"""Training-free layout control for two-subject prompts (src/spherical_routing/regional.py) on the
pool A pair benchmark: early-step regional attention masks + optional layout-gated LoRA parts,
against the hooked static merge (sph_rc_nm_s0_up_K<K>, same prompts and seeds).

Arm: rg_<fac>_<route>_f<frac>[_<opt>...]_K<K>
  fac    base (no adapter) | nm (rc+pico merge factors, run_spherical_proto build) | raw (own LoRAs)
         | merge baselines for the merge ablation (run_spherical_proto build variants, all K
         subjects kept as nm): np (rc w/o de-crowding), avg, ssrm (SSR-Merge), ssrmc (+ our cal.),
         gp<z|x><lam> (random_gp.py), x<lk|oc><g> (ortho_reg.py)
  route  na  no adapter (fac base)
         st  ungated: nm -> every subject of the merge on every token (== the static merge);
             raw -> the prompt's two own LoRAs on every token
         lg  layout-gated (LayoutRouter): region k -> subject k's part only, background and
             non-subject text -> both; nm keeps the other K-2 subjects passive (weight 1)
         dg  attention-gated (DynamicRouter softmax_max, maps of the previous step from every
             double block, temperature t<v> (default 0.2)); with the +p bias the subject maps
             separate, unlike the plain merge (see spherical-routing results)
  f<frac>  regional attention for steps < round(frac * steps); f0 = none
  opts   ib       image-to-image bounding (region k -/-> region j) in every masked block
         ibs<a>-<b>  image bounding only in single blocks a..b (DreamRenderer's middle layers)
         b<r>     base_ratio r (mix with the unmasked attention)
         n<v>     soft mask: blocked bias -v instead of -inf;  p<v>: own text<->region bias +v
         g<gap>   layout_lr background stripe (fraction of the width)
         em       control without a layout: no blocking, +p<v> between every image token and
                  every subject text token (RegionalAttention emphasis)
         al       automatic layout (FreeFuse-style): a first pass of the base model (no LoRA,
                  no mask) from the same noise for --al_steps steps records the subject maps
                  at double block --al_block; layout_from_maps -> regions (fallback: strips)
         cos      bias schedule p cos(pi/2 i / n_act) over the round(frac N) active steps
         pl<v>    after the active steps: own bias +v only (no blocking) until the end
         z<sets>  scene-mass preservation (regional.py): the attention mass of each image query
                  on the key sets s (scene text), p (padding text), i (image) is held at its
                  unbiased value; e.g. zsp: the extra subject mass comes only from the subject
                  tokens (the other subject), zs: only from padding + image keys
         ii<v>    soft image repulsion: bias -v between the image tokens of the two regions
         sg<s>    swap guidance (regional.SwapGuidance): v + s M (v - v_swap), v_swap = the
                  prediction with the two regions exchanged; one extra pass per guided step
         sgm<b|l> swap only the bias (b) / only the LoRA gating (l, route lg); default both
         sgw<a>-<b>, sge<k>  guided steps a..b every k-th (default 0-8, every 2nd)
         fo<b>    forward feature push (regional.FeaturePush): in double-block image features,
                  region k loses beta * its positive projection on region j's distinctive
                  direction; fow<a>-<b> steps (default 0-8, every step), fob<a>-<b> double blocks
                  (default all)
         ng<s>    region-swap NAG (regional.nag_combine, NAG tau 2.5 alpha 0.25): in every biased
                  attention, the subject-region image rows get the NAG extrapolation away from
                  the attention output under the swapped layout; ngw<a>-<b> steps (default 0-8),
                  nga<v> alpha, ngt<v> tau
         lu<f>    lg: layout gating only for steps < round(f steps), then ungated (== st)
         w<v>     lg: weight v of the other subject's part inside region k (soft gating)
         t<v>     dg: mask temperature
         lo<lr|tb|bs>  layout template (regional.layout_named): lr left/right strips (default),
                  tb top/bottom strips, bs big-small (a large foreground box, b small background
                  box; pairs only)
         ld<a>-<b> / ls<a>-<b>  bias only in double blocks a..b / single blocks a..b (if either
                  is given, the other block type has no bias unless also given)
Layout: subject a (first in the prompt) left half, subject b right half (unless al or lo).
--protocols triples: three subjects per prompt (TRIPLE_TEMPLATES, cyclic triples of the pair
permutation; gen_sph_triples/<arm>; T5 + CLIP embeddings cached in data/processed/
triples_embeds.pt, encoded before the transformer loads), layout three strips (lo lr / tb);
the swap guidance negative is the cyclic shift of the regions.  Scored by eval_layout.py.
--protocols t1 (route st only): single-subject prompts (n_prompts per subject, gen_sph/t1_<arm>);
the layout is one region (the whole image) for the prompt's subject, so +p is emphasis of that
subject; f0 arms give the hooked static merge.

  python scripts/run_regional_proto.py --stage gen --arms rg_nm_st_f0.3_K10 --seeds 0 1
  python scripts/run_regional_proto.py --stage eval --arms ... --tag _rg1   (-> sph_pairs_K10_rg1)
  python scripts/run_spherical_proto.py --stage table --K 10                (pooled, paired)
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common
import run_spherical_proto as rsp


FACS = ("base", "nm", "raw", "np", "avg", "ssrm", "ssrmc")


def parse_arm(arm):
    p = arm.split("_")
    assert p[0] == "rg" and p[-1].startswith("K") and p[3].startswith("f"), arm
    a = dict(fac=p[1], route=p[2], frac=float(p[3][1:]), K=int(p[-1][1:]), bound="none",
             bound_single=None, base_ratio=0.0, neg=float("-inf"), pos=0.0, gap=0.0,
             auto=False, emphasis=False, sched="const", pos_late=None, preserve="",
             double_blocks=None, single_blocks=None, other=0.0, temp=0.2, rep=0.0,
             sg=0.0, sg_mode="all", sg_win=(0, 8), sg_every=2, fo=0.0, fo_win=(0, 8),
             fo_blocks=None, until=None, nag=0.0, nag_win=(0, 8), nag_alpha=0.25,
             nag_tau=2.5, layout="lr")
    assert a["fac"] in FACS or re.fullmatch(r"gp[zx][0-9.]+|x(lk|oc)[0-9.]+", a["fac"]), arm   # random_gp / ortho_reg
    assert a["route"] in ("na", "st", "lg", "dg"), arm
    assert (a["fac"] == "base") == (a["route"] == "na"), arm
    for o in p[4:-1]:
        if o.startswith("lo"):
            a["layout"] = o[2:]
        elif o == "al":
            a["auto"] = True
        elif o == "em":
            a["emphasis"] = True
        elif o == "cos":
            a["sched"] = "cos"
        elif o.startswith("pl"):
            a["pos_late"] = float(o[2:])
        elif o.startswith("ii"):
            a["rep"] = float(o[2:])
        elif o.startswith("ngw"):
            a["nag_win"] = tuple(map(int, o[3:].split("-")))
        elif o.startswith("nga"):
            a["nag_alpha"] = float(o[3:])
        elif o.startswith("ngt"):
            a["nag_tau"] = float(o[3:])
        elif o.startswith("ng"):
            a["nag"] = float(o[2:])
        elif o.startswith("lu"):
            a["until"] = float(o[2:])
        elif o.startswith("fow"):
            a["fo_win"] = tuple(map(int, o[3:].split("-")))
        elif o.startswith("fob"):
            b0, b1 = map(int, o[3:].split("-"))
            a["fo_blocks"] = list(range(b0, b1 + 1))
        elif o.startswith("fo"):
            a["fo"] = float(o[2:])
        elif o.startswith("sgm"):
            a["sg_mode"] = o[3:]
        elif o.startswith("sgw"):
            a["sg_win"] = tuple(map(int, o[3:].split("-")))
        elif o.startswith("sge"):
            a["sg_every"] = int(o[3:])
        elif o.startswith("sg"):
            a["sg"] = float(o[2:])
        elif o[0] == "w":
            a["other"] = float(o[1:])
        elif o[0] == "t":
            a["temp"] = float(o[1:])
        elif o[0] == "z":
            a["preserve"] = o[1:]
        elif o[:2] in ("ld", "ls"):
            b0, b1 = map(int, o[2:].split("-"))
            a["double_blocks" if o[1] == "d" else "single_blocks"] = list(range(b0, b1 + 1))
        elif o == "ib":
            a["bound"] = "all"
        elif o.startswith("ibs"):
            b0, b1 = map(int, o[3:].split("-"))
            a["bound"], a["bound_single"] = "single", list(range(b0, b1 + 1))
        elif o[0] == "b":
            a["base_ratio"] = float(o[1:])
        elif o[0] == "n":
            a["neg"] = -float(o[1:])
        elif o[0] == "p":
            a["pos"] = float(o[1:])
        elif o[0] == "g":
            a["gap"] = float(o[1:])
        else:
            raise ValueError(f"arm option {o!r} in {arm}")
    # the preserve path computes attention itself, so dg's map capture processors would not run
    assert not (a["route"] == "dg" and a["preserve"]), arm
    assert a["layout"] in ("lr", "tb", "bs") and not (a["auto"] and a["layout"] != "lr"), arm
    assert not a["sg"] or (a["frac"] > 0 and a["route"] != "dg"), arm
    assert not a["nag"] or (a["frac"] > 0 and not (a["emphasis"] or a["preserve"]
                                                   or a["base_ratio"])), arm
    assert a["until"] is None or a["route"] == "lg", arm
    assert not a["fo"] or (a["frac"] > 0 and not a["emphasis"]), arm
    assert a["sg_mode"] == "b" or not a["sg"] or a["route"] == "lg", arm
    if a["double_blocks"] is not None or a["single_blocks"] is not None:
        a["double_blocks"] = a["double_blocks"] or []
        a["single_blocks"] = a["single_blocks"] or []
    return a


def stage_gen(args):
    from diffusers import FluxPipeline
    from transformers import T5TokenizerFast
    import generate_pairs as gp
    from spherical_routing import regional as rg
    from spherical_routing.dynamic_hook import DynamicRouter

    tok = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    emb = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    emb3 = None
    if "triples" in args.protocols:
        emb3 = gp.encode([m["prompt"] for K in sorted({parse_arm(x)["K"] for x in args.arms})
                          for m in triple_prompts(K, args.n_pairs, args.n_templates)],
                         os.path.join(common.DATA, "triples_embeds.pt"), args.device)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    tr = pipe.transformer
    print(f"pipeline ready in {time.time() - t0:.0f}s", flush=True)
    side = args.res // 16                                  # VAE /8, 2x2 patches
    emb_t1 = None
    cache = {}
    for arm in args.arms:
        a = parse_arm(arm)
        K = a["K"]
        subs = common.SUBJECTS[:K]
        jobs = []
        if "pairs" in args.protocols:
            jobs.append(("pairs", os.path.join(rsp.GEN_PAIRS, arm + args.outsuffix),
                         gp.pair_prompts(K, args.n_pairs, args.n_templates), emb))
        if "triples" in args.protocols:
            jobs.append(("triples", os.path.join(GEN_TRIPLES, arm + args.outsuffix),
                         triple_prompts(K, args.n_pairs, args.n_templates), emb3))
        if "t1" in args.protocols and a["route"] == "st" and not a["auto"] and not a["sg"] \
                and not a["fo"] and not a["nag"]:
            if emb_t1 is None:
                emb_t1 = torch.load(os.path.join(common.DATA, "t1_embeds.pt"), weights_only=False)
            jobs.append(("t1", os.path.join(rsp.GEN_T1, f"t1_{arm}{args.outsuffix}"),
                         [{"subject": x, "prompt": q} for x in subs
                          for q in common.all_prompts(x)[:args.n_prompts]], emb_t1))
        jobs = [j for j in jobs if not os.path.exists(os.path.join(j[1], "meta.json"))]
        if not jobs:
            print(f"[{arm}] done, skip", flush=True)
            continue
        rt = ra = rec = None
        if a["auto"]:                                  # before the RegionalAttention (wraps it)
            rec = rg.MapRecorder(tr, blocks=(args.al_block,))
        if a["fac"] != "base":
            key = (a["fac"], K)
            if key not in cache:
                cache.clear()
                torch.cuda.empty_cache()
                cache[key] = rsp.load_factors("rc", K, a["fac"])[0]
            if a["route"] == "st":
                rt = DynamicRouter(tr, cache[key], gating=False, capture_blocks=[],
                                   dtype=torch.bfloat16)
            elif a["route"] == "dg":
                rt = DynamicRouter(tr, cache[key], gating=True, mask="softmax_max",
                                   temp=a["temp"], dtype=torch.bfloat16)
            else:
                rt = rg.LayoutRouter(tr, cache[key], until=a["until"], dtype=torch.bfloat16)
        if a["frac"] > 0:
            nd, ns = len(tr.transformer_blocks), len(tr.single_transformer_blocks)
            bd = range(nd) if a["bound"] == "all" else ()
            bs = range(ns) if a["bound"] == "all" else (a["bound_single"] or ())
            ra = rg.RegionalAttention(tr, frac=a["frac"], double_blocks=a["double_blocks"],
                                      single_blocks=a["single_blocks"], bound_double=bd,
                                      bound_single=bs, base_ratio=a["base_ratio"], neg=a["neg"],
                                      pos=a["pos"], emphasis=a["emphasis"], sched=a["sched"],
                                      pos_late=a["pos_late"], preserve=a["preserve"],
                                      rep=a["rep"], nag=a["nag"], nag_tau=a["nag_tau"],
                                      nag_alpha=a["nag_alpha"],
                                      nag_steps=range(a["nag_win"][0], a["nag_win"][1] + 1))
        sg = None
        if a["sg"]:
            sg = rg.SwapGuidance(tr, ra, rt, scale=a["sg"], mode=a["sg_mode"],
                                 steps=range(a["sg_win"][0], a["sg_win"][1] + 1, a["sg_every"]))
        fp = None
        if a["fo"]:
            fp = rg.FeaturePush(tr, ra, beta=a["fo"], blocks=a["fo_blocks"],
                                steps=range(a["fo_win"][0], a["fo_win"][1] + 1))
        t1 = time.time()
        try:
            for proto, outdir, meta, em in jobs:
                os.makedirs(outdir, exist_ok=True)
                layouts = {}
                for i, m in enumerate(meta):
                    ss_ = m["subjects"] if proto != "t1" else [m["subject"]]
                    tsets = rsp.token_sets(tok, m["prompt"], ss_)
                    n_valid = len(tok(m["prompt"], max_length=512, truncation=True).input_ids)
                    regions = rg.layout_named(a["layout"], side, len(ss_), a["gap"]) \
                        if proto != "t1" else [torch.ones(side * side, dtype=torch.bool)]
                    E, P = em[m["prompt"]]
                    act = [subs.index(x) for x in ss_]
                    for sd in args.seeds:
                        fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
                        if os.path.exists(fn):
                            continue
                        seed = sd * 1000 + i
                        if rec is not None:
                            if rt is not None:
                                rt.active = None               # LoRA hooks pass through
                            if ra is not None:
                                ra.off()
                            rec.start(tsets)
                            pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                                 pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                                 num_inference_steps=args.steps, guidance_scale=args.guidance,
                                 height=args.res, width=args.res, output_type="latent",
                                 generator=torch.Generator(device=args.device).manual_seed(seed),
                                 callback_on_step_end=_stop_after(args.al_steps))
                            regions, ok = rg.layout_from_maps(rec.stop()[0], side)
                            layouts[f"{i:03d}_s{sd}"] = dict(
                                ok=ok, frac=[float(r.float().mean()) for r in regions],
                                cx=[float((torch.nonzero(r).flatten() % side).float().mean())
                                    / side for r in regions])
                        if rt is not None:
                            pas = ([k for k in range(K) if k not in act] if a["fac"] != "raw"
                                   else [])
                            if a["route"] in ("st", "dg"):     # st: gating off, act + pas at 1
                                rt.set_prompt(act, tsets, args.steps, txt_len=E.shape[0],
                                              passive=pas)
                            else:
                                rt.set_layout(act, tsets, regions, args.steps,
                                              txt_len=E.shape[0], passive=pas, other=a["other"])
                        if ra is not None:
                            ra.set_layout(regions, tsets, args.steps, txt_len=E.shape[0],
                                          n_valid=n_valid)
                        g = torch.Generator(device=args.device).manual_seed(seed)
                        im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                                  pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                                  num_inference_steps=args.steps, guidance_scale=args.guidance,
                                  height=args.res, width=args.res, generator=g).images[0]
                        for c in (rt, ra):
                            assert c is None or c.step == args.steps - 1, (c.step, args.steps)
                        im.save(fn)
                json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
                           "guidance": args.guidance, "arm": {**a, "neg": str(a["neg"])},
                           "layout": "auto" if rec else a["layout"], "auto_layouts": layouts,
                           **({"subjects": subs} if proto == "t1" else {})},
                          open(os.path.join(outdir, "meta.json"), "w"), indent=1)
                print(f"[{arm}] {proto}: {len(meta) * len(args.seeds)} images, "
                      f"{time.time() - t1:.0f}s", flush=True)
        finally:
            if fp is not None:
                fp.remove()
            if sg is not None:
                sg.remove()
            if ra is not None:
                ra.remove()
            if rt is not None:
                rt.remove()
            if rec is not None:
                rec.remove()
    print("done", flush=True)


GEN_TRIPLES = os.path.join(common.ROOT, "gen_sph_triples")
TRIPLE_TEMPLATES = [
    "a {a}, a {b} and a {c} sitting together on a wooden table",
    "a {a}, a {b} and a {c} in a grassy field",
]


def triple_prompts(K, n, n_templates):
    """n cyclic triples (perm[i], perm[i+1], perm[i+2]) of generate_pairs' seeded permutation
    (the pairs extended by the next subject) x n_templates"""
    import generate_pairs as gp
    pairs = gp.pairs_for(K, K)
    out = []
    for i in range(min(n, K)):
        abc = [pairs[i][0], pairs[i][1], pairs[(i + 1) % K][1]]
        for t in TRIPLE_TEMPLATES[:n_templates]:
            out.append({"subjects": abc, "prompt": t.format(
                **{k: f"{common.TOKEN} {common.CLASSES[x]}" for k, x in zip("abc", abc)})})
    return out


def _stop_after(k):
    """callback_on_step_end: interrupt the pipeline after step k - 1 (the remaining steps are
    skipped; with output_type="latent" nothing is decoded)."""
    def cb(pipe, i, t, kw):
        if i + 1 >= k:
            pipe._interrupt = True
        return kw
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("gen", "eval"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--protocols", nargs="+", default=["pairs"], choices=("pairs", "t1", "triples"))
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_templates", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="_rg")
    ap.add_argument("--al_steps", type=int, default=5)
    ap.add_argument("--al_block", type=int, default=18)
    ap.add_argument("--outsuffix", default="")
    args = ap.parse_args()
    for arm in args.arms:
        parse_arm(arm.split(f"_K{args.K}")[0] + f"_K{args.K}")
    if args.stage == "gen":
        stage_gen(args)
    else:
        rsp.stage_eval(args)


if __name__ == "__main__":
    main()

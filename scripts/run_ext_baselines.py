"""External multi-LoRA composition baselines on the pool A pair benchmark (same prompts, seeds,
steps, guidance and resolution as run_regional_proto; outputs gen_sph_pairs/<arm><outsuffix>,
evaluated with run_regional_proto --stage eval / pair_image_checks / paired_vs).

Arm: xb_<method>[_<opt>...]_K<K>, always on the prompt's two own LoRAs (rc_K<K>_raw factors,
the same LoRA hooks as rg_raw_*):
  lsw   LoRA Switch (Zhong et al. 2024): one LoRA active at a time, switched every tau steps
        (opt t<tau>, default 5), order = prompt order
  lcp   LoRA Composite (Zhong et al. 2024): every step, the velocity is the mean of the
        predictions with each LoRA alone (FLUX-dev is guidance-distilled, so the per-LoRA
        cond/uncond CFG of the paper reduces to one prediction per LoRA); 2x cost
  sum   naive merge of the two LoRAs (== rg_raw_st_f0; sanity check of this script)

  python scripts/run_ext_baselines.py --stage gen --arms xb_lsw_K10 xb_lcp_K10 --seeds 6 7 8 9 10 11 --outsuffix _s6-11
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common
import run_spherical_proto as rsp


def parse_arm(arm):
    p = arm.split("_")
    assert p[0] == "xb" and p[-1].startswith("K"), arm
    a = dict(method=p[1], K=int(p[-1][1:]), tau=5)
    assert a["method"] in ("lsw", "lcp", "sum"), arm
    for o in p[2:-1]:
        if o[0] == "t":
            a["tau"] = int(o[1:])
        else:
            raise ValueError(f"arm option {o!r} in {arm}")
    return a


def one_lora(rt):
    """per prompt subject j: the router's active L with every other subject's column zeroed (the
    router's mask keeps one column per prompt subject)"""
    out = []
    for j in range(len(rt.active)):
        d = {}
        for n, L in rt._L_act.items():
            L = L.clone()
            L[:, [c for c in range(L.shape[1]) if c != j]] = 0
            d[n] = L
        out.append(d)
    return out


class Switch:
    """LoRA Switch on a DynamicRouter (ungated): before each transformer call, only the LoRA of
    subject act[(step // tau) % 2] is applied."""

    def __init__(self, tr, rt, tau):
        self.rt, self.tau = rt, tau
        self.h = tr.register_forward_pre_hook(self._pre)       # registered after rt._pre (step++)

    def set_prompt(self):
        self._L = one_lora(self.rt)

    def _pre(self, mod, args):
        self.rt._L_act = self._L[(self.rt.step // self.tau) % len(self._L)]

    def remove(self):
        self.h.remove()


class Composite:
    """LoRA Composite: transformer(...) -> mean over the active LoRAs of the prediction with that
    LoRA alone. The router's step counter advances once per LoRA pass, which only matters for
    gating (off here)."""

    def __init__(self, tr, rt):
        self.tr, self.rt = tr, rt
        self.orig = tr.forward
        tr.forward = self._forward
        self.calls = 0

    def set_prompt(self):
        self._L = one_lora(self.rt)
        self.calls = 0

    def _forward(self, *args, **kw):
        outs = []
        for L in self._L:
            self.rt._L_act = L
            outs.append(self.orig(*args, **kw))
        self.calls += 1
        if isinstance(outs[0], tuple):
            return (sum(o[0] for o in outs) / len(outs),) + tuple(outs[0][1:])
        outs[0].sample = sum(o.sample for o in outs) / len(outs)
        return outs[0]

    def remove(self):
        self.tr.forward = self.orig


def stage_gen(args):
    from diffusers import FluxPipeline
    from transformers import T5TokenizerFast
    import generate_pairs as gp
    from spherical_routing.dynamic_hook import DynamicRouter

    tok = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    emb = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    tr = pipe.transformer
    for arm in args.arms:
        a = parse_arm(arm)
        K = a["K"]
        subs = common.SUBJECTS[:K]
        outdir = os.path.join(rsp.GEN_PAIRS, arm + args.outsuffix)
        if os.path.exists(os.path.join(outdir, "meta.json")):
            print(f"[{arm}] done, skip", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        meta = gp.pair_prompts(K, args.n_pairs, args.n_templates)
        fac = rsp.load_factors("rc", K, "raw")[0]
        rt = DynamicRouter(tr, fac, gating=False, capture_blocks=[], dtype=torch.bfloat16)
        ctl = (Switch(tr, rt, a["tau"]) if a["method"] == "lsw" else
               Composite(tr, rt) if a["method"] == "lcp" else None)
        t1, n, lat = time.time(), 0, []
        try:
            for i, m in enumerate(meta):
                tsets = rsp.token_sets(tok, m["prompt"], m["subjects"])
                E, P = emb[m["prompt"]]
                act = [subs.index(x) for x in m["subjects"]]
                for sd in args.seeds:
                    fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
                    if os.path.exists(fn):
                        continue
                    rt.set_prompt(act, tsets, args.steps, txt_len=E.shape[0])
                    if ctl is not None:
                        ctl.set_prompt()
                    torch.cuda.synchronize()
                    t2 = time.time()
                    g = torch.Generator(device=args.device).manual_seed(sd * 1000 + i)
                    im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                              pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                              num_inference_steps=args.steps, guidance_scale=args.guidance,
                              height=args.res, width=args.res, generator=g).images[0]
                    torch.cuda.synchronize()
                    lat.append(time.time() - t2)
                    if a["method"] == "lcp":
                        assert ctl.calls == args.steps, (ctl.calls, args.steps)
                    else:
                        assert rt.step == args.steps - 1, (rt.step, args.steps)
                    im.save(fn)
                    n += 1
        finally:
            if ctl is not None:
                ctl.remove()
            rt.remove()
        json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
                   "guidance": args.guidance, "arm": a, "layout": "none",
                   "latency_s": sum(lat) / max(1, len(lat))},
                  open(os.path.join(outdir, "meta.json"), "w"), indent=1)
        print(f"[{arm}] {n} images, {time.time() - t1:.0f}s, "
              f"{sum(lat) / max(1, len(lat)):.2f}s/img", flush=True)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("gen",))
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_templates", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outsuffix", default="")
    args = ap.parse_args()
    for arm in args.arms:
        parse_arm(arm)
    stage_gen(args)


if __name__ == "__main__":
    main()

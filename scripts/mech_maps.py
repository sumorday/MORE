"""Mechanism data for the paper figure (Fig. 3): per-step subject attention maps and region
feature similarity of the static merge vs the routed pipeline, with and without swap guidance.

For each arm (run_regional_proto arm names) and pair prompt x seed of the K-subject benchmark,
the image is regenerated with two recorders on double block --block (main pass only; the extra
negative pass of SwapGuidance, ra.swapped = True, is skipped):
  maps[t, k, n]  attention mass of image query n on subject k's text tokens at step t, mean over
                 heads, computed WITH the additive attention bias of that step (what the block
                 actually uses; dynamic_hook.subject_maps + bias)
  fsim[t]        cosine between the mean image features (block output) of region a and region b
                 (lr layout halves, also for the static arm) - the region-feature homogenization
  fown[t, k]     cosine between subject k's region mean feature and the mean feature of all
                 image tokens (for reference)
Output: <out>.pt {arm: {"maps": (n_img, T, 2, N) fp16, "fsim": (n_img, T), "keys": [(i, sd)]}}
plus the regenerated images under <out>_img/<arm>/.

  python scripts/mech_maps.py --K 10 --seeds 6 --out results/mech_K10
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common
import run_regional_proto as rrp
import run_spherical_proto as rsp

F_OPTS = "f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb"
ARMS = ["rg_nm_st_f0", "rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1", "rg_nm_lg_" + F_OPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[6])
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--block", type=int, default=18)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from diffusers import FluxPipeline
    from transformers import T5TokenizerFast
    import generate_pairs as gp
    from spherical_routing import regional as rg
    from spherical_routing.dynamic_hook import DynamicRouter, MapCaptureProcessor

    dev = args.device
    tok = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    emb = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(dev)
    pipe.set_progress_bar_config(disable=True)
    tr = pipe.transformer
    side = 512 // 16
    K = args.K
    subs = common.SUBJECTS[:K]
    meta = gp.pair_prompts(K, args.n_pairs, 2)
    regions = rg.layout_named("lr", side, 2, 0.0)

    class Rec:
        """map + feature recorder of one block (main pass only)"""
        def __init__(self):
            self.ra, self.tsets, self.maps, self.fsim, self.mask = None, None, [], [], None

        def on(self):
            return self.tsets is not None and not (self.ra is not None and self.ra.swapped)

        def wants_maps(self):
            return self.on()

        def add_map(self, query, key, txt_len):
            q = query[:, txt_len:].transpose(1, 2).float()
            k = key.transpose(1, 2).float()
            s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            if self.mask is not None:
                s = s + self.mask[..., txt_len:, :].float()
            p = torch.softmax(s, -1)
            self.maps.append(torch.stack([p[..., t].sum(-1) for t in self.tsets], 1)
                             .mean(2)[0].half().cpu())

    rec = Rec()

    class Cap(MapCaptureProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     image_rotary_emb=None):
            rec.mask = attention_mask
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask,
                                    image_rotary_emb)

    blk = tr.transformer_blocks[args.block]
    blk.attn.set_processor(Cap(rec))
    ra_mask = [r.to(dev) for r in regions]

    def fhook(mod, inp, out):
        if not rec.on():
            return
        h = out[1].float()[0]                                  # (N, D) image stream
        a, b = (h[m].mean(0) for m in ra_mask)
        rec.fsim.append(float(torch.nn.functional.cosine_similarity(a, b, 0)))
    hh = blk.register_forward_hook(fhook)

    res = {}
    cache = {}
    for arm0 in args.arms:
        arm = f"{arm0}_K{K}"
        a = rrp.parse_arm(arm)
        if a["fac"] not in cache:
            cache.clear()
            torch.cuda.empty_cache()
            cache[a["fac"]] = rsp.load_factors("rc", K, a["fac"])[0]
        if a["route"] == "st":
            rt = DynamicRouter(tr, cache[a["fac"]], gating=False, capture_blocks=[],
                               dtype=torch.bfloat16)
        else:
            rt = rg.LayoutRouter(tr, cache[a["fac"]], until=a["until"], dtype=torch.bfloat16)
        ra = sg = None
        if a["frac"] > 0:
            ra = rg.RegionalAttention(tr, frac=a["frac"], neg=a["neg"], pos=a["pos"],
                                      sched=a["sched"], rep=a["rep"])
        if a["sg"]:
            sg = rg.SwapGuidance(tr, ra, rt, scale=a["sg"], mode=a["sg_mode"],
                                 steps=range(a["sg_win"][0], a["sg_win"][1] + 1, a["sg_every"]))
        rec.ra = ra
        odir = f"{args.out}_img/{arm}"
        os.makedirs(odir, exist_ok=True)
        M, FS, keys = [], [], []
        t0 = time.time()
        try:
            for i, m in enumerate(meta):
                tsets = rsp.token_sets(tok, m["prompt"], m["subjects"])
                n_valid = len(tok(m["prompt"], max_length=512, truncation=True).input_ids)
                E, P = emb[m["prompt"]]
                act = [subs.index(x) for x in m["subjects"]]
                pas = [k for k in range(K) if k not in act]
                for sd in args.seeds:
                    seed = sd * 1000 + i
                    if a["route"] == "st":
                        rt.set_prompt(act, tsets, args.steps, txt_len=E.shape[0], passive=pas)
                    else:
                        rt.set_layout(act, tsets, regions, args.steps, txt_len=E.shape[0],
                                      passive=pas, other=a["other"])
                    if ra is not None:
                        ra.set_layout(regions, tsets, args.steps, txt_len=E.shape[0],
                                      n_valid=n_valid)
                    rec.tsets = [torch.as_tensor(t, dtype=torch.long, device=dev) for t in tsets]
                    rec.maps, rec.fsim = [], []
                    im = pipe(prompt_embeds=E.unsqueeze(0).to(dev),
                              pooled_prompt_embeds=P.unsqueeze(0).to(dev),
                              num_inference_steps=args.steps, guidance_scale=3.5,
                              height=512, width=512,
                              generator=torch.Generator(device=dev).manual_seed(seed)).images[0]
                    rec.tsets = None
                    assert len(rec.maps) == args.steps == len(rec.fsim), (len(rec.maps),
                                                                           len(rec.fsim))
                    im.save(os.path.join(odir, f"{i:03d}_s{sd}.png"))
                    M.append(torch.stack(rec.maps))
                    FS.append(torch.tensor(rec.fsim))
                    keys.append((i, sd))
        finally:
            for c in (sg, ra, rt):
                if c is not None:
                    c.remove()
        res[arm] = dict(maps=torch.stack(M), fsim=torch.stack(FS), keys=keys,
                        prompts=[meta[i]["prompt"] for i, _ in keys])
        print(f"[{arm}] {len(keys)} images {time.time() - t0:.0f}s fsim(step0-7) "
              f"{res[arm]['fsim'][:, :8].mean():.4f} fsim(all) {res[arm]['fsim'].mean():.4f}",
              flush=True)
        torch.save(res, args.out + ".pt")
    hh.remove()
    print("done", flush=True)


if __name__ == "__main__":
    main()

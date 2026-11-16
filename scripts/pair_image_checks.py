"""Image-level checks of pair images that the detect-then-DINO metric does not see:
  clipT    CLIP ViT-B/32 cosine of the image with the prompt (without the "sks" token)
  collage  CLIP zero-shot probability of "split screen" vs "single photo" prompts (hard layout
           masks can produce two separate scenes side by side)
  clipS    CLIP cosine with the scene part of the prompt only ("a photo of a scene <rest>")
  lum/sat  mean luma (0-1) and HSV saturation (style shift: darker / more saturated)
  simRef   (paired only) CLIP image-image cosine with the --ref image of the same prompt + seed
Per arm: means over the images + per-image values; paired against --ref on shared files.

  python scripts/pair_image_checks.py --arms rg_nm_st_f0.3_ib_K10 ... --ref sph_rc_nm_s0_up_K10
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import common

GEN_PAIRS = os.path.join(common.ROOT, "gen_sph_pairs")
CLIP = os.path.join(common.MODELS, "clip-vit-base-patch32")
SINGLE = ["a photo", "a single photograph of one scene"]
SPLIT = ["a split screen image of two different photos side by side",
         "a collage of two separate pictures with a vertical dividing line"]


def _unit(out):
    """projected CLIP embedding (transformers >= 5 returns an output object, the projection in
    pooler_output; see eval_metrics.py), L2-normalised"""
    f = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
    return f / f.norm(dim=-1, keepdim=True)


def _scene(m):
    """prompt without its subjects: the text after the last subject phrase"""
    b = f"{common.TOKEN} {common.CLASSES[m['subjects'][-1]]}"
    return "a photo of a scene " + m["prompt"][m["prompt"].rindex(b) + len(b):].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", default="sph_rc_nm_s0_up_K10")
    ap.add_argument("--out", default=os.path.join(common.ROOT, "results", "rg_imgchecks_K10.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP).to(args.device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)

    @torch.no_grad()
    def text_feat(texts):
        t = proc(text=texts, return_tensors="pt", padding=True).to(args.device)
        return _unit(model.get_text_features(**t))

    tq = text_feat(SINGLE + SPLIT)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    arms = list(dict.fromkeys(args.arms + [args.ref]))
    feat_of = {}
    for arm in arms:
        # an arm's images may be split over <arm> and <arm>_s<a>-<b> (seed ranges run separately)
        dirs = [os.path.join(GEN_PAIRS, x) for x in sorted(os.listdir(GEN_PAIRS))
                if x == arm or (x.startswith(arm + "_s") and x[len(arm) + 2:].replace("-", "").isdigit())]
        meta = json.load(open(os.path.join(dirs[0], "meta.json")))
        src = {}
        for d in dirs:
            src.update({f: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")})
        fs = sorted(src)
        prompts = [meta["prompts"][int(f[:3])]["prompt"].replace(f"{common.TOKEN} ", "")
                   for f in fs]
        scenes = [_scene(meta["prompts"][int(f[:3])]) for f in fs]
        ct, co, cs, lum, sat, feats = [], [], [], [], [], []
        for i in range(0, len(fs), 32):
            ims = [Image.open(src[f]).convert("RGB") for f in fs[i:i + 32]]
            with torch.no_grad():
                x = proc(images=ims, return_tensors="pt").to(args.device)
                fi = _unit(model.get_image_features(**x))
            ft = text_feat(prompts[i:i + 32])
            ct += (fi * ft).sum(-1).tolist()
            cs += (fi * text_feat(scenes[i:i + 32])).sum(-1).tolist()
            p = torch.softmax(100 * fi @ tq.t(), -1)
            co += p[:, len(SINGLE):].sum(-1).tolist()
            feats.append(fi.cpu())
            for im in ims:
                a = np.asarray(im, dtype=np.float32) / 255
                lum.append(float((a @ np.array([0.299, 0.587, 0.114], np.float32)).mean()))
                sat.append(float(np.asarray(im.convert("HSV"), np.float32)[..., 1].mean() / 255))
        feats = torch.cat(feats)
        feat_of[arm] = {f: feats[j] for j, f in enumerate(fs)}
        res[arm] = dict(n=len(fs), clipT=float(np.mean(ct)), collage=float(np.mean(co)),
                        collage_gt05=float((np.asarray(co) > 0.5).mean()),
                        clipS=float(np.mean(cs)), lum=float(np.mean(lum)), sat=float(np.mean(sat)),
                        per_image={f: [a, b, c, d, e] for f, a, b, c, d, e
                                   in zip(fs, ct, co, cs, lum, sat)})
    R, FR = res[args.ref]["per_image"], feat_of[args.ref]
    for arm in arms:
        r = res[arm]
        line = (f"{arm:44s} n={r['n']:3d} clipT {r['clipT']:.4f} clipS {r['clipS']:.4f} "
                f"lum {r['lum']:.3f} sat {r['sat']:.3f} collage>0.5 {r['collage_gt05']:.3f}")
        keys = sorted(set(r["per_image"]) & set(R))
        if arm != args.ref and keys:
            se = lambda x: f"{x.mean():+.4f}±{x.std() / len(x) ** .5:.4f}"
            d = {n: np.array([r["per_image"][k][j] - R[k][j] for k in keys])
                 for j, n in ((0, "clipT"), (2, "clipS"), (3, "lum"), (4, "sat"))}
            sim = np.array([float(feat_of[arm][k] @ FR[k]) for k in keys])
            line += (f" | n_paired {len(keys)} " + " ".join(f"d{n} {se(x)}" for n, x in d.items())
                     + f" simRef {sim.mean():.3f}")
            r["simRef"] = float(sim.mean())
        print(line)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

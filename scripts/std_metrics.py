"""Standard image metrics for generated pair / triple / layout images (run with the metrics venv,
envs/metrics: hpsv2 + image-reward + transformers 4.46 on top of the main env):
  hps      HPSv2.1 (ViT-H-14, models_ext/metrics/hpsv2/HPS_v2.1_compressed.pt): image-text
           cosine of the preference-tuned CLIP, prompt without the "sks" token
  ir       ImageReward v1.0 (models_ext/metrics/imagereward)
  clipT    CLIP ViT-L/14 image-text cosine (prompt without "sks")
  clipI    CLIP ViT-L/14 image-image cosine with each subject's reference images (mean over the
           references), averaged over the prompt's subjects;  clipImin  the minimum over them
Arms are pooled over their seed dirs (<arm>, <arm>_s<a>-<b>); per-image values are cached in
--out (an arm is recomputed only if its image count changed).  Summary + paired differences
against --ref (bootstrap SE over images).

  envs/metrics/bin/python scripts/std_metrics.py --genroot gen_sph_pairs --seeds 6 7 8 9 10 11 \\
      --arms rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb_K10 xb_ff_K10 --ref sph_rc_nm_s0_up_K10 \\
      --out results/std_metrics_pairs_K10.json
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
from eval_layout import arm_dirs

MET = os.path.join(common.ROOT, "models_ext", "metrics")
FIELDS = ("hps", "ir", "clipT", "clipI", "clipImin")


def load_hps(dev):
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    model, _, prep = create_model_and_transforms(
        "ViT-H-14", None, precision="amp", device=dev, jit=False, force_quick_gelu=False,
        force_custom_text=False, force_patch_dropout=False, force_image_size=None,
        pretrained_image=False, image_mean=None, image_std=None, light_augmentation=True,
        aug_cfg={}, output_dict=True, with_score_predictor=False, with_region_predictor=False)
    ck = torch.load(os.path.join(MET, "hpsv2", "HPS_v2.1_compressed.pt"), map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tok = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def score(ims, prompts):
        x = torch.stack([prep(im) for im in ims]).to(dev)
        t = tok(prompts).to(dev)
        with torch.cuda.amp.autocast():
            o = model(x, t)
        return (o["image_features"] * o["text_features"]).sum(-1).float().cpu().tolist()
    return score


def load_ir(dev):
    import transformers
    from ImageReward.models.BLIP import blip
    bert = os.path.join(MET, "bert-base-uncased")

    class Tok(transformers.BertTokenizer):                     # offline bert-base-uncased
        @classmethod
        def from_pretrained(cls, name, *a, **k):
            return transformers.BertTokenizer.from_pretrained(
                bert if name == "bert-base-uncased" else name, *a, **k)
    blip.BertTokenizer = Tok
    import ImageReward as RM
    model = RM.load(os.path.join(MET, "imagereward", "ImageReward.pt"), device=dev,
                    med_config=os.path.join(MET, "imagereward", "med_config.json"))

    @torch.no_grad()
    def score(ims, prompts):
        out = []
        for im, p in zip(ims, prompts):
            s = model.score(p, [im])                   # float for one image (list for several)
            out.append(float(s[0] if isinstance(s, list) else s))
        return out
    return score


def load_clip(dev):
    from transformers import CLIPModel, CLIPProcessor
    path = os.path.join(MET, "clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(path).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(path)

    def unit(f):
        f = (f.pooler_output if hasattr(f, "pooler_output") else f).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def img(ims):
        return unit(model.get_image_features(**proc(images=ims, return_tensors="pt").to(dev)))

    @torch.no_grad()
    def txt(t):
        return unit(model.get_text_features(**proc(text=t, return_tensors="pt", padding=True,
                                                   truncation=True).to(dev)))
    return img, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genroot", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", nargs="*", default=[], choices=("hps", "ir"))
    args = ap.parse_args()
    dev = args.device
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    hps = None if "hps" in args.skip else load_hps(dev)
    ir = None if "ir" in args.skip else load_ir(dev)
    cimg, ctxt = load_clip(dev)
    refs = {}
    for arm in list(dict.fromkeys(args.arms + args.ref)):
        dirs = arm_dirs(args.genroot, arm)
        if not dirs:
            print("missing", arm, flush=True)
            continue
        meta = json.load(open(os.path.join(dirs[0], "meta.json")))
        src = {}
        for d in dirs:
            src.update({f: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")})
        fs = sorted(f for f in src if args.seeds is None
                    or int(f.split("_s")[1][:-4]) in args.seeds)
        if arm in res and res[arm]["n"] == len(fs):
            print("cached", arm, flush=True)
            continue
        per = {}
        for b in range(0, len(fs), 16):
            chunk = fs[b:b + 16]
            ims = [Image.open(src[f]).convert("RGB") for f in chunk]
            ms = [meta["prompts"][int(f[:3])] for f in chunk]
            prompts = [m["prompt"].replace(f"{common.TOKEN} ", "") for m in ms]
            h = hps(ims, prompts) if hps else [float("nan")] * len(ims)
            r = ir(ims, prompts) if ir else [float("nan")] * len(ims)
            fi = cimg(ims)
            ct = (fi * ctxt(prompts)).sum(-1).tolist()
            for j, (f, m) in enumerate(zip(chunk, ms)):
                ci = []
                for s in m["subjects"]:
                    if s not in refs:
                        refs[s] = cimg([Image.open(p).convert("RGB")
                                        for p in common.ref_images(s)])
                    ci.append(float((fi[j:j + 1] @ refs[s].t()).mean()))
                per[f] = [h[j], r[j], ct[j], float(np.mean(ci)), float(np.min(ci))]
        res[arm] = dict(n=len(fs), per_image=per,
                        **{k: float(np.nanmean([v[i] for v in per.values()]))
                           for i, k in enumerate(FIELDS)})
        print(f"{arm:60s} n={len(fs)} " + " ".join(f"{k} {res[arm][k]:.4f}" for k in FIELDS),
              flush=True)
        tmp = args.out + ".tmp"
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, args.out)
    rng = np.random.default_rng(0)
    for ref in args.ref or [None]:
        print(f"\n=== {'summary' if ref is None else 'paired diff vs ' + ref}")
        for arm in args.arms:
            if arm not in res:
                continue
            P = res[arm]["per_image"]
            parts = [f"{k} {res[arm][k]:.4f}" for k in FIELDS]
            if ref is not None and ref in res and arm != ref:
                Q = res[ref]["per_image"]
                keys = sorted(set(P) & set(Q))
                if keys:
                    idx = rng.integers(0, len(keys), (2000, len(keys)))
                    for i, k in enumerate(FIELDS):
                        d = np.array([P[f][i] - Q[f][i] for f in keys])
                        parts.append(f"d{k} {d.mean():+.4f}±{d[idx].mean(1).std():.4f}")
            print(f"  {arm:60s} n={res[arm]['n']:3d} " + " ".join(parts))


if __name__ == "__main__":
    main()

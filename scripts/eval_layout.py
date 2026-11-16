"""Detect-then-DINO scoring for any number of subjects per image plus layout adherence and a
layout-agnostic collage check (the layout-generalisation runs of run_regional_proto.py:
lo tb / lo bs pairs and --protocols triples).

Per image: each subject is localised with OWLv2 (query = its class); the K boxes are the
disjoint (pairwise IoU < --iou) choice among each query's top --cand boxes that maximises the
sum of detector scores (eval_pairs_det.assign for K = 2).  Then
  dino_k   DINOv2 CLS cosine of subject k's square crop with k's reference images (mean)
  det_k    detector score of the chosen box;  all = every det_k >= --det_thr
  hit_k    the box centre lies in subject k's layout region (meta "layout", regional.py),
  collage  CLIP B/32 zero-shot probability of split-screen vs single-photo prompts for three
           split groups (v vertical = pair_image_checks' prompts, h horizontal, t three-panel),
           each a softmax against the single-photo prompts alone.  The group matching the
           layout of the arm is used (lr / bs pairs v, tb h, triples max(v, t)); a paired
           comparison scores both arms with the layout arm's group (a max over all groups
           flags too many single photos with strong horizontal edges)
Arm summary: min/mean DINO, all-detected, collage-aware allC / minC / meanC (collage > 0.5 ->
0, as paired_vs --clean), layout hit rate (per subject, all), and paired differences against
--ref arms with the same prompts (bootstrap SE).

  python scripts/eval_layout.py --genroot gen_sph_triples --out results/layout_triples_K10.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common
from eval_metrics import load_imgs
from eval_pairs_det import box_iou, nms_top, square_crop, to_tensor

CLIP = os.path.join(common.MODELS, "clip-vit-base-patch32")
SINGLE = ["a photo", "a single photograph of one scene"]
SPLIT_GROUPS = [["a split screen image of two different photos side by side",
                 "a collage of two separate pictures with a vertical dividing line"],
                ["a split screen image of two different photos stacked top and bottom",
                 "a collage of two separate pictures with a horizontal dividing line"],
                ["a triptych of three separate photos side by side"]]


def assign_k(boxes, scores, k=5, iou=0.5):
    """scores: list of (n,) per subject -> box index per subject (disjoint, max sum)."""
    cands = [nms_top(boxes, s, k, iou) for s in scores]
    I = box_iou(boxes, boxes)
    best, arg = None, [c[0] for c in cands]
    for combo in itertools.product(*cands):
        if len(set(combo)) < len(combo) or any(float(I[a, b]) >= iou for a, b in
                                               itertools.combinations(combo, 2)):
            continue
        v = float(sum(s[i] for s, i in zip(scores, combo)))
        if best is None or v > best:
            best, arg = v, list(combo)
    return arg


def regions_of(meta, n_sub, side):
    from spherical_routing import regional as rg
    lay = meta.get("layout", "lr")
    if lay == "auto":
        return None
    gap = float(meta.get("arm", {}).get("gap", 0.0) or 0.0)
    return [r.view(side, side) for r in rg.layout_named(lay, side, n_sub, gap)]


def arm_dirs(root, arm):
    return [os.path.join(root, x) for x in sorted(os.listdir(root))
            if x == arm or (x.startswith(arm + "_s") and x[len(arm) + 2:].replace("-", "").isdigit())]


def col_group(r):
    """collage group of an arm result (see the module docstring)"""
    return "vt" if r["K"] == 3 else ("h" if r["layout"] == "tb" else "v")


def summarize(rows, K, thr, grp="v"):
    x = np.asarray([r[2:] for r in rows], dtype=np.float64)
    d, s, h = x[:, :K], x[:, K:2 * K], x[:, 2 * K:3 * K]
    c = {g: x[:, 3 * K + j] for j, g in enumerate("vht")}
    col = np.max([c[g] for g in grp], 0)
    ok = (col <= 0.5).astype(float)
    det = (s >= thr)
    return dict(min=d.min(1), mean=d.mean(1), all=det.all(1).astype(float),
                allC=det.all(1) * ok, minC=d.min(1) * ok, meanC=d.mean(1) * ok, col=1 - ok,
                hit=np.where(det, h, 0).mean(1), hitAll=(det & (h > 0.5)).all(1) * ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genroot", required=True)
    ap.add_argument("--arms", nargs="+", default=None, help="arm names (seed dirs pooled)")
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--det_thr", type=float, default=0.2)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--dino", default=os.path.join(common.MODELS, "dinov2-base"))
    ap.add_argument("--owl", default=os.path.join(common.ROOT, "models_ext",
                                                  "owlv2-base-patch16-ensemble"))
    args = ap.parse_args()
    dev = args.device
    root = args.genroot
    arms = args.arms or sorted(d for d in os.listdir(root)
                               if os.path.exists(os.path.join(root, d, "meta.json")))
    from transformers import (AutoModel, CLIPModel, CLIPProcessor, Owlv2ForObjectDetection,
                              Owlv2Processor)
    dino = AutoModel.from_pretrained(args.dino).to(dev).eval()
    owl = Owlv2ForObjectDetection.from_pretrained(args.owl).to(dev).eval()
    proc = Owlv2Processor.from_pretrained(args.owl)
    clip = CLIPModel.from_pretrained(CLIP).to(dev).eval()
    cproc = CLIPProcessor.from_pretrained(CLIP)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    def unit(o):
        f = (o.pooler_output if hasattr(o, "pooler_output") else o).float()
        return f / f.norm(dim=-1, keepdim=True)

    def dfeat(x):
        return F.normalize(dino((x.to(dev) - mean) / std).last_hidden_state[:, 0].float(), dim=-1)

    with torch.no_grad():
        tq = [unit(clip.get_text_features(**cproc(text=SINGLE + g, return_tensors="pt",
                                                  padding=True).to(dev))) for g in SPLIT_GROUPS]

    def detect(im, classes):
        inp = proc(text=[[f"a photo of a {c}" for c in classes]], images=im,
                   return_tensors="pt").to(dev)
        o = owl(**inp)
        s = torch.sigmoid(o.logits[0].float())
        cx, cy, w, h = o.pred_boxes[0].float().unbind(-1)
        side = max(im.size)
        b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * side
        return b.cpu(), [s[:, j].cpu() for j in range(len(classes))]

    res = json.load(open(args.out)).get("arms", {}) if os.path.exists(args.out) else {}
    ref = {}
    side = args.res // 16
    with torch.no_grad():
        for arm in list(dict.fromkeys(arms + args.ref)):
            dirs = arm_dirs(root, arm)
            if not dirs:
                print("missing", arm, flush=True)
                continue
            meta = json.load(open(os.path.join(dirs[0], "meta.json")))
            src = {}
            for d in dirs:
                src.update({f: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")})
            fs = sorted(f for f in src if args.seeds is None
                        or int(f.split("_s")[1][:-4]) in args.seeds)
            if arm in res and res[arm]["n_images"] == len(fs):
                print("cached", arm, flush=True)
                continue
            K = len(meta["prompts"][0]["subjects"])
            regs = regions_of(meta, K, side)
            rows = []
            for f in fs:
                i, sd = int(f[:3]), int(f.split("_s")[1][:-4])
                subs = meta["prompts"][i]["subjects"]
                for s in subs:
                    if s not in ref:
                        ref[s] = dfeat(load_imgs(common.ref_images(s)))
                im = Image.open(src[f]).convert("RGB")
                boxes, sc = detect(im, [common.CLASSES[s] for s in subs])
                idx = assign_k(boxes, sc, args.cand, args.iou)
                feats = dfeat(to_tensor([square_crop(im, boxes[j]) for j in idx]))
                dn = [float((feats[k:k + 1] @ ref[s].t()).mean()) for k, s in enumerate(subs)]
                dt = [float(sc[k][j]) for k, j in enumerate(idx)]
                hit = []
                for k, j in enumerate(idx):
                    if regs is None:
                        hit.append(float("nan"))
                        continue
                    x0, y0, x1, y1 = boxes[j].tolist()
                    cx = min(max(int((x0 + x1) / 2 * side / im.size[0]), 0), side - 1)
                    cy = min(max(int((y0 + y1) / 2 * side / im.size[1]), 0), side - 1)
                    hit.append(float(regs[k][cy, cx]))
                x = cproc(images=[im], return_tensors="pt").to(dev)
                fi = unit(clip.get_image_features(**x))
                col = [float(torch.softmax(100 * fi @ t.t(), -1)[0, len(SINGLE):].sum())
                       for t in tq]
                rows.append([i, sd, *dn, *dt, *hit, *col])
            res[arm] = dict(n_images=len(rows), K=K, layout=meta.get("layout", "lr"),
                            per_image=rows)
            m = summarize(rows, K, args.det_thr, col_group(res[arm]))
            print(f"{arm:60s} n={len(rows)} " + " ".join(f"{k} {np.nanmean(v):.3f}"
                                                         for k, v in m.items()), flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump({"det_thr": args.det_thr, "arms": res}, open(args.out, "w"))
    report(res, arms, args.ref, args.det_thr)


def report(res, arms, refs, thr):
    rng = np.random.default_rng(0)
    for ref in refs or [None]:
        print(f"\n=== {'summary' if ref is None else 'paired diff vs ' + ref}")
        for arm in arms:
            if arm not in res:
                continue
            r = res[arm]
            A = {(int(x[0]), int(x[1])): x for x in r["per_image"]}
            ma = summarize(list(A.values()), r["K"], thr, col_group(r))
            parts = [f"{k} {np.nanmean(v):.3f}" for k, v in ma.items()
                     if k in ("col", "allC", "minC", "meanC", "hit", "hitAll")]
            if ref is not None and ref in res and arm != ref:
                B = {(int(x[0]), int(x[1])): x for x in res[ref]["per_image"]}
                keys = sorted(set(A) & set(B))
                a = summarize([A[k] for k in keys], r["K"], thr, col_group(r))
                b = summarize([B[k] for k in keys], r["K"], thr, col_group(r))
                idx = rng.integers(0, len(keys), (2000, len(keys)))
                parts += [f"d{f} {(a[f] - b[f]).mean():+.3f}±{(a[f] - b[f])[idx].mean(1).std():.3f}"
                          for f in ("allC", "minC", "meanC")]
            print(f"  {arm:60s} n={r['n_images']:3d} " + " ".join(parts))


if __name__ == "__main__":
    main()

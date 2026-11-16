"""Identity confusion of multi-subject images (MuDI D&C / MultiBind-style slot confusion) and
the boxes used for the paper's qualitative figure.

Per image (subjects s_1..s_K of the prompt): OWLv2 boxes as eval_layout.py (disjoint, max sum of
scores), DINOv2 CLS of each square crop, then
  S[k, j] = mean cosine of crop k with the reference images of subject j       (K x K)
  conf    = fraction of detected subjects k with argmax_j S[k, j] != k  (crop looks more like
            another prompt subject than like its own)
  xsim    = mean off-diagonal S[k, j] (identity leakage),  own = mean diagonal
  cc      = mean cosine between the crops of different subjects (homogenization)
Entries: <genroot>:<arm>[:<seeds comma>]; arm dirs pooled as eval_layout.arm_dirs.
Output json {entry: {"rows": [[i, sd, own, xsim, conf, cc, det_min, boxes...]], summary}}.

  python scripts/id_confusion.py --entries gen_sph_pairs:xb_ff_K10:6,7,8,9,10,11 ... --out x.json
"""
import argparse
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
from eval_layout import arm_dirs, assign_k
from eval_metrics import load_imgs
from eval_pairs_det import square_crop, to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det_thr", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino", default=os.path.join(common.MODELS, "dinov2-base"))
    ap.add_argument("--owl", default=os.path.join(common.ROOT, "models_ext",
                                                  "owlv2-base-patch16-ensemble"))
    args = ap.parse_args()
    dev = args.device
    from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor
    dino = AutoModel.from_pretrained(args.dino).to(dev).eval()
    owl = Owlv2ForObjectDetection.from_pretrained(args.owl).to(dev).eval()
    proc = Owlv2Processor.from_pretrained(args.owl)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    def dfeat(x):
        return F.normalize(dino((x.to(dev) - mean) / std).last_hidden_state[:, 0].float(), dim=-1)

    def detect(im, classes):
        inp = proc(text=[[f"a photo of a {c}" for c in classes]], images=im,
                   return_tensors="pt").to(dev)
        o = owl(**inp)
        s = torch.sigmoid(o.logits[0].float())
        cx, cy, w, h = o.pred_boxes[0].float().unbind(-1)
        side = max(im.size)
        b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * side
        return b.cpu(), [s[:, j].cpu() for j in range(len(classes))]

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    ref = {}
    with torch.no_grad():
        for e in args.entries:
            root, arm, *sd = e.split(":")
            seeds = set(map(int, sd[0].split(","))) if sd else None
            dirs = arm_dirs(os.path.join(common.ROOT, root), arm)
            if not dirs:
                print("missing", e, flush=True)
                continue
            meta = json.load(open(os.path.join(dirs[0], "meta.json")))
            src = {}
            for d in dirs:
                src.update({f: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")})
            fs = sorted(f for f in src if seeds is None or int(f.split("_s")[1][:-4]) in seeds)
            if e in res and len(res[e]["rows"]) == len(fs):
                print("cached", e, flush=True)
                continue
            rows = []
            for f in fs:
                i, s_ = int(f[:3]), int(f.split("_s")[1][:-4])
                subs = meta["prompts"][i]["subjects"]
                for s in subs:
                    if s not in ref:
                        ref[s] = dfeat(load_imgs(common.ref_images(s)))
                im = Image.open(src[f]).convert("RGB")
                boxes, sc = detect(im, [common.CLASSES[s] for s in subs])
                idx = assign_k(boxes, sc, 5, 0.5)
                ft = dfeat(to_tensor([square_crop(im, boxes[j]) for j in idx]))
                K = len(subs)
                S = np.array([[float((ft[k:k + 1] @ ref[s].t()).mean()) for s in subs]
                              for k in range(K)])
                det = np.array([float(sc[k][j]) for k, j in enumerate(idx)])
                ok = det >= args.det_thr
                off = ~np.eye(K, dtype=bool)
                conf = float(np.mean([S[k].argmax() != k for k in range(K) if ok[k]])) \
                    if ok.any() else float("nan")
                C = (ft @ ft.t()).cpu().numpy()
                rows.append([i, s_, float(np.diag(S).mean()), float(S[off].mean()), conf,
                             float(C[off].mean()), float(det.min()),
                             [[round(v, 1) for v in boxes[j].tolist()] for j in idx],
                             det.round(3).tolist(), S.round(4).tolist()])
            x = np.array([r[2:7] for r in rows], dtype=float)
            summ = dict(n=len(rows), own=x[:, 0].mean(), xsim=x[:, 1].mean(),
                        conf=float(np.nanmean(x[:, 2])), cc=x[:, 3].mean(),
                        conf_det=float(np.nanmean(np.where(x[:, 4] >= args.det_thr, x[:, 2],
                                                           np.nan))))
            res[e] = dict(rows=rows, summary=summ)
            print(f"{e:80s} " + " ".join(f"{k} {v:.4f}" for k, v in summ.items()), flush=True)
            json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()

"""Detect-then-DINO scoring of two-subject images (generate_pairs.py outputs).

Global-CLS DINO of a two-object image is dominated by layout and background: the
no-adapter base already scores min-DINO 0.086 and every merge sits at that floor, so it
cannot rank composition.  Here each subject is first localised with an open-vocabulary
detector (OWLv2, query = the subject's prompt class, e.g. "cat", "monster toy"), then
DINOv2 CLS of the box crop is compared with that subject's reference images.

Box choice: per query the top-`--cand` boxes after NMS; the two subjects get the pair
(b_a, b_b) with IoU < `--iou` that maximises s_a(b_a) + s_b(b_b), so one hybrid object
cannot be credited to both subjects.  Per image we report DINO_a, DINO_b of the crops,
the detector scores, and "both" = both detector scores >= `--det_thr`.

  python scripts/eval_pairs_det.py --genroot gen_pairs --out results/pairs_det_s01.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import common
from eval_metrics import load_imgs


def box_iou(a, b):
    """IoU of xyxy boxes a (n,4) and b (m,4) -> (n,m)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(-1)       # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter).clamp(min=1e-12)


def nms_top(boxes, scores, k, iou):
    """Indices of up to k highest-scoring boxes, greedily dropping IoU >= iou overlaps."""
    keep = []
    for i in torch.argsort(scores, descending=True).tolist():
        if all(float(box_iou(boxes[i:i + 1], boxes[j:j + 1])) < iou for j in keep):
            keep.append(i)
            if len(keep) == k:
                break
    return keep


def assign(boxes, sa, sb, k=5, iou=0.5):
    """(i_a, i_b) maximising sa[i_a] + sb[i_b] over top-k candidates with IoU < iou.

    Falls back to the two queries' top boxes if no disjoint pair exists."""
    ca, cb = nms_top(boxes, sa, k, iou), nms_top(boxes, sb, k, iou)
    best, arg = None, (ca[0], cb[0])
    for i in ca:
        for j in cb:
            if i == j or float(box_iou(boxes[i:i + 1], boxes[j:j + 1])) >= iou:
                continue
            v = float(sa[i] + sb[j])
            if best is None or v > best:
                best, arg = v, (i, j)
    return arg


def square_crop(im, box, pad=1.1):
    """Square crop around an xyxy pixel box (side = pad * max(w, h)), kept inside im."""
    W, H = im.size
    x0, y0, x1, y1 = [float(v) for v in box]
    s = min(max(x1 - x0, y1 - y0) * pad, W, H)
    s = max(s, 16.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    l = min(max(cx - s / 2, 0.0), W - s)
    t = min(max(cy - s / 2, 0.0), H - s)
    return im.crop((int(round(l)), int(round(t)), int(round(l + s)), int(round(t + s))))


def to_tensor(ims, size=224):
    import numpy as np
    return torch.stack([torch.from_numpy(np.array(im.convert("RGB").resize((size, size),
                        Image.BICUBIC))).permute(2, 0, 1).float() / 255.0 for im in ims])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genroot", required=True)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--det_thr", type=float, default=0.2)
    ap.add_argument("--dump", default=None, help="save the two crops of every image here (debug)")
    ap.add_argument("--dino", default=os.path.join(common.MODELS, "dinov2-base"))
    ap.add_argument("--owl", default=os.path.join(common.ROOT, "models_ext",
                                                  "owlv2-base-patch16-ensemble"))
    args = ap.parse_args()
    dev = args.device
    tags = args.tags or sorted(d for d in os.listdir(args.genroot)
                               if os.path.exists(os.path.join(args.genroot, d, "meta.json")))
    from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor
    dino = AutoModel.from_pretrained(args.dino).to(dev).eval()
    owl = Owlv2ForObjectDetection.from_pretrained(args.owl).to(dev).eval()
    proc = Owlv2Processor.from_pretrained(args.owl)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    def dfeat(x):
        return F.normalize(dino((x.to(dev) - mean) / std).last_hidden_state[:, 0].float(), dim=-1)

    def detect(im, qa, qb):
        """xyxy pixel boxes (n,4) and sigmoid scores per query (n,), (n,)."""
        inp = proc(text=[[f"a photo of a {qa}", f"a photo of a {qb}"]], images=im,
                   return_tensors="pt").to(dev)
        o = owl(**inp)
        s = torch.sigmoid(o.logits[0].float())                 # (patches, 2)
        cx, cy, w, h = o.pred_boxes[0].float().unbind(-1)      # normalised to the padded square
        side = max(im.size)                                    # OWLv2 pads bottom/right
        b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * side
        return b.cpu(), s[:, 0].cpu(), s[:, 1].cpu()

    ref = {}
    results = {}
    with torch.no_grad():
        for tag in tags:
            d = os.path.join(args.genroot, tag)
            meta = json.load(open(os.path.join(d, "meta.json")))
            per_img = []
            for i, m in enumerate(meta["prompts"]):
                a, b = m["subjects"]
                for s in (a, b):
                    if s not in ref:
                        ref[s] = dfeat(load_imgs(common.ref_images(s)))
                for sd in meta["seeds"]:
                    fn = os.path.join(d, f"{i:03d}_s{sd}.png")
                    if not os.path.exists(fn):
                        continue
                    im = Image.open(fn).convert("RGB")
                    boxes, sa, sb = detect(im, common.CLASSES[a], common.CLASSES[b])
                    ia, ib = assign(boxes, sa, sb, args.cand, args.iou)
                    crops = [square_crop(im, boxes[ia]), square_crop(im, boxes[ib])]
                    if args.dump:
                        os.makedirs(os.path.join(args.dump, tag), exist_ok=True)
                        for c, s, sc in zip(crops, (a, b), (sa[ia], sb[ib])):
                            c.save(os.path.join(args.dump, tag,
                                                f"{i:03d}_s{sd}_{s}_{float(sc):.2f}.png"))
                    f = dfeat(to_tensor(crops))
                    va = float((f[0:1] @ ref[a].t()).mean())
                    vb = float((f[1:2] @ ref[b].t()).mean())
                    per_img.append([i, sd, va, vb, float(sa[ia]), float(sb[ib])])
            n = len(per_img)
            both = [min(x[4], x[5]) >= args.det_thr for x in per_img]
            res = {"n_images": n,
                   "min_dino": sum(min(x[2], x[3]) for x in per_img) / n,
                   "mean_dino": sum((x[2] + x[3]) / 2 for x in per_img) / n,
                   "gap_dino": sum(abs(x[2] - x[3]) for x in per_img) / n,
                   "both_detected": sum(both) / n,
                   # crop DINO counted as 0 for a subject whose detector score < det_thr
                   "min_dino_det": sum(min(x[2] * (x[4] >= args.det_thr),
                                           x[3] * (x[5] >= args.det_thr)) for x in per_img) / n,
                   "per_image": per_img}
            results[tag] = res
            print(f"{tag:56s} cropMin {res['min_dino']:.4f} cropMean {res['mean_dino']:.4f} "
                  f"both {res['both_detected']:.3f} minDet {res['min_dino_det']:.4f} n={n}",
                  flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"det_thr": args.det_thr, "cand": args.cand, "iou": args.iou, "arms": results},
              open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()

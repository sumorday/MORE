"""Score generated merge images: DINOv2 / CLIP-I subject similarity + CLIP-T prompt alignment.

Both metrics are averaged per subject over reference images / prompts, so the
score is comparable across K and across arms.
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


def load_imgs(paths, size=224):
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2)).resize((size, size), Image.BICUBIC)
        out.append(torch.from_numpy(__import__("numpy").array(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genroot", required=True)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dino", default=os.path.join(common.MODELS, "dinov2-base"))
    ap.add_argument("--clip", default=os.path.join(common.MODELS, "clip-vit-base-patch32"))
    args = ap.parse_args()

    dev = args.device
    root_meta = os.path.join(args.genroot, "meta.json")
    tags = args.tags or sorted(
        [d for d in os.listdir(args.genroot) if os.path.isdir(os.path.join(args.genroot, d))]
    )
    # generate_table1.py writes one metadata file per arm; older runs wrote a
    # single one at the root.  Fall back to the root file when a tag lacks its own.
    metas, subjects = {}, []
    for tag in tags:
        mf = os.path.join(args.genroot, tag, "meta.json")
        if not os.path.exists(mf):
            mf = root_meta
        if not os.path.exists(mf):
            print(f"skip {tag}: no metadata")
            continue
        metas[tag] = json.load(open(mf))
        for s in metas[tag]["subjects"]:
            if s not in subjects:
                subjects.append(s)
    tags = [t for t in tags if t in metas]
    if not tags:
        raise SystemExit("no arms with metadata found")

    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPTokenizer

    dino_proc = AutoImageProcessor.from_pretrained(args.dino)
    dino = AutoModel.from_pretrained(args.dino).to(dev).eval()
    clip = CLIPModel.from_pretrained(args.clip).to(dev).eval()
    ctok = CLIPTokenizer.from_pretrained(args.clip)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    # CLIP uses its own normalization constants, not ImageNet's -- using the
    # DINO mean/std here (as an earlier version of this script did) silently
    # shifts every CLIP-I score; it doesn't crash because both are 3-channel
    # unit-range tensors of the same shape.
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)

    def clip_feat(x):
        # x is raw [0,1]; apply CLIP's own normalization here.
        # transformers>=5 returns the vision tower output object; the projected
        # image embedding lives in `pooler_output` (it is overwritten with
        # visual_projection(pooled) before being returned).
        out = clip.get_image_features(pixel_values=(x - clip_mean) / clip_std)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(emb.float(), dim=-1)

    text_cache = {}

    def text_feat(pr):
        # CLIP-T (DreamBooth protocol): prompt with the identifier token removed
        if pr not in text_cache:
            t = ctok([pr.replace(f"{common.TOKEN} ", "")], padding=True, return_tensors="pt").to(dev)
            out = clip.get_text_features(**t)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            text_cache[pr] = F.normalize(emb.float(), dim=-1)[0]
        return text_cache[pr]

    # Reference features per subject: both metrics are image<->reference
    # similarities averaged over the subject's reference images, matching the
    # SSR paper's protocol.  CLIP-T (image vs prompt without the identifier) is
    # reported separately as "clip_t".
    ref_feat, ref_clip = {}, {}
    with torch.no_grad():
        for s in subjects:
            x = load_imgs(common.ref_images(s)).to(dev)
            ref_feat[s] = F.normalize(dino((x - mean) / std).last_hidden_state[:, 0].float(), dim=-1)
            ref_clip[s] = clip_feat(x)

    results = {}
    for tag in tags:
        d = os.path.join(args.genroot, tag)
        if not os.path.isdir(d):
            continue
        meta = metas[tag]
        prompts = meta["prompts"]
        # collect all images in the canonical order of `prompts`
        files = []
        missing = 0
        for i, m in enumerate(prompts):
            got = []
            for sd in meta["seeds"]:
                fn = os.path.join(d, f"{i:03d}_s{sd}.png")
                if os.path.exists(fn):
                    got.append(fn)
                else:
                    missing += 1
            if got:
                files.append((i, m["subject"], m["prompt"], got))
        if not files:
            continue

        acc_dino, acc_clip, acc_ct, n = 0.0, 0.0, 0.0, 0
        per_subject = {s: [0.0, 0.0, 0] for s in subjects}
        with torch.no_grad():
            buf, meta_buf = [], []

            def flush():
                nonlocal acc_dino, acc_clip, acc_ct, n
                if not buf:
                    return
                x = torch.stack(buf).to(dev)
                df = F.normalize(dino((x - mean) / std).last_hidden_state[:, 0].float(), dim=-1)
                cf = clip_feat(x)
                for j, (s, pr) in enumerate(meta_buf):
                    ds = float((df[j] @ ref_feat[s].t()).mean())
                    cs = float((cf[j] @ ref_clip[s].t()).mean())
                    acc_dino += ds
                    acc_clip += cs
                    acc_ct += float(cf[j] @ text_feat(pr))
                    per_subject[s][0] += ds
                    per_subject[s][1] += cs
                    per_subject[s][2] += 1
                    n += 1
                buf.clear()
                meta_buf.clear()

            for i, s, pr, got in files:
                for fn in got:
                    buf.append(load_imgs([fn])[0])
                    meta_buf.append((s, pr))
                    if len(buf) >= args.batch:
                        flush()
            flush()

        res = {
            "n_images": n,
            "missing": missing,
            "dino": acc_dino / max(n, 1),
            "clip": acc_clip / max(n, 1),
            "clip_t": acc_ct / max(n, 1),
            "per_subject_dino": {s: (v[0] / v[2] if v[2] else None) for s, v in per_subject.items()},
            "per_subject_clip": {s: (v[1] / v[2] if v[2] else None) for s, v in per_subject.items()},
        }
        results[tag] = res
        print(f"{tag:24s} DINO {res['dino']:.4f}  CLIP {res['clip']:.4f}  CLIP-T {res['clip_t']:.4f}  n={n}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()

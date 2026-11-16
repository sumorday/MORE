"""Precompute DreamBooth text embeddings (T5 + CLIP) and VAE latents for one subject.

Writes $DATA/<subject>.pt with:
  latents  (N, 16, 64, 64)  float32
  embeds   (25, 512, 4096)  bf16   T5 prompt embeddings
  pooled   (25, 768)        bf16   CLIP pooled
  batches  (N,)  long      index into embeds (image i trained with prompt perm[i])
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(common.DATA, exist_ok=True)
    torch.manual_seed(0)

    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL

    dtype = torch.bfloat16
    prompts = common.all_prompts(args.subject)

    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"), max_length=512)
    te2 = T5EncoderModel.from_pretrained(os.path.join(common.MODEL, "text_encoder_2"), torch_dtype=dtype).to(args.device)
    tok1 = CLIPTokenizer.from_pretrained(os.path.join(common.MODEL, "tokenizer"))
    te1 = CLIPTextModel.from_pretrained(os.path.join(common.MODEL, "text_encoder"), torch_dtype=dtype).to(args.device)
    te1.requires_grad_(False).eval()
    te2.requires_grad_(False).eval()

    with torch.no_grad():
        e2 = tok2(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(args.device)
        embeds = te2(e2.input_ids, attention_mask=e2.attention_mask)[0]
        e1 = tok1(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(args.device)
        pooled = te1(e1.input_ids).pooler_output
    embeds, pooled = embeds.cpu(), pooled.cpu()
    del te1, te2
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(os.path.join(common.MODEL, "vae"), torch_dtype=dtype).to(args.device)
    vae.eval()
    cfg = vae.config
    imgs = common.ref_images(args.subject)
    latents = []
    with torch.no_grad():
        for p in imgs:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((args.res, args.res), Image.LANCZOS)
            x = torch.from_numpy(__import__("numpy").array(im)).float().div(127.5).sub(1.0)
            x = x.permute(2, 0, 1).unsqueeze(0).to(args.device, dtype)
            z = vae.encode(x).latent_dist.sample()
            z = (z - cfg.shift_factor) * cfg.scaling_factor
            latents.append(z.cpu().float())
    latents = torch.cat(latents, 0)

    perm = torch.arange(latents.shape[0]) % len(prompts)
    out = os.path.join(common.DATA, f"{args.subject}.pt")
    torch.save(
        {"latents": latents, "embeds": embeds, "pooled": pooled,
         "perm": perm, "prompts": prompts, "images": imgs},
        out,
    )
    print(f"saved {out}: latents {tuple(latents.shape)} embeds {tuple(embeds.shape)}")


if __name__ == "__main__":
    main()

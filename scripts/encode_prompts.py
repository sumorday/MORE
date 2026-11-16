"""Pre-encode every Table-1 prompt with T5 + CLIP, so generation can run without
the 9.5GB T5 encoder resident beside the 12B transformer (32GB card).

Writes data/processed/t1_embeds.pt: {prompt: (t5 (512,4096), pooled (768,))}
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--prompt_offset", type=int, default=0,
                    help="encode ENVS[offset : offset + n_prompts] (held-out prompts)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(common.DATA, "t1_embeds.pt"))
    args = ap.parse_args()

    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    prompts = []
    for s in common.SUBJECTS:
        prompts += common.all_prompts(s)[args.prompt_offset: args.prompt_offset + args.n_prompts]
    prompts = sorted(set(prompts))
    print(f"{len(prompts)} unique prompts")

    tok1 = CLIPTokenizer.from_pretrained(os.path.join(common.MODEL, "tokenizer"))
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"), max_length=512)
    te1 = CLIPTextModel.from_pretrained(os.path.join(common.MODEL, "text_encoder"),
                                        torch_dtype=torch.bfloat16).to(args.device).eval()
    te2 = T5EncoderModel.from_pretrained(os.path.join(common.MODEL, "text_encoder_2"),
                                         torch_dtype=torch.bfloat16).to(args.device).eval()

    out = {}
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            chunk = prompts[i: i + 8]
            e2 = tok2(chunk, padding="max_length", max_length=512, truncation=True,
                      return_tensors="pt").to(args.device)
            E = te2(e2.input_ids, attention_mask=e2.attention_mask)[0].cpu()
            e1 = tok1(chunk, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").to(args.device)
            P = te1(e1.input_ids).pooler_output.cpu()
            for j, p in enumerate(chunk):
                out[p] = (E[j], P[j])
            print(f"{i+len(chunk)}/{len(prompts)}", flush=True)

    torch.save(out, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

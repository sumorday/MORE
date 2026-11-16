"""Two-subject composition protocol for large K: one FLUX load for all arms.

A K-subject merge is prompted with pairs of its own subjects in one image
("a sks dog and a sks teapot ..."), so a merge that keeps each subject alone
but lets one dominate the other is caught.  Pairs are fixed per K: a seeded
permutation of the K subjects, consecutive (cyclic) pairs, first --n_pairs.
Each pair gets --n_templates scene templates x --seeds.  eval_pairs.py scores
min over the two subjects of DINOv2 similarity.

Prompts are encoded here (T5 + CLIP, freed before the transformer loads) and
cached in data/processed/pairs_embeds.pt keyed by prompt.

  python scripts/generate_pairs.py --arms s2_avg_K10 s2_ssr_full_tau0.2_g0.8_K10 \\
      --outroot gen_pairs

Reference arm pairsum<W>_K<K>: for each pair only its two own LoRAs (--loras_root) are
active, each at adapter weight W (no merge, no other subjects): what plain LoRA
composition of the two subjects gives under the same prompts and seeds.
Arm base_K<K>: no adapter (floor).
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common
from generate_table1 import load_lora


def load_pair(pipe, root, a, b, w):
    from safetensors.torch import load_file
    for name, s in (("pa", a), ("pb", b)):
        sd = load_file(os.path.join(root, s, "pytorch_lora_weights.safetensors"))
        sd = {k.replace(".base_model.model.", "."): v for k, v in sd.items()}
        pipe.load_lora_weights(sd, adapter_name=name)
    pipe.set_adapters(["pa", "pb"], adapter_weights=[w, w])

TEMPLATES = [
    "a {a} and a {b} sitting together on a wooden table",
    "a {a} and a {b} in a grassy field",
    "a {a} and a {b} on a beach at sunset",
    "a {a} and a {b} in a cozy living room",
]


def pairs_for(K, n_pairs):
    subs = common.SUBJECTS[:K]
    perm = list(range(K))
    random.Random(K).shuffle(perm)
    out = [(subs[perm[i]], subs[perm[(i + 1) % K]]) for i in range(K)]
    return out[:n_pairs]


def pair_prompts(K, n_pairs, n_templates):
    out = []
    for a, b in pairs_for(K, n_pairs):
        for t in TEMPLATES[:n_templates]:
            out.append({"subjects": [a, b],
                        "prompt": t.format(a=f"{common.TOKEN} {common.CLASSES[a]}",
                                           b=f"{common.TOKEN} {common.CLASSES[b]}")})
    return out


def encode(prompts, path, device):
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    todo = sorted({p for p in prompts if p not in cache})
    if not todo:
        return cache
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    tok1 = CLIPTokenizer.from_pretrained(os.path.join(common.MODEL, "tokenizer"))
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"), max_length=512)
    te1 = CLIPTextModel.from_pretrained(os.path.join(common.MODEL, "text_encoder"),
                                        torch_dtype=torch.bfloat16).to(device).eval()
    te2 = T5EncoderModel.from_pretrained(os.path.join(common.MODEL, "text_encoder_2"),
                                         torch_dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        for i in range(0, len(todo), 8):
            chunk = todo[i: i + 8]
            e2 = tok2(chunk, padding="max_length", max_length=512, truncation=True,
                      return_tensors="pt").to(device)
            E = te2(e2.input_ids, attention_mask=e2.attention_mask)[0].cpu()
            e1 = tok1(chunk, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").to(device)
            P = te1(e1.input_ids).pooler_output.cpu()
            for j, p in enumerate(chunk):
                cache[p] = (E[j], P[j])
    del te1, te2
    torch.cuda.empty_cache()
    tmp = path + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)
    print(f"encoded {len(todo)} prompts -> {path}", flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_templates", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outroot", default=os.path.join(common.ROOT, "gen_pairs"))
    ap.add_argument("--embeds", default=os.path.join(common.DATA, "pairs_embeds.pt"))
    ap.add_argument("--loras_root", default=os.path.join(common.ROOT, "loras_seed2"),
                    help="own LoRAs for pairsum<W>_K<K> arms")
    args = ap.parse_args()

    Ks = sorted({int(a.split("_K")[1]) for a in args.arms})
    plan = {K: pair_prompts(K, args.n_pairs, args.n_templates) for K in Ks}
    emb = encode([m["prompt"] for v in plan.values() for m in v], args.embeds, args.device)

    from diffusers import FluxPipeline
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline ready in {time.time()-t0:.0f}s", flush=True)

    for arm in args.arms:
        K = int(arm.split("_K")[1])
        outdir = os.path.join(args.outroot, arm)
        meta_f = os.path.join(outdir, "meta.json")
        if os.path.exists(meta_f):
            print(f"[{arm}] done, skip", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        pairsum = arm.startswith("pairsum")
        nolora = arm.startswith("base_K")
        if pairsum:
            w_pair = float(arm[len("pairsum"):].split("_K")[0])
        elif not nolora:
            load_lora(pipe, os.path.join(common.MERGED, f"{arm}.safetensors"))
        meta = plan[K]
        loaded = None
        for i, m in enumerate(meta):
            if pairsum and loaded != m["subjects"]:
                if loaded is not None:
                    pipe.delete_adapters(["pa", "pb"])
                load_pair(pipe, args.loras_root, *m["subjects"], w_pair)
                loaded = m["subjects"]
            for sd in args.seeds:
                fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
                if os.path.exists(fn):
                    continue
                g = torch.Generator(device=args.device).manual_seed(sd * 1000 + i)
                E, P = emb[m["prompt"]]
                im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                          pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                          num_inference_steps=args.steps, guidance_scale=args.guidance,
                          height=args.res, width=args.res, generator=g).images[0]
                im.save(fn)
        json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
                   "guidance": args.guidance}, open(meta_f, "w"), indent=1)
        print(f"[{arm}] {len(meta) * len(args.seeds)} images", flush=True)
        if not nolora:
            pipe.delete_adapters(["pa", "pb"] if pairsum else "m")
    print("done", args.outroot, flush=True)


if __name__ == "__main__":
    main()

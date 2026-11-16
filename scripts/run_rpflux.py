"""Regional Prompting FLUX (Chen et al., arXiv 2411.02395; official code ext/Regional-Prompting-FLUX,
ported to diffusers 0.42 in ext/rpflux_port: transformer_flux.py with absolute imports, the
pipeline verbatim) + naive merge of the prompt's two own LoRAs (peft adapters at weight 1), on the
pool A pair benchmark: same prompts, seeds, steps, guidance and resolution as run_regional_proto,
the same oracle layout (subject a left half, subject b right half).

RP-FLUX as published (infer_flux_regional.py defaults): base prompt = the pair prompt (the same
T5 / pooled embeddings as every other arm), one regional prompt per half,
"a sks <class>" + the prompt's scene text; regional attention mask (text <-> own region, image
self-attention within a region) for the first mask_inject_steps = 10 steps, mixed with the
unmasked base-prompt attention at base_ratio = 0.3, injection in every double / single block;
the halves cover the image, so there is no background region.

Arm: xb_rp_K<K> -> gen_sph_pairs/xb_rp_K<K><outsuffix>   (--lora_w v != 1: xb_rp_w<v>_K<K>)
  python scripts/run_rpflux.py --K 10 --outsuffix _s6-11
"""
import argparse
import json
import os
import sys
import time

import torch

RP_ROOT = os.path.join(os.path.dirname(__file__), "..", "ext", "rpflux_port")
sys.path.insert(0, os.path.abspath(RP_ROOT))
sys.path.insert(1, os.path.dirname(__file__))
import common
import run_spherical_proto as rsp


def build_pipe(subjects, device):
    from safetensors.torch import load_file
    from transformer_flux_rp import FluxTransformer2DModel as RPTransformer
    from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0

    tr = RPTransformer.from_pretrained(common.MODEL, subfolder="transformer",
                                       torch_dtype=torch.bfloat16)
    pipe = RegionalFluxPipeline.from_pretrained(common.MODEL, transformer=tr,
                                                torch_dtype=torch.bfloat16)
    # transformer + T5 do not fit a 32 GB GPU: the text encoders stay on the CPU (fp32) and
    # encode_prompt runs there, cached (the regional prompts repeat over seeds; latency excludes
    # text encoding, as for every arm on the precomputed pair embeddings)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    pipe.text_encoder.float()
    pipe.text_encoder_2.float()
    enc, cache = pipe.encode_prompt, {}

    def encode_prompt(prompt=None, prompt_embeds=None, pooled_prompt_embeds=None, device=None,
                      **kw):
        if prompt_embeds is not None:
            out = enc(prompt=prompt, prompt_embeds=prompt_embeds,
                      pooled_prompt_embeds=pooled_prompt_embeds, device=device, **kw)
            return tuple(x.to(device, torch.bfloat16) for x in out)
        if prompt not in cache:
            with torch.no_grad():
                cache[prompt] = enc(prompt=prompt, device="cpu", **kw)
        return tuple(x.to(device, torch.bfloat16) for x in cache[prompt])
    pipe.encode_prompt = encode_prompt
    pipe.set_progress_bar_config(disable=True)
    for s in subjects:
        # strip the peft base_model.model infix (generate_table1.load_lora: diffusers 0.42
        # otherwise dispatches the file to its fal-Kontext converter)
        sd = load_file(os.path.join(rsp.LORAS_ROOT, s, "pytorch_lora_weights.safetensors"))
        pipe.load_lora_weights({k.replace(".base_model.model.", "."): v for k, v in sd.items()},
                               adapter_name=s)
    procs = {}
    for name, p in pipe.transformer.attn_processors.items():   # infer_flux_regional.py
        procs[name] = (RegionalFluxAttnProcessor2_0()
                       if "transformer_blocks" in name and name.endswith("attn.processor") else p)
    pipe.transformer.set_attn_processor(procs)
    n_rp = sum(isinstance(p, RegionalFluxAttnProcessor2_0)
               for p in pipe.transformer.attn_processors.values())
    print(f"{len(subjects)} adapters, {n_rp} regional processors", flush=True)
    assert n_rp == len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
    return pipe


def regional_prompts(m):
    a, b = m["subjects"]
    last = f"{common.TOKEN} {common.CLASSES[b]}"
    scene = m["prompt"][m["prompt"].rindex(last) + len(last):]
    return [f"a {common.TOKEN} {common.CLASSES[s]}{scene}" for s in (a, b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_templates", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--mask_inject_steps", type=int, default=10)
    ap.add_argument("--base_ratio", type=float, default=0.3)
    ap.add_argument("--lora_w", type=float, default=1.0,
                    help="weight of each own LoRA (arm suffix _w<v> if != 1; 0 = port sanity check)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outsuffix", default="")
    ap.add_argument("--limit", type=int, default=None, help="first n prompts only (smoke test)")
    args = ap.parse_args()
    import generate_pairs as gp

    K = args.K
    subs = common.SUBJECTS[:K]
    arm = f"xb_rp_K{K}" if args.lora_w == 1 else f"xb_rp_w{args.lora_w:g}_K{K}"
    outdir = os.path.join(rsp.GEN_PAIRS, arm + args.outsuffix)
    if os.path.exists(os.path.join(outdir, "meta.json")):
        print(f"[{arm}] done, skip")
        return
    os.makedirs(outdir, exist_ok=True)
    meta = gp.pair_prompts(K, args.n_pairs, args.n_templates)
    emb = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    t0 = time.time()
    pipe = build_pipe(subs, args.device)
    print(f"pipeline ready in {time.time() - t0:.0f}s", flush=True)
    R = args.res
    left = torch.zeros(R, R, device=args.device)
    left[:, :R // 2] = 1.0
    masks = [left, 1.0 - left]
    t1, n, lat, log = time.time(), 0, [], {}
    for i, m in enumerate(meta[:args.limit]):
        a, b = m["subjects"]
        rps = regional_prompts(m)
        log[f"{i:03d}"] = rps
        E, P = emb[m["prompt"]]
        for rp in rps:                                  # fill the text cache before timing
            pipe.encode_prompt(prompt=rp, prompt_2=rp, device=args.device, max_sequence_length=512)
        for sd in args.seeds:
            fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
            if os.path.exists(fn):
                continue
            pipe.set_adapters([a, b], [args.lora_w, args.lora_w])
            for p in pipe.transformer.attn_processors.values():
                if hasattr(p, "regional_mask"):
                    p.regional_mask = None          # the processor caches the first mask it sees
            torch.cuda.synchronize()
            t2 = time.time()
            g = torch.Generator(device=args.device).manual_seed(sd * 1000 + i)
            im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device, torch.bfloat16),
                      pooled_prompt_embeds=P.unsqueeze(0).to(args.device, torch.bfloat16),
                      num_samples=1, width=R, height=R, mask_inject_steps=args.mask_inject_steps,
                      guidance_scale=args.guidance, num_inference_steps=args.steps, generator=g,
                      joint_attention_kwargs={"regional_prompts": rps, "regional_masks": masks,
                                              "double_inject_blocks_interval": 1,
                                              "single_inject_blocks_interval": 1,
                                              "base_ratio": args.base_ratio}).images[0]
            torch.cuda.synchronize()
            lat.append(time.time() - t2)
            im.save(fn)
            n += 1
        print(f"[{arm}] prompt {i}: {n} images, {time.time() - t1:.0f}s", flush=True)
    json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
               "guidance": args.guidance,
               "arm": dict(method="rpflux", K=K, mask_inject_steps=args.mask_inject_steps,
                           base_ratio=args.base_ratio, lora_w=args.lora_w),
               "layout": "lr", "latency_s": sum(lat) / max(1, len(lat)),
               "regional_prompts": log}, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    print(f"[{arm}] {n} images, {time.time() - t1:.0f}s, {sum(lat) / max(1, len(lat)):.2f}s/img")


if __name__ == "__main__":
    main()

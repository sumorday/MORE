"""LoRAShop (Dalva et al., arXiv 2505.23758, NeurIPS 2025; official code ext/LoRAShop, MIT) on the
pool A pair benchmark: the same prompts, seeds, steps, guidance and resolution as
run_regional_proto, the prompt's two own LoRAs (loras_seed2, rank 32, alpha = rank).

Needs diffusers 0.31 (upstream pin): run with envs/ls031/bin/python (venv over the main env with
diffusers 0.31.0, transformers 4.46.3, huggingface_hub 0.26.5, peft 0.13.2).

LoRAShop as published (main.py + flux_real_pipeline.RealGenerationPipeline):
  - prior pass: 5 steps of the base model (LoRAs off); double-block image->concept-token attention
    of each target_prompt span -> one_homogeneous_blob(sigma 1, thr 0.8) -> sigmoid(10 (m - mean))
    -> keep > quantile 0.6 -> argmax over concepts with confidence 0.1 -> disjoint seg masks
    (masks from the last prior step);
  - generation from the same noise with every double / single block run once per LoRA plus once
    LoRA-free, the LoRA outputs blended inside their seg masks (base output elsewhere); attention
    blending from edit_start_step (main.py passes 0), double-block FF blending from step 3;
  - both passes use the pipeline's second-order midpoint solver (2 transformer calls per step);
  - main.py defaults: 30 steps, guidance 3.5, 1024x1024, max_sequence_length 256, bf16; the README
    example uses 50 steps. target_prompt[i] (a substring of the prompt) drives LoRA i.
Benchmark settings used here instead: 28 steps (= 56 transformer calls + 10 prior calls), 512x512,
guidance 3.5, the cached 512-token T5 / CLIP pooled embeds of data/pairs_embeds.pt (the same
conditioning as every other arm; T5 is not loaded). target_prompt = the "sks <class>" spans.
Deviations (bookkeeping only, the method is unchanged):
  - all K LoRAs are loaded once by subject name before register_transformer_blocks (adapters must
    be injected before the blocks are wrapped) and the blocks' set_adapter maps target i to the
    prompt's i-th own subject instead of the hard-coded f"default_{i}";
  - encode_prompt returns the cached embeds (upstream's version only defines prompt_length when
    it runs the text encoders itself);
  - token spans: upstream get_substring_tokens (first occurrence); run_spherical_proto.token_sets
    only if a span occurs more than once in the prompt.

Arm: xb_ls_K<K> -> gen_sph_pairs/xb_ls_K<K><outsuffix>
  envs/ls031/bin/python scripts/run_lorashop.py --K 10 --outsuffix _s6-11
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import torch

LS_ROOT = os.path.join(os.path.dirname(__file__), "..", "ext", "LoRAShop")
sys.path.insert(0, os.path.abspath(LS_ROOT))
sys.path.insert(1, os.path.dirname(__file__))
import common
import run_spherical_proto as rsp

CUR = []        # adapter names of the current prompt's targets, in target_prompt order


def _set_adapter(self, module_to_set, adapter_idx):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in module_to_set.modules():
        if isinstance(module, BaseTunerLayer):
            module.set_adapter(CUR[adapter_idx])


def build_pipe(subjects, device):
    from safetensors.torch import load_file
    from transformers import T5TokenizerFast
    import flux_blocks
    from flux_real_pipeline import RealGenerationPipeline

    # add_prefix_space=None: the saved config's add_prefix_space forces a slow (spiece) conversion
    # under transformers 4.46 and the snapshot only ships tokenizer.json (same ids)
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"),
                                           add_prefix_space=None)
    pipe = RealGenerationPipeline.from_pretrained(
        common.MODEL, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=tok2).to(device)
    pipe.set_progress_bar_config(disable=True)
    for s in subjects:
        sd = load_file(os.path.join(rsp.LORAS_ROOT, s, "pytorch_lora_weights.safetensors"))
        pipe.load_lora_weights({k.replace(".base_model.model.", "."): v for k, v in sd.items()},
                               adapter_name=s)
    pipe.register_transformer_blocks()
    flux_blocks.TransformerBlock.set_adapter = _set_adapter
    flux_blocks.SingleTransformerBlock.set_adapter = _set_adapter
    n = sum(isinstance(m, flux_blocks.TransformerBlock) for m in pipe.transformer.modules())
    n1 = sum(isinstance(m, flux_blocks.SingleTransformerBlock) for m in pipe.transformer.modules())
    print(f"{len(subjects)} adapters, {n} + {n1} LoRAShop blocks", flush=True)
    assert n == 19 and n1 == 38
    return pipe


def activate(pipe, names):
    """Make names[0] the only active adapter of every LoRA layer, all scalings 1 (alpha = rank)."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    CUR[:] = names
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer):
            for k in m.scaling:
                m.scaling[k] = 1.0
            m.set_adapter(names[0])


def targets(pipe, prompt, subjects):
    spans = [f"{common.TOKEN} {common.CLASSES[s]}" for s in subjects]
    if all(prompt.count(t) == 1 for t in spans):
        return spans, type(pipe).get_target_token_idxs(pipe, prompt, spans)
    return spans, rsp.token_sets(pipe.tokenizer_2, prompt, subjects)


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
    ap.add_argument("--edit_start_step", type=int, default=0, help="main.py passes 0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outsuffix", default="")
    ap.add_argument("--limit", type=int, default=None, help="first n prompts only (smoke test)")
    ap.add_argument("--save_masks", action="store_true", help="seg masks -> <outdir>_masks/")
    args = ap.parse_args()
    import diffusers
    import generate_pairs as gp
    assert diffusers.__version__ == "0.31.0", diffusers.__version__

    K = args.K
    subs = common.SUBJECTS[:K]
    arm = f"xb_ls_K{K}"
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
    cond = {}
    pipe.encode_prompt = lambda **kw: cond["v"]
    t1, n, lat, log, area = time.time(), 0, [], {}, {}
    for i, m in enumerate(meta[:args.limit]):
        a, b = m["subjects"]
        spans, idx = targets(pipe, m["prompt"], [a, b])
        E, P = emb[m["prompt"]]
        E = E.unsqueeze(0).to(args.device, torch.bfloat16)
        ids = pipe.tokenizer_2(m["prompt"], padding="max_length", max_length=E.shape[1],
                               truncation=True, return_tensors="pt").input_ids
        cond["v"] = (E, P.unsqueeze(0).to(args.device, torch.bfloat16),
                     torch.zeros(E.shape[1], 3, device=args.device, dtype=torch.bfloat16),
                     torch.argmin(ids))
        log[f"{i:03d}"] = dict(targets=spans, token_idxs=idx)
        for sd in args.seeds:
            fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
            if os.path.exists(fn):
                continue
            activate(pipe, [a, b])
            torch.cuda.synchronize()
            t2 = time.time()
            g = torch.Generator(device=args.device).manual_seed(sd * 1000 + i)
            pipe.get_target_token_idxs = lambda prompt, target_prompt: idx
            with contextlib.redirect_stdout(io.StringIO()):
                im = pipe(prompt=m["prompt"], target_prompt=spans, guidance_scale=args.guidance,
                          num_inference_steps=args.steps, max_sequence_length=E.shape[1],
                          height=args.res, width=args.res, generator=g,
                          edit_start_step=args.edit_start_step).images[0]
            torch.cuda.synchronize()
            lat.append(time.time() - t2)
            im.save(fn)
            n += 1
            # the prior-pass seg masks (1, (res/16)^2, 1) stay in the pipeline's kwargs
            msk = [x.view(args.res // 16, args.res // 16).float().cpu()
                   for x in pipe.joint_attention_kwargs["target_mask"]]
            area[f"{i:03d}_s{sd}"] = [round(x.mean().item(), 4) for x in msk]
            if args.save_masks:
                from PIL import Image
                os.makedirs(outdir + "_masks", exist_ok=True)
                rgb = torch.stack([msk[0], msk[1], torch.zeros_like(msk[0])], -1)
                Image.fromarray((rgb * 255).byte().numpy()).resize((args.res, args.res), 0).save(
                    os.path.join(outdir + "_masks", f"{i:03d}_s{sd}.png"))
        print(f"[{arm}] prompt {i}: {n} images, {time.time() - t1:.0f}s", flush=True)
    json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
               "guidance": args.guidance,
               "arm": dict(method="lorashop", K=K, edit_start_step=args.edit_start_step),
               "layout": "auto (LoRAShop prior masks)", "latency_s": sum(lat) / max(1, len(lat)),
               "token_pos": log, "mask_area": area},
              open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    print(f"[{arm}] {n} images, {time.time() - t1:.0f}s, {sum(lat) / max(1, len(lat)):.2f}s/img")


if __name__ == "__main__":
    main()

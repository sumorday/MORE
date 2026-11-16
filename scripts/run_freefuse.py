"""FreeFuse (Liu et al., arXiv 2510.23515; official code ext/FreeFuse, master 2026-11) on the pool A
pair benchmark: the same prompts, seeds, steps, guidance and resolution as run_regional_proto, the
prompt's two own LoRAs (loras_seed2, peft adapters at weight 1).

FreeFuse as published (main_freefuse_flux*.py): phase 1 runs the base model (LoRAs off) for
aggreate_lora_score_step=5 steps and extracts concept sim maps at double block 18; balanced argmax
+ EOS background exclusion -> per-LoRA masks + attention bias (scale 4, positive 2,
bidirectional); phase 2 regenerates from the same noise with each LoRA confined to its mask and
kept off the other subject's tokens. Concept text = the "sks <class>" span (FreeFuse's
find_concept_positions, restricted to the occurrence run_spherical_proto.token_sets assigns).
Background = EOS token (option 1 of main_freefuse_flux_bg_except.py, fully automatic).

Arm: xb_ff_K<K> -> gen_sph_pairs/xb_ff_K<K><outsuffix>
  python scripts/run_freefuse.py --K 10 --outsuffix _s6-11
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import torch

FF_ROOT = os.path.join(os.path.dirname(__file__), "..", "ext", "FreeFuse")
sys.path.insert(0, os.path.abspath(FF_ROOT))
sys.path.insert(1, os.path.dirname(__file__))
import common
import run_spherical_proto as rsp


def build_pipe(subjects, device):
    from safetensors.torch import load_file
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxAttention
    from peft.tuners.lora.layer import LoraLayer, Linear
    from transformers import T5TokenizerFast
    from src.pipeline.freefuse_flux_direct_extract_attn_bais_background_except_pipeline import \
        FreeFuseFluxDirectExtractAttnBaisBGExceptPipeline
    from src.attn_processor.freefuse_attn_processor import FreeFuseFluxAttnProcessor
    from src.models.freefuse_transformer_flux import (FluxTransformer2DModel,
                                                      FreeFuseFluxTransformerBlock,
                                                      FreeFuseFluxAttention)
    from src.tuner.freefuse_lora_layer import FreeFuseLinear

    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    pipe = FreeFuseFluxDirectExtractAttnBaisBGExceptPipeline.from_pretrained(
        common.MODEL, torch_dtype=torch.bfloat16, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=tok2).to(device)
    pipe.set_progress_bar_config(disable=True)
    for s in subjects:
        # strip the peft base_model.model infix (generate_table1.load_lora: diffusers 0.42
        # otherwise dispatches the file to its fal-Kontext converter)
        sd = load_file(os.path.join(rsp.LORAS_ROOT, s, "pytorch_lora_weights.safetensors"))
        pipe.load_lora_weights({k.replace(".base_model.model.", "."): v for k, v in sd.items()},
                               adapter_name=s)
    # main_freefuse_flux.py, verbatim order
    pipe.transformer.__class__ = FluxTransformer2DModel
    procs = {}
    for name, p in pipe.transformer.attn_processors.items():
        if "single_transformer_blocks" in name:
            procs[name] = p
        elif "transformer_blocks.18.attn" in name:
            procs[name] = FreeFuseFluxAttnProcessor()
            procs[name].cal_concept_sim_map = True
        else:
            procs[name] = FreeFuseFluxAttnProcessor()
    pipe.transformer.set_attn_processor(procs)
    for m in pipe.transformer.modules():
        if isinstance(m, LoraLayer) and isinstance(m, Linear):
            FreeFuseLinear.init_from_lora_linear(m)
    for m in pipe.transformer.modules():
        if isinstance(m, FluxTransformerBlock):
            m.__class__ = FreeFuseFluxTransformerBlock
        if isinstance(m, FluxAttention):
            m.__class__ = FreeFuseFluxAttention
    n_ff = sum(isinstance(m, FreeFuseLinear) for m in pipe.transformer.modules())
    print(f"{len(subjects)} adapters, {n_ff} FreeFuseLinear layers", flush=True)
    assert n_ff > 0
    return pipe


def reset(pipe):
    """Per-image state reset. Upstream reset_freefuse calls a missing reset_attn_map and deletes
    freefuse_mask (singular; the layers store freefuse_masks), and main_freefuse_flux.py never
    calls it (one image per process). Phase 1 runs with the LoRAs disabled and phase 2 overwrites
    the masks, so clearing the layer masks and the token maps is enough."""
    from src.tuner.freefuse_lora_layer import FreeFuseLinear
    for m in pipe.transformer.modules():
        if isinstance(m, FreeFuseLinear):
            m.freefuse_masks = None
            m.freefuse_token_pos_maps = None
    for k in ("freefuse_mask", "freefuse_token_pos_maps", "cross_attn_map"):
        if hasattr(pipe.transformer, k):
            delattr(pipe.transformer, k)


def concept_positions(pipe, prompt, subjects):
    """FreeFuse's token positions of each "sks <class>" span, restricted to the occurrence that
    token_sets assigns to the subject (the two spans can share text, e.g. two toys)."""
    import main_freefuse_flux as mff
    ours = rsp.token_sets(pipe.tokenizer_2, prompt, subjects)
    out = {}
    for s, t in zip(subjects, ours):
        ff = mff.find_concept_positions(pipe, prompt, {s: f"{common.TOKEN} {common.CLASSES[s]}"})[s][0]
        t = [int(x) for x in t]
        keep = [p for p in ff if p in t]
        out[s] = [keep or t]
    return out


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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outsuffix", default="")
    ap.add_argument("--limit", type=int, default=None, help="first n prompts only (smoke test)")
    args = ap.parse_args()
    import generate_pairs as gp
    import main_freefuse_flux as mff

    K = args.K
    subs = common.SUBJECTS[:K]
    arm = f"xb_ff_K{K}"
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
    t1, n, lat, log = time.time(), 0, [], {}
    for i, m in enumerate(meta[:args.limit]):
        a, b = m["subjects"]
        pos = concept_positions(pipe, m["prompt"], [a, b])
        eos = mff.find_first_eos_index(pipe, m["prompt"])
        texts = mff.get_prompt_token_texts(pipe, m["prompt"])
        E, P = emb[m["prompt"]]
        log[f"{i:03d}"] = dict(pos=pos, eos=eos)
        for sd in args.seeds:
            fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
            if os.path.exists(fn):
                continue
            pipe.set_adapters([a, b], [1.0, 1.0])
            reset(pipe)
            pipe.transformer.set_freefuse_token_pos_maps(pos)
            torch.cuda.synchronize()
            t2 = time.time()
            g = torch.Generator(device=args.device).manual_seed(sd * 1000 + i)
            with contextlib.redirect_stdout(io.StringIO()):          # per-call prints
                im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device, torch.bfloat16),
                          pooled_prompt_embeds=P.unsqueeze(0).to(args.device, torch.bfloat16),
                          concept_embeds_map=None, height=args.res, width=args.res,
                          num_inference_steps=args.steps, generator=g,
                          joint_attention_kwargs={"freefuse_token_pos_maps": pos,
                                                  "concept_token_texts": {a: texts, b: texts},
                                                  "eos_token_index": eos,
                                                  "background_token_positions": None},
                          guidance_scale=args.guidance, true_cfg_scale=0).images[0]
            torch.cuda.synchronize()
            lat.append(time.time() - t2)
            im.save(fn)
            n += 1
        print(f"[{arm}] prompt {i}: {n} images, {time.time() - t1:.0f}s", flush=True)
    json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
               "guidance": args.guidance, "arm": dict(method="freefuse", K=K),
               "layout": "auto (FreeFuse)", "latency_s": sum(lat) / max(1, len(lat)),
               "token_pos": log}, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    print(f"[{arm}] {n} images, {time.time() - t1:.0f}s, {sum(lat) / max(1, len(lat)):.2f}s/img")


if __name__ == "__main__":
    main()

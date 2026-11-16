"""Test spectral-adaptive ridge + interference-guided tau on K=3.

Compares:
  - RGC baseline (trace, no adaptive_tau)
  - spectral sigma_max
  - spectral Frobenius
  - spectral condition-aware
  - spectral + interference (adaptive_tau=True)

Usage:
  python scripts/test_spectral.py --device cuda:1
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common
import ssrlib

from safetensors.torch import load_file


def load_pairs(subjects, root=None):
    root = root or common.LORAS
    return [common.extract_lora_pairs(load_file(
        os.path.join(root, s, "pytorch_lora_weights.safetensors"))) for s in subjects]


def prompt_embeds(subs, tag, dev, dtype):
    allp = sorted({f"a {common.TOKEN} {common.CLASSES[s]}" for s in subs})
    # emb_{tag}.pt files written before the prompt list was stored are trusted only
    # when the prompt count matches; otherwise the cache is keyed by the prompt list
    path = os.path.join(common.DATA, f"emb_{tag}.pt")
    if os.path.exists(path):
        out = torch.load(path, weights_only=False)
        if out.get("prompts", allp) == allp and out["E"].shape[0] == len(allp):
            return out
        h = hashlib.md5("|".join(allp).encode()).hexdigest()[:8]
        path = os.path.join(common.DATA, f"emb_{tag}_{h}.pt")
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    te1 = CLIPTextModel.from_pretrained(os.path.join(common.MODEL, "text_encoder"),
                                        torch_dtype=dtype).to(dev).eval()
    tok1 = CLIPTokenizer.from_pretrained(os.path.join(common.MODEL, "tokenizer"))
    te2 = T5EncoderModel.from_pretrained(os.path.join(common.MODEL, "text_encoder_2"),
                                         torch_dtype=dtype).to(dev).eval()
    tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"),
                                           max_length=512)
    with torch.no_grad():
        e2 = tok2(allp, padding="max_length", max_length=512, truncation=True,
                  return_tensors="pt").to(dev)
        E = te2(e2.input_ids, attention_mask=e2.attention_mask)[0]
        e1 = tok1(allp, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").to(dev)
        P = te1(e1.input_ids).pooler_output
    out = {"E": E.cpu(), "P": P.cpu(), "prompts": allp}
    torch.save(out, path)
    del te1, te2, tok1, tok2
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--loras_root", default=None,
                    help="directory holding <subject>/pytorch_lora_weights.safetensors; "
                         "defaults to common.LORAS")
    ap.add_argument("--prefix", default="", help="prefix for arm/checkpoint names")
    ap.add_argument("--stats_dir", default=None,
                    help="save the calibration stats here as stats_K{K}.pt so that "
                         "solve_from_stats.py can sweep arm configs on CPU")
    args = ap.parse_args()

    dev = args.device
    dtype = torch.bfloat16
    t_start = time.time()

    subs = args.subjects[:args.K] if args.subjects else common.SUBJECTS[:args.K]
    tag = f"K{args.K}"
    print(f"=== Spectral-Adaptive Ridge Test: {tag} tau={args.tau} ===\n", flush=True)

    emb = prompt_embeds(subs, tag, dev, dtype)

    from diffusers import FluxTransformer2DModel
    print(f"loading transformer on {dev} ...", flush=True)
    model = FluxTransformer2DModel.from_pretrained(
        os.path.join(common.MODEL, "transformer"), torch_dtype=dtype)
    if args.offload:
        from accelerate.big_modeling import cpu_offload
        model.eval()
        model.requires_grad_(False)
        cpu_offload(model, torch.device(dev))
    else:
        model = model.to(dev)
        model.eval()
        model.requires_grad_(False)
    print(f"transformer ready ({time.time()-t_start:.0f}s)\n", flush=True)

    from train_lora import latent_ids, pack_latents

    task_pairs = load_pairs(subs, args.loras_root)
    comb = ssrlib.concat_pairs(task_pairs)
    base = [f"a {common.TOKEN} {common.CLASSES[s]}" for s in subs]
    idx = {p: i for i, p in enumerate(sorted(set(base)))}
    E = emb["E"].to(dev)
    P = emb["P"].to(dev)
    ts = [1 / 8.0]

    def run_task(mdl, k, n_steps):
        for si, tv in enumerate(ts):
            j = idx[base[k]]
            gen = torch.Generator(device=dev).manual_seed(args.seed + si)
            x0 = torch.randn(1, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
            noise = torch.randn(1, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
            t = torch.tensor([tv], device=dev, dtype=dtype)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
            mdl(hidden_states=pack_latents(xt), timestep=t,
                encoder_hidden_states=E[j: j + 1], pooled_projections=P[j: j + 1],
                img_ids=latent_ids(32, 32, dev), txt_ids=torch.zeros(512, 3, device=dev),
                guidance=torch.full((1,), 3.5, device=dev, dtype=dtype), return_dict=False)

    print(f"[{tag}] calibrating with interference ...", flush=True)
    t0 = time.time()
    stats = ssrlib.accumulate(comb, task_pairs, model, run_task,
                              n_steps=len(ts), seed=args.seed,
                              keep_tasks=True, compute_interference=True,
                              lora_device=dev if args.offload else None,
                              log=True)
    print(f"[{tag}] calibrated in {time.time()-t0:.0f}s\n", flush=True)

    # Check interference values
    interf = sorted(s.get("interference", 0.0) for s in stats.values() if s["n"] > 0)
    print(f"Interference: median {interf[len(interf)//2]:.3f}, "
          f"p90 {interf[int(0.9*len(interf))]:.3f}, max {interf[-1]:.3f}\n", flush=True)

    diag_summary = ssrlib.diagnose_stats(stats, comb, task_pairs, tag=tag)
    print("", flush=True)
    if args.stats_dir:
        os.makedirs(args.stats_dir, exist_ok=True)
        sp = os.path.join(args.stats_dir, f"stats_{tag}.pt")
        torch.save(stats, sp)
        print(f"[{tag}] wrote {sp}", flush=True)

    # Test different spectral modes
    modes = [
        ("trace", False, "RGC baseline (mean diag)"),
        ("sigma", False, "spectral radius (σ_max)"),
        ("frob", False, "Frobenius norm"),
        ("cond", False, "condition-aware (σ_max·φ(κ))"),
        ("sigma", True, "σ_max + interference"),
    ]

    results = {}
    for spectral_mode, adaptive_tau, desc in modes:
        arm = f"{spectral_mode}" + ("_adp" if adaptive_tau else "")
        print(f"[{arm}] {desc}", flush=True)
        t0 = time.time()
        merged, diag = ssrlib.ssr_solve(
            stats, comb, ridge="rel", tau=args.tau,
            spectral_mode=spectral_mode, adaptive_tau=adaptive_tau,
            device=dev)
        # Extract ridge_rel values for statistics
        ridge_vals = sorted([d["ridge_rel"] for d in diag.values() if "ridge_rel" in d])
        if ridge_vals:
            med = ridge_vals[len(ridge_vals) // 2]
            p90 = ridge_vals[int(0.9 * len(ridge_vals))]
            maxval = ridge_vals[-1]
        else:
            med = p90 = maxval = 0.0
        print(f"  ridge: median {med:.3f}, p90 {p90:.3f}, max {maxval:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        out = os.path.join(common.MERGED, f"{args.prefix}spectral_{arm}_{tag}.safetensors")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        from safetensors.torch import save_file
        save_file(common.emit_peft_state(merged), out)
        print(f"  wrote {out}\n", flush=True)

        results[arm] = {
            "spectral_mode": spectral_mode,
            "adaptive_tau": adaptive_tau,
            "tau": args.tau,
            "ridge_median": med,
            "ridge_p90": p90,
            "ridge_max": maxval,
            "diag_G": diag_summary,
        }

    json.dump(results, open(os.path.join(common.MERGED, f"{args.prefix}spectral_test_{tag}.json"), "w"), indent=1)
    print(f"\ndone in {(time.time()-t_start)/60:.1f} min")
    print(f"\nNext: generate images and evaluate metrics")
    print(f"  python scripts/generate_table1.py --arms " +
          " ".join(f"{args.prefix}spectral_{k}_{tag}" for k in results.keys()))


if __name__ == "__main__":
    main()

"""Calibration statistics on the LoRA's own training distribution (richer than test_spectral).

test_spectral.py calibrates each task with ONE forward: prompt "a sks <class>", t = 1/8,
x0 = pure noise.  That is far from inference (28 steps over the whole noise range,
scene prompts, real image content).  Here each task k runs, with its own LoRA injected,
  prompts  : common.ENVS[--prompt_ids] scene prompts of subject k (default 5..9, disjoint
             from the 4 eval prompts ENVS[0:4]); embeddings from the precompute cache
  x0       : the subject's own training latents (cycled per prompt), or noise (--x0 noise)
  t        : --timesteps (default 0.1 .. 0.9), x_t = (1 - t) x0 + t n  (train_lora.py)
one batched forward per prompt.  Accumulation is ssrlib.accumulate unchanged
(keep_tasks=True), so the output has the layout of data/seed2/stats_K{K}.pt and every
downstream solver (ssr_rc.py, fair_route.py, ...) reads it via --stats_root.

--pair_prompts N adds, per task k, N two-subject prompts "a sks <k> and a sks <j> ..."
(generate_pairs templates [2:4], k first or second in turn; partners j never form one of
the evaluated pairs generate_pairs.pairs_for(K, 10)), noise x0, same timesteps, with only
LoRA k injected: the merge then also has to keep each subject's own behaviour on inputs
that contain a second subject.

--legacy reproduces test_spectral's run_task exactly (same embeddings, RNG, t = 1/8);
--check compares the result layer by layer with an existing stats file, so the
plumbing is verified before the new statistics are trusted.

  python scripts/calib_stats.py --K 10 20 --out_dir data/seed2_cal
  python scripts/calib_stats.py --K 10 --legacy --check data/seed2/stats_K10.pt
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common
import ssrlib
from solve_from_stats import load_pairs


def check(stats, ref_path, n_layers=12):
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    names = [n for n in stats if stats[n]["n"] > 0]
    worst = 0.0
    for name in names[:: max(1, len(names) // n_layers)]:
        a, b = stats[name], ref[name]
        assert a["n"] == b["n"], (name, a["n"], b["n"])
        for key in ("G", "Q"):
            d = float((a[key].cpu() - b[key]).norm() / b[key].norm())
            worst = max(worst, d)
        for k, t in a["tasks"].items():
            u = b["tasks"][k]
            assert (t["start"], t["end"]) == (u["start"], u["end"])
            worst = max(worst, float((t["G"] - u["G"]).norm() / u["G"].norm()),
                        float((t["Qb"] - u["Qb"]).norm() / u["Qb"].norm()))
    print(f"check vs {ref_path}: max rel diff {worst:.2e} over {n_layers} layers", flush=True)
    return worst


def pair_prompt_texts(K, n):
    """{k: [n two-subject prompts containing subject k]}, partners outside the eval pairs."""
    import random
    from generate_pairs import TEMPLATES, pairs_for
    subs = common.SUBJECTS[:K]
    held = {frozenset(p) for p in pairs_for(K, 10)}
    rng = random.Random(1000 + K)
    out = {}
    for k, a in enumerate(subs):
        cand = [b for b in subs if b != a and frozenset((a, b)) not in held]
        for i, b in enumerate(rng.sample(cand, n)):
            x, y = (a, b) if i % 2 == 0 else (b, a)
            t = TEMPLATES[2 + i % 2]
            out.setdefault(k, []).append(t.format(a=f"{common.TOKEN} {common.CLASSES[x]}",
                                                  b=f"{common.TOKEN} {common.CLASSES[y]}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[10])
    ap.add_argument("--loras_root", default="loras_seed2")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--timesteps", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--prompt_ids", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    ap.add_argument("--x0", default="real", choices=("real", "noise"))
    ap.add_argument("--pair_prompts", type=int, default=0)
    ap.add_argument("--txt_trim", action="store_true",
                    help="drop T5 padding: text stream = the prompt's own tokens (+ EOS) only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy", action="store_true", help="test_spectral calibration")
    ap.add_argument("--check", default=None, help="reference stats file to compare with")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert not set(args.prompt_ids) & {0, 1, 2, 3} or args.legacy, \
        "prompt_ids overlap the eval prompts ENVS[0:4]"

    dev, dtype = args.device, torch.bfloat16
    pair_texts, pair_cache = {}, {}
    if args.pair_prompts and not args.legacy:
        # encode before the transformer is on the GPU (T5 + FLUX do not fit in 32 GB)
        from generate_pairs import encode
        pair_texts = {K: pair_prompt_texts(K, args.pair_prompts) for K in args.K}
        for K in args.K:
            print(f"[K{K}] pair prompts of {common.SUBJECTS[0]}: {pair_texts[K][0]}", flush=True)
        os.makedirs(args.out_dir, exist_ok=True)
        pair_cache = encode([p for v in pair_texts.values() for ps in v.values() for p in ps],
                            os.path.join(args.out_dir, "pairs_calib_embeds.pt"), dev)
        torch.cuda.empty_cache()
    from diffusers import FluxTransformer2DModel
    from train_lora import latent_ids, pack_latents
    t0 = time.time()
    model = FluxTransformer2DModel.from_pretrained(
        os.path.join(common.MODEL, "transformer"), torch_dtype=dtype).to(dev)
    model.eval()
    model.requires_grad_(False)
    print(f"transformer ready ({time.time()-t0:.0f}s)", flush=True)
    img_ids = latent_ids(32, 32, dev)
    txt_ids = torch.zeros(512, 3, device=dev)
    tok_len = lambda p: 512                                  # noqa: E731
    if args.txt_trim:
        assert not args.legacy
        # precompute.py / generate_pairs.encode run T5 with the padding attention mask, so
        # the first L outputs (prompt tokens + EOS) equal those of an unpadded encode
        from transformers import T5TokenizerFast
        tok2 = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"),
                                               max_length=512)
        tok_len = lambda p: len(tok2(p, max_length=512, truncation=True).input_ids)  # noqa: E731

    for K in args.K:
        subs = common.SUBJECTS[:K]
        task_pairs = load_pairs(subs, args.loras_root)
        comb = ssrlib.concat_pairs(task_pairs)

        if args.legacy:
            # verbatim test_spectral.run_task (same embedding cache, RNG and t)
            from test_spectral import prompt_embeds
            emb = prompt_embeds(subs, f"K{K}", dev, dtype)
            base = [f"a {common.TOKEN} {common.CLASSES[s]}" for s in subs]
            idx = {p: i for i, p in enumerate(sorted(set(base)))}
            E, P = emb["E"].to(dev), emb["P"].to(dev)
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
                        img_ids=img_ids, txt_ids=txt_ids,
                        guidance=torch.full((1,), 3.5, device=dev, dtype=dtype),
                        return_dict=False)
            n_steps = 1
        else:
            pair_emb = {k: [(pair_cache[p], tok_len(p)) for p in v]
                        for k, v in pair_texts.get(K, {}).items()}
            data = {s: torch.load(os.path.join(common.DATA, f"{s}.pt"), weights_only=False)
                    for s in subs}
            for s in subs:
                assert data[s]["prompts"] == common.all_prompts(s), s
            L = {(k, j): tok_len(data[s]["prompts"][j]) for k, s in enumerate(subs)
                 for j in args.prompt_ids}
            if args.txt_trim:
                print(f"[K{K}] trimmed text lengths of {subs[0]}: "
                      f"{[L[(0, j)] for j in args.prompt_ids]}", flush=True)
            ts = torch.tensor(args.timesteps, device=dev, dtype=dtype)
            nt = len(args.timesteps)

            def run_task(mdl, k, n_steps):
                d = data[subs[k]]
                gen = torch.Generator(device=dev).manual_seed(args.seed * 1000 + k)
                for pi, j in enumerate(args.prompt_ids):
                    if args.x0 == "real":
                        lat = d["latents"][pi % d["latents"].shape[0]]
                        x0 = lat.to(dev, dtype).unsqueeze(0).expand(nt, -1, -1, -1)
                    else:
                        x0 = torch.randn(nt, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
                    noise = torch.randn(nt, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
                    xt = (1 - ts.view(-1, 1, 1, 1)) * x0 + ts.view(-1, 1, 1, 1) * noise
                    n_txt = L[(k, j)]
                    e = d["embeds"][j][:n_txt].to(dev, dtype).unsqueeze(0).expand(nt, -1, -1)
                    p = d["pooled"][j].to(dev, dtype).unsqueeze(0).expand(nt, -1)
                    mdl(hidden_states=pack_latents(xt), timestep=ts,
                        encoder_hidden_states=e, pooled_projections=p,
                        img_ids=img_ids, txt_ids=txt_ids[:n_txt],
                        guidance=torch.full((nt,), 3.5, device=dev, dtype=dtype),
                        return_dict=False)
                for (E2, P2), n_txt in pair_emb.get(k, []):
                    x0 = torch.randn(nt, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
                    noise = torch.randn(nt, 16, 64, 64, device=dev, dtype=dtype, generator=gen)
                    xt = (1 - ts.view(-1, 1, 1, 1)) * x0 + ts.view(-1, 1, 1, 1) * noise
                    mdl(hidden_states=pack_latents(xt), timestep=ts,
                        encoder_hidden_states=E2[:n_txt].to(dev, dtype).unsqueeze(0).expand(nt, -1, -1),
                        pooled_projections=P2.to(dev, dtype).unsqueeze(0).expand(nt, -1),
                        img_ids=img_ids, txt_ids=txt_ids[:n_txt],
                        guidance=torch.full((nt,), 3.5, device=dev, dtype=dtype),
                        return_dict=False)
            n_steps = len(args.prompt_ids) + args.pair_prompts

        t1 = time.time()
        stats = ssrlib.accumulate(comb, task_pairs, model, run_task, n_steps=n_steps,
                                  seed=args.seed, keep_tasks=True, compute_interference=True)
        print(f"[K{K}] calibrated in {time.time()-t1:.0f}s", flush=True)
        if args.check:
            check(stats, args.check.format(K=K))
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, f"stats_K{K}.pt")
            assert not os.path.exists(out), f"{out} exists"
            for s in stats.values():          # same layout as test_spectral (CPU tensors)
                for key in ("G", "Q", "mu"):
                    if s.get(key) is not None:
                        s[key] = s[key].cpu()
            torch.save(stats, out)
            print(f"[K{K}] wrote {out}", flush=True)
        del stats
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

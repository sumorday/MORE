"""Spherical max-margin + cross-step residual routing: one-click prototype benchmark on pool A
(loras_seed2, calibration data/seed2_calNmT3) against the static merges.  Library: src/spherical_routing.

Stages (--stage, run in this order; each reads the previous one's files):
  diag   (CPU)  per layer: cosines of the differential parts (common mode removed / not removed,
                data / fro metric) of the pico merge and of the raw own LoRAs; what max_margin does
                at m0 = 0.2 and at the quantile m0; alpha_shared / alpha_spec / kappa of the
                undamped merge; reconstruction check of the factored merge against the static
                arm's safetensors.  -> results/sph_diag_<base>_K<K>.json
  build  (CPU)  per-layer runtime factors (bf16) of the variants  -> data/sph/<base>_K<K>_<var>.pt
                  nm      pico merge, Cw = I  (no margin: the static merge, split)
                  mm<m0>  pico merge, hard max-margin (data metric) at m0 (float or q<p>)
                  ad<m0>  Component 2: per-layer damping strength s0 w_l (thresh 0) and a soft
                          margin with lam_l = lam0 w_l, w_l = 1 - alpha_spec (undamped merge)
                  raw     own LoRAs B_k A_k, no shared part (gated multi-LoRA baseline)
                  merge baselines in the same factored form (merge ablation under routing; the
                  build checks each against its static safetensors arm):
                  np      rc merge without de-crowding (s2calNmT3_ssr_rc_*_g1)
                  avg     task arithmetic B_k A_k / K (s2_avg)
                  ssrm    SSR-Merge as published, R = Q (G/n + 1e-4 I)^-1, 1-prompt calibration
                          data/seed2 (s2_ssr_lam0.0001);  ssrmc: same on our calibration
                  gp<z|x><lam>  ssrmc + random-path gradient penalty (random_gp.py): R pulled
                          toward I/K in the z (latent) or x (weight) Frobenius metric
                  x<lk|oc><g>  ssrmc with a cross-subject interference penalty (ortho_reg.py):
                          lk crosstalk-weighted LS (g = 1: ssrmc), oc off-diagonal G scaled by g
  gen    (GPU)  single-subject (generate_table1 protocol: ENVS[:4], seeds 0 1, seed sd*1000+i)
                and two-subject (generate_pairs: 10 pairs x 2 templates, seeds 0 1) images of
                each arm, one FLUX load.  -> gen_sph/t1_<arm>, gen_sph_pairs/<arm> (+ meta.json)
                --check first compares the hooked static merge (all subjects, ungated) with the
                static safetensors loaded through diffusers on a few images.
  eval   (GPU)  eval_metrics.py / eval_pairs_det.py on the arms -> results/sph_{t1,pairs}_K<K>.json
  table         the arms next to the static merges, base, pairsum (existing result files)

Arm name: sph_<base>_<var>_s<s_split>_<gate>[_t<temp>][_c<b0>-<b1>]_K<K>, gate g = attention-map
softmax mask, gb = softmax with background channel, gm = softmax / per-token max (dominant subject
keeps its full part), u = ungated (every prompt subject on every token); a trailing "p" (gp, gmp,
up) keeps all other subjects of the merge as passive (weight 1 everywhere: up == the static
merge, gating only redistributes the prompt subjects' parts).  t<temp>: mask temperature,
c<b0>-<b1>: double blocks whose attention maps are used.
In the single-subject protocol the prompt's subject is known, so the dynamic arm keeps the
shared part + that subject's own differential part only (an oracle-style lookup, not a merge).

  python scripts/run_spherical_proto.py --stage diag --K 10
  python scripts/run_spherical_proto.py --stage gen --K 10 --arms sph_rc_nm_s0.2_g_K10 ...
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import common
import ssrlib
import random_gp
import ortho_reg
from solve_from_stats import load_pairs
from spherical_routing import curvature as cv
from spherical_routing import spherical_svm as ss

D = torch.float64
STATS_ROOT = os.path.join(common.ROOT, "data", "seed2_calNmT3")
LORAS_ROOT = os.path.join(common.ROOT, "loras_seed2")
FAC_ROOT = os.path.join(common.ROOT, "data", "sph")
GEN_T1 = os.path.join(common.ROOT, "gen_sph")
GEN_PAIRS = os.path.join(common.ROOT, "gen_sph_pairs")
PICO = dict(strength=0.5, thresh=0.5, m=8, rescale=False, ctrl="share")
# main-line static arms of pool A (rank / tau per K as in the paper); "full" = ssr_full
BASES = {("rc", 10): dict(rank=8, tau=0.05), ("rc", 20): dict(rank=16, tau=0.1),
         ("full", 10): dict(rank=None, tau=0.05), ("full", 20): dict(rank=None, tau=0.1)}


def static_stem(base, K):
    c = BASES[(base, K)]
    mode = "ssr_full" if c["rank"] is None else f"ssr_rc_r{c['rank']}"
    return f"s2calNmT3_{mode}_tau{c['tau']:g}_g1_pico_K{K}"


def load_inputs(K):
    t0 = time.time()
    stats = torch.load(os.path.join(STATS_ROOT, f"stats_K{K}.pt"), map_location="cpu",
                       weights_only=False)
    subs = common.SUBJECTS[:K]
    comb = ssrlib.concat_pairs(load_pairs(subs, LORAS_ROOT))
    print(f"[K{K}] {len(comb)} LoRA layers, {len(stats)} stats layers, {time.time() - t0:.0f}s",
          flush=True)
    return stats, comb, subs


def offdiag(S):
    iu = torch.triu_indices(S.shape[0], S.shape[0], 1)
    return S[iu[0], iu[1]]


def cos_summary(S):
    v = offdiag(S)
    return dict(max=float(v.max()), q80=float(torch.quantile(v, 0.8)), mean=float(v.mean()),
                mean_abs=float(v.abs().mean()), frac_gt02=float((v > 0.2).double().mean()))


def mtype(name):
    """single.to_qkv / double.attn_img / double.attn_txt / double.ff_img / double.ff_txt."""
    if name.startswith("single"):
        return "single.qkv"
    if "ff_context" in name:
        return "double.ff_txt"
    if ".ff." in name:
        return "double.ff_img"
    if any(t in name for t in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")):
        return "double.attn_txt"
    return "double.attn_img"


# ---------------------------------------------------------------------------------- diag
def stage_diag(args):
    stats, comb, subs = load_inputs(args.K)
    c = BASES[(args.base, args.K)]
    stem = static_stem(args.base, args.K)
    from safetensors.torch import load_file
    saved = common.extract_lora_pairs(load_file(os.path.join(common.MERGED, f"{stem}.safetensors")))
    gen = torch.Generator().manual_seed(0)
    out, recon = {}, []
    t0 = time.time()
    for li, (name, (A, B, rd, acts)) in enumerate(comb.items()):
        s = stats.get(name)
        if s is None or s["n"] == 0 or len(acts) != args.K:
            out[name] = dict(skip=True)
            continue
        rt = ss.route_layer(s, A, B, rd, c["tau"], 1.0, PICO, rank=c["rank"])
        rt0 = ss.route_layer(s, A, B, rd, c["tau"], 1.0, None, rank=c["rank"])  # undamped
        U, U0 = rt["U_s"], rt["U_s"][:, :0]
        Kr = A.shape[0]
        I = torch.eye(Kr, dtype=D)
        rec = dict(type=mtype(name), n_shared=int(U.shape[1]), g=rt["g"],
                   shares_top=[round(float(x), 4) for x in rt["shares"][:3]])
        for metric in ("data", "fro"):
            W = ss.latent_metric(s, A, metric)
            S, n, _ = ss.cosines(ss.part_gram(rt["B"], rt["R"], rt["g"], U, W, rd))
            S_nc, _, _ = ss.cosines(ss.part_gram(rt["B"], rt["R"], rt["g"], U0, W, rd))
            S_raw, _, _ = ss.cosines(ss.part_gram(rt["B_orig"], I, 1.0, U, W, rd))
            S_rawnc, _, _ = ss.cosines(ss.part_gram(rt["B_orig"], I, 1.0, U0, W, rd))
            rec[metric] = dict(merge=cos_summary(S), merge_nocm=cos_summary(S_nc),
                               raw=cos_summary(S_raw), raw_nocm=cos_summary(S_rawnc),
                               part_norm_cv=float(n.std() / n.mean()))
            if metric == "data":
                for m0 in (0.2, "q0.8"):
                    mv = ss.resolve_m0(S, m0)
                    _, info = ss.max_margin(S, mv, lam=10.0, iters=20, hard=True)
                    rec[f"mm_{m0}"] = {k: (round(v, 5) if isinstance(v, float) else v)
                                       for k, v in info.items()}
        M0 = rt0["g"] * rt0["B"] @ rt0["R"]
        M1 = rt["g"] * rt["B"] @ rt["R"]
        a_sh, a_sp, o_sh, o_sp = cv.retention(rt["B_orig"], M0, U, s["tasks"])
        p_sh, p_sp, _, _ = cv.retention(rt["B_orig"], M1, U, s["tasks"])
        rec.update(alpha_shared=a_sh, alpha_spec=a_sp, own_shared=o_sh, own_spec=o_sp,
                   alpha_shared_pico=p_sh, alpha_spec_pico=p_sp,
                   kappa_spec=cv.kappa(a_sh, a_sp, "spec"), kappa_cmrr=cv.kappa(a_sh, a_sp, "cmrr"))
        # reconstruction: factored static merge vs the saved arm (fp32 file), random probes
        f = ss.merge_factors(rt, A, rd)
        A_o, B_o = saved[name]
        X = torch.randn(A.shape[1], 4, generator=gen, dtype=D)
        y_ref = B_o.double() @ (A_o.double() @ X)
        y = ss.static_B(f) @ (f.A_r @ X)
        rec["recon_rel"] = float((y - y_ref).norm() / y_ref.norm())
        recon.append(rec["recon_rel"])
        out[name] = rec
        if li % 40 == 0:
            print(f"  {li:3d} {name:52s} cos(data) max {rec['data']['merge']['max']:.3f} "
                  f"nocm {rec['data']['merge_nocm']['max']:.3f} raw {rec['data']['raw']['max']:.3f} "
                  f"a_sp {a_sp:.3f} recon {rec['recon_rel']:.1e}  {time.time() - t0:.0f}s",
                  flush=True)
    summ = diag_summary(out)
    summ["recon_rel_max"] = max(recon)
    os.makedirs(os.path.join(common.ROOT, "results"), exist_ok=True)
    fn = os.path.join(common.ROOT, "results", f"sph_diag_{args.base}_K{args.K}.json")
    json.dump(dict(static_arm=stem, summary=summ, layers=out), open(fn, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("saved", fn)


def _q(v, qs=(0.1, 0.5, 0.9)):
    t = torch.tensor(v, dtype=D)
    return [round(float(torch.quantile(t, q)), 4) for q in qs] if len(v) else None


def diag_summary(out):
    L = [r for r in out.values() if not r.get("skip")]
    summ = dict(n_layers=len(L), n_skipped=len(out) - len(L),
                n_with_shared=sum(r["n_shared"] > 0 for r in L))
    for metric in ("data", "fro"):
        for part in ("merge", "merge_nocm", "raw", "raw_nocm"):
            summ[f"{metric}.{part}"] = {
                k: _q([r[metric][part][k] for r in L]) for k in ("max", "q80", "mean", "mean_abs")}
            summ[f"{metric}.{part}"]["layers_max_gt0.2"] = sum(
                r[metric][part]["max"] > 0.2 for r in L)
    for m0 in ("0.2", "q0.8"):
        I = [r[f"mm_{m0}"] for r in L]
        act = [x for x in I if x["active_pairs"] > 0]
        summ[f"mm_{m0}"] = dict(layers_active=len(act),
                                pairs_active=sum(x["active_pairs"] for x in I),
                                m0=_q([x["m0"] for x in I]),
                                fidelity_active=_q([x["fidelity"] for x in act]),
                                infeasible=sum(not x["feasible"] for x in I))
    for k in ("alpha_shared", "alpha_spec", "alpha_shared_pico", "alpha_spec_pico", "kappa_spec",
              "kappa_cmrr"):
        summ[k] = _q([r[k] for r in L if math.isfinite(r[k])])
    by = {}
    for r in L:
        by.setdefault(r["type"], []).append(r)
    summ["by_type"] = {t: dict(n=len(v), cos_data_max=_q([r["data"]["merge"]["max"] for r in v]),
                               cos_data_nocm_max=_q([r["data"]["merge_nocm"]["max"] for r in v]),
                               alpha_spec=_q([r["alpha_spec"] for r in v]),
                               n_shared=sum(r["n_shared"] for r in v))
                       for t, v in sorted(by.items())}
    return summ


# --------------------------------------------------------------------------------- build
def parse_m0(x):
    return x if x.startswith("q") else float(x)


def stage_build(args):
    stats, comb, subs = load_inputs(args.K)
    c = BASES[(args.base, args.K)]
    os.makedirs(FAC_ROOT, exist_ok=True)
    todo = [v for v in args.variants
            if args.overwrite or not os.path.exists(fac_path(args.base, args.K, v))]
    if not todo:
        print("all variants exist")
        return
    facs = {v: {} for v in todo}
    info = {v: {} for v in todo}
    leg = None
    if "ssrm" in todo:
        leg = torch.load(os.path.join(common.ROOT, "data", "seed2", f"stats_K{args.K}.pt"),
                         map_location="cpu", weights_only=False)
    saved, recon = {}, {}
    from safetensors.torch import load_file
    for v in todo:
        if v in BASELINE_STATIC:
            stem = BASELINE_STATIC[v](args.base, args.K)
            saved[v] = common.extract_lora_pairs(
                load_file(os.path.join(common.MERGED, f"{stem}.safetensors")))
            recon[v] = []
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    for li, (name, (A, B, rd, acts)) in enumerate(comb.items()):
        s = stats.get(name)
        if s is None or s["n"] == 0 or len(acts) != args.K:
            raise RuntimeError(f"{name}: no stats / not all subjects (static arm passes it through)")
        rt = rt0 = None
        for v in todo:
            if v == "raw":
                facs[v][name] = ss.raw_factors(A, B, rd)
                continue
            if v[:3] in ("xlk", "xoc"):
                R, xi = ortho_reg.xtalk_solve(s, float(v[3:]), v[1:3])
                U_s, _ = ss.shared_dirs(B.to(D), rd, 0.5, 8)
                facs[v][name] = ss.merge_factors(dict(B=B.to(D), R=R, g=1.0, U_s=U_s), A, rd)
                own, leak = ortho_reg.diagnose(s, R)
                info[v][name] = dict(g=1.0, own=own, leak=leak, **xi)
                continue
            if v.startswith("gp"):
                # ssrmc + random-path gradient penalty (scripts/random_gp.py), closed form
                sp, lam = v[2], float(v[3:])
                R, cg = random_gp.gp_solve(s["G"], s["Q"], s["n"], args.K, lam, sp, B, A)
                U_s, _ = ss.shared_dirs(B.to(D), rd, 0.5, 8)
                facs[v][name] = ss.merge_factors(dict(B=B.to(D), R=R, g=1.0, U_s=U_s), A, rd)
                info[v][name] = dict(g=1.0, c=cg, dist_avg=float(
                    (R - torch.eye(R.shape[0], dtype=D) / args.K).norm()))
                continue
            if v in BASELINE_STATIC:
                if v == "np":
                    rb = ss.route_layer(s, A, B, rd, c["tau"], 1.0, None, rank=c["rank"])
                else:
                    rb = baseline_route(v, leg.get(name) if v == "ssrm" else s, A, B, rd)
                f = ss.merge_factors(rb, A, rd)
                A_o, B_o = saved[v][name]
                X = torch.randn(A.shape[1], 4, generator=gen, dtype=D)
                y_ref = B_o.double() @ (A_o.double() @ X)
                recon[v].append(float((ss.static_B(f) @ (f.A_r @ X) - y_ref).norm()
                                      / y_ref.norm().clamp_min(1e-300)))
                facs[v][name] = f
                info[v][name] = dict(g=rb["g"], recon_rel=recon[v][-1])
                continue
            if v == "nm" or v.startswith("mm"):
                rt = rt or ss.route_layer(s, A, B, rd, c["tau"], 1.0, PICO, rank=c["rank"])
                Cw, inf = None, {}
                if v.startswith("mm"):
                    W = ss.latent_metric(s, A, "data")
                    S, n, _ = ss.cosines(ss.part_gram(rt["B"], rt["R"], rt["g"], rt["U_s"], W, rd))
                    C, inf = ss.max_margin(S, ss.resolve_m0(S, parse_m0(v[2:])), lam=10.0,
                                           iters=20, hard=True)
                    Cw = ss.weight_coefs(C, S, n)
                facs[v][name] = ss.merge_factors(rt, A, rd, Cw)
                info[v][name] = dict(g=rt["g"], n_shared=int(rt["U_s"].shape[1]), **inf)
            elif v.startswith("ad"):
                # Component 2: crowding of the undamped merge sets this layer's damping + margin
                rt0 = rt0 or ss.route_layer(s, A, B, rd, c["tau"], 1.0, None, rank=c["rank"])
                a_sh, a_sp, _, _ = cv.retention(rt0["B_orig"], rt0["g"] * rt0["B"] @ rt0["R"],
                                                rt0["U_s"], s["tasks"])
                sch = cv.layer_schedule(a_sh, a_sp, "spec", "frac", lam0=args.lam0, s0=args.s0)
                pico = dict(strength=sch["damp_strength"], thresh=0.0, m=8, rescale=False,
                            ctrl="share")
                ra = ss.route_layer(s, A, B, rd, c["tau"], 1.0, pico, rank=c["rank"])
                W = ss.latent_metric(s, A, "data")
                S, n, _ = ss.cosines(ss.part_gram(ra["B"], ra["R"], ra["g"], ra["U_s"], W, rd))
                Cw, inf = None, {}
                if sch["lam_margin"] > 0:
                    C, inf = ss.max_margin(S, ss.resolve_m0(S, parse_m0(v[2:])),
                                           lam=sch["lam_margin"], iters=20, hard=False)
                    Cw = ss.weight_coefs(C, S, n)
                facs[v][name] = ss.merge_factors(ra, A, rd, Cw)
                info[v][name] = dict(g=ra["g"], alpha_spec=a_sp, **sch, **inf)
            else:
                raise ValueError(f"variant {v!r}")
        # bf16 per layer as we go: float64 factors of several K20 variants do not fit in memory
        for v in todo:
            facs[v][name] = {k: (None if t is None else t.to(torch.bfloat16).contiguous())
                             for k, t in vars(facs[v][name]).items()}
        if li % 40 == 0:
            print(f"  {li:3d} {name}  {time.time() - t0:.0f}s", flush=True)
    for v, r in recon.items():
        print(f"[{v}] reconstruction vs {BASELINE_STATIC[v](args.base, args.K)}: max rel "
              f"{max(r):.2e}, median {sorted(r)[len(r) // 2]:.2e}", flush=True)
        if max(r) > 1e-3:
            raise RuntimeError(f"{v}: factored merge does not match its static arm")
    for v in todo:
        fn = fac_path(args.base, args.K, v)
        tmp = fn + ".tmp"
        torch.save(dict(factors=facs[v], info=info[v], subjects=subs, base=args.base,
                        static_arm=static_stem(args.base, args.K)), tmp)
        os.replace(tmp, fn)
        L = list(info[v].values())
        act = [x for x in L if x.get("active_pairs", 0) > 0]
        print(f"[{v}] -> {fn} ({os.path.getsize(fn) / 2**30:.2f} GiB); margin active in "
              f"{len(act)}/{len(L)} layers"
              + (f", fidelity med {sorted(x['fidelity'] for x in act)[len(act) // 2]:.3f}"
                 if act else ""), flush=True)


# merge-baseline variants -> the static safetensors arm the factored build must reproduce
BASELINE_STATIC = {
    "np": lambda base, K: static_stem(base, K).replace("_pico", ""),
    "avg": lambda base, K: f"s2_avg_K{K}",
    "ssrm": lambda base, K: f"s2_ssr_lam0.0001_K{K}",
    "ssrmc": lambda base, K: f"s2calNmT3_ssr_lam0.0001_K{K}",
}


def baseline_route(v, s, A_comb, B_comb, r_dims):
    """route_layer-style dict (g 1, same common-mode split) of a baseline dW = B_comb R A_comb:
    avg R = I/K; ssrm/ssrmc SSR-Merge as published, R = Q (G/n + 1e-4 I)^-1 (ssrlib.ssr_solve
    ridge abs; a layer without stats passes the LoRAs through, R = I, as ssr_solve does)."""
    Kr = A_comb.shape[0]
    if v == "avg":
        R = torch.eye(Kr, dtype=D) / len(r_dims)
    elif s is None or s["n"] == 0:
        R = torch.eye(Kr, dtype=D)
    else:
        R = ssrlib._ridge_solve(s["G"].to(D), s["Q"].to(D), 1e-4 * s["n"], "zero")
    B = B_comb.to(D)
    U_s, _ = ss.shared_dirs(B, r_dims, 0.5, 8)
    return dict(B=B, R=R, g=1.0, U_s=U_s)


def fac_path(base, K, var):
    return os.path.join(FAC_ROOT, f"{base}_K{K}_{var}.pt")


def load_factors(base, K, var):
    d = torch.load(fac_path(base, K, var), map_location="cpu", weights_only=False)
    return {n: ss.LayerFactors(**t) for n, t in d["factors"].items()}, d


# ----------------------------------------------------------------------------------- gen
GATES = {"g": "softmax", "gb": "softmax_bg", "gm": "softmax_max", "u": None}


def parse_arm(arm, args=None):
    """sph_<base>_<var>_s<s_split>_<gate>[_t<temp>][_c<b0>-<b1>]_K<K> -> dict; temp / capture
    blocks (double blocks b0..b1) default to --temp / all blocks."""
    p = arm.split("_")
    assert p[0] == "sph" and p[-1].startswith("K") and p[3].startswith("s"), arm
    gate = p[4]
    passive = gate.endswith("p")
    gate = gate[:-1] if passive else gate
    assert gate in GATES, arm
    a = dict(base=p[1], var=p[2], s_split=float(p[3][1:]), gate=gate, passive=passive,
             K=int(p[-1][1:]), temp=None if args is None else args.temp, capture=None)
    for o in p[5:-1]:
        if o.startswith("t"):
            a["temp"] = float(o[1:])
        elif o.startswith("c"):
            b0, b1 = map(int, o[1:].split("-"))
            a["capture"] = list(range(b0, b1 + 1))
        else:
            raise ValueError(f"arm option {o!r} in {arm}")
    return a


def token_sets(tok, prompt, subjects):
    """T5 token indices of each subject's "sks <class>" span (first unused occurrence)."""
    enc = tok(prompt, return_offsets_mapping=True, max_length=512, truncation=True)
    offs = enc["offset_mapping"]
    out, used = [], []
    for s in subjects:
        phrase = f"{common.TOKEN} {common.CLASSES[s]}"
        start = 0
        while True:
            c0 = prompt.find(phrase, start)
            if c0 < 0:
                raise ValueError(f"{phrase!r} not in {prompt!r}")
            c1 = c0 + len(phrase)
            if all(c1 <= u0 or c0 >= u1 for u0, u1 in used):
                break
            start = c0 + 1
        used.append((c0, c1))
        idx = [i for i, (a, b) in enumerate(offs) if b > a and a < c1 and b > c0]
        assert idx, (prompt, s)
        out.append(idx)
    return out


def stage_gen(args):
    from diffusers import FluxPipeline
    from transformers import T5TokenizerFast
    from spherical_routing.dynamic_hook import DynamicRouter
    import generate_pairs as gp
    from generate_table1 import load_lora

    tok = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    emb_t1 = torch.load(os.path.join(common.DATA, "t1_embeds.pt"), weights_only=False)
    emb_pr = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline ready in {time.time() - t0:.0f}s", flush=True)

    def run(E, P, seed):
        g = torch.Generator(device=args.device).manual_seed(seed)
        return pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                    pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                    num_inference_steps=args.steps, guidance_scale=args.guidance,
                    height=args.res, width=args.res, generator=g).images[0]

    if args.check:
        check_static(args, pipe, run, emb_t1, load_lora, DynamicRouter)

    cache = {}
    for arm in args.arms:
        a = parse_arm(arm, args)
        K = a["K"]
        subs = common.SUBJECTS[:K]
        jobs = []
        if "t1" in args.protocols:
            meta = [{"subject": s, "prompt": p} for s in subs
                    for p in common.all_prompts(s)[:args.n_prompts]]
            jobs.append(("t1", os.path.join(GEN_T1, f"t1_{arm}{args.outsuffix}"), meta, emb_t1))
        if "pairs" in args.protocols:
            meta = gp.pair_prompts(K, args.n_pairs, args.n_templates)
            jobs.append(("pairs", os.path.join(GEN_PAIRS, arm + args.outsuffix), meta, emb_pr))
        jobs = [j for j in jobs if not os.path.exists(os.path.join(j[1], "meta.json"))]
        if not jobs:
            print(f"[{arm}] done, skip", flush=True)
            continue
        key = (a["base"], K, a["var"])
        if key not in cache:
            cache.clear()
            torch.cuda.empty_cache()
            cache[key] = load_factors(*key)[0]
        rt = DynamicRouter(pipe.transformer, cache[key], s_split=a["s_split"], ramp=args.ramp,
                           temp=a["temp"], mask=GATES[a["gate"]] or "softmax",
                           bg=args.bg, gating=a["gate"] != "u", capture_blocks=a["capture"],
                           dtype=torch.bfloat16)
        t1 = time.time()
        try:
            for proto, outdir, meta, emb in jobs:
                os.makedirs(outdir, exist_ok=True)
                logs = {}
                for i, m in enumerate(meta):
                    ss_ = m["subjects"] if proto == "pairs" else [m["subject"]]
                    tsets = token_sets(tok, m["prompt"], ss_)
                    E, P = emb[m["prompt"]]
                    for sd in args.seeds:
                        fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
                        if os.path.exists(fn):
                            continue
                        act = [subs.index(x) for x in ss_]
                        rt.set_prompt(act, tsets, args.steps, txt_len=E.shape[0],
                                      passive=[k for k in range(K) if k not in act]
                                      if a["passive"] else ())
                        im = run(E, P, sd * 1000 + i)
                        assert rt.step == args.steps - 1, (rt.step, args.steps)
                        im.save(fn)
                        if rt.log:
                            logs[f"{i:03d}_s{sd}"] = rt.log[-1]
                json.dump({"prompts": meta, "seeds": args.seeds, "steps": args.steps,
                           "guidance": args.guidance, "arm": a, "temp": a["temp"],
                           "ramp": args.ramp, "bg": args.bg, "mask_log_last_step": logs,
                           **({"subjects": subs} if proto == "t1" else {})},
                          open(os.path.join(outdir, "meta.json"), "w"), indent=1)
                print(f"[{arm}] {proto}: {len(meta) * len(args.seeds)} images, "
                      f"{time.time() - t1:.0f}s", flush=True)
        finally:
            rt.remove()
    print("done", flush=True)


def check_static(args, pipe, run, emb, load_lora, DynamicRouter):
    """Hooked static merge (all subjects, ungated, s_split 0) vs the static safetensors."""
    import numpy as np
    K = args.K
    subs = common.SUBJECTS[:K]
    f, _ = load_factors(args.base, K, "nm")
    prompts = [common.all_prompts(s)[0] for s in subs[:2]]
    seeds = [0, 4]                                          # sd 0 of t1 images 0 and 4
    ims_h, ims_s = [], []
    rt = DynamicRouter(pipe.transformer, f, s_split=0.0, gating=False, dtype=torch.bfloat16)
    for p, sd in zip(prompts, seeds):
        rt.set_prompt(list(range(K)), [[0]] * K, args.steps)
        ims_h.append(np.asarray(run(*emb[p], sd), dtype=np.float32))
    rt.remove()
    del f
    torch.cuda.empty_cache()
    load_lora(pipe, os.path.join(common.MERGED, f"{static_stem(args.base, K)}.safetensors"))
    for p, sd in zip(prompts, seeds):
        ims_s.append(np.asarray(run(*emb[p], sd), dtype=np.float32))
    pipe.delete_adapters("m")
    plain = np.asarray(run(*emb[prompts[0]], seeds[0]), dtype=np.float32)
    d = [float(np.abs(a - b).mean()) for a, b in zip(ims_h, ims_s)]
    d_plain = float(np.abs(plain - ims_s[0]).mean())
    print(f"[check] hooked static vs safetensors: mean |dpixel| {d} (0-255); "
          f"no-adapter vs static {d_plain:.2f}", flush=True)
    if max(d) > 0.25 * d_plain:
        raise RuntimeError("hooked static merge does not reproduce the static arm")


def _smooth_norm(a, k=3):
    """(..., N) maps -> 3x3 box-smoothed on the latent grid, each map rescaled to mean 1."""
    import torch.nn.functional as F
    sh, N = a.shape[:-1], a.shape[-1]
    side = int(round(math.sqrt(N)))
    g = F.avg_pool2d(a.reshape(-1, 1, side, side).float(), k, stride=1, padding=k // 2,
                     count_include_pad=False).reshape(*sh, N)
    return g / g.mean(-1, keepdim=True).clamp_min(1e-12)


def map_stats(a, temps=(0.5, 0.2, 0.1)):
    """a (2, N) smoothed mean-1 maps of the two subjects -> separability numbers."""
    x = a - a.mean(-1, keepdim=True)
    corr = float((x[0] * x[1]).sum() / (x[0].norm() * x[1].norm()).clamp_min(1e-12))
    out = dict(corr=corr, peak=[float(v) for v in a.max(-1).values])
    for T in temps:
        w = torch.softmax(a.clamp_min(1e-12).log() / T, 0)
        out[f"frac90_T{T}"] = float((w.max(0).values > 0.9).float().mean())
        out[f"share_T{T}"] = [float(v) for v in (w.argmax(0)[None] == torch.arange(2)[:, None])
                              .float().mean(-1)]
    return out


def stage_maps(args):
    """Gating diagnostic: per double block and step, the two subjects' joint-attention maps on the
    first pair prompts (seed 0), generated with the first arm of --arms (gated).
    -> results/sph_maps_K<K>.json (per block / block-range / step: correlation of the two maps,
       fraction of tokens a temperature-T mask assigns with weight > 0.9) + results/sph_maps_K<K>.png"""
    import numpy as np
    from PIL import Image
    from diffusers import FluxPipeline
    from transformers import T5TokenizerFast
    from spherical_routing.dynamic_hook import DynamicRouter, subject_maps
    import generate_pairs as gp

    a = parse_arm(args.arms[0], args)
    K = a["K"]
    subs = common.SUBJECTS[:K]
    tok = T5TokenizerFast.from_pretrained(os.path.join(common.MODEL, "tokenizer_2"))
    emb = torch.load(os.path.join(common.DATA, "pairs_embeds.pt"), weights_only=False)
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    f, _ = load_factors(a["base"], K, a["var"])
    rt = DynamicRouter(pipe.transformer, f, s_split=a["s_split"], temp=a["temp"],
                       mask=GATES[a["gate"]] or "softmax", bg=args.bg,
                       gating=True, dtype=torch.bfloat16)
    keep = sorted({1, 3, 7, 14, 21, args.steps - 1})
    rec = {}

    def add_map(query, key, txt_len):                     # DynamicRouter.add_map + recording
        with torch.no_grad():
            m = subject_maps(query, key, txt_len, [t.to(query.device) for t in rt.token_sets])
        rt._acc = m if rt._acc is None else rt._acc + m
        rt._n_acc += 1
        if rt.step in keep:
            rec.setdefault(rt.step, []).append(m[0].cpu())

    rt.add_map = add_map
    meta = gp.pair_prompts(K, args.n_pairs, args.n_templates)[:args.n_prompts]
    nb = len(pipe.transformer.transformer_blocks)
    ranges = {"all": range(nb), "0-6": range(0, 7), "7-12": range(7, 13), "13-18": range(13, nb)}
    stats, vis = [], []
    try:
        for i, m in enumerate(meta):
            tsets = token_sets(tok, m["prompt"], m["subjects"])
            E, P = emb[m["prompt"]]
            rec.clear()
            act = [subs.index(x) for x in m["subjects"]]
            rt.set_prompt(act, tsets, args.steps, txt_len=E.shape[0],
                          passive=[k for k in range(K) if k not in act] if a["passive"] else ())
            g = torch.Generator(device=args.device).manual_seed(i)
            im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                      pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                      num_inference_steps=args.steps, guidance_scale=args.guidance,
                      height=args.res, width=args.res, generator=g).images[0]
            st = {"prompt": m["prompt"], "steps": {}}
            for s, lst in rec.items():
                A = _smooth_norm(torch.stack(lst))        # (nb, 2, N)
                assert A.shape[0] == nb, (A.shape, nb)
                st["steps"][s] = dict(block=[map_stats(A[b]) for b in range(nb)],
                                      **{r: map_stats(_smooth_norm(torch.stack(
                                          [torch.stack(lst)[b] for b in rr]).mean(0)))
                                         for r, rr in ranges.items()})
            stats.append(st)
            last = _smooth_norm(torch.stack(rec[args.steps - 1]).mean(0))   # all-block average
            vis.append((im, last))
            print(f"[{i}] {m['prompt'][:60]}  corr(all, last step) "
                  f"{st['steps'][args.steps - 1]['all']['corr']:+.3f}", flush=True)
    finally:
        rt.remove()

    def agg(key, s, field):
        return float(np.mean([st["steps"][s][key][field] for st in stats]))
    summ = {}
    for s in keep:
        summ[s] = {r: {fld: agg(r, s, fld) for fld in ("corr", "frac90_T0.5", "frac90_T0.2",
                                                         "frac90_T0.1")} for r in ranges}
        summ[s]["block_corr"] = [float(np.mean([st["steps"][s]["block"][b]["corr"] for st in stats]))
                                 for b in range(nb)]
        print(f"step {s:2d}: " + "  ".join(f"{r} corr {summ[s][r]['corr']:+.3f} "
                                           f"f90(T.5/.2/.1) {summ[s][r]['frac90_T0.5']:.2f}/"
                                           f"{summ[s][r]['frac90_T0.2']:.2f}/"
                                           f"{summ[s][r]['frac90_T0.1']:.2f}" for r in ranges))
        print("   per-block corr " + " ".join(f"{v:+.2f}" for v in summ[s]["block_corr"]))
    out = os.path.join(common.ROOT, "results", f"sph_maps_K{K}{args.tag}")
    json.dump({"arm": args.arms[0], "summary": summ, "images": stats}, open(out + ".json", "w"))
    # png rows: image | map subject a | map subject b | mask_a at temp args.temp
    S = 192
    canvas = Image.new("RGB", (4 * S, len(vis) * S))
    side = int(round(math.sqrt(vis[0][1].shape[-1])))
    for r, (im, A) in enumerate(vis):
        w = torch.softmax(A.clamp_min(1e-12).log() / args.temp, 0)
        tiles = [im.resize((S, S))]
        for x in (A[0] / A.max(), A[1] / A.max(), w[0]):
            arr = (x.reshape(side, side).clamp(0, 1).numpy() * 255).astype(np.uint8)
            tiles.append(Image.fromarray(arr).resize((S, S), Image.NEAREST).convert("RGB"))
        for c, t in enumerate(tiles):
            canvas.paste(t, (c * S, r * S))
    canvas.save(out + ".png")
    print(f"-> {out}.json / .png", flush=True)


# ---------------------------------------------------------------------------------- eval
def stage_eval(args):
    py = sys.executable
    sd = os.path.dirname(__file__)
    t1 = [f"t1_{a}" for a in args.arms if os.path.exists(os.path.join(GEN_T1, f"t1_{a}", "meta.json"))]
    pr = [a for a in args.arms if os.path.exists(os.path.join(GEN_PAIRS, a, "meta.json"))]
    if t1:
        subprocess.run([py, os.path.join(sd, "eval_metrics.py"), "--genroot", GEN_T1, "--tags",
                        *t1, "--out", os.path.join(common.ROOT, "results",
                                                   f"sph_t1_K{args.K}{args.tag}.json")], check=True)
    if pr:
        subprocess.run([py, os.path.join(sd, "eval_pairs_det.py"), "--genroot", GEN_PAIRS,
                        "--tags", *pr, "--out", os.path.join(common.ROOT, "results",
                                                             f"sph_pairs_K{args.K}{args.tag}.json")],
                       check=True)


# --------------------------------------------------------------------------------- table
def _load(fn):
    fn = os.path.join(common.ROOT, "results", fn)
    if not os.path.exists(fn):
        return {}
    d = json.load(open(fn))
    return d.get("arms", d)


def stage_table(args):
    K = args.K
    subs = common.SUBJECTS[:K]
    t1, pr = {}, {}
    for fn in sorted(os.listdir(os.path.join(common.ROOT, "results"))):
        if fn.startswith(f"sph_t1_K{K}"):
            t1.update(_load(fn))
        if fn.startswith(f"sph_pairs_K{K}"):
            pr.update(_load(fn))
    # static references (seeds 0 1)
    ref_t1 = {"static rc+pico": _load(f"scr_pico_K{K}_s01.json").get(
                  f"t1_{static_stem('rc', K)}"),
              "static full+pico": _load(f"v2svc_K{K}_s01.json").get(f"t1_{static_stem('full', K)}")}
    base = _load("v2_base_s01.json").get("t1_base_K20")
    if base:
        ps = base["per_subject_dino"]
        ref_t1["base (no adapter)"] = dict(dino=sum(ps[s] for s in subs) / K, per_subject_dino=ps)
    ref_pr = {"static rc+pico": _load(f"pairs_det_full_K{K}_s01.json").get(static_stem("rc", K)),
              "static full+pico": _load(f"pairs_det_full_K{K}_s01.json").get(static_stem("full", K)),
              "base (no adapter)": _load(f"pairs_det_full_K{K}_s01.json").get(f"base_K{K}"),
              "pairsum0.7 (own 2 LoRAs)": _load(f"pairs_det_K{K}_s01.json").get(f"pairsum0.7_K{K}")}

    def worst(r):
        ps = r.get("per_subject_dino") or {}
        v = sorted(ps[s] for s in subs if ps.get(s) is not None)
        return sum(v[:3]) / 3 if len(v) >= 3 else float("nan")

    print(f"\n=== K{K} single subject (40 prompts x seeds 0 1 per K10; DINO / CLIP-I / worst-3)")
    for name, r in list(ref_t1.items()) + [(k[3:], v) for k, v in t1.items()]:
        if r:
            print(f"  {name:44s} DINO {r['dino']:.4f}  CLIP-I {r.get('clip', float('nan')):.4f}  "
                  f"worst3 {worst(r):.4f}")
    print(f"\n=== K{K} two subjects (detect-then-DINO crops; n=40)")
    for name, r in list(ref_pr.items()) + list(pr.items()):
        if r:
            print(f"  {name:44s} min {r['min_dino']:.4f}  mean {r['mean_dino']:.4f}  "
                  f"both {r['both_detected']:.3f}  minDet {r['min_dino_det']:.4f}")
    pooled_pairs(pr, K)


def _pair_metrics(rows, thr=0.2):
    """per-image rows [i, sd, dino_a, dino_b, det_a, det_b] -> (min, mean, both, minDet) arrays."""
    import numpy as np
    x = np.asarray([r[2:] for r in rows], dtype=np.float64)
    da, db = x[:, 0], x[:, 1]
    ok_a, ok_b = x[:, 2] >= thr, x[:, 3] >= thr
    return dict(min=np.minimum(da, db), mean=(da + db) / 2, both=(ok_a & ok_b).astype(float),
                minDet=np.minimum(da * ok_a, db * ok_b))


def pooled_pairs(pr, K, ref=None, n_boot=2000):
    """Arms pooled over their extra-seed dirs (<arm>_s*), and paired differences against the
    hooked static merge (same prompts and seeds; bootstrap SE over images)."""
    import numpy as np
    ref = ref or f"sph_rc_nm_s0_up_K{K}"
    groups = {}
    for name, r in pr.items():
        base = name.split(f"_K{K}")[0] + f"_K{K}"
        groups.setdefault(base, {}).update({(int(x[0]), int(x[1])): x for x in r["per_image"]})
    if ref not in groups:
        return
    R = groups[ref]
    rng = np.random.default_rng(0)
    print(f"\n=== K{K} two subjects, pooled seeds; paired diff vs {ref} (bootstrap SE)")
    for name, rows in sorted(groups.items()):
        keys = sorted(set(rows) & set(R))
        m = _pair_metrics([rows[k] for k in sorted(rows)])
        line = (f"  {name:44s} n={len(rows):3d} min {m['min'].mean():.4f} mean "
                f"{m['mean'].mean():.4f} both {m['both'].mean():.3f} minDet {m['minDet'].mean():.4f}")
        if name != ref and len(keys) >= 10:
            a, b = _pair_metrics([rows[k] for k in keys]), _pair_metrics([R[k] for k in keys])
            idx = rng.integers(0, len(keys), (n_boot, len(keys)))
            parts = []
            for f in ("min", "mean", "both"):
                d = a[f] - b[f]
                parts.append(f"d{f} {d.mean():+.4f}±{d[idx].mean(1).std():.4f}")
            line += f" | n_paired {len(keys)} " + " ".join(parts)
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("diag", "build", "gen", "maps", "eval", "table"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--base", default="rc", choices=("rc", "full"))
    ap.add_argument("--variants", nargs="+", default=["nm", "mmq0.8", "raw"])
    ap.add_argument("--lam0", type=float, default=10.0)
    ap.add_argument("--s0", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--protocols", nargs="+", default=["t1", "pairs"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_templates", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=0.0)
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outsuffix", default="",
                    help="gen: appended to the output dirs (e.g. _s2-5 for extra --seeds); eval: "
                         "pass the suffixed names as --arms")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    {"diag": stage_diag, "build": stage_build, "gen": stage_gen, "maps": stage_maps,
     "eval": stage_eval,
     "table": stage_table}[args.stage](args)


if __name__ == "__main__":
    main()

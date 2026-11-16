"""Task-Conditioned Projection Routing (TCPR): direction-only SSR routers.

Solve-stage only.  G/Q come from ssrlib.accumulate (the official SSR hook:
G += z z^T, Q[start_k:end_k] += z_k z^T during task k's calibration), saved by
merge_more.py --save_stats.  Nothing here touches accumulation.

Per layer, with G, Q count-normalised (/n) and K active tasks:

    G_reg = G + lam I,         lam = tau * mean(diag G)            (rel ridge)
    ssr_full : R = Q G_reg^-1                                      (official form)
    ssr_dir  : R = Q G_reg^-1, then direction-only + global gain
    ssr_cross: R = (G/K) G_reg^-1 + cross * (Q - G/K) G_reg^-1    (whitened avg
               + scaled cross-task correction; cross=1 == ssr_dir)
    acr      : ssr_cross with a per-layer cross alpha_l read off G/Q
               (--cross norm: ||R_res|| / (||R_avg|| + ||R_res||);
                --cross trace: clip((tr Q - tr G/K) / tr G, 0, 1))
    optional : R <- R * M, M[i,j] = 1 if rows i and column j share a task block,
               else 1 - block_strength                              (Option B)
    regmean  : R = sum_k P_k Gt_k (sum_k Gt_k)^-1,  Gt_k = a G_k + (1-a) diag(G_k)
               P_k selects task k's rows; G_k is task k's own-forward Gram.
               RegMean's objective sum_k ||dW X_k - dW_k X_k||^2 restricted to
               dW = B_comb R A_comb has normal equations
               B_comb^T B_comb (R G - sum_k P_k G_k) = 0, so at a=1 this is
               R = Q G^-1, i.e. SSR with lam=0.  a<1 is RegMean's own
               off-diagonal shrinkage, applied per task.
    norm     : 'col'  R <- g * R / ||R[:, j]||   (Option A, unit columns)
               'data' R <- c * R, c scalar s.t. merged output energy on the
                      calibration data = g^2 * sum_k own-delta energy
                      (== the guni arms of solve_from_stats.py)
               'coldata' col, then data: per-layer gain identical to 'data',
                      so the only difference is the column reweighting
    B_merged = B_comb R,  A_merged = A_comb

Note the two norms use g in different units.  'data' rescales R by one scalar
per layer, so the direction of Delta W is the SSR direction exactly.  'col'
rescales each latent coordinate separately (R D, D diagonal), which changes
Delta W's direction too; check the reported data-space gain before generating.

  python scripts/tcpr.py --mode ssr_full ssr_dir ssr_cross --cross 0 0.5 1 \\
      --K 3 --stats_root data/seed --loras_root loras_seed --prefix seed_ --diagnose 4
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common
import ssrlib
from solve_from_stats import load_pairs

MODES = ("ssr_full", "ssr_dir", "ssr_cross", "regmean", "acr")
NORMS = ("col", "data", "coldata", "none")


def _blocks(r_dims):
    """[(start, end)] of each task's slice in the concatenated rank space."""
    out, s = [], 0
    for r in r_dims:
        out.append((s, s + r))
        s += r
    return out


def _block_mask(r_dims, strength, dtype):
    """M = 1 on same-task blocks, 1 - strength across tasks (0 = R_full, 1 = hard block)."""
    Kr = sum(r_dims)
    M = torch.full((Kr, Kr), 1.0 - float(strength), dtype=dtype)
    for s, e in _blocks(r_dims):
        M[s:e, s:e] = 1.0
    return M


def tcpr_router(G, Q, r_dims, mode="ssr_dir", tau=0.2, cross=1.0,
                block_strength=0.0):
    """Un-normalised router R (float64).  G, Q: count-normalised, (Kr, Kr)."""
    G = G.double()
    Q = Q.double()
    Kr = G.shape[0]
    K = len(r_dims)
    lam = float(tau) * float(torch.diagonal(G).mean())       # lam = tau * mean(diag G)
    G_reg = G + lam * torch.eye(Kr, dtype=G.dtype, device=G.device)
    alpha = None
    if mode == "acr":
        # adaptive common-residual: R = R_avg + alpha_l R_res, alpha_l from G/Q
        Gw = G / K
        if cross == "norm":
            # ||R_res|| / (||R_avg|| + ||R_res||), both share G_reg^-1
            ra = torch.linalg.solve(G_reg, Gw.t()).t().norm()
            rr = torch.linalg.solve(G_reg, (Q - Gw).t()).t().norm()
            alpha = float(rr / (ra + rr + 1e-12))
        elif cross == "trace":
            # clip((tr Q - tr G / K) / tr G, 0, 1)
            alpha = min(max(float((torch.trace(Q) - torch.trace(G) / K)
                                  / (torch.trace(G) + 1e-12)), 0.0), 1.0)
        else:
            alpha = float(cross)
        num = Gw + alpha * (Q - Gw)
    elif mode == "ssr_full" or mode == "ssr_dir":
        num = Q                                               # R = Q G_reg^-1
    else:
        Gw = G / K                                            # R = (G/K) G_reg^-1
        num = Gw + float(cross) * (Q - Gw)                    #   + cross (Q - G/K) G_reg^-1
    try:
        # G_reg symmetric: R = num G_reg^-1 = solve(G_reg, num^T)^T, as official
        R = torch.linalg.solve(G_reg, num.t()).t()
        ok = True
    except Exception:
        R = torch.eye(Kr, dtype=G.dtype, device=G.device)     # official fallback
        ok = False
    if block_strength > 0:
        R = R * _block_mask(r_dims, block_strength, R.dtype).to(R.device)
    return R, lam, ok, alpha


def regmean_router(tasks, r_dims, alpha=0.9):
    """Subspace RegMean: R = sum_k P_k Gt_k (sum_k Gt_k)^-1 (float64)."""
    blocks = _blocks(r_dims)
    if len(tasks) != len(r_dims):
        raise ValueError(f"regmean needs per-task stats for all {len(r_dims)} tasks, "
                         f"got {len(tasks)}")
    Gsum, num = None, None
    for local, k in enumerate(sorted(tasks)):
        Gk = tasks[k]["G"].double() / tasks[k]["n"]
        Gt = alpha * Gk + (1 - alpha) * torch.diag(torch.diagonal(Gk))
        if Gsum is None:
            Gsum, num = torch.zeros_like(Gt), torch.zeros_like(Gt)
        Gsum += Gt
        s, e = blocks[local]
        num[s:e] += Gt[s:e]                                   # P_k Gt_k
    try:
        R = torch.linalg.solve(Gsum, num.t()).t()
        ok = True
    except Exception:
        R = torch.eye(Gsum.shape[0], dtype=Gsum.dtype)
        ok = False
    return R, ok


def _data_gain(s, comb, name, R):
    """sqrt(merged output energy / sum_k own-delta energy) on calibration data."""
    lg = ssrlib._layer_gain(s, comb, name, R.float())
    if lg is None or lg[0] <= 0 or lg[1] <= 0:
        return None
    return math.sqrt(lg[0] / lg[1])


def tcpr_solve(stats, comb, mode="ssr_dir", tau=0.2, global_gain=0.8, cross=1.0,
               block_strength=0.0, norm="col", alpha=0.9):
    """{layer: (A_merged, B_merged)}, {layer: diag}.  See module docstring."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if mode == "ssr_dir":
        cross = 1.0
    merged, diag = {}, {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        if s is None or s["n"] == 0:
            merged[name] = (A_comb, B_comb)                   # passthrough, as ssr_solve
            diag[name] = {"fallback": True}
            continue
        n = s["n"]
        if mode == "regmean":
            R, ok = regmean_router(s.get("tasks") or {}, r_dims, alpha)
            lam, alpha_l = 0.0, None
        else:
            R, lam, ok, alpha_l = tcpr_router(s["G"].cpu() / n, s["Q"].cpu() / n, r_dims,
                                              mode, tau, cross, block_strength)
        d = {"lam": lam, "fallback": not ok, "n": n, "alpha": alpha_l,
             "colnorm_med": float(R.norm(dim=0).median())}
        if mode != "ssr_full":
            if norm in ("col", "coldata"):
                R = R / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
                if norm == "col":
                    R = global_gain * R
            if norm in ("data", "coldata"):
                g0 = _data_gain(s, comb, name, R)
                if g0 is None:
                    raise ValueError(f"{name}: norm='data' needs per-task stats "
                                     f"(stats file built without keep_tasks)")
                R = R * (global_gain / max(g0, 1e-12))
            elif global_gain != 1.0:
                R = R * global_gain
        d["data_gain"] = _data_gain(s, comb, name, R)
        merged[name] = (A_comb, B_comb.float() @ R.float())   # B_merged = B_comb R
        diag[name] = d
    return merged, diag


def diagnose_gq(stats, comb, layers=None, n_show=4):
    """Print G/Q health for a few layers plus an all-layer summary.

    Q is filled block-row-wise, so Q ~= G/K by design; ||Q - G/K|| is the
    cross-task signal that ssr_cross scales.
    """
    names = [n for n in comb if stats.get(n) is not None and stats[n]["n"] > 0]
    if layers is None:
        idx = sorted({round(i * (len(names) - 1) / max(n_show - 1, 1))
                      for i in range(n_show)})
        layers = [names[i] for i in idx]
    rel_dev = []
    for name in names:
        s = stats[name]
        r_dims = comb[name][2]
        K = len(r_dims)
        G = s["G"].cpu().double() / s["n"]
        Q = s["Q"].cpu().double() / s["n"]
        gF, qF, dF = float(G.norm()), float(Q.norm()), float((Q - G / K).norm())
        rel_dev.append(dF / max(gF / K, 1e-300))
        if name not in layers:
            continue
        print(f"[gq] {name}  K={K} n={s['n']}", flush=True)
        print(f"     ||G||_F {gF:.4e}  ||Q||_F {qF:.4e}  ||Q||/||G|| {qF / gF:.4f}"
              f" (1/K={1 / K:.4f})", flush=True)
        print(f"     ||Q-G/K||/||G|| {dF / gF:.4f}  ||Q-G/K||/||G/K|| {dF * K / gF:.4f}",
              flush=True)
        for k, (st, en) in enumerate(_blocks(r_dims)):
            eq = float(Q[st:en].norm() ** 2)
            eg = float(G[st:en].norm() ** 2)
            print(f"     task {k}: row energy Q {eq:.4e}  G {eg:.4e}  Q/G {eq / eg:.4f}",
                  flush=True)
    rel_dev.sort()
    if rel_dev:
        print(f"[gq] all {len(rel_dev)} layers: ||Q-G/K||/||G/K|| median "
              f"{rel_dev[len(rel_dev) // 2]:.3f} p90 {rel_dev[int(0.9 * len(rel_dev))]:.3f}",
              flush=True)


def arm_stem(mode, tau, g, cross, bs, norm, alpha=0.9):
    if mode == "regmean":
        bits = ["regmean", f"a{alpha:g}"]
    else:
        bits = ["tcpr", mode.split("_")[-1], f"tau{tau:g}"]
    if mode != "ssr_full" and not (norm == "none" and g == 1.0):
        bits.append(f"{norm}g{g:g}")
    if mode == "ssr_cross":
        bits.append(f"cr{cross:g}")
    if mode == "acr":
        bits.append(f"a{cross}")
    if bs > 0:
        bits.append(f"bs{bs:g}")
    return "_".join(bits)


def _med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else float("nan")


def _cross_arg(x):
    return x if x in ("norm", "trace") else float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", nargs="+", default=["ssr_dir"], choices=MODES)
    ap.add_argument("--tau", nargs="+", type=float, default=[0.2])
    ap.add_argument("--global_gain", nargs="+", type=float, default=[0.8])
    ap.add_argument("--cross", nargs="+", type=_cross_arg, default=[1.0],
                    help="ssr_cross: number; acr: 'norm', 'trace' or a number")
    ap.add_argument("--block_strength", nargs="+", type=float, default=[0.0])
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.9],
                    help="RegMean off-diagonal scale, used by regmean only")
    ap.add_argument("--norm", default="col", choices=NORMS)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--stats_root", required=True,
                    help="dir with stats_K{K}.pt (use data/seed, NOT data/processed)")
    ap.add_argument("--loras_root", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--diagnose", type=int, default=0,
                    help="print G/Q diagnostics for this many layers first")
    ap.add_argument("--diag_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = f"K{args.K}"
    stats = torch.load(os.path.join(args.stats_root, f"stats_{tag}.pt"),
                       map_location="cpu", weights_only=False)
    subs = args.subjects[:args.K] if args.subjects else common.SUBJECTS[:args.K]
    comb = ssrlib.concat_pairs(load_pairs(subs, args.loras_root))
    print(f"[{tag}] subjects {subs}; {len(stats)} stats layers, {len(comb)} LoRA layers",
          flush=True)
    if args.diagnose:
        diagnose_gq(stats, comb, n_show=args.diagnose)
    if args.diag_only:
        return

    # ssr_full ignores gain/cross/norm and ssr_dir ignores cross: dedupe by stem
    seen = {}
    for mode, tau, g, cr, bs, a in itertools.product(args.mode, args.tau, args.global_gain,
                                                     args.cross, args.block_strength,
                                                     args.alpha):
        if mode == "ssr_cross" and isinstance(cr, str):
            continue                                          # norm/trace are acr-only
        stem = args.prefix + arm_stem(mode, tau, g, cr, bs, args.norm, a)
        seen.setdefault(stem, (mode, tau, g, cr, bs, a))

    from safetensors.torch import save_file
    for stem, (mode, tau, g, cr, bs, a) in seen.items():
        out = os.path.join(common.MERGED, f"{stem}_{tag}.safetensors")
        if args.dry_run:
            print(f"  would write {out}", flush=True)
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"  {stem}: exists, skip (--overwrite to replace)", flush=True)
            continue
        t0 = time.time()
        merged, diag = tcpr_solve(stats, comb, mode, tau, g, cr, bs, args.norm, a)
        save_file(common.emit_peft_state(merged), out)
        cfg = dict(mode=mode, tau=tau, global_gain=g, cross=cr, block_strength=bs,
                   norm=args.norm, alpha=a, subjects=subs, stats_root=args.stats_root,
                   loras_root=args.loras_root)
        json.dump({"cfg": cfg, "layers": diag},
                  open(out.replace(".safetensors", "_stats.json"), "w"), indent=1)
        nfb = sum(bool(d.get("fallback")) for d in diag.values())
        print(f"  {stem:40s} {time.time() - t0:5.1f}s  "
              f"pre-norm colnorm med {_med(d.get('colnorm_med') for d in diag.values()):.3f}  "
              f"data gain med {_med(d.get('data_gain') for d in diag.values()):.3f}  "
              f"fallback {nfb}", flush=True)


if __name__ == "__main__":
    main()

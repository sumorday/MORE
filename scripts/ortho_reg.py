"""Cross-subject interference (crosstalk) penalties for the SSR router (ssrmc), the opposite
direction of the random-path GP (random_gp.py, a pull toward I/K).

ssrmc fits, per output block j (subject j's rows of R, feeding B_j), on every calibration latent z
of every subject k:  min sum_x || R_j z - [k = j] z_j ||^2  ->  R_j = Qb_j (G + ridge)^-1,
G = sum_k G_k (G_k: subject k's own Gram, stats["tasks"][k]["G"]), Qb_j = rows j of G_j.

  xlk<g>  leakage weighting: subject j's block output on the OTHER subjects' latents (crosstalk;
          the target there is 0) gets weight g, its own reconstruction weight 1:
            R_j = Qb_j (G_j + g (G - G_j) + ridge)^-1          (g = 1: ssrmc exactly)
  xoc<g>  the literal proposal: the off-diagonal (cross-subject) blocks of G scaled by g,
            R = Q (blockdiag(G) + g offdiag(G) + ridge)^-1
          for g > 1 the matrix need not be PSD (then the solve is a saddle point, not a
          minimum); min_eig reports it.
ridge = 1e-4 n absolute, as ssrlib.ssr_solve / run_spherical_proto.baseline_route.
"""
import torch

D = torch.float64


def blocks(s):
    return sorted((t["start"], t["end"], k) for k, t in s["tasks"].items())


def xtalk_solve(s, gamma, mode, ridge=1e-4):
    """-> (R, info). s: one layer of stats_K<K>.pt (G, Q, n, tasks)."""
    G = s["G"].to(D)
    Kr = G.shape[0]
    I = torch.eye(Kr, dtype=D)
    lam = ridge * s["n"]
    if mode == "oc":
        M = G.clone()
        for a0, a1, _ in blocks(s):
            for b0, b1, _ in blocks(s):
                if a0 != b0:
                    M[a0:a1, b0:b1] *= gamma
        M = M + lam * I
        R = torch.linalg.solve(M, s["Q"].to(D).t()).t()     # M symmetric
        return R, dict(min_eig=float(torch.linalg.eigvalsh(M)[0] / s["n"]))
    assert mode == "lk", mode
    R = torch.zeros(Kr, Kr, dtype=D)
    for a0, a1, k in blocks(s):
        Gk = s["tasks"][k]["G"].to(D)
        M = Gk + gamma * (G - Gk) + lam * I
        L = torch.linalg.cholesky(M)
        R[a0:a1] = torch.cholesky_solve(Gk[a0:a1].t().contiguous(), L).t()
    return R, {}


def diagnose(s, R):
    """Per-layer energies relative to the target energy yy: own (sum_j own-data residual of
    block j) and leak (sum_j block-j output energy on the other subjects' latents)."""
    G = s["G"].to(D)
    own = leak = 0.0
    for a0, a1, k in blocks(s):
        Gk = s["tasks"][k]["G"].to(D)
        Rj = R[a0:a1]
        own += float((Rj @ Gk * Rj).sum() - 2 * (Rj * Gk[a0:a1]).sum() + torch.trace(Gk[a0:a1, a0:a1]))
        leak += float((Rj @ (G - Gk) * Rj).sum())
    return own / s["yy"], leak / s["yy"]


if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import common
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.5, 2, 4, 10])
    ap.add_argument("--n_layers", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    stats = torch.load(os.path.join(common.ROOT, "data", "seed2_calNmT3", f"stats_K{args.K}.pt"),
                       map_location="cpu", weights_only=False)
    names = [n for n in stats if stats[n]["n"] > 0]
    names = names[::max(1, len(names) // args.n_layers)][:args.n_layers]
    med = lambda v: sorted(v)[len(v) // 2]
    rows = {}
    for n in names:
        s = stats[n]
        R0, _ = xtalk_solve(s, 1.0, "lk")
        R1 = torch.linalg.solve(s["G"].to(D) + 1e-4 * s["n"] * torch.eye(s["G"].shape[0], dtype=D),
                                s["Q"].to(D).t()).t()
        assert float((R0 - R1).norm() / R1.norm()) < 1e-4, n   # lk1 == ssrmc (G stored fp32)
        o0, l0 = diagnose(s, R0)
        for mode in ("lk", "oc"):
            for g in args.gammas:
                R, info = xtalk_solve(s, g, mode)
                o, l = diagnose(s, R)
                rows.setdefault((mode, g), []).append(
                    (float((R - R0).norm() / R0.norm()), o, l, o0, l0, info.get("min_eig", 0.0)))
    print(f"K{args.K}, {len(names)} layers, medians; xlk1 == ssrmc checked")
    print("  mode gamma  |R-R_ssrmc|/|R_ssrmc|  own(R)  leak(R)  own(ssrmc) leak(ssrmc)  min_eig(oc)")
    for (mode, g), v in rows.items():
        c = list(zip(*v))
        print(f"  {mode}   {g:<6g} {med(c[0]):10.3f}            {med(c[1]):.4f}  {med(c[2]):.4f}"
              f"   {med(c[3]):.4f}    {med(c[4]):.4f}     {min(c[5]):+.2e}")

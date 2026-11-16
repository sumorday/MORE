"""Common-mode / differential-mode split of a routed merge on the unit sphere, and max-margin
de-correlation of the differential parts.

Per layer the full-rank merge is dW = M A_comb, M = g B R (B = B_comb, possibly de-crowded;
R the SSR router; g the energy gain), exactly as ssr_rc.rc_solve(mode="ssr_full").  It splits
over subjects and over output directions without approximation:
  M_k  = g B_k R_k                  (B_k: columns of subject k, R_k: its rows of R), sum_k M_k = M
The rank-r merge (rc_solve(mode="ssr_rc"), svd "fro") is dW_c = g M_c A_comb, M_c = B_out T, which
mixes the subjects' B columns; it is split over the INPUT blocks instead (route_layer(rank=r)
returns B = M_c and R = I, so every formula below holds with B_k = column block k of M_c, R = I):
  M_k  = g M_c[:, block k],  part k = g M_c[:, block k] A_k   (the merge's response to subject
                                                               k's LoRA-A features)
  P    = U_s U_s^T                  (shared output directions, ssr_rc.shared_dirs on the
                                     ORIGINAL B_comb; empty -> P = 0)
  common mode    W_c   = P M A_comb
  differential   W_k^p = (I - P) M_k A_comb,       W_c + sum_k W_k^p = dW.

Inner product of two deltas X A_comb, Y A_comb (metric W on the Kr-dim latent z = A_comb x):
  data: W = G / n  -> <X, Y> = mean over calibration inputs of <X z, Y z> (output space)
  fro : W = A_comb A_comb^T -> Frobenius inner product of the out x in matrices
For the differential parts the K x K Gram is a block sum of two Kr x Kr matrices,
  H_ij = g^2 sum_{block (i, j)} (Bp^T Bp) o (R W R^T),   Bp = (I - P) B,
so nothing out x in (or K copies of out x Kr) is formed.

Max-margin.  u_k = W_k^p / ||W_k^p|| lies on the unit sphere of the metric; S = cosine matrix.
  min_{u~}  sum_k ||u~_k - u_k||^2 + lam sum_{i != j} max(0, <u~_i, u~_j> - m0)^2,  ||u~_k|| = 1
is solved restricted to span{u_j} (the merge introduces no new directions):
u~_k = v_k / ||v_k||, v_k = sum_j C_kj u_j, so every quantity is a function of C (K x K) and S:
  <u~_i, u~_j> = (C S C^T)_ij / (n_i n_j),  n_k^2 = (C S C^T)_kk,  ||u~_k - u_k||^2 = 2 - 2 (C S)_kk / n_k.
L-BFGS (<= `iters` iterations per penalty level, float64).  hard=True raises lam by `grow` until
every off-diagonal cosine is below m0 (the hinge acts at m0 - eps, so the result is strictly
below m0), or lam_max is hit (reported infeasible).  Each subject keeps its own norm:
  W~_k = ||W_k^p|| u~_k = sum_j Cw_kj W_j^p,  Cw = diag(||W^p||) diag(1/n) C diag(1/||W^p||).

Runtime factors of one layer (LayerFactors): with w = A_r x, A_r = R A_comb,
  shared   U_s (C_s w),            C_s = U_s^T g B          (= P M A_comb x)
  subject  B_sp (w o L[:, k]),     B_sp = g (I - P) B,  L[block j, k] = Cw_kj   (= W~_k x)
Cw = I gives W~_k = W_k^p, and sum over all k of the subject terms plus the shared term is the
static merge.  The raw own-LoRA reference is the same form with A_r = A_comb, B_sp = B_comb,
no shared part and L = block indicators (subject k's delta B_k A_k x).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import _SCRIPTS  # noqa: F401  (puts scripts/ on sys.path)
import tcpr
from ssr_rc import _energy, _own_energy, compress, pico_calibrate_B, shared_dirs

D = torch.float64


def blocks(r_dims):
    return tcpr._blocks(r_dims)


def sphere_normalize(X, dim=-1, eps=1e-30):
    """Rows (dim=-1) / columns projected to the unit sphere, in float64; zero vectors stay 0."""
    X = X.to(D)
    n = X.norm(dim=dim, keepdim=True)
    return torch.where(n > eps, X / n.clamp_min(eps), torch.zeros_like(X))


def block_sum(X, r_dims):
    """(Kr, Kr) -> (K, K): sum of each (i, j) block."""
    b = blocks(r_dims)
    out = torch.zeros(len(b), len(b), dtype=X.dtype)
    for i, (a0, a1) in enumerate(b):
        for j, (c0, c1) in enumerate(b):
            out[i, j] = X[a0:a1, c0:c1].sum()
    return out


def route_layer(s, A_comb, B_comb, r_dims, tau, g_star=1.0, pico=None, cross=1.0,
                shared=(0.5, 8), rank=None):
    """SSR merge of one layer in factored per-subject form, dW = g B R A_comb.

    Same router, de-crowding and energy gain as ssr_rc.rc_solve(norm="energy"):
      rank None: mode "ssr_full", B = de-crowded B_comb, g * B @ R is that arm's B_out;
      rank r:    mode "ssr_rc" (svd "fro"), B = M_c = B_out T (out, Kr), R = I, so
                 g * B @ A_comb = that arm's B_out @ A_out.
    pico: None or pico_calibrate_B kwargs.  shared: (thresh, m) of shared_dirs on the original
    B_comb (the common-mode directions).
    Returns dict(B, B_orig, R, g, U_s, shares), float64."""
    B_orig = B_comb.to(D)
    B = B_orig
    if pico is not None:
        B, _, _ = pico_calibrate_B(B_orig, r_dims, **pico)
    R, _, ok, _ = tcpr.tcpr_router(s["G"].cpu() / s["n"], s["Q"].cpu() / s["n"], r_dims,
                                   "ssr_full" if cross == 1.0 else "ssr_cross", tau, cross)
    if not ok:
        raise RuntimeError("router solve failed")
    if rank is not None:
        B_out, _, T, _, _ = compress(B @ R, A_comb.to(D), rank)
        B, R = B_out @ T, torch.eye(R.shape[0], dtype=D)
    tasks = s["tasks"]
    g = g_star * math.sqrt(_own_energy(tasks, B_orig) / max(_energy(tasks, B @ R), 1e-300))
    U_s, shares = shared_dirs(B_orig, r_dims, *shared)
    return dict(B=B, B_orig=B_orig, R=R, g=g, U_s=U_s, shares=shares)


def latent_metric(s, A_comb, metric):
    if metric == "data":
        return s["G"].cpu().to(D) / s["n"]
    if metric == "fro":
        A = A_comb.to(D)
        return A @ A.t()
    raise ValueError(f"metric {metric!r}")


def part_gram(B, R, g, U_s, W, r_dims):
    """H (K, K) of the parts g (I - P) B_k R_k A_comb under latent metric W (module docstring).
    U_s with 0 columns -> no common-mode removal; R = I, g = 1 -> raw B_k A_k."""
    Bp = B - U_s @ (U_s.t() @ B) if U_s.shape[1] else B
    return (g * g) * block_sum((Bp.t() @ Bp) * (R @ W @ R.t()), r_dims)


def cosines(H):
    """Cosine matrix and norms of a Gram; zero-norm parts get cos 0 (diag 1 kept)."""
    n = H.diagonal().clamp_min(0).sqrt()
    ok = n > 1e-150 * max(float(n.max()), 1e-300)
    inv = torch.where(ok, 1.0 / n.clamp_min(1e-300), torch.zeros_like(n))
    S = H * inv[:, None] * inv[None, :]
    S.fill_diagonal_(1.0)
    return S, n, ok


def resolve_m0(S, m0):
    """m0: float, or "q<p>" = the p-quantile of this layer's off-diagonal cosines (pairs above it
    are pushed down to it), clipped at 0 so it never asks for negative correlation."""
    if isinstance(m0, str):
        p = float(m0[1:])
        iu = torch.triu_indices(S.shape[0], S.shape[0], 1)
        return max(float(torch.quantile(S[iu[0], iu[1]], p)), 0.0)
    return float(m0)


def max_margin(S, m0, lam=10.0, iters=20, hard=True, eps=1e-3, grow=10.0, lam_max=1e8):
    """K x K max-margin solve on the cosine matrix S (module docstring).  Returns (C, info);
    C = I when no pair is above m0 - eps (no-op)."""
    S = S.to(D)
    K = S.shape[0]
    eye = torch.eye(K, dtype=D)
    off = 1.0 - eye
    target = m0 - eps
    diag = eye.bool()
    offmax = lambda X: float(X.masked_fill(diag, -math.inf).max())   # not (X * off): cos may be < 0
    cos0 = offmax(S) if K > 1 else -1.0
    info = dict(m0=m0, max_cos_before=cos0,
                active_pairs=int((S.masked_fill(diag, -math.inf) > target).sum()) // 2)
    if K < 2 or cos0 <= target:
        info.update(max_cos_after=cos0, lam=0.0, n_iter=0, feasible=True, fidelity=1.0)
        return eye.clone(), info

    def parts(C):
        V = C @ S @ C.t()
        n = V.diagonal().clamp_min(1e-30).sqrt()
        cos = V / (n[:, None] * n[None, :])
        fid = (C @ S).diagonal() / n                      # <u~_k, u_k>
        return cos, fid

    C = eye.clone().requires_grad_(True)
    n_iter = 0
    while True:
        opt = torch.optim.LBFGS([C], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15, history_size=20)

        def closure():
            opt.zero_grad()
            cos, fid = parts(C)
            h = torch.relu(cos - target) * off
            loss = (2.0 - 2.0 * fid).sum() + lam * (h ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        n_iter += opt.state[opt._params[0]]["n_iter"]
        with torch.no_grad():
            cos, fid = parts(C)
            mx = offmax(cos)
        if not hard or mx < m0 - 0.5 * eps or lam >= lam_max:
            break
        lam *= grow
    C = C.detach()
    info.update(max_cos_after=mx, lam=lam, n_iter=n_iter, feasible=mx < m0,
                fidelity=float(fid.mean()), fidelity_min=float(fid.min()))
    return C, info


def weight_coefs(C, S, norms):
    """Cw with W~_k = sum_j Cw_kj W_j^p (norm of W~_k = norms_k)."""
    V = C @ S @ C.t()
    nv = V.diagonal().clamp_min(1e-30).sqrt()
    inv = torch.where(norms > 0, 1.0 / norms.clamp_min(1e-300), torch.zeros_like(norms))
    return (norms / nv)[:, None] * C * inv[None, :]


@dataclass
class LayerFactors:
    """Delta of one layer: shared U_s (C_s w) + subject k: B_sp (w o L[:, k]), w = A_r x."""
    A_r: torch.Tensor                 # (Kr, in)
    B_sp: torch.Tensor                # (out, Kr)
    L: torch.Tensor                   # (Kr, K)
    U_s: torch.Tensor | None = None   # (out, m)
    C_s: torch.Tensor | None = None   # (m, Kr)

    def to(self, device, dtype):
        f = lambda t: None if t is None else t.to(device=device, dtype=dtype).contiguous()
        return LayerFactors(f(self.A_r), f(self.B_sp), f(self.L), f(self.U_s), f(self.C_s))

    def delta(self, x, mask, gamma_sh=1.0, gamma_sp=1.0):
        """Reference (unhooked) delta: x (..., in), mask (..., K) per-token subject weights."""
        w = x @ self.A_r.t()
        out = torch.zeros(*x.shape[:-1], self.B_sp.shape[0], dtype=x.dtype, device=x.device)
        if self.U_s is not None and gamma_sh:
            out = out + gamma_sh * (w @ self.C_s.t()) @ self.U_s.t()
        if gamma_sp:
            out = out + gamma_sp * (w * (mask @ self.L.t())) @ self.B_sp.t()
        return out


def block_indicator(r_dims, Cw=None):
    """L (Kr, K): L[block j, k] = Cw[k, j] (identity when Cw is None)."""
    K = len(r_dims)
    Cw = torch.eye(K, dtype=D) if Cw is None else Cw
    L = torch.zeros(sum(r_dims), K, dtype=D)
    for j, (a0, a1) in enumerate(blocks(r_dims)):
        L[a0:a1, :] = Cw[:, j][None, :]
    return L


def merge_factors(rt, A_comb, r_dims, Cw=None):
    """LayerFactors of the routed merge rt (route_layer) with subject coefficients Cw."""
    B, R, g, U = rt["B"], rt["R"], rt["g"], rt["U_s"]
    A_r = R @ A_comb.to(D)
    if U.shape[1]:
        return LayerFactors(A_r, g * (B - U @ (U.t() @ B)), block_indicator(r_dims, Cw),
                            U.clone(), g * (U.t() @ B))
    return LayerFactors(A_r, g * B, block_indicator(r_dims, Cw))


def raw_factors(A_comb, B_comb, r_dims):
    """Own LoRAs: subject k's delta is B_k A_k x, no shared part."""
    return LayerFactors(A_comb.to(D), B_comb.to(D), block_indicator(r_dims))


def static_B(f):
    """B with B @ A_r == the static merge (all subjects, ungated): U_s C_s + B_sp diag(L 1)."""
    out = f.B_sp * f.L.sum(1)[None, :]
    if f.U_s is not None:
        out = out + f.U_s @ f.C_s
    return out

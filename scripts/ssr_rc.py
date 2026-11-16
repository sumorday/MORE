"""SSR-RC (Route-then-Compress).

  R = Q (G + lam I)^{-1}    # full subspace signal routing (cross=1), lam = tau mean(diag G)
  dW = B_comb R A_comb
  (B_out, A_out) = truncated SVD of dW to rank r (factored, never forms out x in)
  B_out *= g                # data-space energy match to g_star * task energy

Differs from Compress-then-Merge (CtM): we route with the activation Gram/Q
first, then compress; CtM compresses weight subspaces before merging without
this OLS router.  Differs from plain SSR: deployable fixed rank r instead of
K*r.  Solve stage only; G/Q are the official accumulation (ssrlib.accumulate),
read from stats_K{K}.pt.

Modes
  ssr_full        B = B_comb R, A = A_comb, then energy gain (no rank cut;
                  == tcpr --mode ssr_dir --norm data == *_ssr_rel_tau*_guni*)
  ssr_rc          route, compress to rank r, energy gain
  ssr_rc_noscale  route, compress, no gain (ablation)
  mtc_svd         merge-then-compress: dW = sum_k B_k A_k (R = I), SVD_r,
                  energy gain.  sum vs mean differ by a scalar, which the
                  energy gain removes.
  ctm             Compress-then-Merge baseline (data-free fixed-rank: Tucker-2
                  subspace of the norm-balanced task deltas, TA inside it;
                  see compress_ctm), energy gain.  --ctm_beta
  --cross 0       ablation router R = (G/K) G_reg^-1 (cr0) instead of Q G_reg^-1

Factored SVD (Option F).  A_comb = U_a S_a V_a^T, M = B_comb R = U_m S_m V_m^T,
  C = S_m V_m^T U_a S_a  (R_dim x R_dim) = U_c S_c V_c^T,
  dW = (U_m U_c) S_c (V_a V_c)^T,  balanced split s = sqrt(S_c[:r]):
  B_out = U_m U_c[:, :r] s,  A_out = s V_c[:, :r]^T V_a^T.
  A_out = T A_comb with T = s V_c[:, :r]^T S_a^-1 U_a^T, so the compressed
  delta stays in A_comb's row space and its energy on calibration data is
  read off the per-task Grams, as for full SSR.

Data-metric SVD (--svd data).  Both truncations are rank-r M_c in a latent
  metric W, dW_c = M_c A_comb, minimising ||(M - M_c) W^{1/2}||_F:
    fro : W = A_comb A_comb^T  -> ||dW - dW_c||_F        (Option F above)
    data: W = G + rho mean(diag G) I, G = sum_k G_k
                               -> output error on the calibration data
  Optimum M_c = U_r U_r^T M, U_r the top-r left singular vectors of
  M W^{1/2} (no W^-1 needed).  B_out = U_r c, A_out = c^-1 U_r^T M A_comb,
  c_i = sqrt(||row i of U_r^T M A_comb||) to balance the two factors.
  With W = G + lam I (--svd ridge, rho = tau) this is the exact optimum of the
  rank-r constrained ridge RegMean objective in the subspace:
    sum_k ||(M - B_comb P_k) Z_k||^2 + lam ||M||^2
      = ||(M - M_lam) (G + lam I)^{1/2}||_F^2 + const,  M_lam = B_comb Q (G + lam I)^-1
  so route-then-compress is one closed-form reduced-rank regression.

Global rank allocation (--rank_alloc global).  Same total budget as uniform
  rank r (sum_l r_l = L r).  Per layer, the share of metric energy lost at
  rank k is e_l(k) = sum_{j>=k} S_lj^2 / sum_j S_lj^2; greedy water-filling on
  the marginal gains S_lj^2 / sum_j S_lj^2 (descending per layer, so greedy is
  optimal) minimises sum_l e_l(r_l) subject to r_min <= r_l <= R_dim.

Task-balanced metric (--task_balance N, with --svd data).  W = sum_k w_k G_k
  with one weight per task shared by all layers.  e_k = mean over layers of
  task k's relative output error ||(M - M_c) G_k^{1/2}|| / ||M G_k^{1/2}||;
  N rounds of w_k <- w_k (e_k / mean e)^tb_power (mean w = 1), each one closed-form
  truncation per layer.  Equalises per-task error instead of the summed error.

Energy (--norm)
  energy  g = g_star sqrt(E_own / E_c),  E_c = sum_k tr(M_c G_k M_c^T) with
          dW_c = M_c A_comb, E_own = sum_k tr(B_k G_k[kk] B_k^T).  Same rule as
          tcpr --norm data / guni (ssrlib._layer_gain), in float64.
  post_svd_only  alias of energy: g is one scalar, so it commutes with the SVD
  fro     weight-space variant: E_c = ||B_out A_out||_F^2, E_own = sum_k ||B_k A_k||_F^2
  none    no gain

Pico-style B calibration (--pico, data-free, before B_comb R).  Top-m left
  singular vectors u_j of B_comb = [B_1 | ... | B_K];
  share_j = mean_k ||u_j^T B_k|| / ||B_k||_F;  for share_j > thresh,
  B_k <- B_k - strength share_j u_j u_j^T B_k (u_j orthonormal, so the order
  does not matter); --pico_rescale restores each ||B_k||_F.  A_k, G, Q and the
  energy reference (original B_k) are unchanged.
  --pico_ctrl random (tag suffix "rnd", control): the same layers and the same per-direction
  damping amounts, but each damped u_j is replaced by a random unit vector of span(B_comb)
  (orthonormalised; drawn from the rc_solve seed), i.e. B_k <- (I - V diag(c) V^T) B_k.
  --pico_ctrl random_norm (tag "rndn", size-matched control): the same layers and the same
  perturbation norm ||B_comb - B_new||_F, spread over a random subspace of span(B_comb)
  orthogonal to the shared directions ("random" damps one random direction per shared one,
  which carries ~1/rank of the energy and so is a much smaller perturbation).
Pico as published (--pico --pico_mode paper, arm tag picoP; Tang & Yang, arXiv 2604.16826,
  "Crowded in B-Space"): B_comb = U diag(sig) V^T (thin), score s_j = sig_j^2 / sum sig^2,
  keep factor a_j = 1 / (1 + (K - 1) s_j) in [1/K, 1], B_k <- S B_k with
  S = I + U diag(a - 1) U^T (no threshold / strength knobs).  The paper's single post-merge
  rescale is subsumed by the energy gain, whose reference E_own uses the original B_k.

Split energy gain (--split_gain, tag sg; --sg_thresh, --sg_m).  Shared output directions
  U_s = pico's criterion on the ORIGINAL B_comb (no damping).  The energy gain is applied
  separately in span(U_s) and its complement, each restored to the subjects' own output energy
  there (split_gain); the shared gain is capped at the specific one.  Replaces the layer-wide
  gain; layers without a shared direction get the plain gain.

Spectral calibration (--svc_alpha a, tag svc<a>; SVC, arXiv 2602.05536, output side), after
  compression and before the energy gain: dW_c = u diag(sig) v^T (factored),
  s_ri = <u_r^T dW_c, u_r^T B_i A_i> / ||u_r^T B_i A_i||^2 with the raw (un-damped) B_i,
  sig_r <- sig_r K / sum_i max(a, s_ri).  Directions carried at full strength by every task
  keep gamma = 1; a direction carried by one task only gets K / (1 + (K-1) a); the energy
  gain then removes the common scale, so SVC re-weights directions, not the layer norm.
  --svc_count cov (tag svcc<a>): s_ri = clip(<u_r^T dW_c, u_r^T B_i A_i> / sig_r^2, 0, 1), the
  share of the merged component task i accounts for.  Same values in the two limiting cases
  above, but a task that only leaks a small component into direction r gets s_ri <= its
  relative size instead of the published ratio's ~1 / leak.

  python scripts/ssr_rc.py --mode ssr_rc --rank 32 64 96 --tau 0.2 --g_star 0.8 \\
      --K 3 --stats_root data/seed --loras_root loras_seed --prefix seed_
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
import tcpr
from solve_from_stats import load_pairs

MODES = ("ssr_full", "ssr_rc", "ssr_rc_noscale", "mtc_svd", "ctm")
NORMS = ("energy", "post_svd_only", "fro", "none")


def _thin_svd(X):
    """X = U diag(S) Vh in float64; on failure, eig of the smaller Gram."""
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return U, S, Vh, 0
    except Exception:
        pass
    m, n = X.shape
    if m <= n:
        w, U = torch.linalg.eigh(X @ X.t())
        w, U = w.flip(0), U.flip(1)
        S = w.clamp_min(0).sqrt()
        keep = S > S.max() * 1e-12
        Vh = (U.t() @ X) / torch.where(keep, S, torch.ones_like(S))[:, None]
        Vh[~keep] = 0
    else:
        w, V = torch.linalg.eigh(X.t() @ X)
        w, V = w.flip(0), V.flip(1)
        S = w.clamp_min(0).sqrt()
        keep = S > S.max() * 1e-12
        U = (X @ V) / torch.where(keep, S, torch.ones_like(S))[None, :]
        U[:, ~keep] = 0
        Vh = V.t()
    return U, S, Vh, 1


def compress(M, A, r):
    """Rank-r factors of M A without forming it (Option F).

    Returns B_out (out, k), A_out (k, in), T (k, R_dim) with A_out = T A,
    the full spectrum S_c of M A, and the number of SVD fallbacks.
    """
    Ua, Sa, Vah, f1 = _thin_svd(A)                        # A = U_a S_a V_a^T
    Um, Sm, Vmh, f2 = _thin_svd(M)                        # M = U_m S_m V_m^T
    C = (Sm[:, None] * Vmh) @ (Ua * Sa[None, :])          # C = S_m V_m^T U_a S_a
    Uc, Sc, Vch, f3 = _thin_svd(C)
    k = min(int(r), Sc.numel())
    s = Sc[:k].clamp_min(0).sqrt()
    B_out = (Um @ Uc[:, :k]) * s[None, :]                 # U_m U_c[:, :r] s
    A_out = s[:, None] * (Vch[:k] @ Vah)                  # s V_c[:, :r]^T V_a^T
    Sa_inv = torch.where(Sa > Sa.max() * 1e-12, 1.0 / Sa.clamp_min(1e-300),
                         torch.zeros_like(Sa))
    T = s[:, None] * ((Vch[:k] * Sa_inv[None, :]) @ Ua.t())
    return B_out, A_out, T, Sc, f1 + f2 + f3


def _sqrt_psd(W):
    w, V = torch.linalg.eigh(W)
    return (V * w.clamp_min(0).sqrt()[None, :]) @ V.t()


def compress_metric(M, A, W, r):
    """Rank-r factors of M A minimising ||(M - M_c) W^{1/2}||_F (see docstring).

    Same return layout as compress(); S is the spectrum of M W^{1/2}.
    """
    U, S, _, f = _thin_svd(M @ _sqrt_psd(W))
    k = min(int(r), S.numel())
    Ur = U[:, :k]
    P = Ur.t() @ M                                        # M_c = U_r P
    H = P @ A                                             # (k, in)
    c = H.norm(dim=1).clamp_min(1e-30).sqrt()
    return Ur * c[None, :], H / c[:, None], P / c[:, None], S, f


def _top_left(X, r):
    U, _, _, _ = _thin_svd(X)
    return U[:, :min(int(r), U.shape[1])]


def compress_ctm(B, A, r_dims, r, beta=0.5, iters=10):
    """Compress-then-Merge baseline (data-free; Task Arithmetic in the shared subspace).

    Core space (Panariello et al. 2025): B_cat = U_B S_B V_B^T, A_cat = U_A S_A V_A^T,
    M_t = U_B^T B_t A_t V_A.  Surrogates X_t = lam_t M_t / ||M_t||,
    lam_t = beta ||M_t|| + (1 - beta) mean_t ||M_t||.  Tucker-2 (r, r, T) by
    HOSVD + HOOI gives U, V (r columns); O = sum_t U^T M_t V (TA; the scale is
    removed by the energy gain), dW = U_B U O V^T V_A^T.  Same return layout as
    compress(): A_out = T A with T = s V_o^T V^T S_A^-1 U_A^T.
    """
    Ub, _, _, f1 = _thin_svd(B)
    Ua, Sa, Vah, f2 = _thin_svd(A)
    Ms = [(Ub.t() @ B[:, s:e]) @ (A[s:e] @ Vah.t()) for s, e in tcpr._blocks(r_dims)]
    nrm = [float(m.norm()) for m in Ms]
    avg = sum(nrm) / len(nrm)
    Xs = [(beta * n + (1 - beta) * avg) / max(n, 1e-300) * m for n, m in zip(nrm, Ms)]
    U = _top_left(torch.cat(Xs, 1), r)                            # HOSVD
    V = _top_left(torch.cat([x.t() for x in Xs], 1), r)
    for _ in range(iters):                                        # HOOI
        U = _top_left(torch.cat([x @ V for x in Xs], 1), r)
        V = _top_left(torch.cat([x.t() @ U for x in Xs], 1), r)
    O = sum(U.t() @ m @ V for m in Ms)
    Uo, So, Voh, f3 = _thin_svd(O)
    s = So.clamp_min(0).sqrt()
    B_out = (Ub @ (U @ Uo)) * s[None, :]
    W = s[:, None] * (Voh @ V.t())                                # (k, R_dim) in V_A coords
    A_out = W @ Vah
    Sa_inv = torch.where(Sa > Sa.max() * 1e-12, 1.0 / Sa.clamp_min(1e-300),
                         torch.zeros_like(Sa))
    T = (W * Sa_inv[None, :]) @ Ua.t()
    return B_out, A_out, T, So, f1 + f2 + f3


def _energy(tasks, M):
    """sum_k tr(M G_k M^T): energy of dW = M A_comb on the calibration data."""
    MM = M.t() @ M
    return sum(float((MM * t["G"].double()).sum()) for t in tasks.values())


def _own_energy(tasks, B_comb):
    """sum_k tr(B_k G_k[kk] B_k^T): each task's own delta on its own data."""
    e = 0.0
    for t in tasks.values():
        s, en = t["start"], t["end"]
        Bk = B_comb[:, s:en]
        e += float((t["G"].double()[s:en, s:en] * (Bk.t() @ Bk)).sum())
    return e


def _fro2(B, A):
    """||B A||_F^2 = sum((B^T B) * (A A^T))."""
    return float(((B.t() @ B) * (A @ A.t())).sum())


def pico_calibrate_B(B_comb, r_dims, strength=0.5, thresh=0.5, m=8, rescale=False,
                     ctrl="share", gen=None):
    """Damp output directions shared by many tasks.  See module docstring.
    ctrl="random": same damping amounts on random orthonormal directions of span(B_comb)."""
    U, _, _, _ = _thin_svd(B_comb)
    Um = U[:, :min(int(m), U.shape[1])]
    P = Um.t() @ B_comb                                   # u_j^T B_comb
    shares = torch.zeros(Um.shape[1], dtype=B_comb.dtype)
    for s, e in tcpr._blocks(r_dims):
        shares += P[:, s:e].norm(dim=1) / B_comb[:, s:e].norm().clamp_min(1e-12)
    shares /= len(r_dims)                                 # share_j = mean_k c_k
    damp = torch.where(shares > thresh, strength * shares, torch.zeros_like(shares))
    if ctrl == "random":
        sel = damp > 0
        nd = int(sel.sum())
        if nd:
            Z = torch.randn(U.shape[1], nd, generator=gen, dtype=torch.float64).to(U.dtype)
            V, _ = torch.linalg.qr(U @ Z)                 # orthonormal, in span(B_comb)
            Um, damp = V, damp[sel]
            P = Um.t() @ B_comb
        else:
            Um, damp, P = Um[:, :0], damp[:0], P[:0]
    elif ctrl == "random_norm":
        # size-matched control: a perturbation ||B_comb - B_new||_F equal to the share
        # damping's, in a random subspace V of span(B_comb) orthogonal to the shared
        # directions, damped by one factor c = ||dB_share|| / ||V^T B||.  A random unit vector
        # carries ~1/rank of the energy, so V has n directions, doubled until c <= 0.5 (the
        # share damping is 0.5 s_j <= 0.5) or V is the whole complement (then c may be larger).
        sel = damp > 0
        if int(sel.sum()):
            dn = float((Um @ (damp[:, None] * P)).norm())
            Us = Um[:, sel]
            n = 8 * int(sel.sum())
            while True:
                n = min(n, U.shape[1] - Us.shape[1])
                Z = torch.randn(U.shape[1], n, generator=gen, dtype=torch.float64).to(U.dtype)
                W = U @ Z
                V, _ = torch.linalg.qr(W - Us @ (Us.t() @ W))
                PV = V.t() @ B_comb
                c = dn / max(float(PV.norm()), 1e-300)
                if c <= 0.5 or n >= U.shape[1] - Us.shape[1]:
                    break
                n *= 2
            Um, P = V, PV
            damp = torch.full((V.shape[1],), min(c, 1.0), dtype=B_comb.dtype)
        else:
            Um, damp, P = Um[:, :0], damp[:0], P[:0]
    elif ctrl != "share":
        raise ValueError(f"pico ctrl {ctrl!r}")
    B_new = B_comb - Um @ (damp[:, None] * P)             # B_k - (1-scale) u u^T B_k
    if rescale:
        for s, e in tcpr._blocks(r_dims):
            B_new[:, s:e] *= B_comb[:, s:e].norm() / B_new[:, s:e].norm().clamp_min(1e-12)
    return B_new, shares, int((damp > 0).sum())


def pico_paper_B(B_comb, r_dims):
    """Published Pico: S B_comb, S = I + U diag(a - 1) U^T, a_j = 1 / (1 + (K-1) s_j),
    s_j = sig_j^2 / sum sig^2 over the thin SVD of B_comb.  Returns (S B_comb, s, a)."""
    U, S, _, _ = _thin_svd(B_comb)
    s = S.pow(2) / S.pow(2).sum().clamp_min(1e-300)
    a = 1.0 / (1.0 + (len(r_dims) - 1) * s)
    return B_comb + U @ ((a - 1.0)[:, None] * (U.t() @ B_comb)), s, a


def shared_dirs(B_comb, r_dims, thresh=0.5, m=8):
    """Orthonormal U_s (d_out, m_s): top-m left singular vectors u_j of B_comb with
    share_j = mean_k ||u_j^T B_k|| / ||B_k||_F > thresh (pico_calibrate_B's criterion)."""
    U, _, _, _ = _thin_svd(B_comb)
    Um = U[:, :min(int(m), U.shape[1])]
    P = Um.t() @ B_comb
    shares = torch.zeros(Um.shape[1], dtype=B_comb.dtype)
    for s, e in tcpr._blocks(r_dims):
        shares += P[:, s:e].norm(dim=1) / B_comb[:, s:e].norm().clamp_min(1e-12)
    shares /= len(r_dims)
    return Um[:, shares > thresh], shares


def split_gain(B_out, T, U_s, tasks, B_orig, g_star=1.0):
    """Energy gain applied separately in span(U_s) (shared output directions) and in its
    orthogonal complement: each part of dW_c = B_out T A_comb is restored to the subjects'
    own output energy in that part,
      g_s = g* sqrt(E_own(P) / E_c(P)),  g_p = g* sqrt(E_own(I-P) / E_c(I-P)),  P = U_s U_s^T,
      B_out <- g_p B_out + (g_s - g_p) P B_out,   g_s capped at g_p.
    A coherent shared component survives the merge at full scale while subject-specific
    components are diluted, so one layer-wide gain hands the restored energy to the shared
    direction; per-part gains restore each part to its own level.  The cap keeps the shared
    part from being amplified above the specific one when the router has removed it.
    Returns (B_new, g_s, g_p); U_s empty -> the plain energy gain (g_s = g_p = g)."""
    M_c = B_out @ T
    if U_s.shape[1] == 0:
        g = g_star * math.sqrt(_own_energy(tasks, B_orig) / max(_energy(tasks, M_c), 1e-300))
        return g * B_out, g, g
    Pb = U_s @ (U_s.t() @ B_orig)
    Pm = U_s @ (U_s.t() @ M_c)
    e_own_s, e_own_p = _own_energy(tasks, Pb), _own_energy(tasks, B_orig - Pb)
    e_c_s, e_c_p = _energy(tasks, Pm), _energy(tasks, M_c - Pm)
    g_p = g_star * math.sqrt(e_own_p / max(e_c_p, 1e-300))
    g_s = min(g_star * math.sqrt(e_own_s / max(e_c_s, 1e-300)), g_p)
    return g_p * B_out + (g_s - g_p) * (U_s @ (U_s.t() @ B_out)), g_s, g_p


def svc_calibrate_B(B_out, A_out, B_tasks, A_tasks, alpha, count="svc"):
    """Singular Value Calibration (SVC, arXiv 2602.05536, output side) of dW = B_out A_out
    without forming it.  dW = u diag(sig) v^T; a_ri = u_r^T B_i A_i (raw factors B_i, A_i).
      count="svc" (as published): s_ri = <sig_r v_r, a_ri> / ||a_ri||^2
      count="cov" (coverage):     s_ri = clip(<sig_r v_r, a_ri> / sig_r^2, 0, 1), the share of
        the merged component that task i accounts for; bounded, so a task that only leaks a
        small a_ri into direction r cannot claim it (s_ri <= ||a_ri|| / sig_r).
    gamma_r = K / sum_i max(alpha, s_ri),  sig_r <- gamma_r sig_r.
    Returns (B_new, gamma, sig) with B_new A_out = u diag(gamma sig) v^T; A_out unchanged."""
    Qb, Rb = torch.linalg.qr(B_out)                       # B_out = Qb Rb
    Qa, Ra = torch.linalg.qr(A_out.t())                   # A_out = Ra^T Qa^T
    Uc, sig, Vch, _ = _thin_svd(Rb @ Ra.t())
    u, v = Qb @ Uc, Qa @ Vch.t()
    num, den = [], []
    for Bi, Ai in zip(B_tasks, A_tasks):
        p = u.t() @ Bi                                    # (k, r_i): u_r^T B_i
        num.append(sig * (p * (Ai @ v).t()).sum(1))       # <sig_r v_r, a_ri>
        den.append(((p @ (Ai @ Ai.t())) * p).sum(1))      # ||a_ri||^2
    num, den = torch.stack(num), torch.stack(den)
    if count == "svc":
        # floor at 1e-12 of the largest task's ||a_r||^2: round-off leaks (a ~ 1e-16) would
        # otherwise give s = noise / noise^2
        S = num / den.clamp_min(1e-12 * den.max(0).values.clamp_min(1e-300))
    elif count == "cov":
        S = (num / (sig ** 2).clamp_min(1e-300)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"svc count {count!r}")
    gamma = len(B_tasks) / S.clamp_min(alpha).sum(0)
    return B_out + u @ ((gamma - 1.0)[:, None] * (u.t() @ B_out)), gamma, sig


def _metric(s, Rdim, svd, svd_ridge, dtype, task_w=None):
    if task_w is None:
        Gn = s["G"].cpu().double() / s["n"]
    else:                                                 # sum_k w_k G_k (sum_k G_k == G)
        Gn = sum(float(w) * t["G"].cpu().double() for w, t in zip(task_w, s["tasks"].values()))
        Gn = Gn / s["n"]
    return Gn + svd_ridge * float(Gn.diagonal().mean()) * torch.eye(Rdim, dtype=dtype)


def _own_task_energy(s, B_comb):
    """[tr(B_k G_k[kk] B_k^T)]_k, count-normalised: task k's own output energy."""
    out = []
    for t in s["tasks"].values():
        a, e = t["start"], t["end"]
        Bk = B_comb[:, a:e]
        out.append(float((t["G"].double()[a:e, a:e] * (Bk.t() @ Bk)).sum()) / s["n"])
    return out


def _weighted_router(s, w, tau):
    """R_w = Q_w (G_w + lam I)^-1, G_w = sum_k w_k G_k, Q_w = [w_k Qb_k]_k stacked,
    lam = tau mean diag G_w (count-normalised).  w = 1 is the SSR router."""
    tasks = list(s["tasks"].values())
    Kr = tasks[0]["G"].shape[0]
    Gw = sum(float(wk) * t["G"].double() for wk, t in zip(w, tasks)) / s["n"]
    Qw = torch.zeros(Kr, Kr, dtype=torch.float64)
    for wk, t in zip(w, tasks):
        Qw[t["start"]:t["end"]] = float(wk) * t["Qb"].double() / s["n"]
    lam = float(tau) * float(Gw.diagonal().mean())
    Greg = Gw + lam * torch.eye(Kr, dtype=torch.float64)
    return torch.linalg.solve(Greg, Qw.t()).t(), lam


def _router(s, r_dims, mode, tau, cross):
    if mode in ("mtc_svd", "ctm"):
        return torch.eye(sum(r_dims), dtype=torch.float64), None, True
    R, lam, ok, _ = tcpr.tcpr_router(s["G"].cpu() / s["n"], s["Q"].cpu() / s["n"], r_dims,
                                     "ssr_full" if cross == 1.0 else "ssr_cross", tau, cross)
    return R, lam, ok


def allocate_ranks(stats, comb, mode, rank, tau=0.2, cross=1.0, svd="data", svd_ridge=0.0,
                   r_min=1, r_max=None, task_w=None):
    """Greedy water-filling of sum_l r_l = L * rank on the truncation spectra."""
    spec = {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        if s is None or s["n"] == 0:
            continue
        A = A_comb.double()
        R, _, _ = _router(s, r_dims, mode, tau, cross)
        M = B_comb.double() @ R
        if svd == "data":
            S = torch.linalg.svdvals(M @ _sqrt_psd(_metric(s, A.shape[0], svd, svd_ridge,
                                                           A.dtype, task_w)))
        else:
            S = compress(M, A, 1)[3]
        e = S ** 2
        spec[name] = (e / e.sum().clamp_min(1e-300)).tolist()
    hi = {n: min(len(v), r_max or len(v)) for n, v in spec.items()}
    alloc = {n: min(r_min, hi[n]) for n in spec}
    budget = rank * len(spec) - sum(alloc.values())
    # marginal gains are non-increasing per layer, so a global sort is the greedy optimum
    gains = sorted(((v[j], n) for n, v in spec.items() for j in range(alloc[n], hi[n])),
                   reverse=True)
    for _, n in gains[:max(budget, 0)]:
        alloc[n] += 1
    kept = {n: sum(spec[n][:alloc[n]]) for n in spec}
    return alloc, kept


def _task_errors(tasks, M, M_c):
    """Per task k: ||(M - M_c) G_k^{1/2}||_F^2 and ||M G_k^{1/2}||_F^2."""
    D = M - M_c
    DD, MM = D.t() @ D, M.t() @ M
    return [(float((DD * t["G"].double()).sum()), float((MM * t["G"].double()).sum()))
            for t in tasks.values()]


def task_weights(stats, comb, mode, rank, tau=0.2, cross=1.0, svd_ridge=0.0, iters=4,
                 power=2.0, log=None):
    """Task-balanced data metric: W = sum_k w_k G_k with one w per task, shared by all layers.

    e_k = mean over layers of ||(M - M_c) G_k^{1/2}|| / ||M G_k^{1/2}|| (task k's relative
    output error at rank `rank`); multiplicative update w_k <- w_k (e_k / mean e)^power,
    mean w = 1.  Each round is one closed-form truncation per layer.
    """
    L = []
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        if s is None or s["n"] == 0 or not s.get("tasks"):
            continue
        R, _, _ = _router(s, r_dims, mode, tau, cross)
        L.append((s, B_comb.double() @ R, A_comb.double()))
    K = len(L[0][0]["tasks"])
    w = torch.ones(K, dtype=torch.float64)
    hist = []
    for it in range(iters + 1):
        E = torch.zeros(K, dtype=torch.float64)
        for s, M, A in L:
            W = _metric(s, A.shape[0], "data", svd_ridge, A.dtype, w.tolist())
            B_out, _, T, _, _ = compress_metric(M, A, W, rank)
            E += torch.tensor([math.sqrt(a / max(b, 1e-300))
                               for a, b in _task_errors(s["tasks"], M, B_out @ T)])
        e = E / len(L)
        hist.append({"w": [round(float(x), 4) for x in w], "task_err": [round(float(x), 4) for x in e]})
        if log:
            log(f"    task_balance it{it}: mean err {float(e.mean()):.4f} worst {float(e.max()):.4f} "
                f"w {[round(float(x), 2) for x in w]}")
        if it == iters:
            break
        w = w * (e / e.mean()) ** power
        w = w / w.mean()
    return hist[-1]["w"], hist


def rc_solve(stats, comb, mode="ssr_rc", rank=32, tau=0.2, g_star=0.8, norm="energy",
             cross=1.0, pico=None, n_probe=2, seed=0, svd="fro", svd_ridge=0.0,
             ranks=None, ctm_beta=0.5, task_w=None, route_w="uni", prio_w=None,
             svc_alpha=None, svc_count="svc", gain_stats=None, split=None):
    """{layer: (A_out, B_out)}, {layer: diag}.  pico: None or kwargs of pico_calibrate_B.

    split: None, or dict(thresh, m): split_gain in place of the layer-wide energy gain, on the
      shared directions of the original B_comb (shared_dirs).

    svc_alpha: None, or the SVC floor alpha (svc_calibrate_B on the compressed factors,
      overlaps with the raw B_k A_k, before the energy gain); svc_count "svc" | "cov".
    gain_stats: optional stats whose per-task Grams define the energy gain (and the energy
      diagnostics) instead of `stats`; used when `stats` holds refreshed cross statistics
      (refresh_stats.py), so the router changes but the gain rule does not.

    ranks: optional {layer: r_l} from allocate_ranks, overriding the uniform rank.
    ctm_beta: target-norm interpolation of the ctm baseline (compress_ctm).
    task_w: optional per-task weights of the data metric (task_weights).
    route_w: "inv" weights task k's term of the routed ridge RegMean by 1 / own_k
      (own_k = its own output energy in this layer), i.e. the router minimises the
      sum of per-task RELATIVE errors instead of absolute ones, so a subject with a
      small delta in a layer is not traded away for a large one.  The data-metric
      truncation then uses the same weighted Gram.  "uni" = SSR.
      "prio": fixed per-task weights prio_w (mean 1) on the same weighted router, i.e.
      a user-chosen priority for some subjects; same weighted Gram in the truncation.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if mode == "ssr_rc_noscale":
        norm = "none"
    if (route_w == "prio") != (prio_w is not None):
        raise ValueError("route_w='prio' needs prio_w (and only it)")
    if route_w != "uni" and (mode not in ("ssr_full", "ssr_rc", "ssr_rc_noscale")
                             or cross != 1.0 or task_w is not None or ranks is not None):
        raise ValueError("route_w needs the plain SSR router, uniform ranks, no task_balance")
    gen = torch.Generator().manual_seed(seed)
    merged, diag = {}, {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        if s is None or s["n"] == 0:
            merged[name] = (A_comb, B_comb)                   # passthrough, as ssr_solve
            diag[name] = {"passthrough": True}
            continue
        tasks = s.get("tasks") or {}
        if gain_stats is not None:
            tasks = gain_stats[name].get("tasks") or {}
        if norm in ("energy", "post_svd_only") and not tasks:
            raise ValueError(f"{name}: norm={norm!r} needs per-task stats "
                             f"(stats file built without keep_tasks)")
        A = A_comb.double()
        B_orig = B_comb.double()
        Rdim = A.shape[0]
        d = {"tau": tau, "R_dim": Rdim, "rank": None if mode == "ssr_full" else int(rank),
             "pico_enabled": pico is not None,
             "pico_strength": pico.get("strength") if pico else None}

        B = B_orig
        if pico is not None and pico.get("mode") == "paper":
            B, sc, a = pico_paper_B(B_orig, r_dims)
            d["pico_scores"] = [round(float(x), 4) for x in sc[:8]]
            d["pico_n_damped"] = float((1.0 - a).sum())     # removed dims, sum_j (1 - a_j)
        elif pico is not None:
            B, shares, nd = pico_calibrate_B(B_orig, r_dims, gen=gen,
                                             **{k: v for k, v in pico.items() if k != "mode"})
            d["pico_shares"] = [round(float(x), 4) for x in shares]
            d["pico_n_damped"] = nd

        w_l = None
        if mode in ("mtc_svd", "ctm"):
            R = torch.eye(Rdim, dtype=A.dtype)                # dW = sum_k B_k A_k
            lam, ok = None, True
        elif route_w == "inv":
            own = _own_task_energy(s, B_orig)
            mo = sum(own) / len(own)
            w_l = [mo / max(o, 1e-30 * mo) for o in own]      # mean-own scale: lam comparable
            R, lam = _weighted_router(s, w_l, tau)
            ok = True
            d["route_w"] = [round(x, 4) for x in w_l]
        elif route_w == "prio":
            w_l = [float(x) for x in prio_w]
            R, lam = _weighted_router(s, w_l, tau)
            ok = True
            d["route_w"] = [round(x, 4) for x in w_l]
        else:
            R, lam, ok, _ = tcpr.tcpr_router(s["G"].cpu() / s["n"], s["Q"].cpu() / s["n"],
                                             r_dims, "ssr_full" if cross == 1.0 else "ssr_cross",
                                             tau, cross)
        d["lambda"], d["solve_fallback"] = lam, not ok
        M = B @ R                                             # dW = M A_comb
        r_l = ranks[name] if ranks is not None else rank
        d["rank"] = None if mode == "ssr_full" else int(r_l)

        if mode == "ssr_full":
            B_out, A_out, T, nfb = M, A, torch.eye(Rdim, dtype=A.dtype), 0
            d.update(rel_fro_error=0.0, rel_data_error=0.0)
        else:
            if mode == "ctm":
                B_out, A_out, T, Sc, nfb = compress_ctm(B, A, r_dims, r_l, ctm_beta)
            elif svd == "data":
                W = _metric(s, Rdim, svd, svd_ridge, A.dtype, task_w if w_l is None else w_l)
                B_out, A_out, T, Sc, nfb = compress_metric(M, A, W, r_l)
            else:
                B_out, A_out, T, Sc, nfb = compress(M, A, r_l)
            k = B_out.shape[1]
            d["r_eff"] = k
            d["svd"] = svd
            d["singular_value_head"] = [float(x) for x in Sc[:5]]
            # rel ||dW - dW_c||_F, factored
            dM = M - B_out @ T
            d["rel_fro_error"] = math.sqrt(max(_fro2(dM, A), 0.0) / max(_fro2(M, A), 1e-300))
            # random probes x ~ N(0, I): |(B R A) x - (B_out A_out) x| / |(B R A) x|
            X = torch.randn(A.shape[1], n_probe, generator=gen, dtype=A.dtype)
            Y = M @ (A @ X)
            d["rel_probe_error"] = float((Y - B_out @ (A_out @ X)).norm() / Y.norm())
            d["A_out_eq_TA"] = float((A_out - T @ A).norm() / A_out.norm())
        d["svd_fallback"] = nfb

        if svc_alpha is not None:                             # spectral calibration, raw B_k
            blk = tcpr._blocks(r_dims)
            B_out, gam, sig = svc_calibrate_B(B_out, A_out, [B_orig[:, a:b] for a, b in blk],
                                              [A[a:b] for a, b in blk], svc_alpha, svc_count)
            o = torch.argsort(sig, descending=True)[:8]
            d["svc_gamma_head"] = [round(float(x), 4) for x in gam[o]]
            d["svc_gamma_wmean"] = float((gam * sig).sum() / sig.sum().clamp_min(1e-300))

        M_c = B_out @ T                                     # dW_c = M_c A_comb
        if tasks:
            E_own = _own_energy(tasks, B_orig)
            E_b, E_a = _energy(tasks, M), _energy(tasks, M_c)
            d.update(energy_own=E_own, energy_before_svd=E_b, energy_after_svd=E_a,
                     data_gain_before_svd=math.sqrt(E_b / E_own),
                     data_gain_after_svd=math.sqrt(E_a / E_own))
            if mode != "ssr_full":
                d["rel_data_error"] = math.sqrt(_energy(tasks, M - M_c) / max(E_b, 1e-300))
        if split is not None:                                 # per-part energy gain
            if norm not in ("energy", "post_svd_only"):
                raise ValueError("split gain needs norm='energy'")
            U_s, sh = shared_dirs(B_orig, r_dims, split["thresh"], split["m"])
            B_out, g_s, g_p = split_gain(B_out, T, U_s, tasks, B_orig, g_star)
            d.update(split_n=int(U_s.shape[1]), split_g_shared=g_s, split_g_spec=g_p,
                     split_shares=[round(float(x), 4) for x in sh])
            g = 1.0
        elif norm in ("energy", "post_svd_only"):
            g = g_star * math.sqrt(d["energy_own"] / max(d["energy_after_svd"], 1e-300))
        elif norm == "fro":
            ref = sum(_fro2(B_orig[:, a:b], A[a:b]) for a, b in tcpr._blocks(r_dims))
            g = g_star * math.sqrt(ref / max(_fro2(B_out, A_out), 1e-300))
        else:
            g = 1.0
        d["g"] = g
        merged[name] = (A_out.float(), (g * B_out).float())   # B_out *= g, A_out unchanged
        diag[name] = d
    return merged, diag


def arm_stem(mode, rank, tau, g, norm, cross=1.0, pico=None, svd="fro", svd_ridge=0.0,
             alloc="uniform", ctm_beta=0.5, tb=0, route_w="uni", prio_tag="", svc_alpha=None,
             svc_count="svc", split=None):
    if mode == "ssr_rc_noscale":
        norm = "none"
    bits = {"ssr_full": ["ssr_full"], "mtc_svd": ["mtc_svd", f"r{rank}"],
            "ctm": ["ctm", f"r{rank}", f"b{ctm_beta:g}"]}.get(mode, ["ssr_rc", f"r{rank}"])
    if mode not in ("mtc_svd", "ctm"):
        bits.append(f"tau{tau:g}")
    bits.append({"none": "noscale", "fro": f"fro{g:g}"}.get(norm, f"g{g:g}"))
    if svd == "data" and mode not in ("ssr_full", "ctm"):
        bits.append("dsvd" if svd_ridge == 0 else f"dsvd{svd_ridge:g}")
    if tb and svd == "data" and mode not in ("ssr_full", "ctm"):
        bits.append("tb")
    if alloc == "global" and mode != "ssr_full":
        bits.append("ga")
    if cross != 1.0 and mode not in ("mtc_svd", "ctm"):
        bits.append(f"cr{cross:g}")
    if route_w != "uni" and mode not in ("mtc_svd", "ctm"):
        bits.append(f"rw{route_w}{prio_tag}")
    if pico is not None and pico.get("mode") == "paper":
        bits.append("picoP")
    elif pico is not None:
        dflt = (pico["strength"], pico["thresh"], pico["m"]) == (0.5, 0.5, 8)
        bits.append("pico" if dflt else
                    f"pico{pico['strength']:g}t{pico['thresh']:g}m{pico['m']}")
        if pico["rescale"]:
            bits[-1] += "n"
        if pico.get("ctrl", "share") == "random":
            bits[-1] += "rnd"
        elif pico.get("ctrl", "share") == "random_norm":
            bits[-1] += "rndn"
    if svc_alpha is not None:
        bits.append(f"{'svc' if svc_count == 'svc' else 'svcc'}{svc_alpha:g}")
    if split is not None:
        bits.append("sg" if (split["thresh"], split["m"]) == (0.5, 8) else
                    f"sg{split['thresh']:g}m{split['m']}")
    return "_".join(bits)


def _med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", nargs="+", default=["ssr_rc"], choices=MODES)
    ap.add_argument("--rank", nargs="+", type=int, default=[32])
    ap.add_argument("--tau", nargs="+", type=float, default=[0.2])
    ap.add_argument("--g_star", nargs="+", type=float, default=[0.8])
    ap.add_argument("--norm", default="energy", choices=NORMS)
    ap.add_argument("--cross", type=float, default=1.0,
                    help="1 = full SSR router (main); 0 = cr0 ablation")
    ap.add_argument("--pico", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pico_mode", default="thresh", choices=("thresh", "paper"),
                    help="thresh: top-m shared-direction damping (below); paper: published Pico")
    ap.add_argument("--pico_strength", type=float, default=0.5)
    ap.add_argument("--pico_share_thresh", type=float, default=0.5)
    ap.add_argument("--pico_m", type=int, default=8)
    ap.add_argument("--pico_rescale", action="store_true")
    ap.add_argument("--pico_ctrl", default="share", choices=("share", "random", "random_norm"),
                    help="random: control, same damping on random directions of span(B_comb)")
    ap.add_argument("--svd", nargs="+", default=["fro"], choices=("fro", "data"),
                    help="truncation metric: weight Frobenius or calibration-data output")
    ap.add_argument("--svd_ridge", type=float, default=0.0,
                    help="data metric W = G + rho mean(diag G) I")
    ap.add_argument("--rank_alloc", nargs="+", default=["uniform"], choices=("uniform", "global"),
                    help="global: water-fill sum_l r_l = L * rank on the truncation spectra")
    ap.add_argument("--r_min", type=int, default=1)
    ap.add_argument("--task_balance", type=int, default=0,
                    help="rounds of task reweighting of the data metric (--svd data); 0 = off")
    ap.add_argument("--tb_power", type=float, default=2.0)
    ap.add_argument("--ctm_beta", type=float, default=0.5,
                    help="ctm baseline: 0 = norm-equalised task targets, 1 = original norms")
    ap.add_argument("--priority", nargs="+", default=None, metavar="SUBJECT=W",
                    help="with --route_w prio: router weight W for SUBJECT (others 1), "
                         "renormalised to mean 1")
    ap.add_argument("--prio_name", default=None,
                    help="short arm tag for --priority (default: list every SUBJECT W)")
    ap.add_argument("--route_w", nargs="+", default=["uni"], choices=("uni", "inv", "prio"),
                    help="inv: per-layer task weights 1/own_k in the router (relative errors)")
    ap.add_argument("--svc_alpha", nargs="+", type=float, default=[0.0],
                    help="SVC floor alpha (svc_calibrate_B, tag svc<alpha>); 0 = off")
    ap.add_argument("--svc_count", default="svc", choices=("svc", "cov"),
                    help="svc: published overlap ratio; cov: bounded coverage (tag svcc<alpha>)")
    ap.add_argument("--split_gain", action="store_true",
                    help="per-part energy gain on shared / specific output directions (tag sg)")
    ap.add_argument("--sg_thresh", type=float, default=0.5)
    ap.add_argument("--sg_m", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--stats_root", required=True,
                    help="dir with stats_K{K}.pt (use data/seed, NOT data/processed)")
    ap.add_argument("--gain_stats_root", default=None,
                    help="stats dir for the energy gain (original calibration) when --stats_root "
                         "holds refreshed cross statistics (refresh_stats.py); put it in --prefix")
    ap.add_argument("--loras_root", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = f"K{args.K}"
    stats = torch.load(os.path.join(args.stats_root, f"stats_{tag}.pt"),
                       map_location="cpu", weights_only=False)
    gain_stats = None
    if args.gain_stats_root:
        gain_stats = torch.load(os.path.join(args.gain_stats_root, f"stats_{tag}.pt"),
                                map_location="cpu", weights_only=False)
        assert set(gain_stats) == set(stats)
    subs = args.subjects[:args.K] if args.subjects else common.SUBJECTS[:args.K]
    comb = ssrlib.concat_pairs(load_pairs(subs, args.loras_root))
    print(f"[{tag}] subjects {subs}; {len(stats)} stats layers, {len(comb)} LoRA layers",
          flush=True)
    prio_w, prio_tag = None, ""
    if args.priority:
        pw = dict(x.split("=") for x in args.priority)
        assert set(pw) <= set(subs), set(pw) - set(subs)
        raw = [float(pw.get(s, 1.0)) for s in subs]
        prio_w = [x * len(raw) / sum(raw) for x in raw]
        prio_tag = ("_" + args.prio_name if args.prio_name else
                    "".join(f"_{s}{float(w):g}" for s, w in sorted(pw.items())))
        assert list(args.route_w) == ["prio"], "--priority goes with --route_w prio"
    pico = (None if not args.pico else dict(mode="paper") if args.pico_mode == "paper" else
            dict(strength=args.pico_strength, thresh=args.pico_share_thresh,
                 m=args.pico_m, rescale=args.pico_rescale, ctrl=args.pico_ctrl))

    split = dict(thresh=args.sg_thresh, m=args.sg_m) if args.split_gain else None
    # ssr_full ignores rank/svd, mtc_svd ignores tau: dedupe by stem
    seen = {}
    for mode, r, tau, g, sv, al, rw, sa in itertools.product(args.mode, args.rank, args.tau,
                                                             args.g_star, args.svd,
                                                             args.rank_alloc, args.route_w,
                                                             args.svc_alpha):
        sa = sa if sa > 0 else None
        if rw != "uni" and mode in ("mtc_svd", "ctm"):
            continue
        stem = args.prefix + arm_stem(mode, r, tau, g, args.norm, args.cross, pico, sv,
                                      args.svd_ridge, al, args.ctm_beta, args.task_balance, rw,
                                      prio_tag if rw == "prio" else "", sa, args.svc_count, split)
        seen.setdefault(stem, (mode, r, tau, g, sv, al, rw, sa))

    from safetensors.torch import save_file
    for stem, (mode, r, tau, g, sv, al, rw, sa) in seen.items():
        out = os.path.join(common.MERGED, f"{stem}_{tag}.safetensors")
        if args.dry_run:
            print(f"  would write {out}", flush=True)
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"  {stem}: exists, skip (--overwrite to replace)", flush=True)
            continue
        t0 = time.time()
        ranks, task_w, tb_hist = None, None, None
        if args.task_balance and sv == "data" and mode not in ("ssr_full", "ctm"):
            task_w, tb_hist = task_weights(stats, comb, mode, r, tau, args.cross,
                                           args.svd_ridge, args.task_balance, args.tb_power,
                                           log=lambda m: print(m, flush=True))
        if al == "global" and mode != "ssr_full":
            ranks, kept = allocate_ranks(stats, comb, mode, r, tau, args.cross, sv,
                                         args.svd_ridge, r_min=args.r_min, task_w=task_w)
        merged, diag = rc_solve(stats, comb, mode, r, tau, g, args.norm, args.cross, pico,
                                svd=sv, svd_ridge=args.svd_ridge, ranks=ranks,
                                ctm_beta=args.ctm_beta, task_w=task_w, route_w=rw,
                                prio_w=prio_w if rw == "prio" else None, svc_alpha=sa,
                                svc_count=args.svc_count, gain_stats=gain_stats, split=split)
        save_file(common.emit_peft_state(merged), out)
        cfg = dict(mode=mode, rank=r, tau=tau, g_star=g, norm=args.norm, cross=args.cross,
                   pico=pico, svd=sv, svd_ridge=args.svd_ridge, rank_alloc=al,
                   r_min=args.r_min, ctm_beta=args.ctm_beta,
                   task_balance=args.task_balance, tb_power=args.tb_power, task_w=task_w,
                   route_w=rw, prio_w=prio_w if rw == "prio" else None, svc_alpha=sa,
                   svc_count=args.svc_count if sa else None, split=split, tb_hist=tb_hist,
                   subjects=subs,
                   stats_root=args.stats_root, gain_stats_root=args.gain_stats_root,
                   loras_root=args.loras_root)
        json.dump({"cfg": cfg, "layers": diag},
                  open(out.replace(".safetensors", "_stats.json"), "w"), indent=1)
        dv = list(diag.values())
        rk = sorted(merged[n][0].shape[0] for n in merged)
        rk_s = (f"{rk[0]}" if rk[0] == rk[-1] else
                f"{rk[0]}/{rk[len(rk) // 2]}/{rk[-1]} (min/med/max, sum {sum(rk)})")
        print(f"  {stem:42s} {time.time() - t0:5.1f}s  ranks {rk_s}  "
              f"rel_fro med {_med(x.get('rel_fro_error') for x in dv):.3f}  "
              f"rel_data med {_med(x.get('rel_data_error') for x in dv):.3f}  "
              f"dgain pre-svd {_med(x.get('data_gain_before_svd') for x in dv):.3f} "
              f"post-svd {_med(x.get('data_gain_after_svd') for x in dv):.3f}  "
              f"g med {_med(x.get('g') for x in dv):.3f}  "
              f"fallbacks: solve {sum(bool(x.get('solve_fallback')) for x in dv)} "
              f"svd {sum(x.get('svd_fallback', 0) for x in dv)} "
              f"passthrough {sum(bool(x.get('passthrough')) for x in dv)}"
              + (f"  pico damped/layer {sum(x['pico_n_damped'] for x in dv) / len(dv):.2f}"
                 if pico else "")
              + (f"  svc gamma wmean med {_med(x.get('svc_gamma_wmean') for x in dv):.3f}"
                 if sa else "")
              + (f"  split layers {sum(x.get('split_n', 0) > 0 for x in dv)}  g_s/g_p med "
                 f"{_med(x['split_g_shared'] / x['split_g_spec'] for x in dv if x.get('split_n')):.3f}"
                 if split else ""), flush=True)


if __name__ == "__main__":
    main()

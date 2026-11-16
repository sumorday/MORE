"""Per-layer crowding index and adaptive weights (replaces the fixed share threshold / damping
strength of the de-crowding, and scales the max-margin penalty per layer).

Retention of one part of the merge (as scripts/sg_diag.py, one layer).  M = g B R with the
UNDAMPED B_comb (the crowding we want to measure), X = P (common mode) or I - P (differential):
  alpha^X = sum_k <X M z, X B_k z_k>_k / sum_k ||X B_k z_k||_k^2
(<.,.>_k over subject k's calibration inputs: z the full latent A_comb x, z_k its own block).
alpha = 1: the merge keeps each subject's own delta in that part at full scale.

Crowding index kappa_l:
  "spec": kappa = (1 - alpha_spec) / alpha_spec, clipped at 0  (dilution of the differential part)
  "cmrr": kappa = alpha_shared / alpha_spec - 1, clipped at 0  (common mode kept better than the
          differential mode; 0 in a layer without shared directions, P = 0)
Weight w_l in [0, 1):
  "frac":    w = kappa / (1 + kappa)            ("spec": w = 1 - alpha_spec), no extra knob
  "sigmoid": w = sigmoid(beta (kappa - kappa0))  (as the plain lam0 sigmoid(kappa) never gives 0)
Used as lam_margin^(l) = lam0 w_l and as the de-crowding strength s0 w_l (thresh 0: every
top-m direction is damped by s0 w_l share_j, so a layer whose top direction is not shared
or which is not crowded is left almost unchanged).
"""
from __future__ import annotations

import math

import torch

from . import _SCRIPTS  # noqa: F401
from .spherical_svm import D, blocks


def retention(B, M, U_s, tasks):
    """(alpha_shared, alpha_spec, own_shared, own_spec) of one layer; tasks = s["tasks"].
    alpha_shared is nan when U_s is empty."""
    B, M = B.to(D), M.to(D)
    if U_s.shape[1]:
        PB, PM = U_s @ (U_s.t() @ B), U_s @ (U_s.t() @ M)
    else:
        PB, PM = torch.zeros_like(B), torch.zeros_like(M)
    out = []
    for XB, XM in ((PB, PM), (B - PB, M - PM)):
        cr = ow = 0.0
        for t in tasks.values():
            a, e = t["start"], t["end"]
            Gk = t["G"].to(D)
            Bk = XB[:, a:e]
            ow += float(((Bk.t() @ Bk) * Gk[a:e, a:e]).sum())
            cr += float(((XM.t() @ Bk) * Gk[:, a:e]).sum())
        out.append((cr / ow if ow > 0 else float("nan"), ow))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def kappa(alpha_shared, alpha_spec, kind="spec"):
    if kind == "spec":
        a = min(max(alpha_spec, 1e-6), 1.0)
        return (1.0 - a) / a
    if kind == "cmrr":
        if not math.isfinite(alpha_shared):
            return 0.0
        return max(alpha_shared / max(alpha_spec, 1e-6) - 1.0, 0.0)
    raise ValueError(f"kappa kind {kind!r}")


def weight(k, fn="frac", beta=4.0, kappa0=0.5):
    if fn == "frac":
        return k / (1.0 + k)
    if fn == "sigmoid":
        return 1.0 / (1.0 + math.exp(-beta * (k - kappa0)))
    raise ValueError(f"weight fn {fn!r}")


def layer_schedule(alpha_shared, alpha_spec, kind="spec", fn="frac", lam0=10.0, s0=0.5,
                   beta=4.0, kappa0=0.5):
    """dict(kappa, w, lam_margin, damp_strength) of one layer."""
    k = kappa(alpha_shared, alpha_spec, kind)
    w = weight(k, fn, beta, kappa0)
    return dict(kappa=k, w=w, lam_margin=lam0 * w, damp_strength=s0 * w)

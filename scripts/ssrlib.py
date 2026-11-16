"""SSR merging on FLUX, plus the SSR+ variants.

Everything operates on a single FluxTransformer2DModel instance in bf16 on GPU:

  * inject_lora(model, pairs)  -- additive forward hook, no module surgery
  * calibrate(model, ...)      -- K forwards, one per task LoRA, accumulate G/Q
  * ssr_solve(G, Q, ...)       -- router R per layer, with SSR+ options
  * to_pairs(...)              -- merged (A_comb, B_comb @ R)
"""
from __future__ import annotations

import math

import torch


class LoRAInjector:
    """Attach LoRA deltas to nn.Linear modules via forward hooks.

    pairs: {module_name: (A, B)} with A (r, in), B (out, r), delta = B @ A @ x.
    """

    def __init__(self, model, pairs, dtype=torch.bfloat16, device=None):
        self.handles = []
        self.modules = {}
        for name, mod in model.named_modules():
            if name in pairs:
                A, B = pairs[name]
                if device is not None:
                    # offloaded model: weights live on CPU between forwards, so
                    # the hook has to place the deltas where the activations are
                    pdev = torch.device(device)
                else:
                    try:
                        pdev = next(iter(mod.parameters())).device
                    except StopIteration:
                        pdev = None
                if pdev is not None:
                    A, B = A.to(pdev), B.to(pdev)
                self.modules[name] = mod
                self.handles.append(mod.register_forward_hook(self._make(A, B, dtype)))

    def _make(self, A, B, dtype):
        def hook(mod, inputs, output):
            x = inputs[0]
            delta = (x.to(dtype) @ A.t().to(dtype)) @ B.t().to(dtype)
            return output + delta.to(output.dtype)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def concat_pairs(task_pairs):
    """[{layer: (A,B)}] -> {layer: (A_comb, B_comb, r_dims, active_idx)}."""
    names = set()
    for p in task_pairs:
        names |= set(p)
    out = {}
    for n in sorted(names):
        acts = [k for k, p in enumerate(task_pairs) if n in p]
        As = [task_pairs[k][n][0] for k in acts]
        Bs = [task_pairs[k][n][1] for k in acts]
        out[n] = (torch.cat(As, 0), torch.cat(Bs, 1), [a.shape[0] for a in As], acts)
    return out


def accumulate(comb, task_pairs, model, prompts_fn, n_steps=1, seed=0,
               keep_tasks=False, keep_raw=False, compute_interference=False,
               lora_device=None, log=None):
    """Run one calibration forward per task LoRA; return per-layer G, Q.

    Returns {name: {"G", "Q", "n", "mu", "yy"}}.  yy = sum_k ||A_k X_k||_F^2 is
    the target energy that a residual needs in order to be computed at all;
    it is a scalar per layer, so tracking it costs nothing.

    keep_tasks additionally stores each task's own Gk/Qk (on CPU) so that a
    leave-one-task-out selection can be done later without a second pass.
    keep_raw stores each task's own projected inputs (Kr x n, CPU) under
    tasks[k]["Zt0"], i.e. the *full* concatenated latent of that task's own
    forward.  A cross-covariance with refreshed activations needs exactly this
    object; concatenating per-task blocks is not the same thing, because the
    other blocks describe other tasks' data.  At K=9 the pool is ~5 GB.
    compute_interference calculates layer-wise task interference degree based
    on pairwise cosine similarity of LoRA down-projections.
    lora_device pins the injected deltas to a card, which is needed when the
    model itself is CPU-offloaded and its weights are on CPU between forwards.
    """
    torch.manual_seed(seed)
    result = {n: {"G": None, "Q": None, "n": 0, "mu": None, "yy": 0.0,
                  "tasks": {}} for n in comb}
    raw_chunks = {n: [] for n in comb} if keep_raw else None
    for k, pairs in enumerate(task_pairs):
        inj = LoRAInjector(model, pairs, device=lora_device)
        if log is not None:
            print(f"  task {k+1}/{len(task_pairs)} forward ...", flush=True)
        stats = {n: {"G": None, "Q": None, "mu": None, "yy": 0.0, "n": 0}
                 for n in comb}
        handles = []
        for name, mod in model.named_modules():
            if name not in comb:
                continue
            A_comb, _, r_dims, acts = comb[name]
            if k not in acts:
                continue
            local = acts.index(k)
            start = sum(r_dims[:local])
            end = start + r_dims[local]

            def make(name=name, A_comb=A_comb, start=start, end=end):
                def hook(mod, inputs, output):
                    x = inputs[0].detach()
                    x = x.reshape(-1, A_comb.shape[1]).t()
                    z = A_comb.to(x.device, torch.float32) @ x.float()
                    s = stats[name]
                    if s["G"] is None:
                        s["G"] = torch.zeros(z.shape[0], z.shape[0], device=z.device)
                        s["Q"] = torch.zeros_like(s["G"])
                        s["mu"] = torch.zeros(z.shape[0], device=z.device)
                    s["G"] += z @ z.t()
                    s["Q"][start:end] += z[start:end] @ z.t()
                    s["mu"] += z.sum(1)
                    s["n"] += z.shape[1]
                    # E_k H_k is exactly block k of z, so this is the target energy
                    s["yy"] += float((z[start:end] ** 2).sum())
                    if raw_chunks is not None:
                        raw_chunks[name].append(z.to("cpu", torch.float32))
                    return output
                return hook
            handles.append(mod.register_forward_hook(make()))

        with torch.no_grad():
            prompts_fn(model, k, n_steps)
        for h in handles:
            h.remove()
        inj.remove()
        if log is not None:
            print(f"  task {k+1}/{len(task_pairs)} done", flush=True)

        for n, s in stats.items():
            if s["G"] is None:
                continue
            r = result[n]
            if r["G"] is None:
                r["G"] = torch.zeros_like(s["G"])
                r["Q"] = torch.zeros_like(s["Q"])
                r["mu"] = torch.zeros_like(s["mu"])
            r["G"] += s["G"]
            r["Q"] += s["Q"]
            r["mu"] += s["mu"]
            r["n"] += s["n"]
            r["yy"] += s["yy"]
            if keep_tasks:
                _, _, r_dims, acts = comb[n]
                local = acts.index(k)
                start = sum(r_dims[:local])
                end = start + r_dims[local]
                r["tasks"][k] = {"G": s["G"].cpu(),
                                 "Qb": s["Q"][start:end].cpu(),
                                 "mu": (s["mu"] / s["n"]).cpu(),
                                 "start": start, "end": end,
                                 "yy": s["yy"], "n": s["n"]}
            if keep_raw:
                chunks = raw_chunks[n]
                if chunks:
                    r["tasks"][k]["Zt0"] = torch.cat(chunks, dim=1)
                raw_chunks[n] = []
    for n, r in result.items():
        if r["n"] > 0:
            r["mu"] /= r["n"]

    # Compute layer-wise task interference if requested
    if compute_interference:
        for name in result:
            if result[name]["n"] == 0:
                result[name]["interference"] = 0.0
                continue

            # Get down-projections (A matrices) from all tasks contributing to this layer
            _, _, r_dims, acts = comb[name]
            if len(acts) < 2:
                result[name]["interference"] = 0.0
                continue

            # Compute pairwise cosine similarity between task LoRA down-projections
            As = [task_pairs[k][name][0] for k in acts if name in task_pairs[k]]
            if len(As) < 2:
                result[name]["interference"] = 0.0
                continue

            sims = []
            for i in range(len(As)):
                for j in range(i+1, len(As)):
                    A_i_flat = As[i].flatten().float()
                    A_j_flat = As[j].flatten().float()
                    cos_sim = float(torch.cosine_similarity(
                        A_i_flat.unsqueeze(0), A_j_flat.unsqueeze(0)
                    ))
                    sims.append(abs(cos_sim))

            # Mean absolute cosine similarity as interference degree
            result[name]["interference"] = sum(sims) / len(sims) if sims else 0.0

    return result


def _ridge_solve(G, A, lam, prior, K=None):
    """R = (A + c*lam*I)(G + lam*I)^-1, c=1 for 'ta', 1/K for 'avg', 0 for 'zero'.

    G = Z Z^T (raw sums), A = Q = Y Z^T. With prior='zero' and lam=0 this is
    exactly SSR's R = Q G^-1. With prior='ta' the estimator is pulled toward
    R = I (task arithmetic); with prior='avg' toward R = I/K (the uniform
    average), which on this pool is the only arm that survives at full
    strength.
    """
    Kr = G.shape[0]
    I = torch.eye(Kr, device=G.device, dtype=G.dtype)
    if prior == "ta":
        num = A + lam * I
    elif prior == "avg":
        num = A + lam * I / float(K or 1)
    else:
        num = A
    return torch.linalg.solve(G + lam * I, num.t()).t()


def _residual(G, A, yy, n, R):
    """tr(RGR^T) - 2 tr(RA^T) + yy, i.e. mean_k ||B_comb R Z_k - Y_k||^2."""
    return float((R @ G * R).sum() - 2 * (R * A).sum() + yy) / n


def _gcv_curve(G, A, yy, n, prior, grid):
    """GCV score for every lam in grid, using one eigendecomposition of G.

    With A' = V^T A V and R' = E (D+lam)^-1, E = A' + lam*I for prior 'ta':

        tr(R G R^T) = sum_i e_i/(e_i+lam)^2 * sum_j E_ij^2      (row i)
        tr(R A^T)   = sum_i 1/(e_i+lam)     * sum_j E_ij A'_ij  (row i)

    Both weights are indexed by the ROW of E, and the cross term carries
    1/(e+lam) rather than e/(e+lam)^2.  The smoother trace is unchanged,
        df(lam) = sum_i e_i/(e_i+lam),
    so GCV(lam) = (residual(lam)/n) / (1 - df(lam)/n)^2.
    """
    ev, V = torch.linalg.eigh(G)
    ev = ev.clamp_min(0.0)
    B = V.t() @ A @ V                      # A expressed in G's eigenbasis
    Kr = G.shape[0]
    I = torch.eye(Kr, device=G.device, dtype=G.dtype)
    B2 = (B * B).sum(1)                    # rowsum(B^2)
    BB = (B * B).sum(1)                    # rowsum(B*B), E=B for prior 'zero'
    Bd = B
    scores, res, dfs = [], [], []
    for lam in grid:
        if prior == "ta":
            E = B + lam * I
            e2 = (E * E).sum(1)
            eb = (E * B).sum(1)
        else:
            e2, eb = B2, BB
        w2 = ev / (ev + lam).pow(2)
        w1 = 1.0 / (ev + lam)
        quad = float((e2 * w2).sum())
        cross = float((eb * w1).sum())
        r = (quad - 2 * cross + yy) / n
        df = float((ev / (ev + lam)).sum())
        denom = max(1.0 - df / n, 1e-6)
        scores.append(r / denom ** 2)
        res.append(r)
        dfs.append(df)
    return scores, res, dfs


def diagnose_stats(stats, comb, task_pairs=None, tag=""):
    """Print the numerical health of the calibration before any solve.

    Two quantities decide whether a router can separate tasks at all:
      * cos(A_i, A_j) between tasks' down-projections.  Values near 1 mean the
        tasks share a projection subspace, so z_k = A_k x is nearly identical
        across k and G is close to rank-deficient across blocks.  This is a
        property of the LoRA initialisation, not of task conflict.
      * cond(G) on the count-normalised Gram.  The raw-sum G has cond ~ n * cond
        of the normalised one; the report uses the normalised form so it is
        comparable across calibration set sizes.
    """
    names = [n for n, s in stats.items() if s["n"] > 0]
    if not names:
        print(f"[diag {tag}] no calibrated layers", flush=True)
        return {}

    cosines, conds, null_fracs = [], [], []
    if task_pairs is not None and len(task_pairs) > 1:
        for n in names:
            acts = comb[n][3]
            As = [task_pairs[k][n][0].float().flatten() for k in acts
                  if k < len(task_pairs) and n in task_pairs[k]]
            for i in range(len(As)):
                for j in range(i + 1, len(As)):
                    cosines.append(abs(float(torch.cosine_similarity(
                        As[i][None], As[j][None]))))

    for n in names:
        G = stats[n]["G"].double() / stats[n]["n"]
        ev = torch.linalg.eigvalsh(G).clamp_min(0.0)
        top = float(ev[-1])
        if top <= 0:
            continue
        conds.append(top / float(ev[0].clamp_min(1e-300)))
        null_fracs.append(float((ev < 1e-6 * top).float().mean()))

    def _q(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

    out = {
        "A_cos_median": _q(cosines, 0.5),
        "A_cos_max": max(cosines) if cosines else float("nan"),
        "cond_median": _q(conds, 0.5),
        "cond_p90": _q(conds, 0.9),
        "null_frac_median": _q(null_fracs, 0.5),
    }
    print(f"[diag {tag}] layers={len(names)}  "
          f"cos(A_i,A_j) median {out['A_cos_median']:.4f} max {out['A_cos_max']:.4f}  "
          f"cond(G/n) median {out['cond_median']:.3e} p90 {out['cond_p90']:.3e}  "
          f"null-space frac {out['null_frac_median']:.3f}", flush=True)
    return out


def _layer_gain(s, comb, name, R):
    """Data-space gain of the merged delta against the individual deltas.

    Returns (sum_k ||merged delta on task k data||^2,
             sum_k ||task k's own delta on its data||^2).

    merged delta = B_comb R z, so its energy is tr(R G_k R^T BB) with
    BB = B_comb^T B_comb.  Task k's own delta is B_k z_k, energy
    tr(B_k^T B_k G_k).  Both are read off the calibration stats, so this needs
    no forward pass and folds into the weights like any other router.
    """
    tasks = s.get("tasks") or {}
    if not tasks:
        return None
    _, B_c, _, _ = comb[name]
    BB = (B_c.float().t() @ B_c.float()).to(R.device)
    num = 0.0
    den = 0.0
    for k, t in tasks.items():
        Gk = t["G"].float().to(R.device)
        start, end = t["start"], t["end"]
        Bk = B_c[:, start:end].float().to(R.device)
        m = float(((R @ Gk @ R.t()) * BB.t()).sum())
        e = float((Gk[start:end, start:end] * (Bk.t() @ Bk)).sum())
        num += m
        den += e
    return num, den


def ssr_solve(stats, comb, lambda_reg=1e-4, ridge="abs", tau=0.05,
              center=False, shrink=0.0, prior="zero", gcv_grid=25,
              device="cuda", gain=1.0, spectral_mode="trace", adaptive_tau=False,
              gain_equalize=None, gain_clip=None, gain_profile=None,
              cross=1.0):
    """Router R per layer.

    ridge: 'abs'   fixed lambda (SSR's formulation)
           'rel'   lambda = tau * mean(diag(G))  (scale-free, fixed tau)
           'gcv'   per-layer lambda chosen by generalized cross-validation
    prior: 'zero'  regularizer pulls R toward 0      -- SSR
           'ta'    regularizer pulls R toward I      -- SSR+
    spectral_mode: 'trace'  -> tau * mean(diag(G))  (current RGC)
                   'sigma'  -> tau * sigma_max(G)    (spectral radius)
                   'frob'   -> tau * ||G||_F         (Frobenius norm)
                   'cond'   -> tau * sigma_max(G) * phi(kappa)  (condition-aware)
    adaptive_tau: if True, modulate tau by layer-wise interference degree
    gain_equalize: if a float g, rescale R per layer so that the merged delta's
                   output energy on the calibration set equals g^2 times the sum
                   of the individual deltas' energies.  A single tau gives every
                   layer the same RIDGE strength, but because the spectrum of G
                   differs per layer it gives each layer a different amplifier
                   gain; this makes the gain uniform across layers instead.
    """
    del device
    merged = {}
    diag = {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        Kr = A_comb.shape[0]
        if s is None or s["n"] == 0:
            merged[name] = (A_comb, B_comb)
            diag[name] = {"ridge": float("nan"), "cond": float("nan"), "fallback": True}
            continue
        n = s["n"]
        G = s["G"].double()
        # R = Q G^-1: rows of Q are the block (Y_k H_k^T -> block k), so Q goes
        # in untransposed.  _ridge_solve returns A @ G^-1 for symmetric G.
        A = s["Q"].double()
        if cross != 1.0:
            # R = (G/K)(G+lamI)^-1 + cross*(Q - G/K)(G+lamI)^-1: a
            # variance-whitened average plus a scaled cross-task correction.
            Gw = G / float(len(r_dims))
            A = Gw + float(cross) * (A - Gw)
        yy = float(s.get("yy", 0.0))
        if center:
            # Per-task centering: G = sum_k (G_k - n_k m_k m_k^T). A single
            # global mean is wrong here, because mu is (1/n) sum_k z_k.sum(1)
            # and each z_k has its own offset; the cross terms do not cancel.
            tasks = s.get("tasks") or {}
            if len(tasks) != len(r_dims):
                raise ValueError(
                    f"{name}: center=1 needs per-task stats, but the stats file "
                    f"has {len(tasks)} task entries (rebuild with keep_tasks=1)")
            for t in tasks.values():
                m = t["mu"]
                G -= t["n"] * torch.outer(m, m)
                A -= t["n"] * torch.outer(m, m)

        # Compute scale based on spectral_mode
        if ridge == "rel":
            if spectral_mode == "sigma":
                # Spectral radius: maximum eigenvalue (spectral norm for symmetric G)
                ev = torch.linalg.eigvalsh(G).clamp_min(0)
                scale = float(ev[-1])
            elif spectral_mode == "frob":
                # Frobenius norm
                scale = float(torch.norm(G, 'fro'))
            elif spectral_mode == "cond":
                # Condition-number aware: sigma_max * phi(kappa)
                ev = torch.linalg.eigvalsh(G).clamp_min(0)
                kappa = float(ev[-1] / ev[0].clamp_min(1e-30))
                phi_kappa = 1.0 + 0.1 * math.log(max(kappa, 1.0))
                scale = float(ev[-1]) * phi_kappa
            else:  # "trace" (default, current RGC)
                scale = float(torch.diagonal(G).sum() / Kr)
        else:
            scale = float(torch.diagonal(G).sum() / Kr)

        if ridge == "gcv":
            grid = [scale * (10.0 ** e)
                    for e in torch.linspace(-6, 2, gcv_grid).tolist()]
            scores, _, dfs = _gcv_curve(G, A, yy, n, prior, grid)
            best = int(min(range(len(grid)), key=lambda i: scores[i]))
            lam = grid[best]
        else:
            # Adaptive tau based on layer-wise interference
            tau_layer = tau
            if adaptive_tau and ridge == "rel":
                interference = s.get("interference", 0.5)
                # Higher interference -> stronger regularization
                tau_layer = tau * (1.0 + interference)

            # The reference implementation solves on the COUNT-NORMALISED
            # Gram, (G/n + lam I) R^T = Q^T/n, so an absolute lambda has to be
            # rescaled by n to mean the same thing here, where G is the raw sum.
            lam = tau_layer * scale if ridge == "rel" else float(lambda_reg) * n
            dfs = None

        try:
            R = _ridge_solve(G, A, lam, prior, K=len(r_dims))
            ev = torch.linalg.eigvalsh(G).clamp_min(0)
            cond = float(ev[-1] / ev[max(0, ev.numel() - 2)].clamp_min(1e-30))
            ok = True
        except Exception:
            R = torch.eye(Kr, device=G.device, dtype=G.dtype)
            cond, ok = float("nan"), False
        if shrink > 0:
            R = R * (1.0 - shrink) + torch.eye(Kr, device=R.device, dtype=R.dtype) * shrink
        if gain != 1.0:
            R = R * gain
        rel_gain = 1.0
        if gain_equalize is not None:
            lg = _layer_gain(s, comb, name, R.float())
            if lg is not None and lg[0] > 0 and lg[1] > 0:
                rel_gain = float(gain_equalize) * math.sqrt(lg[1] / lg[0])
                R = R * rel_gain
        if gain_clip is not None:
            # Per-layer suppression of the amplification outliers, direction
            # untouched.  Average's gain distribution has a tail reaching 74x;
            # clipping the layer gain at g_max costs only a scalar per layer.
            lg = _layer_gain(s, comb, name, R.float())
            if lg is not None and lg[0] > 0 and lg[1] > 0:
                g = math.sqrt(lg[0] / lg[1])
                if g > gain_clip:
                    R = R * (gain_clip / g)
        if gain_profile is not None and name in gain_profile:
            # Force this layer's data-space gain to a prescribed value while
            # leaving its direction alone.  Decouples "per-layer gain profile"
            # from "per-layer direction" for the ablation.
            lg = _layer_gain(s, comb, name, R.float())
            if lg is not None and lg[0] > 0 and lg[1] > 0:
                g = math.sqrt(lg[0] / lg[1])
                R = R * (float(gain_profile[name]) / max(g, 1e-12))
        merged[name] = (A_comb, torch.mm(B_comb.to(R.device), R.float()))
        diag[name] = {"ridge": lam, "ridge_rel": lam / scale, "cond": cond,
                      "fallback": not ok, "n": n, "gain_scale": rel_gain}
    return merged, diag


def _task_residual(Gk, Qb, yy, start, end, R):
    """f_j(R) = ||B_comb R Z_j - B_j H_j||^2 for one task.

    With B_comb^T B_comb = I, tr((B_comb R Z_j)(B_j H_j)^T) = tr(R Q_j) where Q_j
    carries H_j Z_j^T in ROWS start:end.  tr(R Q_j) = sum_a sum_b R[a,b] Q_j[b,a],
    so the linear term uses the COLUMN slice R[:, start:end] against Q_j's
    transpose.  The quadratic term needs the full G_j.
    """
    quad = float((R @ Gk * R).sum())
    lin = float((R[:, start:end].float() * Qb.t().float()).sum())
    return quad - 2 * lin + yy


def ssr_loto(comb, stats, prior="ta", grid_n=13, span=(-6, 2),
             device="cuda", log=None):
    """Choose the ridge strength per layer by leave-one-task-out calibration.

    GCV assumes i.i.d. residuals in Y; here Y_k = E_k H_k is exact given Z_k,
    so the only residual is model mismatch, which is correlated across tasks
    and invisible to GCV.  Holding out one task and solving on the rest gives
    an honest estimate of how a router transfers to a task it never saw.

    Retention for task j is  1 - f_j(R_{-j}) / yy_j,   i.e. the fraction of
    task j's own delta energy that survives the merge, and selecting lambda to
    maximise it is the training-side analogue of the paper's Recovery Rate.
    """
    del device
    merged, diag = {}, {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        s = stats.get(name)
        Kr = A_comb.shape[0]
        tasks = s.get("tasks") if s else None
        if not tasks or len(tasks) < 2:
            merged[name] = (A_comb, B_comb)
            diag[name] = {"ridge": float("nan"), "fallback": True}
            continue

        ks = sorted(tasks)
        Gk = [tasks[k]["G"].to(torch.float32) for k in ks]
        Qb = [tasks[k]["Qb"].to(torch.float32) for k in ks]
        yy = torch.tensor([tasks[k]["yy"] for k in ks], dtype=torch.float32)
        st = [tasks[k]["start"] for k in ks]
        en = [tasks[k]["end"] for k in ks]
        G = torch.stack(Gk).sum(0)
        A = torch.zeros_like(G)
        for k, q in zip(ks, Qb):
            A[tasks[k]["start"]:tasks[k]["end"]] += q
        scale = float(torch.diagonal(G).sum() / Kr)
        grid = [scale * 10.0 ** e for e in torch.linspace(span[0], span[1], grid_n).tolist()]

        # For every fold j and every lambda, R = (A_-j + lam I)(G_-j + lam I)^-1
        K, L = len(ks), len(grid)
        lam = torch.tensor(grid, dtype=G.dtype).repeat(K, 1)            # K x L
        I = torch.eye(Kr, dtype=G.dtype)
        Gm = (G.unsqueeze(0) - torch.stack(Gk)).unsqueeze(1)            # K x 1 x Kr x Kr
        Am = (A.unsqueeze(0) - A).unsqueeze(1)
        Gb = (Gm + lam.view(K, L, 1, 1) * I).reshape(K * L, Kr, Kr)
        if prior == "ta":
            Ab = (Am + lam.view(K, L, 1, 1) * I).reshape(K * L, Kr, Kr)
        else:
            Ab = Am.expand(K, L, Kr, Kr).reshape(K * L, Kr, Kr)
        R = torch.linalg.solve(Gb, Ab.transpose(-1, -2)).transpose(-1, -2)
        R = R.reshape(K, L, Kr, Kr)

        ret = torch.zeros(K, L, dtype=G.dtype)
        for j in range(K):
            quad = torch.einsum("lab,ac,lcb->l", R[j], Gk[j], R[j])
            lin = torch.einsum("lac,ac->l", R[j, :, :, st[j]:en[j]], Qb[j].t())
            f = quad - 2 * lin + yy[j]
            ret[j] = 1.0 - f / yy[j]
        mean_ret = ret.mean(0)

        bi = int(torch.argmax(mean_ret))
        lam_best = float(grid[bi])
        R_best = _ridge_solve(G, A, lam_best, prior, K=K)
        merged[name] = (A_comb, torch.mm(B_comb.to(R_best.device, R_best.dtype), R_best))
        diag[name] = {"ridge": lam_best, "ridge_rel": lam_best / scale,
                      "loto_ret": float(mean_ret[bi]),
                      "loto_ret_min": float(ret[:, bi].min()),
                      "fallback": False, "n": int(s["n"])}
        if log is not None and name in log:
            log[name] = diag[name]
    return merged, diag



def baselines(comb, mode="ta", K=None):
    """Closed-form baselines inside the same container.

    task_arith:    R = I            (sum of deltas)  -> A_comb, B_comb
    average:       R = I/K on the block structure    -> (1/K) * sum deltas
    """
    out = {}
    for name, (A_comb, B_comb, r_dims, _) in comb.items():
        Kr = A_comb.shape[0]
        if mode == "average":
            K = K or len(r_dims)
            R = torch.eye(Kr, device=A_comb.device) / K
            out[name] = (A_comb, torch.mm(B_comb, R))
        else:
            out[name] = (A_comb, B_comb)
    return out

"""Random-path gradient penalty (WGAN-GP style) for the SSR calibration (ssrmc).

Batch form (the definition; any f, used by test_random_gp.py on the linear layer):
  z_mix = eps z_i + (1 - eps) z_perm(i),  eps ~ U(0, 1) per sample, perm = randperm(B)
  GP    = mse( J_merged(z_mix)^T v,  eps J_a(z_mix)^T v + (1 - eps) J_b(z_mix)^T v )
with a = subject of sample i, b = subject of perm(i), J_k the Jacobian of subject k's own LoRA
map and v a probe (v = 1 is the out.sum() of the recipe, which only constrains J^T 1; v ~ N(0, I)
makes GP an unbiased estimate of the squared Frobenius distance of the Jacobians).

ssrmc is linear per layer, dW = B_comb R A_comb, own LoRA k = B_comb E_k A_comb (E_k selects
block k), so every Jacobian is constant and z_mix drops out.  With balanced subjects
E_{a,b,eps}[eps E_a + (1 - eps) E_b] = I/K, so up to a constant
  z space (latent z = A_comb x, the space of the SSR data term):  GP = ||R - I/K||_F^2
  x space (layer weight, W = B R A):                              GP = ||B_comb (R - I/K) A_comb||_F^2
i.e. a Tikhonov pull of the SSR router toward the uniform average (task arithmetic / K).
Closed form of  min_R  data(R) + 1e-4 ||R||^2 + c GP(R),  data = tr(R G R^T - 2 R Q^T)/n:
  z:  R = (Q/n + c I/K) (G/n + (1e-4 + c) I)^-1        (== ssrlib._ridge_solve prior "avg")
  x:  R G' + c Bg R Ag = Q/n + (c/K) Bg Ag,  G' = G/n + 1e-4 I, Bg = B^T B, Ag = A A^T
      (Sylvester; rows decouple in the eigenbasis of Bg)
c = lam * (trace of the data Hessian) / (trace of the GP Hessian), so lam = 1 weighs GP like the
data term:  z: c = lam tr(G')/Kr;  x: c = lam Kr tr(G') / (tr(Ag) tr(Bg)).
"""
import torch

D = torch.float64
RIDGE = 1e-4


def gp_scale(lam, space, Gp, Bg=None, Ag=None):
    Kr = Gp.shape[0]
    if space == "z":
        return lam * float(torch.trace(Gp)) / Kr
    return lam * Kr * float(torch.trace(Gp)) / (float(torch.trace(Ag)) * float(torch.trace(Bg)))


def gp_solve(G, Q, n, K, lam, space, B=None, A=None, c=None, ridge=RIDGE, chunk=64):
    """SSR router with the random-path GP (see module doc). G, Q raw sums (ssrlib.accumulate),
    B (out, Kr), A (Kr, in) for space "x". Returns (R, c)."""
    G, Q = G.to(D) / n, Q.to(D) / n
    Kr = G.shape[0]
    I = torch.eye(Kr, dtype=D, device=G.device)
    Gp = G + ridge * I
    if space == "z":
        c = gp_scale(lam, "z", Gp) if c is None else c
        return torch.linalg.solve(Gp + c * I, (Q + c * I / K).t()).t(), c
    assert space == "x", space
    Bg = B.to(D).t() @ B.to(D)
    Ag = A.to(D) @ A.to(D).t()
    c = gp_scale(lam, "x", Gp, Bg, Ag) if c is None else c
    s, U = torch.linalg.eigh(Bg)
    s = s.clamp_min(0)
    rhs = (U.t() @ (Q + (c / K) * Bg @ Ag)).contiguous()
    out = torch.empty_like(rhs)
    for i0 in range(0, Kr, chunk):            # SPD per row; batched CPU getrs trips MKL (DLASWP)
        i1 = min(Kr, i0 + chunk)
        M = (Gp[None] + c * s[i0:i1, None, None] * Ag[None]).contiguous()
        out[i0:i1] = torch.cholesky_solve(rhs[i0:i1, :, None].contiguous(),
                                          torch.linalg.cholesky(M))[..., 0]
    return U @ out, c


def random_path_gp(f_merged, f_own, z, subj, probe="gauss", gen=None):
    """Batch-level random-path GP of the recipe, generic f (autograd, create_graph=True so it
    back-propagates into f_merged's parameters). f_own(z, subj) applies each sample's own LoRA."""
    B = z.shape[0]
    perm = torch.randperm(B, generator=gen, device="cpu").to(z.device)
    eps = torch.rand((B,) + (1,) * (z.dim() - 1), generator=gen, device="cpu").to(z.device, z.dtype)
    z_mix = (eps * z + (1 - eps) * z[perm]).detach().requires_grad_(True)
    out = f_merged(z_mix)
    v = (torch.randn(out.shape, generator=gen, device="cpu").to(out.device, out.dtype)
         if probe == "gauss" else torch.ones_like(out))
    g_mix = torch.autograd.grad((out * v).sum(), z_mix, create_graph=True)[0]
    tgt = []
    for s in (subj, subj[perm]):
        zz = z_mix.detach().requires_grad_(True)
        tgt.append(torch.autograd.grad((f_own(zz, s) * v).sum(), zz)[0].detach())
    g_tgt = eps * tgt[0] + (1 - eps) * tgt[1]
    return torch.nn.functional.mse_loss(g_mix, g_tgt)

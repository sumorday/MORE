"""Training-free layout control for two-subject prompts on FLUX (MMDiT): early-step regional
attention masks (Regional-Prompting-FLUX / Stitch / FreeFuse style) + optional region-gated LoRA
parts with a fixed layout.

Layout: one boolean region per prompt subject over the N image tokens (latent grid, row-major);
image tokens in no region are background.  layout_lr: subject 0 left half, subject 1 right half,
with an optional central background stripe (the pair prompts are "a {a} and a {b} ...").
layout_from_maps: automatic layout from the subject attention maps of a short first pass
(MapRecorder; FreeFuse / LoRAShop take it from the base model, first ~5 steps, last double block).

RegionalAttention wraps the processor of selected double / single blocks; for steps
i < round(frac N) it calls the wrapped processor with an additive attention bias over the joint
sequence [text (txt_len) | image (N)] (both block types put the text first):
  blocked (bias = neg, default -inf):
    image query in region k  -> text tokens of subject j != k
    text token of subject k  -> image tokens outside region k (incl. background), text of j != k
    image="bound" blocks only: image query in region k -> image tokens of region j != k
  own (bias = pos, default 0, both directions): region k image <-> text tokens of subject k
  everything else (background image queries, other text tokens) is unrestricted, so every row
  keeps unmasked keys.  emphasis=True (control without a layout): no blocked entries, bias = pos
  between every image token and every subject text token (both directions).  base_ratio r > 0: output = (1 - r) attn(bias) + r attn(no bias) (exact
  for the double blocks too: to_out / to_add_out are affine).  Steps >= round(frac N) and
  unselected blocks are bitwise the wrapped processor.
Steps are counted by a forward pre-hook on the transformer (one transformer call per step, FLUX
dev has no CFG batch); set_layout before every pipeline call.

Bias schedule (step i, n_act = round(frac N) active steps): sched="const" -> own bias pos;
sched="cos" -> pos cos(pi/2 i / n_act).  Blocked entries are applied only for i < n_act; for
i >= n_act, pos_late (if set) puts only the own bias pos_late (no blocking), else the step is the
wrapped processor.  With one region (the whole image) and one token set (single-subject prompts)
regional_bias reduces to emphasis on that subject.

Scene-mass preservation (preserve = letters of the key sets C held fixed: s scene text = valid
non-subject text tokens [0, n_valid), p padding text tokens [n_valid, txt_len), i image tokens):
on every image query row with a bias, C gets the extra logit
  delta = LSE_{j not in C}(s_j + b_j) - LSE_{j not in C}(s_j)        (s: raw logits, b: bias)
so the attention mass on C equals that of the unbiased softmax exactly and the extra mass of the
own subject tokens is taken from the keys outside C only (mass_preserving_attention).  This path
computes the attention itself (fp32, flux_attention) instead of calling the wrapped processor.

LayoutRouter: DynamicRouter with the softmax_max semantics and a fixed image mask from the layout
instead of attention maps: region k -> subject k's part only (weight 1; the other subjects'
parts get weight `other`, default 0), background -> 1 for every active subject, text tokens one-hot on the subjects' own tokens and 1 elsewhere; no capture
processors are installed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusers.models.transformers import transformer_flux as tf

from .dynamic_hook import DynamicRouter, MapCaptureProcessor, subject_maps


def layout_lr(side, n_sub=2, gap=0.0):
    """n_sub vertical strips (left to right) on a side x side grid; a central background stripe
    of width gap * side between adjacent strips.  -> list of (side*side,) bool tensors."""
    cols = torch.arange(side, dtype=torch.float64) + 0.5
    edges = torch.linspace(0, side, n_sub + 1, dtype=torch.float64)
    half = 0.5 * gap * side
    out = []
    for k in range(n_sub):
        lo = edges[k] + (half if k > 0 else 0.0)
        hi = edges[k + 1] - (half if k < n_sub - 1 else 0.0)
        c = (cols >= lo) & (cols < hi)
        out.append(c.unsqueeze(0).expand(side, side).reshape(-1).clone())
    return out


def layout_tb(side, n_sub=2, gap=0.0):
    """layout_lr transposed: n_sub horizontal strips, top to bottom."""
    return [r.view(side, side).t().reshape(-1).clone() for r in layout_lr(side, n_sub, gap)]


def layout_box(side, y0, y1, x0, x1):
    """(side*side,) bool box: rows [y0, y1) x cols [x0, x1) in fractions of the side."""
    c = torch.arange(side, dtype=torch.float64) + 0.5
    rows = (c >= y0 * side) & (c < y1 * side)
    cols = (c >= x0 * side) & (c < x1 * side)
    return (rows[:, None] & cols[None, :]).reshape(-1)


# big-small (foreground / background): subject a a large box low and left, subject b a small
# box high and right (area ratio ~5:1); rows y, cols x in fractions of the side
BS_BOXES = ((0.25, 1.0, 0.0, 0.62), (0.1, 0.45, 0.68, 0.95))


def layout_named(name, side, n_sub=2, gap=0.0):
    """lr / tb strips (n_sub subjects) or bs (2 subjects, BS_BOXES)."""
    if name == "lr":
        return layout_lr(side, n_sub, gap)
    if name == "tb":
        return layout_tb(side, n_sub, gap)
    assert name == "bs" and n_sub == 2, (name, n_sub)
    return [layout_box(side, *b) for b in BS_BOXES]


def swap_regions(regions):
    """the negative layout of SwapGuidance: subject k gets region k+1 (cyclic; 2 subjects: the
    regions exchanged)"""
    return list(regions[1:]) + list(regions[:1])


def regional_bias(txt_len, regions, token_sets, image_bound, neg=float("-inf"), pos=0.0,
                  device=None, dtype=torch.float32):
    """(1, 1, S, S) additive bias over [text | image] (module docstring)."""
    blk, own = regional_parts(txt_len, regions, token_sets, image_bound)
    bias = torch.zeros(blk.shape, dtype=torch.float32)
    bias[own] = pos
    bias[blk] = neg
    return bias.to(device=device, dtype=dtype)[None, None]


def regional_parts(txt_len, regions, token_sets, image_bound):
    """(S, S) bool masks (blocked, own) of regional_bias."""
    N = regions[0].numel()
    S = txt_len + N
    if int(sum(r.long() for r in regions).max()) > 1:
        raise ValueError("regions overlap")
    blk = torch.zeros(S, S, dtype=torch.bool)
    own = torch.zeros(S, S, dtype=torch.bool)
    for k, (r, t) in enumerate(zip(regions, token_sets)):
        img_k = txt_len + torch.nonzero(r, as_tuple=False).flatten()
        not_k = txt_len + torch.nonzero(~r, as_tuple=False).flatten()
        t = torch.as_tensor(t, dtype=torch.long)
        for j, (rj, tj) in enumerate(zip(regions, token_sets)):
            if j == k:
                continue
            tj = torch.as_tensor(tj, dtype=torch.long)
            blk[img_k[:, None], tj[None, :]] = True                 # region k -/-> text j
            blk[t[:, None], tj[None, :]] = True                     # text k -/-> text j
            if image_bound:
                img_j = txt_len + torch.nonzero(rj, as_tuple=False).flatten()
                blk[img_k[:, None], img_j[None, :]] = True          # region k -/-> region j
        blk[t[:, None], not_k[None, :]] = True                      # text k -/-> outside k
        own[img_k[:, None], t[None, :]] = True
        own[t[:, None], img_k[None, :]] = True
    if not (~blk).any(1).all():
        raise ValueError("a query row has every key blocked")
    return blk, own


def region_cross(txt_len, regions):
    """(S, S) bool: image query in region k -> image key in region j != k"""
    N = regions[0].numel()
    S = txt_len + N
    m = torch.zeros(S, S, dtype=torch.bool)
    for k, rk in enumerate(regions):
        for j, rj in enumerate(regions):
            if j != k:
                ik = txt_len + torch.nonzero(rk, as_tuple=False).flatten()
                ij = txt_len + torch.nonzero(rj, as_tuple=False).flatten()
                m[ik[:, None], ij[None, :]] = True
    return m


def emphasis_bias(txt_len, N, token_sets, pos, device=None, dtype=torch.float32):
    """(1, 1, S, S): +pos between all image tokens and all subject text tokens, 0 elsewhere."""
    bias = torch.zeros(txt_len + N, txt_len + N, dtype=torch.float32)
    bias[emphasis_own(txt_len, N, token_sets)] = pos
    return bias.to(device=device, dtype=dtype)[None, None]


def emphasis_own(txt_len, N, token_sets):
    S = txt_len + N
    t = torch.as_tensor(sorted({int(i) for ts in token_sets for i in ts}), dtype=torch.long)
    own = torch.zeros(S, S, dtype=torch.bool)
    own[txt_len:, t] = True
    own[t, txt_len:] = True
    return own


def preserve_cols(txt_len, N, token_sets, n_valid, letters):
    """(S,) bool: the key set C of the scene-mass preservation (module docstring)."""
    c = torch.zeros(txt_len + N, dtype=torch.bool)
    if "s" in letters:
        c[:n_valid] = True
        for t in token_sets:
            c[torch.as_tensor(t, dtype=torch.long)] = False
    if "p" in letters:
        c[n_valid:txt_len] = True
    if "i" in letters:
        c[txt_len:] = True
    return c


def mass_preserving_attention(query, key, value, bias, C, rows, return_probs=False):
    """query/key/value (B, S, H, d); bias (1, 1, S, S) float32; C bool broadcastable to (B, H, S, S)
    (preserved keys); rows (S,) bool (query rows with preservation).  -> (B, S, H, d) in the
    query dtype (and the (B, H, S, S) probabilities).  Rows outside `rows` are plain
    softmax(s + bias)."""
    q, k, v = (x.transpose(1, 2).float() for x in (query, key, value))
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    sb = s + bias
    l1 = torch.logsumexp(sb.masked_fill(C, float("-inf")), -1)
    l0 = torch.logsumexp(s.masked_fill(C, float("-inf")), -1)
    delta = torch.where(rows.to(s.device), l1 - l0, torch.zeros_like(l1))
    del s
    p = torch.softmax(sb.add_(delta.unsqueeze(-1) * C.to(sb.dtype)), -1)
    o = (p @ v).transpose(1, 2).to(query.dtype)
    return (o, p) if return_probs else o


def flux_attention(attn, hidden_states, encoder_hidden_states, image_rotary_emb, attend):
    """FluxAttnProcessor with the attention call replaced by attend(query, key, value) ->
    (B, S, H, d) (same projections, norms, rotary and output layers)."""
    query, key, value, encoder_query, encoder_key, encoder_value = tf._get_qkv_projections(
        attn, hidden_states, encoder_hidden_states)
    query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
    value = value.unflatten(-1, (-1, attn.head_dim))
    if attn.added_kv_proj_dim is not None:
        encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (-1, attn.head_dim)))
        encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (-1, attn.head_dim)))
        encoder_value = encoder_value.unflatten(-1, (-1, attn.head_dim))
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)
    if image_rotary_emb is not None:
        query = tf.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = tf.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
    hidden_states = attend(query, key, value).flatten(2, 3).to(query.dtype)
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [encoder_hidden_states.shape[1],
             hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.contiguous()))
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
        return hidden_states, encoder_hidden_states
    return hidden_states


def nag_combine(pos, neg, rows, scale, tau, alpha):
    """Normalized attention guidance (NAG, Chen et al. 2025, NAGFluxAttnProcessor2_0) on the
    attention outputs (B, S, H, d) before to_out, on the query rows `rows` (S,) bool only:
      g = scale pos - (scale - 1) neg,  g *= min(|g| / |pos|, tau) / (|g| / |pos|)  (L2 over H d),
      out = alpha g + (1 - alpha) pos;   other rows = pos."""
    p, n = pos.float(), neg.float()
    g = scale * p - (scale - 1) * n
    r = g.flatten(2).norm(dim=-1) / p.flatten(2).norm(dim=-1).clamp_min(1e-12)   # (B, S)
    g = g * (torch.minimum(r, torch.full_like(r, tau)) / r.clamp_min(1e-12))[..., None, None]
    out = alpha * g + (1 - alpha) * p
    return torch.where(rows.to(pos.device)[None, :, None, None], out, p).to(pos.dtype)


class RegionalProcessor:
    """Calls the wrapped processor with the regional bias while the controller is active;
    on the NAG steps of the controller (ctl.nag_now) computes the attention itself with the
    bias and with the swapped bias and combines them with nag_combine on the subject-region
    image rows (region-swap NAG: the negative branch is "subject j's tokens in region k")."""

    def __init__(self, ctl, inner, image_bound):
        self.ctl, self.inner, self.image_bound = ctl, inner, image_bound

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        bias = self.ctl.bias_for(self.image_bound, hidden_states)
        if bias is None:
            return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask,
                              image_rotary_emb)
        if self.ctl.nag_now():
            ctl = self.ctl
            bias_n = ctl.bias_for(self.image_bound, hidden_states, swapped=True)
            rows = ctl.nag_rows(bias.device, bias.shape[-1])
            att = lambda q, k, v, b: tf.dispatch_attention_fn(
                q, k, v, attn_mask=b, backend=getattr(self.inner, "_attention_backend", None),
                parallel_config=getattr(self.inner, "_parallel_config", None))
            return flux_attention(
                attn, hidden_states, encoder_hidden_states, image_rotary_emb,
                lambda q, k, v: nag_combine(att(q, k, v, bias), att(q, k, v, bias_n), rows,
                                            ctl.nag, ctl.nag_tau, ctl.nag_alpha))
        if self.ctl.preserve:
            C, rows = self.ctl.preserve_masks(bias.device)
            return flux_attention(attn, hidden_states, encoder_hidden_states, image_rotary_emb,
                                  lambda q, k, v: mass_preserving_attention(q, k, v, bias, C, rows))
        out = self.inner(attn, hidden_states, encoder_hidden_states, bias, image_rotary_emb)
        r = self.ctl.base_ratio
        if r > 0:
            ref = self.inner(attn, hidden_states, encoder_hidden_states, attention_mask,
                             image_rotary_emb)
            if isinstance(out, tuple):
                out = tuple((1 - r) * a + r * b for a, b in zip(out, ref))
            else:
                out = (1 - r) * out + r * ref
        return out


class RegionalAttention:
    """double_blocks / single_blocks: indices whose attention is masked (None = all);
    bound_double / bound_single: the subset that also blocks region-to-region image attention;
    sched / pos_late / preserve: module docstring; rep: soft image repulsion, bias -rep between
    the image tokens of region k and region j != k (both directions), on the blocking steps and
    scaled like pos by the schedule; swapped: the layout with the regions cyclically shifted
    (swap_regions, the negative pass of SwapGuidance); nag / nag_tau / nag_alpha / nag_steps: region-swap NAG
    (RegionalProcessor, nag_combine) on the steps in nag_steps while the bias is on."""

    def __init__(self, transformer, frac=0.3, double_blocks=None, single_blocks=None,
                 bound_double=(), bound_single=(), base_ratio=0.0, neg=float("-inf"), pos=0.0,
                 emphasis=False, sched="const", pos_late=None, preserve="", rep=0.0,
                 nag=0.0, nag_tau=2.5, nag_alpha=0.25, nag_steps=()):
        self.frac, self.base_ratio, self.neg, self.pos = frac, base_ratio, neg, pos
        self.rep, self.swapped = rep, False
        self.nag, self.nag_tau, self.nag_alpha = nag, nag_tau, nag_alpha
        self.nag_steps = set(int(i) for i in nag_steps)
        assert not (nag and (preserve or base_ratio or emphasis)), "nag with preserve/b/em"
        self.emphasis, self.sched, self.pos_late, self.preserve = emphasis, sched, pos_late, preserve
        assert sched in ("const", "cos") and set(preserve) <= set("spi"), (sched, preserve)
        assert not (preserve and base_ratio), "preserve with base_ratio"
        self.step, self.n_steps = -1, None
        self._bias, self._parts = {}, {}
        self.saved = []
        db = range(len(transformer.transformer_blocks)) if double_blocks is None else double_blocks
        sb = range(len(transformer.single_transformer_blocks)) if single_blocks is None \
            else single_blocks
        for blocks, idx, bound in ((transformer.transformer_blocks, db, set(bound_double)),
                                   (transformer.single_transformer_blocks, sb, set(bound_single))):
            for i in idx:
                a = blocks[i].attn
                self.saved.append((a, a.processor))
                a.set_processor(RegionalProcessor(self, a.processor, i in bound))
        self.handle = transformer.register_forward_pre_hook(self._pre)

    def set_layout(self, regions, token_sets, n_steps, txt_len=512, n_valid=None):
        """regions: list of (N,) bool (one per subject), token_sets: matching text-token indices;
        n_valid: number of non-padding text tokens (needed by preserve)."""
        assert len(regions) == len(token_sets) >= 1
        assert n_valid is not None or not self.preserve, "preserve needs n_valid"
        self.regions = [torch.as_tensor(r, dtype=torch.bool).cpu() for r in regions]
        self.token_sets = [list(map(int, t)) for t in token_sets]
        self.txt_len, self.n_steps, self.n_valid = txt_len, n_steps, n_valid
        self.step = -1
        self._bias, self._parts, self._pres = {}, {}, {}

    def _pre(self, module, args):
        self.step += 1

    def off(self):
        """inactive until the next set_layout (e.g. during a first pass)"""
        self.n_steps = None

    def n_active(self):
        return int(round(self.frac * self.n_steps))

    def active(self):
        return self.n_steps is not None and 0 <= self.step < self.n_active()

    def bias_now(self):
        """(pos, blocking on) of the current step, None = no bias (module docstring)."""
        if self.n_steps is None or self.step < 0:
            return None
        n = self.n_active()
        if self.step < n:
            p = self.pos
            if self.sched == "cos":
                p = self.pos * math.cos(0.5 * math.pi * self.step / n)
            return p, True
        if self.pos_late is not None and self.step < self.n_steps:
            return self.pos_late, False
        return None

    def nag_now(self):
        return bool(self.nag) and self.n_steps is not None and self.step in self.nag_steps \
            and len(self.regions) == 2

    def nag_rows(self, device, S):
        """(S,) bool: image tokens of the subject regions"""
        rows = torch.zeros(S, dtype=torch.bool)
        rows[self.txt_len:] = self.regions[0] | self.regions[1]
        return rows.to(device)

    def preserve_masks(self, device):
        if device not in self._pres:
            N = self.regions[0].numel()
            C = preserve_cols(self.txt_len, N, self.token_sets, self.n_valid, self.preserve)
            rows = torch.zeros(self.txt_len + N, dtype=torch.bool)
            rows[self.txt_len:] = True
            self._pres[device] = (C.to(device)[None, None, None], rows.to(device))
        return self._pres[device]

    def bias_for(self, image_bound, x, swapped=None):
        """swapped: None = self.swapped"""
        now = self.bias_now()
        if now is None:
            return None
        pos, blocking = now
        swapped = self.swapped if swapped is None else swapped
        pk = (image_bound, swapped)
        if pk not in self._parts:
            N = self.regions[0].numel()
            regions = swap_regions(self.regions) if swapped else self.regions
            if self.emphasis:
                assert not swapped, "swap with emphasis"
                own = emphasis_own(self.txt_len, N, self.token_sets)
                blk = torch.zeros_like(own)
            else:
                blk, own = regional_parts(self.txt_len, regions, self.token_sets, image_bound)
            self._parts[pk] = (blk, own, region_cross(self.txt_len, regions))
        key = (image_bound, x.device, x.dtype, self.step, swapped)
        if key not in self._bias:
            self._bias = {k: v for k, v in self._bias.items() if k[3] == self.step}
            blk, own, cross = self._parts[pk]
            b = torch.zeros(blk.shape, dtype=torch.float32)
            b[own] = pos
            if blocking and self.rep:
                b[cross] = -self.rep * (pos / self.pos if self.pos else 1.0)
            if blocking:
                b[blk] = self.neg
            # preserve path computes in fp32; the wrapped processor gets the activation dtype
            self._bias[key] = b.to(x.device, torch.float32 if self.preserve else x.dtype)[None, None]
        b = self._bias[key]
        S = b.shape[-1]
        n = x.shape[1] if x.shape[1] == S else x.shape[1] + self.txt_len   # single: joined seq
        if n != S:
            raise ValueError(f"sequence {n} != layout {S}")
        return b

    def remove(self):
        self.handle.remove()
        for a, p in reversed(self.saved):
            a.set_processor(p)
        self.saved.clear()
        self._bias = {}


class MapRecorder:
    """Subject attention maps (dynamic_hook.subject_maps) of selected double blocks, averaged over
    the blocks and the recorded steps; for an automatic layout from a short first pass
    (FreeFuse / LoRAShop: base model, first steps, last double block).  Install BEFORE a
    RegionalAttention (which then wraps the capture processor), remove after it."""

    def __init__(self, transformer, blocks=(18,)):
        self.blocks = transformer.transformer_blocks
        self.saved = {}
        for i in blocks:
            a = self.blocks[i].attn
            self.saved[i] = a.processor
            a.set_processor(MapCaptureProcessor(self))
        self.token_sets, self._acc, self._n = None, None, 0

    def start(self, token_sets):
        self.token_sets = [torch.as_tensor(t, dtype=torch.long) for t in token_sets]
        self._acc, self._n = None, 0

    def wants_maps(self):
        return self.token_sets is not None

    def add_map(self, query, key, txt_len):
        with torch.no_grad():
            m = subject_maps(query, key, txt_len, [t.to(query.device) for t in self.token_sets])
        self._acc = m if self._acc is None else self._acc + m
        self._n += 1

    def stop(self):
        """-> (B, K, N) mean maps; recording off."""
        a = None if self._acc is None else self._acc / self._n
        self.token_sets, self._acc, self._n = None, None, 0
        return a

    def remove(self):
        for i, p in self.saved.items():
            self.blocks[i].attn.set_processor(p)
        self.saved.clear()


def layout_from_maps(a, side, q=0.6, margin=0.1, smooth=3, min_frac=0.03):
    """(K, N) subject maps -> K disjoint regions + background.  Each map is box-smoothed and
    rescaled to mean 1; token i goes to k = argmax_j a_j(i) if a_k(i) is in subject k's top
    (1 - q) fraction and above 1.05 times its mean, and a_k(i) / sum_j a_j(i) >= 1/K + margin;
    a 3x3 majority filter cleans the regions.  Returns (regions, ok): ok False (and layout_lr
    strips) if a region has fewer than min_frac * N tokens."""
    K, N = a.shape
    g = a.float().view(K, 1, side, side)
    if smooth > 1:
        g = F.avg_pool2d(g, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
    a = g.view(K, N)
    a = a / a.mean(-1, keepdim=True).clamp_min(1e-12)
    share = a / a.sum(0, keepdim=True).clamp_min(1e-12)
    top = (a >= torch.quantile(a, q, dim=-1, keepdim=True)) & (a > 1.05)   # flat map: none
    k = a.argmax(0)
    own = F.one_hot(k, K).t().bool() & top & (share >= 1.0 / K + margin)
    f = F.avg_pool2d(own.float().view(K, 1, side, side), 3, stride=1, padding=1,
                     count_include_pad=False).view(K, N)
    regions = [(f[j] > 0.5) & (k == j) for j in range(K)]
    if min(int(r.sum()) for r in regions) < min_frac * N:
        return layout_lr(side, K), False
    return [r.cpu() for r in regions], True


class LayoutRouter(DynamicRouter):
    """DynamicRouter gated by a fixed layout (module docstring).  until: gating only for steps
    < round(until * n_steps), later steps ungated (every active part on every token, == the
    static merge / route st); None = every step."""

    def __init__(self, transformer, factors, until=None, **kw):
        kw.update(mask="softmax_max", gating=True, capture_blocks=[])
        super().__init__(transformer, factors, **kw)
        self.until = until

    def _mask_for(self, stream, x):
        if self.until is not None and self.step >= int(round(self.until * self.n_steps)):
            return torch.ones(1, 1, len(self.active), device=x.device, dtype=x.dtype)
        return super()._mask_for(stream, x)

    def set_layout(self, active, token_sets, regions, n_steps, txt_len=512, passive=(), other=0.0):
        """other: weight of subject j's part in region k != j (0 = hard gating)"""
        self.set_prompt(active, token_sets, n_steps, txt_len=txt_len, passive=passive)
        self._swap_mask = None
        K = len(active)
        assert len(regions) == K
        regions = [torch.as_tensor(r, dtype=torch.bool).cpu() for r in regions]
        if int(sum(r.long() for r in regions).max()) > 1:
            raise ValueError("regions overlap")
        m = torch.ones(1, regions[0].numel(), K)
        for k, r in enumerate(regions):
            m[0, r] = other
            m[0, r, k] = 1.0
        self.img_mask = m

    def set_swapped(self, on):
        """image mask of the swapped layout (swap_regions: region k gets subject k-1's part;
        2 subjects: the columns exchanged); the text mask is unchanged"""
        if on:
            self._swap_mask = self.img_mask
            self.img_mask = torch.roll(self.img_mask, -1, dims=-1)
        elif self._swap_mask is not None:
            self.img_mask, self._swap_mask = self._swap_mask, None

    def wants_maps(self):
        return False

    def _finalize_maps(self):
        pass


class SwapGuidance:
    """Contrastive layout guidance with one extra transformer pass on selected steps:
      v = v + scale * M (v - v_swap),
    v_swap the prediction with the subjects' regions exchanged (region k gets subject j's
    attention bias and, with a LayoutRouter, subject j's LoRA part), M the union of the subject
    regions (background unchanged).  Pushes region k away from "subject j here" (a per-region
    negative prompt that names the other subject, without a second text encoding).
    mode: "all" swap the bias and the LoRA gating, "b" bias only, "l" LoRA gating only.
    steps: step indices with guidance.  Installs an instance forward on the transformer: the
    pre-hooks (step counters) run once per step, the negative pass calls the class forward
    directly (no hooks)."""

    def __init__(self, transformer, ra, rt=None, scale=1.0, steps=(), mode="all"):
        assert mode in ("all", "b", "l") and ra is not None
        assert mode == "b" or isinstance(rt, LayoutRouter), "LoRA swap needs a LayoutRouter"
        self.tr, self.ra, self.rt, self.scale, self.mode = transformer, ra, rt, scale, mode
        self.steps = set(int(i) for i in steps)
        self.n_guided = 0
        self._orig = type(transformer).forward.__get__(transformer)
        transformer.forward = self._forward

    def _swap(self, on):
        if self.mode in ("all", "b"):
            self.ra.swapped = on
        if self.mode in ("all", "l"):
            self.rt.set_swapped(on)

    def _forward(self, *args, **kw):
        out = self._orig(*args, **kw)
        if self.ra.n_steps is None or self.ra.step not in self.steps or not self.scale:
            return out
        self._swap(True)
        try:
            neg = self._orig(*args, **kw)
        finally:
            self._swap(False)
        v, vn = (out[0], neg[0]) if isinstance(out, tuple) else (out.sample, neg.sample)
        M = torch.zeros(v.shape[1], dtype=torch.bool)
        for r in self.ra.regions:
            M |= r
        M = M.to(v.device, v.dtype)[None, :, None]
        g = v + self.scale * M * (v - vn)
        self.n_guided += 1
        if isinstance(out, tuple):
            return (g,) + tuple(out[1:])
        out.sample = g
        return out

    def remove(self):
        del self.tr.forward


class FeaturePush:
    """Forward feature repulsion between the subject regions on the image stream of selected
    double blocks (forward hook on the block output, (encoder_hidden_states, hidden_states)):
      mu_k = mean_{R_k} h,  u_kj = unit(mu_j - mu_k),
      h_n -= beta * relu((h_n - mu_k) . u_kj) u_kj        for n in R_k, j != k  (two regions),
    i.e. the tokens of region k that lean from their region mean towards region j lose that
    component (only along the inter-region axis, only the part towards j).  Centering on the own
    mean matters: with two regions tiling the image, the global mean lies between mu_0 and mu_1
    and d_k = mu_k - mean is exactly -d_j.  Per sample, fp32, on the steps in `steps` (ra.step)
    while ra holds a two-region layout."""

    def __init__(self, transformer, ra, beta=0.5, blocks=None, steps=()):
        self.ra, self.beta, self.steps = ra, beta, set(int(i) for i in steps)
        blocks = range(len(transformer.transformer_blocks)) if blocks is None else blocks
        self.handles = [transformer.transformer_blocks[i].register_forward_hook(self._hook)
                        for i in blocks]
        self.n_pushed = 0

    def _on(self):
        ra = self.ra
        return ra.n_steps is not None and ra.step in self.steps and len(ra.regions) == 2

    def push(self, h):
        """h (B, N, D) -> pushed copy (the formula of the class docstring)."""
        x = h.float()
        R = [r.to(x.device) for r in self.ra.regions]
        mu = [x[:, r].mean(1, keepdim=True) for r in R]                     # (B, 1, D) each
        out = x.clone()
        for k, j in ((0, 1), (1, 0)):
            u = torch.nn.functional.normalize(mu[j] - mu[k], dim=-1)
            a = ((x[:, R[k]] - mu[k]) * u).sum(-1, keepdim=True).clamp_min(0)  # (B, n_k, 1)
            out[:, R[k]] = x[:, R[k]] - self.beta * a * u
        return out.to(h.dtype)

    def _hook(self, mod, inputs, output):
        if not self._on() or not self.beta:
            return output
        enc, h = output
        self.n_pushed += 1
        return enc, self.push(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

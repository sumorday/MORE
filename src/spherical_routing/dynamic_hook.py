"""Inference-time routing of a factored merge on FLUX (MMDiT): timestep schedule + gating by the
joint-attention maps of each subject's prompt tokens.

Per LoRA layer (spherical_svm.LayerFactors, w = A_r x):
  dh = gamma_sh(step) U_s C_s w + gamma_sp(step) [sum_{k active} m_k(token) B_sp (w o L[:, k])
                                                  + sum_{k passive} B_sp (w o L[:, k])]
Active subjects are the prompt's subjects (gated); passive subjects (optional, e.g. all others of
the merge) keep weight 1 on every token, so with passive = the rest the ungated run is the full
static merge.  With the sum-to-1 masks (softmax, softmax_bg) the active subjects' weights sum to 1
per token, i.e. gating also removes ~(K_act - 1)/K_act of their total part; softmax_max keeps the
dominant subject at 1 (uniform maps -> exactly ungated).
Schedule over the step index i of an N-step sampling run (s_split, ramp are fractions of N):
  i <  n_split = round(s_split N):  gamma_sp = 0           (common mode only: layout)
  i >= n_split:  gamma_sp = min(1, (i - n_split + 1) / max(1, round(ramp N)))
  gamma_sh = 1 throughout.
Token weights m_k (softmax / softmax_bg: each row sums to 1 over the active subjects;
softmax_max: each row's max is 1; with one active subject every token gets that subject's full part):
  image tokens   soft mask from the attention maps of the PREVIOUS step (the current step's maps
                 do not exist yet when the first layers run); step 0 or no maps -> uniform
                 (1/K, or 1 for softmax_max).
                 a_k(i) = sum over subject k's text tokens of softmax attention prob of image query
                 i, averaged over heads and the capture blocks; smoothed (3x3 box on the latent
                 grid), each map rescaled to mean 1, m_k = a_k^(1/T) / sum_j a_j^(1/T).
                 mask="softmax_bg" adds a background channel with a_bg = bg (constant, in the same
                 mean-1 units) whose weight is dropped, so low-attention tokens get no subject part.
                 mask="softmax_max" divides the softmax weights by their per-token max: the
                 dominant subject keeps its full part, the others are scaled by a_j^(1/T) / a_max^(1/T)
                 (uniform maps -> weight 1 for all, the ungated merge).
  text tokens    one-hot on subject k's own tokens ("sks <class>"), uniform over the active
                 subjects on every other token (1/K, or 1 for softmax_max).
  gating=False   weight 1 for every active subject on every token (static merge restricted to
                 the prompt's subjects; the ungated ablation).
  joint (single blocks) the [text | image] sequence is split at the text length.
Module stream from its name: single_transformer_blocks -> joint; add_*_proj / to_add_out /
ff_context -> text; other double-block modules -> image.

FLUX has no cross-attention: the maps are the image-query x text-key block of the joint
attention, captured by MapCaptureProcessor, a copy of diffusers' FluxAttnProcessor (same calls,
so the attention output is bitwise unchanged) that additionally forms the probabilities.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.transformers import transformer_flux as tf

from .spherical_svm import LayerFactors


def module_stream(name):
    if name.startswith("single_transformer_blocks"):
        return "joint"
    if any(t in name for t in (".add_q_proj", ".add_k_proj", ".add_v_proj", ".to_add_out",
                               "ff_context")):
        return "text"
    return "image"


class MapCaptureProcessor:
    """FluxAttnProcessor + subject attention maps (image queries -> subject text tokens)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, router):
        self.router = router

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        query, key, value, encoder_query, encoder_key, encoder_value = tf._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (-1, attn.head_dim))
        key = key.unflatten(-1, (-1, attn.head_dim))
        value = value.unflatten(-1, (-1, attn.head_dim))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (-1, attn.head_dim))
            encoder_key = encoder_key.unflatten(-1, (-1, attn.head_dim))
            encoder_value = encoder_value.unflatten(-1, (-1, attn.head_dim))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = tf.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = tf.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None and self.router.wants_maps():
            self.router.add_map(query, key, encoder_hidden_states.shape[1])

        hidden_states = tf.dispatch_attention_fn(query, key, value, attn_mask=attention_mask,
                                                 backend=self._attention_backend,
                                                 parallel_config=self._parallel_config)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


def subject_maps(query, key, txt_len, token_sets):
    """(B, K_act, N_img) float32: per subject, attention mass of each image query on its tokens,
    mean over heads.  query/key (B, S, H, d) after rotary, S = txt_len + N_img."""
    q = query[:, txt_len:].transpose(1, 2).float()             # (B, H, N, d)
    k = key.transpose(1, 2).float()                            # (B, H, S, d)
    p = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
    out = torch.stack([p[..., idx].sum(-1) for idx in token_sets], 1)   # (B, K, H, N)
    return out.mean(2)


class DynamicRouter:
    """Forward hooks on the LoRA layers + map capture + step counter (module docstring).

    factors: {module name: LayerFactors} (any device/dtype; moved to `device`, `dtype`).
    Per image: set_prompt(active, token_sets, n_steps) before the pipeline call.
    """

    def __init__(self, transformer, factors, s_split=0.0, ramp=0.0, temp=0.5, mask="softmax",
                 bg=1.0, smooth=3, capture_blocks=None, dtype=torch.bfloat16, device=None,
                 gamma_sh=1.0, gating=True):
        self.s_split, self.ramp, self.temp, self.mask, self.bg = s_split, ramp, temp, mask, bg
        self.smooth, self.gamma_sh_val, self.gating = smooth, gamma_sh, gating
        self.handles, self.mods, self.f = [], {}, {}
        self.saved_procs = {}
        self.step = 0
        self.n_steps = None
        self.active = None
        self.token_sets = None
        self._acc, self._n_acc = None, 0
        self.img_mask = None                                   # (B, N_img, K_act)
        self.txt_mask = None                                   # (1, txt_len, K_act)
        self.log = []
        named = dict(transformer.named_modules())
        missing = sorted(set(factors) - set(named))
        if missing:
            raise KeyError(f"{len(missing)} factor layers not in the model, e.g. {missing[:3]}")
        for name, f in factors.items():
            mod = named[name]
            dev = device or next(mod.parameters()).device
            self.f[name] = f.to(dev, dtype)
            self.mods[name] = mod
            self.handles.append(mod.register_forward_hook(self._make(name, module_stream(name))))
        self.handles.append(transformer.register_forward_pre_hook(self._pre, with_kwargs=True))
        blocks = transformer.transformer_blocks
        idx = range(len(blocks)) if capture_blocks is None else capture_blocks
        for i in idx:
            a = blocks[i].attn
            self.saved_procs[i] = a.processor
            a.set_processor(MapCaptureProcessor(self))
        self._blocks = blocks

    # ---- per image -------------------------------------------------------------------
    def set_prompt(self, active, token_sets, n_steps, txt_len=512, passive=()):
        """active: subject indices (columns of L); token_sets: list of LongTensor text-token
        indices, one per active subject; passive: subject indices with weight 1 everywhere."""
        assert len(active) == len(token_sets) and len(active) >= 1
        self.active = torch.as_tensor(active, dtype=torch.long)
        self.token_sets = [torch.as_tensor(t, dtype=torch.long) for t in token_sets]
        self.n_steps = n_steps
        self.step = -1                                         # first pre-hook -> 0
        self._acc, self._n_acc, self.img_mask = None, 0, None
        K = len(active)
        m = torch.full((1, txt_len, K), self._uniform(K))
        for j, t in enumerate(self.token_sets):
            m[0, t] = 0.0
            m[0, t, j] = 1.0
        self.txt_mask = m
        self._L_act = {n: f.L[:, self.active.to(f.L.device)] for n, f in self.f.items()}
        pas = sorted(set(int(i) for i in passive))
        assert not set(pas) & set(self.active.tolist()), "passive overlaps active"
        self._L_pas = {n: f.L[:, torch.as_tensor(pas, dtype=torch.long, device=f.L.device)].sum(1)
                       for n, f in self.f.items()} if pas else None
        self.log = []

    def _uniform(self, K):
        """weight of every active subject on a token without information"""
        return 1.0 if self.mask == "softmax_max" else 1.0 / K

    def gammas(self, step=None):
        i = self.step if step is None else step
        n_split = int(round(self.s_split * self.n_steps))
        if i < n_split:
            return self.gamma_sh_val, 0.0
        n_ramp = max(1, int(round(self.ramp * self.n_steps)))
        return self.gamma_sh_val, min(1.0, (i - n_split + 1) / n_ramp)

    def wants_maps(self):
        return self.gating and self.active is not None and len(self.active) > 1

    # ---- maps ------------------------------------------------------------------------
    def add_map(self, query, key, txt_len):
        with torch.no_grad():
            m = subject_maps(query, key, txt_len, [t.to(query.device) for t in self.token_sets])
        self._acc = m if self._acc is None else self._acc + m
        self._n_acc += 1

    def _finalize_maps(self):
        """Accumulated maps of the step that just ended -> image mask for the next step."""
        if self._acc is None:
            return
        a = self._acc / self._n_acc                            # (B, K, N)
        self._acc, self._n_acc = None, 0
        B, K, N = a.shape
        side = int(round(math.sqrt(N)))
        if side * side != N:
            raise ValueError(f"non-square latent grid ({N} tokens)")
        if self.smooth and self.smooth > 1:
            g = a.view(B * K, 1, side, side)
            g = F.avg_pool2d(g, self.smooth, stride=1, padding=self.smooth // 2,
                             count_include_pad=False)
            a = g.view(B, K, N)
        a = a / a.mean(-1, keepdim=True).clamp_min(1e-12)
        logits = a.clamp_min(1e-12).log() / self.temp
        if self.mask == "softmax_bg":
            bgl = torch.full_like(logits[:, :1], math.log(self.bg) / self.temp)
            w = torch.softmax(torch.cat([logits, bgl], 1), 1)[:, :K]
        elif self.mask == "softmax":
            w = torch.softmax(logits, 1)
        elif self.mask == "softmax_max":
            w = torch.softmax(logits, 1)
            w = w / w.max(1, keepdim=True).values
        else:
            raise ValueError(f"mask {self.mask!r}")
        self.img_mask = w.transpose(1, 2).contiguous()        # (B, N, K)
        self.log.append(dict(step=self.step, mask_mean=w.mean((0, 2)).tolist(),
                             mask_max_frac=(w.max(1).values > 0.9).float().mean().item(),
                             excl_frac=(w.min(1).values < 0.1 * w.max(1).values)
                             .float().mean().item()))

    def _pre(self, module, args, kwargs):
        self._finalize_maps()
        self.step += 1
        return None

    # ---- LoRA hooks ------------------------------------------------------------------
    def _mask_for(self, stream, x):
        """(B, S, K_act) token weights for the input x (B, S, in) of a layer of `stream`."""
        B, S, _ = x.shape
        K = len(self.active)
        if K == 1 or not self.gating:
            # ungated: every active subject's part on every token (the static merge restricted
            # to the prompt's subjects)
            return torch.ones(1, 1, K, device=x.device, dtype=x.dtype)
        if self.img_mask is None:
            im = torch.full((1, 1, K), self._uniform(K), device=x.device, dtype=x.dtype)
        else:
            im = self.img_mask.to(x.device, x.dtype)
        tm = self.txt_mask.to(x.device, x.dtype)
        if stream == "image":
            return im.expand(B, S, K) if im.shape[1] == 1 else im
        if stream == "text":
            if tm.shape[1] != S:
                raise ValueError(f"text stream has {S} tokens, mask {tm.shape[1]}")
            return tm
        T = tm.shape[1]
        if im.shape[1] == 1:
            im = im.expand(B, S - T, K)
        return torch.cat([tm.expand(B, T, K), im], 1)

    def _make(self, name, stream):
        def hook(mod, inputs, output):
            if self.active is None:
                return output
            f, L = self.f[name], self._L_act[name]
            x = inputs[0].to(f.A_r.dtype)
            g_sh, g_sp = self.gammas()
            w = x @ f.A_r.t()
            delta = None
            if f.U_s is not None and g_sh:
                delta = g_sh * ((w @ f.C_s.t()) @ f.U_s.t())
            if g_sp:
                m = self._mask_for(stream, x)                  # (B|1, S|1, K)
                lw = m @ L.t()                                 # (B|1, S|1, Kr)
                if self._L_pas is not None:
                    lw = lw + self._L_pas[name]
                sp = (w * lw) @ f.B_sp.t()
                delta = g_sp * sp if delta is None else delta + g_sp * sp
            if delta is None:
                return output
            return output + delta.to(output.dtype)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for i, p in self.saved_procs.items():
            self._blocks[i].attn.set_processor(p)
        self.saved_procs.clear()
        self.f.clear()
        self._L_act, self._L_pas = {}, None
        self._acc = self.img_mask = self.txt_mask = None

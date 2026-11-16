"""DreamBooth LoRA for FLUX.1-dev, matching SSR-Merge's protocol.

rank 32, lr 1e-4, 500 optimizer steps, batch 1, 512px, flow matching,
guidance 3.5, LoRA on all attention + FF projections of the DiT.

Writes loras/<subject>/ with a diffusers-layout adapter (transformer.* prefix).
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import common

TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]


def latent_ids(h, w, device):
    """img_ids for a (h, w) packed latent grid, shape (h*w, 3)."""
    ids = torch.zeros((h, w, 3), device=device)
    ids[..., 1] = torch.arange(h, device=device).view(-1, 1)
    ids[..., 2] = torch.arange(w, device=device).view(1, -1)
    return ids.reshape(-1, 3)


def pack_latents(z):
    b, c, h, w = z.shape
    z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return z.reshape(b, h // 2 * w // 2, c * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--independent_seed", action="store_true",
                    help="offset the seed by the subject's index so each LoRA gets "
                         "its own gaussian init. Without this every adapter starts "
                         "from the same global seed=42 draw, which leaves all "
                         "lora_A matrices at cos 0.996 with each other.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    seed = args.seed
    if args.independent_seed and args.subject in common.SUBJECTS:
        seed = args.seed + common.SUBJECTS.index(args.subject)
    torch.manual_seed(seed)
    print(f"[{args.subject}] seed {seed}")
    dev = args.device

    from diffusers import FluxTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    d = torch.load(os.path.join(common.DATA, f"{args.subject}.pt"), weights_only=False)
    latents, embeds, pooled, perm = d["latents"], d["embeds"], d["pooled"], d["perm"]
    n_img = latents.shape[0]
    print(f"[{args.subject}] {n_img} images, {len(d['prompts'])} prompts")

    model = FluxTransformer2DModel.from_pretrained(
        os.path.join(common.MODEL, "transformer"), torch_dtype=torch.bfloat16
    ).to(dev)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=TARGETS,
    )
    model = get_peft_model(model, lora_cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[{args.subject}] trainable {n_tr/1e6:.2f}M")

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        print("optimizer: AdamW8bit")
    except Exception as e:
        print("bitsandbytes unavailable, falling back to AdamW:", e)
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    H = W = 32  # packed grid: 64x64 latent -> 32x32 tokens
    img_ids = latent_ids(H, W, dev)
    txt_ids = torch.zeros(512, 3, device=dev)
    t0 = time.time()
    model.train()

    for step in range(args.steps):
        i = step % n_img
        x0 = latents[i: i + 1].to(dev, torch.bfloat16)
        n = torch.randn_like(x0)
        t = torch.rand(1, device=dev, dtype=torch.bfloat16)
        xt = (1 - t) * x0 + t * n
        target = n - x0

        e = embeds[perm[i]].to(dev, torch.bfloat16).unsqueeze(0)
        p = pooled[perm[i]].to(dev, torch.bfloat16).unsqueeze(0)

        pred = model(
            hidden_states=pack_latents(xt),
            timestep=t,
            encoder_hidden_states=e,
            pooled_projections=p,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=torch.full((1,), args.guidance, device=dev, dtype=torch.bfloat16),
            return_dict=False,
        )[0]
        loss = F.mse_loss(pred.float(), pack_latents(target).float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"[{args.subject}] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{el/(step+1):.2f}s/it eta {(args.steps-step-1)*el/(step+1)/60:.1f}min", flush=True)

    out = args.out or os.path.join(common.LORAS, args.subject)
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().to(torch.float32).cpu().contiguous()
          for k, v in model.state_dict().items() if "lora_" in k}
    sd = {k.replace(".default.", "."): v for k, v in sd.items()}
    sd = {f"transformer.{k}": v for k, v in sd.items()}
    save_file(sd, os.path.join(out, "pytorch_lora_weights.safetensors"))
    print(f"[{args.subject}] saved {len(sd)} tensors -> {out}")


if __name__ == "__main__":
    main()

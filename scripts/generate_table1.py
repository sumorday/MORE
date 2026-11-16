"""Table-1 protocol: single-subject prompts, one FLUX load for all arms.

For each arm the merged model is prompted with the target subject's own prompts
and scored against that subject's reference images (CLIP-I + DINOv2), which is
the RQ1 protocol of the SSR paper. The pipeline is built once and adapters are
swapped, because each load costs ~25 min on this cluster.

An arm whose safetensors file does not exist yet is waited for (`--wait_s`), so
this can be started while the merge sweep is still writing outputs.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common


def load_lora(pipe, path):
    """Inject an adapter whose keys may carry the peft `base_model.model` infix.

    diffusers 0.42 dispatches any state dict containing "base_model" to its
    fal-Kontext converter, which expects a different key layout and raises
    KeyError on a plain DreamBooth export.  Strip the infix first; every other
    key form (our merged files included) then goes down the normal path.
    """
    from safetensors.torch import load_file
    sd = {k.replace(".base_model.model.", "."): v for k, v in load_file(path).items()}
    pipe.load_lora_weights(sd, adapter_name="m")
    pipe.set_adapters(["m"], adapter_weights=[1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="stems under merged/ ; 'base' = no adapter")
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--prompt_offset", type=int, default=0,
                    help="use ENVS[offset : offset + n_prompts] (held-out prompts)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offload", action="store_true",
                    help="stream pipeline components from CPU RAM; needed when "
                         "other users hold most of every card")
    ap.add_argument("--embeds", default=os.path.join(common.DATA, "t1_embeds.pt"),
                    help="pre-encoded prompts from encode_prompts.py")
    ap.add_argument("--outroot", default=common.GEN)
    ap.add_argument("--loras_root", default=None,
                    help="directory holding <subject>/pytorch_lora_weights.safetensors, "
                         "used for oracle_* arms; defaults to common.LORAS")
    ap.add_argument("--wait_s", type=int, default=0)
    ap.add_argument("--per_subject_seed", action="store_true",
                    help="noise seed from the prompt's index within its subject (as an oracle_ "
                         "arm), so merged and own-LoRA images share initial noise")
    args = ap.parse_args()

    from diffusers import FluxPipeline

    emb = torch.load(args.embeds, weights_only=False)
    print(f"{len(emb)} pre-encoded prompts", flush=True)

    print("loading pipeline ...", flush=True)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(common.MODEL, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None)
    if args.offload:
        # A resident bf16 transformer is 23.8 GB, which no card has free while
        # other users' jobs are running.  Leaf-level offload keeps one block
        # (~420 MB) on the card at a time and costs ~30-60 s per image.
        pipe.enable_sequential_cpu_offload(device=args.device)
    else:
        pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline ready in {time.time()-t0:.0f}s", flush=True)

    for arm in args.arms:
        if arm.startswith("oracle_"):
            # single-LoRA oracle for one subject: upper bound of that row
            subjects = [arm[len("oracle_"):]]
            lora = os.path.join(args.loras_root or common.LORAS, subjects[0],
                                "pytorch_lora_weights.safetensors")
        else:
            K = int(arm.split("_K")[1]) if "_K" in arm else (0 if arm == "base" else None)
            subjects = common.SUBJECTS[:K] if K else common.SUBJECTS[:3]
            # base / base_K{K}: no adapter (base_K20 = base model on the 20-subject prompts)
            lora = (None if arm == "base" or arm.startswith("base_K")
                    else os.path.join(common.MERGED, f"{arm}.safetensors"))
        if lora is not None:
            deadline = time.time() + args.wait_s
            while not os.path.exists(lora) and time.time() < deadline:
                time.sleep(15)
            if not os.path.exists(lora):
                print(f"[{arm}] missing {lora}, skip", flush=True)
                continue
            if args.wait_s:     # file may still be being written: wait until its size is stable
                size = -1
                while os.path.getsize(lora) != size:
                    size = os.path.getsize(lora)
                    time.sleep(20)
            load_lora(pipe, lora)
        outdir = os.path.join(args.outroot, f"t1_{arm}")
        os.makedirs(outdir, exist_ok=True)
        meta_f = os.path.join(outdir, "meta.json")
        if os.path.exists(meta_f):
            print(f"[{arm}] done, skip", flush=True)
            if lora is not None:
                pipe.delete_adapters("m")
            continue

        prompts, meta, gidx = [], [], []
        for s in subjects:
            for q, p in enumerate(common.all_prompts(s)[args.prompt_offset:
                                                        args.prompt_offset + args.n_prompts]):
                prompts.append(p)
                meta.append({"subject": s, "prompt": p})
                gidx.append(q if args.per_subject_seed else len(gidx))

        for i, (p, m) in enumerate(zip(prompts, meta)):
            for sd in args.seeds:
                fn = os.path.join(outdir, f"{i:03d}_s{sd}.png")
                if os.path.exists(fn):
                    continue
                g = torch.Generator(device=args.device).manual_seed(sd * 1000 + gidx[i])
                E, P = emb[p]
                im = pipe(prompt_embeds=E.unsqueeze(0).to(args.device),
                          pooled_prompt_embeds=P.unsqueeze(0).to(args.device),
                          num_inference_steps=args.steps,
                          guidance_scale=args.guidance, height=args.res, width=args.res,
                          generator=g).images[0]
                im.save(fn)
            if i % 8 == 0:
                print(f"[{arm}] {i+1}/{len(prompts)}", flush=True)

        json.dump({"prompts": meta, "subjects": subjects, "seeds": args.seeds,
                   "per_subject_seed": args.per_subject_seed,
                   "steps": args.steps, "guidance": args.guidance}, open(meta_f, "w"), indent=1)
        print(f"[{arm}] {len(prompts)*len(args.seeds)} images", flush=True)

        if lora is not None:
            pipe.delete_adapters("m")

    print("done", args.outroot, flush=True)


if __name__ == "__main__":
    main()

"""Shared paths, subject list, prompt templates, and LoRA key normalization."""
import os
import re

# Repository root (data/, loras_seed2/, merged/, gen_*/, results/ live here); override with MORE_ROOT.
ROOT = os.environ.get("MORE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Local model snapshots (flux1-dev, dinov2-base, clip-vit-base-patch32); override with MORE_MODELS.
MODELS = os.environ.get("MORE_MODELS", os.path.join(ROOT, "models"))
MODEL = os.environ.get("FLUX_MODEL", os.path.join(MODELS, "flux1-dev"))
IMGS = os.path.join(ROOT, "data/dreambooth/imgs")
DATA = os.path.join(ROOT, "data/processed")
LORAS = os.path.join(ROOT, "loras")
MERGED = os.path.join(ROOT, "merged")
GEN = os.path.join(ROOT, "gen")
RESULTS = os.path.join(ROOT, "results")

SUBJECTS = [
    "backpack", "bear_plushie", "cat", "dog", "duck_toy",
    "monster_toy", "pink_sunglasses", "robot_toy", "colorful_sneaker", "vase",
    # K > 10 (added 2026-11-11, from google/dreambooth)
    "rc_car", "teapot", "wolf_plushie", "clock", "candle",
    "berry_bowl", "can", "fancy_boot", "grey_sloth_plushie", "poop_emoji",
]
# Every subject needs its own prompt: with one shared token, subjects of the same
# class get identical prompts and no merge can tell them apart.  Until 2026-11-11
# monster_toy and robot_toy used "toy" like duck_toy, so the seed_*_K10 arms
# (loras_seed, data/seed/stats_K10.pt) have three identical toy prompts; the
# K <= 5 sets never contained a second toy and are unaffected.
CLASSES = {
    "backpack": "backpack", "bear_plushie": "stuffed animal", "cat": "cat",
    "dog": "dog", "duck_toy": "toy", "monster_toy": "monster toy",
    "pink_sunglasses": "glasses", "robot_toy": "robot toy",
    "colorful_sneaker": "sneaker", "vase": "vase",
    "rc_car": "rc car", "teapot": "teapot", "wolf_plushie": "wolf plushie",
    "clock": "clock", "candle": "candle", "berry_bowl": "bowl", "can": "can",
    "fancy_boot": "boot", "grey_sloth_plushie": "sloth plushie",
    "poop_emoji": "poop emoji",
}
TOKEN = "sks"

ENVS = [
    "in the jungle", "in the snow", "on the beach", "on a cobblestone street",
    "on top of pink fabric", "on top of a wooden floor",
    "with a city in the background", "with a mountain in the background",
    "with a blue house in the background", "on top of a purple rug in a forest",
    "with a wheat field in the background",
    "with a tree and autumn leaves in the background",
    "with the Eiffel Tower in the background", "floating on top of water",
    "floating in an ocean of milk",
    "on top of green grass with sunflowers around it", "on top of a mirror",
    "on top of the sidewalk in a crowded street", "on top of a dirt road",
    "on top of a white rug", "red", "purple", "shiny", "wet", "cube shaped",
]


def prompt(subject: str, env: str) -> str:
    return f"a {TOKEN} {CLASSES[subject]} {env}"


def all_prompts(subject: str):
    return [prompt(subject, e) for e in ENVS]


def ref_images(subject: str):
    d = os.path.join(IMGS, subject)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts))


_LORA_KEY = re.compile(
    r"(?:^|\.)base_model\.model\.|(?:^|\.)model\.dit\.|(?:^|\.)dit\."
)
# diffusers saves adapters under a "transformer." (or "model.") key prefix
_TRUNK_KEY = re.compile(r"^(?:transformer|model|unet)\.")


def norm_layer_name(key: str) -> str:
    """Map any PEFT/diffusers LoRA key prefix to the bare module path."""
    k = _LORA_KEY.sub(".", key) if _LORA_KEY.search(key) else key
    k = k.lstrip(".")
    # Only strip once: a doubled prefix means the source layout was odd.
    k = _TRUNK_KEY.sub("", k, count=1)
    # peft appends adapter name: ...lora_A.default.weight / ...lora_A.weight
    return k


def extract_lora_pairs(state_dict, scale_by_alpha=True, alpha_key_suffix=".alpha"):
    """{module: (A, B)} from a flat LoRA state dict, any common key layout.

    A: (r, in), B: (out, r), and B is pre-multiplied by alpha/r so that the
    effective delta is exactly B @ A.
    """
    import torch

    a_keys = {}
    for k, v in state_dict.items():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_A.default.weight"):
            base = k[: k.rindex("lora_A")]
            a_keys[base.rstrip(".")] = (k, v)
    out = {}
    for base, (ak, A) in a_keys.items():
        bk = None
        for cand in (base + ".lora_B.weight", base + ".lora_B.default.weight", base + "lora_B.weight"):
            if cand in state_dict:
                bk = cand
                break
        if bk is None:
            continue
        B = state_dict[bk]
        if scale_by_alpha:
            r = A.shape[0]
            alpha = None
            for cand in (base + ".alpha", base + ".lora_A.alpha"):
                if cand in state_dict and torch.is_tensor(state_dict[cand]) and state_dict[cand].numel() == 1:
                    alpha = float(state_dict[cand].flatten()[0])
                    break
            scale = (alpha / r) if alpha is not None else 1.0
            B = B * scale
        out[norm_layer_name(base)] = (A.float(), B.float())
    return out


def emit_peft_state(pairs, alpha=None, prefix="transformer."):
    """Emit a diffusers/PEFT-compatible LoRA state dict (alpha = 1.0 by default).

    B already carries the effective scale, so alpha is set to rank (scale=1).
    """
    import torch

    out = {}
    for layer, (A, B) in pairs.items():
        r = A.shape[0]
        out[f"{prefix}{layer}.lora_A.weight"] = A.detach().to("cpu", torch.float32).contiguous()
        out[f"{prefix}{layer}.lora_B.weight"] = B.detach().to("cpu", torch.float32).contiguous()
        out[f"{prefix}{layer}.alpha"] = torch.tensor(r, dtype=torch.float32)
    return out

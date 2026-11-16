"""Mix calibration statistics files into one objective (no new forwards, hooks unchanged).

Per layer, file f with weight w_f enters with factor c_f = w_f / n_f (n_f its token
count at that layer), so the mixed objective is
  sum_f w_f * (mean over f's tokens of the squared output error),
i.e. w_f is the share of the objective carried by f's tokens regardless of how many
tokens f has.  Main use: data/seed2_calN (512-token padded text, ~97% of the text tokens
are T5 padding) + a --txt_trim file (text stream = the prompt's own tokens only) up-weights
the subject's identity tokens in the text-stream layers without touching the hooks.
Every field ssrlib.accumulate writes is mixed with the same c_f: G, Q, yy and per task
G, Qb, yy, n; mu as the matching weighted mean; n = sum_f w_f.  All solvers read only
G/n, Q/n (and per-task ratios), so the result is a valid stats file for --stats_root.

  python scripts/mix_stats.py --inputs data/seed2_calN data/seed2_calNT --weights 1 1 \
      --K 10 --out_dir data/seed2_calNmixT
"""
import argparse
import os

import torch


def mix_layer(layers, weights):
    """layers: per-file stats dicts of one layer (all with n > 0)."""
    c = [w / s["n"] for s, w in zip(layers, weights)]
    out = {"n": float(sum(weights))}
    for key in ("G", "Q"):
        out[key] = sum(ci * s[key].double() for ci, s in zip(c, layers)).float()
    out["yy"] = float(sum(ci * s["yy"] for ci, s in zip(c, layers)))
    out["mu"] = (sum(ci * s["n"] * s["mu"].double() for ci, s in zip(c, layers))
                 / out["n"]).float()
    if "interference" in layers[0]:           # depends on the LoRAs only
        out["interference"] = layers[0]["interference"]
    tasks = {}
    for k in layers[0]["tasks"]:
        ts = [s["tasks"][k] for s in layers]
        assert all((t["start"], t["end"]) == (ts[0]["start"], ts[0]["end"]) for t in ts), k
        n_k = sum(ci * t["n"] for ci, t in zip(c, ts))
        tasks[k] = {"G": sum(ci * t["G"].double() for ci, t in zip(c, ts)).float(),
                    "Qb": sum(ci * t["Qb"].double() for ci, t in zip(c, ts)).float(),
                    "mu": (sum(ci * t["n"] * t["mu"].double() for ci, t in zip(c, ts))
                           / n_k).float(),
                    "start": ts[0]["start"], "end": ts[0]["end"],
                    "yy": float(sum(ci * t["yy"] for ci, t in zip(c, ts))), "n": float(n_k)}
    out["tasks"] = tasks
    return out


def mix(stats_list, weights):
    names = list(stats_list[0])
    assert all(set(s) == set(names) for s in stats_list), "layer sets differ"
    out = {}
    for name in names:
        layers = [s[name] for s in stats_list]
        if any(l["n"] == 0 for l in layers):
            assert all(l["n"] == 0 for l in layers), name
            out[name] = layers[0]
            continue
        assert all(set(l["tasks"]) == set(layers[0]["tasks"]) for l in layers), name
        out[name] = mix_layer(layers, weights)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--K", nargs="+", type=int, default=[10])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    assert len(args.inputs) == len(args.weights) and all(w >= 0 for w in args.weights)
    os.makedirs(args.out_dir, exist_ok=True)
    for K in args.K:
        out = os.path.join(args.out_dir, f"stats_K{K}.pt")
        assert not os.path.exists(out), f"{out} exists"
        st = [torch.load(os.path.join(r, f"stats_K{K}.pt"), map_location="cpu",
                         weights_only=False) for r in args.inputs]
        torch.save(mix(st, args.weights), out)
        print(f"[K{K}] {list(zip(args.inputs, args.weights))} -> {out}", flush=True)


if __name__ == "__main__":
    main()

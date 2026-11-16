"""Emit router arms from saved calibration statistics -- no GPU, no model load.

merge_more.py --save_stats writes data/stats_K{K}.pt.  Every solver option
(ridge scale, centering, prior, shrink) is applied AFTER calibration, so a
sweep over them is pure CPU algebra and takes seconds per arm.

  python scripts/solve_from_stats.py --arm ssr_rel:tau=0.02 --arm ssr_rel:tau=0.1 --K 3
  python scripts/solve_from_stats.py --arm ta_rel:tau=0.05 --arm ta_relctr:tau=0.05 --K 3 5
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common
import ssrlib


def load_pairs(subjects, root=None):
    root = root or common.LORAS
    from safetensors.torch import load_file
    return [common.extract_lora_pairs(load_file(
        os.path.join(root, s, "pytorch_lora_weights.safetensors"))) for s in subjects]


def parse_arm(spec):
    """'ta_rel:tau=0.02:center=1' -> ('ta_rel', dict(prior='ta', ridge='rel',
    tau=0.02, center=True))."""
    parts = spec.split(":")
    name = parts[0]
    parts = parts[1:]
    if not parts:
        parts = [n for n in name.split("_") if n in ("rel", "ctr", "relctr", "ta", "gcv")]
    cfg = dict(prior="zero", ridge="abs", center=False)
    # Name tokens always describe the arm; explicit :opt=val override them.
    toks = []
    for n in name.split("_"):
        toks += ["rel", "ctr"] if n == "relctr" else [n]
    for p in toks + parts:
        if p == "rel":
            cfg["ridge"] = "rel"
        elif p == "relctr":
            cfg["ridge"] = "rel"
            cfg["center"] = True
        elif p == "gcv":
            cfg["ridge"] = "gcv"
        elif p in ("ctr", "center"):
            cfg["center"] = True
        elif p == "ta":
            cfg["prior"] = "ta"
        elif p == "avgp":
            cfg["prior"] = "avg"
        elif p.startswith("tau="):
            cfg["tau"] = float(p.split("=")[1])
        elif p.startswith("gain="):
            cfg["gain"] = float(p.split("=")[1])
        elif p.startswith("lam="):
            cfg["lambda_reg"] = float(p.split("=")[1])
        elif p.startswith("shrink="):
            cfg["shrink"] = float(p.split("=")[1])
        elif p.startswith("geq="):
            cfg["gain_equalize"] = float(p.split("=")[1])
        elif p.startswith("clip="):
            cfg["gain_clip"] = float(p.split("=")[1])
        elif p.startswith("gprof="):
            cfg["_gprof"] = p.split("=", 1)[1]
        elif p.startswith("guni="):
            cfg["_guni"] = float(p.split("=", 1)[1])
        elif p.startswith("cross="):
            cfg["cross"] = float(p.split("=")[1])
        elif p == "avg":
            pass  # identity baselines come from make_baseline.py
        elif p in ("ssr", "avgp", "base"):
            pass  # arm family name, carries no options
        else:
            raise SystemExit(f"unknown option {p!r} in {spec!r}")
    return name, cfg


def arm_stem(name, cfg):
    """Stable, self-describing output name; the grid classifier in
    make_table.py reads the leading token, so keep it first."""
    bits = [name]
    if "tau" in cfg:
        bits.append(f"tau{cfg['tau']:g}")
    if "lambda_reg" in cfg:
        bits.append(f"lam{cfg['lambda_reg']:g}")
    if "gain" in cfg:
        bits.append(f"gain{cfg['gain']:g}")
    if cfg.get("shrink"):
        bits.append(f"sh{cfg['shrink']:g}")
    if "gain_equalize" in cfg:
        bits.append(f"geq{cfg['gain_equalize']:g}")
    if "gain_clip" in cfg:
        bits.append(f"clip{cfg['gain_clip']:g}")
    if "_gprof" in cfg:
        bits.append(f"gprof{cfg['_gprof']}")
    if "_guni" in cfg:
        bits.append(f"guni{cfg['_guni']:g}")
    if "cross" in cfg:
        bits.append(f"cr{cfg['cross']:g}")
    return "_".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name[:opt=val...] repeated; opts: tau, lam, shrink, "
                         "center, rel/gcv/ta")
    ap.add_argument("--K", type=int, nargs="+", default=[3])
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--loras_root", default=None,
                    help="directory holding <subject>/pytorch_lora_weights.safetensors")
    ap.add_argument("--stats_root", default=None,
                    help="directory holding stats_K{K}.pt; defaults to common.DATA")
    ap.add_argument("--prefix", default="", help="prefix for output arm names")
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arm]

    for K in args.K:
        tag = f"K{K}"
        sp = os.path.join(args.stats_root or common.DATA, f"stats_{tag}.pt")
        if not os.path.exists(sp):
            print(f"[{tag}] no {sp}, skip", flush=True)
            continue
        stats = torch.load(sp, map_location="cpu", weights_only=False)
        subs = args.subjects[:K] if args.subjects else common.SUBJECTS[:K]
        comb = ssrlib.concat_pairs(load_pairs(subs, args.loras_root))
        print(f"[{tag}] {len(stats)} layers with stats, {len(comb)} LoRA layers",
              flush=True)

        for name, cfg in arms:
            cfg = dict(cfg)
            stem = args.prefix + arm_stem(name, cfg)
            out = os.path.join(common.MERGED, f"{stem}_{tag}.safetensors")
            if args.dry_run:
                print(f"  would write {out}  cfg={cfg}")
                continue
            # gain_profile / uniform-gain arms: resolve the directives into a
            # {layer: target_gain} dict before handing cfg to ssr_solve.
            gprof = cfg.pop("_gprof", None)
            guni = cfg.pop("_guni", None)
            if gprof or guni is not None:
                prof = torch.load(os.path.join(args.stats_root or common.DATA,
                                               f"gainprof_{gprof or 'uni'}_{tag}.pt"),
                                  weights_only=False) if gprof else None
                if guni is not None:
                    prof = {n: guni for n in stats}
                cfg["gain_profile"] = prof
            t0 = time.time()
            merged, diag = ssrlib.ssr_solve(stats, comb, **cfg)
            from safetensors.torch import save_file
            save_file(common.emit_peft_state(merged), out)
            rows = {k: {kk: (vv if isinstance(vv, (int, float, bool)) else str(vv))
                        for kk, vv in v.items()} for k, v in diag.items()}
            for r in rows.values():
                r.update({kk: vv for kk, vv in cfg.items()})
            json.dump(rows, open(out.replace(".safetensors", "_stats.json"), "w"),
                      indent=1)
            lams = [r["ridge_rel"] for r in rows.values()
                    if isinstance(r.get("ridge_rel"), (int, float))]
            lams.sort()
            med = lams[len(lams) // 2] if lams else float("nan")
            print(f"  {stem:28s} {time.time()-t0:5.1f}s  lam_rel med {med:.3e}",
                  flush=True)


if __name__ == "__main__":
    main()

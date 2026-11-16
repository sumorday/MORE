"""Paired pair-metric differences of selected arms against any reference arm (same prompts and
seeds; run_spherical_proto.pooled_pairs with a chosen ref).  CPU only.
With --clean: a second block with collage-aware metrics from the pair_image_checks json files
(results/rg*_imgchecks_K<K>.json): an image counts only if its collage probability is <= 0.5
(split-screen images give clean detector crops and inflate the DINO metrics):
  bothC = both and not collage, minC / meanC = min / mean DINO with collage images set to 0.

With --seeds: only images of these seeds (e.g. the held-out seeds 6-11 generated into
<arm>_s6-11 dirs, which pool with <arm> by name).

  python scripts/paired_vs.py --K 10 --ref rg_nm_st_f1_n0_p2_cos_K10 --grep _ii _sg _fo
"""
import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common
import run_spherical_proto as rsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--grep", nargs="*", default=[], help="keep arms containing any of these")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    pr = {}
    for fn in sorted(os.listdir(os.path.join(common.ROOT, "results"))):
        if fn.startswith(f"sph_pairs_K{args.K}"):
            pr.update(rsp._load(fn))
    if args.seeds:
        pr = {k: {**r, "per_image": [x for x in r["per_image"] if int(x[1]) in args.seeds]}
              for k, r in pr.items()}
        pr = {k: r for k, r in pr.items() if r["per_image"]}
    for ref in args.ref:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rsp.pooled_pairs(pr, args.K, ref=ref)
        for line in buf.getvalue().splitlines():
            if line.startswith("===") or not args.grep or any(g in line for g in args.grep) \
                    or ref in line:
                print(line)
    if args.clean:
        clean_block(pr, args)


def clean_block(pr, args):
    import json
    import numpy as np
    col = {}
    for fn in sorted(os.listdir(os.path.join(common.ROOT, "results"))):
        if fn.endswith(f"_imgchecks_K{args.K}.json"):
            for arm, r in json.load(open(os.path.join(common.ROOT, "results", fn))).items():
                col.setdefault(arm, {}).update({f: v[1] for f, v in r["per_image"].items()})
    groups = {}
    for name, r in pr.items():
        base = name.split(f"_K{args.K}")[0] + f"_K{args.K}"
        groups.setdefault(base, {}).update({(int(x[0]), int(x[1])): x for x in r["per_image"]})

    def metrics(arm, keys):
        m = rsp._pair_metrics([groups[arm][k] for k in keys])
        ok = np.asarray([col[arm][f"{i:03d}_s{sd}.png"] <= 0.5 for i, sd in keys], dtype=float)
        return dict(bothC=m["both"] * ok, minC=m["min"] * ok, meanC=m["mean"] * ok, col=1 - ok)

    rng = np.random.default_rng(0)
    for ref in args.ref:
        if ref not in col or ref not in groups:
            print(f"(no collage scores for {ref})")
            continue
        print(f"=== collage-aware, paired diff vs {ref}")
        for arm in sorted(groups):
            if arm not in col or (args.grep and arm != ref and not any(g in arm for g in args.grep)):
                continue
            keys = sorted(k for k in set(groups[arm]) & set(groups[ref])
                          if f"{k[0]:03d}_s{k[1]}.png" in col[arm]
                          and f"{k[0]:03d}_s{k[1]}.png" in col[ref])
            if len(keys) < 10:
                continue
            a, b = metrics(arm, keys), metrics(ref, keys)
            idx = rng.integers(0, len(keys), (2000, len(keys)))
            parts = [f"{f} {a[f].mean():.3f}" for f in ("col", "bothC", "minC", "meanC")]
            if arm != ref:
                parts += [f"d{f} {(a[f] - b[f]).mean():+.4f}±{(a[f] - b[f])[idx].mean(1).std():.4f}"
                          for f in ("bothC", "minC", "meanC")]
            print(f"  {arm:44s} n={len(keys):3d} " + " ".join(parts))


if __name__ == "__main__":
    main()

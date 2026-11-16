"""Unselected qualitative grids of the supplementary material (no cherry-picking): every prompt of
the benchmark at ONE fixed held-out seed (seed 6, files *_s6.png), ordered by prompt index.

  figB_pairs_K<K>_p<1|2>.pdf  pair benchmark, K = 10 / 20, prompts 1-10 / 11-20:
                              refs | FLUX.1-dev | LoRA sum | LoRA Switch | LoRA Composite |
                              SSR-Merge | RP-FLUX | LoRAShop | FreeFuse | MORE (ours)
                              (the arms of Table 1).  Red frame + tag = the image fails Table 1's
                              bothC: OWLv2 det < 0.2 for a subject, or CLIP collage prob. > 0.5
                              (make_paper_figs.failure, as in Fig. 4)
  figB_layouts.pdf            MORE only, K = 10: top-bottom, big-small, three subjects.  Red frame
                              + tag = fails Table 3's bothC / allC (a subject with det < 0.2, or a
                              collage, eval_layout.py); amber tag = all present but a box centre
                              outside its layout region (not counted by "hit")

Detections: results/id_confusion.json (det_rows of make_paper_figs) where it has the arm, otherwise
the per-image rows of results/sph_pairs_K<K>*.json that Table 1 is computed from (paired_vs.py
--clean; same OWLv2 assignment, identical to id_confusion.json up to rounding where both exist).
Caption drafts with the per-method counts: <OUT>/figB_captions.tex.
Collage scores: make_paper_figs.CHK (rg12_xb_imgchecks) completed in memory by the other
results/*_imgchecks_K<K>.json files for arms it lacks (as paired_vs.py --clean merges them).
The RGB images inside the saved PDFs are re-encoded as JPEG (quality 90, pikepdf) to keep the
supplementary PDF small (~33 MB -> ~7 MB for the five figures).

  python scripts/make_supp_grids.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_paper_figs as M  # noqa: E402  (module level: rcParams, IDC / CHK json loads only)
from make_paper_figs import (img, ref, meta, axes_in, off, frame, failure, det_rows,  # noqa: E402
                             collage, layout_rows, PAIRS, TRIPLES, F, RED, CHK, IDC)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT, OUT = M.ROOT, M.OUT
SEED = 6
AMBER = "#e69f00"
# (arm dir stem, column title): the rows of Table 1 (verified against results/*_paired_clean_K*.txt)
METHODS = [("rg_base_na_f0", "FLUX.1-dev\n(no LoRA)"), ("xb_sum", "LoRA sum"),
           ("xb_lsw", "LoRA Switch"), ("xb_lcp", "LoRA\nComposite"),
           ("rg_ssrm_st_f0", "SSR-Merge"), ("xb_rp_w0.5", "RP-FLUX"), ("xb_ls", "LoRAShop"),
           ("xb_ff", "FreeFuse"), (F, "MORE (ours)")]
LABEL = {"rg_base_na_f0": "FLUX.1-dev (no LoRA)", "xb_lcp": "LoRA Composite"}


def jpegify(path, q=90):
    """re-encode the RGB images of a saved PDF as JPEG in place (matplotlib embeds them losslessly)"""
    import io
    import pikepdf
    from PIL import Image  # noqa: F401  (needed by PdfImage.as_pil_image)
    pdf = pikepdf.open(path, allow_overwriting_input=True)
    for page in pdf.pages:
        for raw in page.images.values():
            im = pikepdf.PdfImage(raw)
            if im.colorspace != "/DeviceRGB" or raw.get("/SMask") is not None:
                continue
            b = io.BytesIO()
            im.as_pil_image().convert("RGB").save(b, "JPEG", quality=q)
            raw.write(b.getvalue(), filter=pikepdf.Name.DCTDecode)
            if "/DecodeParms" in raw:
                del raw["/DecodeParms"]
    pdf.save(path)


# ---------------------------------------------------------------- per-image data
_SPH = {}


def sph_rows(K):
    """{arm_K_dir: {file: (det list,)}} of results/sph_pairs_K<K>*.json (Table 1's source)"""
    if K not in _SPH:
        out = {}
        for fn in sorted(glob.glob(os.path.join(ROOT, "results", f"sph_pairs_K{K}*.json"))):
            d = json.load(open(fn))
            for arm, r in d.get("arms", d).items():
                out.setdefault(arm, {}).update(
                    {f"{int(x[0]):03d}_s{int(x[1])}.png": ([x[4], x[5]],) for x in r["per_image"]})
        _SPH[K] = out
    return _SPH[K]


def complete_chk(K):
    """add collage scores of arms missing in CHK[K] from the other imgchecks files (in memory)"""
    for fn in sorted(glob.glob(os.path.join(ROOT, "results", f"*_imgchecks_K{K}.json"))):
        for arm, r in json.load(open(fn)).items():
            if arm in CHK[K] and os.path.basename(fn) == f"rg12_xb_imgchecks_K{K}.json":
                continue
            e = CHK[K].setdefault(arm, {"per_image": {}})
            for f, v in r["per_image"].items():
                e["per_image"].setdefault(f, v)


SRC = {}


def dets_of(arm, K):
    d = det_rows(PAIRS, arm, K)
    if d:
        SRC[(arm, K)] = "id_confusion.json"
        return d
    SRC[(arm, K)] = "sph_pairs (Table 1 rows)"
    return sph_rows(K)[f"{arm}_K{K}_s6-11"]


def tag_box(ax, tag, fs, color=RED, top=False):
    ax.text(0.03, 0.97 if top else 0.03, tag, transform=ax.transAxes, color="w", fontsize=fs,
            va="top" if top else "bottom", linespacing=0.95,
            bbox=dict(fc=color, ec="none", pad=0.6))


def wrap_tag(tag, n=15):
    return tag.replace(" / ", " /\n") if len(tag) > n else tag


# ---------------------------------------------------------------- pairs
def fig_pairs(K, page, per_page=10):
    m = meta(PAIRS, f"{F}_K{K}_s6-11")
    idx = list(range(page * per_page, min(len(m), (page + 1) * per_page)))
    W, g, top, ph = 7.0, 0.04, 0.24, 0.105          # inches
    x0 = 0.0
    th = (W - x0 - 9 * g - 0.01) / 9.5
    H = top + len(idx) * (th + ph)
    fig = plt.figure(figsize=(W, H))
    flags = {}
    for ri, i in enumerate(idx):
        f = f"{i:03d}_s{SEED}.png"
        y = top + ri * (th + ph)
        subs = m[i]["subjects"]
        for k, s in enumerate(subs):
            ax = axes_in(fig, x0, y + k * th / 2, th / 2 - 0.012, th / 2 - 0.012)
            ax.imshow(ref(s))
            off(ax)
        for c, (arm, lab) in enumerate(METHODS):
            x = x0 + th / 2 + g + c * (th + g)
            ax = axes_in(fig, x, y, th, th)
            ax.imshow(img(PAIRS, f"{arm}_K{K}_s6-11", f))
            off(ax)
            tag = failure(arm, K, f, subs, dets_of(arm, K))
            flags[(arm, i)] = tag
            if tag:
                frame(ax, RED, 1.3)
                tag_box(ax, wrap_tag(tag), 4.6)
            if ri == 0:
                fig.text((x + th / 2) / W, 1 - (top - 0.025) / H, lab, ha="center", va="bottom",
                         fontsize=6.6, linespacing=0.95, fontweight="bold" if arm == F else None)
        fig.text(x0 / W, 1 - (y + th + 0.01) / H,
                 f"#{i + 1}  “" + m[i]["prompt"].replace("sks ", "") + "”",
                 fontsize=5.6, va="top", style="italic")
    fig.text((x0 + th / 4) / W, 1 - (top - 0.025) / H, f"refs.\n$K{{=}}{K}$", ha="center",
             va="bottom", fontsize=6.0, linespacing=0.95)
    fn = f"figB_pairs_K{K}_p{page + 1}.pdf"
    fig.savefig(os.path.join(OUT, fn), dpi=300)
    plt.close(fig)
    jpegify(os.path.join(OUT, fn))
    return fn, H, flags


# ---------------------------------------------------------------- layouts
BS_BOXES = ((0.25, 1.0, 0.0, 0.62), (0.1, 0.45, 0.68, 0.95))     # regional.BS_BOXES (y0 y1 x0 x1)


def fig_layouts():
    blocks = [("Top-bottom layout (first subject top)", PAIRS, F + "_lotb", "pairs", "tb"),
              ("Big-small layout (first subject in the large low-left box, second in the small "
               "top-right box)", PAIRS, F + "_lobs", "pairs", "bs"),
              ("Three subjects (left / middle / right strips)", TRIPLES, F, "triples", "tri")]
    W, ncol, g = 7.0, 10, 0.04
    th = (W - (ncol - 1) * g) / ncol
    bt, nh = 0.17, 0.20                                          # block title, names
    H = len(blocks) * (bt + 2 * (th + nh)) + 2 * 0.06
    fig = plt.figure(figsize=(W, H))
    y = 0.0
    counts = {}
    for title, root, arm, rname, kind in blocks:
        rows = layout_rows(rname, f"{arm}_K10")
        m = meta(root, f"{arm}_K10_s6-11")
        fig.text(0, 1 - (y + 0.02) / H, title, fontsize=7, va="top", fontweight="bold")
        y += bt
        fail = offr = 0
        for i in range(len(m)):
            r, c = divmod(i, ncol)
            x, yy = c * (th + g), y + r * (th + nh)
            f = f"{i:03d}_s{SEED}.png"
            ax = axes_in(fig, x, yy, th, th)
            ax.imshow(img(root, f"{arm}_K10_s6-11", f, size=256))
            off(ax)
            dash = dict(color="w", lw=0.35, ls=(0, (3, 3)), alpha=0.8)
            if kind == "bs":
                for (ya, yb, xa, xb) in BS_BOXES:
                    ax.add_patch(Rectangle((xa * 256 - 0.5, ya * 256 - 0.5), (xb - xa) * 256,
                                           (yb - ya) * 256, fill=False, ec="w", lw=0.35,
                                           ls=(0, (3, 3)), alpha=0.8))
            elif kind == "tb":
                ax.axhline(127.5, **dash)
            else:
                for v in (256 / 3, 512 / 3):
                    ax.axvline(v - 0.5, **dash)
            det, hit, _, col = rows[f]
            subs = m[i]["subjects"]
            if col:
                tag = "collage"
            else:
                miss = [M.name(s) for s, d in zip(subs, det) if d < 0.2]
                tag = ("no " + " / ".join(miss)) if miss else None
            if tag:
                fail += 1
                frame(ax, RED, 1.2)
                tag_box(ax, wrap_tag(tag, 13), 4.3)
            elif min(hit) <= 0.5:
                offr += 1
                tag_box(ax, "off region", 4.3, color=AMBER, top=False)
            fig.text((x + th / 2) / W, 1 - (yy + th + 0.012) / H,
                     "\n".join(M.name(s) for s in subs), ha="center", va="top", fontsize=4.6,
                     linespacing=0.95)
        counts[kind] = (fail, offr, len(m))
        y += 2 * (th + nh) + 0.06
    fig.savefig(os.path.join(OUT, "figB_layouts.pdf"), dpi=300)
    plt.close(fig)
    jpegify(os.path.join(OUT, "figB_layouts.pdf"))
    return H, counts


# ---------------------------------------------------------------- captions + counts
def table1_rate(arm, K):
    """bothC / col over all 120 held-out images (Table 1 criterion) from the same per-image data"""
    d = dets_of(arm, K)
    fs = [f"{i:03d}_s{s}.png" for i in range(20) for s in M.HO]
    bad = [failure(arm, K, f, meta(PAIRS, f"{F}_K{K}_s6-11")[int(f[:3])]["subjects"], d)
           is not None for f in fs]
    col = [collage(arm, K, f) for f in fs]
    return 1 - np.mean(bad), np.mean(col)


def main():
    os.makedirs(OUT, exist_ok=True)
    for K in (10, 20):
        complete_chk(K)
    lines, cnt = [], {}
    for K in (10, 20):
        fl = {}
        for p in (0, 1):
            fn, H, flags = fig_pairs(K, p)
            print(fn, f"height {H:.2f} in")
            fl.update(flags)
        for arm, lab in METHODS:
            tags = [fl[(arm, i)] for i in range(20)]
            n = sum(t is not None for t in tags)
            nc = sum(t == "collage" for t in tags)
            bc, col = table1_rate(arm, K)
            d, m = dets_of(arm, K), meta(PAIRS, f"{F}_K{K}_s6-11")
            per_seed = [sum(failure(arm, K, f"{i:03d}_s{sd}.png", m[i]["subjects"], d) is None
                            for i in range(20)) for sd in M.HO]
            cnt[(arm, K)] = (n, nc, bc, col, per_seed)
            print(f"K{K} {arm:45s} flagged {n:2d}/20 (collage {nc}) seed-6 bothC {1 - n / 20:.2f}"
                  f" | 120-img bothC {bc:.3f} col {col:.3f} | pass/seed {per_seed} | {SRC[(arm, K)]}")
    H, lc = fig_layouts()
    print("figB_layouts.pdf", f"height {H:.2f} in", lc)
    write_captions(cnt, lc)


def write_captions(cnt, lc):

    def lab(a):
        return LABEL.get(a, dict(METHODS)[a].replace("\n", " "))
    rows = []
    for arm, _ in METHODS:
        a, b = cnt[(arm, 10)], cnt[(arm, 20)]
        rows.append(f"{lab(arm)} & {a[0]} ({a[1]}) & {1 - a[0] / 20:.2f} & {a[2]:.3f} & "
                    f"{b[0]} ({b[1]}) & {1 - b[0] / 20:.2f} & {b[2]:.3f} \\\\")
    common_txt = (
        "All 20 prompts of the pair benchmark at one fixed held-out seed (seed 6, the first "
        "held-out seed), ordered by prompt index; no image, prompt or seed was selected. "
        "Every method uses the same prompt and the same seed (seed index 6, \\cref{sec:setup}). "
        "Columns as in \\cref{tab:main}: FLUX.1-dev without LoRA, LoRA sum, LoRA Switch, LoRA "
        "Composite, SSR-Merge (published calibration, whole pool), RP-FLUX (LoRA weight 0.5, "
        "oracle layout), LoRAShop, FreeFuse (both automatic masks), and MORE (whole pool, oracle "
        "left/right layout). A \\textcolor{red}{red frame} marks an image that fails the "
        "automatic coexistence test of \\bothC{}: a subject not detected by OWLv2 "
        "(score ${<}0.2$; tag ``no $\\langle$class$\\rangle$'') or a CLIP-flagged collage "
        "(tag ``collage''). The frames are produced by the metric, not by us; they are not a "
        "judgement of identity, which \\bothC{} does not measure (an unframed image may still "
        "show a wrong identity or a hybrid).")
    def rate_txt(K):
        parts = [f"{lab(a)} {cnt[(a, K)][0]}" for a, _ in METHODS]

        def rng(a):
            p = cnt[(a, K)][4]
            return f"{min(p)}--{max(p)}"
        return (f" Flagged images at this seed (of 20): {', '.join(parts)}. For reference, over"
                f" the six held-out seeds 6--11 MORE passes {rng(F)} of 20 per seed (seed 6:"
                f" {20 - cnt[(F, K)][0]}; \\bothC{{}} {cnt[(F, K)][2]:.3f} in \\cref{{tab:main}})"
                f" and FreeFuse {rng('xb_ff')} (seed 6: {20 - cnt[('xb_ff', K)][0]};"
                f" {cnt[('xb_ff', K)][2]:.3f}).")
    tex = [
        "% figB captions, generated by cvpr2027/scripts/make_supp_grids.py",
        "",
        "% ---- figB_pairs_K10_p1 / p2",
        "\\caption{\\textbf{Unselected comparison, $K{=}10$ pool} (prompts 1--10 / 11--20). "
        + common_txt + rate_txt(10) + "}",
        "\\label{fig:supp_pairs_K10}",
        "",
        "% ---- figB_pairs_K20_p1 / p2",
        "\\caption{\\textbf{Unselected comparison, $K{=}20$ pool} (prompts 1--10 / 11--20). "
        + common_txt.replace("first held-out seed)", "first held-out seed; prompts and subject "
                             "pairs of the $K{=}20$ benchmark)") + rate_txt(20) + "}",
        "\\label{fig:supp_pairs_K20}",
        "",
        "% ---- figB_layouts",
        "\\caption{\\textbf{Unselected layout and three-subject results of MORE} ($K{=}10$ pool, "
        "held-out seed 6, all 20 prompts per setting in prompt order, no selection). Top: "
        "top-bottom layout; middle: big-small layout (large low-left box, small top-right box); "
        "bottom: three subjects in left/middle/right strips. Dashed lines: the layout regions; "
        "the subject names under each image are listed in region order. A \\textcolor{red}{red "
        "frame} marks a failure of the criterion of \\cref{tab:general} (\\bothC{} / \\allC{}): "
        "a subject not detected by OWLv2 (score ${<}0.2$) or a CLIP-flagged collage. An "
        "\\textcolor{orange}{amber tag} marks an image in which all subjects are detected but at "
        "least one box centre lies outside its region (counted by \\bothC{} but not by ``hit''). "
        f"Failures: {lc['tb'][0]}/20 top-bottom, {lc['bs'][0]}/20 big-small, {lc['tri'][0]}/20 "
        f"three subjects; off-region: {lc['tb'][1]}, {lc['bs'][1]}, {lc['tri'][1]}.}}",
        "\\label{fig:supp_layouts}",
        "",
        "% ---- flagged counts at seed 6 (n=20) vs Table 1 bothC (n=120, seeds 6-11)",
        "% columns: method & K10 flagged (of which collage) & seed-6 bothC & Table-1 bothC &"
        " K20 flagged (collage) & seed-6 bothC & Table-1 bothC",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "& \\multicolumn{3}{c}{$K{=}10$} & \\multicolumn{3}{c}{$K{=}20$} \\\\",
        "Method & flagged & \\bothC{} (s6) & \\bothC{} (Tab.~1) & flagged & \\bothC{} (s6) & "
        "\\bothC{} (Tab.~1) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "% layouts (MORE, K=10, seed 6): failures / off-region of 20: "
        + ", ".join(f"{k} {v[0]} / {v[1]}" for k, v in lc.items())
        + "  (Table 3: tb bothC .733 hit .642, bs .475 / .117, triples allC .692 hit .650)",
    ]
    open(os.path.join(OUT, "figB_captions.tex"), "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()

"""All raster figures of the CVPR 2027 paper, built from the generated images and the result files
of this repository (no manual editing).  Fig. 2 (framework) is TikZ inside the paper.

  fig1_teaser.pdf      references (stacked) + ours + exact prompt, one column per setting (K=10 /
                       K=20 pairs, top-bottom layout, 3 subjects); subject colors link frame and words
  fig3_mechanism.pdf   subject attention maps (double block 18) of the static merge / routed (no
                       swap guidance) / full pipeline + region-feature cosine per step
  fig4_qualitative.pdf refs | LoRA sum | RP-FLUX | LoRAShop | FreeFuse | ours, same prompt + seed;
                       red frame = OWLv2 misses a subject (det < 0.2) or CLIP flags a collage
  fig5_ablation.pdf    avg -> SSR-Merge (our calib.) -> our merge -> + routing -> + ii1 + sg2,
                       3 subjects (static / no-sg / full), tb and big-small layouts
  figA_userstudy_<p>.pdf  the 50 blind trials with the author's answers (appendix)

Selection: rows of Fig. 1 / 4 / 5 are picked automatically from the held-out seeds (6-11) by our
detector scores (results/id_confusion.json) so that every prompt pair appears at most once; the
baselines are shown for the same prompt and seed without any baseline-specific selection.

  python scripts/make_paper_figs.py [1 3 4 5 A]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MORE_ROOT", os.path.abspath(os.path.join(HERE, "..")))
OUT = os.path.join(HERE, "..", "figs")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import common  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
                     "pdf.fonttype": 42, "axes.linewidth": 0.5})
PAIRS, TRIPLES = os.path.join(ROOT, "gen_sph_pairs"), os.path.join(ROOT, "gen_sph_triples")
F = "rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb"
NOSG = "rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1"
HO = [6, 7, 8, 9, 10, 11]
TILE = 320                                                    # px per image in the pdf
RED, GREEN = "#d62728", "#2ca02c"

IDC = json.load(open(os.path.join(ROOT, "results", "id_confusion.json")))
CHK = {K: json.load(open(os.path.join(ROOT, "results", f"rg12_xb_imgchecks_K{K}.json")))
       for K in (10, 20)}
META = {}
USED = {}                                       # subject sets already shown, per (root, K)


def meta(root, arm_dir):
    k = (root, arm_dir)
    if k not in META:
        META[k] = json.load(open(os.path.join(root, arm_dir, "meta.json")))["prompts"]
    return META[k]


def img(root, arm_dir, f, size=TILE):
    return np.asarray(Image.open(os.path.join(root, arm_dir, f)).convert("RGB")
                      .resize((size, size), Image.LANCZOS))


def ref(s, size=TILE // 2):
    im = Image.open(common.ref_images(s)[0]).convert("RGB")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return np.asarray(im.resize((size, size), Image.LANCZOS))


def det_rows(root, arm, K, seeds=HO):
    """{file: (det list, boxes)} of results/id_confusion.json"""
    key = f"{os.path.basename(root)}:{arm}_K{K}:{','.join(map(str, seeds))}"
    if key not in IDC:
        return {}
    return {f"{r[0]:03d}_s{r[1]}.png": (r[8], r[7], r[2]) for r in IDC[key]["rows"]}


def collage(arm, K, f):
    v = CHK[K].get(f"{arm}_K{K}", {}).get("per_image", {}).get(f)
    return v is not None and v[1] > 0.5


def axes_in(fig, x, y, w, h):
    """axes at (x, y) inches from the top-left corner, size w x h inches"""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def name(s):
    return common.CLASSES[s]


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def frame(ax, color, lw=2.0):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(color)
        s.set_linewidth(lw)


def failure(arm, K, f, subs, dets):
    """short tag of the failure of an image or None"""
    if collage(arm, K, f):
        return "collage"
    d = dets.get(f)
    if d is None:
        return None
    miss = [name(s) for s, x in zip(subs, d[0]) if x < 0.2]
    return ("no " + " / ".join(miss)) if miss else None


def layout_rows(name_, arm):
    """{file: (det list, hit list, dino list, collage)} of eval_layout.py (tb / bs / triples)"""
    d = json.load(open(os.path.join(ROOT, "results", f"layout_{name_}_K10.json")))["arms"][arm]
    K, out = d["K"], {}
    grp = {"lr": 0, "bs": 0, "tb": 1}[d["layout"]]
    for r in d["per_image"]:
        col = r[2 + 3 * K:]
        c = max(col[0], col[2]) if K == 3 else col[grp]
        out[f"{int(r[0]):03d}_s{int(r[1])}.png"] = (r[2 + K:2 + 2 * K], r[2 + 2 * K:2 + 3 * K],
                                                    r[2:2 + K], c > 0.5)
    return out


def pick(root, K, n, arm=F, skip=0, used=None):
    """files of our arm (held-out seeds) with every subject detected (det >= 0.3, inside its
    layout region for tb / bs / triples) and no collage, ranked by min det x mean DINO; one image
    per subject set (sets in `used` are skipped and added); the ranks skip .. skip + n - 1"""
    if root == TRIPLES or "_lo" in arm:
        rows = layout_rows("triples" if root == TRIPLES else "pairs", f"{arm}_K{K}")
        score = {f: min(d) * np.mean(s) for f, (d, h, s, c) in rows.items()
                 if min(d) >= 0.3 and min(h) > 0 and not c}
    else:
        rows = det_rows(root, arm, K)
        score = {f: min(d) * own for f, (d, b, own) in rows.items()
                 if min(d) >= 0.3 and not collage(arm, K, f)}
    m = meta(root, f"{arm}_K{K}_s6-11")
    used = set() if used is None else used
    out = []
    for f in sorted(score, key=lambda f: -score[f]):
        subs = tuple(sorted(m[int(f[:3])]["subjects"]))
        if subs in used:
            continue
        used.add(subs)
        out.append(f)
        if len(out) == skip + n:
            break
    return out[skip:]


def refs_strip(ax, subs):
    ax.imshow(np.concatenate([ref(s) for s in subs], 1))
    off(ax)


# ---------------------------------------------------------------- Fig. 1
SUBC = ["#2b6cb0", "#c05621", "#6b46c1"]          # subject colors: reference frame = prompt words


def color_prompt(fig, x, y, w, prompt, subs, fs=6.3, lh=0.098):
    """`prompt` centered in [x, x + w] from y (inches) down, wrapped to w; the words "sks <class>"
    of subject k in SUBC[k]"""
    W, H = fig.get_size_inches()
    words, col = prompt.split(), []
    for t in words:
        col.append(None)
    for k, s in enumerate(subs):                       # mark "sks <class words>"
        cls = name(s).split()
        for a in range(len(words)):
            if words[a] == "sks" and [v.strip(",") for v in words[a + 1:a + 1 + len(cls)]] == cls \
                    and col[a] is None:
                for b in range(a, a + 1 + len(cls)):
                    col[b] = SUBC[k]
                break
    dpi = fig.dpi
    fig.set_dpi(1200)                                  # fine text metrics
    r = fig.canvas.get_renderer()

    def width(t):
        tt = fig.text(0, 0, t, fontsize=fs, style="italic")
        e = tt.get_window_extent(r).width / fig.dpi
        tt.remove()
        return e
    sp = width("a a") - width("aa")
    lines, cur, cw = [], [], 0.0
    for t, c in zip(words, col):
        tw = width(t)
        if cur and cw + sp + tw > w:
            lines.append((cur, cw))
            cur, cw = [], 0.0
        cw += (sp if cur else 0) + tw
        cur.append((t, c, tw))
    lines.append((cur, cw))
    fig.set_dpi(dpi)
    for n, (line, lw) in enumerate(lines):
        xx = x + (w - lw) / 2
        for t, c, tw in line:
            fig.text(xx / W, 1 - (y + n * lh) / H, t, fontsize=fs, style="italic", va="top",
                     color=c or "black")
            xx += tw + sp
    return len(lines)


def fig1():
    """columns = settings (K=10 / K=20 left-right, K=10 top-bottom, three subjects), two held-out
    images each: references stacked at the left of our image, the exact prompt below; each
    subject's reference frame and prompt words share a color"""
    picks = []
    for K, root, arm, lab, n in ((10, PAIRS, F, "$K{=}10$ pool, left-right", 2),
                                 (20, PAIRS, F, "$K{=}20$ pool, left-right", 2),
                                 (10, PAIRS, F + "_lotb", "$K{=}10$ pool, top-bottom", 2),
                                 (10, TRIPLES, F, "$K{=}10$ pool, three subjects", 2)):
        d = f"{arm}_K{K}_s6-11"
        picks.append([(root, d, f, lab)
                      for f in pick(root, K, n, arm=arm, used=USED.setdefault((root, K), set()))])
    W, th, g, hd, tx = 7.0, 0.95, 0.03, 0.15, 0.25   # inches: width, image, gap, header, prompt
    pw = th * 1.5 + g
    gap = (W - 4 * pw) / 3
    H = hd + 2 * (th + tx)
    fig = plt.figure(figsize=(W, H))
    for c, col in enumerate(picks):
        x = c * (pw + gap)
        fig.text((x + pw / 2) / W, 1 - 0.01 / H, col[0][3], ha="center", va="top", fontsize=7)
        for r, (root, d, f, lab) in enumerate(col):
            y = hd + r * (th + tx)
            m = meta(root, d)[int(f[:3])]
            n = len(m["subjects"])
            sz = th / max(2, n)
            for k, s in enumerate(m["subjects"]):            # references, stacked at the left
                ax = axes_in(fig, x + (th / 2 - sz) / 2, y + (th - n * sz) / 2 + k * sz + g / 2,
                             sz - g, sz - g)
                ax.imshow(ref(s))
                off(ax)
                frame(ax, SUBC[k], 1.2)
            ax = axes_in(fig, x + th / 2 + g, y, th, th)
            ax.imshow(img(root, d, f))
            off(ax)
            color_prompt(fig, x, y + th + 0.03, pw, m["prompt"], m["subjects"])
    fig.savefig(os.path.join(OUT, "fig1_teaser.pdf"), dpi=300)
    plt.close(fig)
    return [c[2] for col in picks for c in col]


# ---------------------------------------------------------------- Fig. 3
def fig3(i=2, steps=(1, 6, 20)):
    r = torch.load(os.path.join(ROOT, "results", "mech_K10.pt"), weights_only=False)
    arms = [("rg_nm_st_f0_K10", "Static merge"), (NOSG + "_K10", "Routed, no SG"),
            (F + "_K10", "MORE (full)")]
    j = [k for k, (ii, sd) in enumerate(r[arms[0][0]]["keys"]) if ii == i][0]
    subs = meta(PAIRS, F + "_K10_s6-11")[i]["subjects"]
    W, H, lw_, t_, g = 3.3, 3.0, 0.2, 0.41, 0.025              # inches
    fig = plt.figure(figsize=(W, H))
    y0 = 0.27
    for a, (arm, lab) in enumerate(arms):
        y = y0 + a * (t_ + 0.03)
        ax = axes_in(fig, lw_, y, t_, t_)
        ax.imshow(np.asarray(Image.open(os.path.join(ROOT, "results", "mech_K10_img", arm,
                                                     f"{i:03d}_s6.png")).convert("RGB")
                             .resize((256, 256))))
        off(ax)
        fig.text(0.6 * lw_ / W, 1 - (y + t_ / 2) / H, lab.replace(", ", ",\n"), rotation=90,
                 ha="center", va="center", fontsize=6)
        M = r[arm]["maps"][j].float()                             # (T, 2, N)
        for si, t in enumerate(steps):
            for k in range(2):
                x = lw_ + t_ + g + (2 * si + k) * t_ + si * g
                ax = axes_in(fig, x + 0.01, y, t_ - 0.01, t_)
                m = M[t, k].view(32, 32).numpy()
                ax.imshow(m / (m.max() + 1e-8), cmap="magma", vmin=0, vmax=1)
                off(ax)
                ax.axvline(15.5, color="w", lw=0.4, ls=":")
                if a == 0:
                    fig.text((x + t_ / 2) / W, 1 - (y - 0.015) / H, name(subs[k]),
                             ha="center", va="bottom", fontsize=5.5)
                    if k == 0:
                        fig.text((x + t_) / W, 1 - (y - 0.12) / H, f"step $t{{=}}{t}$",
                                 ha="center", va="bottom", fontsize=6)
    fig.text((lw_ + t_ / 2) / W, 1 - (y0 - 0.015) / H, "image", ha="center", va="bottom",
             fontsize=5.5)
    ax = axes_in(fig, 0.42, y0 + 3 * (t_ + 0.03) + 0.12, W - 0.47, 1.0)
    T = np.arange(28)
    for (arm, lab), c in zip(arms, ["#7f7f7f", "#ff7f0e", "#1f77b4"]):
        fs = r[arm]["fsim"].numpy()
        mu, se = fs.mean(0), fs.std(0) / np.sqrt(fs.shape[0])
        ax.plot(T, mu, color=c, lw=1, label=f"{lab} ({fs[:, :8].mean():.3f})")
        ax.fill_between(T, mu - se, mu + se, color=c, alpha=0.2, lw=0)
    ax.axvspan(-0.5, 7.5, color="#eeeeee", zorder=0)
    for t in (0, 2, 4, 6, 8):
        ax.axvline(t, color="#1f77b4", lw=0.3, ls=":")
    ax.text(3.5, 0.9615, "layout gating", ha="center", fontsize=5.5)
    ax.set_xlim(-0.5, 27.5)
    ax.set_xlabel("denoising step", labelpad=1)
    ax.set_ylabel("cos(region A, region B)", fontsize=6)
    ax.legend(fontsize=5.2, frameon=False, loc="lower right", ncol=1)
    ax.tick_params(labelsize=5.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig3_mechanism.pdf"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
METHODS = [("xb_sum", "LoRA sum"), ("xb_rp_w0.5", "RP-FLUX"), ("xb_ls", "LoRAShop"),
           ("xb_ff", "FreeFuse"), (F, "MORE (ours)")]


def fig4(n10=2, n20=2):
    rows = [(10, f) for f in pick(PAIRS, 10, n10, used=USED.setdefault((PAIRS, 10), set()))] + \
        [(20, f) for f in pick(PAIRS, 20, n20, used=USED.setdefault((PAIRS, 20), set()))]
    th, g, top, ph = 0.86, 0.03, 0.17, 0.12                   # inches: tile, gap, title, prompt
    W = 0.16 + th / 2 + len(METHODS) * (th + g)
    fig = plt.figure(figsize=(W, top + len(rows) * (th + ph)))
    for ri, (K, f) in enumerate(rows):
        y = top + ri * (th + ph)
        m = meta(PAIRS, F + f"_K{K}_s6-11")[int(f[:3])]
        subs = m["subjects"]
        fig.text(0.07 / W, 1 - (y + th / 2) / fig.get_size_inches()[1], f"$K{{=}}{K}$",
                 rotation=90, ha="center", va="center", fontsize=7)
        for k, s in enumerate(subs):
            ax = axes_in(fig, 0.15, y + k * th / 2, th / 2 - 0.015, th / 2 - 0.015)
            ax.imshow(ref(s))
            off(ax)
        for c, (arm, lab) in enumerate(METHODS):
            ax = axes_in(fig, 0.15 + th / 2 + g + c * (th + g), y, th, th)
            ax.imshow(img(PAIRS, f"{arm}_K{K}_s6-11", f))
            off(ax)
            tag = failure(arm, K, f, subs, det_rows(PAIRS, arm, K))
            if tag:
                frame(ax, RED, 2.0)
                ax.text(0.03, 0.03, tag, transform=ax.transAxes, color="w", fontsize=5.5,
                        va="bottom", bbox=dict(fc=RED, ec="none", pad=1.0))
            if ri == 0:
                ax.set_title(lab, fontsize=7.5, pad=2, fontweight="bold" if arm == F else None)
        fig.text(0.15 / W, 1 - (y + th + 0.012) / fig.get_size_inches()[1],
                 "\u201c" + m["prompt"].replace("sks ", "") + "\u201d", fontsize=5.8, va="top",
                 style="italic")
    fig.savefig(os.path.join(OUT, "fig4_qualitative.pdf"), dpi=300)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- Fig. 5
def fig5():
    prog = [("rg_avg_st_f0", "Avg. merge"), ("rg_ssrmc_st_f0", "SSR-Merge$^\\dagger$"),
            ("sph_rc_nm_s0_up", "Our merge"), ("rg_nm_lg_f1_n0_p2_cos", "+ routing"),
            (F, "+ ii1 + sg2 (full)")]
    rows = [(K, pick(PAIRS, K, 1, used=USED.setdefault((PAIRS, K), set()))[0]) for K in (10, 20)]
    tri = pick(TRIPLES, 10, 1, used=USED.setdefault((TRIPLES, 10), set()))[0]
    tri_arms = [("rg_nm_st_f0", "static"), (NOSG, "no SG"), (F, "full")]
    lo = []
    for tag, boxes in (("lotb", ((0, .5, 0, 1), (.5, 1, 0, 1))),
                       ("lobs", ((0.25, 1.0, 0.0, 0.62), (0.1, 0.45, 0.68, 0.95)))):
        d = f"{F}_{tag}_K10_s6-11"
        lo.append((d, boxes, "top-bottom" if tag == "lotb" else "big-small"))
    f_tb = pick(PAIRS, 10, 1, arm=F + "_lotb", used=USED.setdefault((PAIRS, 10), set()))[0]
    W, H, th, g, top = 7.0, 1.95, 0.8, 0.03, 0.16              # inches
    fig = plt.figure(figsize=(W, H))

    def ref_col(x, y, subs, h):
        sz = h / max(2, len(subs))
        for k, s in enumerate(subs):
            ax = axes_in(fig, x, y + (h - len(subs) * sz) / 2 + k * sz, sz - 0.012, sz - 0.012)
            ax.imshow(ref(s, 128))
            off(ax)
        return sz

    for ri, (K, f) in enumerate(rows):
        y = top + ri * (th + 0.04)
        subs = meta(PAIRS, F + f"_K{K}_s6-11")[int(f[:3])]["subjects"]
        fig.text(0.06 / W, 1 - (y + th / 2) / H, f"$K{{=}}{K}$", rotation=90, ha="center",
                 va="center", fontsize=7)
        ref_col(0.13, y, subs, th)
        for c, (arm, lab) in enumerate(prog):
            ax = axes_in(fig, 0.13 + th / 2 + g + c * (th + g), y, th, th)
            ax.imshow(img(PAIRS, f"{arm}_K{K}_s6-11", f))
            off(ax)
            if ri == 0:
                ax.set_title(lab, fontsize=7, pad=2,
                             fontweight="bold" if arm == F else None)
    x0 = 0.13 + th / 2 + g + len(prog) * (th + g) + 0.08
    subs = meta(TRIPLES, F + "_K10_s6-11")[int(tri[:3])]["subjects"]
    t3 = (W - x0 - 0.2 - 3 * g) / 3
    ref_col(x0, top, subs, t3 * 1.0)
    for c, (arm, lab) in enumerate(tri_arms):
        ax = axes_in(fig, x0 + 0.2 + g + c * (t3 + g), top, t3, t3)
        ax.imshow(img(TRIPLES, f"{arm}_K10_s6-11", tri))
        off(ax)
        ax.set_title("3 subj., " + lab, fontsize=6.3, pad=2,
                     fontweight="bold" if arm == F else None)
    y2 = top + t3 + 0.17
    t2 = min(H - y2 - 0.01, (W - x0 - 2 * g) / 2.5)
    subs = meta(PAIRS, F + "_lotb_K10_s6-11")[int(f_tb[:3])]["subjects"]
    ref_col(x0, y2, subs, t2)
    for c, (d, boxes, lab) in enumerate(lo):
        ax = axes_in(fig, x0 + t2 / 2 + g + c * (t2 + g), y2, t2, t2)
        ax.imshow(img(PAIRS, d, f_tb))
        off(ax)
        for (ya, yb, xa, xb), col in zip(boxes, ["#ffffff", "#ffdd00"]):
            ax.add_patch(Rectangle((xa * TILE, ya * TILE), (xb - xa) * TILE, (yb - ya) * TILE,
                                   fill=False, ec=col, lw=0.8, ls="--"))
        ax.set_title(lab, fontsize=6.3, pad=2)
    fig.savefig(os.path.join(OUT, "fig5_ablation.pdf"), dpi=300)
    plt.close(fig)
    return rows, tri, f_tb


# ---------------------------------------------------------------- appendix: user study
def figA(per_page=10):
    us = os.path.join(ROOT, "user_study")
    key = json.load(open(os.path.join(us, "key.json")))["trials"]
    ans = json.load(open(os.path.join(us, "answers", "answers_author1.json")))["answers"]
    page = open(os.path.join(us, "index.html"), encoding="utf-8").read()
    import re
    items = {t["t"]: t for t in json.loads(re.search(r"T=(\[.*?\]);\n", page, re.S).group(1))}
    pages = []
    for p0 in range(0, len(key), per_page):
        tr = key[p0:p0 + per_page]
        fig = plt.figure(figsize=(7.0, 8.9))
        th = 1.3
        for j, t in enumerate(tr):
            r, c = divmod(j, 2)
            x, y = c * 3.55, 0.14 + r * (th + 0.46)
            it = items[t["trial"]]
            for k, rr in enumerate(it["refs"]):
                ax = axes_in(fig, x, y + k * th / 2, th / 2 - 0.02, th / 2 - 0.02)
                im = Image.open(os.path.join(us, rr)).convert("RGB")
                w_, h_ = im.size
                m_ = min(w_, h_)
                im = im.crop(((w_ - m_) // 2, (h_ - m_) // 2, (w_ - m_) // 2 + m_,
                              (h_ - m_) // 2 + m_)).resize((128, 128))
                ax.imshow(np.asarray(im))
                off(ax)
            ours = "L" if t["left"] == "ours" else "R"
            for k, side in enumerate("LR"):
                ax = axes_in(fig, x + th / 2 + 0.02 + k * (th + 0.03), y, th, th)
                ax.imshow(np.asarray(Image.open(os.path.join(us, it[side])).convert("RGB")
                                     .resize((256, 256))))
                off(ax)
                ax.set_title(f"{'Left' if side == 'L' else 'Right'}: "
                             f"{'ours' if side == ours else t['baseline']}", fontsize=6, pad=1.5,
                             color="#1f77b4" if side == ours else "k")
                if side == ours:
                    frame(ax, "#1f77b4", 1.2)
            a = [ans.get(f"{t['trial']}_{q}") or "-" for q in ("id", "text", "overall")]
            v = " / ".join({"tie": "tie", "-": "n/a", ours: "ours"}.get(s_, "baseline") for s_ in a)
            fig.text(x / 7.0, 1 - (y + th + 0.04) / 8.9, f"#{t['trial'] + 1}  {it['prompt']}",
                     fontsize=6, va="top", style="italic")
            fig.text(x / 7.0, 1 - (y + th + 0.17) / 8.9,
                     f"author (identity / text / overall): {' / '.join(a)}  \u2192  {v}",
                     fontsize=6, va="top")
        fn = f"figA_userstudy_{p0 // per_page + 1}.pdf"
        fig.savefig(os.path.join(OUT, fn), dpi=200)
        plt.close(fig)
        pages.append(fn)
    return pages


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["1", "3", "4", "5", "A"]
    if "1" in which:
        print("fig1", fig1())
    if "3" in which:
        fig3()
        print("fig3")
    if "4" in which:
        print("fig4", fig4())
    if "5" in which:
        print("fig5", fig5())
    if "A" in which:
        print("figA", figA())

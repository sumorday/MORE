"""Blind A/B user-study materials: ours (final config) vs the baselines on the K10 held-out pair
benchmark (same prompts and seeds), plus the analysis of the returned answer files.

build: --n trials split evenly over the baselines that have images; per trial a random
(prompt, seed) of the pair benchmark (no repeats within a baseline), ours on a random side.
Writes <out>/index.html (self-contained offline form; images copied under <out>/img with
anonymous names, one reference image per subject under <out>/ref), <out>/key.json (hidden:
trial -> arm per side; do not give it to raters).  Raters open index.html, answer three
questions per trial (both identities kept / text match / overall; left, right or tie) and
press the button that downloads answers_<name>.json.

pdf: --pdf -> <out>/blind_test.pdf from the built index.html (same trials, same random sides, no
method names): one landscape page per trial (number, prompt, references, left / right image) with
fillable radio buttons per question (fields <trial>_<question>, same as the html answers).

analyze: --analyze answers_*.json / filled blind_test*.pdf -> per baseline and question: our win /
tie / loss rate over all raters x trials, win rate excluding ties with a 95% Wilson interval.

  python scripts/user_study.py --out user_study
  python scripts/user_study.py --out user_study --pdf
  python scripts/user_study.py --out user_study --analyze user_study/answers/*.json
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common

GEN = os.path.join(common.ROOT, "gen_sph_pairs")
OURS = "rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb_K10"
BASELINES = {"FreeFuse": "xb_ff_K10", "LoRAShop": "xb_ls_K10", "RP-FLUX": "xb_rp_w0.5_K10",
             "SSR-Merge": "rg_ssrm_st_f0_K10", "LoRA-sum": "xb_sum_K10"}
QUESTIONS = [("id", "哪张图同时更好地保留了两个参考主体的身份（外观细节）？",
              "Which image better preserves the identity of BOTH reference subjects?"),
             ("text", "哪张图更符合文字描述（两个主体都出现、场景正确）？",
              "Which image better matches the text (both subjects present, correct scene)?"),
             ("overall", "综合来看你更喜欢哪张图？", "Overall, which image do you prefer?")]


def files_of(arm, seeds):
    out = {}
    for d in sorted(os.listdir(GEN)):
        if d == arm or (d.startswith(arm + "_s") and d[len(arm) + 2:].replace("-", "").isdigit()):
            for f in os.listdir(os.path.join(GEN, d)):
                if f.endswith(".png") and int(f.split("_s")[1][:-4]) in seeds:
                    out[f] = os.path.join(GEN, d, f)
    return out


def build(args):
    import generate_pairs as gp
    rng = random.Random(args.seed)
    prompts = gp.pair_prompts(10, 10, 2)
    ours = files_of(OURS, args.seeds)
    avail = {k: files_of(v, args.seeds) for k, v in BASELINES.items()}
    avail = {k: v for k, v in avail.items() if len(set(v) & set(ours)) >= 20}
    print("baselines:", {k: len(v) for k, v in avail.items()})
    names = sorted(avail)
    per = [args.n // len(names) + (j < args.n % len(names)) for j in range(len(names))]
    trials = []
    for name, m in zip(names, per):
        for f in rng.sample(sorted(set(avail[name]) & set(ours)), m):
            trials.append((name, f))
    rng.shuffle(trials)
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ref"), exist_ok=True)
    key, items = [], []
    for t, (name, f) in enumerate(trials):
        p = prompts[int(f[:3])]
        sides = [("ours", ours[f]), (name, avail[name][f])]
        rng.shuffle(sides)
        im = []
        for s, (arm, src) in zip("LR", sides):
            dst = f"img/t{t:02d}_{s}.png"
            shutil.copy(src, os.path.join(args.out, dst))
            im.append(dst)
        refs = []
        for subj in p["subjects"]:
            dst = f"ref/{subj}{os.path.splitext(common.ref_images(subj)[0])[1]}"
            if not os.path.exists(os.path.join(args.out, dst)):
                shutil.copy(common.ref_images(subj)[0], os.path.join(args.out, dst))
            refs.append(dst)
        key.append(dict(trial=t, file=f, baseline=name, left=sides[0][0], right=sides[1][0]))
        items.append(dict(t=t, prompt=p["prompt"].replace(f"{common.TOKEN} ", ""), refs=refs,
                          L=im[0], R=im[1]))
    json.dump(dict(ours=OURS, baselines=BASELINES, seeds=args.seeds, trials=key),
              open(os.path.join(args.out, "key.json"), "w"), indent=1)
    open(os.path.join(args.out, "index.html"), "w").write(html(items))
    print(f"{len(items)} trials -> {args.out}/index.html (key.json is the hidden mapping)")


def html(items):
    q = json.dumps([[k, zh, en] for k, zh, en in QUESTIONS], ensure_ascii=False)
    data = json.dumps(items, ensure_ascii=False)
    return """<!doctype html><html><head><meta charset="utf-8"><title>User study</title>
<style>body{font-family:sans-serif;max-width:1100px;margin:auto}
.trial{border:1px solid #ccc;padding:12px;margin:18px 0}.refs img{height:120px;margin-right:6px}
.pair img{width:480px;margin:4px}.q{margin:6px 0}</style></head><body>
<h2>多主体图像生成盲测 / Blind A/B study</h2>
<p>每题给出两个参考主体（上排）、一句文字描述和两张生成图（左 / 右）。请按三个问题分别选择左、右或差不多。
全部答完后填写你的代号并点击页面底部按钮下载答案文件，发回给我们。<br>
Each trial shows two reference subjects, a text prompt and two generated images (left / right).
Answer the three questions, then enter a name and press the button at the bottom to download your answers.</p>
<p>代号 / Name: <input id="rater"></p><div id="trials"></div>
<button onclick="save()">下载答案 / Download answers</button> <span id="msg"></span>
<script>
const Q=""" + q + """, T=""" + data + """;
const box=document.getElementById('trials');
T.forEach(t=>{let h=`<div class="trial"><b>#${t.t+1}</b> 文字 / Prompt: <i>${t.prompt}</i>
<div class="refs">参考 / References: ${t.refs.map(r=>`<img src="${r}">`).join('')}</div>
<div class="pair"><img src="${t.L}"><img src="${t.R}"></div>`;
Q.forEach(([k,zh,en])=>{h+=`<div class="q">${zh} / ${en}<br>`+
['L','tie','R'].map(v=>`<label><input type="radio" name="${t.t}_${k}" value="${v}"> ${
{L:'左 Left',tie:'差不多 Tie',R:'右 Right'}[v]}</label>`).join(' ')+`</div>`});
box.insertAdjacentHTML('beforeend',h+'</div>')});
function save(){const r=document.getElementById('rater').value.trim()||'anon';let a={rater:r,answers:{}},miss=0;
T.forEach(t=>Q.forEach(([k])=>{const e=document.querySelector(`input[name="${t.t}_${k}"]:checked`);
if(e)a.answers[`${t.t}_${k}`]=e.value;else miss++}));
if(miss&&!confirm(`还有 ${miss} 个未答，仍然下载？ / ${miss} unanswered, download anyway?`))return;
const b=new Blob([JSON.stringify(a)],{type:'application/json'});const l=document.createElement('a');
l.href=URL.createObjectURL(b);l.download=`answers_${r}.json`;l.click();
document.getElementById('msg').textContent='已下载 / saved';}
</script></body></html>"""


def pdf(args):
    import re
    import tempfile
    from PIL import Image
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))    # viewer-side CJK font, no file
    zh, en = "STSong-Light", "Helvetica"
    page = open(os.path.join(args.out, "index.html"), encoding="utf-8").read()
    items = json.loads(re.search(r"T=(\[.*?\]);\n", page, re.S).group(1))
    out = os.path.join(args.out, "blind_test.pdf")
    W, H = landscape(A4)
    c = canvas.Canvas(out, pagesize=(W, H))
    c.setTitle("Blind A/B study")
    c.setAuthor("")
    c.setCreator("")
    c.setFont(zh, 20)
    c.drawString(40, H - 70, "多主体图像生成盲测 / Blind A/B study")
    c.setFont(zh, 12)
    for i, s in enumerate([
            f"共 {len(items)} 题。每题：上方为文字描述和两个参考主体，中间为左、右两张生成图（方法已隐藏、左右随机）。",
            "每题回答三个问题，点选 左 / 差不多 / 右。",
            "答完后保存 PDF（Acrobat / 预览 均可，直接 Cmd+S 保存即可；不要\"打印为 PDF\"，会丢掉选项）发回。",
            "",
            f"{len(items)} trials. Each page: prompt + two reference subjects, then two generated images",
            "(left / right, methods hidden, random sides). Tick Left / Tie / Right for the three questions,",
            "save the PDF (do not 'print to PDF', which flattens the answers) and send it back."]):
        c.drawString(40, H - 120 - 20 * i, s)
    c.showPage()
    form = c.acroForm
    tmp = tempfile.mkdtemp()

    def jpg(rel):                                              # JPEG q95 keeps the PDF small
        dst = os.path.join(tmp, rel.replace("/", "_") + ".jpg")
        if not os.path.exists(dst):
            Image.open(os.path.join(args.out, rel)).convert("RGB").save(dst, quality=95)
        return dst
    for t in items:
        c.setFont(en, 16)
        c.drawString(30, H - 35, f"#{t['t'] + 1} / {len(items)}")
        c.setFont(zh, 12)
        c.drawString(110, H - 35, "文字 / Prompt:")
        c.setFont("Helvetica-Oblique", 13)
        c.drawString(200, H - 35, t["prompt"])
        c.setFont(zh, 10)
        c.drawString(30, H - 60, "参考 / References:")
        for j, r in enumerate(t["refs"]):
            c.drawImage(jpg(r), 130 + j * 85, H - 135, 78, 78, preserveAspectRatio=True,
                        anchor="sw")
        side, gap = 300, 40                                    # generated images, 512 px -> 300 pt
        x0, y0 = (W - 2 * side - gap) / 2, H - 145 - side
        for j, (lab, im) in enumerate([("左 Left", t["L"]), ("右 Right", t["R"])]):
            x = x0 + j * (side + gap)
            c.drawImage(jpg(im), x, y0, side, side)
            c.setFont(zh, 13)
            c.drawCentredString(x + side / 2, y0 - 16, lab)
        cw = (W - 60) / 3                                      # three question columns below
        for qi, (k, qz, qe) in enumerate(QUESTIONS):
            qx, y = 30 + qi * cw, y0 - 40
            c.setFont(zh, 9)
            c.drawString(qx, y, f"Q{qi + 1}. {qz}")                # CJK has no spaces to wrap at
            c.setFont(en, 7.5)
            for li, s in enumerate(simpleSplit(qe, en, 7.5, cw - 15)):
                c.drawString(qx, y - 13 - 9 * li, s)
            for vi, (v, lab) in enumerate([("L", "左 Left"), ("tie", "差不多 Tie"), ("R", "右 Right")]):
                bx = qx + vi * 75
                form.radio(name=f"{t['t']}_{k}", value=v, selected=False, x=bx, y=y - 68, size=14,
                           buttonStyle="circle", borderWidth=1, shape="circle",
                           tooltip=f"#{t['t'] + 1} Q{qi + 1} {v}")
                c.setFont(zh, 9)
                c.drawString(bx + 17, y - 64, lab)
        c.showPage()
    c.save()
    shutil.rmtree(tmp)
    print(f"{len(items)} trials -> {out}")


def read_answers(fn):
    if fn.endswith(".pdf"):                                    # filled blind_test.pdf form
        from pypdf import PdfReader
        f = PdfReader(fn).get_fields() or {}
        return {k: str(v.get("/V")).lstrip("/") for k, v in f.items()
                if v.get("/V") not in (None, "/Off", "Off", "")}
    return json.load(open(fn))["answers"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def analyze(args):
    key = json.load(open(os.path.join(args.out, "key.json")))["trials"]
    files = sorted(set(sum((glob.glob(p) for p in args.analyze), [])))
    stats = {}
    for fn in files:
        a = read_answers(fn)
        print(f"{fn}: {len(a)} answers")
        for tr in key:
            ours_side = "L" if tr["left"] == "ours" else "R"
            for k, _, _ in QUESTIONS:
                v = a.get(f"{tr['trial']}_{k}")
                if v in (None, ""):                             # unanswered
                    continue
                r = "tie" if v == "tie" else ("win" if v == ours_side else "loss")
                for b in (tr["baseline"], "ALL"):
                    stats.setdefault((b, k), {"win": 0, "tie": 0, "loss": 0})[r] += 1
    print(f"{len(files)} raters")
    for (b, k), s in sorted(stats.items()):
        n = sum(s.values())
        lo, hi = wilson(s["win"], s["win"] + s["loss"])
        print(f"  {b:10s} {k:8s} n={n:4d} win {s['win'] / n:.3f} tie {s['tie'] / n:.3f} "
              f"loss {s['loss'] / n:.3f} | win w/o ties {s['win'] / max(1, s['win'] + s['loss']):.3f}"
              f" [{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(common.ROOT, "user_study"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyze", nargs="*", default=None)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(args)
    elif args.pdf:
        pdf(args)
    else:
        build(args)


if __name__ == "__main__":
    main()

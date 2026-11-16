# MORE: Merge Once, Route Early — code

Training-free multi-subject composition from a pool of K subject LoRAs on FLUX.1-dev.
Stage 1 merges the pool once (calibrated subspace merge, fixed rank, exact factorization
`W0 + sum_k Wk`); Stage 2 routes the merged parts per prompt (early layout gating, soft
text–region attention bias, image–image repulsion, cyclic swap guidance).

This is the research code that produced every number of the paper, cleaned for review
(paths are relative / environment variables; cluster scripts removed). Model weights,
LoRAs, generated images and caches are **not** included.

## Layout

```
scripts/                     flat modules, run from the repository root (python scripts/<x>.py)
  common.py                  paths (MORE_ROOT, MORE_MODELS, FLUX_MODEL), subjects, prompts
  precompute.py, train_lora.py               DreamBooth latents/embeddings, one LoRA per subject
  calib_stats.py, mix_stats.py               Stage 1 calibration statistics (G, Q)
  ssrlib.py, ssr_rc.py, tcpr.py, solve_from_stats.py
                                             router R = Q/n (G/n + lam I)^-1, rank-r compression,
                                             energy gain, shared-direction damping
  run_spherical_proto.py     build: exact factorization of a merge into W0, W1..WK (per layer)
  run_regional_proto.py      Stage 2 sampling (gen) and detector evaluation (eval) of an arm
  generate_pairs.py          pair benchmark (10 pairs x 2 templates per pool), prompt encoding
  eval_pairs_det.py, eval_metrics.py         OWLv2 disjoint boxes + DINOv2 crop identity
  pair_image_checks.py       CLIP collage check
  paired_vs.py               bothC / minC / meanC with collages as failures, paired bootstrap SE
  eval_layout.py             any number of subjects / layouts (top-bottom, big-small, triples)
  std_metrics.py             HPSv2.1, ImageReward, CLIP-T, CLIP-I (envs/metrics)
  id_confusion.py, mech_maps.py              identity-confusion table, attention-map figure
  run_ext_baselines.py       LoRA sum / LoRA Switch / LoRA Composite on FLUX
  run_freefuse.py, run_lorashop.py, run_rpflux.py   wrappers around the official baselines
  random_gp.py, ortho_reg.py                 merge variants of the ablation
  user_study.py              blind A/B trials (offline HTML) and their analysis
  make_paper_figs.py         Figs. 1, 3, 4, 5 and the user-study figures
  make_supp_grids.py         unselected grids of the supplement (all 20 prompts, seed 6, every method)
src/spherical_routing/
  spherical_svm.py           per-layer factors of a merge (LayerFactors)
  dynamic_hook.py            runtime hook: per-token weights of W0, W1..WK
  regional.py                LayoutRouter (gating), RegionalAttention (bias of Eq. 3),
                             SwapGuidance (Eq. 4), layouts (lr / tb / bs / strips)
ext/rpflux_port/             RP-FLUX pipeline ported to current diffusers (Apache-2.0 header kept)
```

Arm names encode the configuration (full grammar in the docstring of
`scripts/run_regional_proto.py`). The paper's method is

```
F = rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb
    nm     our merge (factored)          lg/lu0.3  layout gating for round(0.3*28)=8 steps
    f1_n0  bias in all steps, no blocking p2_cos   +2 own text<->region bias, cosine decay
    ii1    image-image repulsion -p/2    sg2_sgmb  swap guidance w=2 at steps {0,2,4,6,8}, bias only
```

Ablation arms replace `nm` (`np` w/o de-crowding, `avg` task arithmetic, `ssrm` SSR-Merge as
published, `ssrmc` SSR-Merge on our calibration, `raw` own LoRAs, `base` no adapter) or the route
(`st_f0` = static merge, no routing).

## Setup

```
pip install -r requirements.txt          # main env; see the file for the two auxiliary venvs
export MORE_ROOT=$PWD                    # default: the parent of scripts/
export MORE_MODELS=/path/to/models       # flux1-dev/, dinov2-base/, clip-vit-base-patch32/
```

* FLUX.1-dev (`black-forest-labs/FLUX.1-dev`), DINOv2-base, CLIP ViT-B/32 under `$MORE_MODELS`;
  OWLv2 and the metric checkpoints (HPSv2.1, ImageReward v1.0, BERT-base, CLIP ViT-L/14) under
  `$MORE_ROOT/models_ext/` (see the defaults in `eval_pairs_det.py` / `std_metrics.py`).
* The 20 DreamBooth subjects (`google/dreambooth`) under `data/dreambooth/imgs/<subject>/`.
* External baselines (only for Table 1): clone the official FreeFuse and LoRAShop repositories
  into `ext/FreeFuse` and `ext/LoRAShop`.

## Reproduction (K = 10; replace 10 / 8 / 0.05 by 20 / 16 / 0.1 for K = 20)

```
# 0. subject LoRAs: rank 32, lr 1e-4, 500 steps, "a sks <class>"
for s in $(python -c "import sys; sys.path.insert(0,'scripts'); import common; print(' '.join(common.SUBJECTS))"); do
  python scripts/precompute.py $s
  python scripts/train_lora.py $s --out loras_seed2/$s
done

# 1. Stage 1 calibration: noise latents, 5 scenes, t in {.1,.3,.5,.7,.9}; non-padding tokens x3
python scripts/calib_stats.py --K 10 20 --x0 noise            --out_dir data/seed2_calN
python scripts/calib_stats.py --K 10 20 --x0 noise --txt_trim --out_dir data/seed2_calNT
python scripts/mix_stats.py --inputs data/seed2_calN data/seed2_calNT --weights 1 3 \
    --K 10 20 --out_dir data/seed2_calNmT3

# 2. Stage 1 merge (router, relative ridge, rank r, energy gain, shared-direction damping)
python scripts/ssr_rc.py --mode ssr_rc --rank 8 --tau 0.05 --g_star 1.0 --pico --K 10 \
    --stats_root data/seed2_calNmT3 --loras_root loras_seed2 --prefix s2calNmT3_
#    -> merged/s2calNmT3_ssr_rc_r8_tau0.05_g1_pico_K10.safetensors (static merge)
python scripts/run_spherical_proto.py --stage build --K 10 --variants nm
#    -> data/sph/..._nm.pt (exact factorization W0, W1..WK; checked against the static merge)

# 3. Stage 2 sampling on the held-out seeds 6-11 (20 prompts x 6 seeds = 120 images)
F=rg_nm_lg_f1_n0_p2_cos_lu0.3_ii1_sg2_sgmb
python scripts/run_regional_proto.py --stage gen --K 10 --arms ${F}_K10 \
    --seeds 6 7 8 9 10 11 --outsuffix _s6-11          # -> gen_sph_pairs/${F}_K10_s6-11/

# 4. evaluation (Table 1/2 columns)
python scripts/run_regional_proto.py --stage eval --K 10 --tag _ho --arms ${F}_K10_s6-11
python scripts/pair_image_checks.py --arms ${F}_K10 --ref ${F}_K10 \
    --out results/rg_imgchecks_K10.json
python scripts/paired_vs.py --K 10 --seeds 6 7 8 9 10 11 --clean --ref ${F}_K10
envs/metrics/bin/python scripts/std_metrics.py --genroot gen_sph_pairs --seeds 6 7 8 9 10 11 \
    --arms ${F}_K10 --ref ${F}_K10 --out results/std_metrics_pairs_K10.json
```

Generalization (Table 3): append `_lotb` / `_lobs` to the arm for top-bottom / big-small layouts,
`--protocols triples` for three subjects, and score with `eval_layout.py`. Automatic layouts:
append `_al`. Baselines (default seeds 6-11; all with `--outsuffix _s6-11`):
`run_ext_baselines.py --stage gen --arms xb_sum_K10 xb_lsw_K10 xb_lcp_K10`,
`run_freefuse.py --K 10`, `envs/ls031/bin/python scripts/run_lorashop.py --K 10`,
`run_rpflux.py --K 10 --lora_w 0.5`; then
`run_spherical_proto.py --stage eval --K 10 --tag _xb --arms xb_..._K10_s6-11`,
`pair_image_checks.py` on the arms and `paired_vs.py --K 10 --seeds 6 7 8 9 10 11 --clean --grep xb_ --ref ${F}_K10`.

All design choices were made on seeds 0-5 (`--seeds 0 1 2 3 4 5`); seeds 6-11 are reported.
Sampling: 512x512, 28 steps, guidance 3.5, seed 1000*seed_index + prompt_index, the same cached
T5/CLIP embeddings (`data/processed/pairs_embeds.pt`) for every method.

`results/` holds the text outputs of `paired_vs.py`, `eval_layout.py` and `std_metrics.py`
behind Tables 1-3 (`ho_*`: routing ablation, `mabl_*`: merge ablation and automatic layout,
`xb_*`: external baselines, all with collage-aware metrics in the second block).

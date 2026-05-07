# Privacy Auditing in One Run Using Normality

End-to-end codebase for training and **empirically auditing** differentially-private deep-learning algorithms on CIFAR-10. Two mechanisms are supported out of the box:

- **DP-SGD** (centralized, Opacus + Sander/Mahloujifar recipe) on a WideResNet-16-4.
- **DP-FTRL** (federated-learning-style, tree-aggregated Gaussian / Honaker construction) on a Tramer–Boneh ScatterLinear model.

Both pipelines inject *dirac white-box canaries* — a single coordinate per canary, perturbed by `±C` post-clipping at one randomly assigned step/leaf — and produce per-canary scalar scores that can be plugged into multiple lower-bound estimators of the realized `(ε, δ)`.

---

## Install

```bash
conda create --prefix /path/to/venv python=3.12 -y
conda activate /path/to/venv
pip install -r requirements.txt
```

[//]: # (PACE / GT cluster users:)

[//]: # ()
[//]: # (```bash)

[//]: # (bash scripts/local_scripts/cluster_install_gpu.sh)

[//]: # (```)

DP-FTRL additionally requires `kymatio` for the scattering transform (already in `requirements.txt`).

---

## Project layout

```
src/
  auditing.py                 # Steinke / Mahloujifar nonparametric auditors
  train.py                    # DP-SGD training loop with canary injection
  train_dpftrl.py             # DP-FTRL training + DPFTRLState (streaming Honaker tree)
  network_arch.py             # WideResNet (DP-SGD)
  scatter_network.py          # ScatterLinear (DP-FTRL)
  dataset.py                  # CIFAR-10 loaders
  whitebox_auditing/
    ndis_1d.py                # NDIS auditor (parametric_bonferroni, bootstrap_ellipsoid, dp_aware)
    tree_mechanism.py         # (eps, delta) <-> sigma_node calibration for the tree mechanism
scripts/
  gen_scores_DP_whitebox.py        # DP-SGD: train + dump per-canary scores
  gen_scores_DP_FTRL_scatter.py    # DP-FTRL: train + dump per-canary scores
  run_auditing_comparison.py       # Multi-method audit + plots (per-eps, T-ablation, complexity)
  plot_dpftrl_audit.py             # DP-FTRL Bonferroni-CI plot
  run_*_phoenix.sbatch             # Slurm wrappers for each pipeline
```

## Audit methods

| Method                                     | Score                       | Plug used in |
|--------------------------------------------|-----------------------------|--------------|
| **Our method, parametric Bonferroni**      | optimal sum / `√L` (NDIS)   | DP-SGD, DP-FTRL |
| **Our method, bootstrap ellipsoid**        | optimal sum / `√L`          | DP-SGD, DP-FTRL |
| **Steinke et al. 2023** (one-run binomial) | optimal sum                 | DP-SGD, DP-FTRL |
| **Mahloujifar et al. 2024** (f-DP)         | optimal sum                 | DP-SGD, DP-FTRL |
| **Andrew et al. 2024** (max cosine)        | `max_t cos(e_{c_i}, G_t)`   | DP-FTRL only (`--with-andrew`) |

NDIS is what *we* propose; the others are reproduced as baselines on the same per-canary scalars to keep comparisons apples-to-apples.

---

## Pipeline 1 — DP-SGD audit

1. Train + dump per-canary scores:

```bash
python scripts/gen_scores_DP_whitebox.py \
    --epsilon 8 --delta 1e-5 \
    --target-steps 2500 \
    --logical-batch-size 4096 --max-physical-batch-size 128 \
    --aug-multiplicity 16 \
    --canary-count 5000 --pkeep 0.5 \
    --lr 4.0 --ema-decay 0.9999 \
    --data-dir ./data --log-dir ./logs
```

This produces an `exp_dir = ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10/` containing
`in_scores_sum_*.csv`, `out_scores_sum_*.csv`, `in_scores_ndis_*.csv`, `out_scores_ndis_*.csv`,
`hparams.json`, `inclusion_mask.csv`, `canary_directions.csv`, and per-epoch checkpoints.

2. Compare audit methods on a sweep over target eps:

```bash
python scripts/run_auditing_comparison.py \
    --exp-dirs ./data/<run_eps1> ./data/<run_eps2> ./data/<run_eps4> ./data/<run_eps8>
```

Outputs `fig/privacy_bounds_comparison_multi_eps.{png,pdf}` (Steinke / Mahloujifar / NDIS bootstrap-ellipsoid vs theoretical ε) and `fig/auditing_comparison_final.csv`.

3. Sample-complexity ablation on a single run:

```bash
python scripts/run_auditing_comparison.py --complexity \
    --exp-dir ./data/<run>
```

---

## Pipeline 2 — DP-FTRL audit (ScatterLinear)

1. Train + dump per-canary scores (single-pass over CIFAR-10):

```bash
python scripts/gen_scores_DP_FTRL_scatter.py \
    --epsilon 8 --delta 1e-5 \
    --target-steps 128 \
    --logical-batch-size 380 --max-physical-batch-size 390 \
    --canary-count 5000 --pkeep 0.5 \
    --lr 1.0 \
    --data-dir ./data --log-dir ./logs
```

Produces `exp_dir = ./data/dpftrl-scatter-canaries-<seed>-5000-0.5-cifar10/` with
`in_scores_optimal_*.csv`, `out_scores_optimal_*.csv`, `in_scores_ndis_*.csv`, `out_scores_ndis_*.csv`,
`in_scores_andrew_*.csv`, `out_scores_andrew_*.csv`, plus `hparams.json` / `inclusion_mask.csv` / `canary_coords.csv` / `canary_leaves.csv`.

2. Compare audit methods on the DP-FTRL eps sweep (`--with-andrew` enables the Andrew baseline and hides the Steinke / Mahloujifar lines, since those are designed for DP-SGD-style score distributions):

```bash
python scripts/run_auditing_comparison.py --with-andrew \
    --exp-dirs ./data/dpftrl-scatter-eps1 ./data/dpftrl-scatter-eps2 \
               ./data/dpftrl-scatter-eps4 ./data/dpftrl-scatter-eps8
```

Outputs `fig/privacy_bounds_comparison_multi_eps.{png,pdf}` (NDIS vs Andrew) and the matching CSV.

3. T-ablation at fixed eps:

```bash
python scripts/run_auditing_comparison.py --ablation-T --with-andrew \
    --exp-dirs ./data/dpftrl-scatter-T64 ./data/dpftrl-scatter-T128 \
               ./data/dpftrl-scatter-T256 ./data/dpftrl-scatter-T512
```

Outputs `fig/ablation_T_eps<eps>.{png,pdf}` + matching CSV.

4. Provable Bonferroni-CI plot for one eps sweep:

```bash
python scripts/plot_dpftrl_audit.py \
    --exp-dirs ./data/dpftrl-scatter-eps{1,2,4,8} \
    --fig-path ./fig/dp-ftrl-audit-bonferroni-ci.png
```

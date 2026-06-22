# Reproducibility Guide

This repository contains the code needed to reproduce the analysis and the
deployable **BurnoutGuard** Moodle plugin. It is released under the GNU GPL v3
license (see `LICENSE`). Nothing is access-on-request.

## 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # xgboost, lightgbm, shap, scikit-learn,
                                          # imbalanced-learn, fairlearn, scipy, pandas, matplotlib
```
All experiments use a fixed random seed (`42`). Tested with Python 3.11–3.13,
xgboost 3.1, lightgbm 4.6, scikit-learn 1.8, shap 0.50, fairlearn 0.13.

## 2. Data

The Open University Learning Analytics Dataset (OULAD) is public:
<https://analyse.kmi.open.ac.uk/open_dataset>. Place the seven CSVs in `data/`
(or run `python data/download_oulad.py`).

## 3. Target definition

Academic risk is operationalized as **course withdrawal**
(`final_result == "Withdrawn"`), evaluated on the VLE-active analytical sample
(29,228 student–course registrations, 24.5% positive).

## 4. Reproducing each result

| Result | Script | Output |
|---|---|---|
| Class-imbalance ablation (CV ± SD) | `revision_pipeline.py` | `results/metrics/revision/ablation_cv.csv`, `ablation_ttests.json` |
| Model comparison (CV ± SD) | `revision_pipeline.py` | `results/metrics/revision/model_comparison_cv.csv` |
| Leakage-robustness | `revision_pipeline.py` | `results/metrics/revision/leakage_robust_cv.csv`, `leakage_tests.json` |
| Subgroup fairness (FPR/FNR/AUROC/ECE) | `revision_pipeline.py` | `results/metrics/revision/fairness_extended.json` |
| Fairness-constrained models (DP/EO) | `revision_fairness.py` | `results/metrics/revision/fairness_constrained.json` |
| SHAP global importance | `revision_shap.py` | `results/metrics/revision/shap_importance_withdrawn.csv` |
| Early prediction + bootstrap CIs | `revision_early.py` | `results/metrics/revision/early_prediction_withdrawn.csv` |
| Architecture control (plain XGBoost on truncated features) | `revision_arch_control.py` | `results/metrics/revision/early_arch_control_xgb.csv` |
| DeLong, leave-presentations-out CV, reliability diagram | `revision_followup.py` | `results/metrics/revision/{delong_lr_vs_xgb,cross_presentation_cv}.json`, `figures/calibration.png` |
| Pairwise DeLong/bootstrap, fairness bootstrap CIs, intersectional, per-group calibration | `revision_followup2.py` | `results/metrics/revision/{model_stats,fairness_bootstrap_ci,fairness_intersectional}.json`, `figures/calibration_groups.png` |
| TabNet deep-learning baseline (GPU) | `train_tabnet_hpc.py` | `results/tabnet_cv.json` |
| Figures (SHAP, ROC, early prediction) | `revision_figures.py` | `figures/*.png` |
| Pipeline diagram | `revision_pipeline_fig.py` | `figures/pipeline.png` |

```bash
python revision_pipeline.py     # ablation, model comparison, leakage, subgroup fairness
python revision_fairness.py     # fairlearn demographic-parity-constrained models
python revision_shap.py         # SHAP global importance
python revision_early.py        # temporally-truncated early prediction (weeks 2–20)
python revision_arch_control.py # plain XGBoost on the early features (architecture control)
python revision_followup.py     # DeLong, cross-presentation CV, reliability diagram
python revision_followup2.py    # pairwise model stats, fairness CIs, intersectional, calibration
python revision_figures.py      # regenerate figures
python revision_pipeline_fig.py # pipeline diagram
```

`run_real_pipeline.py` and `src/` provide the data loading and feature utilities
that the scripts above build on. The TabNet baseline (`train_tabnet_hpc.py`)
requires a CUDA GPU and `pytorch-tabnet`; it is the only step that benefits from
a GPU. Each script writes machine-readable CSV/JSON under
`results/metrics/revision/` and logs the reported values.

## 5. BurnoutGuard plugin

The deployable system (FastAPI backend + Moodle local plugin + Docker compose) is
under `burnoutguard/`. See `burnoutguard/README.md` for deployment. Explanations
are template-based by default; an optional LLM tier (OpenAI-compatible endpoint)
can be enabled, in which case the per-student risk factors are sent to that
endpoint — disable it, or point it at a self-hosted model, for fully local
operation.

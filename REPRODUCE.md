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
| Class-imbalance ablation (CV ± SD) | `pipeline.py` | `results/metrics/ablation_cv.csv`, `ablation_ttests.json` |
| Model comparison (CV ± SD) | `pipeline.py` | `results/metrics/model_comparison_cv.csv` |
| Ensemble under the same 5-fold CV (mean ± SD) | `ensemble_cv.py` | `results/metrics/ensemble_cv.json` |
| External validation on a second dropout dataset (UCI id 697) | `external_validation.py` | `results/metrics/external_validation_uci.json` |
| Early prediction with presentation-grouped folds; SHAP, calibration and fairness at weeks 4/8 | `horizons.py` | `results/metrics/{early_grouped_vs_random,horizon_fairness_calibration}.json`, `figures/shap_early.png` |
| Presentation-grouped folds for the stacking early model (one week per run) | `horizons_stacking.py <week>` | `results/metrics/early_grouped_stacking.json` |
| Post-hoc calibration at the deployment horizons (isotonic / Platt) | `horizons_calibration.py` | `results/metrics/horizon_calibration.json` |
| Operating points: fairness at calibrated / capacity thresholds, precision@k | `operating_point.py` | `results/metrics/operating_point.json` |
| Full-course fairness for the disability attribute | `fairness_disability.py` | `results/metrics/fairness_disability.json` |
| Leakage-robustness | `pipeline.py` | `results/metrics/leakage_robust_cv.csv`, `leakage_tests.json` |
| Subgroup fairness (FPR/FNR/AUROC/ECE) | `pipeline.py` | `results/metrics/fairness_extended.json` |
| Fairness-constrained models (DP/EO) | `fairness_constrained.py` | `results/metrics/fairness_constrained.json` |
| SHAP global importance | `shap_analysis.py` | `results/metrics/shap_importance_withdrawn.csv` |
| Early prediction + bootstrap CIs | `early_prediction.py` | `results/metrics/early_prediction_withdrawn.csv` |
| Architecture control (plain XGBoost on truncated features) | `early_architecture_control.py` | `results/metrics/early_arch_control_xgb.csv` |
| DeLong, leave-presentations-out CV, reliability diagram | `generalization_tests.py` | `results/metrics/{delong_lr_vs_xgb,cross_presentation_cv}.json`, `figures/calibration.png` |
| Pairwise DeLong/bootstrap, fairness bootstrap CIs, intersectional, per-group calibration | `model_stats_fairness.py` | `results/metrics/{model_stats,fairness_bootstrap_ci,fairness_intersectional}.json`, `figures/calibration_groups.png` |
| TabNet deep-learning baseline (GPU) | `tabnet_baseline.py` | `results/tabnet_cv.json` |
| Figures (SHAP, ROC, early prediction) | `make_figures.py` | `figures/*.png` |
| Pipeline diagram | `make_pipeline_diagram.py` | `figures/pipeline.png` |

```bash
python pipeline.py     # ablation, model comparison, leakage, subgroup fairness
python fairness_constrained.py     # fairlearn demographic-parity-constrained models
python shap_analysis.py         # SHAP global importance
python early_prediction.py        # temporally-truncated early prediction (weeks 2–20)
python early_architecture_control.py # plain XGBoost on the early features (architecture control)
python generalization_tests.py     # DeLong, cross-presentation CV, reliability diagram
python model_stats_fairness.py    # pairwise model stats, fairness CIs, intersectional, calibration
python make_figures.py      # regenerate figures
python make_pipeline_diagram.py # pipeline diagram
```

`oulad_data.py` and `src/` provide the data loading and feature utilities
that the scripts above build on. The TabNet baseline (`tabnet_baseline.py`)
requires a CUDA GPU and `pytorch-tabnet`; it is the only step that benefits from
a GPU. Each script writes machine-readable CSV/JSON under
`results/metrics/` and logs the reported values.

## 5. BurnoutGuard plugin

The deployable system (FastAPI backend + Moodle local plugin + Docker compose) is
under `burnoutguard/`. See `burnoutguard/README.md` for deployment. The backend
ships a fitted isotonic calibrator and applies it before any probability is shown;
regenerate model and calibrator with `python burnoutguard/backend/fit_calibrator.py
--data data --week 8`. `POST /batch_predict` accepts a `capacity` parameter that
flags the highest-risk fraction of a cohort instead of using a fixed threshold. Explanations
are template-based by default; an optional LLM tier (OpenAI-compatible endpoint)
can be enabled, in which case the per-student risk factors are sent to that
endpoint — disable it, or point it at a self-hosted model, for fully local
operation.

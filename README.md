# BurnoutGuard — Explainable and Fair Early Warning System for Academic Risk

Repository: <https://github.com/lspusal/burnoutguard>

An open-source pipeline and Moodle plugin for predicting student **withdrawal**
from learning-analytics data, with SHAP-based explanations and a fairness audit.

The system is built around two ideas:

- **Early prediction.** Using only the data available in the first weeks of a
  course, it estimates each student's risk of withdrawing, leaving time to
  intervene. Full-course performance is reported only as an upper bound.
- **Transparency and equity.** Predictions come with SHAP-based explanations, and
  the model is audited for demographic parity, error-rate parity, calibration,
  and intersectional effects.

## Repository layout

```
.
├── data/                 # OULAD download helper (the data itself is not committed)
├── src/                  # data loading and feature-engineering utilities
├── revision_*.py         # analysis scripts (one per result; see REPRODUCE.md)
├── run_real_pipeline.py  # shared data/feature layer
├── train_tabnet_hpc.py   # optional TabNet deep-learning baseline (GPU)
├── burnoutguard/         # deployable system: FastAPI backend + Moodle plugin
├── REPRODUCE.md          # step-by-step reproduction guide
├── requirements.txt
└── LICENSE               # GNU GPL v3
```

## Quick start

1. Reproduce the analysis: follow [`REPRODUCE.md`](REPRODUCE.md).
2. Deploy the plugin: follow [`burnoutguard/README.md`](burnoutguard/README.md).

All experiments use a fixed random seed (42) and the public OULAD dataset, so the
results are reproducible. Released under the GNU GPL v3 license.

#!/usr/bin/env python3
"""Ensemble (0.5*XGBoost + 0.5*LightGBM) under the SAME 5-fold stratified CV as
the other models, so Table 5 can report mean +/- SD for the ensemble too."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from pipeline import build_features, proba, met
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SEED = 42
data = load_oulad_data("data/")
df, X, y, helpers = build_features(data)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for tr, te in skf.split(X, y):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    pe = 0.5*proba("xgboost", Xtr, ytr, Xte, "smote_weight") + 0.5*proba("lightgbm", Xtr, ytr, Xte, "smote_weight")
    rows.append(met(yte.values, pe))
d = pd.DataFrame(rows)
summ = {c: (d[c].mean(), d[c].std(ddof=1)) for c in d}
for c in ["auroc","auprc","f1","precision","recall","specificity","brier"]:
    m,s = summ[c]; print(f"{c:12s} {m:.4f} +/- {s:.4f}")
Path("results/metrics").mkdir(parents=True, exist_ok=True)
json.dump({c:{"mean":float(summ[c][0]),"std":float(summ[c][1])} for c in d},
          open("results/metrics/ensemble_cv.json","w"), indent=2)

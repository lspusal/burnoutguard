#!/usr/bin/env python3
"""Post-hoc calibration at the deployment horizons (weeks 4 and 8).
The early models are trained with SMOTE + class weighting, which inflates the
predicted probabilities. Here we fit a post-hoc calibrator (Platt / isotonic) on
a held-out validation slice of the TRAINING data only, and report Brier and ECE
on the test set before and after. Seed 42."""
import warnings, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import pandas as pd
from horizons import extract_full, xgb_proba, ece
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)

vle = pd.read_csv("data/studentVle.csv"); info = pd.read_csv("data/studentInfo.csv")
res = {}
for wk in [4, 8]:
    f, feat = extract_full(vle, info, wk)
    X, y = f[feat].values, f["y"].values
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    isub, ival = train_test_split(itr, test_size=0.2, stratify=y[itr], random_state=SEED)
    model, p_te = xgb_proba(X[isub], y[isub], X[ite])
    p_val = model.predict_proba(X[ival])[:, 1]
    raw = {"brier": float(brier_score_loss(y[ite], p_te)), "ece": ece(y[ite], p_te),
           "auroc": float(roc_auc_score(y[ite], p_te))}
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y[ival])
    p_iso = iso.predict(p_te)
    pl = LogisticRegression().fit(p_val.reshape(-1, 1), y[ival])
    p_pl = pl.predict_proba(p_te.reshape(-1, 1))[:, 1]
    res[f"week{wk}"] = {"uncalibrated": raw,
        "isotonic": {"brier": float(brier_score_loss(y[ite], p_iso)), "ece": ece(y[ite], p_iso),
                     "auroc": float(roc_auc_score(y[ite], p_iso))},
        "platt": {"brier": float(brier_score_loss(y[ite], p_pl)), "ece": ece(y[ite], p_pl),
                  "auroc": float(roc_auc_score(y[ite], p_pl))}}
    r = res[f"week{wk}"]
    print(f"week {wk}: raw Brier={r['uncalibrated']['brier']:.3f} ECE={r['uncalibrated']['ece']:.3f} | "
          f"isotonic Brier={r['isotonic']['brier']:.3f} ECE={r['isotonic']['ece']:.3f} | "
          f"Platt Brier={r['platt']['brier']:.3f} ECE={r['platt']['ece']:.3f} | "
          f"AUROC {r['uncalibrated']['auroc']:.3f}->{r['isotonic']['auroc']:.3f}")
json.dump(res, open(OUT / "horizon_calibration.json", "w"), indent=2)
print("saved horizon_calibration.json")

#!/usr/bin/env python3
"""Recompute SHAP global importance (Table 6) with the Withdrawn target and the
43-feature matrix. Trains XGBoost (SMOTE+weight) on a train split and computes
TreeSHAP on up to 2000 test rows."""
import warnings, logging
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb, shap
from pipeline import build_features
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)

data = load_oulad_data("data/")
df, X, y, helpers = build_features(data)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
spw = (ytr == 0).sum() / (ytr == 1).sum()
m = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
    n_jobs=-1, verbosity=0)
m.fit(Xb, yb)
samp = Xte.sample(min(2000, len(Xte)), random_state=SEED)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(samp)
mean_abs = np.abs(sv).mean(axis=0)
imp = pd.DataFrame({"feature": samp.columns, "mean_abs_shap": mean_abs})
imp["importance_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(OUT / "shap_importance_withdrawn.csv", index=False)
log.info("Top 12 SHAP features (Withdrawn target):")
for _, r in imp.head(12).iterrows():
    log.info(f"  {r.feature:24s} {r.mean_abs_shap:.4f}  {r.importance_pct:.1f}%")
top3 = imp.head(3).importance_pct.sum()
log.info(f"Top-3 cumulative importance: {top3:.1f}%")

# SHAP dependence for last_access_day (absolute course day, not a recency).
j = list(samp.columns).index("last_access_day")
lad, sj = samp["last_access_day"].values, sv[:, j]
log.info(f"last_access_day: range {lad.min():.0f}-{lad.max():.0f}, "
         f"corr(value, SHAP) = {np.corrcoef(lad, sj)[0, 1]:.3f}")
edges = [-np.inf, 0, 25, 50, 75, 100, 150, 200, np.inf]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (lad > lo) & (lad <= hi)
    if m.sum() >= 20:
        rows.append({"day_from": lo, "day_to": hi, "n": int(m.sum()),
                     "median_shap": float(np.median(sj[m])), "mean_shap": float(sj[m].mean())})
        log.info(f"  days ({lo:g}, {hi:g}]  n={m.sum():5d}  median SHAP {np.median(sj[m]):+.3f}")
pd.DataFrame(rows).to_csv(OUT / "shap_dependence_last_access_day.csv", index=False)

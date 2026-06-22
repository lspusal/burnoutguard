#!/usr/bin/env python3
"""Architecture control for the early-prediction comparison.

The early-prediction results use a stacking ensemble, while the full-course
results use an XGBoost+LightGBM ensemble. To separate the effect of *reduced
information* (temporal truncation) from the effect of the *different
architecture*, we train a plain XGBoost on exactly the same temporally
truncated early features and compare it with the stacking ensemble at each week.
If the two are close, the drop from the full-course figure is driven by
information, not architecture. Withdrawal target, seed=42."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from train_ews_enhanced import extract_enhanced_features
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SEED = 42
OUT = Path("results/metrics/revision"); OUT.mkdir(parents=True, exist_ok=True)

vle = pd.read_csv("data/studentVle.csv")
info = pd.read_csv("data/studentInfo.csv")
info["final_result"] = info["final_result"].replace({"Fail": "Pass"})  # -> Withdrawn-only target

rows = []
for wk in [2, 4, 6, 8]:
    X, y = extract_enhanced_features(vle, info.copy(), wk)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr_s, ytr)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    m = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
        n_jobs=-1, verbosity=0)
    m.fit(Xb, yb)
    p = m.predict_proba(Xte_s)[:, 1]
    au = roc_auc_score(yte, p)
    rows.append({"week": wk, "xgb_auroc": float(au), "xgb_f1": float(f1_score(yte, (p >= .5).astype(int)))})
    print(f"week {wk}: plain XGBoost AUROC={au:.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "early_arch_control_xgb.csv", index=False)
# stacking AUROC from the earlier run, for side-by-side
stack = {2: 0.683, 4: 0.717, 6: 0.763, 8: 0.784}
print("\nweek  plain_XGB  stacking  diff")
for _, r in df.iterrows():
    w = int(r["week"]); print(f"{w:4d}  {r.xgb_auroc:.3f}      {stack[w]:.3f}     {r.xgb_auroc-stack[w]:+.3f}")
print("DONE_ARCH")

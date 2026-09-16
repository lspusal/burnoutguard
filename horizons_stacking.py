#!/usr/bin/env python3
"""Presentation-grouped generalization for the STACKING early-prediction model,
i.e. the model whose week-2/4/6/8 numbers are reported, so that the grouped
check is run on the reported model rather than only on a plain-XGBoost proxy.
Random StratifiedKFold vs GroupKFold by course presentation. Seed 42.
Usage: python horizons_stacking.py <week>   (results appended to JSON)"""
import sys, os, json, warnings
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from horizons import extract_full
from early_model import train_enhanced_model
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
JS = OUT / "early_grouped_stacking.json"

wk = int(sys.argv[1])
vle = pd.read_csv("data/studentVle.csv"); info = pd.read_csv("data/studentInfo.csv")
f, feat = extract_full(vle, info, wk)
X, y = f[feat].values, f["y"].values
grp = (f["code_module"].astype(str) + "_" + f["code_presentation"].astype(str)).values

def run(splits):
    out = []
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        m = train_enhanced_model(sc.transform(X[tr]), y[tr], None, None)
        out.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    return np.array(out)

r = run(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
g = run(GroupKFold(5).split(X, y, grp))
rec = {"week": wk, "random_mean": float(r.mean()), "random_std": float(r.std(ddof=1)),
       "grouped_mean": float(g.mean()), "grouped_std": float(g.std(ddof=1)),
       "delta": float(g.mean() - r.mean())}
allr = json.load(open(JS)) if JS.exists() else []
allr = [x for x in allr if x["week"] != wk] + [rec]
json.dump(sorted(allr, key=lambda x: x["week"]), open(JS, "w"), indent=2)
print(f"week {wk}: stacking random={r.mean():.3f}+/-{r.std(ddof=1):.3f}  "
      f"grouped={g.mean():.3f}+/-{g.std(ddof=1):.3f}  delta={g.mean()-r.mean():+.3f}")

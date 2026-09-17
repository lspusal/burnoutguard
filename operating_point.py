#!/usr/bin/env python3
"""Deployment operating point at the early horizons (weeks 4 and 8).

(A) Fairness gaps recomputed at usable operating points, to follow through on the
    caveat that threshold 0.5 on uncalibrated scores is not what a deployment
    should use: (i) at 0.5 on isotonically calibrated scores, and (ii) under a
    capacity rule that flags the top 20% of students.
(B) Precision@k / recall@k for an advisor who can only contact the top k.
    Note: isotonic calibration is monotone, so top-k sets (and therefore
    precision@k) are identical before and after calibration.
Seed 42."""
import sys, os, json, warnings
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix
from horizons import extract_full, xgb_proba
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)

def gaps(y, pred, g):
    vals = {"sel": [], "fpr": [], "fnr": []}
    for lvl in np.unique(g):
        m = g == lvl
        if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
        tn, fp, fn, tp = confusion_matrix(y[m], pred[m], labels=[0, 1]).ravel()
        vals["sel"].append(pred[m].mean())
        if fp + tn: vals["fpr"].append(fp / (fp + tn))
        if fn + tp: vals["fnr"].append(fn / (fn + tp))
    return {k: (float(max(v) - min(v)) if len(v) > 1 else float("nan")) for k, v in vals.items()}

def prec_rec_at_k(y, score, k):
    n = int(round(k * len(y)))
    idx = np.argsort(score)[::-1][:n]
    tp = y[idx].sum()
    return float(tp / n), float(tp / y.sum())

vle = pd.read_csv("data/studentVle.csv"); info = pd.read_csv("data/studentInfo.csv")
res = {}
for wk in [4, 8]:
    f, feat = extract_full(vle, info, wk)
    X, y = f[feat].values, f["y"].values
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    isub, ival = train_test_split(itr, test_size=0.2, stratify=y[itr], random_state=SEED)
    model, p_te = xgb_proba(X[isub], y[isub], X[ite])
    p_val = model.predict_proba(X[ival])[:, 1]
    p_cal = IsotonicRegression(out_of_bounds="clip").fit(p_val, y[ival]).predict(p_te)
    yte = y[ite]; meta = f.iloc[ite]

    attrs = {
        "gender": meta["gender"].values,
        "age_band": meta["age_band"].values,
        "imd_band": None,  # set below using the same median split as the horizon audit
        "disability": (meta["disability"].values == "Y").astype(int),
    }
    imd_num = (meta["imd_band"].astype(str).str.extract(r"(\d+)").astype(float)
               .fillna(50).values.ravel())
    attrs["imd_band"] = (imd_num <= np.median(imd_num)).astype(int)
    # all three scenarios share the same model and split, so they are comparable
    # Regla de capacidad: seleccionar EXACTAMENTE el 20% de mayor riesgo por
    # ranking. Un corte por percentil no vale: la isotónica produce mesetas con
    # muchos empates y ">= percentil 80" acaba marcando bastante más del 20%.
    n_flag = int(round(0.20 * len(p_cal)))
    top = np.zeros(len(p_cal), dtype=int)
    top[np.argsort(p_cal)[::-1][:n_flag]] = 1
    scen = {"raw_0.5": (p_te >= 0.5).astype(int),
            "cal_0.5": (p_cal >= 0.5).astype(int),
            "capacity_top20": top}
    out = {"n_test": int(len(yte)), "base_rate": float(yte.mean()),
           "sel_rate": {k: float(v.mean()) for k, v in scen.items()}, "fairness": {}}
    for sname, pred in scen.items():
        out["fairness"][sname] = {a: gaps(yte, pred, g) for a, g in attrs.items()}
    out["precision_at_k"] = {f"top{int(k*100)}": dict(zip(["precision", "recall"], prec_rec_at_k(yte, p_te, k)))
                             for k in (0.10, 0.20)}
    res[f"week{wk}"] = out
    print(f"=== week {wk}  (n={len(yte)}, base={yte.mean():.3f}) ===")
    for k, v in out["precision_at_k"].items():
        print(f"  {k}: precision={v['precision']:.3f} recall={v['recall']:.3f}")
    for sname in scen:
        print(f"  [{sname}] global selection={out['sel_rate'][sname]:.3f}")
        for a, gp in out["fairness"][sname].items():
            print(f"     {a:11s} DP={gp['sel']:.3f} FPR={gp['fpr']:.3f} FNR={gp['fnr']:.3f}")
json.dump(res, open(OUT / "operating_point.json", "w"), indent=2)
print("SAVED operating_point.json")

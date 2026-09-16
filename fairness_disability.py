#!/usr/bin/env python3
"""Full-course fairness for the DISABILITY attribute:
subgroup metrics (n, base rate, selection, AUROC, FPR, FNR, ECE) and parity
gaps with bootstrap 95% CIs, on the held-out test set. Same protocol as the
other full-course fairness results (XGBoost, SMOTE+weight, seed 42)."""
import warnings, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from pipeline import build_features, proba
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(SEED)


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1]) if i < bins-1 else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def rates(y, pred, a, kind):
    vals = []
    for g in np.unique(a):
        m = a == g; tn, fp, fn, tp = confusion_matrix(y[m], pred[m], labels=[0, 1]).ravel()
        v = (fp/(fp+tn) if fp+tn else np.nan) if kind == "fpr" else (fn/(fn+tp) if fn+tp else np.nan)
        if not np.isnan(v): vals.append(v)
    return (max(vals) - min(vals)) if len(vals) > 1 else np.nan

def dp(pred, a):
    r = [pred[a == g].mean() for g in np.unique(a)]
    return max(r) - min(r)


def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)
    itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
    yv = yte.values
    pb = proba("xgboost", Xtr, ytr, Xte, "smote_weight")
    pred = (pb >= 0.5).astype(int)
    dis = (Xte["has_disability"].values == 1).astype(int)   # 1 = has disability

    # subgroup metrics
    sub = {}
    for lvl, name in [(1, "disabled"), (0, "not_disabled")]:
        m = dis == lvl; tn, fp, fn, tp = confusion_matrix(yv[m], pred[m], labels=[0, 1]).ravel()
        sub[name] = {"n": int(m.sum()), "base_rate": float(yv[m].mean()),
            "selection_rate": float(pred[m].mean()), "auroc": float(roc_auc_score(yv[m], pb[m])),
            "fpr": float(fp/(fp+tn)), "fnr": float(fn/(fn+tp)), "ece": ece(yv[m], pb[m])}
    # parity + bootstrap CI
    point = {"dp": dp(pred, dis), "fpr_parity": rates(yv, pred, dis, "fpr"),
             "fnr_parity": rates(yv, pred, dis, "fnr")}
    n = len(yv); bd = {k: [] for k in point}
    for _ in range(1000):
        s = rng.randint(0, n, n)
        bd["dp"].append(dp(pred[s], dis[s]))
        bd["fpr_parity"].append(rates(yv[s], pred[s], dis[s], "fpr"))
        bd["fnr_parity"].append(rates(yv[s], pred[s], dis[s], "fnr"))
    ci = {k: {"point": float(point[k]), "ci_lo": float(np.nanpercentile(bd[k], 2.5)),
              "ci_hi": float(np.nanpercentile(bd[k], 97.5))} for k in point}
    json.dump({"subgroups": sub, "parity_ci": ci}, open(OUT / "fairness_disability.json", "w"), indent=2)
    for nm, d in sub.items():
        print(f"{nm:14s} n={d['n']:4d} base={d['base_rate']:.3f} sel={d['selection_rate']:.3f} "
              f"AUROC={d['auroc']:.3f} FPR={d['fpr']:.3f} FNR={d['fnr']:.3f} ECE={d['ece']:.3f}")
    print("parity: DP=%.3f [%.3f,%.3f]  FPR=%.3f [%.3f,%.3f]  FNR=%.3f [%.3f,%.3f]" % (
        ci["dp"]["point"], ci["dp"]["ci_lo"], ci["dp"]["ci_hi"],
        ci["fpr_parity"]["point"], ci["fpr_parity"]["ci_lo"], ci["fpr_parity"]["ci_hi"],
        ci["fnr_parity"]["point"], ci["fnr_parity"]["ci_lo"], ci["fnr_parity"]["ci_hi"]))


if __name__ == "__main__":
    main()

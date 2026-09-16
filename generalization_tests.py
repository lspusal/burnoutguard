#!/usr/bin/env python3
"""Follow-up revision analyses:
 (#1) Proper DeLong test (correlated ROC AUCs) on the held-out test set,
      Logistic Regression vs XGBoost, withdrawal target.
 (#2) Cross-presentation generalization: GroupKFold by course presentation
      (leave-presentations-out), XGBoost SMOTE+weight.
 (#4) Reliability (calibration) curve for the ensemble -> figures/calibration.png
Seed=42."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from pipeline import build_features, proba, KEYS
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("figures")

# ---------- DeLong implementation (Sun & Xu 2014, fast version) ----------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_roc_test(y_true, p1, p2):
    """Two-sided DeLong test for two correlated ROC AUCs. Returns (auc1, auc2, p)."""
    order = np.argsort(-np.concatenate([p1[y_true==1], p1[y_true==0]]))  # not used; structured below
    y = y_true.astype(int)
    preds = np.vstack([p1, p2])
    pos = preds[:, y == 1]; neg = preds[:, y == 0]
    m, n, k = pos.shape[1], neg.shape[1], 2
    tx = np.array([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([_compute_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    l = np.array([1, -1])
    var = l @ cov @ l
    from scipy import stats
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p), float(z)

def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)

    # ---- (#1) DeLong: LR vs XGBoost on held-out test ----
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    p_lr = proba("logistic_regression", Xtr, ytr, Xte, "smote_weight")
    p_xgb = proba("xgboost", Xtr, ytr, Xte, "smote_weight")
    a_lr, a_xgb, p_delong, z = delong_roc_test(yte.values, p_lr, p_xgb)
    print(f"[#1 DeLong] LR AUC={a_lr:.4f}  XGB AUC={a_xgb:.4f}  z={z:.3f}  p={p_delong:.3e}")
    json.dump({"lr_auc": a_lr, "xgb_auc": a_xgb, "z": z, "p_value": p_delong},
              open(OUT / "delong_lr_vs_xgb.json", "w"), indent=2)

    # ---- (#2) Cross-presentation generalization (GroupKFold) ----
    groups = (df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)).values
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        p = proba("xgboost", X.iloc[tr], y.iloc[tr], X.iloc[te], "smote_weight")
        aucs.append(roc_auc_score(y.iloc[te], p))
    aucs = np.array(aucs)
    print(f"[#2 cross-presentation] {n_groups} presentations, GroupKFold-5: "
          f"AUROC={aucs.mean():.4f}±{aucs.std(ddof=1):.4f}  folds={np.round(aucs,4).tolist()}")
    json.dump({"n_presentations": int(n_groups), "auroc_mean": float(aucs.mean()),
               "auroc_std": float(aucs.std(ddof=1)), "folds": aucs.tolist()},
              open(OUT / "cross_presentation_cv.json", "w"), indent=2)

    # ---- (#4) Calibration curve (ensemble) ----
    p_lgbm = proba("lightgbm", Xtr, ytr, Xte, "smote_weight")
    ens = 0.5 * p_xgb + 0.5 * p_lgbm
    frac_pos, mean_pred = calibration_curve(yte, ens, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, "o-", color="#1f77b4", lw=2, label="Ensemble")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed withdrawal frequency")
    plt.title("Reliability diagram — withdrawal prediction (held-out test)")
    plt.legend(loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "calibration.png", dpi=200); plt.close()
    print("[#4] saved figures/calibration.png")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Additional analyses:
 (#4) Pairwise DeLong tests between all models + bootstrap AUROC-difference CIs.
 (#2) Bootstrap 95% CIs for fairness metrics (DP, FPR parity, FNR parity);
      intersectional subgroups (Gender x Age, Gender x IMD);
      per-group reliability (calibration) curves -> figures/calibration_groups.png
Seed=42, held-out test set."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from pipeline import build_features, proba
from generalization_tests import delong_roc_test
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("figures")
rng = np.random.RandomState(SEED)
MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]
NB = 1000


def dp_diff(pred, a):
    r = [pred[a == g].mean() for g in np.unique(a) if (a == g).sum() > 0]
    return max(r) - min(r)

def rate_diff(y, pred, a, kind):
    vals = []
    for g in np.unique(a):
        m = a == g
        yt, pr = y[m], pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
        if kind == "fpr":
            v = fp / (fp + tn) if (fp + tn) else np.nan
        else:
            v = fn / (fn + tp) if (fn + tp) else np.nan
        if not np.isnan(v):
            vals.append(v)
    return (max(vals) - min(vals)) if len(vals) > 1 else np.nan


def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
    yv = yte.values
    probs = {m: proba(m, Xtr, ytr, Xte, "smote_weight") for m in MODELS}
    probs["ensemble"] = 0.5 * probs["xgboost"] + 0.5 * probs["lightgbm"]
    order = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "ensemble"]

    # ---- (#4) pairwise DeLong + bootstrap AUROC-difference CIs ----
    delong_p = {}; aucs = {m: roc_auc_score(yv, probs[m]) for m in order}
    for i, a in enumerate(order):
        for b in order[i+1:]:
            _, _, p, z = delong_roc_test(yv, probs[a], probs[b])
            delong_p[f"{a}_vs_{b}"] = {"auc_a": aucs[a], "auc_b": aucs[b], "z": z, "p": p}
    # bootstrap AUROC diff CIs for key pairs
    pairs = [("xgboost", "logistic_regression"), ("xgboost", "random_forest"),
             ("ensemble", "xgboost"), ("xgboost", "decision_tree")]
    boot_diff = {}
    n = len(yv)
    for a, b in pairs:
        diffs = []
        for _ in range(NB):
            s = rng.randint(0, n, n)
            if len(np.unique(yv[s])) < 2: continue
            diffs.append(roc_auc_score(yv[s], probs[a][s]) - roc_auc_score(yv[s], probs[b][s]))
        diffs = np.array(diffs)
        boot_diff[f"{a}_minus_{b}"] = {"mean": float(diffs.mean()),
            "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}
    json.dump({"aucs": aucs, "delong_pairwise": delong_p, "bootstrap_auroc_diff": boot_diff},
              open(OUT / "model_stats.json", "w"), indent=2)
    print("[#4] DeLong pairwise + bootstrap AUROC diffs saved")
    for k, v in boot_diff.items():
        print(f"     {k}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    # ---- (#2) fairness metrics with bootstrap CIs ----
    H = df.iloc[ite][helpers].reset_index(drop=True)
    gender = (Xte["gender_M"].values == 1).astype(int)            # 1=male
    med = H["_imd_num"].median()
    imd = (H["_imd_num"].values <= med).astype(int)               # 1=more deprived
    age3 = np.where(H["_age55"].values == 1, 2, np.where(H["_age3555"].values == 1, 1, 0))
    pred = (probs["xgboost"] >= 0.5).astype(int)
    attrs = {"gender": gender, "age": age3, "imd_band": imd}

    fair_ci = {}
    for name, a in attrs.items():
        point = {"dp": dp_diff(pred, a), "fpr_parity": rate_diff(yv, pred, a, "fpr"),
                 "fnr_parity": rate_diff(yv, pred, a, "fnr")}
        bd = {"dp": [], "fpr_parity": [], "fnr_parity": []}
        for _ in range(NB):
            s = rng.randint(0, n, n)
            bd["dp"].append(dp_diff(pred[s], a[s]))
            bd["fpr_parity"].append(rate_diff(yv[s], pred[s], a[s], "fpr"))
            bd["fnr_parity"].append(rate_diff(yv[s], pred[s], a[s], "fnr"))
        fair_ci[name] = {k: {"point": float(point[k]),
            "ci_lo": float(np.nanpercentile(bd[k], 2.5)),
            "ci_hi": float(np.nanpercentile(bd[k], 97.5))} for k in point}
    json.dump(fair_ci, open(OUT / "fairness_bootstrap_ci.json", "w"), indent=2)
    print("[#2] fairness bootstrap CIs saved")
    for nm, d in fair_ci.items():
        print(f"     {nm}: DP={d['dp']['point']:.3f} [{d['dp']['ci_lo']:.3f},{d['dp']['ci_hi']:.3f}] "
              f"FPR={d['fpr_parity']['point']:.3f} [{d['fpr_parity']['ci_lo']:.3f},{d['fpr_parity']['ci_hi']:.3f}] "
              f"FNR={d['fnr_parity']['point']:.3f} [{d['fnr_parity']['ci_lo']:.3f},{d['fnr_parity']['ci_hi']:.3f}]")

    # ---- (#2) intersectional subgroups ----
    def subg(mask):
        m = mask
        if m.sum() < 20 or len(np.unique(yv[m])) < 2:
            return {"n": int(m.sum()), "note": "too small"}
        yt, pr, pb = yv[m], pred[m], probs["xgboost"][m]
        tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
        return {"n": int(m.sum()), "base_rate": float(yt.mean()), "selection_rate": float(pr.mean()),
                "auroc": float(roc_auc_score(yt, pb)),
                "fpr": float(fp/(fp+tn) if fp+tn else np.nan),
                "fnr": float(fn/(fn+tp) if fn+tp else np.nan)}
    inter = {}
    glab = {1: "male", 0: "female"}
    alab = {0: "<35", 1: "35-55", 2: "55+"}
    ilab = {1: "more_deprived", 0: "less_deprived"}
    gxa = {f"{glab[g]}_{alab[al]}": subg((gender == g) & (age3 == al)) for g in (1, 0) for al in (0, 1, 2)}
    gxi = {f"{glab[g]}_{ilab[il]}": subg((gender == g) & (imd == il)) for g in (1, 0) for il in (1, 0)}
    sel = [v["selection_rate"] for v in gxa.values() if "selection_rate" in v]
    inter = {"gender_x_age": gxa, "gender_x_imd": gxi,
             "gender_x_age_dp_range": float(max(sel) - min(sel)) if sel else None}
    json.dump(inter, open(OUT / "fairness_intersectional.json", "w"), indent=2)
    print("[#2] intersectional subgroups saved; gender x age DP range =",
          round(inter["gender_x_age_dp_range"], 3) if inter["gender_x_age_dp_range"] else None)

    # ---- (#2) per-group calibration curves ----
    pb = probs["xgboost"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("Gender", {"Male": gender == 1, "Female": gender == 0}),
              ("Age band", {"<35": age3 == 0, "35--55": age3 == 1}),
              ("IMD band", {"More deprived": imd == 1, "Less deprived": imd == 0})]
    for ax, (title, groups) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        for lab, m in groups.items():
            try:
                fp_, mp_ = calibration_curve(yv[m], pb[m], n_bins=5, strategy="quantile")
                ax.plot(mp_, fp_, "o-", lw=1.8, label=f"{lab} (n={int(m.sum())})")
            except Exception:
                pass
        ax.set_title(title); ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed withdrawal frequency"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Per-group calibration (held-out test set)")
    plt.tight_layout(); plt.savefig(FIG / "calibration_groups.png", dpi=200); plt.close()
    print("[#2] saved figures/calibration_groups.png")


if __name__ == "__main__":
    main()

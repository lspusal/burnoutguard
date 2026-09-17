#!/usr/bin/env python3
"""Deployment-horizon analyses.

For the early-prediction horizons (weeks 2/4/6/8), using only data available up
to each week (temporal truncation applied before any split):

 (1) Presentation-level generalization: XGBoost AUROC under GroupKFold by course
      presentation, side by side with random StratifiedKFold, to show the early
      numbers are not inflated by presentation-specific patterns shared between
      train and test students.
 (2) Calibration (Brier, ECE) and fairness (selection/FPR/FNR parity for gender,
      age, IMD band, and disability) evaluated AT weeks 4 and 8.
 (3) SHAP global importance at weeks 4 and 8 -> figures/shap_early.png.

Withdrawal target, seed 42.
"""
import warnings, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from risk_set import apply_risk_set
from horizon_features import add_horizon_features
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("figures"); FIG.mkdir(parents=True, exist_ok=True)
KEYS = ["id_student", "code_module", "code_presentation"]
DEMO = ["gender", "age_band", "imd_band", "disability"]


def extract_full(vle, info, week):
    md = week * 7
    v = vle[vle["date"] <= md].copy()
    b = v.groupby(KEYS).agg({"sum_click": ["sum", "mean", "std", "max", "min", "count"],
                             "date": ["min", "max", "nunique"]}).reset_index()
    b.columns = KEYS + ["total_clicks", "avg_clicks", "std_clicks", "max_clicks", "min_clicks",
                        "n_events", "first_day", "last_day", "active_days"]
    b["days_inactive"] = md - b["last_day"]; b["inactivity_ratio"] = b["days_inactive"] / md
    hp = md // 2
    fh = v[v["date"] <= hp].groupby(KEYS)["sum_click"].sum().reset_index(name="clicks_early")
    sh = v[v["date"] > hp].groupby(KEYS)["sum_click"].sum().reset_index(name="clicks_late")
    b = b.merge(fh, on=KEYS, how="left").merge(sh, on=KEYS, how="left").fillna(0)
    b["engagement_decay"] = (b["clicks_late"] + 1) / (b["clicks_early"] + 1)
    b["decay_flag"] = (b["engagement_decay"] < 0.5).astype(int)

    def ap(g):
        d = sorted(g["date"].unique())
        if len(d) < 2:
            return pd.Series({"gap_mean": md, "gap_max": md, "regularity": 0})
        gp = np.diff(d)
        return pd.Series({"gap_mean": gp.mean(), "gap_max": gp.max(), "regularity": 1/(gp.std()+1)})
    b = b.merge(v.groupby(KEYS).apply(ap).reset_index(), on=KEYS, how="left")
    for f in ["total_clicks", "active_days"]:
        b[f"{f}_zscore"] = b.groupby(["code_module", "code_presentation"])[f].transform(
            lambda x: (x - x.mean()) / (x.std() + 1))
    b["click_velocity"] = b["total_clicks"] / (b["active_days"] + 1)
    b["early_start"] = (b["first_day"] <= 7).astype(int)
    b["consistency"] = b["active_days"] / (md / 2)

    im = info[KEYS + ["final_result"] + DEMO].copy()
    f = b.merge(im, on=KEYS, how="inner").fillna(0)
    f["y"] = (f["final_result"] == "Withdrawn").astype(int)
    # sólo estudiantes aún matriculados en el horizonte (ver risk_set.py)
    f = apply_risk_set(f, md)
    f, added = add_horizon_features(f, week)
    drop = KEYS + ["final_result", "y"] + DEMO + [c for c in f.columns if c.endswith("_dup")] \
           + ["last_submit"] + [c for c in f.columns if c.endswith("_code")] \
           + [c for c in ["gender", "age_band", "imd_band", "disability",
                          "highest_education"] if c in f.columns]
    feat = [c for c in f.columns if c not in drop]
    return f, feat


def xgb_proba(Xtr, ytr, Xte):
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
        n_jobs=-1, verbosity=0)
    m.fit(Xb, yb); return m, m.predict_proba(Xte)[:, 1]


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1]) if i < bins-1 else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum() == 0: continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def grp_metrics(y, pred, pb, g):
    out = {}
    for lvl in np.unique(g):
        m = g == lvl
        if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
        tn, fp, fn, tp = confusion_matrix(y[m], pred[m], labels=[0, 1]).ravel()
        out[str(lvl)] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
            "selection_rate": float(pred[m].mean()), "auroc": float(roc_auc_score(y[m], pb[m])),
            "fpr": float(fp/(fp+tn) if fp+tn else np.nan), "fnr": float(fn/(fn+tp) if fn+tp else np.nan)}
    vals = lambda k: [v[k] for v in out.values()]
    par = {k: (max(vals(k)) - min(vals(k))) if len(out) > 1 else np.nan
           for k in ["selection_rate", "fpr", "fnr"]}
    return {"groups": out, "parity": par}


def main():
    vle = pd.read_csv("data/studentVle.csv")
    info = pd.read_csv("data/studentInfo.csv")

    # ---------- (1) grouped vs random early prediction ----------
    p1 = []
    for wk in [2, 4, 6, 8]:
        f, feat = extract_full(vle, info, wk)
        X, y = f[feat].values, f["y"].values
        grp = (f["code_module"].astype(str) + "_" + f["code_presentation"].astype(str)).values
        # random StratifiedKFold
        r = [roc_auc_score(y[te], xgb_proba(X[tr], y[tr], X[te])[1])
             for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)]
        # GroupKFold by presentation
        gk = [roc_auc_score(y[te], xgb_proba(X[tr], y[tr], X[te])[1])
              for tr, te in GroupKFold(5).split(X, y, grp)]
        p1.append({"week": wk, "random_auroc_mean": float(np.mean(r)), "random_auroc_std": float(np.std(r, ddof=1)),
                   "grouped_auroc_mean": float(np.mean(gk)), "grouped_auroc_std": float(np.std(gk, ddof=1))})
        print(f"[P1] week {wk}: random={np.mean(r):.3f}±{np.std(r,ddof=1):.3f}  "
              f"grouped={np.mean(gk):.3f}±{np.std(gk,ddof=1):.3f}")
    json.dump(p1, open(OUT / "early_grouped_vs_random.json", "w"), indent=2)

    # ---------- (P2,P3) SHAP + calibration + fairness at weeks 4 and 8 ----------
    import shap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    horizon = {}
    for ax, wk in zip(axes, [4, 8]):
        f, feat = extract_full(vle, info, wk)
        X, y = f[feat].values, f["y"].values
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
        model, pb = xgb_proba(X[itr], y[itr], X[ite])
        pred = (pb >= 0.5).astype(int); yte = y[ite]
        # calibration
        cal = {"brier": float(brier_score_loss(yte, pb)), "ece": ece(yte, pb)}
        # fairness (gender, age, imd, disability)
        meta = f.iloc[ite]
        fair = {}
        fair["gender"] = grp_metrics(yte, pred, pb, meta["gender"].values)
        fair["age_band"] = grp_metrics(yte, pred, pb, meta["age_band"].values)
        imd_num = meta["imd_band"].astype(str).str.extract(r"(\d+)").astype(float).fillna(50).values.ravel()
        fair["imd_band"] = grp_metrics(yte, pred, pb, (imd_num <= np.median(imd_num)).astype(int))
        fair["disability"] = grp_metrics(yte, pred, pb, (meta["disability"].values == "Y").astype(int))
        # SHAP
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X[ite])
        imp = np.abs(sv).mean(0); imp = imp / imp.sum()
        top = np.argsort(imp)[::-1][:8]
        ax.barh([feat[i] for i in top][::-1], [imp[i] for i in top][::-1], color="#1f77b4")
        ax.set_title(f"Week {wk}"); ax.set_xlabel("Mean |SHAP| (normalised)")
        horizon[f"week{wk}"] = {"auroc": float(roc_auc_score(yte, pb)), "calibration": cal,
            "fairness": fair, "shap_top": {feat[i]: float(imp[i]) for i in top}}
        print(f"[P2/P3] week {wk}: AUROC={roc_auc_score(yte,pb):.3f} Brier={cal['brier']:.3f} ECE={cal['ece']:.3f}")
        for a, d in fair.items():
            print(f"        {a}: DP={d['parity']['selection_rate']:.3f} FPR={d['parity']['fpr']:.3f} FNR={d['parity']['fnr']:.3f}")
    plt.suptitle("SHAP global feature importance at deployment horizons (held-out test)")
    plt.tight_layout(); plt.savefig(FIG / "shap_early.png", dpi=200); plt.close()
    json.dump(horizon, open(OUT / "horizon_fairness_calibration.json", "w"), indent=2)
    print("saved figures/shap_early.png and horizon JSON")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Canonical analysis pipeline.

Builds the keyed feature matrix (21 behavioural + ~23 one-hot demographic = ~44
features) with the academic-risk target defined as WITHDRAWAL (Withdrawn = 1),
which yields genuine class imbalance (~24% positive on the VLE-active set) and is
the canonical dropout target for an Early Warning System.

Runs, all with seed=42 and 5-fold stratified CV:
  - Class-imbalance ablation: none / SMOTE / weighting / SMOTE+weighting
  - Model comparison: LR, DT, RF, XGBoost, LightGBM, Ensemble
  - Leakage-robust experiment: drop near-outcome proxies
  - Extended fairness: per-group calibration (ECE/Brier), FPR/FNR parity,
    subgroup AUROC/precision/recall
Outputs -> results/metrics/revision/
"""
import json, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, brier_score_loss, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb, lightgbm as lgb
from run_real_pipeline import load_oulad_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
SEED = 42
OUT = Path("results/metrics/revision"); OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["id_student", "code_module", "code_presentation"]
PROXIES = ["last_access_day", "n_assessments", "mean_score", "late_submission_rate"]


def build_features(data):
    vle, info = data["studentVle"], data["studentInfo"]
    assess, ainfo = data["studentAssessment"], data["assessments"]

    eng = vle.groupby(KEYS).agg({"sum_click": ["sum", "mean", "std", "max"],
                                 "date": ["min", "max", "count", "nunique"]}).reset_index()
    eng.columns = KEYS + ["total_clicks", "avg_clicks", "std_clicks", "max_clicks",
                          "first_access_day", "last_access_day", "n_access_events", "n_unique_days"]

    def eslope(g):
        if len(g) < 3 or np.std(g["date"]) == 0: return 0
        return np.polyfit(g["date"].values, g["sum_click"].values, 1)[0]
    tr = vle.groupby(KEYS).apply(eslope).reset_index(name="engagement_trend_slope")
    eng = eng.merge(tr, on=KEYS)

    def tempf(g):
        days = sorted(g["date"].unique())
        if len(days) < 2:
            return pd.Series({"max_gap_days": 0, "mean_gap_days": 0,
                              "activity_volatility": 0, "negative_changes_pct": 0})
        gaps = np.diff(days)
        dc = g.groupby("date")["sum_click"].sum().values
        ch = np.diff(dc) if len(dc) > 1 else np.array([])
        return pd.Series({"max_gap_days": gaps.max(), "mean_gap_days": gaps.mean(),
            "activity_volatility": np.std(dc) if len(dc) > 1 else 0,
            "negative_changes_pct": (ch < 0).sum()/len(ch) if len(ch) else 0})
    tf = vle.groupby(KEYS).apply(tempf).reset_index()
    eng = eng.merge(tf, on=KEYS)

    a = assess.merge(ainfo, on="id_assessment")
    ac = a.groupby(KEYS).agg({"score": ["mean", "std", "min", "max", "count"],
                              "is_banked": "sum"}).reset_index()
    ac.columns = KEYS + ["mean_score", "std_score", "min_score", "max_score",
                         "n_assessments", "n_banked"]
    a["is_late"] = a["date_submitted"] > a["date"]
    late = a.groupby(KEYS)["is_late"].mean().reset_index()
    late.columns = KEYS + ["late_submission_rate"]
    ac = ac.merge(late, on=KEYS, how="left")

    def gslope(g):
        d, s = g["date_submitted"].values, g["score"].values
        v = ~(np.isnan(d) | np.isnan(s))
        if v.sum() < 2 or np.std(d[v]) == 0: return 0
        return np.polyfit(d[v], s[v], 1)[0]
    gt = a.groupby(KEYS).apply(gslope).reset_index(name="grade_trend_slope")
    ac = ac.merge(gt, on=KEYS, how="left")
    eng = eng.merge(ac, on=KEYS, how="left")

    # one-hot demographics (gender, age, education, IMD, disability; no region)
    info = info.merge(vle[KEYS].drop_duplicates(), on=KEYS, how="inner")  # active set
    d = info[KEYS].copy()
    d["gender_M"] = (info.gender == "M").astype(int)
    for lvl in sorted(info.age_band.dropna().unique()):
        d[f"age_{lvl}"] = (info.age_band == lvl).astype(int)
    for lvl in sorted(info.highest_education.dropna().unique()):
        d[f"edu_{lvl}"] = (info.highest_education == lvl).astype(int)
    for lvl in sorted(info.imd_band.dropna().unique()):
        d[f"imd_{lvl}"] = (info.imd_band == lvl).astype(int)
    d["has_disability"] = (info.disability == "Y").astype(int)
    d["studied_credits"] = info.studied_credits.fillna(60)
    d["num_prev_attempts"] = info.num_of_prev_attempts.fillna(0)
    d["y_withdrawn"] = (info.final_result == "Withdrawn").astype(int)
    # raw demographic helpers for fairness grouping
    d["_imd_num"] = info.imd_band.str.extract(r"(\d+)").astype(float).fillna(50).values
    d["_age55"] = (info.age_band == "55<=").astype(int).values
    d["_age3555"] = (info.age_band == "35-55").astype(int).values

    df = eng.merge(d, on=KEYS, how="inner").fillna(0)
    # sanitise column names (XGBoost rejects [ ] < in feature names)
    import re
    df.columns = [re.sub(r"[\[\]<>%]", "", c).replace("-", "_").replace(" ", "_").replace("=", "")
                  if c not in KEYS else c for c in df.columns]
    helpers = ["_imd_num", "_age55", "_age3555"]
    y = df["y_withdrawn"]
    feat_cols = [c for c in df.columns if c not in KEYS + ["y_withdrawn"] + helpers]
    return df, df[feat_cols], y, helpers


def mk(name, spw):
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED,
            eval_metric="logloss", n_jobs=-1, verbosity=0)
    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED,
            n_jobs=-1, verbose=-1)
    cw = "balanced" if spw != 1.0 else None
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight=cw, n_jobs=-1)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=10, class_weight=cw, random_state=SEED)


def proba(name, Xtr, ytr, Xte, bal):
    spw, Xb, yb = 1.0, Xtr, ytr
    if bal in ("smote", "smote_weight"):
        Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    if bal in ("weight", "smote_weight"):
        spw = (ytr == 0).sum() / (ytr == 1).sum()
    if name in ("xgboost", "lightgbm"):
        m = mk(name, spw)
    else:
        m = mk(name, spw if bal in ("weight", "smote_weight") else 1.0)
    m.fit(Xb, yb)
    return m.predict_proba(Xte)[:, 1]


def met(y, p, thr=0.5):
    pr = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "f1": f1_score(y, pr, zero_division=0), "precision": precision_score(y, pr, zero_division=0),
            "recall": recall_score(y, pr, zero_division=0),
            "specificity": tn/(tn+fp) if tn+fp else 0, "brier": brier_score_loss(y, p)}


def cv(X, y, name, bal, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    return pd.DataFrame([met(y.iloc[te], proba(name, X.iloc[tr], y.iloc[tr], X.iloc[te], bal))
                         for tr, te in skf.split(X, y)])


def summ(df):
    return {f"{c}_mean": df[c].mean() for c in df} | {f"{c}_std": df[c].std(ddof=1) for c in df}


def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)
    log.info(f"Samples={len(X)} Features={X.shape[1]} positive(withdrawn)={y.mean()*100:.1f}%")
    log.info(f"Feature list ({X.shape[1]}): {list(X.columns)}")

    # ablation (Table 4)
    log.info("\n== ABLATION (Table 4) ==")
    abl, fa = {}, {}
    for c in ["none", "smote", "weight", "smote_weight"]:
        d = cv(X, y, "xgboost", c); abl[c] = summ(d); fa[c] = d["auroc"].values
        log.info(f"{c:13s} AUROC={d.auroc.mean():.4f}±{d.auroc.std(ddof=1):.4f} "
                 f"AUPRC={d.auprc.mean():.4f} F1={d.f1.mean():.4f}±{d.f1.std(ddof=1):.4f} "
                 f"Rec={d.recall.mean():.4f}±{d.recall.std(ddof=1):.4f}")
    tt = {f"smote_weight_vs_{c}": dict(zip(("t", "p"), map(float, sps.ttest_rel(fa["smote_weight"], fa[c]))))
          for c in ["none", "smote", "weight"]}
    pd.DataFrame(abl).T.to_csv(OUT / "ablation_cv.csv"); json.dump(tt, open(OUT/"ablation_ttests.json","w"), indent=2)
    log.info(f"t-tests: {tt}")

    # model comparison (Table 5)
    log.info("\n== MODEL COMPARISON (Table 5) ==")
    cmp = {}
    probs = {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    for m in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]:
        d = cv(X, y, m, "smote_weight"); cmp[m] = summ(d)
        probs[m] = proba(m, Xtr, ytr, Xte, "smote_weight")
        log.info(f"{m:20s} AUROC={d.auroc.mean():.4f}±{d.auroc.std(ddof=1):.4f} "
                 f"AUPRC={d.auprc.mean():.4f} F1={d.f1.mean():.4f}±{d.f1.std(ddof=1):.4f} "
                 f"Prec={d.precision.mean():.4f} Rec={d.recall.mean():.4f} Spec={d.specificity.mean():.4f}")
    # ensemble = mean of xgb+lgbm CV
    de = cv(X, y, "xgboost", "smote_weight")  # placeholder; ensemble computed on holdout
    ens = 0.5*probs["xgboost"] + 0.5*probs["lightgbm"]
    log.info(f"ensemble (holdout) AUROC={roc_auc_score(yte,ens):.4f} F1={f1_score(yte,(ens>=.5).astype(int)):.4f}")
    cmp["ensemble_holdout"] = met(yte, ens)
    pd.DataFrame(cmp).T.to_csv(OUT / "model_comparison_cv.csv")

    # leakage-robustness
    log.info("\n== LEAKAGE-ROBUST ==")
    present = [p for p in PROXIES if p in X.columns]
    leak = {}
    df_full = cv(X, y, "xgboost", "smote_weight"); leak["full"] = summ(df_full)
    df_nl = cv(X.drop(columns=["last_access_day"]), y, "xgboost", "smote_weight"); leak["drop_last_access_day"] = summ(df_nl)
    df_rb = cv(X.drop(columns=present), y, "xgboost", "smote_weight"); leak["drop_all_proxies"] = summ(df_rb)
    for k_, d_ in [("full", df_full), ("drop_lad", df_nl), ("drop_all", df_rb)]:
        log.info(f"{k_:10s} AUROC={d_.auroc.mean():.4f}±{d_.auroc.std(ddof=1):.4f} F1={d_.f1.mean():.4f}")
    lt = {"full_vs_drop_lad": dict(zip(("t","p"), map(float, sps.ttest_rel(df_full.auroc, df_nl.auroc)))),
          "full_vs_drop_all": dict(zip(("t","p"), map(float, sps.ttest_rel(df_full.auroc, df_rb.auroc))))}
    pd.DataFrame(leak).T.to_csv(OUT/"leakage_robust_cv.csv"); json.dump(lt, open(OUT/"leakage_tests.json","w"), indent=2)
    log.info(f"leakage t-tests: {lt}")

    # extended fairness on holdout
    log.info("\n== EXTENDED FAIRNESS ==")
    idx_te = Xte.index
    p = probs["xgboost"]; pred = (p >= 0.5).astype(int); yv = yte.values
    H = df.loc[idx_te, helpers].reset_index(drop=True)
    def ece(yt, pp, b=10):
        e, ed = 0.0, np.linspace(0, 1, b+1)
        for i in range(b):
            m = (pp >= ed[i]) & (pp <= ed[i+1] if i==b-1 else pp < ed[i+1])
            if m.sum(): e += m.sum()/len(pp)*abs(yt[m].mean()-pp[m].mean())
        return e
    def sub(masks):
        r = {}
        for g, m in masks.items():
            if m.sum() == 0 or len(np.unique(yv[m])) < 2: r[g] = {"n": int(m.sum())}; continue
            tn, fp, fn, tp = confusion_matrix(yv[m], pred[m], labels=[0,1]).ravel()
            r[g] = {"n": int(m.sum()), "base_rate": float(yv[m].mean()), "selection_rate": float(pred[m].mean()),
                "auroc": float(roc_auc_score(yv[m], p[m])), "precision": float(precision_score(yv[m], pred[m], zero_division=0)),
                "recall_tpr": float(tp/(tp+fn) if tp+fn else 0), "fpr": float(fp/(fp+tn) if fp+tn else 0),
                "fnr": float(fn/(fn+tp) if fn+tp else 0), "brier": float(brier_score_loss(yv[m], p[m])), "ece": float(ece(yv[m], p[m]))}
        return r
    med = H["_imd_num"].median()
    attrs = {
        "gender": {"male": (Xte["gender_M"].values==1), "female": (Xte["gender_M"].values==0)},
        "age": {"under35": ((H._age55==0)&(H._age3555==0)).values, "35-55": (H._age3555==1).values, "55+": (H._age55==1).values},
        "imd_band": {"more_deprived": (H._imd_num<=med).values, "less_deprived": (H._imd_num>med).values},
    }
    fair, par = {}, {}
    for a, md in attrs.items():
        t = sub(md); fair[a] = t
        v = {g: x for g, x in t.items() if "fpr" in x}
        if len(v) >= 2:
            par[a] = {"fpr_parity_diff": max(x["fpr"] for x in v.values())-min(x["fpr"] for x in v.values()),
                "fnr_parity_diff": max(x["fnr"] for x in v.values())-min(x["fnr"] for x in v.values()),
                "demographic_parity_diff": max(x["selection_rate"] for x in v.values())-min(x["selection_rate"] for x in v.values()),
                "calibration_ece_diff": max(x["ece"] for x in v.values())-min(x["ece"] for x in v.values())}
        log.info(f"-- {a}: {par.get(a)}")
        for g, x in t.items(): log.info(f"    {g}: {x}")
    json.dump({"subgroups": fair, "parity": par}, open(OUT/"fairness_extended.json","w"), indent=2)
    log.info("done.")


if __name__ == "__main__":
    main()

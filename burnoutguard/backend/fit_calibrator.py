#!/usr/bin/env python3
"""Entrena el modelo del backend y ajusta su calibrador post-hoc.

El modelo se entrena con remuestreo (SMOTE) y pesos de clase, lo que infla las
probabilidades. Sin calibrar, la puntuación ordena bien a los estudiantes pero no
puede leerse como probabilidad, que es justo lo que muestra un panel docente.
Aquí se ajusta un calibrador isotónico sobre una porción de validación retenida y
se guardan modelo y calibrador en models/.

Las features se construyen desde OULAD con truncamiento temporal en la semana
indicada y en el MISMO orden que backend/main.py:BurnoutPredictor.extract_features

    python fit_calibrator.py --data ../../data --week 8

Semilla 42.
"""
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

SEED = 42
KEYS = ["id_student", "code_module", "code_presentation"]
FEATURES = ["total_clicks", "days_since_last_access", "active_days", "avg_clicks_per_day",
            "avg_grade", "grade_diff", "completion_rate", "days_enrolled", "week_of_course"]


def ece(y, p, bins=10):
    y, p, e = np.asarray(y), np.asarray(p), 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def build(data_dir, week):
    md = week * 7
    vle = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    info = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    sa = pd.read_csv(os.path.join(data_dir, "studentAssessment.csv"))
    asm = pd.read_csv(os.path.join(data_dir, "assessments.csv"))

    v = vle[vle["date"] <= md]
    eng = v.groupby(KEYS).agg(total_clicks=("sum_click", "sum"),
                              last_day=("date", "max"),
                              active_days=("date", "nunique")).reset_index()
    eng["days_since_last_access"] = md - eng["last_day"]
    eng["avg_clicks_per_day"] = eng["total_clicks"] / eng["active_days"].clip(lower=1)

    a = sa.merge(asm, on="id_assessment")
    a = a[a["date_submitted"] <= md]
    gr = a.groupby(KEYS).agg(avg_grade=("score", "mean"),
                             assignments_completed=("score", "size")).reset_index()
    # asignaciones ya programadas a esa altura del curso
    tot = (asm[asm["date"] <= md].groupby(["code_module", "code_presentation"])
           .size().reset_index(name="assignments_total"))

    df = eng.merge(gr, on=KEYS, how="left").merge(tot, on=["code_module", "code_presentation"], how="left")
    df["avg_grade"] = df["avg_grade"].fillna(0.0)
    df["assignments_completed"] = df["assignments_completed"].fillna(0)
    df["assignments_total"] = df["assignments_total"].fillna(1).clip(lower=1)
    df["completion_rate"] = df["assignments_completed"] / df["assignments_total"]
    course_avg = (df.groupby(["code_module", "code_presentation"])["avg_grade"]
                  .transform("mean"))
    df["grade_diff"] = df["avg_grade"] - course_avg
    df["days_enrolled"] = md
    df["week_of_course"] = week

    df = df.merge(info[KEYS + ["final_result"]], on=KEYS, how="inner")
    y = (df["final_result"] == "Withdrawn").astype(int).values
    return df[FEATURES].fillna(0).values.astype(float), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--week", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models"))
    args = ap.parse_args()

    X, y = build(args.data, args.week)
    print(f"n={len(y)}  features={X.shape[1]}  withdrawal_rate={y.mean():.3f}  week={args.week}")

    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    isub, ival = train_test_split(itr, test_size=0.2, stratify=y[itr], random_state=SEED)

    spw = (y[isub] == 0).sum() / (y[isub] == 1).sum()
    Xb, yb = SMOTE(random_state=SEED).fit_resample(X[isub], y[isub])
    model = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED,
                              eval_metric="logloss", n_jobs=-1, verbosity=0).fit(Xb, yb)

    p_val = model.predict_proba(X[ival])[:, 1]
    p_te = model.predict_proba(X[ite])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(p_val, y[ival])
    p_cal = cal.predict(p_te)

    rep = {"week": args.week, "n": int(len(y)), "withdrawal_rate": float(y.mean()),
           "auroc_raw": float(roc_auc_score(y[ite], p_te)),
           "auroc_calibrated": float(roc_auc_score(y[ite], p_cal)),
           "brier_raw": float(brier_score_loss(y[ite], p_te)),
           "brier_calibrated": float(brier_score_loss(y[ite], p_cal)),
           "ece_raw": ece(y[ite], p_te), "ece_calibrated": ece(y[ite], p_cal)}
    print(json.dumps(rep, indent=2))

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "xgboost_model.joblib"))
    joblib.dump({"calibrator": cal, "method": "isotonic", "report": rep},
                os.path.join(args.out, "calibrator.joblib"))
    json.dump(rep, open(os.path.join(args.out, "calibration_report.json"), "w"), indent=2)
    print(f"saved model + calibrator to {args.out}")


if __name__ == "__main__":
    main()

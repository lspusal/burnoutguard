#!/usr/bin/env python3
"""Fairness-constrained model (Table 9 'Fair Model' columns) recomputed for the
Withdrawn target using fairlearn ExponentiatedGradient with a DemographicParity
constraint. Reports baseline vs fair DP and Equal-Opportunity (TPR) differences
and F1 for gender, age, IMD. Complements the extended subgroup metrics already
in fairness_extended.json."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from revision_pipeline import build_features
from run_real_pipeline import load_oulad_data
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)
SEED = 42
OUT = Path("results/metrics/revision"); OUT.mkdir(parents=True, exist_ok=True)


def dp_diff(pred, a):
    rates = [pred[a == g].mean() for g in np.unique(a)]
    return max(rates) - min(rates)


def eo_diff(y, pred, a):  # equal opportunity = TPR difference
    tprs = []
    for g in np.unique(a):
        m = (a == g) & (y == 1)
        if m.sum() == 0:
            continue
        tprs.append(pred[m].mean())
    return max(tprs) - min(tprs) if len(tprs) > 1 else 0.0


def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)
    H = df[helpers].copy()
    age = np.where(H["_age55"] == 1, 2, np.where(H["_age3555"] == 1, 1, 0))
    sens = {"gender": X["gender_M"].values,
            "age": age,
            "imd_band": (H["_imd_num"].values <= H["_imd_num"].median()).astype(int)}

    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    def base_est():
        return xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED,
            eval_metric="logloss", n_jobs=-1, verbosity=0)

    base = base_est(); base.fit(Xb, yb)
    base_pred = base.predict(Xte)

    rows = {}
    for name, a in sens.items():
        a_te = a[ite]; a_tr = a[itr]
        b_dp = dp_diff(base_pred, a_te); b_eo = eo_diff(yte.values, base_pred, a_te)
        b_f1 = f1_score(yte, base_pred)
        # fairness-constrained (in-processing) on training data
        try:
            mit = ExponentiatedGradient(base_est(), constraints=DemographicParity(), eps=0.02)
            mit.fit(Xtr, ytr, sensitive_features=a_tr)
            f_pred = mit.predict(Xte)
            f_dp = dp_diff(f_pred, a_te); f_eo = eo_diff(yte.values, f_pred, a_te)
            f_f1 = f1_score(yte, f_pred)
        except Exception as e:
            log.warning(f"{name}: mitigator failed ({e})"); f_dp = f_eo = f_f1 = None
        rows[name] = {"baseline_dp": b_dp, "fair_dp": f_dp, "baseline_eo": b_eo,
                      "fair_eo": f_eo, "baseline_f1": b_f1, "fair_f1": f_f1}
        log.info(f"{name:10s} DP {b_dp:.3f}->{f_dp if f_dp is None else round(f_dp,3)} "
                 f"EO {b_eo:.3f}->{f_eo if f_eo is None else round(f_eo,3)} "
                 f"F1 {b_f1:.3f}->{f_f1 if f_f1 is None else round(f_f1,3)}")
    json.dump(rows, open(OUT / "fairness_constrained.json", "w"), indent=2)
    log.info("saved fairness_constrained.json")


if __name__ == "__main__":
    main()

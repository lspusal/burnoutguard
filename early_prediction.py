#!/usr/bin/env python3
"""Early-prediction (Table 8) recomputed with the WITHDRAWN target + bootstrap
95% CIs and consecutive-week bootstrap significance. Reuses the temporal feature
extraction and stacking model from early_model.py; the target is switched
to Withdrawn by relabelling 'Fail'->'Pass' in a copy of studentInfo so that the
reused isin(['Withdrawn','Fail']) reduces to (final_result=='Withdrawn')."""
import warnings, logging, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from early_model import extract_enhanced_features, train_enhanced_model
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(SEED)


def boot_ci(y, p, n=1000):
    y = np.asarray(y); p = np.asarray(p)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), np.array(aucs)


def main():
    vle = pd.read_csv("data/studentVle.csv")
    info = pd.read_csv("data/studentInfo.csv")
    info_w = info.copy()
    info_w["final_result"] = info_w["final_result"].replace({"Fail": "Pass"})  # -> Withdrawn-only target

    rows, prev = [], None
    for wk in [2, 4, 6, 8, 12, 20]:
        X, y = extract_enhanced_features(vle, info_w.copy(), wk)
        if X is None:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
        sc = StandardScaler()
        Xtr_s, Xv_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte)
        model = train_enhanced_model(Xtr_s, ytr, Xv_s, yv)
        proba = model.predict_proba(Xte_s)[:, 1]
        pred = model.predict(Xte_s)
        au = roc_auc_score(yte, proba)
        lo, hi, aucs = boot_ci(yte.values, proba)
        # consecutive-week bootstrap significance (diff of bootstrap AUCs > 0)
        p_inc = None
        if prev is not None and len(prev) == len(aucs):
            p_inc = float((np.array(aucs) <= np.array(prev)).mean())  # P(this <= prev)
        rows.append({"week": wk, "auroc": au, "ci_lo": lo, "ci_hi": hi,
                     "f1": f1_score(yte, pred), "accuracy": accuracy_score(yte, pred),
                     "n_features": X.shape[1], "pos_rate": float(y.mean()),
                     "boot_p_vs_prev": p_inc})
        log.info(f"week {wk:2d}: AUROC={au:.4f} ({lo:.4f}-{hi:.4f}) F1={f1_score(yte,pred):.4f} "
                 f"Acc={accuracy_score(yte,pred):.4f} pos={y.mean()*100:.1f}% p_vs_prev={p_inc}")
        prev = aucs
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "early_prediction_withdrawn.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

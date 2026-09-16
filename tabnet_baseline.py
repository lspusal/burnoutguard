#!/usr/bin/env python3
"""TabNet deep-learning baseline for the withdrawal target, 5-fold stratified CV,
to compare against the gradient-boosting models (Table 5). Runs on GPU.
Uses the same feature matrix (build_features) and SMOTE+CV protocol as the GBMs."""
import warnings, logging, json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pipeline import build_features
from oulad_data import load_oulad_data
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)
SEED = 42
OUT = Path("results"); OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"device={DEV} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) else 0.0)


def fit_tabnet(Xtr, ytr, Xval, yval):
    clf = TabNetClassifier(seed=SEED, device_name=DEV, verbose=0,
                           n_d=32, n_a=32, n_steps=5, gamma=1.5,
                           optimizer_params=dict(lr=2e-2))
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric=["auc"],
            max_epochs=120, patience=20, batch_size=2048, virtual_batch_size=256)
    return clf


def main():
    data = load_oulad_data("data/")
    df, X, y, helpers = build_features(data)
    Xv, yv = X.values.astype(np.float32), y.values.astype(int)
    log.info(f"X={Xv.shape} pos={yv.mean():.3f}")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(skf.split(Xv, yv)):
        Xtr_full, ytr_full = Xv[tr], yv[tr]
        # inner val split for early stopping
        Xi, Xval, yi, yval = train_test_split(Xtr_full, ytr_full, test_size=0.15,
                                              stratify=ytr_full, random_state=SEED)
        sc = StandardScaler().fit(Xi)
        Xi_s, Xval_s, Xte_s = sc.transform(Xi), sc.transform(Xval), sc.transform(Xv[te])
        Xi_r, yi_r = SMOTE(random_state=SEED).fit_resample(Xi_s, yi)
        clf = fit_tabnet(Xi_r.astype(np.float32), yi_r, Xval_s.astype(np.float32), yval)
        p = clf.predict_proba(Xte_s.astype(np.float32))[:, 1]
        m = metrics(yv[te], p); rows.append(m)
        log.info(f"fold {k}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f} rec={m['recall']:.4f}")
    dfm = pd.DataFrame(rows)
    summary = {f"{c}_mean": float(dfm[c].mean()) for c in dfm} | \
              {f"{c}_std": float(dfm[c].std(ddof=1)) for c in dfm}
    summary["device"] = DEV
    json.dump(summary, open(OUT / "tabnet_cv.json", "w"), indent=2)
    log.info("TabNet CV: AUROC=%.4f±%.4f AUPRC=%.4f F1=%.4f±%.4f prec=%.4f rec=%.4f spec=%.4f" % (
        dfm.auroc.mean(), dfm.auroc.std(ddof=1), dfm.auprc.mean(),
        dfm.f1.mean(), dfm.f1.std(ddof=1), dfm.precision.mean(), dfm.recall.mean(), dfm.specificity.mean()))
    print("DONE_TABNET")


if __name__ == "__main__":
    main()

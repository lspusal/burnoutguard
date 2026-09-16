#!/usr/bin/env python3
"""External validation on a second, independent dropout dataset: UCI 'Predict
Students' Dropout and Academic Success' (Realinho et al., Portugal; 4,424
students). SAME model comparison and protocol as the OULAD experiments
(Table 5): Logistic Regression, Decision Tree, Random Forest, TabNet, XGBoost,
LightGBM, and the XGBoost+LightGBM ensemble; SMOTE + class weighting; 5-fold
stratified CV; seed 42. Target: Dropout (1) vs {Enrolled, Graduate} (0)."""
import warnings, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb, lightgbm as lgb
warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("results/metrics"); OUT.mkdir(parents=True, exist_ok=True)

from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=697)
X = ds.data.features.copy()
tgt = ds.data.targets.iloc[:, 0].astype(str).str.strip()
y = (tgt == "Dropout").astype(int).values
for c in X.columns:
    if X[c].dtype == object:
        X[c] = pd.factorize(X[c])[0]
Xv = X.fillna(0).values.astype(float)
print(f"n={len(y)}  features={Xv.shape[1]}  dropout_rate={y.mean():.3f}")

def met(yt, p, thr=0.5):
    pr = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
    return dict(auroc=roc_auc_score(yt, p), auprc=average_precision_score(yt, p),
        f1=f1_score(yt, pr, zero_division=0), precision=precision_score(yt, pr, zero_division=0),
        recall=recall_score(yt, pr, zero_division=0), specificity=tn/(tn+fp) if tn+fp else 0)

def proba(name, Xtr, ytr, Xte):
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    if name == "xgboost":
        m = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED, eval_metric="logloss",
            n_jobs=-1, verbosity=0)
    elif name == "lightgbm":
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1)
    elif name == "random_forest":
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif name == "logistic_regression":
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    elif name == "decision_tree":
        m = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=SEED)
    m.fit(Xb, yb); return m.predict_proba(Xte)[:, 1]

def proba_tabnet(Xtr, ytr, Xte):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StandardScaler().fit(Xtr)
    Xi, Xval, yi, yval = train_test_split(sc.transform(Xtr), ytr, test_size=0.15,
                                          stratify=ytr, random_state=SEED)
    Xi_r, yi_r = SMOTE(random_state=SEED).fit_resample(Xi, yi)
    clf = TabNetClassifier(seed=SEED, device_name=dev, verbose=0, n_d=32, n_a=32, n_steps=5,
                           gamma=1.5, optimizer_params=dict(lr=2e-2))
    clf.fit(Xi_r.astype(np.float32), yi_r, eval_set=[(Xval.astype(np.float32), yval)],
            eval_metric=["auc"], max_epochs=120, patience=20, batch_size=1024, virtual_batch_size=128)
    return clf.predict_proba(sc.transform(Xte).astype(np.float32))[:, 1]

MODELS = ["logistic_regression", "decision_tree", "random_forest", "tabnet", "xgboost", "lightgbm", "ensemble"]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
res = {m: [] for m in MODELS}
for tr, te in skf.split(Xv, y):
    p = {}
    for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]:
        p[name] = proba(name, Xv[tr], y[tr], Xv[te])
    p["tabnet"] = proba_tabnet(Xv[tr], y[tr], Xv[te])
    p["ensemble"] = 0.5 * p["xgboost"] + 0.5 * p["lightgbm"]
    for name in MODELS:
        res[name].append(met(y[te], p[name]))
summary = {"n": int(len(y)), "n_features": int(Xv.shape[1]), "dropout_rate": float(y.mean())}
for name in MODELS:
    d = pd.DataFrame(res[name])
    summary[name] = {c: {"mean": float(d[c].mean()), "std": float(d[c].std(ddof=1))} for c in d}
    print(f"{name:20s} AUROC={d.auroc.mean():.3f}+/-{d.auroc.std(ddof=1):.3f} AUPRC={d.auprc.mean():.3f} "
          f"F1={d.f1.mean():.3f} P={d.precision.mean():.3f} R={d.recall.mean():.3f} S={d.specificity.mean():.3f}")
json.dump(summary, open(OUT / "external_validation_uci.json", "w"), indent=2)
print("saved external_validation_uci.json")

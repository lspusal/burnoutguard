#!/usr/bin/env python3
"""Regenerate the data-dependent figures for the withdrawal target so they
match the revised tables: SHAP summary + bar, model ROC curves, and early-prediction
performance. Saves into figures/. Seed=42."""
import warnings, logging
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb, lightgbm as lgb, shap
from revision_pipeline import build_features, mk, proba
from run_real_pipeline import load_oulad_data
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
SEED = 42
FIG = Path("figures"); FIG.mkdir(parents=True, exist_ok=True)

data = load_oulad_data("data/")
df, X, y, helpers = build_features(data)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---- ROC curves (model comparison) ----
plt.figure(figsize=(7, 6))
names = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
probs = {}
for key, lab in names.items():
    p = proba(key, Xtr, ytr, Xte, "smote_weight"); probs[key] = p
    fpr, tpr, _ = roc_curve(yte, p)
    plt.plot(fpr, tpr, lw=1.8, label=f"{lab} (AUC={roc_auc_score(yte,p):.3f})")
ens = 0.5 * probs["xgboost"] + 0.5 * probs["lightgbm"]
fpr, tpr, _ = roc_curve(yte, ens)
plt.plot(fpr, tpr, lw=2.6, color="black", label=f"Ensemble (AUC={roc_auc_score(yte,ens):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves — withdrawal prediction (held-out test set)")
plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG / "model_roc_curves.png", dpi=200); plt.close()
print("saved model_roc_curves.png")

# ---- SHAP summary + bar ----
Xb, yb = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
spw = (ytr == 0).sum() / (ytr == 1).sum()
m = mk("xgboost", spw); m.fit(Xb, yb)
samp = Xte.sample(min(2000, len(Xte)), random_state=SEED)
sv = shap.TreeExplainer(m).shap_values(samp)
shap.summary_plot(sv, samp, show=False, max_display=12)
plt.title("SHAP summary — withdrawal prediction", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
print("saved shap_summary.png")
shap.summary_plot(sv, samp, plot_type="bar", show=False, max_display=12)
plt.title("SHAP feature importance — withdrawal prediction", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "shap_bar.png", dpi=200, bbox_inches="tight"); plt.close()
print("saved shap_bar.png")

# ---- Early prediction performance ----
ep = pd.read_csv("results/metrics/revision/early_prediction_withdrawn.csv")
plt.figure(figsize=(7.5, 5))
wk = ep["week"].values.astype(float)
plt.plot(wk, ep["auroc"], "o-", lw=2, color="#1f77b4", label="Early prediction (temporally truncated)")
plt.fill_between(wk, ep["ci_lo"], ep["ci_hi"], alpha=0.2, color="#1f77b4")
plt.axhline(0.938, ls="--", color="gray", label="Full-course (upper bound, 0.938)")
plt.axhline(0.7, ls=":", color="red", alpha=0.6, label="Practical-utility threshold")
for x, a in zip(wk, ep["auroc"]):
    plt.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
plt.xlabel("Week of course (data available)"); plt.ylabel("AUROC")
plt.title("Early prediction of withdrawal across time horizons")
plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3); plt.ylim(0.6, 1.0)
plt.tight_layout(); plt.savefig(FIG / "ews_temporal_performance.png", dpi=200); plt.close()
print("saved ews_temporal_performance.png")
print("All figures regenerated.")

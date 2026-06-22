#!/usr/bin/env python3
"""Regenerate the pipeline diagram (figures/pipeline.png) with the corrected
feature count (43, not 45). Matches the original 5-box flow layout."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(parents=True, exist_ok=True)
boxes = [
    ("Data\nCollection", "VLE Logs\nAssessments\nDemographics", "#3FC5B7"),
    ("Feature\nEngineering", "43 Features\n4 Domains", "#42A6CC"),
    ("Model\nTraining", "XGBoost\nLightGBM\nEnsemble", "#93CBA6"),
    ("SHAP\nAnalysis", "Global\nLocal\nDependence", "#D79CE0"),
    ("Risk\nPrediction", "Risk Score\nExplanation\nIntervention", "#FCE08C"),
]
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
ax.set_title("Early Warning System Pipeline", fontsize=20, fontweight="bold", pad=20)
w, h, gap = 15, 11, 4.25
x0 = 4
centers = []
for i, (title, sub, color) in enumerate(boxes):
    x = x0 + i * (w + gap)
    box = FancyBboxPatch((x, 18), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                         linewidth=2, edgecolor="#333333", facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, 18 + h/2, title, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(x + w/2, 14, sub, ha="center", va="top", fontsize=12, color="#444444")
    centers.append((x, x + w))
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(centers[i+1][0] - 0.3, 23.5), xytext=(centers[i][1] + 0.3, 23.5),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
plt.tight_layout()
plt.savefig(FIG / "pipeline.png", dpi=200, bbox_inches="tight")
print("saved figures/pipeline.png")

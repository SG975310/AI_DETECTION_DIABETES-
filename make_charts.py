import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    data = json.load(f)

results = data["model_results"]
roc = data["roc_curves"]
models = list(results.keys())

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- Chart 1: Metric comparison bar chart ----
metrics = ["accuracy", "precision", "recall", "f1_score"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(models))
width = 0.2
colors = ["#2a78d6", "#199e70", "#eda100", "#e34948"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, (m, label) in enumerate(zip(metrics, metric_labels)):
    vals = [results[model][m] for model in models]
    ax.bar(x + i*width, vals, width, label=label, color=colors[i])

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Model Performance Comparison — Diabetes Prediction", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_model_comparison.png", dpi=150)
plt.close()

# ---- Chart 2: ROC curves ----
fig, ax = plt.subplots(figsize=(7, 6))
model_colors = {"Logistic Regression": "#2a78d6", "Random Forest": "#199e70", "XGBoost": "#e34948"}
for model in models:
    fpr = roc[model]["fpr"]
    tpr = roc[model]["tpr"]
    auc = results[model]["roc_auc"]
    ax.plot(fpr, tpr, label=f"{model} (AUC = {auc:.3f})", color=model_colors[model], linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess (AUC = 0.5)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title("ROC Curves — Diabetes Prediction Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart_roc_curves.png", dpi=150)
plt.close()

# ---- Chart 3: Feature importance ----
fi = data["feature_importance"]
feats = list(fi.keys())
vals = list(fi.values())

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feats[::-1], vals[::-1], color="#2a78d6")
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_title("Feature Importance — Random Forest Model", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_feature_importance.png", dpi=150)
plt.close()

print("Charts saved: chart_model_comparison.png, chart_roc_curves.png, chart_feature_importance.png")

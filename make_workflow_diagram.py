"""Generates a simple workflow diagram showing the ML pipeline stages."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IMAGES_DIR = "images"

stages = [
    ("1. Raw Data", "768 patients, 8 features\n(Pima Indians Diabetes Dataset)"),
    ("2. Clean & Preprocess", "Replace placeholder zeros\nwith column median"),
    ("3. Train/Test Split", "80% train (614) / 20% test (154)\nstratified by class"),
    ("4. Train 3 Models", "Logistic Regression\nRandom Forest\nXGBoost"),
    ("5. Evaluate", "Accuracy, Precision, Recall,\nF1 Score, ROC-AUC, 5-fold CV"),
    ("6. Compare & Report", "Best single-split model vs.\nmost stable cross-validated model"),
]

fig, ax = plt.subplots(figsize=(7, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(stages) * 2 + 1)
ax.axis("off")

box_color = "#E6F1FB"
edge_color = "#2a78d6"
text_color = "#0C447C"

y_positions = []
for i, (title, subtitle) in enumerate(stages):
    y = len(stages) * 2 - i * 2
    y_positions.append(y)
    box = FancyBboxPatch(
        (1, y - 0.7), 8, 1.4,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.3, edgecolor=edge_color, facecolor=box_color,
    )
    ax.add_patch(box)
    ax.text(5, y + 0.25, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=text_color)
    ax.text(5, y - 0.25, subtitle, ha="center", va="center",
            fontsize=10, color="#333333")

    if i > 0:
        prev_y = y_positions[i - 1]
        arrow = FancyArrowPatch(
            (5, prev_y - 0.7), (5, y + 0.7),
            arrowstyle="-|>", mutation_scale=20,
            linewidth=1.3, color="#555555",
        )
        ax.add_patch(arrow)

plt.title("Diabetes ML Pipeline — Workflow", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/workflow_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Workflow diagram saved.")

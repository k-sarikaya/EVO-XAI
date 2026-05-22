#!/usr/bin/env python
"""
regenerate_fig3_fig4.py
Regenerates Figures 3 and 4 from the verified canonical CSV files.
Run from the project root:  python code/regenerate_fig3_fig4.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "hiss_submission", "CBM_Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                     "font.size": 10, "axes.labelsize": 11,
                     "axes.titlesize": 12, "legend.fontsize": 9})

COLORS   = {"NSGA-II": "#2ecc71", "DE": "#3498db", "Hybrid_PSO_DE": "#e74c3c"}
DATASETS = ["child", "adolescent", "adult"]
DS_LABEL = {"child": "Child", "adolescent": "Adolescent", "adult": "Adult"}

# ──────────────────────────────────────────────────────────────────────
# FIGURE 3  — MCC vs VIF Threshold (VIF 5–13 only, matching manuscript)
# ──────────────────────────────────────────────────────────────────────
print("Generating Figure 3 — MCC vs VIF Threshold ...")

df_vif = pd.read_csv(os.path.join(RESULTS_DIR, "mcc_vif_all_algorithms.csv"))
# Focus on VIF 5-13 (manuscript coverage)
df_vif = df_vif[df_vif["VIF"].between(5, 13)]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for idx, ds in enumerate(DATASETS):
    ax   = axes[idx]
    data = df_vif[df_vif["Dataset"] == ds]
    for algo in ["NSGA-II", "DE", "Hybrid_PSO_DE"]:
        sub = data[data["Algorithm"] == algo].sort_values("VIF")
        ax.plot(sub["VIF"], sub["MCC"], "o-",
                color=COLORS[algo], label=algo.replace("_", " "),
                linewidth=2, markersize=6)

    ax.axvline(x=13, color="gray", linestyle="--", alpha=0.7, label="VIF=13 (Optimal)")
    ax.set_xlabel("VIF Threshold")
    ax.set_ylabel("MCC")
    ax.set_title(f"{DS_LABEL[ds]} Dataset")
    ax.set_xlim(4, 14)
    ax.set_xticks(range(5, 14))
    ax.legend(loc="lower right", fontsize=8)

    # Annotate verified VIF=13 NSGA-II value
    v13_nsga = data[(data["Algorithm"] == "NSGA-II") & (data["VIF"] == 13)]["MCC"].values
    if v13_nsga.size:
        ax.annotate(f"{v13_nsga[0]:.3f}", xy=(13, v13_nsga[0]),
                    xytext=(12.2, v13_nsga[0] + 0.04),
                    fontsize=8, color=COLORS["NSGA-II"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["NSGA-II"]))

plt.suptitle("MCC Performance Across VIF Thresholds", fontsize=14, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "Figure_3.png")
plt.savefig(out3, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved: {out3}")

# Spot-check: adolescent VIF=13 NSGA-II
chk = df_vif[(df_vif["Dataset"]=="adolescent") & (df_vif["VIF"]==13) & (df_vif["Algorithm"]=="NSGA-II")]["MCC"]
print(f"  Spot-check: Adolescent VIF=13 NSGA-II MCC = {chk.values[0]:.4f}  (expected 0.769)")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 4  — Algorithm Comparison at VIF=13
# ──────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 4 — Algorithm Comparison at VIF=13 ...")

df_algo = pd.read_csv(os.path.join(RESULTS_DIR, "algorithm_comparison_vif13.csv"))
ALGOS   = ["NSGA-II", "DE", "Hybrid_PSO_DE"]
METRICS = ["MCC", "Sensitivity", "Specificity"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x     = np.arange(len(DATASETS))
width = 0.25

for col_idx, metric in enumerate(METRICS):
    ax = axes[col_idx]
    col = metric  # column name in CSV

    for bar_idx, algo in enumerate(ALGOS):
        values = []
        errors = []
        for ds in DATASETS:
            row = df_algo[(df_algo["Algorithm"] == algo) & (df_algo["Dataset"] == ds)]
            values.append(float(row[col].values[0]))
            if metric == "MCC":
                errors.append(float(row["MCC_std"].values[0]))

        bars = ax.bar(x + bar_idx * width, values, width,
                      label=algo.replace("_", " "), color=list(COLORS.values())[bar_idx])

        if metric == "MCC":
            ax.errorbar(x + bar_idx * width, values, yerr=errors,
                        fmt="none", color="black", capsize=3)

        # Annotate bar values
        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Algorithm")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DS_LABEL[ds] for ds in DATASETS])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.15)

plt.suptitle("Algorithm Performance Comparison at VIF=13", fontsize=14, fontweight="bold")
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "Figure_4.png")
plt.savefig(out4, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved: {out4}")

# Spot-check: NSGA-II adolescent MCC
chk2 = df_algo[(df_algo["Algorithm"]=="NSGA-II") & (df_algo["Dataset"]=="adolescent")]["MCC"]
print(f"  Spot-check: NSGA-II Adolescent MCC = {chk2.values[0]:.4f}  (expected 0.769)")

print("\nDone. Both figures regenerated from verified canonical CSVs.")

#!/usr/bin/env python
"""
VIF Threshold Sensitivity Analysis (5--13) with Evolutionary Selection

Goal: produce a single canonical CSV and Figure 3 that are consistent with:
- the feature-selection pipeline described in the manuscript, and
- the evaluation protocol used in algorithm_comparison_vif13.py (5-fold CV MCC/Sens/Spec/AUC).

Outputs
- results/vif_threshold_sensitivity.csv
- CBM_Figures/Figure_3.png

Notes
- Uses a RandomForest classifier for evaluation (as in algorithm_comparison_vif13.py).
- Uses n_jobs=1 for portability in restricted Windows environments.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import arff

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from statsmodels.stats.outliers_influence import variance_inflation_factor

from deap import base, creator, tools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# Keep the sweep aligned with the manuscript's sensitivity table and to keep runtime reasonable.
VIF_THRESHOLDS = [5, 8, 10, 13]
ALGORITHMS = ("NSGA-II", "DE", "Hybrid_PSO_DE")


def load_datasets(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    paths = {
        "child": os.path.join(data_dir, "child", "Autism-Child-Data.arff"),
        "adolescent": os.path.join(data_dir, "adolescent", "Autism-Adolescent-Data.arff"),
        "adult": os.path.join(data_dir, "adult", "Autism-Adult-Data.arff"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r") as f:
            data = arff.load(f)
        cols = [a[0] for a in data["attributes"]]
        df = pd.DataFrame(data["data"], columns=cols)

        target_col = "Class/ASD" if "Class/ASD" in df.columns else df.columns[-1]
        y = df[target_col].apply(lambda x: 1 if str(x).upper() in ["YES", "TRUE", "1"] else 0)
        X = df.drop(columns=[target_col])

        X_enc = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == "object":
                num = pd.to_numeric(X[col], errors="coerce")
                if num.notna().sum() > 0.5 * len(X):
                    X_enc[col] = num
                else:
                    X_enc[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
            else:
                X_enc[col] = X[col]

        X_enc = X_enc.fillna(X_enc.median())
        out = X_enc.copy()
        out["Class"] = y.values
        datasets[name] = out
    return datasets


def chi_square_filter(X: pd.DataFrame, y: pd.Series) -> list[str]:
    X_pos = X - X.min().min() + 0.001
    try:
        _, p_vals = chi2(X_pos, y)
        results = (
            pd.DataFrame({"Feature": X.columns, "P": p_vals})
            .sort_values("P")
            .reset_index(drop=True)
        )
        sel = results[results["P"] < 0.05]["Feature"].tolist()
        return sel if len(sel) >= 5 else results.head(5)["Feature"].tolist()
    except Exception:
        return list(X.columns)


def vif_filter(X: pd.DataFrame, features: list[str], vif_threshold: int, min_features: int = 5) -> list[str]:
    X_df = X[features].copy() + 0.001
    try:
        vif_data = [
            {"Feature": col, "VIF": variance_inflation_factor(X_df.values, i)}
            for i, col in enumerate(X_df.columns)
        ]
        vif_df = pd.DataFrame(vif_data)
        high_vif = vif_df[vif_df["VIF"] > vif_threshold]["Feature"].tolist()
        ind = [f for f in features if f not in high_vif]
        if len(ind) < min_features:
            ind = vif_df.sort_values("VIF")["Feature"].tolist()[:min_features]
        return ind
    except Exception:
        return features[:min_features]


def _ensure_deap_creators() -> None:
    if not hasattr(creator, "FitMultiVIF"):
        creator.create("FitMultiVIF", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndVIF"):
        creator.create("IndVIF", list, fitness=creator.FitMultiVIF)


def _fitness(ind, X: pd.DataFrame, y: pd.Series) -> tuple[float, int]:
    sel = [i for i, g in enumerate(ind) if (g > 0.5 if isinstance(g, float) else g == 1)]
    if not sel:
        return 0.0, X.shape[1]
    X_sel = X.iloc[:, sel].values
    y_arr = y.values
    mc = min(np.bincount(y_arr.astype(int)))
    if mc < 2:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y_arr, test_size=0.2, random_state=42, stratify=y_arr
        )
    clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
    clf.fit(X_tr, y_tr)
    return f1_score(y_te, clf.predict(X_te)), len(sel)


def nsga2_selection(X: pd.DataFrame, y: pd.Series, features: list[str], pop: int = 12, gens: int = 8) -> list[str]:
    _ensure_deap_creators()
    n = len(features)
    if n == 0:
        return []
    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.IndVIF, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    toolbox.register("evaluate", lambda i: _fitness(i, X, y))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    p = toolbox.pop(n=pop)
    for i in p:
        i.fitness.values = toolbox.evaluate(i)
    for _ in range(gens):
        offspring = [toolbox.clone(i) for i in toolbox.select(p, len(p))]
        for i in range(1, len(offspring), 2):
            if np.random.rand() < 0.7:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for i in offspring:
            if np.random.rand() < 0.3:
                toolbox.mutate(i)
                del i.fitness.values
        for i in offspring:
            if not i.fitness.valid:
                i.fitness.values = toolbox.evaluate(i)
        p = toolbox.select(p + offspring, pop)
    best = max(p, key=lambda x: x.fitness.values[0])
    selected = [features[i] for i, g in enumerate(best) if g == 1]
    return selected


def de_selection(X: pd.DataFrame, y: pd.Series, features: list[str], pop: int = 12, gens: int = 8, F: float = 0.8, CR: float = 0.9) -> list[str]:
    n = len(features)
    if n == 0:
        return []
    P = np.random.rand(pop, n)
    fit = np.array([_fitness(P[i], X, y)[0] - 0.01 * sum(P[i] > 0.5) for i in range(pop)])
    for _ in range(gens):
        for i in range(pop):
            cand = [j for j in range(pop) if j != i]
            r1, r2, r3 = np.random.choice(cand, 3, replace=False)
            mut = np.clip(P[r1] + F * (P[r2] - P[r3]), 0, 1)
            trial = P[i].copy()
            for j in range(n):
                if np.random.rand() < CR:
                    trial[j] = mut[j]
            tf = _fitness(trial, X, y)[0] - 0.01 * sum(trial > 0.5)
            if tf > fit[i]:
                P[i], fit[i] = trial, tf
    best = P[np.argmax(fit)]
    selected = [features[i] for i in range(n) if best[i] > 0.5]
    return selected


def hybrid_pso_de_selection(X: pd.DataFrame, y: pd.Series, features: list[str], pop: int = 12, gens: int = 8, w: float = 0.7, c1: float = 1.5, c2: float = 1.5, F: float = 0.5, CR: float = 0.5) -> list[str]:
    n = len(features)
    if n == 0:
        return []
    pos = np.random.rand(pop, n)
    vel = (np.random.rand(pop, n) - 0.5) * 0.1
    pbest = pos.copy()
    pfit = np.array([_fitness(pos[i], X, y)[0] - 0.01 * sum(pos[i] > 0.5) for i in range(pop)])
    gb = pbest[np.argmax(pfit)].copy()
    gfit = pfit.max()
    for _ in range(gens):
        for i in range(pop):
            r1, r2 = np.random.rand(2)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gb - pos[i])
            pso_pos = np.clip(pos[i] + vel[i], 0, 1)
            cand = [j for j in range(pop) if j != i]
            if len(cand) >= 3:
                ri = np.random.choice(cand, 3, replace=False)
                mut = np.clip(pos[ri[0]] + F * (pos[ri[1]] - pos[ri[2]]), 0, 1)
                trial = pso_pos.copy()
                for j in range(n):
                    if np.random.rand() < CR:
                        trial[j] = mut[j]
            else:
                trial = pso_pos
            pf = _fitness(pso_pos, X, y)[0] - 0.01 * sum(pso_pos > 0.5)
            tf = _fitness(trial, X, y)[0] - 0.01 * sum(trial > 0.5)
            pos[i] = trial if tf > pf else pso_pos
            cf = max(pf, tf)
            if cf > pfit[i]:
                pbest[i], pfit[i] = pos[i].copy(), cf
                if cf > gfit:
                    gb, gfit = pos[i].copy(), cf
    selected = [features[i] for i in range(n) if gb[i] > 0.5]
    return selected


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict[str, tuple[float, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"mcc": [], "sens": [], "spec": [], "f1": [], "auc": []}
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        metrics["mcc"].append(matthews_corrcoef(y_te, y_pred))
        metrics["sens"].append(tp / (tp + fn) if tp + fn > 0 else 0)
        metrics["spec"].append(tn / (tn + fp) if tn + fp > 0 else 0)
        metrics["f1"].append(f1_score(y_te, y_pred))
        metrics["auc"].append(roc_auc_score(y_te, y_prob))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def main() -> None:
    print("=" * 70)
    print("VIF THRESHOLD SENSITIVITY (5--13) - ALL ALGORITHMS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("CBM_Figures", exist_ok=True)

    datasets = load_datasets()
    algos = {
        "NSGA-II": nsga2_selection,
        "DE": de_selection,
        "Hybrid_PSO_DE": hybrid_pso_de_selection,
    }

    rows = []
    for ds_name, df in datasets.items():
        X0 = df.drop(columns=["Class"])
        y0 = df["Class"]
        stage1 = chi_square_filter(X0, y0)
        X1 = X0[stage1]

        for vif in VIF_THRESHOLDS:
            stage2 = vif_filter(X1, stage1, vif_threshold=vif)
            X2 = X1[stage2] if stage2 else X1[[stage1[0]]]
            for algo_name, algo_fn in algos.items():
                selected = algo_fn(X2, y0, stage2)
                if not selected:
                    selected = stage2[:1]
                Xf = X2[selected]
                res = evaluate_model(Xf, y0)
                rows.append(
                    {
                        "Dataset": ds_name,
                        "VIF_Threshold": vif,
                        "Algorithm": algo_name,
                        "After_ChiSq": len(stage1),
                        "After_VIF": len(stage2),
                        "Final": len(selected),
                        "MCC": res["mcc"][0],
                        "MCC_std": res["mcc"][1],
                        "Sensitivity": res["sens"][0],
                        "Specificity": res["spec"][0],
                        "AUC": res["auc"][0],
                    }
                )
            print(f"[{ds_name}] VIF={vif} done")

    out_csv = os.path.join("results", "vif_threshold_sensitivity.csv")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")

    # Figure 3: MCC vs VIF threshold (per dataset), lines = algorithms
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"NSGA-II": "#3498db", "DE": "#e74c3c", "Hybrid_PSO_DE": "#2ecc71"}
    markers = {"NSGA-II": "o", "DE": "s", "Hybrid_PSO_DE": "^"}
    ds_order = ["child", "adolescent", "adult"]

    for ax, ds_name in zip(axes, ds_order):
        for algo in ALGORITHMS:
            dd = df_out[(df_out["Dataset"] == ds_name) & (df_out["Algorithm"] == algo)].sort_values("VIF_Threshold")
            ax.plot(
                dd["VIF_Threshold"],
                dd["MCC"],
                marker=markers[algo],
                color=colors[algo],
                linewidth=2,
                markersize=6,
                label=algo,
            )
        ax.set_title(ds_name.title())
        ax.set_xlabel("VIF Threshold")
        ax.set_xticks(VIF_THRESHOLDS)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("MCC")
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("Performance (MCC) vs VIF Threshold: Algorithm Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_fig = os.path.join("CBM_Figures", "Figure_3.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_fig}")


if __name__ == "__main__":
    main()

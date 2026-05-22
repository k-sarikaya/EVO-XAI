#!/usr/bin/env python
"""
Reproduce HISS submission artifacts in a deterministic, headless way.

This script is the "source of truth" for the reported core metrics:
- VIF threshold sweep (for Figure 3 / Table 2 narrative)
- Algorithm comparison at VIF=13 (NSGA-II/DE/Hybrid) (for Figure 4 narrative)

It is designed to run on Windows in restricted environments:
- headless matplotlib backend (Agg)
- n_jobs=1 everywhere (avoid multiprocessing / joblib pipe issues)
- fixed RNG seeds

Outputs (written under evo_xai_project/):
- results/hiss_vif_sweep.csv
- results/hiss_algo_comparison_vif13.csv
- hiss_submission/CBM_Figures/Figure_3.png
- hiss_submission/CBM_Figures/Figure_4.png
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import arff

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import chi2
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix

from statsmodels.stats.outliers_influence import variance_inflation_factor

from deap import base, creator, tools

from imblearn.over_sampling import ADASYN, SMOTE
import xgboost as xgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


SEED = 42
N_SPLITS = 5

# Keep the sweep aligned with the manuscript table (5/8/10/13).
VIF_SWEEP = [5, 8, 10, 13]

# Evolutionary settings: keep modest for runtime; deterministic via numpy RNG.
POP = 20
GENS = 15


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    results: Path
    fig_dir: Path


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)


def load_dataset_arff(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = arff.load(f)
    cols = [a[0] for a in data["attributes"]]
    df = pd.DataFrame(data["data"], columns=cols)

    target_col = "Class/ASD" if "Class/ASD" in df.columns else df.columns[-1]
    y = df[target_col].apply(lambda x: 1 if str(x).upper() in {"YES", "TRUE", "1"} else 0)
    X = df.drop(columns=[target_col])

    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == "object":
            num = pd.to_numeric(X[col], errors="coerce")
            if num.notna().sum() > 0.5 * len(X):
                X_enc[col] = num
            else:
                X_enc[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
        else:
            X_enc[col] = X[col]
    return X_enc, y


def _mice_impute_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = SEED, max_iter: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    MICE-style imputation (chained equations) without test leakage.

    Fit the IterativeImputer on X_train, transform both X_train and X_test.
    """
    imp = IterativeImputer(
        random_state=seed,
        max_iter=max_iter,
        initial_strategy="median",
        sample_posterior=False,
        skip_complete=True,
    )
    Xtr = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)
    return Xtr, Xte


def _balance_xy(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN on training folds; fallback to SMOTE; fallback to no-op.
    """
    try:
        # Keep neighbors conservative to avoid failure on small minority folds.
        k = max(1, min(3, int(np.bincount(y.astype(int)).min() - 1)))
        if k < 1:
            return X, y
        return ADASYN(random_state=seed, n_neighbors=k).fit_resample(X, y)
    except Exception:
        try:
            k = max(1, min(2, int(np.bincount(y.astype(int)).min() - 1)))
            if k < 1:
                return X, y
            return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        except Exception:
            return X, y


def _make_base_learners(seed: int = SEED) -> tuple[object, object, object]:
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=1)
    xgb_m = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric="logloss",
    )
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
        ]
    )
    return rf, xgb_m, svm


def _fit_stacking_oof(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict[str, object]:
    """
    Leakage-safe stacking:
    - build OOF predictions for meta-learner using inner 5-fold CV
    - fit final base learners on full (balanced) outer training split
    """
    inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3), dtype=float)

    for tr_i, va_i in inner.split(X, y):
        X_tr, y_tr = X[tr_i], y[tr_i]
        X_va = X[va_i]

        X_bal, y_bal = _balance_xy(X_tr, y_tr, seed=seed)
        rf, xgb_m, svm = _make_base_learners(seed=seed)
        rf.fit(X_bal, y_bal)
        xgb_m.fit(X_bal, y_bal)
        svm.fit(X_bal, y_bal)

        oof[va_i, 0] = rf.predict_proba(X_va)[:, 1]
        oof[va_i, 1] = xgb_m.predict_proba(X_va)[:, 1]
        oof[va_i, 2] = svm.predict_proba(X_va)[:, 1]

    meta = LogisticRegression(max_iter=1000, random_state=seed)
    meta.fit(oof, y)

    X_bal, y_bal = _balance_xy(X, y, seed=seed)
    rf, xgb_m, svm = _make_base_learners(seed=seed)
    rf.fit(X_bal, y_bal)
    xgb_m.fit(X_bal, y_bal)
    svm.fit(X_bal, y_bal)

    return {"rf": rf, "xgb": xgb_m, "svm": svm, "meta": meta}


def _predict_stacking(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    meta_X = np.column_stack(
        [
            model["rf"].predict_proba(X)[:, 1],
            model["xgb"].predict_proba(X)[:, 1],
            model["svm"].predict_proba(X)[:, 1],
        ]
    )
    return model["meta"].predict_proba(meta_X)[:, 1]


def chi_square_filter(X: pd.DataFrame, y: pd.Series, p_thresh: float = 0.05, min_features: int = 5) -> list[str]:
    X_pos = X - X.min().min() + 0.001
    stat, p_vals = chi2(X_pos, y)
    df = pd.DataFrame({"Feature": X.columns, "P": p_vals}).sort_values("P")
    sel = df[df["P"] < p_thresh]["Feature"].tolist()
    return sel if len(sel) >= min_features else df.head(min_features)["Feature"].tolist()


def vif_filter(X: pd.DataFrame, features: list[str], vif_threshold: int, min_features: int = 5) -> list[str]:
    X_df = X[features].copy() + 0.001
    vif_data = [
        {"Feature": col, "VIF": variance_inflation_factor(X_df.values, i)}
        for i, col in enumerate(X_df.columns)
    ]
    vif_df = pd.DataFrame(vif_data).sort_values("VIF")
    keep = vif_df[vif_df["VIF"] <= vif_threshold]["Feature"].tolist()
    if len(keep) < min_features:
        keep = vif_df["Feature"].tolist()[:min_features]
    return keep


def _ensure_deap_creators() -> None:
    if not hasattr(creator, "FitMultiHISS"):
        creator.create("FitMultiHISS", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndHISS"):
        creator.create("IndHISS", list, fitness=creator.FitMultiHISS)


def _fs_fitness(ind, X: pd.DataFrame, y: pd.Series) -> tuple[float, int]:
    sel = [i for i, g in enumerate(ind) if g == 1]
    if not sel:
        return 0.0, X.shape[1]

    X_sel = X.iloc[:, sel].values
    y_arr = y.values
    mc = min(np.bincount(y_arr.astype(int)))
    if mc < 2:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=SEED)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y_arr, test_size=0.2, random_state=SEED, stratify=y_arr
        )
    clf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=1)
    clf.fit(X_tr, y_tr)
    return f1_score(y_te, clf.predict(X_te)), len(sel)


def nsga2_select(X: pd.DataFrame, y: pd.Series, features: list[str]) -> list[str]:
    _ensure_deap_creators()
    n = len(features)
    if n == 0:
        return []

    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.IndHISS, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    toolbox.register("evaluate", lambda i: _fs_fitness(i, X, y))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.pop(n=POP)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for _ in range(GENS):
        offspring = [toolbox.clone(i) for i in toolbox.select(pop, len(pop))]
        for i in range(1, len(offspring), 2):
            if np.random.rand() < 0.7:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for ind in offspring:
            if np.random.rand() < 0.3:
                toolbox.mutate(ind)
                del ind.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, POP)

    best = max(pop, key=lambda x: x.fitness.values[0])
    sel_idx = [i for i, g in enumerate(best) if g == 1]
    if not sel_idx:
        sel_idx = [int(np.argmax([g for g in best]))]
    return [features[i] for i in sel_idx]


def de_select(X: pd.DataFrame, y: pd.Series, features: list[str]) -> list[str]:
    n = len(features)
    if n == 0:
        return []
    P = np.random.rand(POP, n)
    fit = np.array([_fs_fitness((P[i] > 0.5).astype(int), X, y)[0] - 0.01 * sum(P[i] > 0.5) for i in range(POP)])
    F = 0.8
    CR = 0.9
    for _ in range(GENS):
        for i in range(POP):
            cand = [j for j in range(POP) if j != i]
            r1, r2, r3 = np.random.choice(cand, 3, replace=False)
            mut = np.clip(P[r1] + F * (P[r2] - P[r3]), 0, 1)
            trial = P[i].copy()
            jrand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = mut[j]
            tf = _fs_fitness((trial > 0.5).astype(int), X, y)[0] - 0.01 * sum(trial > 0.5)
            if tf > fit[i]:
                P[i], fit[i] = trial, tf
    best = P[np.argmax(fit)]
    sel = [features[i] for i in range(n) if best[i] > 0.5]
    return sel if sel else [features[int(np.argmax(best))]]


def hybrid_pso_de_select(X: pd.DataFrame, y: pd.Series, features: list[str]) -> list[str]:
    n = len(features)
    if n == 0:
        return []
    pos = np.random.rand(POP, n)
    vel = (np.random.rand(POP, n) - 0.5) * 0.1
    pbest = pos.copy()
    pfit = np.array([_fs_fitness((pos[i] > 0.5).astype(int), X, y)[0] - 0.01 * sum(pos[i] > 0.5) for i in range(POP)])
    gb = pbest[np.argmax(pfit)].copy()
    gfit = pfit.max()

    w, c1, c2, F, CR = 0.7, 1.5, 1.5, 0.5, 0.5
    for _ in range(GENS):
        for i in range(POP):
            r1, r2 = np.random.rand(2)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gb - pos[i])
            pso_pos = np.clip(pos[i] + vel[i], 0, 1)

            cand = [j for j in range(POP) if j != i]
            ri = np.random.choice(cand, 3, replace=False)
            mut = np.clip(pos[ri[0]] + F * (pos[ri[1]] - pos[ri[2]]), 0, 1)
            trial = pso_pos.copy()
            for j in range(n):
                if np.random.rand() < CR:
                    trial[j] = mut[j]

            pf = _fs_fitness((pso_pos > 0.5).astype(int), X, y)[0] - 0.01 * sum(pso_pos > 0.5)
            tf = _fs_fitness((trial > 0.5).astype(int), X, y)[0] - 0.01 * sum(trial > 0.5)
            pos[i] = trial if tf > pf else pso_pos
            cf = max(pf, tf)
            if cf > pfit[i]:
                pbest[i], pfit[i] = pos[i].copy(), cf
                if cf > gfit:
                    gb, gfit = pos[i].copy(), cf

    sel = [features[i] for i in range(n) if gb[i] > 0.5]
    return sel if sel else [features[int(np.argmax(gb))]]


def evaluate_pipeline_cv(
    X_raw: pd.DataFrame,
    y: pd.Series,
    *,
    vif_threshold: int,
    algo_fn,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """
    Outer 5-fold evaluation of the full pipeline without test leakage:
    - imputation fit on each outer-train split (MICE / IterativeImputer)
    - chi-square + VIF filtering + evolutionary selection fit on outer-train
    - stacking ensemble fit on outer-train; evaluated on outer-test
    """
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    mccs, sens, spec, aucs = [], [], [], []
    n_chi, n_vif, n_final = [], [], []

    for tr_idx, te_idx in outer.split(X_raw, y):
        X_tr_raw, X_te_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        X_tr, X_te = _mice_impute_train_test(X_tr_raw, X_te_raw, seed=seed)

        stage1 = chi_square_filter(X_tr, y_tr)
        n_chi.append(len(stage1))
        X1_tr, X1_te = X_tr[stage1], X_te[stage1]
        stage2 = vif_filter(X1_tr, stage1, vif_threshold=vif_threshold)
        n_vif.append(len(stage2))
        X2_tr, X2_te = X1_tr[stage2], X1_te[stage2]

        selected = algo_fn(X2_tr, y_tr, stage2)
        if not selected:
            selected = stage2[:1]
        n_final.append(len(selected))

        model = _fit_stacking_oof(X2_tr[selected].values, y_tr.values.astype(int), seed=seed)
        y_prob = _predict_stacking(model, X2_te[selected].values)
        y_pred = (y_prob > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_te.values, y_pred).ravel()
        mccs.append(matthews_corrcoef(y_te.values, y_pred))
        sens.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        spec.append(tn / (tn + fp) if tn + fp > 0 else 0.0)
        aucs.append(roc_auc_score(y_te.values, y_prob))

    def ms(a: list[float]) -> tuple[float, float]:
        return float(np.mean(a)), float(np.std(a))

    out: dict[str, tuple[float, float]] = {"mcc": ms(mccs), "sens": ms(sens), "spec": ms(spec), "auc": ms(aucs)}
    # Count summaries are returned as (mean, std) to keep a uniform shape.
    out["after_chisq"] = ms([float(v) for v in n_chi])
    out["after_vif"] = ms([float(v) for v in n_vif])
    out["final"] = ms([float(v) for v in n_final])
    return out


def compute_vif_sweep(paths: Paths, datasets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    rows = []
    for ds, (X0, y0) in datasets.items():
        for vif in VIF_SWEEP:
            res = evaluate_pipeline_cv(X0, y0, vif_threshold=vif, algo_fn=nsga2_select, seed=SEED)
            rows.append(
                {
                    "Dataset": ds,
                    "Original": int(X0.shape[1]),
                    "VIF_Threshold": vif,
                    "After_ChiSq": res["after_chisq"][0],
                    "After_VIF": res["after_vif"][0],
                    "Final": res["final"][0],
                    "MCC": res["mcc"][0],
                    "MCC_std": res["mcc"][1],
                    "Sensitivity": res["sens"][0],
                    "Specificity": res["spec"][0],
                    "AUC": res["auc"][0],
                }
            )
            print(f"[sweep] {ds} VIF={vif} MCC={res['mcc'][0]:.3f}")
    df = pd.DataFrame(rows)
    out = paths.results / "hiss_vif_sweep.csv"
    df.to_csv(out, index=False)
    print(f"[OK] {out}")
    return df


def compute_algo_comparison_vif13(paths: Paths, datasets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    rows = []
    algo_map = {
        "NSGA-II": nsga2_select,
        "DE": de_select,
        "Hybrid_PSO_DE": hybrid_pso_de_select,
    }
    for ds, (X0, y0) in datasets.items():
        for algo_name, algo_fn in algo_map.items():
            res = evaluate_pipeline_cv(X0, y0, vif_threshold=13, algo_fn=algo_fn, seed=SEED)
            rows.append(
                {
                    "Dataset": ds,
                    "Algorithm": algo_name,
                    "Original": int(X0.shape[1]),
                    "VIF_Threshold": 13,
                    "After_ChiSq": res["after_chisq"][0],
                    "After_VIF": res["after_vif"][0],
                    "Final": res["final"][0],
                    "MCC": res["mcc"][0],
                    "MCC_std": res["mcc"][1],
                    "Sensitivity": res["sens"][0],
                    "Specificity": res["spec"][0],
                    "AUC": res["auc"][0],
                }
            )
            print(f"[algo] {ds} {algo_name} MCC={res['mcc'][0]:.3f}")
    df = pd.DataFrame(rows)
    out = paths.results / "hiss_algo_comparison_vif13.csv"
    df.to_csv(out, index=False)
    print(f"[OK] {out}")
    return df


def plot_figure_3(paths: Paths, df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"child": "#3498db", "adolescent": "#e74c3c", "adult": "#2ecc71"}
    markers = {"child": "o", "adolescent": "s", "adult": "^"}
    order = ["child", "adolescent", "adult"]

    for ax, ds in zip(axes, order):
        d = df[df["Dataset"] == ds].sort_values("VIF_Threshold")
        ax.plot(d["VIF_Threshold"], d["MCC"], marker=markers[ds], color=colors[ds], linewidth=2)
        ax.set_title(ds.title())
        ax.set_xlabel("VIF Threshold")
        ax.set_xticks(VIF_SWEEP)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MCC")
    fig.suptitle("Performance (MCC) vs VIF Threshold (NSGA-II final model)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = paths.fig_dir / "Figure_3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_figure_4(paths: Paths, df: pd.DataFrame) -> None:
    # 3 panels: MCC, Sens, Spec at VIF=13
    plt.style.use("seaborn-v0_8-whitegrid")
    metrics = ["MCC", "Sensitivity", "Specificity"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    algos = ["NSGA-II", "DE", "Hybrid_PSO_DE"]
    ds_order = ["child", "adolescent", "adult"]
    colors = {"NSGA-II": "#3498db", "DE": "#e74c3c", "Hybrid_PSO_DE": "#2ecc71"}

    for ax, metric in zip(axes, metrics):
        width = 0.25
        x = np.arange(len(ds_order))
        for i, algo in enumerate(algos):
            vals = []
            for ds in ds_order:
                v = float(df[(df["Dataset"] == ds) & (df["Algorithm"] == algo)][metric].values[0])
                vals.append(v)
            ax.bar(x + (i - 1) * width, vals, width=width, label=algo, color=colors[algo])
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in ds_order])
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(loc="lower left", fontsize=9, frameon=True)
    fig.suptitle("Algorithm comparison at VIF=13", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = paths.fig_dir / "Figure_4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> None:
    set_seeds(SEED)
    root = Path(__file__).resolve().parents[1]
    paths = Paths(
        root=root,
        data=root / "data",
        results=root / "results",
        fig_dir=root / "hiss_submission" / "CBM_Figures",
    )
    paths.results.mkdir(exist_ok=True)
    paths.fig_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "child": load_dataset_arff(paths.data / "child" / "Autism-Child-Data.arff"),
        "adolescent": load_dataset_arff(paths.data / "adolescent" / "Autism-Adolescent-Data.arff"),
        "adult": load_dataset_arff(paths.data / "adult" / "Autism-Adult-Data.arff"),
    }

    df_sweep = compute_vif_sweep(paths, datasets)
    plot_figure_3(paths, df_sweep)

    df_cmp = compute_algo_comparison_vif13(paths, datasets)
    plot_figure_4(paths, df_cmp)

    print("\nDone. Next steps:")
    print("- Update manuscript tables/text to match results/hiss_*.csv.")
    print("- Re-run the LaTeX build to ensure the PDF reflects updated figures and numbers.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
verify_nsga2_vif13.py -- NSGA-II at VIF=13, 5-seed x 5-fold verification.
Self-contained script for verifying adolescent MCC claim.
Run from project root or with full path.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, '..'))

from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, StackingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.feature_selection import chi2
from sklearn.metrics           import matthews_corrcoef
import xgboost as xgb
from imblearn.over_sampling    import ADASYN, SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from deap import base, creator, tools
import arff

VIF_THRESHOLD = 13
SEEDS         = [42, 7, 123, 256, 999]
N_OUTER       = 5
POP_SIZE      = 30
N_GENS        = 30
DATA_DIR      = os.path.join(PROJECT_DIR, 'data')
RESULT_DIR    = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)
CLAIMS        = {'child': 0.703, 'adolescent': 0.785, 'adult': 0.593}

PATHS = {
    'child':      os.path.join(DATA_DIR, 'child',      'Autism-Child-Data.arff'),
    'adolescent': os.path.join(DATA_DIR, 'adolescent', 'Autism-Adolescent-Data.arff'),
    'adult':      os.path.join(DATA_DIR, 'adult',      'Autism-Adult-Data.arff'),
}

# DEAP creator (guarded)
if not hasattr(creator, 'FitnessV13'):
    creator.create('FitnessV13', base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, 'IndividualV13'):
    creator.create('IndividualV13', list, fitness=creator.FitnessV13)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_and_preprocess(path):
    """Load ARFF, encode to numeric, return (X: DataFrame, y: Series)."""
    with open(path, 'r') as f:
        ds = arff.load(f)
    cols = [a[0] for a in ds['attributes']]
    df   = pd.DataFrame(ds['data'], columns=cols)

    # Detect target
    for cand in ['Class/ASD', 'Class', 'class', 'austim', 'Class/ASD Traits ']:
        if cand in df.columns:
            df = df.rename(columns={cand: '_target'})
            break
    else:
        df = df.rename(columns={df.columns[-1]: '_target'})

    # Binarise target
    df['_target'] = df['_target'].apply(
        lambda v: 1 if str(v).strip().lower() in ('yes','true','1','asd') else 0)

    # Encode features
    for col in df.columns:
        if col == '_target':
            continue
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            uniq = df[col].dropna().unique()
            # try numeric first
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str)).astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop cols that are entirely NaN (IterativeImputer would shrink the shape)
    X = df.drop('_target', axis=1)
    X = X.dropna(axis=1, how='all')
    # Simple median fill for remaining NaN (< 1% of values in these datasets)
    X = X.fillna(X.median())
    y = df['_target'].astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def chi2_filter(X, y, p=0.05, min_feats=5):
    X_pos = (X - X.min() + 0.001).values
    _, pvals = chi2(X_pos, y.values)
    sel = [c for c, p_ in zip(X.columns, pvals) if p_ < p]
    if len(sel) < min_feats:
        sel = list(X.columns[np.argsort(pvals)[:min_feats]])
    return sel


def vif_filter(X, threshold):
    cols = list(X.columns)
    while len(cols) > 1:
        sub  = X[cols].values.astype(float)
        vifs = [variance_inflation_factor(sub, i) for i in range(len(cols))]
        if max(vifs) <= threshold:
            break
        cols.pop(int(np.argmax(vifs)))
    return cols


def nsga2_select(X_vif, y, seed):
    np.random.seed(seed)
    feats = list(X_vif.columns)
    n     = len(feats)
    tb    = base.Toolbox()
    tb.register('attr_bool', np.random.randint, 0, 2)
    tb.register('individual', tools.initRepeat, creator.IndividualV13,
                tb.attr_bool, n=n)
    tb.register('population', tools.initRepeat, list, tb.individual)

    def evaluate(ind):
        idx = [i for i, g in enumerate(ind) if g == 1]
        if not idx:
            return (1.0, n)
        Xs  = X_vif.iloc[:, idx]
        clf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
        sc  = cross_val_score(clf, Xs, y, cv=3, scoring='f1')
        return (1.0 - sc.mean(), len(idx))

    tb.register('evaluate', evaluate)
    tb.register('mate',   tools.cxTwoPoint)
    tb.register('mutate', tools.mutFlipBit, indpb=0.1)
    tb.register('select', tools.selNSGA2)

    pop = tb.population(n=POP_SIZE)
    for ind, fit in zip(pop, map(tb.evaluate, pop)):
        ind.fitness.values = fit
    for _ in range(N_GENS):
        off = tb.select(pop, len(pop))
        off = [tb.clone(o) for o in off]
        for i in range(1, len(off), 2):
            if np.random.random() < 0.8:
                tb.mate(off[i-1], off[i])
                del off[i-1].fitness.values, off[i].fitness.values
        for i in range(len(off)):
            if np.random.random() < 0.1:
                tb.mutate(off[i]); del off[i].fitness.values
        invalid = [o for o in off if not o.fitness.valid]
        for ind, fit in zip(invalid, map(tb.evaluate, invalid)):
            ind.fitness.values = fit
        pop = tb.select(pop + off, POP_SIZE)

    pareto   = sorted([p for p in pop if p.fitness.valid],
                      key=lambda x: x.fitness.values)
    best_idx = [i for i, g in enumerate(pareto[0]) if g == 1]
    return [feats[i] for i in best_idx] if best_idx else feats[:3]


def eval_stacking(X, y, features, seed):
    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    X_sel = X[features]
    mccs  = []
    for tr, te in outer.split(X_sel, y):
        Xtr, Xte = X_sel.iloc[tr].copy(), X_sel.iloc[te].copy()
        ytr, yte = y.iloc[tr].copy(), y.iloc[te].copy()
        try:
            Xtr_b, ytr_b = ADASYN(random_state=seed, n_neighbors=3).fit_resample(Xtr, ytr)
        except Exception:
            Xtr_b, ytr_b = SMOTE(random_state=seed, k_neighbors=2).fit_resample(Xtr, ytr)
        Xtr_b = pd.DataFrame(Xtr_b, columns=features)
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_b)
        Xte_s = sc.transform(Xte)
        stack = StackingClassifier(
            estimators=[
                ('rf',  RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=seed,
                                           verbosity=0, eval_metric='logloss')),
                ('svm', SVC(probability=True, random_state=seed)),
            ],
            final_estimator=LogisticRegression(random_state=seed),
            cv=3, n_jobs=-1,
        )
        stack.fit(Xtr_s, ytr_b)
        mccs.append(matthews_corrcoef(yte, stack.predict(Xte_s)))
    return float(np.mean(mccs)), float(np.std(mccs))


# ── MAIN ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"NSGA-II  VIF={VIF_THRESHOLD}  --  {len(SEEDS)} seeds x {N_OUTER}-fold stacking CV")
print("=" * 70)

rows = []
for name in ['child', 'adolescent', 'adult']:
    path = PATHS[name]
    if not os.path.exists(path):
        print(f"\nSKIP {name}: file not found ({path})"); continue

    print(f"\n[{name.upper()}]  loading ...")
    X, y = load_and_preprocess(path)
    print(f"  n={len(X)}, features={len(X.columns)}, ASD={int(y.sum())}")

    chi_feats = chi2_filter(X, y)
    print(f"  After chi2: {len(chi_feats)} features")

    X_chi     = X[chi_feats]
    vif_feats = vif_filter(X_chi, VIF_THRESHOLD)
    X_vif     = X_chi[vif_feats]
    print(f"  After VIF<={VIF_THRESHOLD}: {len(vif_feats)} -> {vif_feats}")

    seed_mccs = []
    for seed in SEEDS:
        feats         = nsga2_select(X_vif, y, seed=seed)
        mcc_m, mcc_sd = eval_stacking(X, y, feats, seed=seed)
        seed_mccs.append(mcc_m)
        print(f"    seed={seed:>4d}  feats={len(feats)}  MCC={mcc_m:.4f}+-{mcc_sd:.4f}  {feats}")

    agg_mean = float(np.mean(seed_mccs))
    agg_std  = float(np.std(seed_mccs))
    claim    = CLAIMS[name]
    verdict  = "MATCH" if abs(agg_mean - claim) < 0.05 else "MISMATCH"
    print(f"  >> Aggregate: MCC={agg_mean:.4f}+-{agg_std:.4f}  "
          f"manuscript={claim}  [{verdict}]")

    rows.append({'Dataset': name, 'VIF': VIF_THRESHOLD, 'Algorithm': 'NSGA-II',
                 'N_seeds': len(SEEDS), 'MCC_mean': round(agg_mean, 4),
                 'MCC_std': round(agg_std, 4), 'MCC_min': round(min(seed_mccs), 4),
                 'MCC_max': round(max(seed_mccs), 4), 'Manuscript': claim,
                 'Verdict': verdict,
                 'Per_seed': str([round(m, 4) for m in seed_mccs])})

out = os.path.join(RESULT_DIR, 'verified_nsga2_vif13.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved -> {out}")
print(f"\n{'Dataset':<12} {'Verified':>10} {'Manuscript':>12} {'Verdict':>10}")
print("-" * 48)
for r in rows:
    print(f"{r['Dataset']:<12} {r['MCC_mean']:>10.4f} {r['Manuscript']:>12.3f} {r['Verdict']:>10}")

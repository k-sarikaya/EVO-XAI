#!/usr/bin/env python
"""
MCC vs VIF Threshold Graph for All Evolutionary Algorithms
Compares: NSGA-II, DE, Hybrid PSO+DE across VIF thresholds 5-13
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import arff

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.metrics import matthews_corrcoef, f1_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
from deap import base, creator, tools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[OK] MCC vs VIF for All Algorithms - Libraries loaded")


def load_datasets(data_dir='data'):
    datasets = {}
    paths = {
        'child': os.path.join(data_dir, 'child', 'Autism-Child-Data.arff'),
        'adolescent': os.path.join(data_dir, 'adolescent', 'Autism-Adolescent-Data.arff'),
        'adult': os.path.join(data_dir, 'adult', 'Autism-Adult-Data.arff')
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = arff.load(f)
        cols = [a[0] for a in data['attributes']]
        df = pd.DataFrame(data['data'], columns=cols)
        target_col = 'Class/ASD' if 'Class/ASD' in df.columns else df.columns[-1]
        y = df[target_col].apply(lambda x: 1 if str(x).upper() in ['YES', 'TRUE', '1'] else 0)
        X = df.drop(columns=[target_col])
        X_enc = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == 'object':
                num = pd.to_numeric(X[col], errors='coerce')
                if num.notna().sum() > 0.5 * len(X):
                    X_enc[col] = num
                else:
                    X_enc[col] = LabelEncoder().fit_transform(X[col].fillna('missing').astype(str))
            else:
                X_enc[col] = X[col]
        X_enc = X_enc.fillna(X_enc.median())
        result = X_enc.copy()
        result['Class'] = y.values
        datasets[name] = result
    return datasets


def chi_square_filter(X, y):
    X_pos = X - X.min().min() + 0.001
    try:
        _, p_vals = chi2(X_pos, y)
        results = pd.DataFrame({'Feature': X.columns, 'P': p_vals}).sort_values('P')
        sel = results[results['P'] < 0.05]['Feature'].tolist()
        return sel if len(sel) >= 5 else results.head(5)['Feature'].tolist()
    except:
        return list(X.columns)


def vif_filter(X, features, vif_thresh, min_feat=5):
    X_df = X[features].copy() + 0.001
    try:
        vif_data = [{'Feature': col, 'VIF': variance_inflation_factor(X_df.values, i)} 
                    for i, col in enumerate(X_df.columns)]
        vif_df = pd.DataFrame(vif_data)
        high = vif_df[vif_df['VIF'] > vif_thresh]['Feature'].tolist()
        ind = [f for f in features if f not in high]
        if len(ind) < min_feat:
            ind = vif_df.sort_values('VIF')['Feature'].tolist()[:min_feat]
        return ind
    except:
        return features[:min_feat]


def fitness(ind, X, y, features):
    sel = [i for i, g in enumerate(ind) if (g > 0.5 if isinstance(g, float) else g == 1)]
    if not sel:
        return 0.0, len(features)
    try:
        X_sel = X.iloc[:, sel].values
        y_arr = y.values
        mc = min(np.bincount(y_arr.astype(int)))
        if mc < 2:
            X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
        clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return f1_score(y_te, clf.predict(X_te)), len(sel)
    except:
        return 0.0, len(sel)


# DEAP setup
if not hasattr(creator, "FitMulti"):
    creator.create("FitMulti", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Ind"):
    creator.create("Ind", list, fitness=creator.FitMulti)


def nsga2(X, y, features, pop=15, gens=10):
    n = len(features)
    if n == 0: return [], 0.0
    tb = base.Toolbox()
    tb.register("attr", np.random.randint, 0, 2)
    tb.register("ind", tools.initRepeat, creator.Ind, tb.attr, n=n)
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("evaluate", lambda i: fitness(i, X, y, features))
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", tools.mutFlipBit, indpb=0.1)
    tb.register("select", tools.selNSGA2)
    
    p = tb.pop(n=pop)
    for i in p: i.fitness.values = tb.evaluate(i)
    for _ in range(gens):
        off = [tb.clone(i) for i in tb.select(p, len(p))]
        for i in range(1, len(off), 2):
            if np.random.random() < 0.8:
                tb.mate(off[i-1], off[i])
                del off[i-1].fitness.values, off[i].fitness.values
        for i in off:
            if np.random.random() < 0.1:
                tb.mutate(i)
                if hasattr(i.fitness, 'values'): del i.fitness.values
        for i in off:
            if not i.fitness.valid: i.fitness.values = tb.evaluate(i)
        p = tb.select(p + off, pop)
    best = max(p, key=lambda x: x.fitness.values[0])
    return [features[i] for i, g in enumerate(best) if g == 1], best.fitness.values[0]


def de(X, y, features, pop=15, gens=10, F=0.8, CR=0.9):
    n = len(features)
    if n == 0: return [], 0.0
    P = np.random.rand(pop, n)
    fit = np.array([fitness(P[i], X, y, features)[0] - 0.01*sum(P[i]>0.5) for i in range(pop)])
    for _ in range(gens):
        for i in range(pop):
            cand = [j for j in range(pop) if j != i]
            r1, r2, r3 = np.random.choice(cand, 3, replace=False)
            mut = np.clip(P[r1] + F*(P[r2]-P[r3]), 0, 1)
            trial = P[i].copy()
            for j in range(n):
                if np.random.rand() < CR: trial[j] = mut[j]
            tf = fitness(trial, X, y, features)[0] - 0.01*sum(trial>0.5)
            if tf > fit[i]:
                P[i], fit[i] = trial, tf
    best = P[np.argmax(fit)]
    return [features[i] for i in range(n) if best[i] > 0.5] or [features[np.argmax(best)]], fitness(best, X, y, features)[0]


def pso_de(X, y, features, pop=15, gens=10, w=0.7, c1=1.5, c2=1.5, F=0.5, CR=0.5):
    n = len(features)
    if n == 0: return [], 0.0
    pos = np.random.rand(pop, n)
    vel = (np.random.rand(pop, n) - 0.5) * 0.1
    pbest = pos.copy()
    pfit = np.array([fitness(pos[i], X, y, features)[0] - 0.01*sum(pos[i]>0.5) for i in range(pop)])
    gb = pbest[np.argmax(pfit)].copy()
    gfit = pfit.max()
    
    for _ in range(gens):
        for i in range(pop):
            r1, r2 = np.random.rand(2)
            vel[i] = w*vel[i] + c1*r1*(pbest[i]-pos[i]) + c2*r2*(gb-pos[i])
            pso_pos = np.clip(pos[i] + vel[i], 0, 1)
            cand = [j for j in range(pop) if j != i]
            if len(cand) >= 3:
                ri = np.random.choice(cand, 3, replace=False)
                mut = np.clip(pos[ri[0]] + F*(pos[ri[1]]-pos[ri[2]]), 0, 1)
                trial = pso_pos.copy()
                for j in range(n):
                    if np.random.rand() < CR: trial[j] = mut[j]
            else:
                trial = pso_pos
            pf = fitness(pso_pos, X, y, features)[0] - 0.01*sum(pso_pos>0.5)
            tf = fitness(trial, X, y, features)[0] - 0.01*sum(trial>0.5)
            pos[i] = trial if tf > pf else pso_pos
            cf = max(pf, tf)
            if cf > pfit[i]:
                pbest[i], pfit[i] = pos[i].copy(), cf
                if cf > gfit: gb, gfit = pos[i].copy(), cf
    return [features[i] for i in range(n) if gb[i] > 0.5] or [features[np.argmax(gb)]], fitness(gb, X, y, features)[0]


def evaluate_cv(X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mccs = []
    for tr, te in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr].values, y.iloc[tr].values)
        mccs.append(matthews_corrcoef(y.iloc[te].values, clf.predict(X.iloc[te].values)))
    return np.mean(mccs)


def main():
    print("\n" + "="*70)
    print("MCC vs VIF THRESHOLD - ALL ALGORITHMS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    datasets = load_datasets()
    
    vif_thresholds = list(range(5, 21)) # VIF 5 to 20
    algorithms = {'NSGA-II': nsga2, 'DE': de, 'Hybrid_PSO_DE': pso_de}
    results = []
    
    for ds_name, df in datasets.items():
        print(f"\n[{ds_name}]")
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        for vif in vif_thresholds:
            print(f"  VIF={vif}: ", end="")
            s1 = chi_square_filter(X, y)
            s2 = vif_filter(X[s1], s1, vif)
            X_s2 = X[s2]
            
            for algo_name, algo_fn in algorithms.items():
                sel, _ = algo_fn(X_s2, y, s2)
                if not sel: sel = s2[:1]
                mcc = evaluate_cv(X_s2[sel], y)
                results.append({'Dataset': ds_name, 'VIF': vif, 'Algorithm': algo_name, 'MCC': mcc, 'Features': len(sel)})
                print(f"{algo_name[:4]}={mcc:.2f} ", end="")
            print()
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/mcc_vif_all_algorithms.csv', index=False)
    print("\n[OK] results/mcc_vif_all_algorithms.csv")
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'NSGA-II': '#3498db', 'DE': '#e74c3c', 'Hybrid_PSO_DE': '#2ecc71'}
    markers = {'NSGA-II': 'o', 'DE': 's', 'Hybrid_PSO_DE': '^'}
    
    for idx, ds_name in enumerate(datasets.keys()):
        ax = axes[idx]
        for algo in algorithms.keys():
            data = df_results[(df_results['Dataset'] == ds_name) & (df_results['Algorithm'] == algo)]
            ax.plot(data['VIF'], data['MCC'], marker=markers[algo], 
                   label=algo.replace('_', ' '), color=colors[algo], linewidth=2, markersize=6)
        ax.set_xlabel('VIF Threshold', fontsize=12)
        ax.set_ylabel('MCC', fontsize=12)
        ax.set_title(f'{ds_name.title()} Dataset', fontsize=13, fontweight='bold')
        ax.set_xticks( range(5, 21, 2) )
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 0.95)
    
    plt.suptitle('Performance (MCC) vs VIF Threshold: Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('CBM_Figures/Figure_3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] CBM_Figures/Figure_3.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    pivot = df_results.pivot_table(index=['Dataset', 'VIF'], columns='Algorithm', values='MCC')
    print(pivot.to_string())
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

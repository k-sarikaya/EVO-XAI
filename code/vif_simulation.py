#!/usr/bin/env python
"""
VIF Threshold Sensitivity Analysis
Tests VIF thresholds: 8, 9, 10, 11, 12
Generates feature selection plots and classification metrics
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
import xgboost as xgb

from deap import base, creator, tools

from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("[OK] VIF Threshold Simulation - Libraries loaded")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets(data_dir='data'):
    """Load ARFF datasets"""
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
        
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == 'object':
                numeric_vals = pd.to_numeric(X[col], errors='coerce')
                if numeric_vals.notna().sum() > 0.5 * len(X):
                    X_encoded[col] = numeric_vals
                else:
                    X_encoded[col] = LabelEncoder().fit_transform(X[col].fillna('missing').astype(str))
            else:
                X_encoded[col] = X[col]
        
        X_encoded = X_encoded.fillna(X_encoded.median())
        
        result = X_encoded.copy()
        result['Class'] = y.values
        datasets[name] = result
    
    return datasets


# ============================================================================
# FEATURE SELECTION FUNCTIONS
# ============================================================================

def chi_square_filter(X, y, p_threshold=0.05):
    """Stage 1: Chi-square filtering"""
    X_pos = X - X.min().min() + 0.001
    
    try:
        chi2_stats, p_values = chi2(X_pos, y)
        
        results = pd.DataFrame({
            'Feature': X.columns,
            'P_Value': p_values,
            'Significant': p_values < p_threshold
        }).sort_values('P_Value')
        
        selected = results[results['Significant']]['Feature'].tolist()
        
        if len(selected) < 5:
            selected = results.head(max(5, len(selected)))['Feature'].tolist()
        
        return selected
        
    except:
        return list(X.columns)


def multicollinearity_filter(X, feature_names, vif_threshold=10, min_features=5):
    """Stage 2: VIF and correlation check"""
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X[feature_names].copy()
    
    X_df = X_df + 0.001
    
    # VIF Analysis
    try:
        vif_data = []
        for i, col in enumerate(X_df.columns):
            try:
                vif = variance_inflation_factor(X_df.values, i)
                vif_data.append({'Feature': col, 'VIF': vif})
            except:
                vif_data.append({'Feature': col, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()
        
    except:
        vif_df = pd.DataFrame()
        high_vif_features = []
    
    # Correlation Analysis (0.95 threshold)
    corr_matrix = X_df.corr()
    redundant_from_corr = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.95:
                redundant_from_corr.add(feature_names[j])
    
    all_redundant = set(high_vif_features) | redundant_from_corr
    independent_features = [f for f in feature_names if f not in all_redundant]
    
    # Ensure minimum features
    if len(independent_features) < min_features:
        if len(vif_df) > 0:
            sorted_by_vif = vif_df.sort_values('VIF')['Feature'].tolist()
            independent_features = sorted_by_vif[:min_features]
        else:
            independent_features = feature_names[:min_features]
    
    return independent_features


# Setup DEAP
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def nsga2_selection(X, y, features, pop_size=15, gens=10):
    """Stage 3: NSGA-II optimization"""
    n = len(features)
    
    if n == 0:
        return [], 0.0
    
    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.Individual, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    
    def evaluate(ind):
        sel = [i for i, g in enumerate(ind) if g == 1]
        if not sel:
            return (1.0, n)
        try:
            X_sel = X.iloc[:, sel].values
            y_arr = y.values
            
            min_class = min(np.bincount(y_arr.astype(int)))
            if min_class < 2:
                X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42)
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
            
            clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            return (1 - f1_score(y_te, y_pred), len(sel))
        except:
            return (1.0, len(sel))
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    pop = toolbox.pop(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    
    for g in range(gens):
        offspring = [toolbox.clone(i) for i in toolbox.select(pop, len(pop))]
        for i in range(1, len(offspring), 2):
            if np.random.random() < 0.8 and i < len(offspring):
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
        for ind in offspring:
            if np.random.random() < 0.1:
                toolbox.mutate(ind)
                if hasattr(ind.fitness, 'values'):
                    del ind.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, pop_size)
    
    best = min(pop, key=lambda x: x.fitness.values[0])
    err, nf = best.fitness.values
    selected_features = [features[i] for i, g in enumerate(best) if g == 1]
    
    return selected_features, 1-err


def evaluate_model(X, y):
    """5-fold CV evaluation"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'mcc': [], 'sens': [], 'spec': [], 'f1': [], 'auc': []}
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        
        # Simple RF for speed
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        metrics['mcc'].append(matthews_corrcoef(y_te, y_pred))
        metrics['sens'].append(tp/(tp+fn) if tp+fn > 0 else 0)
        metrics['spec'].append(tn/(tn+fp) if tn+fp > 0 else 0)
        metrics['f1'].append(f1_score(y_te, y_pred))
        metrics['auc'].append(roc_auc_score(y_te, y_prob))
    
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("VIF THRESHOLD SENSITIVITY ANALYSIS")
    print(f"Testing VIF thresholds: 8, 9, 10, 11, 12")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    # Load Data
    print("\n[1] Loading datasets...")
    datasets = load_datasets()
    print(f"    Loaded: {list(datasets.keys())}")
    
    # VIF thresholds to test
    vif_thresholds = [8, 9, 10, 11, 12]
    
    # Store results
    all_results = []
    feature_counts = []
    
    for vif_threshold in vif_thresholds:
        print(f"\n{'='*70}")
        print(f"VIF THRESHOLD = {vif_threshold}")
        print(f"{'='*70}")
        
        for name, df in datasets.items():
            print(f"\n  [{name}]")
            
            X = df.drop(columns=['Class'])
            y = df['Class']
            
            original_n = len(X.columns)
            
            # Stage 1: Chi-Square
            stage1_features = chi_square_filter(X, y)
            X_stage1 = X[stage1_features]
            chi_n = len(stage1_features)
            
            # Stage 2: VIF with current threshold
            stage2_features = multicollinearity_filter(X_stage1, stage1_features, vif_threshold=vif_threshold)
            X_stage2 = X_stage1[stage2_features]
            vif_n = len(stage2_features)
            
            # Stage 3: NSGA-II
            stage3_features, best_f1 = nsga2_selection(X_stage2, y, stage2_features)
            
            if not stage3_features:
                stage3_features = stage2_features[:1]
            
            X_stage3 = X_stage2[stage3_features]
            final_n = len(stage3_features)
            
            print(f"    Features: {original_n} -> {chi_n} -> {vif_n} -> {final_n}")
            
            # Evaluate
            if final_n > 0:
                results = evaluate_model(X_stage3, y)
                print(f"    MCC={results['mcc'][0]:.3f}, Sens={results['sens'][0]:.1%}, Spec={results['spec'][0]:.1%}")
            else:
                results = {'mcc': (0, 0), 'sens': (0, 0), 'spec': (0, 0), 'f1': (0, 0), 'auc': (0, 0)}
            
            # Store
            all_results.append({
                'VIF_Threshold': vif_threshold,
                'Dataset': name,
                'Original': original_n,
                'After_ChiSq': chi_n,
                'After_VIF': vif_n,
                'Final': final_n,
                'MCC': results['mcc'][0],
                'Sensitivity': results['sens'][0],
                'Specificity': results['spec'][0],
                'F1': results['f1'][0],
                'AUC': results['auc'][0]
            })
            
            feature_counts.append({
                'VIF_Threshold': vif_threshold,
                'Dataset': name,
                'Original': original_n,
                'After_ChiSq': chi_n,
                'After_VIF': vif_n,
                'Final': final_n
            })
    
    # Create DataFrames
    results_df = pd.DataFrame(all_results)
    feature_df = pd.DataFrame(feature_counts)
    
    # Save results
    results_df.to_csv('results/vif_simulation_results.csv', index=False)
    print("\n[OK] Saved: results/vif_simulation_results.csv")
    
    # ===== CREATE PLOTS =====
    print("\n[2] Creating plots...")
    
    # Plot 1: Feature Selection Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets_list = list(datasets.keys())
    
    for i, dataset in enumerate(datasets_list):
        df_subset = feature_df[feature_df['Dataset'] == dataset]
        
        x = np.arange(len(vif_thresholds))
        width = 0.2
        
        ax = axes[i]
        ax.bar(x - 1.5*width, df_subset['Original'], width, label='Original', color='#3498db')
        ax.bar(x - 0.5*width, df_subset['After_ChiSq'], width, label='After Chi²', color='#2ecc71')
        ax.bar(x + 0.5*width, df_subset['After_VIF'], width, label='After VIF', color='#e74c3c')
        ax.bar(x + 1.5*width, df_subset['Final'], width, label='Final (NSGA-II)', color='#9b59b6')
        
        ax.set_xlabel('VIF Threshold')
        ax.set_ylabel('Number of Features')
        ax.set_title(f'{dataset.title()} Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(vif_thresholds)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim(0, 25)
    
    plt.suptitle('Feature Selection Across VIF Thresholds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/vif_feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/vif_feature_selection_comparison.png")
    
    # Plot 2: Classification Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['MCC', 'Sensitivity', 'Specificity']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        for j, dataset in enumerate(datasets_list):
            df_subset = results_df[results_df['Dataset'] == dataset]
            ax.plot(vif_thresholds, df_subset[metric], marker='o', label=dataset.title(), 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('VIF Threshold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs VIF Threshold')
        ax.set_xticks(vif_thresholds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Classification Metrics Across VIF Thresholds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/vif_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/vif_metrics_comparison.png")
    
    # Plot 3: Combined heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MCC Heatmap
    pivot_mcc = results_df.pivot(index='Dataset', columns='VIF_Threshold', values='MCC')
    sns.heatmap(pivot_mcc, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0], 
                vmin=0.4, vmax=0.8, center=0.6)
    axes[0].set_title('MCC by Dataset and VIF Threshold', fontsize=12, fontweight='bold')
    
    # Feature count heatmap
    pivot_features = feature_df.pivot(index='Dataset', columns='VIF_Threshold', values='Final')
    sns.heatmap(pivot_features, annot=True, fmt='.0f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Final Feature Count by Dataset and VIF Threshold', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/vif_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/vif_heatmaps.png")
    
    # ===== PRINT SUMMARY =====
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    
    print("\n[Feature Selection by VIF Threshold]")
    print(feature_df.to_string(index=False))
    
    print("\n[Classification Metrics by VIF Threshold]")
    print(results_df[['VIF_Threshold', 'Dataset', 'MCC', 'Sensitivity', 'Specificity']].to_string(index=False))
    
    # Best VIF threshold
    avg_mcc = results_df.groupby('VIF_Threshold')['MCC'].mean()
    best_vif = avg_mcc.idxmax()
    print(f"\n[RECOMMENDATION] Best VIF Threshold: {best_vif} (Avg MCC = {avg_mcc[best_vif]:.3f})")
    
    print("\n" + "="*70)
    print("[SUCCESS] SIMULATION COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - results/vif_simulation_results.csv")
    print("  - results/vif_feature_selection_comparison.png")
    print("  - results/vif_metrics_comparison.png")
    print("  - results/vif_heatmaps.png")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()

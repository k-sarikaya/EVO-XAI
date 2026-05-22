#!/usr/bin/env python
"""
EVO-XAI Enhanced Pipeline - Simplified Version
Three-Stage Feature Selection (No ADASYN during NSGA-II)
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

print("[OK] All libraries imported")


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
        print(f"  [OK] {name}: {len(result)} samples, {len(X_encoded.columns)} features")
    
    return datasets


def chi_square_filter(X, y, p_threshold=0.05):
    """Stage 1: Chi-square filtering"""
    print("\n" + "="*60)
    print("STAGE 1: CHI-SQUARE FILTERING")
    print("="*60)
    
    X_pos = X - X.min().min() + 0.001
    
    try:
        chi2_stats, p_values = chi2(X_pos, y)
        
        results = pd.DataFrame({
            'Feature': X.columns,
            'Chi2_Stat': chi2_stats,
            'P_Value': p_values,
            'Significant': p_values < p_threshold
        }).sort_values('P_Value')
        
        print(f"\nOriginal features: {len(X.columns)}")
        
        for _, row in results.iterrows():
            status = "[KEEP]" if row['Significant'] else "[DROP]"
            print(f"  {status} {row['Feature']}: Chi2={row['Chi2_Stat']:.2f}, p={row['P_Value']:.4f}")
        
        selected = results[results['Significant']]['Feature'].tolist()
        
        if len(selected) < 5:
            selected = results.head(max(5, len(selected)))['Feature'].tolist()
        
        print(f"\nResult: {len(selected)}/{len(X.columns)} features retained")
        
        return selected, results
        
    except Exception as e:
        print(f"  [WARN] Chi-square failed: {e}")
        return list(X.columns), None


def multicollinearity_filter(X, feature_names, vif_threshold=10, corr_threshold=0.95, min_features=5):
    """Stage 2: VIF and correlation check (with minimum feature guarantee)"""
    print("\n" + "="*60)
    print("STAGE 2: MULTICOLLINEARITY CHECK")
    print("="*60)
    
    print(f"\nStarting with {len(feature_names)} features")
    print(f"Minimum features to retain: {min_features}")
    
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X[feature_names].copy()
    
    X_df = X_df + 0.001
    
    # VIF Analysis
    print("\n--- VIF Analysis ---")
    
    try:
        vif_data = []
        for i, col in enumerate(X_df.columns):
            try:
                vif = variance_inflation_factor(X_df.values, i)
                vif_data.append({'Feature': col, 'VIF': vif})
            except:
                vif_data.append({'Feature': col, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        
        for _, row in vif_df.iterrows():
            status = "[HIGH VIF]" if row['VIF'] > vif_threshold else "[OK]"
            print(f"  {status} {row['Feature']}: VIF = {row['VIF']:.2f}")
        
        high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()
        
    except Exception as e:
        print(f"  [WARN] VIF failed: {e}")
        vif_df = pd.DataFrame()
        high_vif_features = []
    
    # Correlation Analysis
    print("\n--- Correlation Analysis ---")
    
    corr_matrix = X_df.corr()
    
    corr_pairs = []
    redundant_from_corr = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > corr_threshold:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: r = {corr:.3f}")
                redundant_from_corr.add(feature_names[j])
    
    if not redundant_from_corr:
        print(f"  [OK] No feature pairs with |r| > {corr_threshold}")
    
    all_redundant = set(high_vif_features) | redundant_from_corr
    independent_features = [f for f in feature_names if f not in all_redundant]
    
    # Ensure minimum features are retained
    if len(independent_features) < min_features:
        print(f"\n  [WARN] Would reduce to {len(independent_features)} features, below minimum {min_features}")
        # Keep top features by VIF (lowest VIF = least redundant)
        if len(vif_df) > 0:
            sorted_by_vif = vif_df.sort_values('VIF')['Feature'].tolist()
            independent_features = sorted_by_vif[:min_features]
        else:
            independent_features = feature_names[:min_features]
        print(f"  [OK] Keeping top {min_features} features by lowest VIF")
    
    print(f"\n--- Summary ---")
    print(f"Removed due to high VIF: {len(high_vif_features)}")
    print(f"Removed due to correlation: {len(redundant_from_corr)}")
    print(f"Independent features: {len(independent_features)}")
    
    return independent_features, vif_df, list(all_redundant)


def plot_correlation_matrix(X, feature_names, save_path):
    """Plot correlation heatmap"""
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X[feature_names]
    
    corr = X_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


# Setup DEAP
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def nsga2_selection(X, y, features, pop_size=15, gens=10):
    """Stage 3: NSGA-II optimization (simplified evaluation)"""
    print("\n" + "="*60)
    print("STAGE 3: NSGA-II OPTIMIZATION")
    print("="*60)
    
    n = len(features)
    print(f"\nOptimizing {n} features")
    
    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.Individual, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    
    def evaluate(ind):
        """Simple train/test split evaluation"""
        sel = [i for i, g in enumerate(ind) if g == 1]
        if not sel:
            return (1.0, n)
        try:
            X_sel = X.iloc[:, sel].values
            y_arr = y.values
            
            # Check if we can stratify
            min_class = min(np.bincount(y_arr.astype(int)))
            if min_class < 2:
                # Can't stratify, use random split
                X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42)
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
            
            clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            return (1 - f1_score(y_te, y_pred), len(sel))
        except Exception as e:
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
        print(f"  Gen {g+1}/{gens}")
    
    best = min(pop, key=lambda x: x.fitness.values[0])
    err, nf = best.fitness.values
    selected_features = [features[i] for i, g in enumerate(best) if g == 1]
    
    print(f"\nBest: {int(nf)} features, F1 = {(1-err)*100:.1f}%")
    print(f"Selected: {selected_features}")
    
    return selected_features, 1-err


def evaluate_model(X, y, name):
    """5-fold CV evaluation"""
    print(f"\n    Evaluating {name}...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'mcc': [], 'sens': [], 'spec': [], 'f1': [], 'auc': []}
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        
        # Train ensemble
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        xgb_m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
        svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        
        rf.fit(X_tr, y_tr)
        xgb_m.fit(X_tr, y_tr)
        svm.fit(X_tr, y_tr)
        
        # Meta-learner
        meta_train_X = np.column_stack([
            rf.predict_proba(X_tr)[:, 1],
            xgb_m.predict_proba(X_tr)[:, 1],
            svm.predict_proba(X_tr)[:, 1]
        ])
        meta = LogisticRegression(random_state=42, max_iter=1000)
        meta.fit(meta_train_X, y_tr)
        
        # Predict
        meta_test_X = np.column_stack([
            rf.predict_proba(X_te)[:, 1],
            xgb_m.predict_proba(X_te)[:, 1],
            svm.predict_proba(X_te)[:, 1]
        ])
        y_prob = meta.predict_proba(meta_test_X)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        metrics['mcc'].append(matthews_corrcoef(y_te, y_pred))
        metrics['sens'].append(tp/(tp+fn) if tp+fn > 0 else 0)
        metrics['spec'].append(tn/(tn+fp) if tn+fp > 0 else 0)
        metrics['f1'].append(f1_score(y_te, y_pred))
        metrics['auc'].append(roc_auc_score(y_te, y_prob))
        
        print(f"      Fold {fold}: MCC={metrics['mcc'][-1]:.3f}")
    
    results = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    print(f"    Summary: MCC={results['mcc'][0]:.3f}, Sens={results['sens'][0]:.1%}, Spec={results['spec'][0]:.1%}")
    return results


def main():
    print("\n" + "="*70)
    print("EVO-XAI ENHANCED PIPELINE WITH MULTICOLLINEARITY CHECK")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    print("\n[PHASE 1] Loading Datasets")
    datasets = load_datasets()
    
    if not datasets:
        print("[ERROR] No datasets loaded!")
        return
    
    all_results = {}
    feature_history = {}
    
    for name, df in datasets.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING: {name.upper()}")
        print(f"{'='*70}")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        original_n = len(X.columns)
        
        # Stage 1: Chi-Square
        stage1_features, _ = chi_square_filter(X, y)
        X_stage1 = X[stage1_features]
        
        # Stage 2: Multicollinearity
        stage2_features, vif_df, removed = multicollinearity_filter(X_stage1, stage1_features)
        X_stage2 = X_stage1[stage2_features]
        
        plot_correlation_matrix(X_stage1, stage1_features, f'results/correlation_{name}.png')
        
        # Stage 3: NSGA-II
        stage3_features, best_f1 = nsga2_selection(X_stage2, y, stage2_features)
        X_stage3 = X_stage2[stage3_features]
        
        feature_history[name] = {
            'original': original_n,
            'after_chi_sq': len(stage1_features),
            'after_vif': len(stage2_features),
            'final': len(stage3_features),
            'features': stage3_features,
            'reduction': (1 - len(stage3_features)/original_n) * 100
        }
        
        print(f"\n{'='*60}")
        print(f"SUMMARY - {name.upper()}")
        print(f"{'='*60}")
        print(f"Original:        {original_n}")
        print(f"After Chi-sq:    {len(stage1_features)}")
        print(f"After VIF/Corr:  {len(stage2_features)}")
        print(f"After NSGA-II:   {len(stage3_features)}")
        print(f"Reduction:       {feature_history[name]['reduction']:.1f}%")
        print(f"Final features:  {stage3_features}")
        
        # Evaluate
        print(f"\n[EVALUATION]")
        results = evaluate_model(X_stage3, y, name)
        all_results[name] = results
    
    # Save results
    print("\n" + "="*70)
    print("[SAVING RESULTS]")
    print("="*70)
    
    # Feature selection summary
    rows = []
    for name, hist in feature_history.items():
        rows.append({
            'Dataset': name,
            'Original': hist['original'],
            'After_ChiSq': hist['after_chi_sq'],
            'After_VIF': hist['after_vif'],
            'Final': hist['final'],
            'Reduction_%': hist['reduction'],
            'Features': ', '.join(hist['features'])
        })
    
    pd.DataFrame(rows).to_csv('results/feature_selection_summary.csv', index=False)
    print("  [OK] results/feature_selection_summary.csv")
    
    # Performance metrics
    perf_rows = []
    for name, res in all_results.items():
        perf_rows.append({
            'Dataset': name,
            'MCC_mean': res['mcc'][0],
            'MCC_std': res['mcc'][1],
            'Sens_mean': res['sens'][0],
            'Spec_mean': res['spec'][0],
            'F1_mean': res['f1'][0],
            'AUC_mean': res['auc'][0]
        })
    
    pd.DataFrame(perf_rows).to_csv('results/performance_metrics.csv', index=False)
    print("  [OK] results/performance_metrics.csv")
    
    # Final summary
    print("\n" + "="*70)
    print("[SUCCESS] PIPELINE COMPLETE")
    print("="*70)
    
    print("\nFeature Selection:")
    for name, hist in feature_history.items():
        print(f"  {name}: {hist['original']} -> {hist['final']} ({hist['reduction']:.1f}% reduction)")
    
    print("\nPerformance:")
    for name, res in all_results.items():
        print(f"  {name}: MCC={res['mcc'][0]:.3f}, Sens={res['sens'][0]:.1%}, Spec={res['spec'][0]:.1%}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()

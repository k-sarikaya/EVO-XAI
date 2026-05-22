#!/usr/bin/env python
"""
Evolutionary Algorithm Comparison Study
Compares: NSGA-II, Differential Evolution (DE), Hybrid PSO+DE
VIF Threshold: 8
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
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

from deap import base, creator, tools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("[OK] Evolutionary Algorithm Comparison - Libraries loaded")


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
# FEATURE SELECTION FUNCTIONS (Chi-Square and VIF)
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


def multicollinearity_filter(X, feature_names, vif_threshold=8, min_features=5):
    """Stage 2: VIF and correlation check with VIF=8"""
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X[feature_names].copy()
    
    X_df = X_df + 0.001
    
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
    
    corr_matrix = X_df.corr()
    redundant_from_corr = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.95:
                redundant_from_corr.add(feature_names[j])
    
    all_redundant = set(high_vif_features) | redundant_from_corr
    independent_features = [f for f in feature_names if f not in all_redundant]
    
    if len(independent_features) < min_features:
        if len(vif_df) > 0:
            sorted_by_vif = vif_df.sort_values('VIF')['Feature'].tolist()
            independent_features = sorted_by_vif[:min_features]
        else:
            independent_features = feature_names[:min_features]
    
    return independent_features


# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def fitness_function(individual, X, y, features):
    """Evaluate feature subset fitness"""
    # Convert to binary selection
    if isinstance(individual, np.ndarray):
        sel = [i for i, g in enumerate(individual) if g > 0.5]
    else:
        sel = [i for i, g in enumerate(individual) if g == 1]
    
    if not sel:
        return 0.0, len(features)
    
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
        f1 = f1_score(y_te, y_pred)
        
        return f1, len(sel)
    except:
        return 0.0, len(sel)


# ============================================================================
# ALGORITHM 1: NSGA-II
# ============================================================================

# Setup DEAP once
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Max F1, Min features
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def nsga2_selection(X, y, features, pop_size=20, gens=15):
    """NSGA-II multi-objective optimization"""
    n = len(features)
    
    if n == 0:
        return [], 0.0, 0
    
    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.Individual, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    
    def evaluate(ind):
        f1, nf = fitness_function(ind, X, y, features)
        return (f1, nf)
    
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
    
    # Best: max F1
    best = max(pop, key=lambda x: x.fitness.values[0])
    f1, nf = best.fitness.values
    selected_features = [features[i] for i, g in enumerate(best) if g == 1]
    
    return selected_features, f1, int(nf)


# ============================================================================
# ALGORITHM 2: DIFFERENTIAL EVOLUTION (DE)
# ============================================================================

def differential_evolution_selection(X, y, features, pop_size=20, gens=15, F=0.8, CR=0.9):
    """
    Differential Evolution for feature selection
    F: Differential weight (mutation scaling factor)
    CR: Crossover probability
    """
    n = len(features)
    
    if n == 0:
        return [], 0.0, 0
    
    # Initialize population (continuous [0,1])
    population = np.random.rand(pop_size, n)
    
    # Evaluate initial population
    fitness = []
    for ind in population:
        f1, nf = fitness_function(ind, X, y, features)
        fitness.append(f1 - 0.01 * nf)  # Penalize more features
    fitness = np.array(fitness)
    
    for g in range(gens):
        for i in range(pop_size):
            # Select 3 random individuals (different from i)
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = np.copy(population[i])
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Selection
            trial_f1, trial_nf = fitness_function(trial, X, y, features)
            trial_fitness = trial_f1 - 0.01 * trial_nf
            
            if trial_fitness > fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
    
    # Find best individual
    best_idx = np.argmax(fitness)
    best = population[best_idx]
    
    selected_features = [features[i] for i in range(n) if best[i] > 0.5]
    
    if not selected_features:
        selected_features = [features[np.argmax(best)]]
    
    f1, nf = fitness_function(best, X, y, features)
    
    return selected_features, f1, len(selected_features)


# ============================================================================
# ALGORITHM 3: HYBRID PSO+DE
# ============================================================================

def hybrid_pso_de_selection(X, y, features, pop_size=20, gens=15, w=0.7, c1=1.5, c2=1.5, F=0.5, CR=0.5):
    """
    Hybrid PSO + DE for feature selection
    PSO parameters: w (inertia), c1 (cognitive), c2 (social)
    DE parameters: F (mutation scale), CR (crossover rate)
    """
    n = len(features)
    
    if n == 0:
        return [], 0.0, 0
    
    # Initialize particles
    positions = np.random.rand(pop_size, n)
    velocities = np.random.rand(pop_size, n) * 0.1 - 0.05
    
    # Personal best
    p_best = positions.copy()
    p_best_fitness = np.zeros(pop_size)
    
    for i in range(pop_size):
        f1, nf = fitness_function(positions[i], X, y, features)
        p_best_fitness[i] = f1 - 0.01 * nf
    
    # Global best
    g_best_idx = np.argmax(p_best_fitness)
    g_best = p_best[g_best_idx].copy()
    g_best_fitness = p_best_fitness[g_best_idx]
    
    for g in range(gens):
        for i in range(pop_size):
            # ===== PSO Update =====
            r1, r2 = np.random.rand(2)
            
            # Update velocity
            velocities[i] = (w * velocities[i] + 
                            c1 * r1 * (p_best[i] - positions[i]) + 
                            c2 * r2 * (g_best - positions[i]))
            
            # Update position
            pso_position = positions[i] + velocities[i]
            pso_position = np.clip(pso_position, 0, 1)
            
            # ===== DE Mutation =====
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) >= 3:
                r1_idx, r2_idx, r3_idx = np.random.choice(candidates, 3, replace=False)
                de_mutant = positions[r1_idx] + F * (positions[r2_idx] - positions[r3_idx])
                de_mutant = np.clip(de_mutant, 0, 1)
                
                # DE Crossover
                trial = np.copy(pso_position)
                for j in range(n):
                    if np.random.rand() < CR:
                        trial[j] = de_mutant[j]
            else:
                trial = pso_position
            
            # ===== Hybrid: Choose better =====
            pso_f1, pso_nf = fitness_function(pso_position, X, y, features)
            pso_fit = pso_f1 - 0.01 * pso_nf
            
            trial_f1, trial_nf = fitness_function(trial, X, y, features)
            trial_fit = trial_f1 - 0.01 * trial_nf
            
            if trial_fit > pso_fit:
                positions[i] = trial
                current_fitness = trial_fit
            else:
                positions[i] = pso_position
                current_fitness = pso_fit
            
            # Update personal best
            if current_fitness > p_best_fitness[i]:
                p_best[i] = positions[i].copy()
                p_best_fitness[i] = current_fitness
                
                # Update global best
                if current_fitness > g_best_fitness:
                    g_best = positions[i].copy()
                    g_best_fitness = current_fitness
    
    # Get selected features
    selected_features = [features[i] for i in range(n) if g_best[i] > 0.5]
    
    if not selected_features:
        selected_features = [features[np.argmax(g_best)]]
    
    f1, nf = fitness_function(g_best, X, y, features)
    
    return selected_features, f1, len(selected_features)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(X, y):
    """5-fold CV evaluation"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'mcc': [], 'sens': [], 'spec': [], 'f1': [], 'auc': []}
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        
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
    print("EVOLUTIONARY ALGORITHM COMPARISON STUDY")
    print("Algorithms: NSGA-II, Differential Evolution, Hybrid PSO+DE")
    print("VIF Threshold: 8")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    # Load Data
    print("\n[1] Loading datasets...")
    datasets = load_datasets()
    print(f"    Loaded: {list(datasets.keys())}")
    
    # Algorithms to compare
    algorithms = {
        'NSGA-II': nsga2_selection,
        'DE': differential_evolution_selection,
        'Hybrid_PSO_DE': hybrid_pso_de_selection
    }
    
    # Store results
    all_results = []
    feature_selection_results = []
    
    for name, df in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {name.upper()}")
        print(f"{'='*70}")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        original_n = len(X.columns)
        
        # Stage 1: Chi-Square
        stage1_features = chi_square_filter(X, y)
        X_stage1 = X[stage1_features]
        chi_n = len(stage1_features)
        print(f"  Chi-Square: {original_n} -> {chi_n} features")
        
        # Stage 2: VIF with threshold=8
        stage2_features = multicollinearity_filter(X_stage1, stage1_features, vif_threshold=8)
        X_stage2 = X_stage1[stage2_features]
        vif_n = len(stage2_features)
        print(f"  VIF Filter: {chi_n} -> {vif_n} features")
        
        # Stage 3: Compare algorithms
        for algo_name, algo_func in algorithms.items():
            print(f"\n  [{algo_name}]")
            
            # Run optimization
            selected_features, best_f1, n_features = algo_func(X_stage2, y, stage2_features)
            
            if not selected_features:
                selected_features = stage2_features[:1]
            
            X_final = X_stage2[selected_features]
            final_n = len(selected_features)
            
            print(f"    Features: {vif_n} -> {final_n}")
            print(f"    Selected: {selected_features}")
            
            # Evaluate
            if final_n > 0:
                results = evaluate_model(X_final, y)
                print(f"    MCC={results['mcc'][0]:.3f}, Sens={results['sens'][0]:.1%}, Spec={results['spec'][0]:.1%}")
            else:
                results = {'mcc': (0, 0), 'sens': (0, 0), 'spec': (0, 0), 'f1': (0, 0), 'auc': (0, 0)}
            
            # Store results
            all_results.append({
                'Algorithm': algo_name,
                'Dataset': name,
                'Original': original_n,
                'After_ChiSq': chi_n,
                'After_VIF': vif_n,
                'Final': final_n,
                'Reduction_%': (1 - final_n/original_n) * 100,
                'MCC': results['mcc'][0],
                'MCC_std': results['mcc'][1],
                'Sensitivity': results['sens'][0],
                'Specificity': results['spec'][0],
                'F1': results['f1'][0],
                'AUC': results['auc'][0]
            })
            
            feature_selection_results.append({
                'Algorithm': algo_name,
                'Dataset': name,
                'Features': ', '.join(selected_features),
                'Count': final_n
            })
    
    # Create DataFrames
    results_df = pd.DataFrame(all_results)
    features_df = pd.DataFrame(feature_selection_results)
    
    # Save results
    results_df.to_csv('results/algorithm_comparison_results.csv', index=False)
    features_df.to_csv('results/algorithm_feature_selection.csv', index=False)
    print("\n[OK] Saved: results/algorithm_comparison_results.csv")
    print("[OK] Saved: results/algorithm_feature_selection.csv")
    
    # ===== CREATE PLOTS =====
    print("\n[2] Creating plots...")
    
    # Plot 1: Feature Count Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets_list = list(datasets.keys())
    x = np.arange(len(datasets_list))
    width = 0.25
    
    for i, algo in enumerate(['NSGA-II', 'DE', 'Hybrid_PSO_DE']):
        counts = results_df[results_df['Algorithm'] == algo]['Final'].values
        bars = ax.bar(x + i*width - width, counts, width, label=algo)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Selection Comparison by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets_list])
    ax.legend()
    ax.set_ylim(0, max(results_df['Final']) + 2)
    
    plt.tight_layout()
    plt.savefig('results/algorithm_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/algorithm_feature_comparison.png")
    
    # Plot 2: Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['MCC', 'Sensitivity', 'Specificity']
    colors = {'NSGA-II': '#3498db', 'DE': '#e74c3c', 'Hybrid_PSO_DE': '#2ecc71'}
    
    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        
        x = np.arange(len(datasets_list))
        width = 0.25
        
        for i, algo in enumerate(['NSGA-II', 'DE', 'Hybrid_PSO_DE']):
            values = results_df[results_df['Algorithm'] == algo][metric].values
            bars = ax.bar(x + i*width - width, values, width, label=algo, color=colors[algo])
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Algorithm')
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in datasets_list])
        ax.legend()
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Classification Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/algorithm_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/algorithm_metrics_comparison.png")
    
    # Plot 3: Heatmap comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MCC Heatmap
    pivot_mcc = results_df.pivot(index='Dataset', columns='Algorithm', values='MCC')
    pivot_mcc = pivot_mcc[['NSGA-II', 'DE', 'Hybrid_PSO_DE']]
    sns.heatmap(pivot_mcc, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0], 
                vmin=0.4, vmax=0.8, center=0.6)
    axes[0].set_title('MCC by Dataset and Algorithm', fontsize=12, fontweight='bold')
    
    # Feature count heatmap
    pivot_features = results_df.pivot(index='Dataset', columns='Algorithm', values='Final')
    pivot_features = pivot_features[['NSGA-II', 'DE', 'Hybrid_PSO_DE']]
    sns.heatmap(pivot_features, annot=True, fmt='.0f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Feature Count by Dataset and Algorithm', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/algorithm_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [OK] results/algorithm_heatmaps.png")
    
    # ===== PRINT SUMMARY TABLES =====
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\n[FEATURE SELECTION COMPARISON]")
    print("-"*70)
    feature_summary = results_df[['Algorithm', 'Dataset', 'Original', 'After_ChiSq', 'After_VIF', 'Final', 'Reduction_%']]
    print(feature_summary.to_string(index=False))
    
    print("\n[CLASSIFICATION METRICS COMPARISON]")
    print("-"*70)
    metrics_summary = results_df[['Algorithm', 'Dataset', 'MCC', 'Sensitivity', 'Specificity', 'F1', 'AUC']]
    print(metrics_summary.to_string(index=False))
    
    print("\n[SELECTED FEATURES]")
    print("-"*70)
    print(features_df.to_string(index=False))
    
    # Overall ranking
    print("\n[ALGORITHM RANKING BY AVERAGE MCC]")
    print("-"*70)
    avg_mcc = results_df.groupby('Algorithm')['MCC'].mean().sort_values(ascending=False)
    for rank, (algo, mcc) in enumerate(avg_mcc.items(), 1):
        print(f"  {rank}. {algo}: {mcc:.4f}")
    
    best_algo = avg_mcc.idxmax()
    print(f"\n[RECOMMENDATION] Best Algorithm: {best_algo} (Avg MCC = {avg_mcc[best_algo]:.4f})")
    
    print("\n" + "="*70)
    print("[SUCCESS] ALGORITHM COMPARISON COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - results/algorithm_comparison_results.csv")
    print("  - results/algorithm_feature_selection.csv")
    print("  - results/algorithm_feature_comparison.png")
    print("  - results/algorithm_metrics_comparison.png")
    print("  - results/algorithm_heatmaps.png")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()

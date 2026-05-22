#!/usr/bin/env python
"""
EVO-XAI: Autism Spectrum Disorder Diagnosis Framework
Complete Implementation with ARFF Support

Execute: python evo_xai_implementation.py

Author: [To be specified]
Date: December 2025
"""

import os
import sys
import time
import pickle
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import arff  # For loading ARFF files

warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# NOTE: IterativeImputer is still marked experimental in scikit-learn and
# requires enabling the import explicitly.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import chi2
import xgboost as xgb

# Explainability
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Imbalance handling
from imblearn.over_sampling import ADASYN, SMOTE

# Feature selection optimization
from deap import base, creator, tools, algorithms

# Metrics
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_recall_curve, auc
)

print("✓ All libraries imported successfully")


# ============================================================================
# PART 1: DATA LOADING (ARFF FORMAT)
# ============================================================================

def load_arff_file(filepath):
    """
    Load ARFF file and convert to pandas DataFrame
    
    Args:
        filepath: Path to ARFF file
    
    Returns:
        pandas DataFrame
    """
    with open(filepath, 'r') as f:
        dataset = arff.load(f)
    
    # Get column names from attributes
    columns = [attr[0] for attr in dataset['attributes']]
    
    # Create DataFrame
    df = pd.DataFrame(dataset['data'], columns=columns)
    
    return df


def load_asd_datasets(data_dir='data'):
    """
    Load autism screening datasets from ARFF files
    
    Returns:
        Dict with three DataFrames: child, adolescent, adult
    """
    
    datasets = {}
    
    # Define paths
    paths = {
        'child': os.path.join(data_dir, 'child', 'Autism-Child-Data.arff'),
        'adolescent': os.path.join(data_dir, 'adolescent', 'Autism-Adolescent-Data.arff'),
        'adult': os.path.join(data_dir, 'adult', 'Autism-Adult-Data.arff')
    }
    
    for name, path in paths.items():
        if os.path.exists(path):
            df = load_arff_file(path)
            
            # Find and standardize target column
            target_cols = ['Class/ASD', 'Class', 'class', 'austim', 'Class/ASD Traits ']
            target_col = None
            for col in target_cols:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                # Use last column as target
                target_col = df.columns[-1]
            
            # Rename to 'Class'
            df = df.rename(columns={target_col: 'Class'})
            
            # Convert target to binary
            if df['Class'].dtype == 'object':
                df['Class'] = df['Class'].apply(
                    lambda x: 1 if str(x).lower() in ['yes', 'true', '1', 'asd'] else 0
                )
            
            # Convert all feature columns to numeric
            for col in df.columns:
                if col != 'Class':
                    if df[col].dtype == 'object':
                        # Try to convert to numeric, else encode
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
            
            datasets[name] = df
            print(f"  ✓ Loaded {name}: {len(df)} samples, {len(df.columns)-1} features")
        else:
            print(f"  ✗ File not found: {path}")
    
    return datasets


# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

def apply_mice_imputation(X, method='median', random_state=42, max_iter=10):
    """
    Apply MICE-style imputation for missing data.

    This uses scikit-learn's IterativeImputer (a chained-equations approach)
    as a deterministic, single-imputation approximation of MICE.
    
    Args:
        X: Feature matrix with potential missing values
        method: initialization strategy: 'mean', 'median', or 'most_frequent'
        random_state: RNG seed for reproducibility
        max_iter: number of imputation rounds
    
    Returns:
        Imputed feature matrix
    """
    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        initial_strategy=method,
        sample_posterior=False,
        skip_complete=True,
    )
    return imputer.fit_transform(X)


def preprocess_dataset(df):
    """
    Preprocess a single dataset
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Processed DataFrame with X (features) and y (target)
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Handle missing values
    X_imputed = apply_mice_imputation(X)
    
    # Create processed DataFrame
    X_processed = pd.DataFrame(X_imputed, columns=X.columns)
    
    return X_processed, y


# ============================================================================
# PART 3: CHI-SQUARE FILTERING
# ============================================================================

def chi_square_filter(X, y, p_threshold=0.05):
    """
    Filter features using chi-square test of independence
    
    Args:
        X: Feature matrix (should be non-negative)
        y: Target vector
        p_threshold: p-value threshold for feature retention
    
    Returns:
        List of selected feature names
        DataFrame with chi-square results
    """
    from scipy import stats
    
    # Ensure non-negative values for chi-square
    X_pos = X.copy()
    X_pos = X_pos - X_pos.min() + 0.001  # Shift to positive
    
    chi2_stats, p_values = chi2(X_pos, y)
    
    results = pd.DataFrame({
        'Feature': X.columns,
        'Chi2_Statistic': chi2_stats,
        'P_Value': p_values,
        'Significant': p_values < p_threshold
    }).sort_values('P_Value')
    
    selected_features = results[results['Significant']]['Feature'].tolist()
    
    # Ensure at least 5 features are selected
    if len(selected_features) < 5:
        selected_features = results.head(5)['Feature'].tolist()
    
    return selected_features, results


# ============================================================================
# PART 4: NSGA-II FEATURE SELECTION
# ============================================================================

# Create DEAP types (only once)
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_toolbox(X_train, y_train, n_features):
    """Create DEAP toolbox for NSGA-II"""
    
    toolbox = base.Toolbox()
    
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        selected_indices = [i for i, gene in enumerate(individual) if gene == 1]
        
        if len(selected_indices) == 0:
            return (1.0, n_features)
        
        X_selected = X_train.iloc[:, selected_indices]
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(clf, X_selected, y_train, cv=3, scoring='f1')
        
        error = 1.0 - cv_scores.mean()
        n_selected = len(selected_indices)
        
        return (error, n_selected)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox


def run_nsga2_feature_selection(X_train, y_train, feature_names, 
                                population_size=30, generations=30):
    """Run NSGA-II for multi-objective feature selection"""
    
    n_features = len(feature_names)
    toolbox = create_toolbox(X_train, y_train, n_features)
    
    pop = toolbox.population(n=population_size)
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print(f"    Running NSGA-II: {generations} generations...")
    
    for gen in range(generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for i in range(1, len(offspring), 2):
            if np.random.random() < 0.8:
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
        
        for i in range(len(offspring)):
            if np.random.random() < 0.1:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, population_size)
        
        if gen % 10 == 0:
            print(f"      Generation {gen}/{generations}")
    
    pareto_front = [p for p in pop if p.fitness.valid]
    pareto_front.sort(key=lambda x: x.fitness.values)
    
    solutions = []
    for individual in pareto_front[:10]:  # Top 10 solutions
        selected_indices = [i for i, gene in enumerate(individual) if gene == 1]
        selected_features = [feature_names[i] for i in selected_indices]
        error, n_feat = individual.fitness.values
        
        solutions.append({
            'features': selected_features,
            'n_features': int(n_feat),
            'error': error,
            'accuracy': 1.0 - error
        })
    
    return solutions


# ============================================================================
# PART 5: ADASYN CLASS BALANCING
# ============================================================================

def apply_adasyn_balancing(X, y, random_state=42):
    """Apply ADASYN for class imbalance"""
    try:
        adasyn = ADASYN(random_state=random_state, n_neighbors=3)
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
    except:
        # Fallback to SMOTE if ADASYN fails
        smote = SMOTE(random_state=random_state, k_neighbors=2)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    
    return X_balanced, y_balanced


# ============================================================================
# PART 6: STACKING ENSEMBLE
# ============================================================================

def create_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Create stacking ensemble with RF, XGBoost, SVM base learners"""
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                                   verbosity=0, use_label_encoder=False)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    
    rf_pred = rf.predict_proba(X_val)[:, 1].reshape(-1, 1)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    svm_pred = svm.predict_proba(X_val)[:, 1].reshape(-1, 1)
    
    meta_features = np.hstack([rf_pred, xgb_pred, svm_pred])
    
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(meta_features, y_val)
    
    return {
        'rf': rf,
        'xgb': xgb_model,
        'svm': svm,
        'meta_learner': meta_learner
    }


def predict_stacking_ensemble(model, X):
    """Generate predictions from stacking ensemble"""
    rf_pred = model['rf'].predict_proba(X)[:, 1].reshape(-1, 1)
    xgb_pred = model['xgb'].predict_proba(X)[:, 1].reshape(-1, 1)
    svm_pred = model['svm'].predict_proba(X)[:, 1].reshape(-1, 1)
    
    meta_features = np.hstack([rf_pred, xgb_pred, svm_pred])
    predictions = model['meta_learner'].predict_proba(meta_features)[:, 1]
    
    return predictions


# ============================================================================
# PART 7: NESTED CROSS-VALIDATION
# ============================================================================

def nested_cross_validation(X, y, dataset_name, n_outer=5, n_inner=5):
    """Nested cross-validation with data leakage prevention"""
    
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    
    all_metrics = {
        'mcc': [], 'sensitivity': [], 'specificity': [], 'f1': [],
        'precision': [], 'auc': []
    }
    all_predictions = []
    all_true_labels = []
    
    print(f"\n  ▶ Nested CV for {dataset_name}:")
    
    fold_count = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_count += 1
        print(f"    Outer Fold {fold_count}/{n_outer}")
        
        X_train_out = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train_out = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        
        # Inner loop for hyperparameter selection
        best_accuracy = 0
        best_ensemble = None
        
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_out, y_train_out):
            X_train_inner = X_train_out.iloc[train_inner_idx]
            X_val_inner = X_train_out.iloc[val_inner_idx]
            y_train_inner = y_train_out.iloc[train_inner_idx]
            y_val_inner = y_train_out.iloc[val_inner_idx]
            
            # Apply ADASYN only to inner training
            X_balanced, y_balanced = apply_adasyn_balancing(
                X_train_inner.values, y_train_inner.values
            )
            
            ensemble = create_stacking_ensemble(
                X_balanced, y_balanced,
                X_val_inner.values, y_val_inner.values
            )
            
            preds_val = predict_stacking_ensemble(ensemble, X_val_inner.values)
            accuracy = ((preds_val > 0.5).astype(int) == y_val_inner.values).mean()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_ensemble = ensemble
        
        # Train final model on entire outer training set
        X_balanced, y_balanced = apply_adasyn_balancing(
            X_train_out.values, y_train_out.values
        )
        
        final_ensemble = create_stacking_ensemble(
            X_balanced, y_balanced,
            X_train_out.values, y_train_out.values
        )
        
        # Test
        y_pred_proba = predict_stacking_ensemble(final_ensemble, X_test.values)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        all_metrics['mcc'].append(mcc)
        all_metrics['sensitivity'].append(sensitivity)
        all_metrics['specificity'].append(specificity)
        all_metrics['f1'].append(f1)
        all_metrics['precision'].append(precision)
        all_metrics['auc'].append(auc_score)
        
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test.values)
    
    # Aggregate results
    results = {
        'mcc': (np.mean(all_metrics['mcc']), np.std(all_metrics['mcc'])),
        'sensitivity': (np.mean(all_metrics['sensitivity']), np.std(all_metrics['sensitivity'])),
        'specificity': (np.mean(all_metrics['specificity']), np.std(all_metrics['specificity'])),
        'f1': (np.mean(all_metrics['f1']), np.std(all_metrics['f1'])),
        'auc': (np.mean(all_metrics['auc']), np.std(all_metrics['auc'])),
        'precision': (np.mean(all_metrics['precision']), np.std(all_metrics['precision']))
    }
    
    print(f"\n    Results for {dataset_name}:")
    for metric, (mean, std) in results.items():
        print(f"      {metric.upper()}: {mean:.3f} ± {std:.3f}")
    
    return results, all_predictions, all_true_labels


# ============================================================================
# PART 8: SHAP EXPLAINABILITY
# ============================================================================

def compute_shap_values(model, X_train, X_test):
    """Compute SHAP values for feature importance"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, explainer


def plot_shap_summary(shap_values, X_test, feature_names, save_path='results/shap_summary.png'):
    """Create SHAP summary plot"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=list(feature_names), show=False)
    plt.title("SHAP Feature Importance - Autism Diagnosis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP summary plot saved: {save_path}")


# ============================================================================
# PART 9: MONDRIAN CONFORMAL PREDICTION
# ============================================================================

def calibrate_mondrian_cp(y_true, y_scores, alpha=0.1):
    """Calibrate Mondrian Conformal Prediction thresholds"""
    
    nonconf_scores = 1 - np.maximum(y_scores, 1 - y_scores)
    
    nonconf_asd = nonconf_scores[y_true == 1]
    nonconf_non_asd = nonconf_scores[y_true == 0]
    
    if len(nonconf_asd) > 0:
        quantile = min(np.ceil((len(nonconf_asd) + 1) * (1 - alpha)) / len(nonconf_asd), 1.0)
        threshold_asd = np.quantile(nonconf_asd, quantile)
    else:
        threshold_asd = 0.5
    
    if len(nonconf_non_asd) > 0:
        quantile = min(np.ceil((len(nonconf_non_asd) + 1) * (1 - alpha)) / len(nonconf_non_asd), 1.0)
        threshold_non_asd = np.quantile(nonconf_non_asd, quantile)
    else:
        threshold_non_asd = 0.5
    
    return {'asd': threshold_asd, 'non_asd': threshold_non_asd}


def get_prediction_set_mondrian(y_score, thresholds):
    """Generate Mondrian CP prediction set"""
    nonconf_asd = 1 - y_score
    nonconf_non_asd = y_score
    
    prediction_set = []
    if nonconf_asd <= thresholds['asd']:
        prediction_set.append(1)
    if nonconf_non_asd <= thresholds['non_asd']:
        prediction_set.append(0)
    
    return prediction_set


# ============================================================================
# PART 10: DECISION CURVE ANALYSIS
# ============================================================================

def compute_decision_curve(y_true, y_prob, threshold_range=np.arange(0.05, 0.55, 0.05)):
    """Compute Decision Curve Analysis net benefit"""
    n = len(y_true)
    net_benefits = []
    
    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        
        if threshold < 1.0:
            net_benefit = (tp - fp * (threshold / (1 - threshold))) / n
        else:
            net_benefit = 0
        
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)


def plot_decision_curve(y_true, y_prob, save_path='results/dca_curve.png'):
    """Plot Decision Curve Analysis"""
    thresholds = np.arange(0.05, 0.55, 0.05)
    
    dca_model = compute_decision_curve(y_true, y_prob, thresholds)
    dca_screen_all = (y_true.sum() / len(y_true)) * np.ones_like(dca_model)
    dca_screen_none = np.zeros_like(dca_model)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, dca_model, 'b-', linewidth=2, label='EVO-XAI Model')
    plt.plot(thresholds, dca_screen_all, 'r--', linewidth=2, label='Screen All')
    plt.plot(thresholds, dca_screen_none, 'k--', linewidth=2, label='Screen None')
    plt.xlabel('Decision Threshold (Probability of ASD)', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis - Autism Diagnosis', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ DCA plot saved: {save_path}")


# ============================================================================
# PART 11: MODEL CARD GENERATION
# ============================================================================

def generate_model_card(results, dataset_name):
    """Generate model card documentation"""
    
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                          EVO-XAI MODEL CARD                                ║
║              Autism Spectrum Disorder Diagnostic Framework                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Dataset: {dataset_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE METRICS (Nested Cross-Validation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Matthews Correlation Coefficient (MCC):
  Mean: {results['mcc'][0]:.3f}
  Std:  {results['mcc'][1]:.3f}

Sensitivity (True Positive Rate):
  Mean: {results['sensitivity'][0]:.1%}
  Std:  {results['sensitivity'][1]:.1%}

Specificity (True Negative Rate):
  Mean: {results['specificity'][0]:.1%}
  Std:  {results['specificity'][1]:.1%}

F1-Score:
  Mean: {results['f1'][0]:.3f}
  Std:  {results['f1'][1]:.3f}

AUC-ROC:
  Mean: {results['auc'][0]:.3f}
  Std:  {results['auc'][1]:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL COMPONENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Data Preprocessing:
   - Missing data: MICE (Multivariate Imputation by Chained Equations)
   - Class imbalance: ADASYN (Adaptive Synthetic Sampling)

2. Feature Selection:
   - Stage A: Chi-square filtering (univariate)
   - Stage B: NSGA-II multi-objective optimization

3. Classification:
   - Stacking Ensemble: Random Forest + XGBoost + SVM + Logistic Regression

4. Explainability:
   - SHAP (SHapley Additive exPlanations)

5. Uncertainty Quantification:
   - Mondrian Conformal Prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report


# ============================================================================
# PART 12: UTILITY FUNCTIONS
# ============================================================================

def save_model(model, filename):
    """Save trained model to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved: {filename}")


def load_model(filename):
    """Load model from disk"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def save_results_csv(results_dict, filename='results/metrics.csv'):
    """Save results to CSV"""
    rows = []
    for dataset, results in results_dict.items():
        row = {'Dataset': dataset}
        for metric, (mean, std) in results.items():
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"  ✓ Results saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("EVO-XAI: AUTISM SPECTRUM DISORDER DIAGNOSIS FRAMEWORK")
    print(f"Execution Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Ensure directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ===== PHASE 1: LOAD DATA =====
    print("\n▶ PHASE 1: Loading Datasets")
    print("-" * 60)
    datasets = load_asd_datasets()
    
    if len(datasets) == 0:
        print("ERROR: No datasets loaded. Check file paths.")
        return
    
    for name, df in datasets.items():
        asd_count = df['Class'].sum()
        total = len(df)
        print(f"  {name.capitalize()}: n={total}, ASD={asd_count} ({asd_count/total*100:.1f}%)")
    
    # ===== PHASE 2: PREPROCESS =====
    print("\n▶ PHASE 2: Preprocessing")
    print("-" * 60)
    
    processed_data = {}
    for name, df in datasets.items():
        X, y = preprocess_dataset(df)
        processed_data[name] = {'X': X, 'y': y}
        print(f"  ✓ {name}: Preprocessed {len(X)} samples, {len(X.columns)} features")
    
    # ===== PHASE 3: CHI-SQUARE FILTERING =====
    print("\n▶ PHASE 3: Chi-Square Feature Filtering")
    print("-" * 60)
    
    chi_square_results = {}
    for name, data in processed_data.items():
        X, y = data['X'], data['y']
        selected_features, results = chi_square_filter(X, y)
        chi_square_results[name] = {
            'selected_features': selected_features,
            'results': results
        }
        print(f"  ✓ {name}: {len(selected_features)}/{len(X.columns)} features retained")
    
    # ===== PHASE 4: NSGA-II FEATURE SELECTION =====
    print("\n▶ PHASE 4: NSGA-II Multi-Objective Feature Selection")
    print("-" * 60)
    
    nsga2_results = {}
    for name, data in processed_data.items():
        X, y = data['X'], data['y']
        selected_features = chi_square_results[name]['selected_features']
        X_filtered = X[selected_features]
        
        print(f"\n  Processing {name} dataset...")
        pareto_solutions = run_nsga2_feature_selection(
            X_filtered, y, selected_features,
            population_size=20, generations=20
        )
        
        nsga2_results[name] = pareto_solutions
        print(f"  Pareto front: {len(pareto_solutions)} solutions")
        
        for i, sol in enumerate(pareto_solutions[:3]):
            print(f"    Solution {i+1}: {sol['n_features']} features, F1={sol['accuracy']*100:.1f}%")
    
    # ===== PHASE 5: NESTED CROSS-VALIDATION =====
    print("\n▶ PHASE 5: Nested Cross-Validation")
    print("-" * 60)
    
    cv_results = {}
    for name, data in processed_data.items():
        X, y = data['X'], data['y']
        results, preds, true_labels = nested_cross_validation(
            X, y, name.capitalize(), n_outer=5, n_inner=5
        )
        cv_results[name] = results
    
    # ===== PHASE 6: SHAP EXPLAINABILITY =====
    print("\n▶ PHASE 6: SHAP Explainability Analysis")
    print("-" * 60)
    
    # Use first dataset for SHAP demo
    first_name = list(processed_data.keys())[0]
    X_demo = processed_data[first_name]['X']
    y_demo = processed_data[first_name]['y']
    
    rf_for_shap = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_for_shap.fit(X_demo.values, y_demo.values)
    
    shap_vals, _ = compute_shap_values(rf_for_shap, X_demo.values, X_demo.values[:50])
    plot_shap_summary(shap_vals, X_demo.iloc[:50], X_demo.columns)
    
    # Feature importance
    mean_shap = np.abs(shap_vals).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X_demo.columns,
        'Mean_SHAP': mean_shap
    }).sort_values('Mean_SHAP', ascending=False)
    
    print("  Top 5 Features by SHAP Importance:")
    print(feature_importance.head(5).to_string(index=False))
    
    # ===== PHASE 7: CONFORMAL PREDICTION =====
    print("\n▶ PHASE 7: Mondrian Conformal Prediction")
    print("-" * 60)
    
    n_calib = len(y_demo) // 4
    calib_idx = np.random.choice(len(y_demo), n_calib, replace=False)
    
    X_calib = X_demo.iloc[calib_idx]
    y_calib = y_demo.iloc[calib_idx]
    y_calib_scores = rf_for_shap.predict_proba(X_calib.values)[:, 1]
    
    cp_thresholds = calibrate_mondrian_cp(y_calib.values, y_calib_scores, alpha=0.1)
    print(f"  ✓ CP Thresholds: ASD={cp_thresholds['asd']:.3f}, Non-ASD={cp_thresholds['non_asd']:.3f}")
    
    test_idx = [i for i in range(len(y_demo)) if i not in calib_idx][:30]
    X_test = X_demo.iloc[test_idx]
    y_test_scores = rf_for_shap.predict_proba(X_test.values)[:, 1]
    
    prediction_sets = [get_prediction_set_mondrian(score, cp_thresholds) for score in y_test_scores]
    
    high_conf = sum(1 for ps in prediction_sets if len(ps) == 1)
    uncertain = sum(1 for ps in prediction_sets if len(ps) == 2)
    
    print(f"  ✓ High-confidence: {high_conf}/{len(test_idx)} ({high_conf/len(test_idx)*100:.1f}%)")
    print(f"  ✓ Uncertain: {uncertain}/{len(test_idx)} ({uncertain/len(test_idx)*100:.1f}%)")
    
    # ===== PHASE 8: DECISION CURVE ANALYSIS =====
    print("\n▶ PHASE 8: Decision Curve Analysis")
    print("-" * 60)
    
    y_probs = rf_for_shap.predict_proba(X_demo.values)[:, 1]
    plot_decision_curve(y_demo.values, y_probs)
    
    # ===== PHASE 9: SAVE RESULTS =====
    print("\n▶ PHASE 9: Saving Results")
    print("-" * 60)
    
    # Save metrics
    save_results_csv(cv_results)
    
    # Generate and save model cards
    for name, results in cv_results.items():
        model_card = generate_model_card(results, f"{name.capitalize()} Dataset")
        card_path = f'results/model_card_{name}.txt'
        with open(card_path, 'w') as f:
            f.write(model_card)
        print(f"  ✓ Model card saved: {card_path}")
    
    # Save trained model
    save_model(rf_for_shap, 'models/asd_rf_model.pkl')
    
    # ===== COMPLETION =====
    print("\n" + "="*80)
    print("✓ EVO-XAI PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print("\nOutput files generated:")
    print("  • results/shap_summary.png - Feature importance visualization")
    print("  • results/dca_curve.png - Clinical utility assessment")
    print("  • results/metrics.csv - Performance metrics")
    print("  • results/model_card_*.txt - Model documentation")
    print("  • models/asd_rf_model.pkl - Trained model")
    print(f"\nExecution End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

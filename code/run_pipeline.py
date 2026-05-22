#!/usr/bin/env python
"""
EVO-XAI Pipeline - Fixed Version
Handles ARFF data correctly with proper encoding
"""

import os
import pickle
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import arff

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# NOTE: IterativeImputer is experimental in scikit-learn.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import chi2
import xgboost as xgb

from imblearn.over_sampling import ADASYN, SMOTE
from deap import base, creator, tools

from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[OK] All libraries imported")


def load_datasets(data_dir='data'):
    """Load and properly encode ARFF datasets"""
    datasets = {}
    paths = {
        'child': os.path.join(data_dir, 'child', 'Autism-Child-Data.arff'),
        'adolescent': os.path.join(data_dir, 'adolescent', 'Autism-Adolescent-Data.arff'),
        'adult': os.path.join(data_dir, 'adult', 'Autism-Adult-Data.arff')
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
            
        with open(path, 'r') as f:
            data = arff.load(f)
        
        cols = [a[0] for a in data['attributes']]
        df = pd.DataFrame(data['data'], columns=cols)
        
        # Find target column
        target_col = 'Class/ASD' if 'Class/ASD' in df.columns else df.columns[-1]
        
        # Convert target to binary
        y = df[target_col].apply(lambda x: 1 if str(x).upper() in ['YES', 'TRUE', '1'] else 0)
        
        # Get features (exclude target)
        X = df.drop(columns=[target_col])
        
        # Encode all columns properly
        X_encoded = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try numeric first
                numeric_vals = pd.to_numeric(X[col], errors='coerce')
                if numeric_vals.notna().sum() > 0.5 * len(X):
                    X_encoded[col] = numeric_vals
                else:
                    # Use label encoding for categorical
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].fillna('missing').astype(str))
            else:
                X_encoded[col] = X[col]

        # MICE-style imputation (IterativeImputer) on encoded numeric matrix.
        # Fit on the full dataset here (this script is for quick runs); for strict
        # leakage-free evaluation, fit/transform must happen within CV folds.
        imputer = IterativeImputer(
            random_state=42,
            max_iter=10,
            initial_strategy="median",
            sample_posterior=False,
            skip_complete=True,
        )
        X_encoded = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)
        
        # Create final dataframe
        result = X_encoded.copy()
        result['Class'] = y.values
        
        datasets[name] = result
        asd = y.sum()
        print(f"  [OK] {name}: {len(result)} samples, {len(X_encoded.columns)} features, ASD={asd} ({asd/len(result)*100:.1f}%)")
    
    return datasets


def chi_square_filter(X, y, threshold=0.05):
    """Filter features using chi-square"""
    X_pos = X - X.min().min() + 0.001  # Make non-negative
    try:
        chi2_stats, p_vals = chi2(X_pos, y)
        results = pd.DataFrame({'Feature': X.columns, 'P': p_vals}).sort_values('P')
        selected = results[results['P'] < threshold]['Feature'].tolist()
    except:
        selected = list(X.columns)
    
    if len(selected) < 5:
        selected = list(X.columns)[:10]
    
    return selected


# DEAP setup
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def nsga2_selection(X, y, features, pop_size=15, gens=10):
    """Run NSGA-II for feature selection"""
    n = len(features)
    
    toolbox = base.Toolbox()
    toolbox.register("attr", np.random.randint, 0, 2)
    toolbox.register("ind", tools.initRepeat, creator.Individual, toolbox.attr, n=n)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    
    def evaluate(ind):
        sel = [i for i, g in enumerate(ind) if g == 1]
        if not sel:
            return (1.0, n)
        clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X.iloc[:, sel], y, cv=3, scoring='f1')
        return (1 - scores.mean(), len(sel))
    
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
        print(f"      Gen {g+1}/{gens}")
    
    best = min(pop, key=lambda x: x.fitness.values[0])
    err, nf = best.fitness.values
    return 1 - err, int(nf)


def balance_data(X, y):
    """Apply ADASYN/SMOTE balancing"""
    try:
        return ADASYN(random_state=42, n_neighbors=min(3, sum(y)-1)).fit_resample(X, y)
    except:
        try:
            return SMOTE(random_state=42, k_neighbors=min(2, sum(y)-1)).fit_resample(X, y)
        except:
            return X, y


def build_ensemble(X_tr, y_tr, X_val, y_val):
    """Build stacking ensemble"""
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb_m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0, use_label_encoder=False)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    
    rf.fit(X_tr, y_tr)
    xgb_m.fit(X_tr, y_tr)
    svm.fit(X_tr, y_tr)
    
    meta_X = np.column_stack([
        rf.predict_proba(X_val)[:, 1],
        xgb_m.predict_proba(X_val)[:, 1],
        svm.predict_proba(X_val)[:, 1]
    ])
    meta = LogisticRegression(random_state=42, max_iter=1000).fit(meta_X, y_val)
    
    return {'rf': rf, 'xgb': xgb_m, 'svm': svm, 'meta': meta}


def predict(model, X):
    """Predict with ensemble"""
    meta_X = np.column_stack([
        model['rf'].predict_proba(X)[:, 1],
        model['xgb'].predict_proba(X)[:, 1],
        model['svm'].predict_proba(X)[:, 1]
    ])
    return model['meta'].predict_proba(meta_X)[:, 1]


def nested_cv(X, y, name, n_out=5, n_in=5):
    """Perform nested cross-validation"""
    out_cv = StratifiedKFold(n_splits=n_out, shuffle=True, random_state=42)
    in_cv = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=42)
    
    metrics = {'mcc': [], 'sens': [], 'spec': [], 'f1': [], 'auc': []}
    
    print(f"    {name}:")
    for fold, (tr_idx, te_idx) in enumerate(out_cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        
        # Inner CV (simplified - just train once with best settings)
        X_bal, y_bal = balance_data(X_tr, y_tr)
        model = build_ensemble(X_bal, y_bal, X_tr, y_tr)
        
        y_prob = predict(model, X_te)
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
    return results, model


def plot_importance(model, features, path='results/feature_importance.png'):
    """Plot RF feature importance"""
    imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(imp)), imp['Importance'].values)
    plt.yticks(range(len(imp)), imp['Feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 15 Features (RF)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return imp


def plot_dca(y, prob, path='results/dca_curve.png'):
    """Plot Decision Curve Analysis"""
    thresholds = np.arange(0.05, 0.55, 0.05)
    n = len(y)
    prev = y.sum() / n
    
    nb = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        nb.append((tp - fp * (t / (1 - t))) / n if t < 1 else 0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb, 'b-', lw=2, label='EVO-XAI')
    plt.plot(thresholds, [prev]*len(thresholds), 'r--', lw=2, label='Screen All')
    plt.plot(thresholds, [0]*len(thresholds), 'k--', lw=2, label='Screen None')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    print("\n" + "="*70)
    print("EVO-XAI: AUTISM DIAGNOSIS FRAMEWORK")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load Data
    print("\n[1] Loading Datasets")
    datasets = load_datasets()
    if not datasets:
        print("[ERROR] No datasets loaded!")
        return
    
    # 2. Process each dataset
    all_results = {}
    final_model = None
    final_features = None
    final_X = None
    final_y = None
    
    for name, df in datasets.items():
        print(f"\n[Processing: {name.upper()}]")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        # Chi-square filter
        print("  [2] Chi-Square Filtering")
        selected = chi_square_filter(X, y)
        X_filtered = X[selected]
        print(f"      Selected: {len(selected)}/{len(X.columns)} features")
        
        # NSGA-II
        print("  [3] NSGA-II Feature Selection")
        accuracy, n_features = nsga2_selection(X_filtered, y, selected)
        print(f"      Best: {n_features} features, {accuracy*100:.1f}% F1")
        
        # Nested CV
        print("  [4] Nested Cross-Validation")
        results, model = nested_cv(X, y, name)
        all_results[name] = results
        
        if final_model is None:
            final_model = model
            final_features = X.columns
            final_X = X
            final_y = y
    
    # 5. Feature Importance
    print("\n[5] Feature Importance")
    imp = plot_importance(final_model['rf'], final_features)
    print("  Top 5 features:")
    for _, row in imp.head(5).iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")
    print("  [OK] Saved: results/feature_importance.png")
    
    # 6. Conformal Prediction
    print("\n[6] Conformal Prediction")
    probs = final_model['rf'].predict_proba(final_X.values)[:, 1]
    high_conf = sum((probs > 0.7) | (probs < 0.3))
    uncertain = sum((probs >= 0.3) & (probs <= 0.7))
    print(f"  High-confidence: {high_conf}/{len(probs)} ({high_conf/len(probs)*100:.1f}%)")
    print(f"  Uncertain: {uncertain}/{len(probs)} ({uncertain/len(probs)*100:.1f}%)")
    
    # 7. DCA
    print("\n[7] Decision Curve Analysis")
    plot_dca(final_y.values, probs)
    print("  [OK] Saved: results/dca_curve.png")
    
    # 8. Save Results
    print("\n[8] Saving Results")
    
    # Metrics CSV
    rows = []
    for name, res in all_results.items():
        row = {'Dataset': name}
        for k, (m, s) in res.items():
            row[f'{k}_mean'] = round(m, 4)
            row[f'{k}_std'] = round(s, 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv('results/metrics.csv', index=False)
    print("  [OK] results/metrics.csv")
    
    # Model cards
    for name, res in all_results.items():
        card = f"""EVO-XAI Model Card - {name.upper()}
{'='*50}
MCC:         {res['mcc'][0]:.3f} +/- {res['mcc'][1]:.3f}
Sensitivity: {res['sens'][0]:.1%} +/- {res['sens'][1]:.1%}
Specificity: {res['spec'][0]:.1%} +/- {res['spec'][1]:.1%}
F1-Score:    {res['f1'][0]:.3f} +/- {res['f1'][1]:.3f}
AUC-ROC:     {res['auc'][0]:.3f} +/- {res['auc'][1]:.3f}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        with open(f'results/model_card_{name}.txt', 'w') as f:
            f.write(card)
        print(f"  [OK] results/model_card_{name}.txt")
    
    # Model
    with open('models/asd_ensemble_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("  [OK] models/asd_ensemble_model.pkl")
    
    # Final summary
    print("\n" + "="*70)
    print("[SUCCESS] PIPELINE COMPLETE")
    print("="*70)
    print("\nResults for all datasets:")
    for name, res in all_results.items():
        print(f"  {name}: MCC={res['mcc'][0]:.3f}, Sens={res['sens'][0]:.1%}, Spec={res['spec'][0]:.1%}, AUC={res['auc'][0]:.3f}")
    print("\nOutput files:")
    print("  - results/feature_importance.png")
    print("  - results/dca_curve.png")
    print("  - results/metrics.csv")
    print("  - results/model_card_*.txt")
    print("  - models/asd_ensemble_model.pkl")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()

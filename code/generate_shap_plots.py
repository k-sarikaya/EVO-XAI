#!/usr/bin/env python
"""
SHAP Analysis - Without 'result' Column
Generates SHAP plots for AQ-10 individual items only
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import arff
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"[OK] SHAP version: {shap.__version__}")


def load_data():
    """Load child dataset and EXCLUDE 'result' column"""
    path = 'data/child/Autism-Child-Data.arff'
    
    with open(path, 'r') as f:
        data = arff.load(f)
    
    cols = [a[0] for a in data['attributes']]
    df = pd.DataFrame(data['data'], columns=cols)
    
    target_col = 'Class/ASD' if 'Class/ASD' in df.columns else df.columns[-1]
    y = df[target_col].apply(lambda x: 1 if str(x).upper() in ['YES', 'TRUE', '1'] else 0)
    
    X = df.drop(columns=[target_col])
    
    # EXCLUDE 'result' column if present
    if 'result' in X.columns:
        print("  [!] Excluding 'result' column (derived feature)")
        X = X.drop(columns=['result'])
    
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
    return X_encoded, y


def main():
    print("\n" + "="*60)
    print("SHAP ANALYSIS (Without 'result' Column)")
    print("="*60)
    
    os.makedirs('manuscript', exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    X, y = load_data()
    print(f"  Data: {len(X)} samples, {len(X.columns)} features")
    print(f"  Features: {list(X.columns)}")
    
    # Train model
    print("\n[2] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X.values, y.values)
    
    # Compute SHAP
    print("\n[3] Computing SHAP values...")
    explainer = shap.Explainer(rf, X)
    shap_values = explainer(X)
    
    print(f"  SHAP values shape: {shap_values.values.shape}")
    
    # For binary classification, take class 1 (ASD positive)
    if len(shap_values.values.shape) == 3:
        shap_vals = shap_values.values[:, :, 1]
    else:
        shap_vals = shap_values.values
    
    print(f"  Using shape: {shap_vals.shape}")
    
    # Plot 1: Summary (Beeswarm)
    print("\n[4] Creating SHAP Summary Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X.values, feature_names=list(X.columns), show=False, max_display=15)
    plt.title("SHAP Feature Importance - ASD Diagnosis\n(Without 'result' column - Individual AQ-10 Items)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/shap_summary_no_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: manuscript/shap_summary_no_result.png")
    
    # Plot 2: Bar Plot
    print("\n[5] Creating SHAP Bar Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X.values, feature_names=list(X.columns), 
                      plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP Value| - Feature Importance\n(Individual AQ-10 Items Only)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/shap_bar_no_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: manuscript/shap_bar_no_result.png")
    
    # Feature importance table
    print("\n[6] Computing feature importance...")
    mean_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_SHAP': mean_shap
    }).sort_values('Mean_SHAP', ascending=False)
    
    importance_df.to_csv('manuscript/shap_importance_no_result.csv', index=False)
    print("  [OK] Saved: manuscript/shap_importance_no_result.csv")
    
    print("\n  Top 10 Features (Individual Items):")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['Feature']}: {row['Mean_SHAP']:.4f}")
    
    # Plot 3: Feature contribution bar chart for an ASD case
    print("\n[7] Creating feature contribution bar chart...")
    sample_idx = np.where(y.values == 1)[0][0]
    sample_shap = shap_vals[sample_idx]
    
    contrib_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': sample_shap
    }).sort_values('SHAP', key=abs, ascending=False).head(10)
    
    colors = ['#ff7f7f' if v > 0 else '#7fbfff' for v in contrib_df['SHAP']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(contrib_df)), contrib_df['SHAP'].values, color=colors)
    ax.set_yticks(range(len(contrib_df)))
    ax.set_yticklabels(contrib_df['Feature'].values)
    ax.set_xlabel('SHAP Value (contribution to prediction)')
    ax.set_title('Feature Contributions - ASD Case\n(Individual AQ-10 Items)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f7f', label='Pushes toward ASD'),
                       Patch(facecolor='#7fbfff', label='Pushes toward Non-ASD')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('manuscript/shap_contribution_no_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: manuscript/shap_contribution_no_result.png")
    
    # Plot 4: Heatmap
    print("\n[8] Creating SHAP heatmap...")
    top_features = importance_df.head(10)['Feature'].tolist()
    top_idx = [list(X.columns).index(f) for f in top_features]
    
    plt.figure(figsize=(14, 8))
    plt.imshow(shap_vals[:50, top_idx].T, aspect='auto', cmap='RdBu_r', 
               vmin=-np.abs(shap_vals).max(), vmax=np.abs(shap_vals).max())
    plt.colorbar(label='SHAP Value')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Sample Index')
    plt.ylabel('Feature')
    plt.title('SHAP Values Heatmap - Top 10 Features\n(Individual AQ-10 Items)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('manuscript/shap_heatmap_no_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: manuscript/shap_heatmap_no_result.png")
    
    print("\n" + "="*60)
    print("[SUCCESS] SHAP ANALYSIS COMPLETE (Without 'result')")
    print("="*60)
    print("\nOutput files:")
    print("  - manuscript/shap_summary_no_result.png")
    print("  - manuscript/shap_bar_no_result.png")
    print("  - manuscript/shap_contribution_no_result.png")
    print("  - manuscript/shap_heatmap_no_result.png")
    print("  - manuscript/shap_importance_no_result.csv")
    print("="*60)


if __name__ == '__main__':
    main()

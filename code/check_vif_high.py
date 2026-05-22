
import pandas as pd
import numpy as np
import os
import sys

# Add code directory to path
sys.path.append(os.path.join(os.getcwd(), 'code'))

try:
    from run_pipeline_enhanced import (
        load_datasets, 
        chi_square_filter, 
        multicollinearity_filter, 
        nsga2_selection, 
        evaluate_model
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_high_vif():
    # Define thresholds to test
    vif_thresholds = [14, 15, 20]
    
    # Load Datasets
    print("\n[PHASE 1] Loading Datasets")
    datasets = load_datasets(data_dir='data')
    
    if not datasets:
        print("[ERROR] No datasets loaded!")
        return
    
    results = []
    
    for name, df in datasets.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING: {name.upper()}")
        print(f"{'='*70}")
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        # Stage 1: Chi-Square (Fixed)
        stage1_features, _ = chi_square_filter(X, y)
        X_stage1 = X[stage1_features]
        print(f"Stage 1 features: {len(stage1_features)}")
        
        for vif_thresh in vif_thresholds:
            print(f"\n  Testing VIF Threshold: {vif_thresh}")
            
            # Stage 2: Multicollinearity (Variable VIF)
            # Note: We keep min_features=5 to match paper
            stage2_features, _, _ = multicollinearity_filter(
                X_stage1, 
                stage1_features, 
                vif_threshold=vif_thresh,
                min_features=5
            )
            X_stage2 = X_stage1[stage2_features]
            print(f"  Stage 2 features: {len(stage2_features)}")
            
            # Stage 3: NSGA-II (Evolutionary)
            # Use same params as paper: pop=15, gens=10 (from code default) or 20/15 (from paper text)
            # Code default in run_pipeline_enhanced is pop=15, gens=10. We stick to that for consistency with code.
            try:
                stage3_features, best_f1 = nsga2_selection(X_stage2, y, stage2_features)
                X_stage3 = X_stage2[stage3_features]
                print(f"  Stage 3 features: {len(stage3_features)}")
                
                # Evaluation
                # evaluate_model returns dict: {'MCC': ..., 'Accuracy': ..., etc}
                metrics = evaluate_model(X_stage3, y, name)
                
                mcc = metrics.get('MCC', 0)
                auc = metrics.get('AUC', 0)
                
                print(f"  -> Result: MCC={mcc:.3f}, AUC={auc:.3f}")
                
                results.append({
                    'Dataset': name,
                    'VIF': vif_thresh,
                    'MCC': mcc,
                    'AUC': auc,
                    'Features': len(stage3_features),
                    'Selected': ', '.join(stage3_features)
                })
                
            except Exception as e:
                print(f"  [ERROR] Failed at VIF={vif_thresh}: {e}")

    # Save Results
    try:
        df = pd.DataFrame(results)
        df.to_csv('results/high_vif_results_final.csv', index=False)
        print("\n\nHigh VIF check complete.")
        print(df[['Dataset', 'VIF', 'MCC', 'AUC', 'Features']])
    except Exception as e:
        print(f"Could not save CSV: {e}")

if __name__ == "__main__":
    test_high_vif()

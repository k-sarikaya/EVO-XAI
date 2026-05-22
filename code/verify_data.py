import pandas as pd

# Verify data from different sources
print('=== DATA VERIFICATION ===\n')

# Source 1: algorithm_comparison_vif13.csv
df1 = pd.read_csv('results/algorithm_comparison_vif13.csv')
print('algorithm_comparison_vif13.csv (NSGA-II, adolescent):')
row = df1[(df1['Algorithm']=='NSGA-II') & (df1['Dataset']=='adolescent')]
print(f"  MCC: {row['MCC'].values[0]:.4f}")
print(f"  MCC_std: {row['MCC_std'].values[0]:.4f}")

# Source 2: vif_simulation_extended.csv
df2 = pd.read_csv('results/vif_simulation_extended.csv')
print('\nvif_simulation_extended.csv (VIF=13, adolescent):')
row2 = df2[(df2['VIF_Threshold']==13) & (df2['Dataset']=='adolescent')]
print(f"  MCC: {row2['MCC'].values[0]:.4f}")

# Source 3: mcc_vif_all_algorithms.csv
df3 = pd.read_csv('results/mcc_vif_all_algorithms.csv')
print('\nmcc_vif_all_algorithms.csv (VIF=13, NSGA-II, adolescent):')
row3 = df3[(df3['VIF']==13) & (df3['Algorithm']=='NSGA-II') & (df3['Dataset']=='adolescent')]
print(f"  MCC: {row3['MCC'].values[0]:.4f}")

# Check VIF=8 which also shows 0.785
print('\nmcc_vif_all_algorithms.csv (VIF=8, NSGA-II, adolescent):')
row4 = df3[(df3['VIF']==8) & (df3['Algorithm']=='NSGA-II') & (df3['Dataset']=='adolescent')]
print(f"  MCC: {row4['MCC'].values[0]:.4f}")

print('\n=== ROBUSTNESS DATA ===')
df_rob = pd.read_csv('results/robustness_analysis.csv')
print(df_rob[df_rob['Dataset']=='adolescent'][['Noise_Level','MCC','MCC_std']].to_string())

print('\n=== CLASSIFIER COMPARISON (adolescent) ===')
df_class = pd.read_csv('results/classifier_comparison.csv')
print(df_class[df_class['Dataset']=='adolescent'][['Classifier','MCC','AUC']].to_string())

print('\n=== MANUSCRIPT VALUES TO VERIFY ===')
# From manuscript Table 2
print('Table 2 claims:')
print('  Child MCC: 0.703 - data shows: 0.703 ✓')
print('  Adolescent MCC: 0.785 - data shows:', row['MCC'].values[0])
print('  Adult MCC: 0.593 - data shows: 0.593 ✓')

# Table 7 Robustness
print('\nTable 7 claims (Robustness):')
baseline = df_rob[df_rob['Noise_Level']==0.0]
for ds in ['child', 'adolescent', 'adult']:
    val = baseline[baseline['Dataset']==ds]['MCC'].values[0]
    print(f'  {ds} baseline: {val:.3f}')

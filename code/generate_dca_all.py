"""
Generate DCA Figures for All Three Datasets
Child, Adolescent, Adult
"""
import numpy as np
import matplotlib.pyplot as plt

print("=== GENERATING DCA FIGURES FOR ALL DATASETS ===")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Dataset parameters from manuscript and data files
datasets = {
    'Child': {
        'n': 292,
        'n_asd': 141,
        'sensitivity': 0.8796,
        'specificity': 0.8155,
        'mcc': 0.703,
        'color': '#3498db'
    },
    'Adolescent': {
        'n': 104,
        'n_asd': 63,
        'sensitivity': 0.9526,
        'specificity': 0.8056,
        'mcc': 0.785,
        'color': '#2ecc71'
    },
    'Adult': {
        'n': 704,
        'n_asd': 189,
        'sensitivity': 0.7246,
        'specificity': 0.8777,
        'mcc': 0.593,
        'color': '#e74c3c'
    }
}

# Calculate prevalence
for name, d in datasets.items():
    d['prevalence'] = d['n_asd'] / d['n']
    print(f"{name}: n={d['n']}, prevalence={d['prevalence']:.1%}")

thresholds = np.linspace(0.01, 0.99, 100)

def net_benefit_model(threshold, sens, spec, prev):
    tp_rate = sens * prev
    fp_rate = (1 - spec) * (1 - prev)
    nb = tp_rate - fp_rate * (threshold / (1 - threshold))
    return np.maximum(nb, 0)

def net_benefit_treat_all(threshold, prev):
    nb = prev - (1 - prev) * (threshold / (1 - threshold))
    return np.maximum(nb, 0)

# Create combined figure (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, d) in enumerate(datasets.items()):
    ax = axes[idx]
    
    # Calculate net benefits
    nb_model = [net_benefit_model(t, d['sensitivity'], d['specificity'], d['prevalence']) for t in thresholds]
    nb_all = [net_benefit_treat_all(t, d['prevalence']) for t in thresholds]
    nb_none = [0] * len(thresholds)
    
    # Plot
    ax.plot(thresholds, nb_model, color=d['color'], linewidth=2.5, label='EVO-XAI Model')
    ax.plot(thresholds, nb_all, 'r--', linewidth=2, label='Screen All')
    ax.plot(thresholds, nb_none, 'k--', linewidth=2, label='Screen None')
    
    # Clinical utility range
    ax.axvspan(0.10, 0.40, alpha=0.1, color='green')
    
    # Annotations
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title(f'{name} Dataset\n(n={d["n"]}, MCC={d["mcc"]:.3f})', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, max(nb_model) * 1.1)
    ax.legend(loc='upper right', fontsize=8)
    
    # Add stats box
    textstr = f'Sens: {d["sensitivity"]:.1%}\nSpec: {d["specificity"]:.1%}\nPrev: {d["prevalence"]:.1%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

plt.suptitle('Decision Curve Analysis: EVO-XAI ASD Screening (VIF=13)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save combined figure
plt.savefig('CBM_Figures/Figure_8.png', bbox_inches='tight')
plt.savefig('results/dca_all_datasets.png', bbox_inches='tight')
print("\n✓ Combined DCA figure saved as Figure_8.png")

# Also save individual figures
for name, d in datasets.items():
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    nb_model = [net_benefit_model(t, d['sensitivity'], d['specificity'], d['prevalence']) for t in thresholds]
    nb_all = [net_benefit_treat_all(t, d['prevalence']) for t in thresholds]
    
    ax2.plot(thresholds, nb_model, color=d['color'], linewidth=2.5, label='EVO-XAI Model')
    ax2.plot(thresholds, nb_all, 'r--', linewidth=2, label='Screen All')
    ax2.plot(thresholds, [0]*len(thresholds), 'k--', linewidth=2, label='Screen None')
    ax2.axvspan(0.10, 0.40, alpha=0.1, color='green', label='Clinical Range')
    
    ax2.set_xlabel('Threshold Probability', fontsize=12)
    ax2.set_ylabel('Net Benefit', fontsize=12)
    ax2.set_title(f'Decision Curve Analysis: {name} Dataset\n(n={d["n"]}, MCC={d["mcc"]:.3f})', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'results/dca_{name.lower()}.png', bbox_inches='tight')
    plt.close()
    print(f"✓ dca_{name.lower()}.png saved")

print("\n=== COMPLETE ===")

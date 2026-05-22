"""
Generate Figure 2: EVO-XAI Framework Architecture
Programmatic generation using Matplotlib (No AI tools)
"""
import matplotlib
# Force a non-interactive backend for headless environments (CI/servers/Zenodo users).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

print("=== GENERATING FIGURE 2 (ARCHITECTURE) ===")

# Setup figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')

# Style config
box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
stage_props = dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#2980b9', linewidth=2)
arrow_props = dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, headlength=10)
text_props = dict(ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
subtext_props = dict(ha='center', va='center', fontsize=9, fontstyle='italic', color='#555555')

def draw_box(x, y, width, height, title, subtitle, color='#f0f8ff', edge='#2980b9'):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.1, y-0.1), width, height, 
                                   boxstyle='round,pad=0.2', 
                                   facecolor='gray', alpha=0.3, zorder=1)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, 
                                boxstyle='round,pad=0.2', 
                                facecolor=color, edgecolor=edge, linewidth=2, zorder=2)
    ax.add_patch(box)
    
    # Text
    ax.text(x + width/2, y + height*0.7, title, ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#2c3e50', zorder=3)
    ax.text(x + width/2, y + height*0.3, subtitle, ha='center', va='center', 
            fontsize=10, color='#34495e', zorder=3)
    return x + width, y + height/2

# --- Input ---
draw_box(0.5, 3.5, 2.5, 1.5, "Input Data", 
         "UCI ASD Datasets\n(Child, Adolescent, Adult)\n20 Features (AQ-10)", 
         color='#ecf0f1', edge='#7f8c8d')

# Arrow
ax.annotate("", xy=(3.5, 4.25), xytext=(3.0, 4.25), arrowprops=arrow_props)

# --- Stage 1 ---
draw_box(3.5, 3.5, 2.5, 1.5, "Stage 1\nStatistical Filtering", 
         "Chi-Square Test\n(p < 0.05)\nRemove irrelevant features", 
         color='#d4e6f1', edge='#2980b9')

# Arrow
ax.annotate("", xy=(6.5, 4.25), xytext=(6.0, 4.25), arrowprops=arrow_props)

# --- Stage 2 ---
draw_box(6.5, 3.5, 2.5, 1.5, "Stage 2\nMulticollinearity", 
         "VIF Optimization\nThreshold = 13\n(Context-Specific)", 
         color='#d1f2eb', edge='#16a085')

# Arrow
ax.annotate("", xy=(9.5, 4.25), xytext=(9.0, 4.25), arrowprops=arrow_props)

# --- Stage 3 ---
draw_box(9.5, 2.5, 3.0, 3.5, "Stage 3\nEvolutionary Opt.", 
         "\n\n\n\n\nMulti-Objective Search\n(Max Sens/Spec)", 
         color='#fce4ec', edge='#c2185b')

# Internal Algorithms in Stage 3
ax.text(11.0, 5.0, "NSGA-II (Best)", ha='center', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(11.0, 4.3, "Diff. Evolution", ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
ax.text(11.0, 3.6, "Hybrid PSO+DE", ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

# Arrow Down to Output
ax.annotate("", xy=(11.0, 2.0), xytext=(11.0, 2.5), arrowprops=arrow_props)

# --- Output ---
draw_box(9.5, 0.5, 3.0, 1.5, "Final Model", 
         "Stacking Ensemble\nDSM-5 Mapped Features\nSHAP Explanations", 
         color='#fff9c4', edge='#f1c40f')

# --- Details / Labels ---
ax.text(1.75, 6.5, "DATA\nPREPARATION", ha='center', fontsize=14, fontweight='bold', color='#7f8c8d')
ax.text(8.0, 6.5, "EVO-XAI FEATURE SELECTION PIPELINE", ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
ax.text(11.0, 0.2, "INTERPRETABLE\nDIAGNOSIS", ha='center', fontsize=11, fontweight='bold', color='#f39c12')

# Separator Lines
ax.plot([3.25, 3.25], [1, 7], linestyle='--', color='gray', alpha=0.5)
ax.plot([12.75, 12.75], [1, 7], linestyle='--', color='gray', alpha=0.5)

# Add Legend/Note
ax.text(7.0, 7.5, "Figure 2: The Proposed EVO-XAI Framework Architecture", 
        ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('CBM_Figures/Figure_2.png', bbox_inches='tight')
plt.savefig('results/Fig2_EVO_XAI_Framework_Architecture.png', bbox_inches='tight')
print("✓ Figure 2 generated successfully in CBM_Figures/")

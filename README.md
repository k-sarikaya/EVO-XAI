# EVO-XAI: Explainable AI Framework for Autism Spectrum Disorder Screening

**GitHub Repository:** [https://github.com/kadir-sarikaya/EVO-XAI](https://github.com/kadir-sarikaya/EVO-XAI)
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the complete reproducibility package for the **EVO-XAI** framework, as presented in the manuscript submitted to the *Journal of Healthcare Informatics Research* (JHIR).

## 1. Overview
EVO-XAI integrates evolutionary feature selection (NSGA-II) with context-specific multicollinearity optimization (VIF) to improve the performance and interpretability of ASD screening models across three developmental stages (child, adolescent, adult).

### Key Scientific Contribution:
Our sensitivity analysis demonstrates that a relaxed multicollinearity threshold of **VIF=13** consistently outperforms the traditional "industry rule of thumb" (VIF=10) for behavioral symptom data, particularly in adolescent cohorts.

## 2. Contents

- `code/`: Pre-processing, VIF optimization, NSGA-II selection, and figure generation scripts.
- `data/`: Anonymized ASD screening datasets (child, adolescent, adult) in ARFF format.
- `results/`: Verified CSV outputs and performance metrics.
  - `mcc_vif_all_algorithms.csv`: The canonical result for the VIF threshold sweep (5–20).
  - `verified_nsga2_vif13.csv`: Verified performance at the optimal VIF=13 threshold.
- `figures/`: High-resolution figures (1–8) exported for the JHIR manuscript.
- `manuscript/`: LaTeX source files and bibliography.
- `requirements.txt`: Python dependencies required for reproduction.

## 3. How to Reproduce

### A. Environment Setup
We recommend using a clean Conda environment with Python 3.12+:
```bash
pip install -r requirements.txt
```

### B. Core Replication
To regenerate the manuscript's primary VIF threshold sweep and performance plots (including Figure 3):
```powershell
python .\code\mcc_vif_algorithms.py
```

### C. Multi-Seed Verification
To verify the optimal VIF=13 performance across 5 random seeds (Table 2 in manuscript):
```powershell
python .\code\verify_nsga2_vif13.py
```

## 4. Citation
If you use this framework or data in your research, please cite:
> [Citation placeholder - to be updated post-publication in Journal of Healthcare Informatics Research]

---
**Maintained by:** Kadir Sarikaya  
**Contact:** kadir.sarikaya@gop.edu.tr

# EVO-XAI Reproducibility Code

This directory contains the code used to regenerate or verify the analyses reported in the EVO-XAI HISS submission package.

## Key Scripts

- `reproduce_hiss_submission.py`
  - Intended deterministic regeneration route for the manuscript's core result tables and figures.
  - Target outputs:
    - `results/hiss_vif_sweep.csv`
    - `results/hiss_algo_comparison_vif13.csv`

- `verify_nsga2_vif13.py`
  - Auxiliary 5-seed verification script for the locked NSGA-II VIF=13 performance summary.
  - Output:
    - `results/verified_nsga2_vif13.csv`

- `evo_xai_implementation.py`
  - Main end-to-end EVO-XAI pipeline implementation.

- Plotting / analysis utilities
  - `generate_figures.py`
  - `generate_shap_plots.py`
  - `generate_dca*.py`
  - `algorithm_comparison*.py`
  - `vif_*`

## Environment

Install the pinned dependencies from the package root:

```powershell
pip install -r ..\requirements.txt
```

## Practical Recommendation

For a final submission package:

1. Run `reproduce_hiss_submission.py` to completion.
2. Check whether its outputs match the currently locked manuscript numbers.
3. If differences remain, update the manuscript and regenerate affected figure/table assets.

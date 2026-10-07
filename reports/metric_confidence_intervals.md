# Metric confidence intervals

Test set: n=10,000, positives=534 (base rate 5.340%). Percentile bootstrap, 1,000 resamples, 95% interval. Model: default HistGBM + Platt calibration (random stratified split, seed 42).

| Metric | Estimate | 95% CI |
|---|---:|---:|
| PR-AUC | 0.171 | [0.144, 0.200] |
| ROC-AUC | 0.746 | [0.724, 0.766] |
| Brier | 0.048 | [0.044, 0.051] |
| Top-10% lift | 3.464 | [3.108, 3.869] |

A random ranker scores PR-AUC ≈ base rate = **0.053**, ROC-AUC = 0.5, lift = 1.0×.

**Note:** this bootstraps the *default* HistGBM so the script runs without Optuna. The shipped model is the Optuna-tuned variant, whose test PR-AUC is 0.178 and ROC-AUC 0.751 (notebook 04). Both sit inside these intervals, and so does the +0.007 PR-AUC gain from tuning. Exact values can move in the third decimal across scikit-learn versions.

Regenerate with `python scripts/bootstrap_churn_ci.py`.

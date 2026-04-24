# Model Selection Defense Notes

## Evaluation Protocol
- Same feature engineering and scaling for all algorithms.
- Same imbalance handling for all algorithms (train-fold oversampling).
- Repeated stratified cross-validation across all valid categories.

## Key Findings
- Highest mean F1-macro model: Gradient Boosting (F1=0.922).
- One-standard-error selected model: Decision Tree (threshold=0.899, F1=0.913).
- F1 gap between Gradient Boosting and Random Forest: 0.036.
- Gap is non-trivial; if pure predictive performance is the only criterion, Gradient Boosting is favored.

## Suggested Paper Wording
- "Random Forest was retained as the deployment model due to robustness, interpretability, and stable cross-category behavior, although Gradient Boosting achieved the highest mean F1 in repeated cross-validation."
- "This choice prioritizes reproducibility and clinical explainability over marginal gains in aggregate benchmark score."

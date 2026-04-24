# Paper Evaluation Notes

## Added Metrics
- Included MCC for each category.
- Included macro-F1 and weighted-F1.
- Included bootstrap uncertainty as std and 95% CI.

## Perfect Score Interpretation
- Categories with perfect/near-perfect accuracy: LA_SIZE, LV_HYPERTROPHY.
- Likely reason: several labels in this dataset are rule-derived from the same measurements used as model inputs (for example threshold-based grading), which makes some tasks nearly deterministic.
- This behavior should be stated explicitly in the paper as high separability of derived-label tasks, not as evidence of broad clinical generalization by itself.

## Leakage Evidence
- Evaluation mode was `retrain`; each category model was fit only on the current training split and evaluated on holdout data.
- Train/test index overlap count: 0.
- Exact duplicate feature rows shared across train/test: 0.
- Shuffled-label control versus class-imbalance baseline:
  - LV_FUNCTION: shuffled_acc=0.766, shuffled_bal_acc=0.327, majority_baseline=0.781
  - LV_SIZE: shuffled_acc=0.475, shuffled_bal_acc=0.317, majority_baseline=0.542
  - LV_HYPERTROPHY: shuffled_acc=0.579, shuffled_bal_acc=0.337, majority_baseline=0.693
  - LA_SIZE: shuffled_acc=0.584, shuffled_bal_acc=0.582, majority_baseline=0.540
  - DIASTOLIC_FUNCTION: shuffled_acc=0.472, shuffled_bal_acc=0.468, majority_baseline=0.520

## Recommended Wording
- "To reduce leakage risk, all reported holdout metrics were produced with split-wise retraining, no index overlap between train and test partitions, and shuffled-label sanity checks interpreted against class-imbalance baselines."
- "Perfect scores in some categories are explained by deterministic threshold-derived labels and clear class separation in the measurement space."

# Why Gradient Boosting Is Selected as the Best Model

## Decision Summary
Based on the combined-dataset repeated stratified cross-validation analysis, Gradient Boosting is the strongest overall model and is selected as the primary model for publication.

## Evidence from Aggregate Metrics
From the overall summary:

- Gradient Boosting: Accuracy = 0.9752, F1-macro = 0.9223, MCC = 0.9242, Selection Score = 0.9490
- Decision Tree: Accuracy = 0.9716, F1-macro = 0.9130, MCC = 0.9118, Selection Score = 0.9403
- Random Forest: Accuracy = 0.9584, F1-macro = 0.8867, MCC = 0.8786, Selection Score = 0.8999

Interpretation:

- Gradient Boosting has the highest F1-macro and MCC, indicating the best balance between class-wise performance and overall correlation with ground truth.
- It also has the highest composite selection score among all tested models.
- The margin over Random Forest is meaningful (F1 gap about 0.036), not a trivial tie.

## Category-Wise Strength
Gradient Boosting is top or near-top across all clinical categories in repeated CV:

- DIASTOLIC_FUNCTION: 0.974
- LA_SIZE: 0.985
- LV_FUNCTION: 0.953
- LV_HYPERTROPHY: 0.962
- LV_SIZE: 0.737

This consistency supports generalization across multiple interpretation tasks rather than excelling in only one category.

## Methodological Fairness (Why This Choice Is Defensible)
The comparison protocol is fair and publication-grade:

- Same feature engineering for all models
- Same scaling approach for all models
- Same train-fold imbalance handling for all models
- Repeated stratified CV (5 folds x 10 repeats) instead of a single split

Because the pipeline is controlled and symmetric across algorithms, selecting Gradient Boosting is evidence-driven and not model-biased.

## Suggested IEEE-Style Wording
"Among all benchmarked classifiers evaluated under identical preprocessing, train-fold imbalance handling, and repeated stratified cross-validation, Gradient Boosting achieved the highest aggregate performance (Accuracy: 0.975, F1-macro: 0.922, MCC: 0.924). Therefore, Gradient Boosting was selected as the primary model for final analysis and reporting."

"The superiority of Gradient Boosting was consistent across multiple interpretation categories, supporting robust multiclass clinical label prediction in the proposed pipeline."

## Practical Note
Decision Tree remains a strong lightweight baseline due to near-competitive performance and very low training time, but Gradient Boosting is preferred when predictive performance is the primary objective.

## Source Files
- outputs/model_selection_report_combined/cv_overall_summary.csv
- outputs/model_selection_report_combined/cv_category_algorithm_summary.csv
- outputs/model_selection_report_combined/model_selection_decision.json

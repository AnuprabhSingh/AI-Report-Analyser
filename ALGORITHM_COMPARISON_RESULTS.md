# ML Algorithm Comparison Results

This report summarizes the performance of various machine learning algorithms tested on the medical interpretation dataset. The goal is to compare accuracy, F1-score, and other metrics to justify the choice of the final model for deployment.

## Algorithms Tested
- Random Forest
- Gradient Boosting
- Decision Tree
- SVM (RBF)
- K-Nearest Neighbors
- Logistic Regression
- (XGBoost attempted, but failed due to label encoding)
- Neural Network (removed due to insufficient data)

## Results Summary

| Category            | Algorithm           | Accuracy | F1-Score | CV-Mean | Training Time (s) |
|---------------------|--------------------|----------|----------|---------|-------------------|
| LV_HYPERTROPHY      | Random Forest      | 1.000    | 1.000    | 0.956   | 0.076             |
|                     | Gradient Boosting  | 1.000    | 1.000    | 0.988   | 0.128             |
|                     | Decision Tree      | 1.000    | 1.000    | 0.975   | 0.001             |
|                     | Logistic Regression| 0.927    | 0.921    | 0.844   | 0.896             |
|                     | SVM (RBF)          | 0.902    | 0.857    | 0.812   | 0.002             |
|                     | KNN                | 0.829    | 0.794    | 0.819   | 0.003             |
| LA_SIZE             | Random Forest      | 1.000    | 1.000    | 0.975   | 0.055             |
|                     | Gradient Boosting  | 1.000    | 1.000    | 0.988   | 0.031             |
|                     | Decision Tree      | 1.000    | 1.000    | 0.988   | 0.001             |
|                     | SVM (RBF)          | 0.775    | 0.755    | 0.787   | 0.001             |
|                     | KNN                | 0.725    | 0.687    | 0.781   | 0.000             |
|                     | Logistic Regression| 0.600    | 0.533    | 0.625   | 0.013             |
| DIASTOLIC_FUNCTION  | Gradient Boosting  | 0.918    | 0.919    | 0.854   | 0.072             |
|                     | Random Forest      | 0.878    | 0.878    | 0.719   | 0.055             |
|                     | Decision Tree      | 0.796    | 0.796    | 0.828   | 0.001             |
|                     | SVM (RBF)          | 0.673    | 0.673    | 0.620   | 0.001             |
|                     | KNN                | 0.571    | 0.571    | 0.557   | 0.000             |
|                     | Logistic Regression| 0.510    | 0.503    | 0.594   | 0.013             |

## Overall Average Accuracy

| Algorithm           | Average Accuracy |
|--------------------|------------------|
| Gradient Boosting  | 97.3%            |
| Random Forest      | 95.9%            |
| Decision Tree      | 93.2%            |
| SVM (RBF)          | 78.4%            |
| KNN                | 70.9%            |
| Logistic Regression| 67.9%            |

## Key Observations
- **Gradient Boosting** and **Random Forest** consistently deliver the highest accuracy and F1-scores.
- **Decision Tree** also performs well but may overfit.
- **SVM, KNN, Logistic Regression** are less accurate for this dataset.
- **Neural Network** was not tested due to insufficient data (<1000 samples required).
- **XGBoost** requires label encoding (not used here).

## Recommendations
- **Deploy Gradient Boosting** for best accuracy.
- **Random Forest** is a strong alternative for interpretability and speed.
- Keep rule-based logic for LV_FUNCTION and LV_SIZE due to data limitations.

## How to Reproduce
- See `compare_algorithms.py` for code and methodology.
- Run: `python compare_algorithms.py`
- All results are reproducible with your current dataset.

---

*Prepared for project demonstration and review.*

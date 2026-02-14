# Quick Reference Card - Advanced Features

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt
pip install shap scipy

# Run complete demo
python demo_advanced_features.py
```

---

## 📊 Feature Cheat Sheet

### 1. Model Explainability

```python
from src.explainability import ModelExplainer

explainer = ModelExplainer()

# Feature importance
explainer.plot_feature_importance('EF', top_n=10)

# SHAP values
explainer.plot_shap_summary(data, 'EF', plot_type='bar')

# Partial dependence
explainer.plot_partial_dependence(data, 'EF', 
                                  features=['EF', 'age', 'LVID_D'])

# Single prediction explanation
report = explainer.generate_explanation_report(data, 'EF', instance_idx=0)
print(report['top_contributors'])
```

### 2. Sensitivity Analysis

```python
from src.sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer()

# One-at-a-time
oat = analyzer.one_at_a_time_sensitivity(base_case, 'EF',
                                         variation_range=(-0.2, 0.2))
analyzer.plot_oat_sensitivity(oat, 'EF')

# Monte Carlo
mc = analyzer.monte_carlo_simulation(base_case, 'EF',
                                     error_std=0.05, n_simulations=1000)
print(f"Mean: {mc['mean']:.2f}, 95% CI: {mc['confidence_interval_95']}")

# Comprehensive report
report = analyzer.generate_sensitivity_report(base_case, 'EF')
print(f"Robustness: {report['robustness_score']:.2f}")
```

### 3. Multi-Class Severity Grading

```python
from src.severity_grading import MultiClassSeverityGrader

grader = MultiClassSeverityGrader()

# Comprehensive grading
report = grader.comprehensive_grading(measurements, patient_info)

# Access results
dd = report['grades']['diastolic_dysfunction']
print(f"Grade: {dd['grade']}, Confidence: {dd['confidence']:.0%}")

lvh = report['grades']['lvh']
print(f"Grade: {lvh['grade']}, Geometry: {lvh['geometry']}")

sf = report['grades']['systolic_function']
print(f"Grade: {sf['grade']}")

# Overall
print(f"Severity: {report['severity_summary']['overall_score']:.1f}/10")
print(f"Level: {report['severity_summary']['severity_level']}")

# Visualize
grader.plot_severity_dashboard(report, save_path='severity.png')
```

### 4. Risk Stratification

```python
from src.risk_stratification import ClinicalRiskStratifier

stratifier = ClinicalRiskStratifier()

# Clinical factors
clinical_factors = {
    'diabetes': True,
    'smoking': False,
    'hypertension': True,
    'ckd': False
}

# Comprehensive risk
risk = stratifier.compute_composite_risk_index(
    measurements, patient_info, clinical_factors
)

# Results
print(f"Composite Score: {risk['composite_score']:.1f}/100")
print(f"Risk Tier: {risk['risk_tier']}")
print(f"HF Risk (5y): {risk['heart_failure_risk']['five_year']:.1f}%")
print(f"Mortality (5y): {risk['mortality_risk']['five_year']:.1f}%")
print(f"Follow-up: {risk['follow_up_interval']}")

# Visualize
stratifier.plot_risk_dashboard(risk, patient_info, save_path='risk.png')
```

---

## 📈 Output Interpretation

### Severity Grades

| Grade | Diastolic Dysfunction | LVH | Systolic Function |
|-------|----------------------|-----|-------------------|
| 0 | Normal | Normal | Normal (EF≥55%) |
| 1 | Impaired Relaxation | Mild | Mild (EF 45-54%) |
| 2 | Pseudonormal | Moderate | Moderate (EF 30-44%) |
| 3 | Restrictive | Severe | Severe (EF<30%) |

### Risk Categories

| Score | Category | Action |
|-------|----------|--------|
| 0-25 | Low | Routine monitoring |
| 25-50 | Moderate | Regular follow-up (6-12 mo) |
| 50-75 | High | Intensive management (3-6 mo) |
| 75-100 | Very High | Urgent intervention (1-3 mo) |

### Robustness Score

| Score | Interpretation |
|-------|----------------|
| >0.9 | Excellent - highly stable |
| 0.7-0.9 | Good - acceptable sensitivity |
| 0.5-0.7 | Fair - noticeable sensitivity |
| <0.5 | Poor - high sensitivity to errors |

---

## 🎨 Visualization Quick Reference

```python
# Create outputs directory
import os
os.makedirs('outputs', exist_ok=True)

# Generate all visualizations
explainer.plot_feature_importance('EF', save_path='outputs/importance.png')
explainer.plot_shap_summary(data, 'EF', save_path='outputs/shap.png')
analyzer.plot_monte_carlo_results(mc, 'EF', save_path='outputs/mc.png')
grader.plot_severity_dashboard(severity, save_path='outputs/severity.png')
stratifier.plot_risk_dashboard(risk, patient_info, save_path='outputs/risk.png')
```

---

## 🔧 Common Patterns

### Complete Analysis Pipeline

```python
# 1. Load data
from src.extractor import MedicalReportExtractor
extractor = MedicalReportExtractor()
data = extractor.extract_from_pdf('report.pdf')

# 2. Analyze
from src.severity_grading import MultiClassSeverityGrader
from src.risk_stratification import ClinicalRiskStratifier

grader = MultiClassSeverityGrader()
stratifier = ClinicalRiskStratifier()

severity = grader.comprehensive_grading(data['measurements'], data['patient'])
risk = stratifier.compute_composite_risk_index(data['measurements'], data['patient'])

# 3. Report
print(f"\n=== CLINICAL SUMMARY ===")
print(f"Severity: {severity['severity_summary']['severity_level']}")
print(f"Risk: {risk['risk_tier']}")
print(f"\nKey Findings:")
for concern in severity['severity_summary']['primary_concerns']:
    print(f"  • {concern}")
print(f"\nRecommendations:")
for rec in risk['recommendations'][:3]:
    print(f"  • {rec}")
```

### Batch Processing

```python
import pandas as pd
from pathlib import Path

results = []
for json_file in Path('data/processed').glob('*.json'):
    data = json.load(open(json_file))
    
    severity = grader.comprehensive_grading(
        data['measurements'], data['patient']
    )
    
    risk = stratifier.compute_composite_risk_index(
        data['measurements'], data['patient']
    )
    
    results.append({
        'file': json_file.name,
        'severity_score': severity['severity_summary']['overall_score'],
        'risk_score': risk['composite_score'],
        'risk_tier': risk['risk_tier']
    })

df = pd.DataFrame(results)
df.to_csv('batch_results.csv', index=False)
print(f"Processed {len(results)} reports")
```

---

## 💡 Tips & Best Practices

### 1. Explainability
- Use SHAP for tree-based models
- PDP shows average effects
- ICE shows individual responses
- Always validate with clinical knowledge

### 2. Sensitivity Analysis
- Test ±5-10% for measurement errors
- Use ≥1000 Monte Carlo simulations
- Report confidence intervals for critical predictions
- Consider clinical tolerance levels

### 3. Severity Grading
- Check confidence scores (>0.7 recommended)
- Require ≥3 parameters for reliable grading
- Consider geometry for LVH interpretation
- Integrate with other clinical data

### 4. Risk Stratification
- Include all available clinical factors
- Update scores with new data
- Use for patient counseling
- Follow institutional intervention guidelines

---

## 🚨 Troubleshooting

### SHAP not working?
```bash
pip install shap --no-build-isolation
# or
conda install -c conda-forge shap
```

### Out of memory?
```python
# Reduce sample size
explainer.compute_shap_values(data.sample(100), 'EF')
```

### Missing parameters?
```python
# Check confidence scores
if grade['confidence'] < 0.5:
    print("Warning: Low confidence due to missing data")
```

---

## 📚 Documentation Links

- **Full Guide:** [ADVANCED_FEATURES_GUIDE.md](ADVANCED_FEATURES_GUIDE.md)
- **Summary:** [EXTENSION_SUMMARY.md](EXTENSION_SUMMARY.md)
- **Main README:** [README.md](README.md)

---

## 🎯 Quick Test

```bash
# Verify installation
python -c "
from src.explainability import ModelExplainer
from src.sensitivity_analysis import SensitivityAnalyzer
from src.severity_grading import MultiClassSeverityGrader
from src.risk_stratification import ClinicalRiskStratifier
print('✅ All modules imported successfully!')
"

# Run demo
python demo_advanced_features.py
```

---

**Version:** 2.0 | **Last Updated:** Feb 2026

# 🚀 Advanced Features Guide

## Overview

This guide documents the advanced features added to the Medical Report Interpretation System, extending it from basic interpretation to comprehensive clinical analysis with explainability, sensitivity analysis, multi-class grading, and risk stratification.

---

## Table of Contents

1. [Model Explainability](#1-model-explainability)
2. [Sensitivity Analysis](#2-sensitivity-analysis)
3. [Multi-Class Severity Grading](#3-multi-class-severity-grading)
4. [Risk Stratification](#4-risk-stratification)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [API Reference](#api-reference)

---

## 1. Model Explainability

### Overview

The explainability module provides transparency into ML model predictions using SHAP (SHapley Additive exPlanations), Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE) curves.

### Features

- **SHAP Values**: Understand feature contributions to predictions
- **Feature Importance**: Identify most influential parameters
- **Partial Dependence Plots**: Visualize feature effects on predictions
- **ICE Plots**: See individual prediction responses
- **Waterfall Plots**: Explain single predictions step-by-step

### Usage

```python
from src.explainability import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model_dir='models')

# Load your data
import pandas as pd
data = pd.DataFrame({
    'age': [60], 'sex_encoded': [1], 'EF': [55],
    'LVID_D': [5.0], 'IVS_D': [1.1], ...
})

# Get feature importance
importance_df = explainer.get_feature_importance('EF')
print(importance_df.head())

# Plot SHAP summary
explainer.plot_shap_summary(data, 'EF', plot_type='bar')

# Plot partial dependence
explainer.plot_partial_dependence(data, 'EF', features=['EF', 'age'])

# Generate explanation for single prediction
report = explainer.generate_explanation_report(data, 'EF', instance_idx=0)
print(f"Top contributors: {report['top_contributors']}")
```

### Key Methods

| Method | Description |
|--------|-------------|
| `compute_shap_values()` | Calculate SHAP values for predictions |
| `plot_shap_summary()` | Create summary plot of feature importance |
| `plot_shap_waterfall()` | Explain individual prediction |
| `plot_shap_dependence()` | Show feature effects and interactions |
| `plot_partial_dependence()` | Create PDP plots |
| `plot_ice_curves()` | Generate ICE plots |
| `get_feature_importance()` | Extract feature importance from model |

### Interpretation Guide

**SHAP Values:**
- **Positive values** push prediction higher
- **Negative values** push prediction lower
- **Magnitude** indicates strength of effect

**Partial Dependence:**
- Shows **average** effect of feature on prediction
- Useful for understanding non-linear relationships
- Can reveal threshold effects in clinical parameters

---

## 2. Sensitivity Analysis

### Overview

Sensitivity analysis tests model robustness by varying inputs and quantifying uncertainty in predictions due to measurement errors.

### Features

- **One-at-a-Time (OAT) Analysis**: Test individual parameter sensitivity
- **Monte Carlo Simulation**: Quantify prediction uncertainty
- **Global Sensitivity**: Rank feature importance across population
- **Feature Interaction**: Analyze joint effects of parameters
- **Robustness Scoring**: Assess model stability

### Usage

```python
from src.sensitivity_analysis import SensitivityAnalyzer

# Initialize analyzer
analyzer = SensitivityAnalyzer(model_dir='models')

# Define base case
base_case = pd.DataFrame({
    'age': [60], 'sex_encoded': [1], 'EF': [55],
    'LVID_D': [5.0], ...
})

# One-at-a-time sensitivity
oat_results = analyzer.one_at_a_time_sensitivity(
    base_case, 'EF',
    features=['EF', 'LVID_D'],
    variation_range=(-0.2, 0.2),  # ±20%
    n_steps=20
)

analyzer.plot_oat_sensitivity(oat_results, 'EF')

# Monte Carlo simulation
mc_results = analyzer.monte_carlo_simulation(
    base_case, 'EF',
    error_std=0.05,  # 5% measurement error
    n_simulations=1000
)

print(f"Mean: {mc_results['mean']:.3f}")
print(f"95% CI: {mc_results['confidence_interval_95']}")

analyzer.plot_monte_carlo_results(mc_results, 'EF')

# Comprehensive report
report = analyzer.generate_sensitivity_report(base_case, 'EF')
print(f"Robustness score: {report['robustness_score']:.3f}")
print(f"Interpretation: {report['robustness_interpretation']}")
```

### Key Methods

| Method | Description |
|--------|-------------|
| `one_at_a_time_sensitivity()` | Vary each feature independently |
| `monte_carlo_simulation()` | Propagate measurement uncertainty |
| `global_sensitivity_analysis()` | Correlation-based sensitivity ranking |
| `feature_interaction_sensitivity()` | Test two-feature interactions |
| `generate_sensitivity_report()` | Comprehensive analysis |

### Clinical Interpretation

**Robustness Score:**
- **> 0.9**: Excellent - predictions highly stable
- **0.7-0.9**: Good - acceptable sensitivity
- **0.5-0.7**: Fair - noticeable sensitivity
- **< 0.5**: Poor - high sensitivity to errors

**Use Cases:**
- Validate model reliability for clinical use
- Identify parameters requiring precise measurement
- Quantify confidence intervals for predictions
- Assess impact of measurement errors

---

## 3. Multi-Class Severity Grading

### Overview

Extends binary classification to graduated severity assessment across multiple cardiac conditions.

### Supported Gradings

#### Diastolic Dysfunction
- **Normal**: Normal diastolic function
- **Grade 1**: Impaired relaxation
- **Grade 2**: Pseudonormal pattern
- **Grade 3**: Restrictive pattern

#### Left Ventricular Hypertrophy (LVH)
- **Normal**: No hypertrophy
- **Mild LVH**: Early hypertrophy
- **Moderate LVH**: Significant hypertrophy
- **Severe LVH**: Advanced hypertrophy

#### Systolic Function
- **Normal**: EF ≥ 55%
- **Mild**: EF 45-54%
- **Moderate**: EF 30-44%
- **Severe**: EF < 30%

### Usage

```python
from src.severity_grading import MultiClassSeverityGrader

# Initialize grader
grader = MultiClassSeverityGrader()

# Patient measurements
measurements = {
    'EF': 62, 'FS': 35,
    'MV_E_A': 0.7, 'E_prime': 6.5, 'E_E_prime': 14,
    'LA_volume_index': 42,
    'LVID_D': 5.2, 'IVS_D': 1.2, 'LVPW_D': 1.1,
    'LV_mass_index': 120
}

patient_info = {'age': 65, 'sex': 'M'}

# Comprehensive grading
report = grader.comprehensive_grading(measurements, patient_info)

# Access results
print(f"Diastolic: {report['grades']['diastolic_dysfunction']['grade']}")
print(f"LVH: {report['grades']['lvh']['grade']}")
print(f"Systolic: {report['grades']['systolic_function']['grade']}")

print(f"Overall Severity: {report['severity_summary']['overall_score']:.1f}/10")
print(f"Level: {report['severity_summary']['severity_level']}")

# Display recommendations
for rec in report['clinical_recommendations']:
    print(f"• {rec}")

# Visualize
grader.plot_severity_dashboard(report)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `grade_diastolic_dysfunction()` | Grade diastolic function (0-3) |
| `grade_lvh()` | Grade LVH severity with geometry |
| `grade_systolic_function()` | Grade systolic function |
| `comprehensive_grading()` | Complete multi-parameter assessment |
| `plot_severity_dashboard()` | Visual summary dashboard |

### Clinical Guidelines

Grading based on:
- **ASE/EACVI Guidelines** for diastolic dysfunction
- **Standard criteria** for LVH and geometry
- **International standards** for systolic function

### Output Format

```python
{
    'grades': {
        'diastolic_dysfunction': {
            'grade': 'Grade 2 - Pseudonormal',
            'confidence': 0.85,
            'numeric_grade': 2,
            'description': '...'
        },
        'lvh': {
            'grade': 'Moderate LVH',
            'geometry': 'Concentric LVH',
            'confidence': 0.75,
            'numeric_grade': 2
        },
        'systolic_function': {
            'grade': 'Normal',
            'confidence': 1.0,
            'numeric_grade': 0
        }
    },
    'severity_summary': {
        'overall_score': 4.5,
        'severity_level': 'Moderate',
        'primary_concerns': [...]
    },
    'clinical_recommendations': [...]
}
```

---

## 4. Risk Stratification

### Overview

Comprehensive risk assessment system computing multiple risk scores and providing integrated clinical recommendations.

### Risk Scores

1. **Cardiovascular Risk Score**: Overall cardiac risk (0-100)
2. **Heart Failure Risk**: Probability of developing HF
3. **Mortality Risk**: 1, 5, and 10-year mortality estimates
4. **Composite Risk Index**: Integrated assessment

### Usage

```python
from src.risk_stratification import ClinicalRiskStratifier

# Initialize stratifier
stratifier = ClinicalRiskStratifier()

# Patient data
measurements = {
    'EF': 50, 'LVID_D': 5.5, 'IVS_D': 1.3,
    'LA_DIMENSION': 4.2, 'MV_E_A': 0.75,
    'E_E_prime': 12, 'LV_MASS': 260,
    'MR_grade': 1, 'AR_grade': 0
}

patient_info = {'age': 62, 'sex': 'M'}

clinical_factors = {
    'diabetes': True,
    'smoking': False,
    'hypertension': True,
    'ckd': False
}

# Compute comprehensive risk
risk = stratifier.compute_composite_risk_index(
    measurements, patient_info, clinical_factors
)

# Display results
print(f"Composite Score: {risk['composite_score']:.1f}/100")
print(f"Risk Tier: {risk['risk_tier']}")

print("\nCardiovascular Risk:")
print(f"  Score: {risk['cardiovascular_risk']['score']:.1f}")
print(f"  Category: {risk['cardiovascular_risk']['category']}")

print("\nHeart Failure Risk:")
print(f"  1-year: {risk['heart_failure_risk']['one_year']:.1f}%")
print(f"  5-year: {risk['heart_failure_risk']['five_year']:.1f}%")

print("\nMortality Risk:")
print(f"  5-year: {risk['mortality_risk']['five_year']:.1f}%")
print(f"  10-year: {risk['mortality_risk']['ten_year']:.1f}%")

print("\nRecommendations:")
for rec in risk['recommendations']:
    print(f"  • {rec}")

print(f"\nFollow-up: {risk['follow_up_interval']}")

# Visualize
stratifier.plot_risk_dashboard(risk, patient_info)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `compute_cardiovascular_risk_score()` | Overall CV risk |
| `compute_heart_failure_risk()` | HF development risk |
| `compute_mortality_risk()` | Mortality estimates |
| `compute_composite_risk_index()` | Integrated assessment |
| `plot_risk_dashboard()` | Comprehensive visualization |

### Risk Categories

| Score Range | Category | Action |
|-------------|----------|--------|
| 0-25 | Low Risk | Routine monitoring |
| 25-50 | Moderate Risk | Regular follow-up |
| 50-75 | High Risk | Intensive management |
| 75-100 | Very High Risk | Urgent intervention |

### Risk Weights

Default weight distribution:
- **Systolic function (EF)**: 25%
- **Diastolic dysfunction**: 20%
- **LVH**: 15%
- **LV dilation**: 10%
- **LA enlargement**: 10%
- **Valvular disease**: 10%
- **Age**: 10%

### Clinical Factors Impact

| Factor | Risk Multiplier |
|--------|-----------------|
| Diabetes | 1.5x |
| Smoking | 1.8x |
| Chronic Kidney Disease | 2.0x |
| Hypertension | 1.3x |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Additional Dependencies

For full functionality:

```bash
pip install shap scipy
```

### Optional (for deep learning features):

```bash
pip install torch transformers
```

---

## Quick Start

### 1. Run Complete Demo

```bash
python demo_advanced_features.py
```

This demonstrates all features with example cases and generates visualizations in `outputs/`.

### 2. Integrate Into Workflow

```python
# Complete analysis pipeline
from src.explainability import ModelExplainer
from src.sensitivity_analysis import SensitivityAnalyzer
from src.severity_grading import MultiClassSeverityGrader
from src.risk_stratification import ClinicalRiskStratifier

# Your patient data
measurements = {...}
patient_info = {'age': 65, 'sex': 'M'}

# 1. Severity grading
grader = MultiClassSeverityGrader()
severity = grader.comprehensive_grading(measurements, patient_info)

# 2. Risk assessment
stratifier = ClinicalRiskStratifier()
risk = stratifier.compute_composite_risk_index(measurements, patient_info)

# 3. Model explainability (if using ML predictions)
explainer = ModelExplainer()
explanation = explainer.generate_explanation_report(data, category='EF')

# 4. Sensitivity analysis
analyzer = SensitivityAnalyzer()
sensitivity = analyzer.generate_sensitivity_report(data, category='EF')
```

---

## API Reference

### Common Parameters

- **measurements**: `Dict[str, float]` - Cardiac measurements
- **patient_info**: `Dict[str, Any]` - Demographics (age, sex)
- **clinical_factors**: `Dict[str, Any]` - Risk factors (diabetes, smoking, etc.)
- **category**: `str` - Model category for analysis
- **save_path**: `Optional[str]` - Path to save visualizations

### Return Types

Most methods return structured dictionaries with:
- Numeric scores/grades
- Confidence levels
- Interpretations
- Clinical recommendations
- Supporting data

### Visualization Options

All plotting functions support:
- `save_path`: Save figure to file
- `dpi`: Resolution (default 300)
- `figsize`: Figure dimensions
- Custom styling through matplotlib

---

## Best Practices

### 1. Model Explainability
- Run SHAP analysis on representative dataset
- Use PDP plots to identify non-linear effects
- Validate explanations with clinical knowledge
- Document important features for transparency

### 2. Sensitivity Analysis
- Test with realistic measurement error ranges (±5-10%)
- Run Monte Carlo with sufficient simulations (≥1000)
- Consider clinical tolerance when interpreting results
- Report confidence intervals for critical predictions

### 3. Severity Grading
- Ensure sufficient parameters for confident grading
- Review confidence scores before clinical use
- Consider geometry alongside severity
- Integrate with other clinical findings

### 4. Risk Stratification
- Include all available clinical factors
- Update scores as new data becomes available
- Use for patient counseling and shared decision-making
- Follow institutional guidelines for interventions

---

## Troubleshooting

### SHAP Installation Issues

```bash
# If SHAP fails to install
pip install shap --no-build-isolation

# Or use conda
conda install -c conda-forge shap
```

### Memory Issues with Large Datasets

```python
# Use sampling for SHAP
background = shap.sample(data, 100)  # Reduced sample size

# Or use TreeExplainer (faster for tree models)
explainer = shap.TreeExplainer(model)
```

### Missing Clinical Parameters

The grading and risk systems are designed to work with partial data:
- Confidence scores reflect data completeness
- Missing parameters reduce confidence but still provide estimates
- "Indeterminate" grade indicates insufficient data

---

## References

### Clinical Guidelines
- ASE/EACVI Guidelines for Diastolic Function Assessment
- ASE Guidelines for LVH and Geometry
- ACC/AHA Heart Failure Risk Assessment

### Technical References
- Lundberg et al. (2017) - SHAP values
- Friedman (2001) - Partial Dependence Plots
- Saltelli et al. (2008) - Global Sensitivity Analysis

---

## Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review demo script for examples
3. Examine source code docstrings
4. Create issue with reproducible example

---

## License

Same as main project.

---

## Changelog

### Version 2.0 (Current)
- ✅ Added model explainability (SHAP, PDP, ICE)
- ✅ Implemented sensitivity analysis
- ✅ Extended to multi-class severity grading
- ✅ Added comprehensive risk stratification
- ✅ Created integrated demo system
- ✅ Enhanced documentation

### Version 1.0
- Basic interpretation engine
- Rule-based classification
- PDF extraction
- REST API

---

**Last Updated:** February 2026

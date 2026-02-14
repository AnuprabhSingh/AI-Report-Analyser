"""
Advanced Features Demo Script
Demonstrates all newly implemented features:
1. Model Explainability (SHAP, PDP, ICE plots)
2. Sensitivity Analysis
3. Multi-class Severity Grading
4. Risk Stratification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.explainability import ModelExplainer
from src.sensitivity_analysis import SensitivityAnalyzer
from src.severity_grading import MultiClassSeverityGrader
from src.risk_stratification import ClinicalRiskStratifier


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def _safe_name(name: str) -> str:
    return name.lower().replace(' ', '_')


def load_processed_samples(processed_dir: str = 'data/processed', max_samples: int = 300):
    """Load processed JSON samples from training dataset or individual files."""
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        return []

    dataset_path = processed_path / 'training_dataset.json'
    samples = []

    if dataset_path.exists():
        try:
            with open(dataset_path, 'r') as f:
                samples = json.load(f)
        except Exception:
            samples = []
    else:
        json_files = [p for p in processed_path.glob('*.json') if p.name != 'training_dataset.json']
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    samples.append(json.load(f))
            except Exception:
                continue

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


def load_model_metadata(model_dir: str = 'models'):
    """Load model metadata (feature names, categories)."""
    metadata_path = Path(model_dir) / 'model_metadata.json'
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def build_feature_dataframe(samples, feature_names):
    """Build feature dataframe aligned to model metadata."""
    rows = []
    for sample in samples:
        measurements = sample.get('measurements', {})
        patient = sample.get('patient', {})

        row = {}
        for feature in feature_names:
            if feature == 'age':
                row['age'] = patient.get('age', 0)
            elif feature == 'sex':
                sex = patient.get('sex', 'M')
                row['sex'] = 1 if str(sex).upper() == 'M' else 0
            else:
                row[feature] = measurements.get(feature, 0)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Replace zeros with median for measurement columns
    for col in df.columns:
        if col not in ['age', 'sex']:
            non_zero = df[df[col] > 0][col]
            if not non_zero.empty:
                median_val = non_zero.median()
                df[col] = df[col].replace(0, median_val)

    # Fill missing age with median if available
    if 'age' in df.columns:
        non_zero_age = df[df['age'] > 0]['age']
        if not non_zero_age.empty:
            df['age'] = df['age'].replace(0, non_zero_age.median())

    # Fill missing sex with mode if available
    if 'sex' in df.columns:
        if df['sex'].dropna().empty:
            df['sex'] = df['sex'].fillna(0)
        else:
            df['sex'] = df['sex'].fillna(df['sex'].mode().iloc[0])

    df = df.fillna(0)
    return df


def get_cases_from_samples(samples, max_cases: int = 3):
    """Extract patient cases from processed samples."""
    cases = []
    for sample in samples:
        measurements = sample.get('measurements', {})
        patient = sample.get('patient', {})
        if measurements and patient:
            case_name = patient.get('name', sample.get('file_name', 'Processed Case'))
            cases.append({
                'name': f"Processed Case - {case_name}",
                'measurements': measurements,
                'patient_info': {
                    'age': patient.get('age', 50),
                    'sex': patient.get('sex', 'M')
                },
                'clinical_factors': {
                    'diabetes': False,
                    'smoking': False,
                    'hypertension': False,
                    'ckd': False
                }
            })
        if len(cases) >= max_cases:
            break
    return cases


def demo_explainability():
    """Demonstrate model explainability features."""
    print_section_header("1. MODEL EXPLAINABILITY DEMO")
    
    # Check if models exist
    model_dir = 'models'
    if not Path(model_dir).exists() or not list(Path(model_dir).glob('model_*.pkl')):
        print("⚠ No trained models found. Please train models first.")
        print("  Run: python train_interpretation_model.py")
        return
    
    try:
        # Initialize explainer
        print("Initializing Model Explainer...")
        explainer = ModelExplainer(model_dir=model_dir)

        metadata = load_model_metadata(model_dir)
        if not metadata:
            print("⚠ Model metadata not found. Using fallback sample data.")
            sample_data = pd.DataFrame({
                'age': [55, 65, 45, 70],
                'sex': [1, 0, 1, 0],
                'EF': [60, 45, 65, 35],
                'LVID_D': [4.8, 5.5, 4.5, 6.0],
                'IVS_D': [1.0, 1.4, 0.9, 1.6],
                'LA_DIMENSION': [3.5, 4.2, 3.3, 4.8],
                'MV_E_A': [1.2, 0.7, 1.4, 2.2]
            })
        else:
            samples = load_processed_samples()
            if not samples:
                print("⚠ No processed data found. Using fallback sample data.")
                sample_data = pd.DataFrame({
                    'age': [55, 65, 45, 70],
                    'sex': [1, 0, 1, 0],
                    'EF': [60, 45, 65, 35],
                    'LVID_D': [4.8, 5.5, 4.5, 6.0],
                    'IVS_D': [1.0, 1.4, 0.9, 1.6],
                    'LA_DIMENSION': [3.5, 4.2, 3.3, 4.8],
                    'MV_E_A': [1.2, 0.7, 1.4, 2.2]
                })
            else:
                sample_data = build_feature_dataframe(samples, metadata['feature_names'])
                print(f"Loaded processed data: {len(sample_data)} samples")

        print(f"Sample data shape: {sample_data.shape}")
        print("\n✓ Model Explainer initialized successfully")
        
        # Get available models
        if explainer.models:
            categories_to_run = []
            for name, model in explainer.models.items():
                classes = getattr(model, 'classes_', [])
                importances = getattr(model, 'feature_importances_', None)
                if classes is not None and len(classes) > 1:
                    if importances is None or np.sum(importances) > 0:
                        categories_to_run.append(name)

            if not categories_to_run:
                categories_to_run = list(explainer.models.keys())

            for category in categories_to_run:
                print(f"\n📊 Analyzing model: {category}")
                category_slug = _safe_name(category)
            
                # Feature importance
                print("\n1.1 Feature Importance Analysis")
                importance_df = explainer.get_feature_importance(category, X=sample_data)
                if not importance_df.empty:
                    print("\nTop 5 Important Features:")
                    print(importance_df.head().to_string(index=False))
                    
                    # Plot feature importance
                    print("\n  Creating feature importance plot...")
                    explainer.plot_feature_importance(category, top_n=7,
                                                     save_path=f'outputs/feature_importance_{category_slug}.png',
                                                     X=sample_data)
                
                # SHAP values
                print("\n1.2 SHAP Analysis")
                try:
                    print("  Computing SHAP values...")
                    explainer.plot_shap_summary(sample_data, category, plot_type='bar',
                                               save_path=f'outputs/shap_summary_{category_slug}.png')
                    print("  ✓ SHAP summary plot created")
                except ImportError:
                    print("  ⚠ SHAP not installed. Install with: pip install shap")
                
                # Partial Dependence Plots
                print("\n1.3 Partial Dependence Plots")
                features_to_plot = ['EF', 'LVID_D', 'IVS_D']
                available_features = [
                    f for f in features_to_plot
                    if f in sample_data.columns and sample_data[f].var() > 0
                ]
                if available_features:
                    print(f"  Creating PDP for: {', '.join(available_features)}")
                    explainer.plot_partial_dependence(sample_data, category, 
                                                     available_features,
                                                     save_path=f'outputs/pdp_plots_{category_slug}.png')
                    print("  ✓ PDP plots created")
                else:
                    print("  ⚠ Skipping PDP (no variable features available)")
                
                # Generate explanation report
                print("\n1.4 Individual Prediction Explanation")
                report = explainer.generate_explanation_report(sample_data, category, instance_idx=0)
                print(f"\n  Prediction: {report['prediction']}")
                if report['top_contributors']:
                    print("\n  Top Contributing Features:")
                    for feature, contribution in report['top_contributors'][:3]:
                        print(f"    • {feature}: {contribution:.4f}")
        
        print("\n✅ Explainability demo completed!")
        
    except Exception as e:
        print(f"❌ Error in explainability demo: {e}")
        import traceback
        traceback.print_exc()


def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis features."""
    print_section_header("2. SENSITIVITY ANALYSIS DEMO")
    
    # Check if models exist
    model_dir = 'models'
    if not Path(model_dir).exists() or not list(Path(model_dir).glob('model_*.pkl')):
        print("⚠ No trained models found. Skipping sensitivity analysis.")
        return
    
    try:
        # Initialize analyzer
        print("Initializing Sensitivity Analyzer...")
        analyzer = SensitivityAnalyzer(model_dir=model_dir)

        metadata = load_model_metadata(model_dir)
        samples = load_processed_samples()
        if metadata and samples:
            feature_df = build_feature_dataframe(samples, metadata['feature_names'])
            base_case = feature_df.sample(1, random_state=42)
            print("\nBase Case Parameters (from processed data):")
            print(base_case.to_string(index=False))
        else:
            # Fallback base case
            base_case = pd.DataFrame({
                'age': [60],
                'sex': [1],
                'EF': [55],
                'LVID_D': [5.0],
                'IVS_D': [1.1],
                'LA_DIMENSION': [3.8],
                'MV_E_A': [1.0]
            })
            print("\nBase Case Parameters (fallback):")
            print(base_case.to_string(index=False))
        
        if analyzer.models:
            category = list(analyzer.models.keys())[0]
            print(f"\n📊 Analyzing model: {category}")
            
            # One-at-a-time sensitivity
            print("\n2.1 One-at-a-Time (OAT) Sensitivity Analysis")
            features_to_analyze = [
                f for f in ['EF', 'LVID_D', 'IVS_D']
                if f in base_case.columns and not pd.isna(base_case[f].iloc[0])
            ]
            oat_results = analyzer.one_at_a_time_sensitivity(
                base_case, category, 
                features=features_to_analyze,
                variation_range=(-0.15, 0.15),
                n_steps=15
            )
            
            if oat_results:
                print(f"  Analyzed {len(oat_results)} features")
                print("\n  Creating sensitivity plots...")
                analyzer.plot_oat_sensitivity(oat_results, category,
                                             save_path='outputs/oat_sensitivity.png')
                print("  ✓ OAT sensitivity plots created")
            else:
                print("  ⚠ OAT results empty; skipping plot")
            
            # Monte Carlo simulation
            print("\n2.2 Monte Carlo Uncertainty Quantification")
            print("  Running 1000 simulations with 5% measurement error...")
            mc_results = analyzer.monte_carlo_simulation(
                base_case, category,
                error_std=0.05,
                n_simulations=1000
            )
            
            if mc_results:
                print(f"\n  Results Summary:")
                print(f"    Mean Prediction: {mc_results['mean']:.4f}")
                print(f"    Std Deviation: {mc_results['std']:.4f}")
                print(f"    95% CI: [{mc_results['confidence_interval_95'][0]:.4f}, "
                      f"{mc_results['confidence_interval_95'][1]:.4f}]")
                print(f"    Coefficient of Variation: {mc_results['coefficient_of_variation']:.2%}")
                
                print("\n  Creating Monte Carlo plots...")
                analyzer.plot_monte_carlo_results(mc_results, category,
                                                 save_path='outputs/monte_carlo.png')
                print("  ✓ Monte Carlo plots created")
            
            # Global sensitivity
            print("\n2.3 Global Sensitivity Analysis")
            if metadata and samples:
                sample_data = feature_df.copy()
            else:
                sample_data = pd.DataFrame({
                    'age': np.random.randint(40, 80, 100),
                    'sex': np.random.randint(0, 2, 100),
                    'EF': np.random.uniform(35, 70, 100),
                    'LVID_D': np.random.uniform(4.0, 6.5, 100),
                    'IVS_D': np.random.uniform(0.8, 1.6, 100),
                    'LA_DIMENSION': np.random.uniform(3.0, 5.0, 100),
                    'MV_E_A': np.random.uniform(0.6, 2.0, 100)
                })
            
            sensitivity_df = analyzer.global_sensitivity_analysis(
                sample_data, category, n_samples=100
            )
            
            if not sensitivity_df.empty:
                print("\n  Top 5 Most Sensitive Features:")
                print(sensitivity_df.head().to_string(index=False))
                
                print("\n  Creating global sensitivity plots...")
                analyzer.plot_global_sensitivity(sensitivity_df,
                                                save_path='outputs/global_sensitivity.png')
                print("  ✓ Global sensitivity plots created")
            
            # Generate comprehensive report
            print("\n2.4 Comprehensive Sensitivity Report")
            report = analyzer.generate_sensitivity_report(base_case, category)
            
            if report.get('robustness_score'):
                print(f"\n  Robustness Score: {report['robustness_score']:.3f}")
                print(f"  Interpretation: {report['robustness_interpretation']}")
        
        print("\n✅ Sensitivity analysis demo completed!")
        
    except Exception as e:
        print(f"❌ Error in sensitivity analysis demo: {e}")
        import traceback
        traceback.print_exc()


def demo_severity_grading():
    """Demonstrate multi-class severity grading."""
    print_section_header("3. MULTI-CLASS SEVERITY GRADING DEMO")
    
    try:
        # Initialize grader
        print("Initializing Severity Grader...")
        grader = MultiClassSeverityGrader()
        print("✓ Severity Grader initialized")
        
        # Load processed cases if available
        samples = load_processed_samples()
        cases = get_cases_from_samples(samples, max_cases=3)

        if not cases:
            # Fallback sample patient cases
            cases = [
                {
                    'name': 'Patient A - Moderate Diastolic Dysfunction',
                    'measurements': {
                        'EF': 62,
                        'MV_E_A': 0.7,
                        'E_prime': 6.5,
                        'E_E_prime': 14,
                        'LA_volume_index': 42,
                        'LVID_D': 5.2,
                        'IVS_D': 1.2,
                        'LVPW_D': 1.1,
                        'LV_mass_index': 120
                    },
                    'patient_info': {'age': 65, 'sex': 'M'}
                },
                {
                    'name': 'Patient B - Severe LVH with Systolic Dysfunction',
                    'measurements': {
                        'EF': 38,
                        'FS': 18,
                        'MV_E_A': 0.6,
                        'LVID_D': 6.0,
                        'IVS_D': 1.7,
                        'LVPW_D': 1.6,
                        'LV_mass_index': 155,
                        'relative_wall_thickness': 0.53
                    },
                    'patient_info': {'age': 58, 'sex': 'M'}
                },
                {
                    'name': 'Patient C - Normal Function',
                    'measurements': {
                        'EF': 65,
                        'FS': 38,
                        'MV_E_A': 1.2,
                        'LVID_D': 4.5,
                        'IVS_D': 0.9,
                        'LVPW_D': 0.9,
                        'LV_mass_index': 95
                    },
                    'patient_info': {'age': 45, 'sex': 'F'}
                }
            ]
        
        for i, case in enumerate(cases, 1):
            print(f"\n{'─' * 80}")
            print(f"Case {i}: {case['name']}")
            print(f"{'─' * 80}")
            
            # Comprehensive grading
            grading_report = grader.comprehensive_grading(
                case['measurements'],
                case['patient_info']
            )
            
            # Display results
            print("\n📋 GRADING RESULTS:")
            
            # Systolic function
            if 'systolic_function' in grading_report['grades']:
                sf = grading_report['grades']['systolic_function']
                print(f"\n  Systolic Function: {sf['grade']}")
                print(f"    Confidence: {sf['confidence']:.2%}")
                print(f"    {sf['description']}")
            
            # Diastolic dysfunction
            if 'diastolic_dysfunction' in grading_report['grades']:
                dd = grading_report['grades']['diastolic_dysfunction']
                print(f"\n  Diastolic Function: {dd['grade']}")
                print(f"    Confidence: {dd['confidence']:.2%}")
                print(f"    {dd['description']}")
                print(f"    Parameters evaluated: {', '.join(dd['parameters_evaluated'])}")
            
            # LVH
            if 'lvh' in grading_report['grades']:
                lvh = grading_report['grades']['lvh']
                print(f"\n  LVH Assessment: {lvh['grade']}")
                print(f"    Confidence: {lvh['confidence']:.2%}")
                print(f"    Geometry: {lvh['geometry']}")
                print(f"    Parameters evaluated: {', '.join(lvh['parameters_evaluated'])}")
            
            # Severity summary
            summary = grading_report['severity_summary']
            print(f"\n  Overall Severity Score: {summary['overall_score']:.1f}/10")
            print(f"  Severity Level: {summary['severity_level']}")
            print(f"\n  Primary Concerns:")
            for concern in summary['primary_concerns']:
                print(f"    • {concern}")
            
            # Recommendations
            print(f"\n  Clinical Recommendations:")
            for rec in grading_report['clinical_recommendations']:
                print(f"    • {rec}")
            
            # Create dashboard for first patient
            if i == 1:
                print("\n  Creating severity dashboard...")
                grader.plot_severity_dashboard(grading_report,
                                              save_path='outputs/severity_dashboard.png')
                print("  ✓ Dashboard saved to outputs/severity_dashboard.png")
        
        print("\n✅ Severity grading demo completed!")
        
    except Exception as e:
        print(f"❌ Error in severity grading demo: {e}")
        import traceback
        traceback.print_exc()


def demo_risk_stratification():
    """Demonstrate risk stratification features."""
    print_section_header("4. RISK STRATIFICATION DEMO")
    
    try:
        # Initialize risk stratifier
        print("Initializing Risk Stratifier...")
        stratifier = ClinicalRiskStratifier()
        print("✓ Risk Stratifier initialized")
        
        # Load processed cases if available
        samples = load_processed_samples()
        cases = get_cases_from_samples(samples, max_cases=3)

        if not cases:
            # Fallback sample patient cases with varying risk profiles
            cases = [
                {
                    'name': 'Low Risk Patient',
                    'measurements': {
                        'EF': 65,
                        'LVID_D': 4.5,
                        'IVS_D': 0.9,
                        'LA_DIMENSION': 3.5,
                        'MV_E_A': 1.2,
                        'E_E_prime': 8,
                        'LV_MASS': 180
                    },
                    'patient_info': {'age': 45, 'sex': 'M'},
                    'clinical_factors': {
                        'diabetes': False,
                        'smoking': False,
                        'hypertension': False
                    }
                },
                {
                    'name': 'Moderate Risk Patient',
                    'measurements': {
                        'EF': 50,
                        'LVID_D': 5.5,
                        'IVS_D': 1.3,
                        'LA_DIMENSION': 4.2,
                        'MV_E_A': 0.75,
                        'E_E_prime': 12,
                        'LV_MASS': 260
                    },
                    'patient_info': {'age': 62, 'sex': 'M'},
                    'clinical_factors': {
                        'diabetes': True,
                        'smoking': False,
                        'hypertension': True
                    }
                },
                {
                    'name': 'High Risk Patient',
                    'measurements': {
                        'EF': 35,
                        'LVID_D': 6.2,
                        'IVS_D': 1.6,
                        'LA_DIMENSION': 4.8,
                        'MV_E_A': 2.1,
                        'E_E_prime': 16,
                        'LV_MASS': 320,
                        'MR_grade': 2,
                        'AR_grade': 1
                    },
                    'patient_info': {'age': 72, 'sex': 'M'},
                    'clinical_factors': {
                        'diabetes': True,
                        'smoking': True,
                        'hypertension': True,
                        'ckd': True
                    }
                }
            ]
        
        for i, case in enumerate(cases, 1):
            print(f"\n{'─' * 80}")
            print(f"Case {i}: {case['name']}")
            print(f"Age: {case['patient_info']['age']}, Sex: {case['patient_info']['sex']}")
            print(f"{'─' * 80}")
            
            # Compute comprehensive risk assessment
            risk_assessment = stratifier.compute_composite_risk_index(
                case['measurements'],
                case['patient_info'],
                case['clinical_factors']
            )
            
            # Display results
            print("\n🎯 RISK ASSESSMENT:")
            
            print(f"\n  Composite Risk Score: {risk_assessment['composite_score']:.1f}/100")
            print(f"  Risk Tier: {risk_assessment['risk_tier']}")
            
            # Cardiovascular risk
            cv_risk = risk_assessment['cardiovascular_risk']
            print(f"\n  Cardiovascular Risk:")
            print(f"    Score: {cv_risk['score']:.1f}")
            print(f"    Category: {cv_risk['category']}")
            print(f"    Percentile: {cv_risk['percentile']:.0f}th")
            
            # Heart failure risk
            hf_risk = risk_assessment['heart_failure_risk']
            print(f"\n  Heart Failure Risk:")
            print(f"    Score: {hf_risk['score']:.1f}")
            print(f"    Category: {hf_risk['category']}")
            print(f"    1-year risk: {hf_risk['one_year']:.1f}%")
            print(f"    5-year risk: {hf_risk['five_year']:.1f}%")
            
            # Mortality risk
            mort_risk = risk_assessment['mortality_risk']
            print(f"\n  Mortality Risk:")
            print(f"    Category: {mort_risk['category']}")
            print(f"    1-year: {mort_risk['one_year']:.1f}%")
            print(f"    5-year: {mort_risk['five_year']:.1f}%")
            print(f"    10-year: {mort_risk['ten_year']:.1f}%")
            
            # Recommendations
            print(f"\n  Clinical Recommendations:")
            for rec in risk_assessment['recommendations']:
                print(f"    • {rec}")
            
            print(f"\n  Recommended Follow-up: {risk_assessment['follow_up_interval']}")
            
            # Create dashboard for high-risk patient
            if 'High Risk' in case['name']:
                print("\n  Creating risk stratification dashboard...")
                stratifier.plot_risk_dashboard(risk_assessment, case['patient_info'],
                                              save_path='outputs/risk_dashboard.png')
                print("  ✓ Dashboard saved to outputs/risk_dashboard.png")
        
        print("\n✅ Risk stratification demo completed!")
        
    except Exception as e:
        print(f"❌ Error in risk stratification demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  ADVANCED FEATURES DEMONSTRATION")
    print("  Medical Report Interpretation System - Extended Capabilities")
    print("=" * 80)
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    print("\n✓ Created outputs directory for visualizations")
    
    # Run demonstrations
    print("\nRunning all demonstrations...")
    print("This may take a few minutes...\n")
    
    try:
        # 1. Explainability
        demo_explainability()
        
        # 2. Sensitivity Analysis
        demo_sensitivity_analysis()
        
        # 3. Severity Grading
        demo_severity_grading()
        
        # 4. Risk Stratification
        demo_risk_stratification()
        
        # Summary
        print("\n" + "=" * 80)
        print("  DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📁 Output files saved in: outputs/")
        print("\nGenerated files:")
        output_dir = Path('outputs')
        if output_dir.exists():
            for file in sorted(output_dir.glob('*.png')):
                print(f"  • {file.name}")
        
        print("\n✅ All advanced features demonstrated successfully!")
        print("\nNext steps:")
        print("  1. Review generated visualizations in outputs/")
        print("  2. Integrate features into your workflow")
        print("  3. Customize parameters for your specific use case")
        print("  4. Refer to documentation for detailed API usage")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

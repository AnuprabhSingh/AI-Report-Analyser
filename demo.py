#!/usr/bin/env python3
"""
Medical Report Interpretation System - Demo
Complete pipeline demonstration
"""

import json
from pathlib import Path
from src.extractor import MedicalReportExtractor
from src.predictor import ClinicalPredictor


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def main():
    """Run complete demo."""
    
    print_section("MEDICAL REPORT INTERPRETATION - DEMO")
    
    print("\nThis demo shows: PDF -> Extraction -> ML Interpretation\n")
    
    # Sample PDF
    sample_pdf = 'data/sample_reports/21-01-04-203828_APPE GAYATHRI DEVI 45 YRS_20210104_203828_20210104_204134.pdf'
    
    if not Path(sample_pdf).exists():
        print(f"ERROR: PDF not found: {sample_pdf}")
        return
    
    # STEP 1: Extract
    print_section("STEP 1: EXTRACTION")
    print(f"\nProcessing: {Path(sample_pdf).name}")
    
    extractor = MedicalReportExtractor()
    data = extractor.extract_from_pdf(sample_pdf)
    
    if not data:
        print("ERROR: Extraction failed")
        return
    
    print("\nPatient Info:")
    for k, v in data['patient'].items():
        print(f"  {k}: {v}")
    
    print(f"\nMeasurements: {len(data['measurements'])}")
    for param, value in list(data['measurements'].items())[:10]:
        print(f"  {param:20s}: {value:>8.2f}")
    
    # STEP 2: Interpret
    print_section("STEP 2: INTERPRETATION")
    
    predictor = ClinicalPredictor(use_ml=True)
    method = "ML-Based" if predictor.ml_available else "Rule-Based"
    print(f"\nMethod: {method}")
    
    interpretations = predictor.predict(
        data['measurements'],
        data['patient']
    )
    
    # STEP 3: Results
    print_section("CLINICAL REPORT")
    
    print(f"\nPatient: {data['patient'].get('name', 'Unknown')}")
    print(f"Age: {data['patient'].get('age', 'Unknown')}")
    print(f"Sex: {data['patient'].get('sex', 'Unknown')}")
    
    print("\nFINDINGS:")
    print("-" * 80)
    for category, text in interpretations.items():
        print(f"\n{category}:")
        print(f"  {text}")
    
    # Save
    output = 'data/processed/demo_interpretation.json'
    result = {
        'file_name': data['file_name'],
        'patient': data['patient'],
        'measurements': data['measurements'],
        'interpretations': interpretations,
        'method': method
    }
    
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print_section("SUMMARY")
    print(f"""
Success!
- Measurements: {len(data['measurements'])}
- Interpretations: {len(interpretations)}
- Method: {method}
- Saved to: {output}
""")
    
    if predictor.ml_available:
        print("ML models working correctly!")
    else:
        print("Using rule-based (train models: python run_training_workflow.py)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

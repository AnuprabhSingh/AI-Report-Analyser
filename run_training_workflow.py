#!/usr/bin/env python3
"""
Complete ML Training Workflow
Step-by-step guide to train interpretation models from PDF reports.
"""

import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files and dependencies exist."""
    print("=" * 80)
    print("CHECKING REQUIREMENTS")
    print("=" * 80)
    
    # Check PDF files
    pdf_dir = Path('data/sample_reports')
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    print(f"\n✓ PDF reports found: {len(pdf_files)}")
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found in data/sample_reports/")
        return False
    
    # Check required Python files
    required_files = [
        'src/extractor.py',
        'src/predictor.py',
        'src/rule_engine.py',
        'prepare_training_data.py',
        'train_interpretation_model.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"❌ Missing: {file}")
            all_exist = False
    
    return all_exist and len(pdf_files) > 0


def run_step(step_num, title, command, script_name):
    """Run a pipeline step."""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: {title}")
    print("=" * 80)
    
    response = input(f"\nRun {script_name}? (y/n): ").strip().lower()
    
    if response == 'y':
        print(f"\nExecuting: {command}")
        print("-" * 80)
        
        result = subprocess.run(command, shell=True)
        
        if result.returncode == 0:
            print(f"\n✓ Step {step_num} completed successfully")
            return True
        else:
            print(f"\n❌ Step {step_num} failed")
            return False
    else:
        print(f"⊘ Skipped step {step_num}")
        return False


def main():
    """Run complete training workflow."""
    
    print("\n" + "🎯" * 40)
    print("ML INTERPRETATION MODEL TRAINING WORKFLOW")
    print("🎯" * 40)
    
    print("""
This workflow will:
  1. Extract measurements from all PDF reports (~240 files)
  2. Generate rule-based interpretations for training labels
  3. Train ML models to predict interpretations from datapoints
  4. Save trained models for deployment
  5. Test the trained models
""")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix issues and try again.")
        sys.exit(1)
    
    print("\n✓ All requirements met!")
    
    # Step 1: Prepare training data
    step1_success = run_step(
        1,
        "Extract Data from PDFs",
        "python prepare_training_data.py",
        "prepare_training_data.py"
    )
    
    if not step1_success:
        print("\n⚠ Cannot proceed without training data")
        return
    
    # Check if dataset was created
    dataset_path = Path('data/processed/training_dataset.json')
    if not dataset_path.exists():
        print(f"\n❌ Training dataset not found: {dataset_path}")
        return
    
    # Step 2: Train models
    step2_success = run_step(
        2,
        "Train ML Models",
        "python train_interpretation_model.py",
        "train_interpretation_model.py"
    )
    
    if not step2_success:
        print("\n⚠ Model training incomplete")
        return
    
    # Step 3: Test trained models
    step3_success = run_step(
        3,
        "Test Trained Models",
        "python predict_with_ml.py",
        "predict_with_ml.py"
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    
    print(f"\nStep 1 (Data Extraction): {'✓ Complete' if step1_success else '✗ Failed/Skipped'}")
    print(f"Step 2 (Model Training): {'✓ Complete' if step2_success else '✗ Failed/Skipped'}")
    print(f"Step 3 (Model Testing): {'✓ Complete' if step3_success else '✗ Failed/Skipped'}")
    
    if step1_success and step2_success:
        print("\n" + "🎉" * 40)
        print("SUCCESS! Models are trained and ready to use")
        print("🎉" * 40)
        
        print("\nNext steps:")
        print("  • Test models: python predict_with_ml.py")
        print("  • Use in API: Update src/predictor.py to use ML models")
        print("  • Process new PDFs: python main.py interpret <pdf_file>")
    else:
        print("\n⚠ Workflow incomplete. Please review errors above.")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

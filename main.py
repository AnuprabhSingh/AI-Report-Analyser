#!/usr/bin/env python3
"""
Medical Report Interpretation System - Main CLI
Command-line interface for the automated medical report interpretation system.
"""

import argparse
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.extractor import MedicalReportExtractor
from src.predictor import ClinicalPredictor
from src.rule_engine import ClinicalRuleEngine
from src.model_trainer import ClinicalMLTrainer
from src.utils import format_clinical_output, save_json


def extract_command(args):
    """Handle extract command - extract data from PDF(s)."""
    print("=" * 60)
    print("EXTRACTING DATA FROM PDF")
    print("=" * 60)
    
    extractor = MedicalReportExtractor()
    
    if os.path.isfile(args.input):
        # Single file
        print(f"\nProcessing: {args.input}")
        data = extractor.extract_from_pdf(args.input)
        
        if data:
            # Save output
            output_path = args.output or 'extracted_data.json'
            save_json(data, output_path)
            print(f"\n✓ Saved to: {output_path}")
            
            # Print summary
            print(f"\nExtracted {len(data.get('measurements', {}))} measurements:")
            for param, value in data.get('measurements', {}).items():
                print(f"  • {param}: {value}")
        else:
            print("\n✗ Failed to extract data")
            return 1
    
    elif os.path.isdir(args.input):
        # Directory
        output_dir = args.output or 'data/processed'
        results = extractor.process_directory(args.input, output_dir)
        print(f"\n✓ Processed {len(results)} files")
    
    else:
        print(f"✗ Invalid input path: {args.input}")
        return 1
    
    return 0


def interpret_command(args):
    """Handle interpret command - generate interpretation from JSON or PDF."""
    print("=" * 60)
    print("GENERATING CLINICAL INTERPRETATION")
    print("=" * 60)
    
    # Initialize predictor
    model_dir = args.model_dir if hasattr(args, 'model_dir') else None
    predictor = ClinicalPredictor(model_dir=model_dir, use_ml=args.use_ml if hasattr(args, 'use_ml') else False)
    
    # Check input type
    if args.input.endswith('.json'):
        # JSON input
        print(f"\nLoading data from: {args.input}")
        result = predictor.predict_from_json(args.input)
    
    elif args.input.endswith('.pdf'):
        # PDF input - extract first
        print(f"\nExtracting data from: {args.input}")
        extractor = MedicalReportExtractor()
        extracted_data = extractor.extract_from_pdf(args.input)
        
        if not extracted_data:
            print("✗ Failed to extract data from PDF")
            return 1
        
        measurements = extracted_data['measurements']
        patient_info = extracted_data['patient']
        
        interpretations = predictor.predict(measurements, patient_info)
        
        result = {
            'file_name': os.path.basename(args.input),
            'patient': patient_info,
            'measurements': measurements,
            'interpretations': interpretations
        }
    
    else:
        print(f"✗ Unsupported file type: {args.input}")
        return 1
    
    # Display results
    print("\n" + "=" * 60)
    print("PATIENT INFORMATION")
    print("=" * 60)
    patient = result.get('patient', {})
    print(f"Name: {patient.get('name', 'N/A')}")
    print(f"Age: {patient.get('age', 'N/A')} years")
    print(f"Sex: {patient.get('sex', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("MEASUREMENTS")
    print("=" * 60)
    for param, value in result.get('measurements', {}).items():
        print(f"{param:20s}: {value:6.2f}")
    
    print("\n" + "=" * 60)
    print("CLINICAL INTERPRETATION")
    print("=" * 60)
    for category, interpretation in result.get('interpretations', {}).items():
        print(f"\n{category}:")
        print(f"  {interpretation}")
    
    # Save output if requested
    if args.output:
        save_json(result, args.output)
        print(f"\n✓ Saved to: {args.output}")
    
    print("\n" + "=" * 60)
    return 0


def train_command(args):
    """Handle train command - train ML models."""
    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)
    
    trainer = ClinicalMLTrainer(model_type='classification')
    
    if os.path.exists(args.data_dir):
        print(f"\nLoading training data from: {args.data_dir}")
        df, labels = trainer.load_training_data(args.data_dir)
    else:
        print(f"\n⚠ Data directory not found: {args.data_dir}")
        print("Generating synthetic training data...")
        df, labels = trainer.create_synthetic_training_data(n_samples=args.n_samples)
    
    print(f"\nTraining data: {len(df)} samples")
    print(f"Features: {list(df.columns)}")
    
    # Train models
    print("\nTraining models...")
    if 'EF' in labels:
        metrics = trainer.train_ef_classifier(df, labels['EF'])
        print(f"  EF Classifier - Accuracy: {metrics['test_accuracy']:.3f}")
    
    # Save models
    output_dir = args.output or 'models'
    print(f"\nSaving models to: {output_dir}")
    trainer.save_models(output_dir)
    
    print("\n✓ Training completed!")
    print("=" * 60)
    return 0


def batch_command(args):
    """Handle batch command - process multiple files."""
    print("=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    
    if not os.path.isdir(args.input):
        print(f"✗ Input must be a directory: {args.input}")
        return 1
    
    # Extract PDFs
    print("\n1. Extracting data from PDFs...")
    extractor = MedicalReportExtractor()
    processed_dir = args.output or 'data/processed'
    results = extractor.process_directory(args.input, processed_dir)
    
    # Generate interpretations
    print(f"\n2. Generating interpretations for {len(results)} reports...")
    predictor = ClinicalPredictor(use_ml=False)
    
    interpretations_dir = os.path.join(os.path.dirname(processed_dir), 'interpretations')
    predictor.batch_predict(processed_dir, interpretations_dir)
    
    print(f"\n✓ Batch processing completed!")
    print(f"  Processed: {len(results)} reports")
    print(f"  Interpretations saved to: {interpretations_dir}")
    print("=" * 60)
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Automated Medical Report Interpretation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data from a PDF
  python main.py extract input.pdf -o output.json
  
  # Extract from all PDFs in a directory
  python main.py extract data/sample_reports/ -o data/processed/
  
  # Generate interpretation from JSON
  python main.py interpret data/processed/report.json
  
  # Generate interpretation from PDF
  python main.py interpret input.pdf -o interpretation.json
  
  # Train ML models
  python main.py train -d data/processed/ -o models/
  
  # Batch process multiple files
  python main.py batch data/sample_reports/ -o data/processed/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract data from PDF')
    extract_parser.add_argument('input', help='PDF file or directory')
    extract_parser.add_argument('-o', '--output', help='Output JSON file or directory')
    
    # Interpret command
    interpret_parser = subparsers.add_parser('interpret', help='Generate clinical interpretation')
    interpret_parser.add_argument('input', help='JSON or PDF file')
    interpret_parser.add_argument('-o', '--output', help='Output JSON file')
    interpret_parser.add_argument('--model-dir', help='Directory with trained models')
    interpret_parser.add_argument('--use-ml', action='store_true', help='Use ML models if available')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('-d', '--data-dir', default='data/processed/',
                             help='Directory with training data (JSON files)')
    train_parser.add_argument('-o', '--output', default='models',
                             help='Output directory for trained models')
    train_parser.add_argument('-n', '--n-samples', type=int, default=200,
                             help='Number of synthetic samples if no data')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
    batch_parser.add_argument('input', help='Directory with PDF files')
    batch_parser.add_argument('-o', '--output', help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'extract':
            return extract_command(args)
        elif args.command == 'interpret':
            return interpret_command(args)
        elif args.command == 'train':
            return train_command(args)
        elif args.command == 'batch':
            return batch_command(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

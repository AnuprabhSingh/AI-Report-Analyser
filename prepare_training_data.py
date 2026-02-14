#!/usr/bin/env python3
"""
Prepare training dataset from all PDF reports.
Extracts measurements and generates interpretations for training.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from src.extractor import MedicalReportExtractor
from src.predictor import ClinicalPredictor

def prepare_training_data(input_dir: str, output_dir: str, dataset_filename: str):
    """Process all PDFs and create training dataset."""
    
    print("=" * 80)
    print("PREPARING TRAINING DATASET FROM PDF REPORTS")
    print("=" * 80)
    
    # Initialize components
    extractor = MedicalReportExtractor()
    predictor = ClinicalPredictor()
    
    # Paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files (recursive)
    pdf_files = list(input_dir.rglob('*.pdf'))
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found in data/sample_reports/")
        return
    
    # Storage for training data
    training_data = []
    failed_extractions = []
    
    print("\nProcessing PDFs...")
    print("-" * 80)
    
    # Process each PDF with progress bar
    for pdf_file in tqdm(pdf_files, desc="Extracting"):
        try:
            # Extract measurements
            result = extractor.extract_from_pdf(str(pdf_file))
            
            if not result or not result.get('measurements'):
                failed_extractions.append(str(pdf_file.name))
                continue
            
            # Generate interpretations using rule engine
            interpretations = predictor.predict(
                measurements=result['measurements'],
                patient_info=result.get('patient', {})
            )
            
            # Create training sample
            training_sample = {
                'file_name': result['file_name'],
                'patient': result['patient'],
                'measurements': result['measurements'],
                'interpretations': interpretations,
                'num_measurements': len(result['measurements'])
            }
            
            training_data.append(training_sample)
            
            # Save individual file
            json_filename = pdf_file.stem + '.json'
            json_path = output_dir / json_filename
            with open(json_path, 'w') as f:
                json.dump(training_sample, f, indent=2)
                
        except Exception as e:
            print(f"\n❌ Error processing {pdf_file.name}: {e}")
            failed_extractions.append(str(pdf_file.name))
    
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully processed: {len(training_data)} reports")
    print(f"✗ Failed extractions: {len(failed_extractions)} reports")
    
    if training_data:
        # Save consolidated dataset
        dataset_path = output_dir / dataset_filename
        with open(dataset_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\n✓ Training dataset saved: {dataset_path}")
        
        # Statistics
        total_measurements = sum(sample['num_measurements'] for sample in training_data)
        avg_measurements = total_measurements / len(training_data)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(training_data)}")
        print(f"  Average measurements per report: {avg_measurements:.1f}")
        
        # Parameter frequency
        param_counts = {}
        for sample in training_data:
            for param in sample['measurements'].keys():
                param_counts[param] = param_counts.get(param, 0) + 1
        
        print(f"\nMost Common Parameters:")
        for param, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(training_data)) * 100
            print(f"  {param:20s}: {count:3d} ({percentage:5.1f}%)")
    
    if failed_extractions:
        print(f"\n⚠ Failed extractions ({len(failed_extractions)}):")
        for failed in failed_extractions[:5]:
            print(f"  - {failed}")
        if len(failed_extractions) > 5:
            print(f"  ... and {len(failed_extractions) - 5} more")
    
    print("\n" + "=" * 80)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 80)
    
    return training_data

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training dataset from PDF reports")
    parser.add_argument(
        "--input-dir",
        default="data/sample_reports",
        help="Folder containing PDF reports"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Folder to write processed JSON files"
    )
    parser.add_argument(
        "--dataset-filename",
        default="training_dataset.json",
        help="Filename for consolidated dataset JSON"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    prepare_training_data(args.input_dir, args.output_dir, args.dataset_filename)

#!/usr/bin/env python3
"""
Train a new v2_expanded model using original + new datasets.
This preserves the original models and saves v2 models with a suffix.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict
import joblib
from train_interpretation_model import InterpretationModelTrainer


def load_samples_from_dir(json_dir: str) -> List[Dict]:
    """Load all training samples from a processed JSON directory."""
    samples: List[Dict] = []
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
        if filename in {'all_reports.json', 'training_dataset.json'} or filename.startswith('training_dataset'):
            continue
        filepath = os.path.join(json_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'interpretations' in data:
                    samples.append(data)
        except Exception:
            continue
    return samples


def save_combined_dataset(samples: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)


def save_models_with_suffix(trainer: InterpretationModelTrainer, output_dir: str, suffix: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save scaler
    scaler_path = output_path / f'scaler_{suffix}.pkl'
    joblib.dump(trainer.scaler, scaler_path)

    # Save each model
    for category, model in trainer.models.items():
        model_path = output_path / f'model_{category}_{suffix}.pkl'
        joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        'feature_names': trainer.feature_names,
        'categories': trainer.categories,
        'key_parameters': trainer.key_parameters
    }
    metadata_path = output_path / f'model_metadata_{suffix}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train v2 model with combined datasets")
    parser.add_argument("--original-processed-dir", default="data/processed")
    parser.add_argument("--new-processed-dir", default="data/processed_new")
    parser.add_argument("--combined-dataset", default="data/processed/combined_training_dataset.json")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--suffix", default="v2_expanded")
    return parser.parse_args()


def main():
    args = parse_args()

    original_samples = load_samples_from_dir(args.original_processed_dir)
    new_samples = load_samples_from_dir(args.new_processed_dir)

    combined_samples = original_samples + new_samples

    if not combined_samples:
        print("❌ No samples found to train on.")
        return

    print(f"✓ Original samples: {len(original_samples)}")
    print(f"✓ New samples: {len(new_samples)}")
    print(f"✓ Combined samples: {len(combined_samples)}")

    save_combined_dataset(combined_samples, args.combined_dataset)
    print(f"✓ Combined dataset saved: {args.combined_dataset}")

    trainer = InterpretationModelTrainer()
    trainer.train_models(args.combined_dataset, test_size=0.2)

    save_models_with_suffix(trainer, args.output_dir, args.suffix)
    print(f"✓ v2 models saved to {args.output_dir} with suffix '{args.suffix}'")


if __name__ == "__main__":
    main()

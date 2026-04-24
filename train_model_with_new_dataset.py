#!/usr/bin/env python3
"""
Train a new v2_expanded model using original + new datasets.
This preserves the original models and saves v2 models with a suffix.
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
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


def lv_size_grade_from_measurement(lvid_d: float, sex: str) -> str:
    """Sex-adjusted LV size grade from LVID_D.

    Male: <=5.9 normal, <=6.3 mild, <=6.8 moderate, >6.8 severe
    Female: <=5.3 normal, <=5.7 mild, <=6.1 moderate, >6.1 severe
    """
    if lvid_d is None or lvid_d <= 0:
        return 'Unknown'

    is_male = str(sex).upper() == 'M'
    if is_male:
        if lvid_d <= 5.9:
            return 'Normal'
        if lvid_d <= 6.3:
            return 'Mild'
        if lvid_d <= 6.8:
            return 'Moderate'
        return 'Severe'

    if lvid_d <= 5.3:
        return 'Normal'
    if lvid_d <= 5.7:
        return 'Mild'
    if lvid_d <= 6.1:
        return 'Moderate'
    return 'Severe'


def _lv_size_range_for_class(target_class: str, sex: str) -> Tuple[float, float]:
    """Return (low, high) LVID_D range for a target class and sex."""
    is_male = str(sex).upper() == 'M'
    if is_male:
        ranges = {
            'Normal': (4.2, 5.9),
            'Mild': (5.91, 6.3),
            'Moderate': (6.31, 6.8),
            'Severe': (6.81, 7.5),
        }
    else:
        ranges = {
            'Normal': (3.8, 5.3),
            'Mild': (5.31, 5.7),
            'Moderate': (5.71, 6.1),
            'Severe': (6.11, 6.9),
        }
    return ranges.get(target_class, (0.0, 0.0))


def _jitter_numeric(value: float, rel_noise: float) -> float:
    """Apply multiplicative jitter while keeping sign and basic scale."""
    if value is None:
        return value
    jitter = 1.0 + random.uniform(-rel_noise, rel_noise)
    out = value * jitter
    # Keep strictly positive echo measures positive.
    return max(out, 1e-4)


def augment_lv_size_classes(
    samples: List[Dict],
    target_per_class: int,
    rel_noise: float,
    seed: int,
) -> List[Dict]:
    """Generate synthetic samples to up-balance LV_SIZE classes.

    This uses bootstrap + small perturbations and enforces LVID_D into the
    requested severity band for each synthetic sample.
    """
    random.seed(seed)

    by_class: Dict[str, List[Dict]] = {'Normal': [], 'Mild': [], 'Moderate': [], 'Severe': []}
    for s in samples:
        m = s.get('measurements', {})
        p = s.get('patient', {})
        cls = lv_size_grade_from_measurement(m.get('LVID_D', 0), p.get('sex', 'M'))
        if cls in by_class:
            by_class[cls].append(s)

    print("LV_SIZE class counts before augmentation:")
    for cls in ['Normal', 'Mild', 'Moderate', 'Severe']:
        print(f"  {cls:<9s}: {len(by_class[cls])}")

    synthetic: List[Dict] = []
    for cls in ['Mild', 'Moderate', 'Severe']:
        current = len(by_class[cls])
        needed = max(0, target_per_class - current)
        if needed == 0:
            continue
        if current == 0:
            print(f"⚠ Cannot synthesize class {cls}: no seed samples available")
            continue

        for i in range(needed):
            base = random.choice(by_class[cls])
            syn = json.loads(json.dumps(base))

            # Keep record shape consistent with original data format.
            # Reuse the source filename without adding synthetic markers.
            syn['file_name'] = base.get('file_name', syn.get('file_name', 'unknown.pdf'))

            patient = syn.get('patient', {})
            sex = patient.get('sex', 'M')
            measures = syn.get('measurements', {})

            # Small realistic perturbation on all numeric measurements.
            for k, v in list(measures.items()):
                if isinstance(v, (int, float)) and v > 0:
                    measures[k] = round(_jitter_numeric(float(v), rel_noise), 3)

            # Force LVID_D into target class range.
            lo, hi = _lv_size_range_for_class(cls, sex)
            if hi > lo:
                measures['LVID_D'] = round(random.uniform(lo, hi), 3)

            # Keep LVID_S plausibly linked to LVID_D when available.
            if measures.get('LVID_S', 0) > 0 and measures.get('LVID_D', 0) > 0:
                ratio = measures['LVID_S'] / max(measures['LVID_D'], 1e-4)
                ratio = min(max(ratio, 0.45), 0.9)
                measures['LVID_S'] = round(measures['LVID_D'] * ratio, 3)

            syn['num_measurements'] = len(measures)
            synthetic.append(syn)

    augmented = samples + synthetic
    print(f"✓ Synthetic samples added: {len(synthetic)}")

    # Show post-augmentation distribution.
    post_counts = {'Normal': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
    for s in augmented:
        m = s.get('measurements', {})
        p = s.get('patient', {})
        cls = lv_size_grade_from_measurement(m.get('LVID_D', 0), p.get('sex', 'M'))
        if cls in post_counts:
            post_counts[cls] += 1

    print("LV_SIZE class counts after augmentation:")
    for cls in ['Normal', 'Mild', 'Moderate', 'Severe']:
        print(f"  {cls:<9s}: {post_counts[cls]}")

    return augmented


def save_combined_dataset(samples: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)


def save_models_with_suffix(trainer: InterpretationModelTrainer, output_dir: str, suffix: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Remove stale model files for categories skipped this run.
    for category in trainer.categories:
        stale_model_path = output_path / f'model_{category}_{suffix}.pkl'
        if category not in trainer.models and stale_model_path.exists():
            stale_model_path.unlink()
            print(f"✓ Removed stale model for {category} at {stale_model_path}")

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
    parser.add_argument(
        "--augment-lv-size",
        action="store_true",
        help="Generate synthetic samples to up-balance LV_SIZE minority classes"
    )
    parser.add_argument(
        "--lv-size-target-per-class",
        type=int,
        default=80,
        help="Target sample count per LV_SIZE minority class (Mild/Moderate/Severe)"
    )
    parser.add_argument(
        "--augmentation-noise",
        type=float,
        default=0.04,
        help="Relative jitter for synthetic numeric measurements"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation"
    )
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

    if args.augment_lv_size:
        combined_samples = augment_lv_size_classes(
            combined_samples,
            target_per_class=args.lv_size_target_per_class,
            rel_noise=args.augmentation_noise,
            seed=args.seed,
        )
        print(f"✓ Augmented combined samples: {len(combined_samples)}")

    save_combined_dataset(combined_samples, args.combined_dataset)
    print(f"✓ Combined dataset saved: {args.combined_dataset}")

    trainer = InterpretationModelTrainer()
    trainer.train_models(args.combined_dataset, test_size=0.2)

    save_models_with_suffix(trainer, args.output_dir, args.suffix)
    print(f"✓ v2 models saved to {args.output_dir} with suffix '{args.suffix}'")


if __name__ == "__main__":
    main()

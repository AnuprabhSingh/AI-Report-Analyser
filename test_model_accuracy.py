#!/usr/bin/env python3
"""
Test and Evaluate ML Model Accuracy

This script:
1. Loads the training dataset
2. Splits it into train/test sets (same as training)
3. Evaluates model accuracy on test set
4. Shows detailed metrics: accuracy, precision, recall, F1-score
5. Shows confusion matrices
6. Compares predictions with actual labels
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import joblib
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class ModelAccuracyTester:
    """Test accuracy of trained ML models."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.categories = []
        
        # Load models and metadata
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scaler."""
        print("Loading trained models...")
        
        # Load metadata
        with open(self.models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.categories = metadata['categories']
        
        # Load scaler
        self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
        
        # Load each model
        for category in self.categories:
            model_path = self.models_dir / f'model_{category}.pkl'
            if model_path.exists():
                self.models[category] = joblib.load(model_path)
                print(f"  ✓ Loaded {category} model")
        
        print(f"✓ Loaded {len(self.models)} models\n")
    
    def load_dataset(self, dataset_path: str):
        """Load and prepare dataset."""
        print(f"Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} samples\n")
        
        # Convert to dataframe
        samples = []
        key_parameters = [
            'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'LVPW_D',
            'LA_DIMENSION', 'AORTIC_ROOT', 'MV_E_A', 'LV_MASS'
        ]
        
        for item in data:
            measurements = item['measurements']
            patient = item.get('patient', {})
            interpretations = item['interpretations']
            
            sample = {
                'age': patient.get('age', 0),
                'sex': 1 if patient.get('sex') == 'M' else 0,
            }
            
            # Add measurements
            for param in key_parameters:
                sample[param] = measurements.get(param, 0)
            
            # Extract labels
            sample['labels'] = self._extract_labels(interpretations)
            
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def _extract_labels(self, interpretations: Dict[str, str]) -> Dict[str, str]:
        """Extract classification labels from interpretation text."""
        labels = {}
        
        # Parse LV Function
        lv_function_text = interpretations.get('Left Ventricular Function', '')
        if 'Normal' in lv_function_text or 'normal' in lv_function_text:
            labels['LV_FUNCTION'] = 'Normal'
        elif 'Mild' in lv_function_text:
            labels['LV_FUNCTION'] = 'Mild'
        elif 'Moderate' in lv_function_text:
            labels['LV_FUNCTION'] = 'Moderate'
        elif 'Severe' in lv_function_text:
            labels['LV_FUNCTION'] = 'Severe'
        else:
            labels['LV_FUNCTION'] = 'Unknown'
        
        # Parse LV Size
        lv_size_text = interpretations.get('LV Diastolic Dimension', '')
        if 'Normal' in lv_size_text:
            labels['LV_SIZE'] = 'Normal'
        elif 'Dilated' in lv_size_text or 'enlargement' in lv_size_text:
            labels['LV_SIZE'] = 'Dilated'
        else:
            labels['LV_SIZE'] = 'Unknown'
        
        # Parse LV Hypertrophy
        ivs_text = interpretations.get('Interventricular Septum', '')
        if 'Normal' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'None'
        elif 'Mild' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Mild'
        elif 'Moderate' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Moderate'
        elif 'Severe' in ivs_text:
            labels['LV_HYPERTROPHY'] = 'Severe'
        else:
            labels['LV_HYPERTROPHY'] = 'Unknown'
        
        # Parse LA Size
        la_text = interpretations.get('Left Atrium', '')
        if 'Normal' in la_text:
            labels['LA_SIZE'] = 'Normal'
        elif 'enlarge' in la_text.lower():
            labels['LA_SIZE'] = 'Enlarged'
        else:
            labels['LA_SIZE'] = 'Unknown'
        
        # Parse Diastolic Function
        diastolic_text = interpretations.get('Diastolic Function', '')
        if 'Normal' in diastolic_text:
            labels['DIASTOLIC_FUNCTION'] = 'Normal'
        else:
            labels['DIASTOLIC_FUNCTION'] = 'Abnormal'
        
        return labels
    
    def evaluate_models(self, dataset_path: str, test_size: float = 0.2):
        """Evaluate all models on test set."""
        
        print("=" * 80)
        print("MODEL ACCURACY EVALUATION")
        print("=" * 80)
        print()
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Prepare features
        feature_cols = self.feature_names
        X = df[feature_cols].copy()
        
        # Handle missing values
        for col in feature_cols:
            if col not in ['age', 'sex']:
                median_val = X[X[col] > 0][col].median()
                X[col] = X[col].replace(0, median_val)
        
        X = X.fillna(0)
        
        # Extract labels
        y_dict = {}
        for category in self.categories:
            y_dict[category] = df['labels'].apply(lambda x: x.get(category, 'Unknown'))
        
        # Split data (same as training)
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, X.index, test_size=test_size, random_state=42
        )
        
        print(f"Dataset Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print()
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate each model
        overall_results = {}
        
        for category in self.categories:
            if category not in self.models:
                print(f"⚠ Model not found for {category}, skipping...")
                continue
            
            print("=" * 80)
            print(f"CATEGORY: {category}")
            print("=" * 80)
            
            model = self.models[category]
            y_test = y_dict[category].iloc[indices_test]
            
            # Remove unknown labels
            test_mask = y_test != 'Unknown'
            X_test_cat = X_test_scaled[test_mask]
            y_test_cat = y_test[test_mask]
            
            if len(y_test_cat) == 0:
                print("⚠ No test samples available\n")
                continue
            
            print(f"Test samples: {len(y_test_cat)}")
            print(f"Classes: {sorted(y_test_cat.unique())}")
            print()
            
            # Make predictions
            y_pred = model.predict(X_test_cat)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_cat, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test_cat, y_pred, average='weighted', zero_division=0
            )
            
            print(f"📊 OVERALL METRICS:")
            print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-Score:  {f1:.3f}")
            print()
            
            # Detailed classification report
            print("📋 DETAILED CLASSIFICATION REPORT:")
            print(classification_report(y_test_cat, y_pred, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y_test_cat, y_pred)
            classes = sorted(y_test_cat.unique())
            
            print("📊 CONFUSION MATRIX:")
            print("    Predicted →")
            print("Actual ↓")
            
            # Header
            header = "         " + "  ".join(f"{cls:>8s}" for cls in classes)
            print(header)
            
            # Matrix rows
            for i, actual_class in enumerate(classes):
                row = f"{actual_class:>8s} "
                row += "  ".join(f"{cm[i,j]:>8d}" for j in range(len(classes)))
                print(row)
            
            print()
            
            # Show some example predictions
            print("🔍 SAMPLE PREDICTIONS (first 10):")
            for i in range(min(10, len(y_test_cat))):
                actual = y_test_cat.iloc[i]
                predicted = y_pred[i]
                status = "✓" if actual == predicted else "✗"
                print(f"  {status} Actual: {actual:15s} | Predicted: {predicted:15s}")
            
            print()
            
            # Store results
            overall_results[category] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_samples': len(y_test_cat),
                'classes': classes
            }
        
        # Summary
        print("=" * 80)
        print("📊 OVERALL SUMMARY")
        print("=" * 80)
        print()
        
        # Category-wise breakdown
        print("CATEGORY-WISE BREAKDOWN:")
        print("-" * 80)
        print(f"{'Category':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Samples':>8s}")
        print("-" * 80)
        
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        total_test_samples = 0
        
        for category, results in overall_results.items():
            print(f"{category:<25s} {results['accuracy']:>10.3f} {results['precision']:>10.3f} "
                  f"{results['recall']:>10.3f} {results['f1_score']:>10.3f} {results['test_samples']:>8d}")
            
            all_accuracies.append(results['accuracy'])
            all_precisions.append(results['precision'])
            all_recalls.append(results['recall'])
            all_f1_scores.append(results['f1_score'])
            total_test_samples += results['test_samples']
        
        print("-" * 80)
        
        # Calculate overall metrics
        avg_accuracy = np.mean(all_accuracies)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1_score = np.mean(all_f1_scores)
        
        # Standard deviations
        std_accuracy = np.std(all_accuracies)
        std_precision = np.std(all_precisions)
        std_recall = np.std(all_recalls)
        std_f1_score = np.std(all_f1_scores)
        
        # Min/Max
        min_accuracy = np.min(all_accuracies)
        max_accuracy = np.max(all_accuracies)
        
        print()
        print("OVERALL STATISTICS:")
        print("-" * 80)
        print(f"{'Metric':<30s} {'Average':>12s} {'Min':>12s} {'Max':>12s} {'Std Dev':>12s}")
        print("-" * 80)
        print(f"{'Accuracy':<30s} {avg_accuracy:>12.3f} {min_accuracy:>12.3f} {max_accuracy:>12.3f} {std_accuracy:>12.3f}")
        print(f"{'Precision':<30s} {avg_precision:>12.3f} {np.min(all_precisions):>12.3f} {np.max(all_precisions):>12.3f} {std_precision:>12.3f}")
        print(f"{'Recall':<30s} {avg_recall:>12.3f} {np.min(all_recalls):>12.3f} {np.max(all_recalls):>12.3f} {std_recall:>12.3f}")
        print(f"{'F1-Score':<30s} {avg_f1_score:>12.3f} {np.min(all_f1_scores):>12.3f} {np.max(all_f1_scores):>12.3f} {std_f1_score:>12.3f}")
        print("-" * 80)
        
        print()
        print("OVERALL PERFORMANCE:")
        print("-" * 80)
        print(f"  📊 Average Accuracy:    {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        print(f"  📊 Average Precision:   {avg_precision:.3f}")
        print(f"  📊 Average Recall:      {avg_recall:.3f}")
        print(f"  📊 Average F1-Score:    {avg_f1_score:.3f}")
        print(f"  📈 Accuracy Range:      {min_accuracy:.3f} - {max_accuracy:.3f}")
        print(f"  📊 Total Test Samples:  {total_test_samples}")
        print(f"  📊 Number of Categories: {len(overall_results)}")
        print("-" * 80)
        
        # Performance rating
        print()
        print("PERFORMANCE RATING:")
        print("-" * 80)
        if avg_accuracy >= 0.95:
            rating = "🌟 Excellent (≥95%)"
        elif avg_accuracy >= 0.90:
            rating = "⭐ Very Good (90-95%)"
        elif avg_accuracy >= 0.85:
            rating = "👍 Good (85-90%)"
        elif avg_accuracy >= 0.80:
            rating = "✓ Acceptable (80-85%)"
        else:
            rating = "⚠️ Needs Improvement (<80%)"
        
        print(f"  {rating}")
        print("-" * 80)
        print()
        
        return overall_results
    
    def compare_predictions(self, dataset_path: str, num_samples: int = 5):
        """Compare model predictions with actual interpretations."""
        
        print("=" * 80)
        print("PREDICTION COMPARISON")
        print("=" * 80)
        print()
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Select random samples
        import random
        samples = random.sample(data, min(num_samples, len(data)))
        
        for idx, item in enumerate(samples, 1):
            print(f"\n{'=' * 80}")
            print(f"SAMPLE {idx}: {item['file_name']}")
            print('=' * 80)
            
            measurements = item['measurements']
            patient = item['patient']
            actual_interpretations = item['interpretations']
            
            print(f"\nPatient: Age {patient.get('age', 'N/A')}, Sex {patient.get('sex', 'N/A')}")
            
            # Prepare features
            features = {
                'age': patient.get('age', 0),
                'sex': 1 if patient.get('sex') == 'M' else 0
            }
            
            key_parameters = [
                'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'LVPW_D',
                'LA_DIMENSION', 'AORTIC_ROOT', 'MV_E_A', 'LV_MASS'
            ]
            
            for param in key_parameters:
                features[param] = measurements.get(param, 0)
            
            # Make predictions
            X = np.array([[features[feat] for feat in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            print("\n🔮 ML PREDICTIONS:")
            predictions = {}
            for category, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                predictions[category] = pred
                print(f"  {category:25s}: {pred}")
            
            print("\n📋 ACTUAL LABELS (from rule-based):")
            actual_labels = self._extract_labels(actual_interpretations)
            for category, label in actual_labels.items():
                if category in predictions:
                    match = "✓" if predictions[category] == label else "✗"
                    print(f"  {match} {category:25s}: {label}")
            
            print()


def main():
    """Main testing pipeline."""
    
    # Check if models exist
    if not Path('models/scaler.pkl').exists():
        print("❌ Models not found!")
        print("Please train models first: python train_interpretation_model.py")
        return
    
    # Initialize tester
    tester = ModelAccuracyTester('models')
    
    # Dataset path
    dataset_path = 'data/processed/training_dataset.json'
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Evaluate models
    print("\n")
    results = tester.evaluate_models(dataset_path, test_size=0.2)
    
    # Compare predictions
    print("\n")
    tester.compare_predictions(dataset_path, num_samples=5)
    
    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

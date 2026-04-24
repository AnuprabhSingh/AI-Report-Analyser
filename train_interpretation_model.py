#!/usr/bin/env python3
"""
Train ML model to generate clinical interpretations from measurements.
Uses extracted datapoints to predict interpretation comments.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Any, Dict, List, Tuple


class LVSizeOneClassClassifier:
    """One-class LV size detector used when only Normal class is available.

    The model is trained on scaled features from the Normal-only training data and
    predicts Dilated when either:
    - scaled LVID_D exceeds the clinical threshold mapped into scaled space, or
    - LVID_D is a strong outlier relative to the observed Normal distribution.
    """

    def __init__(self, lvid_d_index: int, scaled_dilation_threshold: float, z_threshold: float = 3.0):
        self.lvid_d_index = lvid_d_index
        self.scaled_dilation_threshold = scaled_dilation_threshold
        self.z_threshold = z_threshold
        self.normal_mean_ = 0.0
        self.normal_std_ = 1.0
        self.feature_importances_ = None

    def fit(self, X, y=None):
        lvid = X[:, self.lvid_d_index]
        self.normal_mean_ = float(np.mean(lvid))
        std = float(np.std(lvid))
        self.normal_std_ = std if std > 1e-8 else 1.0

        # Keep compatibility with existing feature-importance plotting.
        importances = np.zeros(X.shape[1], dtype=float)
        importances[self.lvid_d_index] = 1.0
        self.feature_importances_ = importances
        return self

    def predict(self, X):
        lvid = X[:, self.lvid_d_index]
        z = np.abs((lvid - self.normal_mean_) / self.normal_std_)

        is_dilated = (lvid > self.scaled_dilation_threshold) | (z > self.z_threshold)
        return np.where(is_dilated, 'Dilated', 'Normal')

class InterpretationModelTrainer:
    """Train models to predict clinical interpretations from measurements."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
        # Define key parameters we'll use as features
        self.key_parameters = [
            'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'LVPW_D', 
            'LA_DIMENSION', 'AORTIC_ROOT', 'MV_E_A', 'LV_MASS'
        ]
        
        # Categories to predict
        self.categories = [
            'LV_FUNCTION',       # Normal, Mild, Moderate, Severe dysfunction
            'LV_SIZE',           # Normal, Mild, Moderate, Severe dilatation
            'LV_HYPERTROPHY',    # None, Mild, Moderate, Severe
            'LA_SIZE',           # Normal, Enlarged
            'DIASTOLIC_FUNCTION' # Normal, Abnormal
        ]
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and prepare dataset from JSON."""
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples")
        
        # Convert to feature matrix and labels
        samples = []
        for item in data:
            measurements = item['measurements']
            patient = item.get('patient', {})
            interpretations = item['interpretations']
            
            sample = {
                'file_name': item['file_name'],
                'age': patient.get('age', 0),
                'sex': 1 if patient.get('sex') == 'M' else 0,  # 1=Male, 0=Female
            }
            
            # Add measurements (use 0 if missing)
            for param in self.key_parameters:
                sample[param] = measurements.get(param, 0)
            
            # Extract labels from interpretations (fallback to measurements)
            sample['labels'] = self._extract_labels(interpretations, measurements, patient)
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        print(f"Created dataframe with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
    def _extract_labels(
        self,
        interpretations: Dict[str, str],
        measurements: Dict[str, float],
        patient_info: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """Extract classification labels from interpretation text with measurement fallback."""
        labels = {}
        if patient_info is None:
            patient_info = {}
        
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
        # Fallback to EF/FS if LV_FUNCTION unknown
        if labels['LV_FUNCTION'] == 'Unknown':
            ef = measurements.get('EF', 0) or 0
            fs = measurements.get('FS', 0) or 0
            if ef > 0:
                if ef >= 55:
                    labels['LV_FUNCTION'] = 'Normal'
                elif ef >= 45:
                    labels['LV_FUNCTION'] = 'Mild'
                elif ef >= 30:
                    labels['LV_FUNCTION'] = 'Moderate'
                else:
                    labels['LV_FUNCTION'] = 'Severe'
            elif fs > 0:
                if fs >= 28:
                    labels['LV_FUNCTION'] = 'Normal'
                elif fs >= 22:
                    labels['LV_FUNCTION'] = 'Mild'
                elif fs >= 15:
                    labels['LV_FUNCTION'] = 'Moderate'
                else:
                    labels['LV_FUNCTION'] = 'Severe'
        
        # LV Size grading: measurement-first to ensure consistent multiclass labels.
        # We avoid text-first labeling here because report text is often compressed
        # to "Normal" even when measurements support graded dilatation.
        # Male: <=5.9 normal, <=6.3 mild, <=6.8 moderate, >6.8 severe
        # Female: <=5.3 normal, <=5.7 mild, <=6.1 moderate, >6.1 severe
        labels['LV_SIZE'] = 'Unknown'
        lvid_d = measurements.get('LVID_D', 0) or 0
        sex = str(patient_info.get('sex', 'M')).upper()
        if lvid_d > 0:
            if sex == 'M':
                if lvid_d <= 5.9:
                    labels['LV_SIZE'] = 'Normal'
                elif lvid_d <= 6.3:
                    labels['LV_SIZE'] = 'Mild'
                elif lvid_d <= 6.8:
                    labels['LV_SIZE'] = 'Moderate'
                else:
                    labels['LV_SIZE'] = 'Severe'
            else:
                if lvid_d <= 5.3:
                    labels['LV_SIZE'] = 'Normal'
                elif lvid_d <= 5.7:
                    labels['LV_SIZE'] = 'Mild'
                elif lvid_d <= 6.1:
                    labels['LV_SIZE'] = 'Moderate'
                else:
                    labels['LV_SIZE'] = 'Severe'
        
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
        # Fallback to IVS_D if LV_HYPERTROPHY unknown
        if labels['LV_HYPERTROPHY'] == 'Unknown':
            ivs_d = measurements.get('IVS_D', 0) or 0
            if ivs_d > 0:
                if ivs_d <= 1.0:
                    labels['LV_HYPERTROPHY'] = 'None'
                elif ivs_d <= 1.3:
                    labels['LV_HYPERTROPHY'] = 'Mild'
                elif ivs_d <= 1.6:
                    labels['LV_HYPERTROPHY'] = 'Moderate'
                else:
                    labels['LV_HYPERTROPHY'] = 'Severe'
        
        # Parse LA Size
        la_text = interpretations.get('Left Atrium', '')
        if 'Normal' in la_text:
            labels['LA_SIZE'] = 'Normal'
        elif 'enlarge' in la_text.lower():
            labels['LA_SIZE'] = 'Enlarged'
        else:
            labels['LA_SIZE'] = 'Unknown'
        # Fallback to LA_DIMENSION if LA_SIZE unknown
        if labels['LA_SIZE'] == 'Unknown':
            la_dim = measurements.get('LA_DIMENSION', 0) or 0
            if la_dim > 0:
                labels['LA_SIZE'] = 'Enlarged' if la_dim >= 4.0 else 'Normal'
        
        # Parse Diastolic Function
        diastolic_text = interpretations.get('Diastolic Function', '')
        if 'Normal' in diastolic_text:
            labels['DIASTOLIC_FUNCTION'] = 'Normal'
        else:
            labels['DIASTOLIC_FUNCTION'] = 'Abnormal'
        # Fallback to MV_E_A if diastolic label may be unreliable
        mv_ea = measurements.get('MV_E_A', 0) or 0
        if mv_ea > 0:
            if 0.8 <= mv_ea <= 2.0:
                labels['DIASTOLIC_FUNCTION'] = 'Normal'
            else:
                labels['DIASTOLIC_FUNCTION'] = 'Abnormal'
        
        return labels
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> Tuple:
        """Prepare feature matrix and label dictionaries."""
        
        # Feature columns (exclude file_name and labels)
        feature_cols = ['age', 'sex'] + self.key_parameters
        
        # Handle missing values (replace 0 with median)
        X = df[feature_cols].copy()
        for col in feature_cols:
            if col not in ['age', 'sex']:
                median_val = X[X[col] > 0][col].median()
                X[col] = X[col].replace(0, median_val)
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        # Extract labels for each category
        y_dict = {}
        for category in self.categories:
            y_dict[category] = df['labels'].apply(lambda x: x.get(category, 'Unknown'))
        
        return X, y_dict

    def _random_oversample(self, X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly oversample minority classes to match the majority class size."""
        y_np = y.to_numpy()
        classes, counts = np.unique(y_np, return_counts=True)
        max_count = int(np.max(counts))

        rng = np.random.default_rng(42)
        sampled_indices = []

        for cls, count in zip(classes, counts):
            cls_indices = np.where(y_np == cls)[0]
            sampled_indices.extend(cls_indices.tolist())
            if count < max_count:
                extra = rng.choice(cls_indices, size=max_count - count, replace=True)
                sampled_indices.extend(extra.tolist())

        sampled_indices = np.array(sampled_indices)
        rng.shuffle(sampled_indices)
        return X[sampled_indices], y_np[sampled_indices]
    
    def train_models(self, dataset_path: str, test_size: float = 0.2):
        """Train models for each interpretation category."""
        
        print("=" * 80)
        print("TRAINING INTERPRETATION MODELS")
        print("=" * 80)
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Prepare features and labels
        X, y_dict = self.prepare_features_and_labels(df)
        
        print(f"\nFeatures: {self.feature_names}")
        print(f"Categories to predict: {self.categories}")
        
        # Split data
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, X.index, test_size=test_size, random_state=42
        )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train a model for each category
        results = {}
        
        for category in self.categories:
            print(f"\n{'=' * 80}")
            print(f"Training model for: {category}")
            print('-' * 80)
            
            y_train = y_dict[category].iloc[indices_train]
            y_test = y_dict[category].iloc[indices_test]
            
            # Remove unknown labels
            train_mask = y_train != 'Unknown'
            test_mask = y_test != 'Unknown'
            
            X_train_cat = X_train_scaled[train_mask]
            y_train_cat = y_train[train_mask]
            X_test_cat = X_test_scaled[test_mask]
            y_test_cat = y_test[test_mask]
            
            print(f"Training samples (after filtering): {len(y_train_cat)}")
            print(f"Testing samples (after filtering): {len(y_test_cat)}")
            print(f"Classes: {sorted(y_train_cat.unique())}")
            
            if len(y_train_cat) < 10:
                print(f"⚠ Too few samples for {category}, skipping...")
                continue

            # A classifier needs at least 2 classes; skip one-class targets.
            n_train_classes = y_train_cat.nunique()
            if n_train_classes < 2:
                only_class = y_train_cat.iloc[0] if len(y_train_cat) > 0 else 'Unknown'

                # For LV_SIZE only, train a one-class detector so we still have a model
                # that can flag potentially dilated cases even without positive labels.
                if category == 'LV_SIZE' and only_class == 'Normal':
                    lvid_d_idx = self.feature_names.index('LVID_D')
                    # Convert the clinical threshold (5.9 cm) into scaled feature space.
                    lvid_d_mean = float(self.scaler.mean_[lvid_d_idx])
                    lvid_d_scale = float(self.scaler.scale_[lvid_d_idx])
                    scaled_threshold = (5.9 - lvid_d_mean) / lvid_d_scale if lvid_d_scale > 1e-8 else 5.9

                    model = LVSizeOneClassClassifier(
                        lvid_d_index=lvid_d_idx,
                        scaled_dilation_threshold=scaled_threshold,
                        z_threshold=3.0,
                    )
                    model.fit(X_train_cat, y_train_cat)

                    y_pred = model.predict(X_test_cat)
                    accuracy = accuracy_score(y_test_cat, y_pred)

                    print(
                        "⚠ Only one class observed for LV_SIZE (Normal). "
                        "Training one-class detector with clinical threshold support instead "
                        "of a supervised classifier."
                    )
                    print(f"✓ One-class LV_SIZE accuracy on current test split: {accuracy:.3f}")
                    print("  Note: This metric is limited because test labels are single-class.")

                    self.models[category] = model
                    feature_importance = sorted(
                        zip(self.feature_names, model.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    results[category] = {
                        'accuracy': accuracy,
                        'feature_importance': feature_importance,
                        'model_type': 'one_class'
                    }
                    continue

                print(
                    f"⚠ Only one training class for {category} ({only_class}), "
                    "skipping ML model and using rule-based interpretation for this category..."
                )
                continue

            # Handle skew by up-weighting minority classes.
            class_counts = y_train_cat.value_counts().to_dict()
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            class_weight = {
                cls: len(y_train_cat) / (n_train_classes * count)
                for cls, count in class_counts.items()
            }

            print(f"Class distribution: {class_counts}")
            print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

            # Oversample only in training to reduce class skew for rare labels.
            X_train_fit, y_train_fit = self._random_oversample(X_train_cat, y_train_cat)
            fit_classes, fit_counts = np.unique(y_train_fit, return_counts=True)
            fit_distribution = {str(c): int(n) for c, n in zip(fit_classes, fit_counts)}
            print(f"Balanced training distribution (oversampled): {fit_distribution}")
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_fit, y_train_fit)
            
            # Evaluate
            y_pred = model.predict(X_test_cat)
            accuracy = accuracy_score(y_test_cat, y_pred)
            
            print(f"\n✓ Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test_cat, y_pred, zero_division=0))
            
            # Feature importance
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("\nTop 5 Important Features:")
            for feat, imp in feature_importance[:5]:
                print(f"  {feat:15s}: {imp:.3f}")
            
            # Store model and results
            self.models[category] = model
            results[category] = {
                'accuracy': accuracy,
                'feature_importance': feature_importance
            }
        
        print("\n" + "=" * 80)
        print("✓ MODEL TRAINING COMPLETE")
        print("=" * 80)
        
        return results
    
    def save_models(self, output_dir: str = 'models'):
        """Save trained models and scaler."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Remove stale model files for categories that were skipped this run.
        for category in self.categories:
            model_path = output_path / f'model_{category}.pkl'
            if category not in self.models and model_path.exists():
                model_path.unlink()
                print(f"✓ Removed stale model for {category} at {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, output_path / 'scaler.pkl')
        print(f"✓ Saved scaler to {output_path / 'scaler.pkl'}")
        
        # Save each model
        for category, model in self.models.items():
            model_path = output_path / f'model_{category}.pkl'
            joblib.dump(model, model_path)
            print(f"✓ Saved model for {category} to {model_path}")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categories': self.categories,
            'key_parameters': self.key_parameters
        }
        
        with open(output_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {output_path / 'model_metadata.json'}")
        print(f"\n✓ All models saved to: {output_path}")
    
    def predict(self, measurements: Dict[str, float], patient_info: Dict) -> Dict[str, str]:
        """Generate predictions for new measurements."""
        
        # Prepare features
        features = {
            'age': patient_info.get('age', 0),
            'sex': 1 if patient_info.get('sex') == 'M' else 0
        }
        
        for param in self.key_parameters:
            features[param] = measurements.get(param, 0)
        
        # Convert to array
        X = np.array([[features[feat] for feat in self.feature_names]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict each category
        predictions = {}
        for category, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions[category] = pred
        
        return predictions


def main():
    """Main training pipeline."""
    
    # Initialize trainer
    trainer = InterpretationModelTrainer()
    
    # Train models
    dataset_path = 'data/processed/training_dataset.json'
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run 'python prepare_training_data.py' first")
        return
    
    # Train
    results = trainer.train_models(dataset_path, test_size=0.2)
    
    # Save models
    trainer.save_models('models')
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Use the trained models for predictions")
    print("  2. Test on new PDF reports")
    print("  3. Integrate with API for real-time interpretation")

if __name__ == '__main__':
    main()

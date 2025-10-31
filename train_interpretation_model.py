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
from typing import Dict, List, Tuple

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
            'LV_SIZE',           # Normal, Dilated
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
            
            # Extract labels from interpretations
            sample['labels'] = self._extract_labels(interpretations)
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        print(f"Created dataframe with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
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
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_cat, y_train_cat)
            
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

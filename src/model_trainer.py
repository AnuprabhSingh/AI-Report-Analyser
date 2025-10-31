"""
Machine Learning Model Trainer
Trains models for automated clinical interpretation generation.
Supports both classification and text generation approaches.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random


class ClinicalMLTrainer:
    """
    Trains ML models for clinical interpretation.
    Supports two approaches:
    1. Classification-based: Classify measurements into severity categories
    2. Regression-based: Predict continuous severity scores
    """
    
    def __init__(self, model_type: str = 'classification'):
        """
        Initialize trainer.
        
        Args:
            model_type: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
    
    def load_training_data(self, json_dir: str) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """
        Load training data from processed JSON files.
        
        Args:
            json_dir: Directory containing processed JSON files
            
        Returns:
            DataFrame with features and labels
        """
        all_data = []
        
        # Load all JSON files
        for filename in os.listdir(json_dir):
            if filename.endswith('.json') and filename != 'all_reports.json':
                filepath = os.path.join(json_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(all_data)
        
        # Generate labels using rule engine
        labels = self._generate_labels(df)
        
        return df, labels
    
    def _convert_to_dataframe(self, data_list: List[Dict]) -> pd.DataFrame:
        """Convert list of extracted data to feature DataFrame."""
        rows = []
        
        for record in data_list:
            row = {}
            
            # Add patient features
            patient = record.get('patient', {})
            row['age'] = patient.get('age', 50)
            row['sex_encoded'] = 1 if patient.get('sex', 'M') == 'M' else 0
            
            # Add measurements
            measurements = record.get('measurements', {})
            for key, value in measurements.items():
                row[key] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Fill missing values with median
        df = df.fillna(df.median())
        
        return df
    
    def _generate_labels(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Generate classification labels for each parameter.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary of parameter -> label list
        """
        labels = {}
        
        # EF labels
        if 'EF' in df.columns:
            ef_labels = []
            for ef in df['EF']:
                if ef >= 55:
                    ef_labels.append(0)  # Normal
                elif ef >= 45:
                    ef_labels.append(1)  # Mild dysfunction
                elif ef >= 30:
                    ef_labels.append(2)  # Moderate dysfunction
                else:
                    ef_labels.append(3)  # Severe dysfunction
            labels['EF'] = ef_labels
        
        # LVIDd labels (simplified, not sex-adjusted here)
        if 'LVID_D' in df.columns:
            lvid_labels = []
            for lvid in df['LVID_D']:
                if lvid <= 5.9:
                    lvid_labels.append(0)  # Normal
                elif lvid <= 6.3:
                    lvid_labels.append(1)  # Mild
                elif lvid <= 6.8:
                    lvid_labels.append(2)  # Moderate
                else:
                    lvid_labels.append(3)  # Severe
            labels['LVID_D'] = lvid_labels
        
        return labels
    
    def train_ef_classifier(self, df: pd.DataFrame, labels: List[int]) -> Dict[str, Any]:
        """
        Train classifier for EF interpretation.
        
        Args:
            df: Feature DataFrame
            labels: Classification labels
            
        Returns:
            Training metrics
        """
        # Select relevant features
        feature_cols = ['age', 'sex_encoded']
        if 'EF' in df.columns:
            feature_cols.append('EF')
        if 'FS' in df.columns:
            feature_cols.append('FS')
        if 'LVID_D' in df.columns:
            feature_cols.append('LVID_D')
        if 'LVID_S' in df.columns:
            feature_cols.append('LVID_S')
        
        # Filter columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].values
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save model
        self.models['EF_classifier'] = model
        self.feature_names = feature_cols
        
        print(f"EF Classifier - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and scaler.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler: {scaler_path}")
        
        # Save feature names
        config = {
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config: {config_path}")
    
    def create_synthetic_training_data(self, n_samples: int = 200) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """
        Create synthetic training data for demonstration.
        In production, this would be replaced with real annotated data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame and labels
        """
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Generate realistic cardiac parameters
            ef = np.random.normal(60, 10)
            ef = np.clip(ef, 20, 80)
            
            lvid_d = np.random.normal(4.8, 0.6)
            lvid_d = np.clip(lvid_d, 3.5, 7.0)
            
            lvid_s = lvid_d * np.random.uniform(0.55, 0.75)
            
            ivs_d = np.random.normal(0.9, 0.2)
            ivs_d = np.clip(ivs_d, 0.6, 1.6)
            
            lvpw_d = np.random.normal(0.9, 0.2)
            lvpw_d = np.clip(lvpw_d, 0.6, 1.5)
            
            la_dim = np.random.normal(3.8, 0.5)
            la_dim = np.clip(la_dim, 2.5, 5.5)
            
            mv_ea = np.random.normal(1.2, 0.4)
            mv_ea = np.clip(mv_ea, 0.5, 2.5)
            
            fs = (lvid_d - lvid_s) / lvid_d * 100
            
            age = np.random.randint(20, 80)
            sex = np.random.choice(['M', 'F'])
            
            record = {
                'EF': ef,
                'LVID_D': lvid_d,
                'LVID_S': lvid_s,
                'IVS_D': ivs_d,
                'LVPW_D': lvpw_d,
                'LA_DIMENSION': la_dim,
                'MV_E_A': mv_ea,
                'FS': fs,
                'age': age,
                'sex_encoded': 1 if sex == 'M' else 0
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        labels = self._generate_labels(df)
        
        return df, labels


def main():
    """Demo training pipeline."""
    print("=" * 60)
    print("Medical Report ML Model Training")
    print("=" * 60)
    
    trainer = ClinicalMLTrainer(model_type='classification')
    
    # Generate synthetic data (in production, use real data)
    print("\n1. Generating synthetic training data...")
    df, labels = trainer.create_synthetic_training_data(n_samples=200)
    print(f"   Generated {len(df)} samples")
    print(f"   Features: {list(df.columns)}")
    
    # Train models
    print("\n2. Training EF classifier...")
    if 'EF' in labels:
        metrics = trainer.train_ef_classifier(df, labels['EF'])
        print(f"   Training metrics: {metrics}")
    
    # Save models
    print("\n3. Saving models...")
    output_dir = '../models/'
    trainer.save_models(output_dir)
    
    # Load necessary libraries
    import os
    import random

    # Define the path to the dataset
    pdf_folder = os.path.join('..', 'DataSet')

    # Get all PDF files in the dataset folder
    all_pdfs = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # Shuffle the list of PDFs
    random.shuffle(all_pdfs)

    # Split the PDFs into training and testing sets
    num_test = int(len(all_pdfs) * 0.2)  # 20% for testing
    train_pdfs = all_pdfs[num_test:]
    test_pdfs = all_pdfs[:num_test]

    # Print the training and testing sets
    print('Training PDFs:', train_pdfs)
    print('Testing PDFs:', test_pdfs)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

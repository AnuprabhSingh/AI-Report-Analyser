#!/usr/bin/env python3
"""
Use trained ML models to generate interpretations from measurements.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict

class MLInterpretationPredictor:
    """Generate interpretations using trained ML models."""
    
    def __init__(self, model_dir: str = 'models'):
        """Load trained models."""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models and metadata."""
        
        # Load metadata
        metadata_path = self.model_dir / 'model_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        
        # Load each model
        for category in self.metadata['categories']:
            model_path = self.model_dir / f'model_{category}.pkl'
            if model_path.exists():
                self.models[category] = joblib.load(model_path)
        
        print(f"✓ Loaded {len(self.models)} models from {self.model_dir}")
    
    def predict(self, measurements: Dict[str, float], patient_info: Dict = None) -> Dict[str, str]:
        """Generate interpretations from measurements using ML models."""
        
        if patient_info is None:
            patient_info = {}
        
        # Prepare features
        features = {
            'age': patient_info.get('age', 50),  # Default age
            'sex': 1 if patient_info.get('sex') == 'M' else 0
        }
        
        for param in self.metadata['key_parameters']:
            features[param] = measurements.get(param, 0)
        
        # Convert to array in correct order
        X = np.array([[features[feat] for feat in self.metadata['feature_names']]])
        
        # Handle missing values (replace 0 with reasonable defaults)
        for i, feat in enumerate(self.metadata['feature_names']):
            if X[0, i] == 0 and feat not in ['age', 'sex']:
                # Use typical normal values
                defaults = {
                    'EF': 60, 'FS': 35, 'LVID_D': 4.5, 'LVID_S': 3.0,
                    'IVS_D': 0.9, 'LVPW_D': 0.9, 'LA_DIMENSION': 3.5,
                    'AORTIC_ROOT': 3.0, 'MV_E_A': 1.2, 'LV_MASS': 150
                }
                X[0, i] = defaults.get(feat, 0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict each category
        predictions = {}
        for category, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = max(proba)
            
            predictions[category] = {
                'prediction': pred,
                'confidence': float(confidence)
            }
        
        # Generate natural language interpretations
        interpretations = self._generate_text_interpretations(
            predictions, measurements, patient_info
        )
        
        return interpretations
    
    def _generate_text_interpretations(self, predictions: Dict, 
                                       measurements: Dict, 
                                       patient_info: Dict) -> Dict[str, str]:
        """Convert model predictions to natural language."""
        
        interpretations = {}
        
        # LV Function
        lv_func = predictions.get('LV_FUNCTION', {}).get('prediction', 'Unknown')
        ef_value = measurements.get('EF', 0)
        if ef_value > 0:
            if lv_func == 'Normal':
                interpretations['Left Ventricular Function'] = f"Normal LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Mild':
                interpretations['Left Ventricular Function'] = f"Mildly reduced LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Moderate':
                interpretations['Left Ventricular Function'] = f"Moderately reduced LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Severe':
                interpretations['Left Ventricular Function'] = f"Severely reduced LV systolic function (EF: {ef_value:.1f}%)"
        
        # LV Size
        lv_size = predictions.get('LV_SIZE', {}).get('prediction', 'Unknown')
        lvid_d = measurements.get('LVID_D', 0)
        if lvid_d > 0:
            if lv_size == 'Normal':
                interpretations['LV Diastolic Dimension'] = f"Normal LV size (LVIDd: {lvid_d:.2f} cm)"
            else:
                interpretations['LV Diastolic Dimension'] = f"LV dilatation (LVIDd: {lvid_d:.2f} cm)"
        
        # LV Hypertrophy
        lv_hyp = predictions.get('LV_HYPERTROPHY', {}).get('prediction', 'Unknown')
        ivs_d = measurements.get('IVS_D', 0)
        if ivs_d > 0:
            if lv_hyp == 'None':
                interpretations['Interventricular Septum'] = f"Normal septal thickness (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Mild':
                interpretations['Interventricular Septum'] = f"Mild septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Moderate':
                interpretations['Interventricular Septum'] = f"Moderate septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Severe':
                interpretations['Interventricular Septum'] = f"Severe septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
        
        # LA Size
        la_size = predictions.get('LA_SIZE', {}).get('prediction', 'Unknown')
        la_dim = measurements.get('LA_DIMENSION', 0)
        if la_dim > 0:
            if la_size == 'Normal':
                interpretations['Left Atrium'] = f"Normal LA size (LA: {la_dim:.2f} cm)"
            else:
                interpretations['Left Atrium'] = f"LA enlargement (LA: {la_dim:.2f} cm)"
        
        # Diastolic Function
        diastolic = predictions.get('DIASTOLIC_FUNCTION', {}).get('prediction', 'Unknown')
        mv_ea = measurements.get('MV_E_A', 0)
        if mv_ea > 0:
            if diastolic == 'Normal':
                interpretations['Diastolic Function'] = f"Normal diastolic function (E/A: {mv_ea:.2f})"
            else:
                interpretations['Diastolic Function'] = f"Diastolic dysfunction (E/A: {mv_ea:.2f})"
        
        # Overall Summary
        summary_parts = []
        if lv_func not in ['Normal', 'Unknown']:
            summary_parts.append(f"{lv_func.lower()} LV dysfunction")
        if lv_size == 'Dilated':
            summary_parts.append("LV dilatation")
        if lv_hyp not in ['None', 'Unknown']:
            summary_parts.append("LV hypertrophy")
        if la_size == 'Enlarged':
            summary_parts.append("LA enlargement")
        
        if summary_parts:
            interpretations['Overall Summary'] = f"Overall: Echocardiography shows {', '.join(summary_parts)}"
        else:
            interpretations['Overall Summary'] = "Overall: Echocardiographic parameters within normal limits"
        
        return interpretations


def test_ml_predictor():
    """Test the ML predictor on sample data."""
    
    print("=" * 80)
    print("TESTING ML INTERPRETATION PREDICTOR")
    print("=" * 80)
    
    # Initialize predictor
    try:
        predictor = MLInterpretationPredictor()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train models first:")
        print("  1. python prepare_training_data.py")
        print("  2. python train_interpretation_model.py")
        return
    
    # Test sample
    measurements = {
        'EF': 45,
        'LVID_D': 5.8,
        'LVID_S': 4.2,
        'IVS_D': 1.3,
        'LVPW_D': 1.1,
        'LA_DIMENSION': 4.5,
        'MV_E_A': 0.8
    }
    
    patient_info = {
        'age': 65,
        'sex': 'M',
        'name': 'Test Patient'
    }
    
    print("\nTest Measurements:")
    for param, value in measurements.items():
        print(f"  {param:15s}: {value}")
    
    print(f"\nPatient: {patient_info['name']}, {patient_info['age']}Y, {patient_info['sex']}")
    
    # Generate interpretation
    print("\n" + "-" * 80)
    interpretations = predictor.predict(measurements, patient_info)
    
    print("\nML-Generated Interpretations:")
    print("=" * 80)
    for category, text in interpretations.items():
        print(f"\n{category}:")
        print(f"  {text}")
    
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_ml_predictor()

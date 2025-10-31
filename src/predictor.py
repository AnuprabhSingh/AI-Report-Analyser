"""
Model Predictor
Loads trained models and generates clinical interpretations.
Intelligently combines ML models with rule-based engine.
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from .rule_engine import ClinicalRuleEngine


class ClinicalPredictor:
    """
    Loads ML models and generates predictions for medical interpretations.
    Falls back to rule-based engine if ML models are not available.
    """
    
    def __init__(self, model_dir: Optional[str] = None, use_ml: bool = True):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing trained models (default: 'models/')
            use_ml: Whether to use ML models (if False, uses rule-based only)
        """
        # Default model directory
        if model_dir is None:
            model_dir = 'models'
        
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.metadata = None
        self.ml_available = False
        self.use_ml_flag = use_ml
        self.last_used_ml = False  # Tracks whether the last predict() used ML
        self.last_sources: Dict[str, str] = {}  # Tracks per-category source for last prediction
        # Categories produced by ML text generator
        self.ML_TEXT_CATEGORIES = set([
            'Left Ventricular Function',
            'LV Diastolic Dimension',
            'Interventricular Septum',
            'Left Atrium',
            'Diastolic Function',
            'Overall Summary'
        ])
        
        # Always initialize rule engine as fallback
        self.rule_engine = ClinicalRuleEngine()
        
        # Try to load ML models if requested
        if use_ml:
            self.ml_available = self._load_ml_models()
            if self.ml_available:
                print("✓ ML models loaded successfully")
            else:
                print("ℹ ML models not available, using rule-based engine only")
    
    def _load_ml_models(self) -> bool:
        """
        Load trained ML models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            model_path = Path(self.model_dir)
            
            # Check if model directory exists
            if not model_path.exists():
                return False
            
            # Load metadata
            metadata_path = model_path / 'model_metadata.json'
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load scaler
            scaler_path = model_path / 'scaler.pkl'
            if not scaler_path.exists():
                return False
            
            self.scaler = joblib.load(scaler_path)
            
            # Load each model
            for category in self.metadata['categories']:
                model_file = model_path / f'model_{category}.pkl'
                if model_file.exists():
                    self.models[category] = joblib.load(model_file)
            
            # Check if at least one model was loaded
            return len(self.models) > 0
        
        except Exception as e:
            print(f"⚠ Error loading ML models: {e}")
            return False
    
    def predict(self, measurements: Dict[str, float], 
                patient_info: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate clinical interpretation for measurements.
        Uses ML models if available, falls back to rule-based engine.
        
        Args:
            measurements: Dictionary of parameter values
            patient_info: Patient demographic information
            
        Returns:
            Dictionary of interpretations
        """
        if patient_info is None:
            patient_info = {}
        
        # Always compute rule-based interpretations first
        rule_interpretations = self.rule_engine.interpret_measurements(measurements, patient_info)
        combined = dict(rule_interpretations)  # start with rules
        sources: Dict[str, str] = {k: 'Rule' for k in combined.keys()}

        # Optionally compute ML interpretations and overlay on top of rules
        ml_interpretations = None
        if self.ml_available and self.use_ml_flag:
            try:
                ml_interpretations = self._predict_with_ml(measurements, patient_info)
                # Overlay ML results onto combined output
                for k, v in ml_interpretations.items():
                    combined[k] = v
                    sources[k] = 'ML'
                self.last_used_ml = True
            except Exception as e:
                print(f"⚠ ML prediction failed: {e}, using rule-based results only")
                self.last_used_ml = False
        else:
            self.last_used_ml = False

        # Persist sources map for this prediction
        self.last_sources = sources
        return combined

    def get_sources_for(self, interpretations: Dict[str, str]) -> Dict[str, str]:
        """Return a map of category -> source ("ML" or "Rule").
        Uses last_used_ml flag and known ML categories to annotate.

        If last_used_ml is True, categories in ML_TEXT_CATEGORIES are marked as ML.
        Otherwise everything is marked as Rule. For any additional categories not in
        ML_TEXT_CATEGORIES, mark as Rule.
        """
        # Prefer the detailed map from the last prediction if available
        if getattr(self, 'last_sources', None):
            # Ensure we only return keys present in current interpretations
            return {k: self.last_sources.get(k, 'Rule') for k in interpretations.keys()}
        # Fallback heuristic based on category membership
        sources = {}
        for cat in interpretations.keys():
            sources[cat] = 'ML' if (self.last_used_ml and cat in self.ML_TEXT_CATEGORIES) else 'Rule'
        return sources
    
    def _predict_with_ml(self, measurements: Dict[str, float],
                         patient_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate interpretations using ML models.
        
        Args:
            measurements: Parameter values
            patient_info: Patient info
            
        Returns:
            Dictionary of interpretations
        """
        # Prepare features
        features = {
            'age': patient_info.get('age', 50),
            'sex': 1 if patient_info.get('sex') == 'M' else 0
        }
        
        for param in self.metadata['key_parameters']:
            features[param] = measurements.get(param, 0)
        
        # Handle missing values with reasonable defaults
        defaults = {
            'EF': 60, 'FS': 35, 'LVID_D': 4.5, 'LVID_S': 3.0,
            'IVS_D': 0.9, 'LVPW_D': 0.9, 'LA_DIMENSION': 3.5,
            'AORTIC_ROOT': 3.0, 'MV_E_A': 1.2, 'LV_MASS': 150
        }
        
        for param in self.metadata['key_parameters']:
            if features[param] == 0:
                features[param] = defaults.get(param, 0)
        
        # Convert to array
        X = np.array([[features[feat] for feat in self.metadata['feature_names']]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict each category
        predictions = {}
        for category, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions[category] = pred
        
        # Generate natural language interpretations
        interpretations = self._ml_to_text(predictions, measurements, patient_info)
        
        return interpretations
    
    def _ml_to_text(self, predictions: Dict, measurements: Dict, 
                    patient_info: Dict) -> Dict[str, str]:
        """Convert ML predictions to natural language text."""
        
        interpretations = {}
        
        # LV Function
        lv_func = predictions.get('LV_FUNCTION', 'Unknown')
        ef_value = measurements.get('EF', 0)
        if ef_value > 0:
            if lv_func == 'Normal':
                interpretations['Left Ventricular Function'] = \
                    f"Normal LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Mild':
                interpretations['Left Ventricular Function'] = \
                    f"Mildly reduced LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Moderate':
                interpretations['Left Ventricular Function'] = \
                    f"Moderately reduced LV systolic function (EF: {ef_value:.1f}%)"
            elif lv_func == 'Severe':
                interpretations['Left Ventricular Function'] = \
                    f"Severely reduced LV systolic function (EF: {ef_value:.1f}%)"
        
        # LV Size
        lv_size = predictions.get('LV_SIZE', 'Unknown')
        lvid_d = measurements.get('LVID_D', 0)
        if lvid_d > 0:
            if lv_size == 'Normal':
                interpretations['LV Diastolic Dimension'] = \
                    f"Normal LV size (LVIDd: {lvid_d:.2f} cm)"
            else:
                interpretations['LV Diastolic Dimension'] = \
                    f"LV dilatation (LVIDd: {lvid_d:.2f} cm)"
        
        # LV Hypertrophy
        lv_hyp = predictions.get('LV_HYPERTROPHY', 'Unknown')
        ivs_d = measurements.get('IVS_D', 0)
        if ivs_d > 0:
            if lv_hyp == 'None':
                interpretations['Interventricular Septum'] = \
                    f"Normal septal thickness (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Mild':
                interpretations['Interventricular Septum'] = \
                    f"Mild septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Moderate':
                interpretations['Interventricular Septum'] = \
                    f"Moderate septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
            elif lv_hyp == 'Severe':
                interpretations['Interventricular Septum'] = \
                    f"Severe septal hypertrophy (IVSd: {ivs_d:.2f} cm)"
        
        # LA Size
        la_size = predictions.get('LA_SIZE', 'Unknown')
        la_dim = measurements.get('LA_DIMENSION', 0)
        if la_dim > 0:
            if la_size == 'Normal':
                interpretations['Left Atrium'] = \
                    f"Normal LA size (LA: {la_dim:.2f} cm)"
            else:
                interpretations['Left Atrium'] = \
                    f"LA enlargement (LA: {la_dim:.2f} cm)"
        
        # Diastolic Function
        diastolic = predictions.get('DIASTOLIC_FUNCTION', 'Unknown')
        mv_ea = measurements.get('MV_E_A', 0)
        if mv_ea > 0:
            if diastolic == 'Normal':
                interpretations['Diastolic Function'] = \
                    f"Normal diastolic function (E/A: {mv_ea:.2f})"
            else:
                interpretations['Diastolic Function'] = \
                    f"Diastolic dysfunction (E/A: {mv_ea:.2f})"
        
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
            interpretations['Overall Summary'] = \
                f"Overall: Echocardiography shows {', '.join(summary_parts)}"
        else:
            interpretations['Overall Summary'] = \
                "Overall: Echocardiographic parameters within normal limits"
        
        return interpretations
    
    def predict_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Load data from JSON and generate interpretation.
        
        Args:
            json_path: Path to JSON file with extracted data
            
        Returns:
            Complete interpretation results
        """
        # Load data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        measurements = data.get('measurements', {})
        patient_info = data.get('patient', {})
        
        # Generate interpretations
        interpretations = self.predict(measurements, patient_info)
        sources = self.get_sources_for(interpretations)
        
        # Build result
        result = {
            'file_name': data.get('file_name', 'unknown'),
            'patient': patient_info,
            'measurements': measurements,
            'interpretations': interpretations,
            'sources': sources,
            'method': 'ML-Based' if self.last_used_ml else 'Rule-Based'
        }
        
        return result
    
    def batch_predict(self, json_dir: str, output_dir: str) -> None:
        """
        Process multiple JSON files and generate interpretations.
        
        Args:
            json_dir: Directory with JSON files
            output_dir: Directory to save interpretation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for filename in os.listdir(json_dir):
            if filename.endswith('.json') and filename != 'all_reports.json':
                json_path = os.path.join(json_dir, filename)
                
                try:
                    result = self.predict_from_json(json_path)
                    results.append(result)
                    
                    # Save individual result
                    output_filename = filename.replace('.json', '_interpreted.json')
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"✓ Processed: {filename}")
                
                except Exception as e:
                    print(f"✗ Error processing {filename}: {e}")
        
        # Save consolidated results
        if results:
            consolidated_path = os.path.join(output_dir, 'all_interpretations.json')
            with open(consolidated_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved consolidated results: {consolidated_path}")
        
        return results


def main():
    """Demo usage of predictor with ML models."""
    print("=" * 80)
    print("CLINICAL INTERPRETATION PREDICTOR (ML-Enhanced)")
    print("=" * 80)
    
    # Initialize predictor (tries to load ML models)
    predictor = ClinicalPredictor(use_ml=True)
    
    print(f"\nMethod: {'ML-Based' if predictor.ml_available else 'Rule-Based'}")
    
    # Example prediction
    sample_measurements = {
        'EF': 64.8,
        'LVID_D': 4.65,
        'LVID_S': 2.89,
        'IVS_D': 0.89,
        'LVPW_D': 0.98,
        'LA_DIMENSION': 3.47,
        'MV_E_A': 1.75,
        'FS': 38,
        'LV_MASS': 150
    }
    
    sample_patient = {
        'age': 45,
        'sex': 'F',
        'name': 'Sample Patient'
    }
    
    print("\nPatient Information:")
    print(f"  Name: {sample_patient['name']}")
    print(f"  Age: {sample_patient['age']} years")
    print(f"  Sex: {sample_patient['sex']}")
    
    print("\nMeasurements:")
    for param, value in sample_measurements.items():
        print(f"  {param:20s}: {value:>8.2f}")
    
    print("\nGenerating interpretation...")
    interpretations = predictor.predict(sample_measurements, sample_patient)
    
    print("\n" + "=" * 80)
    print("CLINICAL INTERPRETATION")
    print("=" * 80)
    for category, text in interpretations.items():
        print(f"\n{category}:")
        print(f"  {text}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

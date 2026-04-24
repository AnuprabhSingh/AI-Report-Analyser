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
from typing import Dict, Any, Optional, Tuple
from .rule_engine import ClinicalRuleEngine


class ClinicalPredictor:
    """
    Loads ML models and generates predictions for medical interpretations.
    Falls back to rule-based engine if ML models are not available.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_ml: bool = True,
        routing_mode: str = 'adaptive',
        ml_confidence_threshold: float = 0.70,
        abstain_confidence_threshold: float = 0.45,
        disagreement_margin: float = 0.10,
        enable_abstain: bool = True,
    ):
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
        self.last_used_ml = False  # Tracks whether the last predict() used any ML category
        self.last_sources: Dict[str, str] = {}  # Tracks per-category source for last prediction
        self.last_routing_details: Dict[str, Dict[str, Any]] = {}
        self.last_method_label: str = 'Rule-Based'

        # Hybrid routing configuration for methodological novelty and safety.
        self.routing_mode = routing_mode
        self.ml_confidence_threshold = ml_confidence_threshold
        self.abstain_confidence_threshold = abstain_confidence_threshold
        self.disagreement_margin = disagreement_margin
        self.enable_abstain = enable_abstain

        self.TEXT_TO_MODEL_CATEGORY = {
            'Left Ventricular Function': 'LV_FUNCTION',
            'LV Diastolic Dimension': 'LV_SIZE',
            'Interventricular Septum': 'LV_HYPERTROPHY',
            'Left Atrium': 'LA_SIZE',
            'Diastolic Function': 'DIASTOLIC_FUNCTION',
        }
        self.MODEL_TO_TEXT_CATEGORY = {v: k for k, v in self.TEXT_TO_MODEL_CATEGORY.items()}
        self.REQUIRED_FEATURES = {
            'Left Ventricular Function': ['EF', 'FS'],
            'LV Diastolic Dimension': ['LVID_D'],
            'Interventricular Septum': ['IVS_D'],
            'Left Atrium': ['LA_DIMENSION'],
            'Diastolic Function': ['MV_E_A'],
        }
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

        # Optionally compute ML interpretations and route per category.
        ml_interpretations = None
        routing_details: Dict[str, Dict[str, Any]] = {}
        if self.ml_available and self.use_ml_flag:
            try:
                ml_interpretations, ml_confidences = self._predict_with_ml(measurements, patient_info)

                for category, ml_text in ml_interpretations.items():
                    # Keep overall summary sourced from the routed per-category outputs.
                    if category == 'Overall Summary':
                        continue

                    decision, reason = self._route_category(
                        category=category,
                        measurements=measurements,
                        rule_text=combined.get(category),
                        ml_text=ml_text,
                        ml_confidence=float(ml_confidences.get(category, 0.0)),
                    )

                    routing_details[category] = {
                        'source': decision,
                        'reason': reason,
                        'ml_confidence': float(ml_confidences.get(category, 0.0)),
                    }

                    if decision == 'ML':
                        combined[category] = ml_text
                        sources[category] = 'ML'
                    elif decision == 'Abstain':
                        combined[category] = (
                            f"Uncertain automated interpretation for {category.lower()}; "
                            "manual expert review recommended."
                        )
                        sources[category] = 'Abstain'
                    else:
                        sources[category] = 'Rule'

                combined['Overall Summary'] = self._build_hybrid_summary(combined)
                sources['Overall Summary'] = self._summary_source(sources)
                self.last_used_ml = any(src == 'ML' for src in sources.values())
            except Exception as e:
                print(f"⚠ ML prediction failed: {e}, using rule-based results only")
                self.last_used_ml = False
        else:
            self.last_used_ml = False

        # Persist sources map for this prediction
        self.last_sources = sources
        self.last_routing_details = routing_details
        self.last_method_label = self._derive_method_label(sources)
        return combined

    def get_sources_for(self, interpretations: Dict[str, str]) -> Dict[str, str]:
        """Return a map of category -> source ("ML", "Rule", or "Abstain").
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

    def get_routing_details(self) -> Dict[str, Dict[str, Any]]:
        """Return detailed per-category routing diagnostics for the last prediction."""
        return dict(self.last_routing_details)

    def _route_category(
        self,
        category: str,
        measurements: Dict[str, float],
        rule_text: Optional[str],
        ml_text: str,
        ml_confidence: float,
    ) -> Tuple[str, str]:
        """Choose source per category using confidence, disagreement, and feature availability."""
        if self.routing_mode == 'ml_only':
            return 'ML', 'ml_only_mode'

        if self.routing_mode == 'rule_only':
            return 'Rule', 'rule_only_mode'

        has_required = self._has_required_features(category, measurements)
        if not has_required:
            return 'Rule', 'missing_critical_features'

        disagree = bool(rule_text and ml_text and rule_text != ml_text)
        high_conf = ml_confidence >= self.ml_confidence_threshold
        conflict_low_conf = disagree and ml_confidence < (self.ml_confidence_threshold + self.disagreement_margin)

        if high_conf and not conflict_low_conf:
            return 'ML', 'high_confidence_ml'

        if self.enable_abstain and disagree and ml_confidence < self.abstain_confidence_threshold:
            return 'Abstain', 'low_confidence_disagreement'

        return 'Rule', 'fallback_to_rule'

    def _has_required_features(self, category: str, measurements: Dict[str, float]) -> bool:
        required = self.REQUIRED_FEATURES.get(category, [])
        if not required:
            return True
        return any((measurements.get(feat, 0) or 0) > 0 for feat in required)

    def _summary_source(self, sources: Dict[str, str]) -> str:
        category_sources = [
            src for key, src in sources.items()
            if key in self.TEXT_TO_MODEL_CATEGORY
        ]
        if any(src == 'Abstain' for src in category_sources):
            return 'Abstain'
        if any(src == 'ML' for src in category_sources) and any(src == 'Rule' for src in category_sources):
            return 'Hybrid'
        if any(src == 'ML' for src in category_sources):
            return 'ML'
        return 'Rule'

    def _derive_method_label(self, sources: Dict[str, str]) -> str:
        vals = set(sources.values())
        if 'Abstain' in vals:
            return 'Hybrid-Abstain'
        if 'ML' in vals and 'Rule' in vals:
            return 'Hybrid-Adaptive'
        if 'ML' in vals:
            return 'ML-Based'
        return 'Rule-Based'

    def _build_hybrid_summary(self, interpretations: Dict[str, str]) -> str:
        """Build summary from routed category outputs to keep summary-source consistency."""
        summary_parts = []
        lv_func = interpretations.get('Left Ventricular Function', '')
        lv_size = interpretations.get('LV Diastolic Dimension', '')
        lv_hyp = interpretations.get('Interventricular Septum', '')
        la_size = interpretations.get('Left Atrium', '')
        diastolic = interpretations.get('Diastolic Function', '')

        if 'reduced' in lv_func.lower() or 'dysfunction' in lv_func.lower():
            summary_parts.append('LV systolic dysfunction')
        if 'dilat' in lv_size.lower():
            summary_parts.append('LV dilatation')
        if 'hypertrophy' in lv_hyp.lower():
            summary_parts.append('septal hypertrophy')
        if 'enlarg' in la_size.lower():
            summary_parts.append('LA enlargement')
        if 'dysfunction' in diastolic.lower() and 'normal' not in diastolic.lower():
            summary_parts.append('diastolic dysfunction')
        if 'manual expert review recommended' in ' '.join(interpretations.values()).lower():
            summary_parts.append('one or more uncertain categories requiring expert review')

        if summary_parts:
            return f"Overall: Echocardiography shows {', '.join(summary_parts)}"
        return "Overall: Echocardiographic parameters within normal limits"
    
    def _predict_with_ml(self, measurements: Dict[str, float],
                         patient_info: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, float]]:
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
        
        # Predict each category and capture confidence for routing.
        predictions = {}
        confidences = {}
        for category, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions[category] = pred
            confidence = 0.0
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = float(np.max(proba))
                except Exception:
                    confidence = 0.0
            confidences[category] = confidence
        
        # Generate natural language interpretations
        interpretations = self._ml_to_text(predictions, measurements, patient_info)

        text_confidences = {}
        for model_category, conf in confidences.items():
            text_category = self.MODEL_TO_TEXT_CATEGORY.get(model_category)
            if text_category:
                text_confidences[text_category] = conf

        return interpretations, text_confidences
    
    def _ml_to_text(self, predictions: Dict, measurements: Dict, 
                    patient_info: Dict) -> Dict[str, str]:
        """Convert ML predictions to natural language text."""
        
        interpretations = {}
        
        # LV Function
        lv_func = predictions.get('LV_FUNCTION')
        ef_value = measurements.get('EF', 0)
        if lv_func is not None and ef_value > 0:
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
        lv_size = predictions.get('LV_SIZE')
        lvid_d = measurements.get('LVID_D', 0)
        if lv_size is not None and lvid_d > 0:
            if lv_size == 'Normal':
                interpretations['LV Diastolic Dimension'] = \
                    f"Normal LV size (LVIDd: {lvid_d:.2f} cm)"
            elif lv_size == 'Mild':
                interpretations['LV Diastolic Dimension'] = \
                    f"Mild LV dilatation (LVIDd: {lvid_d:.2f} cm)"
            elif lv_size == 'Moderate':
                interpretations['LV Diastolic Dimension'] = \
                    f"Moderate LV dilatation (LVIDd: {lvid_d:.2f} cm)"
            elif lv_size == 'Severe':
                interpretations['LV Diastolic Dimension'] = \
                    f"Severe LV dilatation (LVIDd: {lvid_d:.2f} cm)"
        
        # LV Hypertrophy
        lv_hyp = predictions.get('LV_HYPERTROPHY')
        ivs_d = measurements.get('IVS_D', 0)
        if lv_hyp is not None and ivs_d > 0:
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
        la_size = predictions.get('LA_SIZE')
        la_dim = measurements.get('LA_DIMENSION', 0)
        if la_size is not None and la_dim > 0:
            if la_size == 'Normal':
                interpretations['Left Atrium'] = \
                    f"Normal LA size (LA: {la_dim:.2f} cm)"
            elif la_size == 'Enlarged':
                interpretations['Left Atrium'] = \
                    f"LA enlargement (LA: {la_dim:.2f} cm)"
        
        # Diastolic Function
        diastolic = predictions.get('DIASTOLIC_FUNCTION')
        mv_ea = measurements.get('MV_E_A', 0)
        if diastolic is not None and mv_ea > 0:
            if diastolic == 'Normal':
                interpretations['Diastolic Function'] = \
                    f"Normal diastolic function (E/A: {mv_ea:.2f})"
            elif diastolic == 'Abnormal':
                interpretations['Diastolic Function'] = \
                    f"Diastolic dysfunction (E/A: {mv_ea:.2f})"
        
        # Overall Summary
        summary_parts = []
        if lv_func not in [None, 'Normal', 'Unknown']:
            summary_parts.append(f"{lv_func.lower()} LV dysfunction")
        if lv_size in ['Mild', 'Moderate', 'Severe', 'Dilated']:
            summary_parts.append("LV dilatation")
        if lv_hyp not in [None, 'None', 'Unknown']:
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
            'method': self.last_method_label,
            'routing_details': self.get_routing_details(),
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

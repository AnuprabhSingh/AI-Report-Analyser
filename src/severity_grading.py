"""
Multi-class Disease Severity Grading Module
Extends binary classification to detailed severity grading for:
- Diastolic Dysfunction: Normal / Grade 1 / Grade 2 / Grade 3
- LVH: Normal / Mild / Moderate / Severe
- Other cardiac parameters with graduated severity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MultiClassSeverityGrader:
    """
    Multi-class grading system for disease severity assessment.
    Implements graduated classification beyond binary detection.
    """
    
    def __init__(self):
        """Initialize severity grader."""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Define severity classification criteria
        self._define_grading_criteria()
    
    def _define_grading_criteria(self):
        """Define clinical criteria for multi-class severity grading."""
        
        # Diastolic Dysfunction Grading (ASE/EACVI Guidelines)
        self.diastolic_dysfunction_criteria = {
            'Normal': {
                'E_A_ratio': (0.8, 2.0),
                'E_prime_septal': (8, 20),  # cm/s
                'E_E_prime': (0, 8),
                'LA_volume_index': (0, 34),  # ml/m²
                'TR_velocity': (0, 2.8)  # m/s
            },
            'Grade 1 - Impaired Relaxation': {
                'E_A_ratio': (0.5, 0.8),
                'E_prime_septal': (7, 8),
                'E_E_prime': (8, 13),
                'LA_volume_index': (34, 40),
                'TR_velocity': (2.8, 3.0),
                'description': 'Mild diastolic dysfunction with impaired relaxation'
            },
            'Grade 2 - Pseudonormal': {
                'E_A_ratio': (0.8, 2.0),
                'E_prime_septal': (0, 7),
                'E_E_prime': (13, 15),
                'LA_volume_index': (40, 45),
                'TR_velocity': (3.0, 3.5),
                'description': 'Moderate diastolic dysfunction with pseudonormalization'
            },
            'Grade 3 - Restrictive': {
                'E_A_ratio': (2.0, 5.0),
                'E_prime_septal': (0, 7),
                'E_E_prime': (15, 50),
                'LA_volume_index': (45, 100),
                'TR_velocity': (3.5, 5.0),
                'description': 'Severe diastolic dysfunction with restrictive pattern'
            }
        }
        
        # LVH Grading
        self.lvh_criteria = {
            'Normal': {
                'male': {
                    'LV_mass_index': (0, 115),  # g/m²
                    'IVS_d': (0.6, 1.0),  # cm
                    'LVPW_d': (0.6, 1.0),  # cm
                    'RWT': (0.24, 0.42)  # Relative wall thickness
                },
                'female': {
                    'LV_mass_index': (0, 95),
                    'IVS_d': (0.6, 0.9),
                    'LVPW_d': (0.6, 0.9),
                    'RWT': (0.24, 0.42)
                }
            },
            'Mild LVH': {
                'male': {
                    'LV_mass_index': (116, 131),
                    'IVS_d': (1.1, 1.3),
                    'LVPW_d': (1.1, 1.3),
                    'RWT': (0.42, 0.47)
                },
                'female': {
                    'LV_mass_index': (96, 108),
                    'IVS_d': (0.9, 1.2),
                    'LVPW_d': (0.9, 1.2),
                    'RWT': (0.42, 0.47)
                }
            },
            'Moderate LVH': {
                'male': {
                    'LV_mass_index': (132, 148),
                    'IVS_d': (1.4, 1.5),
                    'LVPW_d': (1.4, 1.5),
                    'RWT': (0.47, 0.52)
                },
                'female': {
                    'LV_mass_index': (109, 121),
                    'IVS_d': (1.2, 1.4),
                    'LVPW_d': (1.2, 1.4),
                    'RWT': (0.47, 0.52)
                }
            },
            'Severe LVH': {
                'male': {
                    'LV_mass_index': (149, 500),
                    'IVS_d': (1.6, 3.0),
                    'LVPW_d': (1.6, 3.0),
                    'RWT': (0.52, 1.0)
                },
                'female': {
                    'LV_mass_index': (122, 500),
                    'IVS_d': (1.4, 3.0),
                    'LVPW_d': (1.4, 3.0),
                    'RWT': (0.52, 1.0)
                }
            }
        }
        
        # Systolic Function Grading
        self.systolic_dysfunction_criteria = {
            'Normal': {'EF': (55, 70), 'FS': (25, 45)},
            'Mild': {'EF': (45, 54), 'FS': (20, 24)},
            'Moderate': {'EF': (30, 44), 'FS': (15, 19)},
            'Severe': {'EF': (0, 29), 'FS': (0, 14)}
        }
        
        # Valvular Dysfunction Grading
        self.valvular_criteria = {
            'Normal': {
                'MR_grade': 0,
                'AR_grade': 0,
                'description': 'No significant regurgitation'
            },
            'Mild': {
                'MR_grade': 1,
                'AR_grade': 1,
                'regurgitant_volume': (0, 30),  # ml
                'description': 'Mild regurgitation'
            },
            'Moderate': {
                'MR_grade': 2,
                'AR_grade': 2,
                'regurgitant_volume': (30, 60),
                'description': 'Moderate regurgitation'
            },
            'Severe': {
                'MR_grade': 3,
                'AR_grade': 3,
                'regurgitant_volume': (60, 200),
                'description': 'Severe regurgitation'
            }
        }
    
    def grade_diastolic_dysfunction(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        Grade diastolic dysfunction based on multiple parameters.
        
        Args:
            measurements: Dictionary with E/A ratio, E', E/E', LA volume, TR velocity
            
        Returns:
            Dictionary with grade, score, and interpretation
        """
        # Extract relevant measurements
        e_a_ratio = measurements.get('MV_E_A', measurements.get('E_A_ratio', None))
        e_prime = measurements.get('E_prime_septal', measurements.get('E_prime', None))
        e_e_prime = measurements.get('E_E_prime', measurements.get('E_e_prime', None))
        la_volume_index = measurements.get('LA_volume_index', None)
        tr_velocity = measurements.get('TR_velocity', None)
        
        # Track grade preferences for each parameter
        # Higher severity grades get priority when any parameter suggests them
        grade_preference_scores = {
            'Grade 3 - Restrictive': 0,
            'Grade 2 - Pseudonormal': 0,
            'Grade 1 - Impaired Relaxation': 0,
            'Normal': 0
        }
        
        total_params = 0
        params_evaluated = []
        
        # Evaluate each parameter and assign grades in order of severity
        if e_a_ratio is not None:
            total_params += 1
            params_evaluated.append('E/A ratio')
            # Check in order from most to least severe
            if 2.0 <= e_a_ratio <= 5.0:
                grade_preference_scores['Grade 3 - Restrictive'] += 1
            elif 0.8 <= e_a_ratio <= 2.0:
                # E/A 0.8-2.0 matches both Normal and Pseudonormal - use other params
                # Give slight preference to Normal as it's the broader range
                grade_preference_scores['Normal'] += 0.5
            elif 0.5 <= e_a_ratio < 0.8:
                grade_preference_scores['Grade 1 - Impaired Relaxation'] += 1
        
        if e_prime is not None:
            total_params += 1
            params_evaluated.append("E'")
            # Lower E' indicates worse diastolic function
            if e_prime >= 8:
                grade_preference_scores['Normal'] += 1
            elif 7 <= e_prime < 8:
                grade_preference_scores['Grade 1 - Impaired Relaxation'] += 1
            elif e_prime < 7:
                grade_preference_scores['Grade 2 - Pseudonormal'] += 0.5
                grade_preference_scores['Grade 3 - Restrictive'] += 0.5
        
        if e_e_prime is not None:
            total_params += 1
            params_evaluated.append("E/E'")
            # Higher E/E' indicates worse diastolic function
            if e_e_prime <= 8:
                grade_preference_scores['Normal'] += 1
            elif 8 < e_e_prime <= 13:
                grade_preference_scores['Grade 1 - Impaired Relaxation'] += 1
            elif 13 < e_e_prime <= 15:
                grade_preference_scores['Grade 2 - Pseudonormal'] += 1
            elif e_e_prime > 15:
                grade_preference_scores['Grade 3 - Restrictive'] += 1
        
        if la_volume_index is not None:
            total_params += 1
            params_evaluated.append('LA Volume Index')
            # Higher LA volume indicates worse diastolic function
            if la_volume_index <= 34:
                grade_preference_scores['Normal'] += 1
            elif 34 < la_volume_index <= 40:
                grade_preference_scores['Grade 1 - Impaired Relaxation'] += 1
            elif 40 < la_volume_index <= 45:
                grade_preference_scores['Grade 2 - Pseudonormal'] += 1
            elif la_volume_index > 45:
                grade_preference_scores['Grade 3 - Restrictive'] += 1
        
        if tr_velocity is not None:
            total_params += 1
            params_evaluated.append('TR Velocity')
            # Higher TR velocity indicates worse diastolic function
            if tr_velocity <= 2.8:
                grade_preference_scores['Normal'] += 1
            elif 2.8 < tr_velocity <= 3.0:
                grade_preference_scores['Grade 1 - Impaired Relaxation'] += 1
            elif 3.0 < tr_velocity <= 3.5:
                grade_preference_scores['Grade 2 - Pseudonormal'] += 1
            elif tr_velocity > 3.5:
                grade_preference_scores['Grade 3 - Restrictive'] += 1
        
        # Determine grade based on highest preference score
        if total_params == 0:
            return {
                'grade': 'Indeterminate',
                'confidence': 0.0,
                'score': 0,
                'total_criteria': 0,
                'description': 'Insufficient data for grading',
                'parameters_evaluated': []
            }
        
        # Normalize scores
        grade_scores = {}
        for grade, score in grade_preference_scores.items():
            grade_scores[grade] = score / total_params
        
        # Get best grade
        best_grade = max(grade_scores, key=grade_scores.get)
        confidence = grade_scores[best_grade]
        
        # Get description
        description = self.diastolic_dysfunction_criteria.get(best_grade, {}).get(
            'description', 
            f'{best_grade} diastolic function'
        )
        
        return {
            'grade': best_grade,
            'confidence': confidence,
            'grade_scores': grade_scores,
            'total_criteria': total_params,
            'description': description,
            'parameters_evaluated': params_evaluated,
            'numeric_grade': self._diastolic_grade_to_numeric(best_grade)
        }
    
    def grade_lvh(self, measurements: Dict[str, float], sex: str = 'M') -> Dict[str, Any]:
        """
        Grade Left Ventricular Hypertrophy severity.
        
        Args:
            measurements: Dictionary with LV mass, IVS, LVPW, RWT
            sex: Patient sex ('M' or 'F')
            
        Returns:
            Dictionary with grade, score, and interpretation
        """
        # Extract measurements
        lv_mass_index = measurements.get('LV_mass_index', measurements.get('LV_MASS_INDEX', None))
        ivs_d = measurements.get('IVS_D', measurements.get('IVS_d', None))
        lvpw_d = measurements.get('LVPW_D', measurements.get('LVPW_d', None))
        rwt = measurements.get('RWT', measurements.get('relative_wall_thickness', None))
        
        # Calculate RWT if not provided but components are available
        if rwt is None and ivs_d is not None and lvpw_d is not None:
            lvid_d = measurements.get('LVID_D', measurements.get('LVIDd', None))
            if lvid_d is not None and lvid_d > 0:
                rwt = (ivs_d + lvpw_d) / lvid_d
        
        sex_key = 'male' if sex.upper() == 'M' else 'female'
        
        # Grade preference scores
        grade_preference_scores = {
            'Severe LVH': 0,
            'Moderate LVH': 0,
            'Mild LVH': 0,
            'Normal': 0
        }
        
        total_params = 0
        params_evaluated = []
        
        # Define thresholds for LVH severity assessment
        male_lv_thresholds = [(115, None), (131, 116), (148, 132), (500, 149)]
        female_lv_thresholds = [(95, None), (108, 96), (121, 109), (500, 122)]
        
        # Evaluate each parameter - higher values indicate worse hypertrophy
        if lv_mass_index is not None:
            total_params += 1
            params_evaluated.append('LV Mass Index')
            
            # Get thresholds based on sex
            if sex_key == 'male':
                if lv_mass_index <= 115:
                    grade_preference_scores['Normal'] += 1
                elif lv_mass_index <= 131:
                    grade_preference_scores['Mild LVH'] += 1
                elif lv_mass_index <= 148:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
            else:
                if lv_mass_index <= 95:
                    grade_preference_scores['Normal'] += 1
                elif lv_mass_index <= 108:
                    grade_preference_scores['Mild LVH'] += 1
                elif lv_mass_index <= 121:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
        
        if ivs_d is not None:
            total_params += 1
            params_evaluated.append('IVS Thickness')
            
            if sex_key == 'male':
                if ivs_d <= 1.0:
                    grade_preference_scores['Normal'] += 1
                elif ivs_d <= 1.3:
                    grade_preference_scores['Mild LVH'] += 1
                elif ivs_d <= 1.5:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
            else:
                if ivs_d <= 0.9:
                    grade_preference_scores['Normal'] += 1
                elif ivs_d <= 1.2:
                    grade_preference_scores['Mild LVH'] += 1
                elif ivs_d <= 1.4:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
        
        if lvpw_d is not None:
            total_params += 1
            params_evaluated.append('LVPW Thickness')
            
            if sex_key == 'male':
                if lvpw_d <= 1.0:
                    grade_preference_scores['Normal'] += 1
                elif lvpw_d <= 1.3:
                    grade_preference_scores['Mild LVH'] += 1
                elif lvpw_d <= 1.5:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
            else:
                if lvpw_d <= 0.9:
                    grade_preference_scores['Normal'] += 1
                elif lvpw_d <= 1.2:
                    grade_preference_scores['Mild LVH'] += 1
                elif lvpw_d <= 1.4:
                    grade_preference_scores['Moderate LVH'] += 1
                else:
                    grade_preference_scores['Severe LVH'] += 1
        
        if rwt is not None:
            total_params += 1
            params_evaluated.append('Relative Wall Thickness')
            
            if rwt <= 0.42:
                grade_preference_scores['Normal'] += 1
            elif rwt <= 0.47:
                grade_preference_scores['Mild LVH'] += 0.5
            elif rwt <= 0.52:
                grade_preference_scores['Moderate LVH'] += 1
            else:
                grade_preference_scores['Severe LVH'] += 1
        
        # Determine grade
        if total_params == 0:
            return {
                'grade': 'Indeterminate',
                'confidence': 0.0,
                'score': 0,
                'total_criteria': 0,
                'description': 'Insufficient data for LVH grading',
                'parameters_evaluated': []
            }
        
        # Normalize scores
        grade_scores = {}
        for grade, score in grade_preference_scores.items():
            grade_scores[grade] = score / total_params
        
        # Get best grade
        best_grade = max(grade_scores, key=grade_scores.get)
        confidence = grade_scores[best_grade]
        
        # Determine LVH geometry if possible
        geometry = 'Unknown'
        if rwt is not None and lv_mass_index is not None:
            normal_mass_upper = self.lvh_criteria['Normal'][sex_key]['LV_mass_index'][1]
            if lv_mass_index > normal_mass_upper:
                if rwt > 0.42:
                    geometry = 'Concentric LVH'
                else:
                    geometry = 'Eccentric LVH'
            elif rwt > 0.42:
                geometry = 'Concentric Remodeling'
            else:
                geometry = 'Normal Geometry'
        
        return {
            'grade': best_grade,
            'confidence': confidence,
            'grade_scores': grade_scores,
            'total_criteria': total_params,
            'geometry': geometry,
            'description': f'{best_grade} - {geometry}',
            'parameters_evaluated': params_evaluated,
            'numeric_grade': self._lvh_grade_to_numeric(best_grade)
        }
    
    def grade_systolic_function(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        Grade systolic function severity.
        
        Args:
            measurements: Dictionary with EF and/or FS
            
        Returns:
            Dictionary with grade and interpretation
        """
        ef = measurements.get('EF', None)
        fs = measurements.get('FS', None)

        # Derive EF from volumes when not explicitly provided
        if ef is None:
            edv = measurements.get('EDV', None)
            esv = measurements.get('ESV', None)
            if edv is not None and esv is not None and edv > 0:
                ef = ((edv - esv) / edv) * 100.0

        # Derive FS from dimensions when not explicitly provided
        if fs is None:
            lvid_d = measurements.get('LVID_D', measurements.get('LVIDd', None))
            lvid_s = measurements.get('LVID_S', measurements.get('LVIDs', None))
            if lvid_d is not None and lvid_s is not None and lvid_d > 0:
                fs = ((lvid_d - lvid_s) / lvid_d) * 100.0
        
        grade_preference_scores = {
            'Severe': 0,
            'Moderate': 0,
            'Mild': 0,
            'Normal': 0
        }
        
        total_params = 0
        
        # EF-based grading (higher is better)
        if ef is not None:
            total_params += 1
            if ef >= 55:
                grade_preference_scores['Normal'] += 1
            elif ef >= 45:
                grade_preference_scores['Mild'] += 1
            elif ef >= 30:
                grade_preference_scores['Moderate'] += 1
            else:
                grade_preference_scores['Severe'] += 1
        
        # FS-based grading (higher is better)
        if fs is not None:
            total_params += 1
            if fs >= 25:
                grade_preference_scores['Normal'] += 1
            elif fs >= 20:
                grade_preference_scores['Mild'] += 1
            elif fs >= 15:
                grade_preference_scores['Moderate'] += 1
            else:
                grade_preference_scores['Severe'] += 1
        
        if total_params == 0:
            return {
                'grade': 'Indeterminate',
                'confidence': 0.0,
                'description': 'Insufficient data'
            }
        
        # Normalize and determine grade
        grade_scores = {}
        for grade, score in grade_preference_scores.items():
            grade_scores[grade] = score / total_params
        
        best_grade = max(grade_scores, key=grade_scores.get)
        
        # Create description
        descriptions = {
            'Normal': 'Normal left ventricular systolic function',
            'Mild': 'Mild left ventricular systolic dysfunction',
            'Moderate': 'Moderate left ventricular systolic dysfunction',
            'Severe': 'Severe left ventricular systolic dysfunction'
        }
        
        return {
            'grade': best_grade,
            'confidence': grade_scores[best_grade],
            'grade_scores': grade_scores,
            'description': descriptions[best_grade],
            'numeric_grade': self._systolic_grade_to_numeric(best_grade),
            'ef_value': ef,
            'fs_value': fs
        }
    
    def comprehensive_grading(self, measurements: Dict[str, float], 
                            patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive multi-class grading across all categories.
        
        Args:
            measurements: All available measurements
            patient_info: Patient demographic information
            
        Returns:
            Comprehensive grading report
        """
        sex = patient_info.get('sex', 'M')
        age = patient_info.get('age', 50)
        
        report = {
            'patient_info': patient_info,
            'measurements': measurements,
            'grades': {},
            'severity_summary': {},
            'clinical_recommendations': []
        }
        
        # Grade diastolic function
        dd_grade = self.grade_diastolic_dysfunction(measurements)
        report['grades']['diastolic_dysfunction'] = dd_grade
        
        # Grade LVH
        lvh_grade = self.grade_lvh(measurements, sex)
        report['grades']['lvh'] = lvh_grade
        
        # Grade systolic function
        systolic_grade = self.grade_systolic_function(measurements)
        report['grades']['systolic_function'] = systolic_grade
        
        # Compute overall severity score (0-10 scale)
        severity_score = self._compute_overall_severity(report['grades'])
        report['severity_summary'] = {
            'overall_score': severity_score,
            'severity_level': self._interpret_severity_score(severity_score),
            'primary_concerns': self._identify_primary_concerns(report['grades'])
        }
        
        # Generate clinical recommendations
        report['clinical_recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _diastolic_grade_to_numeric(self, grade: str) -> int:
        """Convert diastolic grade to numeric (0-3)."""
        mapping = {
            'Normal': 0,
            'Grade 1 - Impaired Relaxation': 1,
            'Grade 2 - Pseudonormal': 2,
            'Grade 3 - Restrictive': 3,
            'Indeterminate': -1
        }
        return mapping.get(grade, -1)
    
    def _lvh_grade_to_numeric(self, grade: str) -> int:
        """Convert LVH grade to numeric (0-3)."""
        mapping = {
            'Normal': 0,
            'Mild LVH': 1,
            'Moderate LVH': 2,
            'Severe LVH': 3,
            'Indeterminate': -1
        }
        return mapping.get(grade, -1)
    
    def _systolic_grade_to_numeric(self, grade: str) -> int:
        """Convert systolic grade to numeric (0-3)."""
        mapping = {
            'Normal': 0,
            'Mild': 1,
            'Moderate': 2,
            'Severe': 3,
            'Indeterminate': -1
        }
        return mapping.get(grade, -1)
    
    def _compute_overall_severity(self, grades: Dict[str, Dict]) -> float:
        """Compute overall severity score (0-10)."""
        scores = []
        weights = {
            'systolic_function': 0.4,
            'diastolic_dysfunction': 0.35,
            'lvh': 0.25
        }
        
        for category, weight in weights.items():
            if category in grades:
                numeric_grade = grades[category].get('numeric_grade', 0)
                if numeric_grade >= 0:
                    # Scale 0-3 to 0-10
                    score = (numeric_grade / 3.0) * 10.0
                    scores.append(score * weight)
        
        return sum(scores) if scores else 0.0
    
    def _interpret_severity_score(self, score: float) -> str:
        """Interpret overall severity score."""
        if score < 2.0:
            return "Normal/Minimal"
        elif score < 4.0:
            return "Mild"
        elif score < 7.0:
            return "Moderate"
        else:
            return "Severe"
    
    def _identify_primary_concerns(self, grades: Dict[str, Dict]) -> List[str]:
        """Identify primary clinical concerns."""
        concerns = []
        
        if 'systolic_function' in grades:
            grade = grades['systolic_function'].get('grade', 'Normal')
            if grade != 'Normal' and grade != 'Indeterminate':
                concerns.append(f"Systolic dysfunction ({grade})")
        
        if 'diastolic_dysfunction' in grades:
            grade = grades['diastolic_dysfunction'].get('grade', 'Normal')
            if 'Grade' in grade:
                concerns.append(f"Diastolic dysfunction ({grade})")
        
        if 'lvh' in grades:
            grade = grades['lvh'].get('grade', 'Normal')
            if 'LVH' in grade:
                concerns.append(f"Left ventricular hypertrophy ({grade})")
        
        return concerns if concerns else ["No significant abnormalities detected"]
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on grading."""
        recommendations = []
        grades = report.get('grades', {})
        
        # Systolic function recommendations
        if 'systolic_function' in grades:
            grade = grades['systolic_function'].get('grade', 'Normal')
            if grade == 'Severe':
                recommendations.append("Consider heart failure management and specialist referral")
            elif grade == 'Moderate':
                recommendations.append("Monitor cardiac function regularly, consider ACE inhibitors")
            elif grade == 'Mild':
                recommendations.append("Lifestyle modifications and regular follow-up recommended")
        
        # Diastolic dysfunction recommendations
        if 'diastolic_dysfunction' in grades:
            numeric = grades['diastolic_dysfunction'].get('numeric_grade', 0)
            if numeric >= 2:
                recommendations.append("Aggressive blood pressure control and volume management")
            elif numeric == 1:
                recommendations.append("Blood pressure optimization and annual echo follow-up")
        
        # LVH recommendations
        if 'lvh' in grades:
            grade = grades['lvh'].get('grade', 'Normal')
            if 'Severe' in grade:
                recommendations.append("Intensive blood pressure control, consider cardiology consultation")
            elif 'Moderate' in grade or 'Mild' in grade:
                recommendations.append("Blood pressure control and regular monitoring")
        
        if not recommendations:
            recommendations.append("Continue routine cardiac health monitoring")
        
        return recommendations
    
    def plot_severity_dashboard(self, grading_report: Dict[str, Any], 
                               save_path: Optional[str] = None):
        """
        Create visual dashboard of severity grades.
        
        Args:
            grading_report: Output from comprehensive_grading
            save_path: Path to save plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        grades = grading_report.get('grades', {})
        
        # 1. Overall severity gauge
        ax1 = fig.add_subplot(gs[0, :])
        severity_score = grading_report['severity_summary']['overall_score']
        self._plot_severity_gauge(ax1, severity_score)
        
        # 2. Individual grades bar chart
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_individual_grades(ax2, grades)
        
        # 3. Confidence scores
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_confidence_scores(ax3, grades)
        
        # 4. Grade distribution pie chart
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_grade_distribution(ax4, grades)
        
        # 5. Recommendations text
        ax5 = fig.add_subplot(gs[2, 1:])
        self._plot_recommendations(ax5, grading_report['clinical_recommendations'])
        
        plt.suptitle('Multi-Class Severity Grading Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dashboard to {save_path}")
        
        plt.show()
    
    def _plot_severity_gauge(self, ax, score: float):
        """Plot overall severity gauge."""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color segments
        colors = ['green', 'yellow', 'orange', 'red']
        boundaries = [0, 2, 4, 7, 10]
        
        for i in range(len(colors)):
            start_angle = boundaries[i] / 10 * np.pi
            end_angle = boundaries[i+1] / 10 * np.pi
            theta_segment = np.linspace(start_angle, end_angle, 25)
            ax.fill_between(theta_segment, 0, 1, color=colors[i], alpha=0.3)
        
        # Plot needle
        score_angle = score / 10 * np.pi
        ax.plot([score_angle, score_angle], [0, 0.9], 'k-', linewidth=3)
        ax.plot(score_angle, 0.9, 'ko', markersize=10)
        
        # Labels
        ax.set_ylim(0, 1)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0\nNormal', '2.5\nMild', '5\nModerate', '7.5\nSevere', '10'])
        ax.set_yticks([])
        ax.set_title(f'Overall Severity Score: {score:.1f}', fontsize=14, fontweight='bold')
        if 'polar' in ax.spines:
            ax.spines['polar'].set_visible(False)
    
    def _plot_individual_grades(self, ax, grades: Dict):
        """Plot individual category grades."""
        categories = []
        scores = []
        colors = []
        
        color_map = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', -1: 'gray'}
        
        for category, grade_info in grades.items():
            numeric = grade_info.get('numeric_grade', -1)
            if numeric >= 0:
                categories.append(category.replace('_', ' ').title())
                scores.append(numeric)
                colors.append(color_map.get(numeric, 'gray'))
        
        if categories:
            y_pos = np.arange(len(categories))
            ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('Severity Grade (0=Normal, 3=Severe)', fontsize=10)
            ax.set_xlim(0, 3.5)
            ax.set_title('Individual Category Grades', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    def _plot_confidence_scores(self, ax, grades: Dict):
        """Plot confidence scores."""
        categories = []
        confidences = []
        
        for category, grade_info in grades.items():
            conf = grade_info.get('confidence', 0)
            if conf > 0:
                categories.append(category.replace('_', ' ').title())
                confidences.append(conf)
        
        if categories:
            y_pos = np.arange(len(categories))
            bars = ax.barh(y_pos, confidences, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('Confidence', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_title('Grading Confidence', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax.text(conf + 0.02, i, f'{conf:.2f}', va='center', fontsize=9)
    
    def _plot_grade_distribution(self, ax, grades: Dict):
        """Plot grade distribution pie chart."""
        grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for category, grade_info in grades.items():
            numeric = grade_info.get('numeric_grade', -1)
            if numeric >= 0:
                grade_counts[numeric] += 1
        
        labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        sizes = [grade_counts[i] for i in range(4)]
        colors_pie = ['green', 'yellow', 'orange', 'red']
        
        # Only plot non-zero values
        non_zero = [(label, size, color) for label, size, color in zip(labels, sizes, colors_pie) if size > 0]
        if non_zero:
            labels_nz, sizes_nz, colors_nz = zip(*non_zero)
            ax.pie(
                sizes_nz,
                labels=labels_nz,
                colors=colors_nz,
                autopct='%1.0f%%',
                startangle=90,
                wedgeprops={'alpha': 0.7}
            )
            ax.set_title('Grade Distribution', fontsize=12, fontweight='bold')
    
    def _plot_recommendations(self, ax, recommendations: List[str]):
        """Plot clinical recommendations."""
        ax.axis('off')
        recommendations_text = "Clinical Recommendations:\n\n" + "\n\n".join(
            [f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]
        )
        ax.text(0.05, 0.95, recommendations_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


if __name__ == "__main__":
    print("Multi-Class Disease Severity Grading Module")
    print("=" * 50)
    print("Supports graduated classification for:")
    print("- Diastolic Dysfunction: Normal / Grade 1 / Grade 2 / Grade 3")
    print("- LVH: Normal / Mild / Moderate / Severe")
    print("- Systolic Function: Normal / Mild / Moderate / Severe")

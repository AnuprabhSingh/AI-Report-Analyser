"""
Rule-Based Clinical Interpretation Engine
Generates clinical interpretations based on medical guidelines and normal ranges.
"""

from typing import Dict, Any, List, Tuple


class ClinicalRuleEngine:
    """
    Rule-based engine for interpreting echocardiography measurements.
    Based on standard clinical guidelines (ASE/EACVI).
    """
    
    def __init__(self):
        """Initialize with normal ranges and interpretation rules."""
        # Normal ranges based on ASE/EACVI guidelines
        self.normal_ranges = {
            'EF': {
                'normal': (55, 70),
                'mild_dysfunction': (45, 54),
                'moderate_dysfunction': (30, 44),
                'severe_dysfunction': (0, 29)
            },
            'LVID_D': {
                'male': {
                    'normal': (4.2, 5.9),
                    'mild_dilation': (6.0, 6.3),
                    'moderate_dilation': (6.4, 6.8),
                    'severe_dilation': (6.9, 10.0)
                },
                'female': {
                    'normal': (3.9, 5.3),
                    'mild_dilation': (5.4, 5.7),
                    'moderate_dilation': (5.8, 6.1),
                    'severe_dilation': (6.2, 10.0)
                }
            },
            'LVID_S': {
                'male': {
                    'normal': (2.5, 4.0),
                    'mild_dilation': (4.1, 4.3),
                    'moderate_dilation': (4.4, 4.6),
                    'severe_dilation': (4.7, 10.0)
                },
                'female': {
                    'normal': (2.2, 3.5),
                    'mild_dilation': (3.6, 3.8),
                    'moderate_dilation': (3.9, 4.1),
                    'severe_dilation': (4.2, 10.0)
                }
            },
            'IVS_D': {
                'normal': (0.6, 1.0),
                'mild_hypertrophy': (1.1, 1.3),
                'moderate_hypertrophy': (1.4, 1.5),
                'severe_hypertrophy': (1.6, 3.0)
            },
            'LVPW_D': {
                'normal': (0.6, 1.0),
                'mild_hypertrophy': (1.1, 1.3),
                'moderate_hypertrophy': (1.4, 1.5),
                'severe_hypertrophy': (1.6, 3.0)
            },
            'LA_DIMENSION': {
                'male': {
                    'normal': (3.0, 4.0),
                    'mild_enlargement': (4.1, 4.6),
                    'moderate_enlargement': (4.7, 5.2),
                    'severe_enlargement': (5.3, 10.0)
                },
                'female': {
                    'normal': (2.7, 3.8),
                    'mild_enlargement': (3.9, 4.2),
                    'moderate_enlargement': (4.3, 4.6),
                    'severe_enlargement': (4.7, 10.0)
                }
            },
            'MV_E_A': {
                'young_normal': (1.5, 2.5),
                'adult_normal': (0.8, 1.5),
                'impaired_relaxation': (0.5, 0.79),
                'restrictive': (2.0, 5.0)
            },
            'FS': {
                'normal': (25, 45),
                'mild_dysfunction': (20, 24),
                'moderate_dysfunction': (15, 19),
                'severe_dysfunction': (0, 14)
            },
            'LV_MASS': {
                'male': {
                    'normal': (88, 224),
                    'mild_hypertrophy': (225, 258),
                    'moderate_hypertrophy': (259, 292),
                    'severe_hypertrophy': (293, 500)
                },
                'female': {
                    'normal': (67, 162),
                    'mild_hypertrophy': (163, 186),
                    'moderate_hypertrophy': (187, 210),
                    'severe_hypertrophy': (211, 500)
                }
            }
        }
    
    def interpret_measurements(self, measurements: Dict[str, float], 
                              patient_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate clinical interpretations for all measurements.
        
        Args:
            measurements: Dictionary of parameter values
            patient_info: Patient demographic information (age, sex)
            
        Returns:
            Dictionary of parameter -> interpretation string
        """
        interpretations = {}
        sex = patient_info.get('sex', 'M')
        age = patient_info.get('age', 50)
        
        # Interpret each measurement
        if 'EF' in measurements:
            interpretations['Left Ventricular Function'] = self._interpret_ef(measurements['EF'])
        
        if 'LVID_D' in measurements:
            interpretations['LV Diastolic Dimension'] = self._interpret_lvid_d(
                measurements['LVID_D'], sex
            )
        
        if 'LVID_S' in measurements:
            interpretations['LV Systolic Dimension'] = self._interpret_lvid_s(
                measurements['LVID_S'], sex
            )
        
        if 'IVS_D' in measurements:
            interpretations['Interventricular Septum'] = self._interpret_ivs_d(
                measurements['IVS_D']
            )
        
        if 'LVPW_D' in measurements:
            interpretations['LV Posterior Wall'] = self._interpret_lvpw_d(
                measurements['LVPW_D']
            )
        
        if 'LA_DIMENSION' in measurements:
            interpretations['Left Atrium'] = self._interpret_la(
                measurements['LA_DIMENSION'], sex
            )
        
        if 'MV_E_A' in measurements:
            interpretations['Diastolic Function'] = self._interpret_mv_ea(
                measurements['MV_E_A'], age
            )
        
        if 'FS' in measurements:
            interpretations['Fractional Shortening'] = self._interpret_fs(
                measurements['FS']
            )
        
        if 'LV_MASS' in measurements:
            interpretations['LV Mass'] = self._interpret_lv_mass(
                measurements['LV_MASS'], sex
            )
        
        # Generate summary interpretation
        interpretations['Overall Summary'] = self._generate_summary(
            measurements, patient_info, interpretations
        )
        
        return interpretations
    
    def _interpret_ef(self, ef_value: float) -> str:
        """Interpret Ejection Fraction."""
        ranges = self.normal_ranges['EF']
        
        if ranges['normal'][0] <= ef_value <= ranges['normal'][1]:
            return f"Normal LV systolic function (EF: {ef_value:.1f}%)"
        elif ranges['mild_dysfunction'][0] <= ef_value <= ranges['mild_dysfunction'][1]:
            return f"Mildly reduced LV systolic function (EF: {ef_value:.1f}%)"
        elif ranges['moderate_dysfunction'][0] <= ef_value <= ranges['moderate_dysfunction'][1]:
            return f"Moderately reduced LV systolic function (EF: {ef_value:.1f}%)"
        else:
            return f"Severely reduced LV systolic function (EF: {ef_value:.1f}%)"
    
    def _interpret_lvid_d(self, value: float, sex: str) -> str:
        """Interpret LV Internal Diameter in Diastole."""
        ranges = self.normal_ranges['LVID_D'].get(sex.lower(), self.normal_ranges['LVID_D']['male'])
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal LV size (LVIDd: {value:.2f} cm)"
        elif ranges['mild_dilation'][0] <= value <= ranges['mild_dilation'][1]:
            return f"Mild LV dilation (LVIDd: {value:.2f} cm)"
        elif ranges['moderate_dilation'][0] <= value <= ranges['moderate_dilation'][1]:
            return f"Moderate LV dilation (LVIDd: {value:.2f} cm)"
        else:
            return f"Severe LV dilation (LVIDd: {value:.2f} cm)"
    
    def _interpret_lvid_s(self, value: float, sex: str) -> str:
        """Interpret LV Internal Diameter in Systole."""
        ranges = self.normal_ranges['LVID_S'].get(sex.lower(), self.normal_ranges['LVID_S']['male'])
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal LV systolic dimension (LVIDs: {value:.2f} cm)"
        elif ranges['mild_dilation'][0] <= value <= ranges['mild_dilation'][1]:
            return f"Mild increase in LV systolic dimension (LVIDs: {value:.2f} cm)"
        elif ranges['moderate_dilation'][0] <= value <= ranges['moderate_dilation'][1]:
            return f"Moderate increase in LV systolic dimension (LVIDs: {value:.2f} cm)"
        else:
            return f"Severe increase in LV systolic dimension (LVIDs: {value:.2f} cm)"
    
    def _interpret_ivs_d(self, value: float) -> str:
        """Interpret Interventricular Septum thickness."""
        ranges = self.normal_ranges['IVS_D']
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal septal thickness (IVSd: {value:.2f} cm)"
        elif ranges['mild_hypertrophy'][0] <= value <= ranges['mild_hypertrophy'][1]:
            return f"Mild septal hypertrophy (IVSd: {value:.2f} cm)"
        elif ranges['moderate_hypertrophy'][0] <= value <= ranges['moderate_hypertrophy'][1]:
            return f"Moderate septal hypertrophy (IVSd: {value:.2f} cm)"
        else:
            return f"Severe septal hypertrophy (IVSd: {value:.2f} cm)"
    
    def _interpret_lvpw_d(self, value: float) -> str:
        """Interpret LV Posterior Wall thickness."""
        ranges = self.normal_ranges['LVPW_D']
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal posterior wall thickness (LVPWd: {value:.2f} cm)"
        elif ranges['mild_hypertrophy'][0] <= value <= ranges['mild_hypertrophy'][1]:
            return f"Mild posterior wall hypertrophy (LVPWd: {value:.2f} cm)"
        elif ranges['moderate_hypertrophy'][0] <= value <= ranges['moderate_hypertrophy'][1]:
            return f"Moderate posterior wall hypertrophy (LVPWd: {value:.2f} cm)"
        else:
            return f"Severe posterior wall hypertrophy (LVPWd: {value:.2f} cm)"
    
    def _interpret_la(self, value: float, sex: str) -> str:
        """Interpret Left Atrium dimension."""
        ranges = self.normal_ranges['LA_DIMENSION'].get(sex.lower(), 
                                                        self.normal_ranges['LA_DIMENSION']['male'])
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal LA size (LA: {value:.2f} cm)"
        elif ranges['mild_enlargement'][0] <= value <= ranges['mild_enlargement'][1]:
            return f"Mild LA enlargement (LA: {value:.2f} cm)"
        elif ranges['moderate_enlargement'][0] <= value <= ranges['moderate_enlargement'][1]:
            return f"Moderate LA enlargement (LA: {value:.2f} cm)"
        else:
            return f"Severe LA enlargement (LA: {value:.2f} cm)"
    
    def _interpret_mv_ea(self, value: float, age: int) -> str:
        """Interpret Mitral Valve E/A ratio for diastolic function."""
        ranges = self.normal_ranges['MV_E_A']
        
        if age < 50:
            if ranges['young_normal'][0] <= value <= ranges['young_normal'][1]:
                return f"Normal diastolic function (E/A: {value:.2f})"
        else:
            if ranges['adult_normal'][0] <= value <= ranges['adult_normal'][1]:
                return f"Normal diastolic function (E/A: {value:.2f})"
        
        if value < ranges['impaired_relaxation'][1]:
            return f"Impaired relaxation pattern - Grade I diastolic dysfunction (E/A: {value:.2f})"
        elif value >= ranges['restrictive'][0]:
            return f"Restrictive filling pattern (E/A: {value:.2f}) - suggest further evaluation"
        else:
            return f"Borderline diastolic function (E/A: {value:.2f})"
    
    def _interpret_fs(self, value: float) -> str:
        """Interpret Fractional Shortening."""
        ranges = self.normal_ranges['FS']
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal fractional shortening (FS: {value:.1f}%)"
        elif ranges['mild_dysfunction'][0] <= value <= ranges['mild_dysfunction'][1]:
            return f"Mildly reduced fractional shortening (FS: {value:.1f}%)"
        elif ranges['moderate_dysfunction'][0] <= value <= ranges['moderate_dysfunction'][1]:
            return f"Moderately reduced fractional shortening (FS: {value:.1f}%)"
        else:
            return f"Severely reduced fractional shortening (FS: {value:.1f}%)"
    
    def _interpret_lv_mass(self, value: float, sex: str) -> str:
        """Interpret LV Mass."""
        ranges = self.normal_ranges['LV_MASS'].get(sex.lower(), 
                                                    self.normal_ranges['LV_MASS']['male'])
        
        if ranges['normal'][0] <= value <= ranges['normal'][1]:
            return f"Normal LV mass (LVmass: {value:.0f} gm)"
        elif ranges['mild_hypertrophy'][0] <= value <= ranges['mild_hypertrophy'][1]:
            return f"Mild LV hypertrophy (LVmass: {value:.0f} gm)"
        elif ranges['moderate_hypertrophy'][0] <= value <= ranges['moderate_hypertrophy'][1]:
            return f"Moderate LV hypertrophy (LVmass: {value:.0f} gm)"
        else:
            return f"Severe LV hypertrophy (LVmass: {value:.0f} gm)"
    
    def _generate_summary(self, measurements: Dict[str, float], 
                         patient_info: Dict[str, Any],
                         interpretations: Dict[str, str]) -> str:
        """Generate overall clinical summary."""
        summary_points = []
        
        # Check for major findings
        ef_value = measurements.get('EF')
        if ef_value:
            if ef_value >= 55:
                summary_points.append("preserved systolic function")
            elif ef_value >= 45:
                summary_points.append("mildly impaired systolic function")
            else:
                summary_points.append("significantly reduced systolic function")
        
        # Check for LV dilation
        lvid_d = measurements.get('LVID_D')
        sex = patient_info.get('sex', 'M')
        if lvid_d:
            ranges = self.normal_ranges['LVID_D'].get(sex.lower(), 
                                                       self.normal_ranges['LVID_D']['male'])
            if lvid_d > ranges['normal'][1]:
                summary_points.append("LV dilation")
        
        # Check for hypertrophy
        ivs_d = measurements.get('IVS_D')
        if ivs_d and ivs_d > self.normal_ranges['IVS_D']['normal'][1]:
            summary_points.append("LV hypertrophy")
        
        # Check for LA enlargement
        la_dim = measurements.get('LA_DIMENSION')
        if la_dim:
            ranges = self.normal_ranges['LA_DIMENSION'].get(sex.lower(),
                                                            self.normal_ranges['LA_DIMENSION']['male'])
            if la_dim > ranges['normal'][1]:
                summary_points.append("LA enlargement")
        
        # Construct summary
        if not summary_points:
            return "Overall: Echocardiographic parameters within normal limits"
        else:
            return f"Overall: Echocardiography shows {', '.join(summary_points)}"


def main():
    """Demo usage of rule engine."""
    engine = ClinicalRuleEngine()
    
    # Example measurements
    sample_measurements = {
        'EF': 64.8,
        'LVID_D': 4.65,
        'LVID_S': 2.89,
        'IVS_D': 0.89,
        'LVPW_D': 0.98,
        'LA_DIMENSION': 3.47,
        'MV_E_A': 1.75,
        'FS': 38
    }
    
    sample_patient = {
        'age': 45,
        'sex': 'F',
        'name': 'Sample Patient'
    }
    
    # Generate interpretations
    interpretations = engine.interpret_measurements(sample_measurements, sample_patient)
    
    print("=" * 60)
    print("CLINICAL INTERPRETATION")
    print("=" * 60)
    for param, interpretation in interpretations.items():
        print(f"\n{param}:")
        print(f"  {interpretation}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

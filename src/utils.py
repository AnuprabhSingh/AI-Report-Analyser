"""
Utility functions for data cleaning, normalization, and helper operations.
"""

import re
import json
from typing import Dict, Any, Optional


def clean_measurement_value(value_str: str) -> Optional[float]:
    """
    Clean and convert measurement value string to float.
    Handles cases like: '64.8%', '4.65 cm', '1.75', etc.
    
    Args:
        value_str: Raw string value from PDF
        
    Returns:
        Float value or None if parsing fails
    """
    if not value_str or value_str.strip() == '':
        return None
    
    try:
        # Remove common units and special characters
        cleaned = value_str.strip()
        cleaned = re.sub(r'[%cm²³mlsec]', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Handle ranges (take first value)
        if '-' in cleaned:
            cleaned = cleaned.split('-')[0].strip()
        
        # Convert to float
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def normalize_measurement_name(name: str) -> Optional[str]:
    """
    Normalize measurement names to consistent format.
    Filters out invalid parameter names.
    
    Args:
        name: Raw measurement name
        
    Returns:
        Normalized name or None if invalid
    """
    if not name or not isinstance(name, str):
        return None
    
    # Convert to uppercase and replace spaces/special chars with underscore
    normalized = name.upper().strip()
    normalized = re.sub(r'[\s\-/]+', '_', normalized)
    normalized = re.sub(r'[()]', '', normalized)
    
    # Filter out invalid names (only digits, too short, etc.)
    if not normalized or len(normalized) < 2:
        return None
    if normalized.isdigit():  # Pure numbers are not valid parameter names
        return None
    if not any(c.isalpha() for c in normalized):  # Must contain at least one letter
        return None
    
    return normalized


def extract_patient_info(text: str) -> Dict[str, Any]:
    """
    Extract patient demographic information from report text.
    
    Args:
        text: Full text from PDF
        
    Returns:
        Dictionary with patient info (age, sex, etc.)
    """
    patient_info = {}
    
    # Extract age
    age_match = re.search(r'(\d+)\s*(YRS?|YEARS?|Y/O)', text, re.IGNORECASE)
    if age_match:
        patient_info['age'] = int(age_match.group(1))
    
    # Extract sex
    if re.search(r'\bMALE\b', text, re.IGNORECASE) and not re.search(r'\bFEMALE\b', text, re.IGNORECASE):
        patient_info['sex'] = 'M'
    elif re.search(r'\bFEMALE\b', text, re.IGNORECASE):
        patient_info['sex'] = 'F'
    
    # Extract patient name
    name_match = re.search(r'(?:NAME|PATIENT|PT)[:\s]+([A-Z\s]+?)(?:\d|\n|AGE)', text, re.IGNORECASE)
    if name_match:
        patient_info['name'] = name_match.group(1).strip()
    
    return patient_info


def convert_percentage_to_decimal(value: float, is_percentage: bool = False) -> float:
    """
    Convert percentage values to decimal if needed.
    
    Args:
        value: Numeric value
        is_percentage: Whether value is in percentage format
        
    Returns:
        Converted value
    """
    if is_percentage and value > 1:
        return value / 100
    return value


def validate_measurement_range(value: float, param_name: str) -> bool:
    """
    Validate if measurement value is within reasonable physiological ranges.
    
    Args:
        value: Measurement value
        param_name: Parameter name (EF, LVIDd, etc.)
        
    Returns:
        True if value is reasonable, False otherwise
    """
    # Basic sanity checks
    if value is None or not isinstance(value, (int, float)):
        return False
    if value < 0 or value > 10000:  # Extreme outliers
        return False
    
    # Define reasonable ranges for common parameters
    ranges = {
        'EF': (20, 90),           # Ejection Fraction (%)
        'FS': (20, 50),           # Fractional Shortening (%)
        'LVID_D': (3.0, 7.0),     # LV Internal Diameter Diastole (cm)
        'LVID_S': (2.0, 6.0),     # LV Internal Diameter Systole (cm)
        'IVS_D': (0.6, 1.5),      # Interventricular Septum Diastole (cm)
        'IVS_S': (0.8, 2.0),      # Interventricular Septum Systole (cm)
        'LVPW_D': (0.6, 1.5),     # LV Posterior Wall Diastole (cm)
        'LVPW_S': (0.8, 2.0),     # LV Posterior Wall Systole (cm)
        'LA_DIMENSION': (2.0, 6.0),  # Left Atrium (cm)
        'MV_E_A': (0.5, 3.0),     # Mitral Valve E/A ratio
        'MV_E': (40, 150),        # Mitral E velocity (cm/s)
        'MV_A': (30, 120),        # Mitral A velocity (cm/s)
        'AORTIC_ROOT': (2.0, 4.5), # Aortic Root (cm)
        'LA_AO': (0.5, 2.5),      # LA/Ao ratio
        'LV_MASS': (50, 400),     # LV Mass (gm)
        'EDV_TEICH': (50, 250),   # End Diastolic Volume (ml)
        'ESV_TEICH': (20, 150),   # End Systolic Volume (ml)
        'EDV_CUBED': (50, 250),   # End Diastolic Volume (ml)
        'ESV_CUBED': (20, 150),   # End Systolic Volume (ml)
        'SV_TEICH': (30, 150),    # Stroke Volume (ml)
        'SV_CUBED': (30, 150),    # Stroke Volume (ml)
    }
    
    if param_name in ranges:
        min_val, max_val = ranges[param_name]
        return min_val <= value <= max_val
    
    return True  # If no range defined, assume valid


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_clinical_output(measurements: Dict[str, float], 
                          interpretations: Dict[str, str]) -> str:
    """
    Format measurements and interpretations into readable clinical report.
    
    Args:
        measurements: Dictionary of parameter values
        interpretations: Dictionary of interpretation strings
        
    Returns:
        Formatted report string
    """
    output = "=" * 60 + "\n"
    output += "AUTOMATED CLINICAL INTERPRETATION\n"
    output += "=" * 60 + "\n\n"
    
    output += "MEASUREMENTS:\n"
    output += "-" * 60 + "\n"
    for param, value in measurements.items():
        unit = get_measurement_unit(param)
        output += f"{param:20s}: {value:6.2f} {unit}\n"
    
    output += "\n" + "=" * 60 + "\n"
    output += "CLINICAL INTERPRETATION:\n"
    output += "=" * 60 + "\n"
    
    for param, interpretation in interpretations.items():
        output += f"\n• {param}: {interpretation}"
    
    output += "\n\n" + "=" * 60 + "\n"
    
    return output


def get_measurement_unit(param_name: str) -> str:
    """
    Get standard unit for measurement parameter.
    
    Args:
        param_name: Parameter name
        
    Returns:
        Unit string
    """
    units = {
        'EF': '%',
        'LVID_D': 'cm',
        'LVID_S': 'cm',
        'IVS_D': 'cm',
        'LVPW_D': 'cm',
        'LA_DIMENSION': 'cm',
        'MV_E_A': 'ratio',
        'AORTIC_ROOT': 'cm',
        'LV_MASS': 'gm',
        'FS': '%',
    }
    return units.get(param_name, '')


def filter_valid_measurements(measurements: Dict[str, float]) -> Dict[str, float]:
    """
    Filter out invalid or spurious measurements.
    
    Args:
        measurements: Dictionary of extracted measurements
        
    Returns:
        Filtered dictionary with only valid measurements
    """
    # Known valid parameter names
    valid_params = {
        'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'IVS_S', 
        'LVPW_D', 'LVPW_S', 'LA_DIMENSION', 'AORTIC_ROOT',
        'MV_E_A', 'MV_E', 'MV_A', 'LA_AO', 'LV_MASS',
        'EDV_TEICH', 'ESV_TEICH', 'EDV_CUBED', 'ESV_CUBED',
        'SV_TEICH', 'SV_CUBED', 'AI_MAX_VEL', 'MAX_PG_AI',
        'PA_ACC_TIME', 'PI_MAX_VEL', 'MAX_PG_PI'
    }
    
    filtered = {}
    for param_name, value in measurements.items():
        # Check if parameter name is valid
        if param_name in valid_params:
            # Re-validate range
            if validate_measurement_range(value, param_name):
                filtered[param_name] = value
        else:
            # Log skipped parameters for debugging
            print(f"  Skipped invalid parameter: {param_name} = {value}")
    
    return filtered


def prioritize_measurements(measurements: Dict[str, float]) -> Dict[str, float]:
    """
    When multiple versions of same measurement exist, choose the best one.
    For example, prefer EF(Cubed) over EF(Teich).
    
    Args:
        measurements: Dictionary of measurements
        
    Returns:
        Dictionary with prioritized measurements
    """
    prioritized = measurements.copy()
    
    # Priority rules: prefer certain calculation methods
    priority_rules = [
        # (preferred_key, alternative_keys, output_key)
        ('EDV_CUBED', ['EDV_TEICH'], 'EDV'),
        ('ESV_CUBED', ['ESV_TEICH'], 'ESV'),
        ('SV_CUBED', ['SV_TEICH'], 'SV'),
    ]
    
    for preferred, alternatives, output_key in priority_rules:
        if preferred in measurements:
            # Use preferred method
            prioritized[output_key] = measurements[preferred]
            # Remove both the preferred and alternatives to avoid confusion
            prioritized.pop(preferred, None)
            for alt in alternatives:
                prioritized.pop(alt, None)
        elif any(alt in measurements for alt in alternatives):
            # Use first available alternative
            for alt in alternatives:
                if alt in measurements:
                    prioritized[output_key] = measurements[alt]
                    prioritized.pop(alt, None)  # Remove the original
                    break
    
    return prioritized

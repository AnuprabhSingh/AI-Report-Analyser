"""
PDF Medical Report Extractor
Extracts structured data (measurements, patient info) from medical PDF reports.
"""

import re
import json
import os
from typing import Dict, Any, List, Optional
import pdfplumber
from .utils import (
    clean_measurement_value, 
    normalize_measurement_name,
    extract_patient_info,
    validate_measurement_range,
    save_json,
    filter_valid_measurements,
    prioritize_measurements
)


class MedicalReportExtractor:
    """
    Extracts structured medical data from PDF reports.
    """
    
    def __init__(self):
        """Initialize extractor with measurement patterns."""
        # Common measurement patterns in echo reports
        self.measurement_patterns = {
            'EF': [
                r'EF\(Cubed\).*?:\s*([\d\.]+)',  # Prefer Cubed method first
                r'EF\(Teich\).*?:\s*([\d\.]+)',
                r'EF[:\s]+([\d\.]+)%?',
                r'Ejection\s+Fraction[:\s]+([\d\.]+)%?',
                r'LVEF[:\s]+([\d\.]+)%?'
            ],
            'LVID_D': [
                r'LVIDd.*?:\s*([\d\.]+)\s*cm',
                r'LVIDd[:\s]+([\d\.]+)',
                r'LV\s+Internal\s+Diameter.*?Diastole[:\s]+([\d\.]+)'
            ],
            'LVID_S': [
                r'LVIDs.*?:\s*([\d\.]+)\s*cm',
                r'LVIDs[:\s]+([\d\.]+)',
                r'LV\s+Internal\s+Diameter.*?Systole[:\s]+([\d\.]+)'
            ],
            'IVS_D': [
                r'IVSd.*?:\s*([\d\.]+)\s*cm',
                r'IVSd[:\s]+([\d\.]+)',
                r'Interventricular\s+Septum.*?Diastole[:\s]+([\d\.]+)'
            ],
            'IVS_S': [
                r'IVSs.*?:\s*([\d\.]+)\s*cm',
                r'IVSs[:\s]+([\d\.]+)'
            ],
            'LVPW_D': [
                r'LVPWd.*?:\s*([\d\.]+)\s*cm',
                r'LVPWd[:\s]+([\d\.]+)',
                r'LV\s+Posterior\s+Wall.*?Diastole[:\s]+([\d\.]+)'
            ],
            'LVPW_S': [
                r'LVPWs.*?:\s*([\d\.]+)\s*cm',
                r'LVPWs[:\s]+([\d\.]+)'
            ],
            'FS': [
                r'FS.*?:\s*([\d\.]+)\s*%',
                r'FS[:\s]+([\d\.]+)%?',
                r'Fractional\s+Shortening[:\s]+([\d\.]+)%?'
            ],
            'EDV_TEICH': [
                r'EDV\(Teich\).*?:\s*([\d\.]+)\s*ml'
            ],
            'ESV_TEICH': [
                r'ESV\(Teich\).*?:\s*([\d\.]+)\s*ml'
            ],
            'EDV_CUBED': [
                r'EDV\(Cubed\).*?:\s*([\d\.]+)\s*ml'
            ],
            'ESV_CUBED': [
                r'ESV\(Cubed\).*?:\s*([\d\.]+)\s*ml'
            ],
            'LV_MASS': [
                r'LVmass\(C\)d.*?:\s*([\d\.]+)\s*grams',
                r'LV\s+mass[:\s]+([\d\.]+)',
                r'Left\s+Ventricular\s+Mass[:\s]+([\d\.]+)'
            ],
            'SV_TEICH': [
                r'SV\(Teich\).*?:\s*([\d\.]+)\s*ml'
            ],
            'SV_CUBED': [
                r'SV\(Cubed\).*?:\s*([\d\.]+)\s*ml'
            ],
            'AORTIC_ROOT': [
                r'Ao root diam.*?:\s*([\d\.]+)\s*cm',
                r'Aortic\s+Root[:\s]+([\d\.]+)',
                r'Ao\s+root[:\s]+([\d\.]+)'
            ],
            'LA_DIMENSION': [
                r'LA dimension.*?:\s*([\d\.]+)\s*cm',
                r'LA\s+dimension[:\s]+([\d\.]+)',
                r'Left\s+Atrium[:\s]+([\d\.]+)'
            ],
            'LA_AO': [
                r'LA/AO.*?:\s*([\d\.]+)'
            ],
            'MV_E_A': [
                r'MV E/A.*?:\s*([\d\.]+)',
                r'E/A\s+ratio[:\s]+([\d\.]+)'
            ],
            'MV_E': [
                r'MV E point.*?:\s*([\d\.]+)\s*cm/s'
            ],
            'MV_A': [
                r'MV A point.*?:\s*([\d\.]+)\s*cm/s'
            ],
            'AI_MAX_VEL': [
                r'AI max vel.*?:\s*([\d\.]+)\s*cm/s'
            ],
            'MAX_PG_AI': [
                r'Max PG\(AI\).*?:\s*([\d\.]+)\s*mmHg'
            ],
            'PA_ACC_TIME': [
                r'PA acc time.*?:\s*([\d\.]+)\s*sec'
            ],
            'PI_MAX_VEL': [
                r'PI max vel.*?:\s*([\d\.]+)\s*cm/s'
            ],
            'MAX_PG_PI': [
                r'Max PG\(PI\).*?:\s*([\d\.]+)\s*mmHg'
            ]
        }
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all relevant data from PDF report.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted data
        """
        print(f"Processing: {pdf_path}")
        
        # Extract text from PDF
        full_text = self._extract_text_from_pdf(pdf_path)
        
        if not full_text:
            print(f"Warning: No text extracted from {pdf_path}")
            return None
        
        # Extract patient information
        patient_info = extract_patient_info(full_text)
        
        # Extract measurements
        measurements = self._extract_measurements(full_text)
        
        # Extract tables if present
        table_data = self._extract_tables_from_pdf(pdf_path)
        
        # Merge table data with regex-extracted measurements (prefer regex matches)
        if table_data:
            for key, value in table_data.items():
                if key not in measurements:  # Don't overwrite regex matches
                    measurements[key] = value
        
        print(f"  Extracted {len(measurements)} raw measurements")
        
        # Filter out invalid measurements
        measurements = filter_valid_measurements(measurements)
        print(f"  Valid measurements after filtering: {len(measurements)}")
        
        # Prioritize when duplicates exist
        measurements = prioritize_measurements(measurements)
        
        # Build structured output
        result = {
            'file_name': os.path.basename(pdf_path),
            'patient': patient_info,
            'measurements': measurements,
            'raw_text': full_text[:1000],  # First 1000 chars for reference
            'extraction_status': 'success' if measurements else 'partial'
        }
        
        return result
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ''
    
    def _extract_measurements(self, text: str) -> Dict[str, float]:
        """
        Extract measurement values using regex patterns.
        
        Args:
            text: Full text from PDF
            
        Returns:
            Dictionary of measurement name -> value
        """
        measurements = {}
        
        for param_name, patterns in self.measurement_patterns.items():
            for pattern_idx, pattern in enumerate(patterns):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = clean_measurement_value(match.group(1))
                    if value is not None and validate_measurement_range(value, param_name):
                        measurements[param_name] = value
                        print(f"  ✓ Matched {param_name} = {value} (pattern {pattern_idx + 1})")
                        break  # Found valid value, move to next parameter
        
        return measurements
    
    def _extract_tables_from_pdf(self, pdf_path: str) -> Dict[str, float]:
        """
        Extract measurement tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary of measurements extracted from tables
        """
        measurements = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table:
                            continue
                        
                        # Process each row
                        for row in table:
                            if not row or len(row) < 2:
                                continue
                            
                            # First column is typically parameter name
                            param_name = str(row[0]).strip() if row[0] else ''
                            
                            if not param_name:
                                continue
                            
                            # Look for numeric value in subsequent columns
                            for cell in row[1:]:
                                if cell:
                                    value = clean_measurement_value(str(cell))
                                    if value is not None:
                                        normalized_name = normalize_measurement_name(param_name)
                                        if normalized_name and validate_measurement_range(value, normalized_name):
                                            measurements[normalized_name] = value
                                            print(f"  ✓ Table: {normalized_name} = {value}")
                                        break
        
        except Exception as e:
            print(f"  Warning: Error extracting tables: {e}")
        
        return measurements
    
    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON outputs
            
        Returns:
            List of extracted data dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            
            # Extract data
            data = self.extract_from_pdf(pdf_path)
            
            if data:
                results.append(data)
                
                # Save individual JSON
                json_filename = os.path.splitext(pdf_file)[0] + '.json'
                json_path = os.path.join(output_dir, json_filename)
                save_json(data, json_path)
                print(f"Saved: {json_path}")
        
        # Save consolidated output
        if results:
            consolidated_path = os.path.join(output_dir, 'all_reports.json')
            save_json(results, consolidated_path)
            print(f"\nConsolidated output saved: {consolidated_path}")
        
        return results


def main():
    """Demo usage of extractor."""
    extractor = MedicalReportExtractor()
    
    # Example: Process single file
    # data = extractor.extract_from_pdf('path/to/report.pdf')
    # print(json.dumps(data, indent=2))
    
    # Example: Process directory
    input_dir = '../data/sample_reports/'
    output_dir = '../data/processed/'
    results = extractor.process_directory(input_dir, output_dir)
    print(f"\nProcessed {len(results)} reports successfully")


if __name__ == '__main__':
    main()

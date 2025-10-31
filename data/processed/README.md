# Processed Data Directory

This directory contains extracted data from PDF reports in JSON format.

## File Structure:

Each processed report generates a JSON file with:

```json
{
  "file_name": "report.pdf",
  "patient": {
    "age": 45,
    "sex": "F",
    "name": "Patient Name"
  },
  "measurements": {
    "EF": 64.8,
    "LVID_D": 4.65,
    "MV_E_A": 1.75,
    ...
  },
  "raw_text": "...",
  "extraction_status": "success"
}
```

## Consolidated Output:

`all_reports.json` contains all processed reports in a single array for analysis.

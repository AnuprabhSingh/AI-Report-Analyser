# 🏥 Automated Medical Report Interpretation System

**B.Tech Final Year Project - Machine Learning & Healthcare**

An intelligent system that automatically extracts measurements from medical PDF reports (echocardiography) and generates clinical interpretations using rule-based logic and machine learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system automates the interpretation of medical reports by:

1. **Extracting** structured data (measurements) from PDF reports
2. **Interpreting** measurements using clinical guidelines (ASE/EACVI)
3. **Generating** doctor-like clinical summaries
4. **Providing** REST API for integration with other systems

### Key Parameters Supported

- **EF** (Ejection Fraction) - Cardiac pumping efficiency
- **LVIDd/LVIDs** - Left Ventricle dimensions
- **IVSd** - Interventricular Septum thickness
- **LA Dimension** - Left Atrium size
- **MV E/A** - Diastolic function ratio
- **FS** - Fractional Shortening
- **LV Mass** - Left Ventricular mass

---

## ✨ Features

### 🔍 **Intelligent PDF Extraction**
- Extracts text and tables from medical PDFs
- Handles multiple report formats
- Normalizes units and measurements

### 📊 **Clinical Interpretation Engine**
- Rule-based interpretation using medical guidelines
- Sex and age-adjusted normal ranges
- Severity classification (Normal, Mild, Moderate, Severe)

### 🤖 **Machine Learning (Optional)**
- Classifier for automated severity prediction
- Supports training on custom datasets
- Falls back to rule-based if ML unavailable

### 🌐 **REST API**
- Upload PDF and get instant interpretation
- JSON input/output support
- Batch processing capability

### 📈 **Data Analysis**
- Jupyter notebook for exploratory analysis
- Visualization of measurement distributions
- Correlation analysis

---

## 📁 Project Structure

```
medical_interpreter/
│
├── data/
│   ├── sample_reports/          # Input PDF files
│   └── processed/                # Extracted JSON data
│
├── src/
│   ├── extractor.py             # PDF data extraction
│   ├── rule_engine.py           # Clinical interpretation rules
│   ├── model_trainer.py         # ML model training
│   ├── predictor.py             # Prediction/inference
│   ├── utils.py                 # Helper functions
│   └── api.py                   # Flask REST API
│
├── notebooks/
│   └── data_analysis.ipynb      # EDA and visualization
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── main.py                       # CLI entry point
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Navigate to Project

```bash
cd medical_interpreter
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python main.py --help
```

---

## 💻 Usage

### Command Line Interface (CLI)

#### 1️⃣ Extract Data from PDF

```bash
# Single file
python main.py extract path/to/report.pdf -o output.json

# Entire directory
python main.py extract data/sample_reports/ -o data/processed/
```

#### 2️⃣ Generate Clinical Interpretation

```bash
# From extracted JSON
python main.py interpret data/processed/report.json

# Directly from PDF
python main.py interpret path/to/report.pdf -o interpretation.json
```

#### 3️⃣ Train ML Models (Optional)

```bash
# Train on processed data
python main.py train -d data/processed/ -o models/

# Train with synthetic data
python main.py train -n 500
```

#### 4️⃣ Batch Processing

```bash
python main.py batch data/sample_reports/ -o data/processed/
```

---

### Python API Usage

```python
from src.extractor import MedicalReportExtractor
from src.predictor import ClinicalPredictor

# Extract data
extractor = MedicalReportExtractor()
data = extractor.extract_from_pdf('report.pdf')

# Generate interpretation
predictor = ClinicalPredictor()
interpretations = predictor.predict(
    measurements=data['measurements'],
    patient_info=data['patient']
)

# Print results
for category, interpretation in interpretations.items():
    print(f"{category}: {interpretation}")
```

---

### REST API Server

#### Start the Server

```bash
cd src
python api.py
```

Server runs at: `http://localhost:5000`

#### API Endpoints

**1. Upload PDF for Interpretation**

```bash
curl -X POST http://localhost:5000/api/interpret \
  -F "file=@report.pdf"
```

**2. Interpret from JSON**

```bash
curl -X POST http://localhost:5000/api/interpret/json \
  -H "Content-Type: application/json" \
  -d '{
    "measurements": {
      "EF": 64.8,
      "LVID_D": 4.65,
      "MV_E_A": 1.75
    },
    "patient": {
      "age": 45,
      "sex": "F"
    }
  }'
```

**3. Get Supported Parameters**

```bash
curl http://localhost:5000/api/parameters
```

**4. Health Check**

```bash
curl http://localhost:5000/health
```

---

## 🏗️ System Architecture

### Data Flow

```
PDF Report → Extraction → Normalization → Interpretation → Clinical Summary
                ↓              ↓               ↓
            Raw Text      Measurements    Rule Engine
                                              ↓
                                         ML Model (Optional)
```

### Components

1. **Extractor** (`extractor.py`)
   - Uses `pdfplumber` for text extraction
   - Regex patterns for measurement detection
   - Table parsing for structured data

2. **Rule Engine** (`rule_engine.py`)
   - Clinical guideline-based interpretation
   - ASE/EACVI standard ranges
   - Age and sex-adjusted ranges

3. **ML Trainer** (`model_trainer.py`)
   - Random Forest classifier
   - Feature engineering
   - Model persistence

4. **Predictor** (`predictor.py`)
   - Unified interface for rule-based + ML
   - Confidence scoring
   - Graceful fallback

5. **API** (`api.py`)
   - Flask web framework
   - File upload handling
   - JSON serialization

---

## 📊 Data Analysis

Open the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

The notebook includes:
- Distribution analysis of measurements
- Normal vs abnormal classification
- Correlation heatmaps
- Age/sex-based comparisons
- Outlier detection

---

## 🧪 Example Output

```
============================================================
CLINICAL INTERPRETATION
============================================================

Patient Information:
  Name: Sample Patient
  Age: 45 years
  Sex: F

Measurements:
  EF                 :  64.80 %
  LVID_D            :   4.65 cm
  LVID_S            :   2.89 cm
  IVS_D             :   0.89 cm
  LA_DIMENSION      :   3.47 cm
  MV_E_A            :   1.75 ratio

============================================================
CLINICAL INTERPRETATION
============================================================

Left Ventricular Function:
  Normal LV systolic function (EF: 64.8%)

LV Diastolic Dimension:
  Normal LV size (LVIDd: 4.65 cm)

Interventricular Septum:
  Normal septal thickness (IVSd: 0.89 cm)

Left Atrium:
  Normal LA size (LA: 3.47 cm)

Diastolic Function:
  Normal diastolic function (E/A: 1.75)

Overall Summary:
  Echocardiographic parameters within normal limits

============================================================
```

---

## 🔧 Configuration

### Adding New Parameters

Edit `src/utils.py` to add measurement patterns:

```python
self.measurement_patterns = {
    'NEW_PARAM': [
        r'Pattern1[:\s]+(\d+\.?\d*)',
        r'Pattern2[:\s]+(\d+\.?\d*)'
    ]
}
```

### Customizing Normal Ranges

Edit `src/rule_engine.py`:

```python
self.normal_ranges = {
    'NEW_PARAM': {
        'normal': (lower, upper),
        'mild': (lower, upper),
        # ...
    }
}
```

---

## 🐛 Troubleshooting

### PDF Extraction Issues

- **Problem**: No text extracted
  - **Solution**: Check if PDF is scanned image (use OCR)
  - Try: `camelot` for table extraction

### Import Errors

- **Problem**: `ModuleNotFoundError`
  - **Solution**: Ensure virtual environment is activated
  - Run: `pip install -r requirements.txt`

### API Server Issues

- **Problem**: Port already in use
  - **Solution**: Change port in `api.py`:
  ```python
  app.run(host='0.0.0.0', port=8000)
  ```

---

## 📚 References

### Medical Guidelines

- American Society of Echocardiography (ASE) Guidelines
- European Association of Cardiovascular Imaging (EACVI)
- Normal Values for Echocardiography (Lang et al., 2015)

### Technical References

- `pdfplumber` documentation
- scikit-learn for ML models
- Flask REST API best practices

---

## 🤝 Contributing

This is a B.Tech project, but contributions are welcome!

### To Contribute:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📜 License

This project is for educational purposes as part of a B.Tech final year project.

---

## 👨‍💻 Author

**Your Name**  
B.Tech Final Year Project  
Department of Computer Science / AI & ML

---

## 🙏 Acknowledgments

- Medical professionals who provided domain knowledge
- Open-source libraries: pdfplumber, scikit-learn, Flask
- ASE/EACVI for clinical guidelines

---

## 📧 Contact

For questions or support:
- Email: your.email@example.com
- GitHub: @yourusername

---

**⭐ If you find this project helpful, please star it!**

---

## 🗺️ Future Enhancements

- [ ] Deep Learning models (BERT, T5) for text generation
- [ ] OCR support for scanned PDFs
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Real-time monitoring dashboard
- [ ] DICOM image analysis integration
- [ ] Multi-modal learning (images + text)

---

*Last Updated: October 2025*

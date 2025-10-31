# 🤖 ML Model Training Guide

## Training Interpretation Models from Echocardiography Reports

This guide explains how to train machine learning models that automatically generate clinical interpretations from echocardiography datapoints.

---

## 📋 Overview

**Goal**: Train ML models to predict interpretation comments from extracted measurements (EF, LVID_D, IVS_D, etc.)

**Dataset**: ~240 echocardiography PDF reports

**Approach**:
1. Extract measurements from PDFs
2. Generate interpretations using rule-based engine (training labels)
3. Train classification models for each interpretation category
4. Generate natural language interpretations from predictions

---

## 🚀 Quick Start

### Option 1: Automated Workflow (Recommended)

```bash
python run_training_workflow.py
```

This interactive script guides you through:
- ✓ Data extraction from PDFs
- ✓ Model training
- ✓ Model testing

### Option 2: Manual Step-by-Step

```bash
# Step 1: Extract data from all PDFs
python prepare_training_data.py

# Step 2: Train ML models
python train_interpretation_model.py

# Step 3: Test trained models
python predict_with_ml.py
```

---

## 📊 Data Pipeline

### 1. Data Extraction (`prepare_training_data.py`)

**What it does**:
- Processes all PDF files in `data/sample_reports/`
- Extracts measurements (EF, LVID_D, IVS_D, etc.)
- Extracts patient info (age, sex, name)
- Generates rule-based interpretations (training labels)
- Saves individual JSON files + consolidated dataset

**Input**: 
```
data/sample_reports/
  ├── report_001.pdf
  ├── report_002.pdf
  └── ... (240 reports)
```

**Output**:
```
data/processed/
  ├── training_dataset.json          # Consolidated dataset
  ├── report_001.json                # Individual extractions
  ├── report_002.json
  └── ...
```

**Dataset Structure**:
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
    "IVS_D": 1.13,
    "LA_DIMENSION": 2.38,
    "MV_E_A": 1.75,
    ...
  },
  "interpretations": {
    "Left Ventricular Function": "Normal LV systolic function (EF: 64.8%)",
    "LV Diastolic Dimension": "Normal LV size (LVIDd: 4.65 cm)",
    "Interventricular Septum": "Mild septal hypertrophy (IVSd: 1.13 cm)",
    ...
  }
}
```

---

### 2. Model Training (`train_interpretation_model.py`)

**What it does**:
- Loads training dataset
- Prepares features (measurements + patient info)
- Extracts labels from interpretation text
- Trains Random Forest classifier for each category
- Evaluates on test set (80/20 split)
- Saves trained models

**Features Used** (12 total):
- Patient: `age`, `sex`
- Measurements: `EF`, `FS`, `LVID_D`, `LVID_S`, `IVS_D`, `LVPW_D`, `LA_DIMENSION`, `AORTIC_ROOT`, `MV_E_A`, `LV_MASS`

**Categories Predicted** (5 total):
1. **LV_FUNCTION**: Normal | Mild | Moderate | Severe
2. **LV_SIZE**: Normal | Dilated
3. **LV_HYPERTROPHY**: None | Mild | Moderate | Severe
4. **LA_SIZE**: Normal | Enlarged
5. **DIASTOLIC_FUNCTION**: Normal | Abnormal

**Models**:
- Algorithm: Random Forest Classifier
- Trees: 100
- Max depth: 10
- Evaluation: 20% test set

**Output**:
```
models/
  ├── scaler.pkl                     # Feature scaler
  ├── model_LV_FUNCTION.pkl          # LV function classifier
  ├── model_LV_SIZE.pkl              # LV size classifier
  ├── model_LV_HYPERTROPHY.pkl       # Hypertrophy classifier
  ├── model_LA_SIZE.pkl              # LA size classifier
  ├── model_DIASTOLIC_FUNCTION.pkl   # Diastolic function classifier
  └── model_metadata.json            # Feature names & categories
```

---

### 3. Prediction (`predict_with_ml.py`)

**What it does**:
- Loads trained models
- Takes new measurements as input
- Predicts interpretation category for each aspect
- Generates natural language interpretations
- Returns human-readable clinical text

**Example Usage**:

```python
from predict_with_ml import MLInterpretationPredictor

# Initialize
predictor = MLInterpretationPredictor()

# Input measurements
measurements = {
    'EF': 45,
    'LVID_D': 5.8,
    'IVS_D': 1.3,
    'LA_DIMENSION': 4.5,
    'MV_E_A': 0.8
}

patient_info = {
    'age': 65,
    'sex': 'M'
}

# Generate interpretations
interpretations = predictor.predict(measurements, patient_info)

# Output
print(interpretations)
# {
#   'Left Ventricular Function': 'Mildly reduced LV systolic function (EF: 45.0%)',
#   'LV Diastolic Dimension': 'LV dilatation (LVIDd: 5.80 cm)',
#   'Interventricular Septum': 'Moderate septal hypertrophy (IVSd: 1.30 cm)',
#   ...
# }
```

---

## 📈 Expected Results

### Dataset Statistics (240 reports)

| Metric | Value |
|--------|-------|
| **Total reports** | ~240 |
| **Training samples** | ~192 (80%) |
| **Test samples** | ~48 (20%) |
| **Avg measurements per report** | 15-25 |
| **Most common parameters** | EF, LVID_D, IVS_D, LA_DIMENSION |

### Model Performance (Expected)

| Category | Expected Accuracy |
|----------|-------------------|
| LV Function | 75-85% |
| LV Size | 80-90% |
| LV Hypertrophy | 70-80% |
| LA Size | 75-85% |
| Diastolic Function | 70-80% |

**Note**: Actual performance depends on data quality and label distribution.

---

## 🔧 Configuration

### Adjust Training Parameters

Edit `train_interpretation_model.py`:

```python
# Change test/train split
trainer.train_models(dataset_path, test_size=0.2)  # 20% test

# Change model hyperparameters
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)
```

### Add New Features

Edit `train_interpretation_model.py`:

```python
self.key_parameters = [
    'EF', 'FS', 'LVID_D', 'LVID_S', 'IVS_D', 'LVPW_D',
    'LA_DIMENSION', 'AORTIC_ROOT', 'MV_E_A', 'LV_MASS',
    'YOUR_NEW_PARAMETER'  # Add here
]
```

### Add New Categories

Edit `train_interpretation_model.py`:

```python
self.categories = [
    'LV_FUNCTION', 'LV_SIZE', 'LV_HYPERTROPHY',
    'LA_SIZE', 'DIASTOLIC_FUNCTION',
    'YOUR_NEW_CATEGORY'  # Add here
]

# Then implement label extraction in _extract_labels()
```

---

## 🐛 Troubleshooting

### Issue: No PDFs found

**Solution**: Ensure PDF files are in `data/sample_reports/`

```bash
ls data/sample_reports/*.pdf | wc -l
# Should show 240 (or your count)
```

### Issue: Few samples for category

**Cause**: Not enough data for that category in your PDFs

**Solutions**:
- Ensure rule engine generates correct interpretations
- Check label extraction logic in `_extract_labels()`
- May need more diverse dataset

### Issue: Low model accuracy

**Solutions**:
- Increase training data (more PDFs)
- Adjust model hyperparameters (more trees, deeper)
- Add more features (more measurements)
- Improve label quality (better rule engine)
- Try different algorithms (Gradient Boosting, SVM)

### Issue: Missing measurements

**Cause**: Not all reports have all parameters

**Solution**: The system handles this by:
- Using median values for missing data
- Feature scaling handles different ranges
- Models learn to work with incomplete data

---

## 📚 Model Files

After training, you'll have:

```
models/
├── scaler.pkl                      # StandardScaler for features
├── model_LV_FUNCTION.pkl           # 100 tree Random Forest
├── model_LV_SIZE.pkl               # 100 tree Random Forest
├── model_LV_HYPERTROPHY.pkl        # 100 tree Random Forest
├── model_LA_SIZE.pkl               # 100 tree Random Forest
├── model_DIASTOLIC_FUNCTION.pkl    # 100 tree Random Forest
└── model_metadata.json             # Feature names & categories
```

**Total size**: ~5-10 MB (all models)

---

## 🎯 Integration

### Use in Main Predictor

Edit `src/predictor.py` to use ML models:

```python
from predict_with_ml import MLInterpretationPredictor

class ClinicalPredictor:
    def __init__(self, use_ml=True):
        self.rule_engine = ClinicalRuleEngine()
        
        if use_ml:
            try:
                self.ml_predictor = MLInterpretationPredictor()
                self.use_ml = True
            except:
                print("ML models not found, using rules only")
                self.use_ml = False
    
    def predict(self, measurements, patient_info):
        if self.use_ml:
            return self.ml_predictor.predict(measurements, patient_info)
        else:
            return self.rule_engine.interpret(measurements, patient_info)
```

### Use in API

The trained models can be loaded once and used for all requests:

```python
from flask import Flask
from predict_with_ml import MLInterpretationPredictor

app = Flask(__name__)
predictor = MLInterpretationPredictor()  # Load once

@app.route('/api/interpret', methods=['POST'])
def interpret():
    data = request.json
    interpretations = predictor.predict(
        data['measurements'],
        data.get('patient_info', {})
    )
    return jsonify(interpretations)
```

---

## 🔮 Future Enhancements

1. **Deep Learning Models**
   - Use LSTM/Transformer for text generation
   - Generate more natural interpretations
   - Handle longer context

2. **Multi-output Models**
   - Single model predicting all categories
   - Better feature sharing
   - Faster inference

3. **Confidence Scores**
   - Return probability distributions
   - Flag uncertain predictions
   - Active learning for labeling

4. **Online Learning**
   - Update models with new validated data
   - Continuous improvement
   - Adapt to new patterns

5. **Ensemble Methods**
   - Combine rule-based + ML predictions
   - Vote or weighted average
   - Better robustness

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies installed
4. Verify data format matches expected structure

---

**Last Updated**: October 31, 2025

**Files Created**:
- `prepare_training_data.py` - Data extraction
- `train_interpretation_model.py` - Model training
- `predict_with_ml.py` - Inference
- `run_training_workflow.py` - Automated workflow
- `ML_TRAINING_GUIDE.md` - This guide

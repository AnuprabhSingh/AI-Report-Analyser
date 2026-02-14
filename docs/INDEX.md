# 📚 Documentation Index

Welcome to the Medical Report Interpretation System documentation!

## Quick Navigation

### 🚀 Getting Started
- **[README.md](../README.md)** - Project overview, features, and quick start
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup and common tasks
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System design and architecture

### 📖 Main Documentation

| Guide | Description | When to Read |
|-------|-------------|--------------|
| **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** | Complete project documentation | Understanding what was built |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Deploy to cloud platforms | Ready to deploy |
| **[ML_GUIDE.md](ML_GUIDE.md)** | Machine learning models | Training or understanding ML |
| **[ADVANCED_FEATURES_GUIDE.md](../ADVANCED_FEATURES_GUIDE.md)** | Explainability, risk stratification | Using advanced features |

### 📋 Quick Reference

- **[QUICKSTART.md](QUICKSTART.md)** - Getting started in 5 minutes
- **API Reference** - Available in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#api-documentation)
- **CLI Reference** - Available in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#cli-usage)

---

## Documentation Structure

```
medical_interpreter/
├── README.md                          # Main entry point
├── ARCHITECTURE.md                    # System architecture
├── ADVANCED_FEATURES_GUIDE.md        # Advanced features
├── docs/
│   ├── INDEX.md                       # This file
│   ├── QUICKSTART.md                  # Quick start guide
│   ├── PROJECT_OVERVIEW.md            # Complete project reference
│   ├── DEPLOYMENT.md                  # Deployment guide
│   └── ML_GUIDE.md                    # ML training and models
```

---

## By Task

### I want to...

**Set up the project**
→ Start with [QUICKSTART.md](QUICKSTART.md)

**Understand the system**
→ Read [README.md](../README.md) then [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**Deploy to production**
→ Follow [DEPLOYMENT.md](DEPLOYMENT.md)

**Train ML models**
→ Check [ML_GUIDE.md](ML_GUIDE.md)

**Use advanced features**
→ See [ADVANCED_FEATURES_GUIDE.md](../ADVANCED_FEATURES_GUIDE.md)

**Understand architecture**
→ Read [ARCHITECTURE.md](../ARCHITECTURE.md)

**Use the API**
→ See API section in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#api-documentation)

**Use CLI tools**
→ See CLI section in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#cli-usage)

---

## Documentation Overview

### PROJECT_OVERVIEW.md (~1000 lines)
Complete project documentation including:
- Problem statement and solution
- System architecture
- Technologies and implementation
- Features (core + advanced)
- Performance metrics
- Usage examples
- Deployment options
- Project statistics

### DEPLOYMENT.md (~700 lines)
Comprehensive deployment guide:
- Quick start (5-minute deploy)
- Docker deployment
- Cloud platforms (Render, Railway, Heroku)
- Split deployment (Backend + Frontend)
- Environment configuration
- Troubleshooting
- Platform comparisons

### ML_GUIDE.md (~1200 lines)
Machine learning documentation:
- Training workflow
- System architecture
- Model specifications (Gradient Boosting)
- Feature engineering
- Model comparison (v1 vs v2)
- Performance metrics
- Usage and integration
- Troubleshooting

### ADVANCED_FEATURES_GUIDE.md
Advanced analytics features:
- Model explainability (SHAP, PDP)
- Sensitivity analysis
- Multi-class severity grading
- Risk stratification
- Code examples
- Visualization guides

### QUICKSTART.md
Quick setup guide:
- Installation (5 minutes)
- Demo scripts
- Common commands
- CLI reference
- API quick reference
- Troubleshooting basics

---

## File Locations

### Core Application
```
src/
├── api.py                  # REST API server
├── extractor.py            # PDF extraction
├── rule_engine.py          # Clinical guidelines
├── predictor.py            # ML predictions
├── model_trainer.py        # Model training
├── explainability.py       # SHAP, feature importance
├── sensitivity_analysis.py # Uncertainty analysis
├── severity_grading.py     # Multi-class grading
└── risk_stratification.py  # Risk assessment
```

### Scripts
```
medical_interpreter/
├── demo.py                    # Basic demo
├── demo_advanced_features.py  # Advanced demo
├── train_interpretation_model.py  # Train models
├── predict_with_ml.py         # Use trained models
├── compare_models.py          # Compare model versions
├── test_model_accuracy.py     # Evaluate models
└── run_training_workflow.py   # Interactive training
```

### Data & Models
```
data/
├── sample_reports/         # Input PDFs
├── processed/              # Extracted JSON
└── processed_new/          # New dataset

models/
├── model_*.pkl            # Trained models
├── scaler.pkl             # Feature scaler
└── model_metadata*.json   # Model metadata
```

---

## Version History

**Current Version**: 2.0

### Version 2.0 (Current)
- ✅ Advanced features (explainability, sensitivity, risk)
- ✅ Improved ML models (98.11% accuracy)
- ✅ React frontend
- ✅ Comprehensive documentation
- ✅ Docker deployment
- ✅ Cloud-ready (Render, Railway, Vercel)

### Version 1.0
- ✓ PDF extraction
- ✓ Rule-based interpretation
- ✓ Basic ML models (93.75% accuracy)
- ✓ Flask API
- ✓ CLI tools

---

## Contributing

When adding new features or documentation:

1. **Update relevant docs** in `docs/` folder
2. **Keep README.md** as the main entry point
3. **Update this INDEX.md** if adding new docs
4. **Maintain consistency** with existing structure

---

## Need Help?

- **Setup issues**: See [QUICKSTART.md](QUICKSTART.md#troubleshooting)
- **Deployment problems**: See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)
- **ML training issues**: See [ML_GUIDE.md](ML_GUIDE.md#troubleshooting)
- **General questions**: Check [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

---

**Last Updated**: February 2026
**Project**: B.Tech Final Year Project - Medical Report Interpretation System

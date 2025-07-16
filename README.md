# Census Income Prediction - Complete Project

## 🎯 Project Overview

This is a comprehensive data science project that predicts income levels using U.S. Census data. The project demonstrates the complete machine learning pipeline from data exploration to production deployment, achieving **99.3% ROC-AUC** with advanced feature engineering and ensemble methods.

## 📦 project Contents

```
census_income_complete_package/
├── README.md                           # This file - setup instructions
├── WINDOWS_SETUP_GUIDE.md             # Detailed Windows setup guide
├── requirements.txt                    # Python dependencies
├── config.py                          # Configuration settings
├── data/
│   └── raw/                           # Original dataset files
│       ├── census_income_learn.csv    # Training data (199,523 samples)
│       ├── census_income_test.csv     # Test data (99,762 samples)
│       └── census_income_metadata.txt # Dataset documentation
├── src/                               # Source code modules
│   ├── data_pipeline.py              # Phase 1: Data Infrastructure & EDA
│   ├── preprocessing.py              # Phase 2: Data Preprocessing & Feature Engineering
│   └── model_development.py          # Phase 3: Model Development & Selection
├── notebooks/                         # Jupyter notebooks for testing
│   ├── phase1_testing.ipynb          # Phase 1 testing notebook
│   ├── phase2_testing.ipynb          # Phase 2 testing notebook
│   └── phase3_testing.ipynb          # Phase 3 testing notebook
├── reports/                           # Generated reports and visualizations
├── models/                            # Trained model artifacts
├── logs/                              # Execution logs
├── tests/                             # Test scripts
├── docker/                            # Docker configuration
├── api/                               # API endpoints
└── streamlit_app/                     # Web application
    ├── app.py                         # Main Streamlit application
    ├── utils/                         # Utility modules
    │   └── model_loader.py           # Model loading utilities
    └── requirements.txt               # Streamlit dependencies
```

## 🚀 Quick Start (Windows)

### Prerequisites
- **Python 3.11** (Download from https://python.org)
- **Git** (Download from https://git-scm.com)
- **Windows 10/11** with PowerShell or Command Prompt

### Step 1: Extract and Setup
```bash
# Extract the ZIP file to your desired location
# Open PowerShell/Command Prompt and navigate to the project folder
cd path\to\census_income_complete_package

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Individual Phases

#### Phase 1: Data Infrastructure & EDA
```bash
python src/data_pipeline.py
```
**Expected Output:**
- Data quality report
- 7 comprehensive visualizations
- Data leakage analysis
- EDA summary report

#### Phase 2: Data Preprocessing & Feature Engineering
```bash
python src/preprocessing.py
```
**Expected Output:**
- Processed datasets (188 engineered features)
- Preprocessing pipeline
- Feature engineering report
- Balanced training data

#### Phase 3: Model Development & Selection
```bash
python src/model_development.py
```
**Expected Output:**
- 5 trained models (Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network)
- Cross-validation results
- Model comparison report
- Feature importance analysis

### Step 3: Launch Web Application
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
**Access:** http://localhost:8501

## 📊 Expected Results

### Model Performance
- **XGBoost**: 99.30% ROC-AUC, 96.32% Accuracy
- **LightGBM**: 99.30% ROC-AUC, 96.25% Accuracy  
- **Random Forest**: 99.01% ROC-AUC, 95.16% Accuracy
- **Neural Network**: 96.76% ROC-AUC, 91.06% Accuracy
- **Logistic Regression**: 94.11% ROC-AUC, 86.83% Accuracy

### Data Processing
- **Original Features**: 41
- **Engineered Features**: 188 (370% increase)
- **Training Samples**: 199,523 → 152,807 (after duplicate removal)
- **Class Balance**: 93.8% vs 6.2% → 50% vs 50% (balanced)


## 📁 File Descriptions

### Core Scripts
- **`src/data_pipeline.py`**: Complete EDA with 41 feature visualizations, data leakage detection
- **`src/preprocessing.py`**: Advanced feature engineering, missing value handling, class balancing
- **`src/model_development.py`**: 5 ML models with hyperparameter tuning and cross-validation

### Configuration
- **`config.py`**: All project settings, paths, and parameters
- **`requirements.txt`**: Python package dependencies

### Web Application
- **`streamlit_app/app.py`**: Interactive web interface with live predictions
- **`streamlit_app/utils/model_loader.py`**: Model loading and prediction utilities

## 🎯 Phase-by-Phase Execution

### Phase 1: Data Infrastructure & EDA
```bash
python src/data_pipeline.py
```
**Generates:**
- `reports/data_quality_report.csv`
- `reports/data_leakage_report.txt`
- `reports/eda_report.md`
- `reports/visualizations/` (7 charts)

### Phase 2: Data Preprocessing 
```bash
python src/preprocessing.py
```
**Generates:**
- `models/preprocessing_pipeline.pkl`
- `models/processed_data.pkl`
- `reports/preprocessing_report.md`

### Phase 3: Model Development 
```bash
python src/model_development.py
```
**Generates:**
- `models/` (5 trained models)
- `reports/model_comparison.png`
- `reports/model_development_summary.txt`

### Phase 4: Web Application (Instant)
```bash
cd streamlit_app
streamlit run app.py
```
**Features:**
- Interactive dashboard
- Live predictions
- Model comparison
- Feature importance analysis

## 🌐 Web Application Features

### Dashboard
- Real-time performance metrics
- Model comparison charts
- Dataset statistics
- Professional styling

### Live Predictions
- User-friendly input forms
- Real-time classification
- Confidence scoring
- Sample data testing

### Analytics
- Feature importance visualization
- Model performance comparison
- Data insights and statistics
- Business intelligence

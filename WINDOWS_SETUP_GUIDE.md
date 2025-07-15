# Windows Setup Guide - Census Income Prediction Project

## 🖥️ Complete Windows Installation Guide

This guide provides step-by-step instructions for setting up and running the Census Income Prediction project on Windows 10/11.

## 📋 Prerequisites

### 1. Install Python 3.11
1. Go to https://python.org/downloads/
2. Download **Python 3.11.x** (latest version)
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```cmd
   python --version
   ```
   Should show: `Python 3.11.x`

### 2. Install Git (Optional but Recommended)
1. Go to https://git-scm.com/download/win
2. Download and install Git for Windows
3. Use default settings during installation

### 3. Install Visual Studio Code (Recommended)
1. Go to https://code.visualstudio.com/
2. Download and install VS Code
3. Install Python extension in VS Code

## 🚀 Project Setup

### Step 1: Extract Project Files
1. Extract the `census_income_complete_package.zip` file
2. Choose a location like `C:\Projects\census_income_prediction\`
3. You should see the following structure:
   ```
   census_income_prediction/
   ├── README.md
   ├── WINDOWS_SETUP_GUIDE.md
   ├── requirements.txt
   ├── config.py
   ├── data/
   ├── src/
   ├── streamlit_app/
   └── ...
   ```

### Step 2: Open Command Prompt or PowerShell
1. Press `Win + R`, type `cmd`, press Enter
2. Or press `Win + X`, select "Windows PowerShell"
3. Navigate to project directory:
   ```cmd
   cd C:\Projects\census_income_prediction
   ```

### Step 3: Create Virtual Environment
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment (Command Prompt)
venv\Scripts\activate

# OR for PowerShell
venv\Scripts\Activate.ps1
```

**Note**: If PowerShell gives execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies
```cmd
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**If installation fails**, install packages individually:
```cmd
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn xgboost lightgbm
pip install streamlit jupyter notebook
pip install imbalanced-learn shap lime
```

## 🔄 Running Each Phase

### Phase 1: Data Infrastructure & EDA
```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Run Phase 1
python src\data_pipeline.py
```

**Expected Runtime**: 15-20 minutes  
**Expected Output**:
- Console output showing progress
- Files created in `reports/` folder:
  - `data_quality_report.csv`
  - `data_leakage_report.txt`
  - `eda_report.md`
  - `visualizations/` folder with 7 charts

**What it does**:
- Loads and analyzes 300K+ census records
- Creates comprehensive visualizations for all 41 features
- Detects potential data leakage
- Generates data quality assessment

### Phase 2: Data Preprocessing & Feature Engineering
```cmd
# Run Phase 2
python src\preprocessing.py
```

**Expected Runtime**: 10-15 minutes  
**Expected Output**:
- Console output showing preprocessing steps
- Files created in `models/` folder:
  - `preprocessing_pipeline.pkl`
  - `processed_data.pkl`
- File created in `reports/`:
  - `preprocessing_report.md`

**What it does**:
- Handles missing values strategically
- Engineers 148 new features (40 → 188 total)
- Removes duplicates and cleans data
- Balances classes using SMOTE

### Phase 3: Model Development & Selection
```cmd
# Run Phase 3 (This takes the longest)
python src\model_development.py
```

**Expected Runtime**: 30-45 minutes  
**Expected Output**:
- Console output showing model training progress
- Files created in `models/` folder:
  - `logistic_regression_best_model.pkl`
  - `random_forest_best_model.pkl`
  - `xgboost_best_model.pkl`
  - `lightgbm_best_model.pkl`
  - `neural_network_best_model.pkl`
- Files created in `reports/`:
  - `model_comparison.png`
  - `model_development_summary.txt`

**What it does**:
- Trains 5 different ML algorithms
- Performs hyperparameter tuning
- Cross-validates all models
- Generates performance comparison

### Phase 4: Web Application
```cmd
# Navigate to streamlit app
cd streamlit_app

# Install streamlit requirements
pip install -r requirements.txt

# Run the web application
streamlit run app.py
```

**Expected Output**:
- Browser opens automatically to `http://localhost:8501`
- Interactive web application with:
  - Dashboard with model metrics
  - Live prediction interface
  - Model comparison charts
  - Feature importance analysis

## 🧪 Testing with Jupyter Notebooks

### Install Jupyter
```cmd
pip install jupyter notebook
```

### Run Testing Notebooks
```cmd
# Start Jupyter
jupyter notebook

# Open and run notebooks in this order:
# 1. notebooks/phase1_testing.ipynb
# 2. notebooks/phase2_testing.ipynb  
# 3. notebooks/phase3_testing.ipynb
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Python is not recognized"
**Problem**: Python not in PATH  
**Solution**:
1. Reinstall Python with "Add to PATH" checked
2. Or manually add Python to PATH:
   - Add `C:\Users\YourName\AppData\Local\Programs\Python\Python311\`
   - Add `C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts\`

#### 2. "pip is not recognized"
**Problem**: pip not in PATH  
**Solution**:
```cmd
python -m pip install --upgrade pip
```

#### 3. Package Installation Errors
**Problem**: Some packages fail to install  
**Solution**:
```cmd
# Try installing with --user flag
pip install --user package_name

# Or use conda instead
conda install package_name
```

#### 4. Memory Errors During Model Training
**Problem**: Not enough RAM for large dataset  
**Solution**:
1. Edit `config.py`
2. Change `SAMPLE_SIZE = 10000` (reduces dataset size)
3. Or increase virtual memory in Windows

#### 5. Streamlit Port Already in Use
**Problem**: Port 8501 is busy  
**Solution**:
```cmd
streamlit run app.py --server.port 8502
```

#### 6. PowerShell Execution Policy Error
**Problem**: Cannot run activation script  
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 7. Long Path Names (Windows Limitation)
**Problem**: Path too long error  
**Solution**:
1. Extract to shorter path like `C:\census\`
2. Or enable long paths in Windows:
   - Run `gpedit.msc`
   - Navigate to Computer Configuration > Administrative Templates > System > Filesystem
   - Enable "Enable Win32 long paths"

## 📊 Expected Performance

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **CPU**: Any modern processor (training will be faster with more cores)

### Runtime Expectations
- **Phase 1**: 15-20 minutes
- **Phase 2**: 10-15 minutes  
- **Phase 3**: 30-45 minutes (depends on CPU)
- **Web App**: Instant startup

### Model Performance Results
After Phase 3 completion, you should see:
- **XGBoost**: 99.30% ROC-AUC, 96.32% Accuracy
- **LightGBM**: 99.30% ROC-AUC, 96.25% Accuracy
- **Random Forest**: 99.01% ROC-AUC, 95.16% Accuracy
- **Neural Network**: 96.76% ROC-AUC, 91.06% Accuracy
- **Logistic Regression**: 94.11% ROC-AUC, 86.83% Accuracy

## 🎯 Quick Commands Reference

### Virtual Environment
```cmd
# Create
python -m venv venv

# Activate (CMD)
venv\Scripts\activate

# Activate (PowerShell)
venv\Scripts\Activate.ps1

# Deactivate
deactivate
```

### Run Phases
```cmd
# Phase 1: EDA
python src\data_pipeline.py

# Phase 2: Preprocessing  
python src\preprocessing.py

# Phase 3: Model Training
python src\model_development.py

# Phase 4: Web App
cd streamlit_app
streamlit run app.py
```

### Check Installation
```cmd
# Python version
python --version

# Installed packages
pip list

# Package info
pip show pandas
```

## 📁 Output Files Guide

### After Phase 1 (`reports/` folder):
- `data_quality_report.csv` - Data quality metrics
- `data_leakage_report.txt` - Leakage analysis results
- `eda_report.md` - Comprehensive EDA report
- `visualizations/` - 7 visualization files

### After Phase 2 (`models/` folder):
- `preprocessing_pipeline.pkl` - Reusable preprocessing pipeline
- `processed_data.pkl` - Clean, engineered dataset

### After Phase 3 (`models/` folder):
- 5 trained model files (`.pkl` format)
- `model_development_results.pkl` - Performance metrics

## 🌐 Web Application Usage

### Dashboard Features
1. **Performance Metrics**: View model accuracy and ROC-AUC scores
2. **Model Comparison**: Interactive charts comparing all algorithms
3. **Dataset Statistics**: Key insights about the data

### Live Predictions
1. Fill in demographic information
2. Click "Predict Income"
3. View results with confidence scores
4. Test with sample profiles

### Analytics
1. **Feature Importance**: See which factors matter most
2. **Data Insights**: Understand the dataset characteristics
3. **Model Details**: Technical performance metrics

## 🎓 Learning Outcomes

After completing this project, you will have:

1. **Data Science Pipeline**: End-to-end ML project experience
2. **Feature Engineering**: Advanced techniques for improving model performance
3. **Model Comparison**: Experience with multiple ML algorithms
4. **Web Deployment**: Skills in creating interactive data applications
5. **Production Code**: Clean, documented, reusable code structure

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Review error messages** carefully
3. **Check Python version** (must be 3.11.x)
4. **Verify virtual environment** is activated
5. **Try installing packages individually** if batch install fails

## 🏆 Success Indicators

You'll know everything is working when:

✅ All phases run without errors  
✅ Model performance matches expected results  
✅ Web application loads at http://localhost:8501  
✅ Predictions work in the web interface  
✅ All visualizations are generated  

Congratulations! You now have a complete, production-ready data science project running on your Windows machine!


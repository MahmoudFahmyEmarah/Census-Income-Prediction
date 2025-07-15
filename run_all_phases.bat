@echo off
echo ========================================
echo Census Income Prediction - All Phases
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
echo.

REM Run Phase 1
echo ========================================
echo Running Phase 1: Data Infrastructure & EDA
echo Expected time: 15-20 minutes
echo ========================================
python src\data_pipeline.py
if %errorlevel% neq 0 (
    echo Phase 1 failed! Check error messages above.
    pause
    exit /b 1
)
echo Phase 1 completed successfully!
echo.

REM Run Phase 2
echo ========================================
echo Running Phase 2: Data Preprocessing & Feature Engineering
echo Expected time: 10-15 minutes
echo ========================================
python src\preprocessing.py
if %errorlevel% neq 0 (
    echo Phase 2 failed! Check error messages above.
    pause
    exit /b 1
)
echo Phase 2 completed successfully!
echo.

REM Run Phase 3
echo ========================================
echo Running Phase 3: Model Development & Selection
echo Expected time: 30-45 minutes
echo ========================================
python src\model_development.py
if %errorlevel% neq 0 (
    echo Phase 3 failed! Check error messages above.
    pause
    exit /b 1
)
echo Phase 3 completed successfully!
echo.

echo ========================================
echo All phases completed successfully!
echo ========================================
echo.
echo Generated files:
echo - reports\data_quality_report.csv
echo - reports\eda_report.md
echo - reports\visualizations\ (7 charts)
echo - models\preprocessing_pipeline.pkl
echo - models\processed_data.pkl
echo - models\ (5 trained models)
echo - reports\model_comparison.png
echo.
echo To run the web application:
echo 1. cd streamlit_app
echo 2. pip install -r requirements.txt
echo 3. streamlit run app.py
echo.
echo Press any key to exit...
pause > nul


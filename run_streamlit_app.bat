@echo off
echo ========================================
echo Census Income Prediction - Web Application
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

REM Navigate to streamlit app directory
cd streamlit_app

REM Install streamlit requirements
echo Installing Streamlit requirements...
pip install -r requirements.txt
echo.

REM Run Streamlit app
echo ========================================
echo Starting Streamlit Web Application...
echo ========================================
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

echo.
echo Application stopped.
echo Press any key to exit...
pause > nul


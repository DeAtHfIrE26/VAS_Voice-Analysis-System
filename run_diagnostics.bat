@echo off
echo Running MemoTag Voice Analysis System Diagnostics...

:: Activate virtual environment
call .venv\Scripts\activate.bat || (
    echo Failed to activate virtual environment. Please make sure it exists.
    echo Run: python -m venv .venv
    pause
    exit /b 1
)

:: Run the diagnostic script
python test_system.py

:: Display results
echo.
echo Diagnostics complete. Please check the output above and the log file memotag_diagnostics.log
echo.
echo If any issues were found, try the following:
echo  1. Ensure you have FFmpeg installed
echo  2. Install any missing dependencies with: pip install -r requirements.txt
echo  3. Run the test_streamlit.py script to verify Streamlit is working: python -m streamlit run test_streamlit.py
echo.
 
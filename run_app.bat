@echo off
echo Setting up environment for MemoTag Voice Analysis System...

:: Set up database URL (using SQLite for local development)
set DATABASE_URL=sqlite:///./memotag.db

:: Check if OpenAI API key is set
if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY environment variable is not set.
    set /p user_api_key="Enter your OpenAI API key (or press Enter to use Vosk instead): "
    
    if not "%user_api_key%"=="" (
        set OPENAI_API_KEY=%user_api_key%
        echo Using provided OpenAI API key.
    ) else (
        echo No API key provided. Using Vosk for speech recognition (offline mode).
    )
) else (
    echo Using existing OPENAI_API_KEY from environment.
)

:: Activate virtual environment
call .venv\Scripts\activate.bat || (
    echo Failed to activate virtual environment. Please make sure it exists.
    echo Run: python -m venv .venv
    pause
    exit /b 1
)

:: Download a sample audio file if needed
echo Checking for sample audio file...
python download_sample_audio.py

:: Run the NLTK data download script
echo Downloading NLTK data...
python download_nltk_data.py

:: Run the application with improved error handling
echo Starting MemoTag Voice Analysis System with enhanced error handling...
python start_app.py

:: Check for error
if %ERRORLEVEL% neq 0 (
    echo An error occurred while running the application.
    echo Please check the memotag_startup.log file for details.
    pause
    exit /b %ERRORLEVEL%
)

pause 
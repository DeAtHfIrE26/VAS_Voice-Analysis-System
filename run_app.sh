#!/bin/bash
echo "Setting up environment for MemoTag Voice Analysis System..."

# Set up database URL (using SQLite for local development)
export DATABASE_URL="sqlite:///./memotag.db"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY environment variable is not set."
    echo -n "Enter your OpenAI API key (or press Enter to use Vosk instead): "
    read user_api_key
    
    if [ ! -z "$user_api_key" ]; then
        export OPENAI_API_KEY="$user_api_key"
        echo "Using provided OpenAI API key."
    else
        echo "No API key provided. Using Vosk for speech recognition (offline mode)."
    fi
else
    echo "Using existing OPENAI_API_KEY from environment."
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate || {
        echo "Failed to activate virtual environment. Please make sure it exists."
        echo "Run: python -m venv .venv"
        exit 1
    }
else
    echo "Virtual environment '.venv' not found. Creating it..."
    python3 -m venv .venv || {
        echo "Failed to create virtual environment. Is python3-venv installed?"
        echo "Run: sudo apt-get install python3-venv  # For Ubuntu/Debian"
        exit 1
    }
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Make script executable if needed
chmod +x start_app.py

# Run the application with improved error handling
echo "Starting MemoTag Voice Analysis System with enhanced error handling..."
python start_app.py

# Check for error
if [ $? -ne 0 ]; then
    echo "An error occurred while running the application."
    echo "Please check the memotag_startup.log file for details."
    exit 1
fi 
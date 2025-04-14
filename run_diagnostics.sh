#!/bin/bash
echo "Running MemoTag Voice Analysis System Diagnostics..."

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

# Run the diagnostic script
python test_system.py

# Display results
echo
echo "Diagnostics complete. Please check the output above and the log file memotag_diagnostics.log"
echo
echo "If any issues were found, try the following:"
echo "  1. Ensure you have FFmpeg installed"
echo "  2. Install any missing dependencies with: pip install -r requirements.txt"
echo "  3. Run the test_streamlit.py script to verify Streamlit is working: python -m streamlit run test_streamlit.py"
echo
echo "To run the full application with enhanced error handling, run: ./run_app.sh"
echo 
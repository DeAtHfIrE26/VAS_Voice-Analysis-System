#!/usr/bin/env python
"""
Startup script for MemoTag Voice Analysis System.
This script provides better error handling and logging before launching the Streamlit app.
"""

import os
import sys
import subprocess
import logging
import platform
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('memotag_startup.log')
    ]
)

logger = logging.getLogger('MemoTag')

def check_environment():
    """Check if the environment is properly set up."""
    logger.info("Checking environment...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    logger.info(f"Python version: {python_version}")
    
    # Check operating system
    system_info = platform.platform()
    logger.info(f"Operating system: {system_info}")
    
    # Check for required directories
    required_dirs = ['modules', 'utils', 'assets', 'nltk_data']
    for directory in required_dirs:
        if os.path.isdir(directory):
            logger.info(f"Directory '{directory}' found.")
        else:
            logger.warning(f"Directory '{directory}' not found. Creating it...")
            os.makedirs(directory, exist_ok=True)
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    logger.info(f"Running in virtual environment: {in_venv}")
    
    # Check for FFmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
    else:
        logger.warning("FFmpeg not found in PATH. Audio processing may be limited.")
    
    # Check for required environment variables
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        logger.info("DATABASE_URL environment variable is set.")
    else:
        logger.info("DATABASE_URL environment variable not set. Will use SQLite.")
        os.environ['DATABASE_URL'] = "sqlite:///./memotag.db"
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        logger.info("OPENAI_API_KEY environment variable is set.")
    else:
        logger.warning("OPENAI_API_KEY environment variable not set. Whisper API will not be available.")

def download_sample_audio():
    """Download a sample audio file for testing."""
    try:
        from download_sample_audio import download_sample_audio as dl_audio
        sample_path = dl_audio()
        logger.info(f"Sample audio file downloaded or verified at: {sample_path}")
    except Exception as e:
        logger.error(f"Error downloading sample audio: {str(e)}")

def download_nltk_data():
    """Download required NLTK data."""
    try:
        from download_nltk_data import ensure_nltk_data
        ensure_nltk_data()
        logger.info("NLTK data downloaded or verified.")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")

def verify_app_files():
    """Verify that all required application files exist."""
    required_files = [
        'app.py',
        'modules/audio_preprocessing.py', 
        'modules/speech_to_text.py',
        'modules/feature_extraction.py',
        'modules/machine_learning.py',
        'modules/visualization.py',
        'modules/reporting.py',
        'modules/database.py',
        'utils/helpers.py'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            logger.info(f"File '{file_path}' exists.")
        else:
            logger.error(f"File '{file_path}' is missing!")
            all_files_exist = False
    
    return all_files_exist

def run_streamlit():
    """Run the Streamlit application."""
    logger.info("Starting Streamlit application...")
    try:
        # Try to find an available port
        port = 8501
        while port < 8520:
            try:
                # Create a temporary socket to test if port is available
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('localhost', port))
                s.close()
                break
            except OSError:
                logger.info(f"Port {port} is in use, trying next port...")
                port += 1
        
        # Prepare Streamlit command
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Start Streamlit process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Log process output
        logger.info(f"Streamlit started on port {port}. Open http://localhost:{port} in your browser.")
        logger.info("Logging Streamlit output below:")
        
        for line in process.stdout:
            logger.info(line.strip())
        
        for line in process.stderr:
            logger.error(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"Streamlit exited with code {return_code}")
        else:
            logger.info("Streamlit exited successfully")
            
    except Exception as e:
        logger.error(f"Error running Streamlit: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("=" * 80)
        logger.info("Starting MemoTag Voice Analysis System")
        logger.info("=" * 80)
        
        # Run setup checks
        check_environment()
        if not verify_app_files():
            logger.error("Some required files are missing. Please check installation.")
            sys.exit(1)
        
        # Download required data
        download_sample_audio()
        download_nltk_data()
        
        # Run the application
        run_streamlit()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1) 
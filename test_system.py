#!/usr/bin/env python
"""
Diagnostic test script for MemoTag Voice Analysis System.
This script tests all components of the system to verify everything is working correctly.
"""

import os
import sys
import importlib
import subprocess
import platform
import logging
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('memotag_diagnostics.log')
    ]
)

logger = logging.getLogger('MemoTag-Diagnostics')

def print_header(title):
    """Print a formatted header."""
    logger.info("=" * 80)
    logger.info(f" {title} ".center(80, "="))
    logger.info("=" * 80)

def check_python_environment():
    """Check Python environment and dependencies."""
    print_header("PYTHON ENVIRONMENT")
    
    # Python version
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    # Operating system
    os_info = platform.platform()
    logger.info(f"Operating system: {os_info}")
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    logger.info(f"Running in virtual environment: {in_venv}")
    
    # Check pip 
    try:
        import pip
        logger.info(f"Pip version: {pip.__version__}")
    except ImportError:
        logger.error("Pip is not installed or not in PYTHONPATH")
    
    # Check required packages
    required_packages = [
        'streamlit', 'numpy', 'pandas', 'matplotlib', 'openai', 
        'nltk', 'scikit-learn', 'sqlalchemy', 'vosk', 'librosa'
    ]
    
    logger.info("Checking required packages:")
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"  ✓ {package}: {version}")
        except ImportError:
            logger.error(f"  ✗ {package}: Not installed")

def check_system_dependencies():
    """Check system dependencies."""
    print_header("SYSTEM DEPENDENCIES")
    
    # Check for FFmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
        
        # Check FFmpeg version
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=True)
            version_line = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg version: {version_line}")
        except subprocess.SubprocessError:
            logger.error("Could not determine FFmpeg version")
    else:
        logger.warning("FFmpeg not found in PATH. Audio processing may be limited.")

def check_nltk_data():
    """Check NLTK data."""
    print_header("NLTK DATA")
    
    try:
        import nltk
        
        # List NLTK data path
        logger.info(f"NLTK data path: {nltk.data.path}")
        
        # Check 'punkt' tokenizer which is required
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("✓ Punkt tokenizer found")
        except LookupError:
            logger.warning("✗ Punkt tokenizer not found")
            logger.info("Attempting to download punkt...")
            nltk.download('punkt')
    except ImportError:
        logger.error("NLTK not installed")

def check_openai_api():
    """Check OpenAI API configuration."""
    print_header("OPENAI API")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        logger.info("OpenAI API key is set")
        # Don't log or display the actual key for security reasons
        
        # Test API connection (without making an actual request)
        try:
            from openai import OpenAI
            client = OpenAI()
            logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
    else:
        logger.warning("OpenAI API key is not set. Whisper API will not be available.")

def check_application_files():
    """Check application files."""
    print_header("APPLICATION FILES")
    
    required_files = [
        'app.py',
        'start_app.py',
        'download_nltk_data.py',
        'download_sample_audio.py',
        'modules/audio_preprocessing.py', 
        'modules/speech_to_text.py',
        'modules/feature_extraction.py',
        'modules/machine_learning.py',
        'modules/visualization.py',
        'modules/reporting.py',
        'modules/database.py',
        'utils/helpers.py'
    ]
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            logger.info(f"✓ {file_path} exists")
        else:
            logger.error(f"✗ {file_path} is missing!")

def test_audio_processing():
    """Test audio processing capabilities."""
    print_header("AUDIO PROCESSING")
    
    # Check for sample audio file
    sample_path = os.path.join('assets', 'sample_speech.mp3')
    if os.path.isfile(sample_path):
        logger.info(f"Sample audio file found at: {sample_path}")
        
        # Test audio loading
        try:
            from modules.audio_preprocessing import get_audio_info
            audio_info = get_audio_info(sample_path)
            logger.info(f"Successfully loaded audio file: {audio_info}")
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
    else:
        logger.warning(f"Sample audio file not found at: {sample_path}")
        logger.info("Trying to download sample audio...")
        try:
            from download_sample_audio import download_sample_audio
            path = download_sample_audio()
            logger.info(f"Downloaded sample audio to: {path}")
        except Exception as e:
            logger.error(f"Error downloading sample audio: {str(e)}")

def test_database():
    """Test database connections."""
    print_header("DATABASE")
    
    # Check database configuration
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        logger.info(f"DATABASE_URL is set: {db_url if not db_url.startswith('postgresql') else 'postgresql://**redacted**'}")
    else:
        logger.info("DATABASE_URL not set, will use SQLite by default")
    
    # Test database connection
    try:
        from modules.database import get_user_by_username
        result = get_user_by_username("test_user")
        logger.info("Successfully connected to database")
        logger.info(f"Test query result: {result}")
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")

def test_streamlit():
    """Test Streamlit setup."""
    print_header("STREAMLIT")
    
    try:
        import streamlit
        logger.info(f"Streamlit version: {streamlit.__version__}")
        
        # Check if Streamlit config directory exists
        config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
        if os.path.isdir(config_dir):
            logger.info(f"Streamlit config directory exists at: {config_dir}")
        else:
            logger.info("Streamlit config directory does not exist yet")
        
        logger.info("You can run 'python test_streamlit.py' to test the Streamlit setup")
    except ImportError:
        logger.error("Streamlit is not installed")

def run_diagnostics():
    """Run all diagnostic tests."""
    logger.info("=" * 80)
    logger.info(" MemoTag Voice Analysis System - Diagnostics ".center(80, "="))
    logger.info("=" * 80)
    
    try:
        check_python_environment()
        check_system_dependencies()
        check_nltk_data()
        check_openai_api()
        check_application_files()
        test_audio_processing()
        test_database()
        test_streamlit()
        
        print_header("DIAGNOSTIC SUMMARY")
        logger.info("Diagnostics completed. Check the log for any errors or warnings.")
        logger.info("For more detailed logging, run: python start_app.py")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Unhandled exception during diagnostics: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_diagnostics() 
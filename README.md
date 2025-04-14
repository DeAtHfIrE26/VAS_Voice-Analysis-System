# <div align="center">🎙️ MemoTag Voice Analysis System</div>

<div align="center">
  <img src="images/Dashboard.png" alt="MemoTag Dashboard" width="800"/>
</div>

<div align="center">
  <p><em>Advanced voice analysis platform for early detection of cognitive decline markers</em></p>
</div>

<div align="center">
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  ![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
  ![Streamlit: 1.28+](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
  ![Status: Beta](https://img.shields.io/badge/Status-Beta-yellow)
  
</div>

<hr>

## 🔍 Overview

**MemoTag Voice Analysis System** is a cutting-edge application that analyzes vocal patterns to identify early indicators of cognitive decline. Using natural language processing, acoustic analysis, and machine learning, MemoTag extracts meaningful linguistic, acoustic, and temporal features from voice recordings to detect subtle changes that may indicate cognitive impairment.

<div align="center">
  <table>
    <tr>
      <td width="50%">
        <img src="images/Upload.png" alt="Voice Upload" width="100%"/>
        <p align="center"><em>Audio Upload & Analysis</em></p>
      </td>
      <td width="50%">
        <img src="images/Results.png" alt="Analysis Results" width="100%"/>
        <p align="center"><em>Detailed Analysis Results</em></p>
      </td>
    </tr>
  </table>
</div>

## ✨ Key Features

- 🔊 **Dual Speech Recognition Engines**: Uses OpenAI's Whisper API for cloud processing or Vosk for offline analysis
- 🧠 **Comprehensive Feature Extraction**: Analyzes 30+ linguistic, acoustic, and temporal voice markers
- 📊 **Advanced Visualization**: Interactive charts and plots for data analysis
- 🤖 **ML-Powered Analysis**: Employs unsupervised machine learning for anomaly detection
- 🧪 **Cognitive Assessment**: Identifies markers associated with cognitive decline
- 📝 **Detailed Reporting**: Generates comprehensive HTML reports for healthcare professionals
- 💾 **User Management**: Save, retrieve, and compare analyses over time
- 🔄 **Robust Error Handling**: Gracefully manages failures with detailed diagnostics

## 📊 Analysis Workflow

<div align="center">
  <img src="images/Analysis.png" alt="Analysis Process" width="800"/>
</div>

MemoTag follows a sophisticated multi-stage analysis process:

1. **Audio Preprocessing**: Noise reduction, segmentation, and enhancement
2. **Speech-to-Text Conversion**: Accurate transcription using state-of-the-art engines
3. **Feature Extraction**:
   - **Linguistic Features**: Hesitation frequency, word retrieval issues, vocabulary richness
   - **Acoustic Features**: Pitch variation, speech clarity, vocal tremor
   - **Temporal Features**: Speech rate, pause patterns, response latency
4. **Machine Learning Analysis**:
   - Clustering for pattern identification
   - Anomaly detection for cognitive marker identification
   - Feature importance analysis
5. **Visualization & Reporting**: Comprehensive assessment with detailed visualizations

## 🚀 Quick Start

For the fastest setup, use our scripted installers which handle all environment setup automatically:

### Windows
```bash
run_app.bat
```

### macOS/Linux
```bash
chmod +x run_app.sh
./run_app.sh
```

<details>
<summary><b>📋 What these scripts do</b></summary>

1. Set up required environment variables
2. Prompt for OpenAI API key (optional)
3. Activate or create Python virtual environment
4. Download sample audio files for testing
5. Set up NLTK linguistic resources
6. Launch the application with enhanced error handling

</details>

## 🔬 Sample Analysis

<div align="center">
  <img src="images/Features.png" alt="Feature Analysis" width="700"/>
  <p><em>Extracted voice features showing potential cognitive markers</em></p>
</div>

## 📈 Diagnostic Tooling

MemoTag includes comprehensive diagnostic tools to ensure system health:

```bash
# Run system diagnostics
./run_diagnostics.sh   # Linux/macOS
run_diagnostics.bat    # Windows

# Test Streamlit setup independently
python -m streamlit run test_streamlit.py
```

The diagnostic suite checks:
- Python environment and dependencies
- System requirements (FFmpeg, etc.)
- NLTK data availability
- API configuration
- Database connections
- Audio processing capabilities

## 🛠️ Detailed Setup Guide

<details>
<summary><b>1. Environment Setup</b></summary>

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```
</details>

<details>
<summary><b>2. Install Dependencies</b></summary>

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
```

Key dependencies include:
- `streamlit`: Web application framework
- `openai`: For Whisper API integration
- `vosk`: For offline speech recognition
- `librosa`: Audio processing
- `nltk`: Natural language processing
- `scikit-learn`: Machine learning algorithms
- `matplotlib` & `seaborn`: Data visualization
- `sqlalchemy`: Database ORM
</details>

<details>
<summary><b>3. Install FFmpeg</b></summary>

FFmpeg is required for audio processing:

#### Windows
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (Get the "release build")
2. Extract the downloaded zip file
3. Add the bin folder to your PATH environment variable:
   - Search for "Environment Variables" in Windows search
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the path to the bin folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```
</details>

<details>
<summary><b>4. Download Sample Audio</b></summary>

```bash
python download_sample_audio.py
```

This will download a sample MP3 file to the `assets` folder which you can use to test the system.
</details>

<details>
<summary><b>5. NLTK Data</b></summary>

```bash
python download_nltk_data.py
```

This script downloads the required NLTK data packages for linguistic analysis.
</details>

<details>
<summary><b>6. Environment Variables</b></summary>

#### Database Configuration
```bash
# For development with SQLite (default fallback if not set)
# No action needed - will use SQLite by default

# For PostgreSQL (recommended for production)
# Windows
set DATABASE_URL=postgresql://username:password@localhost:5432/memotag
# macOS/Linux
export DATABASE_URL=postgresql://username:password@localhost:5432/memotag
```

#### OpenAI API Key (required for Whisper API)
```bash
# Windows
set OPENAI_API_KEY=your_openai_api_key_here
# macOS/Linux
export OPENAI_API_KEY=your_openai_api_key_here
```

> **Note**: If you don't provide an OpenAI API key, the system will fall back to using Vosk for offline speech recognition.
</details>

<details>
<summary><b>7. Manual Application Startup</b></summary>

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.
</details>

## 📊 Cognitive Assessment Reports

MemoTag generates comprehensive cognitive assessment reports that can be saved, shared, and compared over time.

<div align="center">
  <img src="images/Reccording.png" alt="Recording Analysis" width="700"/>
  <p><em>Voice recording analysis with transcription and feature extraction</em></p>
</div>

The reports include:
- Detailed feature analysis with significance markers
- Anomaly detection results
- Clustering visualizations
- Feature importance rankings
- Risk assessment score

## 💻 System Requirements

- **Python**: 3.9+
- **RAM**: 4GB+ recommended
- **Disk Space**: 500MB for application + dependencies
- **Internet Connection**: Required for Whisper API (optional)
- **Microphone**: Required for direct recording (optional)
- **FFmpeg**: Required for audio processing
- **Processor**: Multi-core recommended for faster analysis

## ⚠️ Troubleshooting

<details>
<summary><b>Common Issues & Solutions</b></summary>

- **Database Connection Issues**: Verify your DATABASE_URL environment variable is correctly set
- **Missing NLTK Data**: Run `python download_nltk_data.py` to download required NLTK resources
- **OpenAI API Errors**: Ensure your OPENAI_API_KEY is valid and has sufficient credits
- **Audio Processing Errors**: Ensure you have FFmpeg installed for audio processing
- **FFmpeg Not Found**: If you see "FFmpeg not available" errors, follow the FFmpeg installation steps in section 3
- **Port Already in Use**: If you see an error that port 8501 is already in use, you can specify a different port with:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Dataframe Display Issues**: If you encounter errors about data types in dataframes, try running with the latest version:
  ```bash
  pip install --upgrade streamlit
  ```
</details>

## 🔍 Diagnostics & Logging

MemoTag includes comprehensive diagnostic and logging capabilities:

- **Application Logs**: Check `memotag_startup.log` for application startup issues
- **Diagnostic Logs**: Check `memotag_diagnostics.log` for system diagnostics
- **Diagnostic Tools**: Run `test_system.py` for complete environment checking
- **Streamlit Test**: Run `test_streamlit.py` to verify Streamlit functionality

## 🚀 Advanced Features

<details>
<summary><b>User Management System</b></summary>

MemoTag includes a complete user management system that allows:
- User registration and authentication
- Saving analyses to user profiles
- Retrieving past analyses
- Comparing changes over time
</details>

<details>
<summary><b>Machine Learning Capabilities</b></summary>

MemoTag's ML system employs:
- **Isolation Forest**: For anomaly detection in voice features
- **K-Means Clustering**: For pattern identification
- **Principal Component Analysis**: For dimensionality reduction
- **Feature Importance Analysis**: For identifying most significant markers
</details>

<details>
<summary><b>Speech-to-Text Options</b></summary>

- **Whisper API** (OpenAI): High-accuracy cloud-based transcription
- **Vosk**: Offline processing for privacy and no-internet scenarios
</details>

## 🧠 Scientific Background

MemoTag analyzes established cognitive markers including:

- Hesitation frequency
- Pause patterns
- Speech rate
- Vocabulary richness
- Word retrieval issues
- Voice quality indicators

Research shows these markers can indicate early cognitive changes before clinical symptoms appear.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- OpenAI for Whisper API integration
- Vosk for offline speech recognition
- Streamlit for the web interface framework
- NLTK for natural language processing capabilities
- scikit-learn for machine learning algorithms
- The scientific community for research on cognitive markers

---

<div align="center">
  <p>🎙️ <b>MemoTag Voice Analysis System</b> — Detecting cognitive changes through voice analysis</p>
  <p><small>Developed with ❤️ by advanced AI systems</small></p>
</div> 
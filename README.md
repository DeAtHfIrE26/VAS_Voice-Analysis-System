# <div align="center"><img src="https://img.shields.io/badge/%F0%9F%8E%99%EF%B8%8F-MemoTag-blue?style=for-the-badge&logo=soundcloud&logoColor=white" alt="MemoTag Voice Analysis System" width="400"/></div>

<div align="center">
  <p>
    <a href="#overview"><img src="https://img.shields.io/badge/%F0%9F%94%8D-Overview-informational?style=flat-square" alt="Overview"/></a>
    <a href="#features"><img src="https://img.shields.io/badge/%E2%9C%A8-Features-success?style=flat-square" alt="Features"/></a>
    <a href="#installation"><img src="https://img.shields.io/badge/%F0%9F%92%BF-Installation-blueviolet?style=flat-square" alt="Installation"/></a>
    <a href="#usage"><img src="https://img.shields.io/badge/%F0%9F%9A%80-Usage-orange?style=flat-square" alt="Usage"/></a>
    <a href="#demo"><img src="https://img.shields.io/badge/%F0%9F%96%A5%EF%B8%8F-Demo-ff69b4?style=flat-square" alt="Demo"/></a>
  </p>
</div>

<div align="center">
  <img src="images/Dashboard.png" alt="MemoTag Dashboard" width="800"/>
  <p><strong><em>Advanced voice analysis platform for early detection of cognitive decline markers</em></strong></p>
</div>

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
    <img src="https://img.shields.io/badge/Python-3.9%2B-brightgreen" alt="Python: 3.9+"/>
    <img src="https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B" alt="Streamlit: 1.28+"/>
    <img src="https://img.shields.io/badge/Status-Beta-yellow" alt="Status: Beta"/>
    <img src="https://img.shields.io/badge/ML-Enabled-8A2BE2" alt="ML: Enabled"/>
    <img src="https://img.shields.io/badge/NLP-Advanced-00FFFF" alt="NLP: Advanced"/>
    <img src="https://img.shields.io/badge/Voice-Analysis-FF6347" alt="Voice: Analysis"/>
  </p>
</div>

<hr>

<div align="center">
  
  ```
  ███╗   ███╗███████╗███╗   ███╗ ██████╗ ████████╗ █████╗  ██████╗ 
  ████╗ ████║██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██╔══██╗██╔════╝ 
  ██╔████╔██║█████╗  ██╔████╔██║██║   ██║   ██║   ███████║██║  ███╗
  ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██╔══██║██║   ██║
  ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝
  ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ 
  ```
  
</div>

<a id="overview"></a>
## 🔍 Overview

<img align="right" src="images/Features.png" width="350"/>

**MemoTag Voice Analysis System** is a groundbreaking application that employs artificial intelligence to analyze vocal patterns and identify early indicators of cognitive decline. This cutting-edge system leverages advances in natural language processing, acoustic analysis, and machine learning to extract and analyze over 30 unique features from voice recordings.

By detecting subtle changes in speech patterns, hesitations, vocabulary usage, and acoustic properties, MemoTag provides healthcare professionals with an invaluable tool for early intervention and monitoring of cognitive health.

The system is designed with both clinical and research applications in mind, offering detailed analysis, visualizations, and comprehensive reporting capabilities that transform voice recordings into actionable cognitive health insights.

<br clear="right"/>

<a id="features"></a>
## ✨ Key Features

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

<details open>
<summary><h3>🔊 Speech Recognition & Analysis</h3></summary>

- **Dual Speech Recognition Engines**:
  - **OpenAI Whisper API**: Cloud-based transcription with unparalleled accuracy
  - **Vosk**: Privacy-focused offline processing (no internet required)
- **Multi-language Support**: Analyze recordings in multiple languages
- **Noise Reduction**: Advanced algorithms for cleaning audio artifacts
- **Automatic Segmentation**: Intelligently divides speech into analyzable segments

</details>

<details>
<summary><h3>🧠 Cognitive Feature Extraction</h3></summary>

**Linguistic Features**:
- Hesitation frequency analysis
- Filler word detection and analysis
- Type-token ratio for vocabulary richness
- Word retrieval difficulty detection
- Uncommon word usage metrics
- Sentence complexity analysis
- Grammatical structure evaluation

**Acoustic Features**:
- Voice tremor analysis
- Pitch variation measurement
- Harmonic-to-noise ratio
- Jitter and shimmer detection
- Vocal energy distribution
- Spectral features analysis

**Temporal Features**:
- Speech rate calculation (words per minute)
- Articulation rate analysis
- Pause pattern recognition
- Response latency measurement
- Speech rhythm analysis

</details>

<details>
<summary><h3>🤖 Machine Learning Analysis</h3></summary>

- **Unsupervised Anomaly Detection**: Identifies deviations from typical speech patterns
- **Clustering Analysis**: Groups similar speech features for pattern identification
- **Feature Importance Ranking**: Highlights most significant cognitive markers
- **Risk Score Calculation**: Provides quantitative assessment of potential cognitive concerns
- **Temporal Analysis**: Tracks changes in speech patterns over time
- **Comparative Analysis**: Benchmarks results against baseline data

</details>

<details>
<summary><h3>📊 Visualization & Reporting</h3></summary>

- **Interactive Dashboards**: Dynamic visualization of key metrics
- **Feature Distribution Plots**: Visual representation of linguistic and acoustic patterns
- **Correlation Analysis**: Identifies relationships between different speech features
- **Anomaly Highlighting**: Visual indicators of potential cognitive markers
- **Comprehensive HTML Reports**: Professional documentation for healthcare providers
- **Export Capabilities**: Save and share results in multiple formats
- **Time-Series Tracking**: Monitor changes across multiple assessments

</details>

<details>
<summary><h3>💾 User & Data Management</h3></summary>

- **User Authentication**: Secure login system
- **Profile Management**: Create and manage multiple user profiles
- **Historical Data Access**: Review and compare past analyses
- **Data Export/Import**: Flexible data transfer options
- **Privacy Controls**: Configure data sharing and storage preferences
- **Database Integration**: Works with SQLite (default) or PostgreSQL (for production)

</details>

<details>
<summary><h3>🛠️ System Integration</h3></summary>

- **Robust Error Handling**: Gracefully manages failures with detailed diagnostics
- **Comprehensive Logging**: Detailed activity and error logging
- **Diagnostic Tools**: Advanced system health checking utilities
- **Flexible Deployment**: Run locally or on a server
- **Cross-platform Support**: Works on Windows, macOS, and Linux
- **API Integration**: Easy integration with OpenAI and other services

</details>

<a id="workflow"></a>
## 📊 Advanced Analysis Workflow

<div align="center">
  <img src="images/Analysis.png" alt="Analysis Process" width="800"/>
</div>

MemoTag employs a sophisticated multi-stage analysis process designed by cognitive neuroscientists and speech pathologists:

<div align="center">

| Stage | Process | Description |
|:-----:|:-------:|:------------|
| 1️⃣ | **Audio Preprocessing** | Noise reduction, normalization, segmentation, and enhancement |
| 2️⃣ | **Speech-to-Text Conversion** | High-accuracy transcription using state-of-the-art engines |
| 3️⃣ | **Feature Extraction** | Analysis of 30+ linguistic, acoustic, and temporal features |
| 4️⃣ | **Machine Learning Analysis** | Pattern recognition, anomaly detection, and feature importance |
| 5️⃣ | **Visualization & Reporting** | Comprehensive assessment with interactive visualizations |

</div>

<a id="installation"></a>
## 💿 Installation & Setup

<a id="quick-start"></a>
### 🚀 Quick Start

For the fastest setup experience, use our scripted installers:

<div align="center">
  <table>
    <tr>
      <th>Windows</th>
      <th>macOS/Linux</th>
    </tr>
    <tr>
      <td>
        <pre><code>run_app.bat</code></pre>
      </td>
      <td>
        <pre><code>chmod +x run_app.sh
./run_app.sh</code></pre>
      </td>
    </tr>
  </table>
</div>

<details>
<summary><b>📋 What these scripts do</b></summary>

1. Set up required environment variables
2. Prompt for OpenAI API key (optional)
3. Activate or create Python virtual environment
4. Download sample audio files for testing
5. Set up NLTK linguistic resources
6. Launch the application with enhanced error handling

</details>

<a id="detailed-setup"></a>
### 🛠️ Detailed Setup Guide

<details>
<summary><h3>📂 1. Environment Setup</h3></summary>

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
<summary><h3>📦 2. Install Dependencies</h3></summary>

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
```

<div align="center">

| Category | Key Dependencies |
|----------|------------------|
| **Web Framework** | `streamlit>=1.28.0` |
| **AI/ML** | `openai>=1.3.0`, `scikit-learn>=1.2.0` |
| **NLP** | `nltk>=3.8.1`, `vosk>=0.3.45` |
| **Audio Processing** | `librosa>=0.11.0`, `soundfile>=0.12.1` |
| **Data Analysis** | `pandas>=2.0.0`, `numpy>=1.24.0` |
| **Visualization** | `matplotlib>=3.7.0`, `seaborn>=0.12.0` |
| **Database** | `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.5` |

</div>
</details>

<details>
<summary><h3>🎛️ 3. Install FFmpeg</h3></summary>

FFmpeg is required for audio processing:

<table>
<tr>
<td>Windows</td>
<td>
1. Download FFmpeg from <a href="https://ffmpeg.org/download.html">ffmpeg.org/download.html</a> (Get the "release build")<br>
2. Extract the downloaded zip file<br>
3. Add the bin folder to your PATH environment variable:<br>
   - Search for "Environment Variables" in Windows search<br>
   - Click "Edit the system environment variables"<br>
   - Click "Environment Variables"<br>
   - Under "System variables", find "Path" and click "Edit"<br>
   - Click "New" and add the path to the bin folder (e.g., <code>C:\ffmpeg\bin</code>)<br>
   - Click "OK" on all dialogs
</td>
</tr>
<tr>
<td>macOS</td>
<td>

```bash
brew install ffmpeg
```

</td>
</tr>
<tr>
<td>Linux</td>
<td>

```bash
sudo apt update
sudo apt install ffmpeg
```

</td>
</tr>
</table>

</details>

<details>
<summary><h3>🔈 4. Download Sample Audio</h3></summary>

```bash
python download_sample_audio.py
```

This will download a sample MP3 file to the `assets` folder which you can use to test the system.
</details>

<details>
<summary><h3>🧮 5. NLTK Data</h3></summary>

```bash
python download_nltk_data.py
```

This script downloads the required NLTK data packages for linguistic analysis.
</details>

<details>
<summary><h3>⚙️ 6. Environment Variables</h3></summary>

<table>
<tr>
<th colspan="2">Database Configuration</th>
</tr>
<tr>
<td>Development<br>(SQLite)</td>
<td>No action needed - will use SQLite by default</td>
</tr>
<tr>
<td>Production<br>(PostgreSQL)</td>
<td>
<b>Windows:</b><br>

```
set DATABASE_URL=postgresql://username:password@localhost:5432/memotag
```

<b>macOS/Linux:</b><br>

```
export DATABASE_URL=postgresql://username:password@localhost:5432/memotag
```
</td>
</tr>
<tr>
<th colspan="2">OpenAI API Key (required for Whisper API)</th>
</tr>
<tr>
<td>Windows</td>
<td>

```
set OPENAI_API_KEY=your_openai_api_key_here
```

</td>
</tr>
<tr>
<td>macOS/Linux</td>
<td>

```
export OPENAI_API_KEY=your_openai_api_key_here
```

</td>
</tr>
</table>

> **Note**: If you don't provide an OpenAI API key, the system will fall back to using Vosk for offline speech recognition.
</details>

<details>
<summary><h3>▶️ 7. Manual Application Startup</h3></summary>

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.
</details>

<a id="usage"></a>
## 📝 Usage Guide

<a id="diagnostics"></a>
### 📈 Diagnostic Tools

MemoTag includes comprehensive diagnostic tools to ensure system health:

```bash
# Run system diagnostics
./run_diagnostics.sh   # Linux/macOS
run_diagnostics.bat    # Windows

# Test Streamlit setup independently
python -m streamlit run test_streamlit.py
```

The diagnostic suite performs these checks:
- Python environment and dependencies
- System requirements (FFmpeg, etc.)
- NLTK data availability
- API configuration
- Database connections
- Audio processing capabilities

<a id="demo"></a>
## 🖼️ Application Showcase

<a id="reports"></a>
### 📊 Cognitive Assessment Reports

<div align="center">
  <img src="images/Reccording.png" alt="Recording Analysis" width="700"/>
  <p><em>Voice recording analysis with transcription and feature extraction</em></p>
</div>

MemoTag generates comprehensive cognitive assessment reports featuring:

- **Feature Analysis**: Detailed breakdown of 30+ voice features with significance markers
- **Anomaly Detection**: Identification of potential cognitive markers with confidence scores
- **Visualization Suite**: Interactive charts showing feature distributions and correlations
- **Risk Assessment**: Quantitative scores with clinical context for healthcare providers
- **Historical Comparison**: Tracking of changes over multiple assessments (when available)
- **Recommendation Engine**: Suggestions for further evaluation or monitoring

<a id="architecture"></a>
## 🏗️ System Architecture

<div align="center">
<pre>
┌───────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                       │
│                      (Streamlit Web Application)                   │
└───────────────┬───────────────────────────────────┬───────────────┘
                │                                   │
┌───────────────▼───────────────┐     ┌─────────────▼───────────────┐
│      Audio Processing Layer    │     │       Analysis Layer        │
│ ┌───────────────────────────┐ │     │ ┌───────────────────────┐   │
│ │ Audio Loading & Validation │ │     │ │ Feature Extraction    │   │
│ └─────────────┬─────────────┘ │     │ └───────────┬───────────┘   │
│               │               │     │             │               │
│ ┌─────────────▼─────────────┐ │     │ ┌───────────▼───────────┐   │
│ │  Noise Reduction & Prep   │ │     │ │ Machine Learning      │   │
│ └─────────────┬─────────────┘ │     │ │  - Anomaly Detection  │   │
│               │               │     │ │  - Feature Importance │   │
│ ┌─────────────▼─────────────┐ │     │ └───────────┬───────────┘   │
│ │    Speech Recognition     │ │     │             │               │
│ │   (Whisper API / Vosk)    │─┼─────┼─────────────┘               │
│ └───────────────────────────┘ │     │ ┌─────────────────────────┐ │
│                               │     │ │ Visualization & Reporting│ │
└───────────────────────────────┘     │ └─────────────┬───────────┘ │
                                      └─────────────────────────────┘
                                                     │
┌────────────────────────────────────────────────────▼──────────────┐
│                         Data Storage Layer                         │
│ ┌────────────────────┐    ┌────────────────┐   ┌────────────────┐ │
│ │   User Profiles    │    │ Analysis Data  │   │ Report Storage │ │
│ └────────────────────┘    └────────────────┘   └────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
</pre>
</div>

<a id="requirements"></a>
## 💻 System Requirements

<div align="center">

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.9+ |
| **RAM** | 4GB+ recommended |
| **Disk Space** | 500MB for application + dependencies |
| **Internet Connection** | Required for Whisper API (optional) |
| **Microphone** | Required for direct recording (optional) |
| **FFmpeg** | Required for audio processing |
| **Processor** | Multi-core recommended for faster analysis |

</div>

<a id="troubleshooting"></a>
## ⚠️ Troubleshooting

<details>
<summary><b>Common Issues & Solutions</b></summary>

<table>
<tr>
<th>Issue</th>
<th>Solution</th>
</tr>
<tr>
<td>Database Connection Issues</td>
<td>Verify your DATABASE_URL environment variable is correctly set</td>
</tr>
<tr>
<td>Missing NLTK Data</td>
<td>Run <code>python download_nltk_data.py</code> to download required NLTK resources</td>
</tr>
<tr>
<td>OpenAI API Errors</td>
<td>Ensure your OPENAI_API_KEY is valid and has sufficient credits</td>
</tr>
<tr>
<td>Audio Processing Errors</td>
<td>Ensure you have FFmpeg installed for audio processing</td>
</tr>
<tr>
<td>FFmpeg Not Found</td>
<td>Follow the FFmpeg installation steps in <a href="#detailed-setup">section 3</a></td>
</tr>
<tr>
<td>Port Already in Use</td>
<td>
Run with a different port:

```bash
streamlit run app.py --server.port 8502
```
</td>
</tr>
<tr>
<td>Dataframe Display Issues</td>
<td>
Update Streamlit:

```bash
pip install --upgrade streamlit
```
</td>
</tr>
</table>

</details>

<a id="logging"></a>
## 🔍 Diagnostics & Logging

MemoTag includes comprehensive diagnostic and logging capabilities:

- **Application Logs**: Check `memotag_startup.log` for application startup issues
- **Diagnostic Logs**: Check `memotag_diagnostics.log` for system diagnostics
- **Diagnostic Tools**: Run `test_system.py` for complete environment checking
- **Streamlit Test**: Run `test_streamlit.py` to verify Streamlit functionality

<a id="advanced"></a>
## 🚀 Advanced Features

<details>
<summary><h3>👤 User Management System</h3></summary>

MemoTag includes a complete user management system that allows:
- User registration and authentication
- Saving analyses to user profiles
- Retrieving past analyses
- Comparing changes over time
</details>

<details>
<summary><h3>🤖 Machine Learning Capabilities</h3></summary>

MemoTag's ML system employs:
- **Isolation Forest**: For anomaly detection in voice features
- **K-Means Clustering**: For pattern identification
- **Principal Component Analysis**: For dimensionality reduction
- **Feature Importance Analysis**: For identifying most significant markers
</details>

<details>
<summary><h3>🔊 Speech-to-Text Options</h3></summary>

- **Whisper API** (OpenAI): High-accuracy cloud-based transcription
- **Vosk**: Offline processing for privacy and no-internet scenarios
</details>

<a id="science"></a>
## 🧠 Scientific Background

<div align="center">
  <table>
    <tr>
      <th colspan="2">Key Cognitive Markers Analyzed</th>
    </tr>
    <tr>
      <td><strong>Marker</strong></td>
      <td><strong>Relevance to Cognitive Function</strong></td>
    </tr>
    <tr>
      <td>Hesitation frequency</td>
      <td>Executive function, processing speed</td>
    </tr>
    <tr>
      <td>Pause patterns</td>
      <td>Working memory, attention</td>
    </tr>
    <tr>
      <td>Speech rate</td>
      <td>Motor control, cognitive processing</td>
    </tr>
    <tr>
      <td>Vocabulary richness</td>
      <td>Semantic memory, language function</td>
    </tr>
    <tr>
      <td>Word retrieval issues</td>
      <td>Language processing, memory access</td>
    </tr>
    <tr>
      <td>Voice quality indicators</td>
      <td>Neuromotor control, emotional regulation</td>
    </tr>
  </table>
</div>

Research shows these markers can indicate early cognitive changes before clinical symptoms appear, providing a valuable window for early intervention.

<a id="license"></a>
## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

<a id="contributing"></a>
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

- OpenAI for Whisper API integration
- Vosk for offline speech recognition
- Streamlit for the web interface framework
- NLTK for natural language processing capabilities
- scikit-learn for machine learning algorithms
- The scientific community for research on cognitive markers

---

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made with Python">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained? yes">
    <img src="https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg" alt="Ask me anything">
  </p>
  
  <p>🎙️ <b>MemoTag Voice Analysis System</b> — Detecting cognitive changes through voice analysis</p>
  <p><small>Developed with ❤️ for advancing cognitive health research and care</small></p>
  
  <a href="#overview">
    <img src="https://img.shields.io/badge/Back%20to%20Top-↑-lightgrey" alt="Back to Top">
  </a>
</div> 
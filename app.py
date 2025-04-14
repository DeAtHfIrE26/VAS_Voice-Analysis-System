import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from io import BytesIO
import nltk

# Initialize NLTK resources with our enhanced download process
try:
    from download_nltk_data import ensure_nltk_data
    ensure_nltk_data()
except Exception as e:
    st.warning(f"Error initializing NLTK resources: {str(e)}. Using fallback mechanisms for linguistic processing.")

# Import custom modules
from modules.audio_preprocessing import preprocess_audio, get_audio_info
from modules.speech_to_text import transcribe_audio
from modules.feature_extraction import (
    extract_linguistic_features,
    extract_acoustic_features,
    extract_temporal_features
)
from modules.machine_learning import (
    perform_clustering,
    detect_anomalies,
    analyze_feature_importance
)
from modules.visualization import (
    plot_feature_distributions,
    plot_anomaly_detection,
    plot_feature_correlation,
    plot_clustering_results
)
from modules.reporting import generate_report
from modules.database import (
    get_user_by_username,
    create_user,
    save_analysis,
    get_analyses_by_user,
    get_analysis_by_id,
    get_all_analyses
)
from utils.helpers import get_supported_formats, download_file

# Page configuration
st.set_page_config(
    page_title="MemoTag Voice Analysis System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'report_html' not in st.session_state:
    st.session_state.report_html = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'view' not in st.session_state:
    st.session_state.view = "main"  # Options: main, login, register, history, view_analysis

# App title and description
st.title("MemoTag Voice Analysis System")
st.markdown("""
This application analyzes voice data to detect early signs of cognitive decline using 
NLP and audio processing techniques. Upload an audio file to begin the analysis.
""")

# Authentication UI
auth_placeholder = st.empty()

with auth_placeholder.container():
    # Login/logout section in the main area
    if not st.session_state.logged_in:
        login_cols = st.columns([1, 1, 3])
        
        with login_cols[0]:
            if st.button("Login"):
                st.session_state.view = "login"
                st.experimental_rerun()
        
        with login_cols[1]:
            if st.button("Register"):
                st.session_state.view = "register"
                st.experimental_rerun()
    else:
        login_cols = st.columns([1, 1, 1, 2])
        
        with login_cols[0]:
            st.write(f"Logged in as: **{st.session_state.user.username}**")
        
        with login_cols[1]:
            if st.button("My Analyses"):
                st.session_state.view = "history"
                st.experimental_rerun()
        
        with login_cols[2]:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.view = "main"
                st.experimental_rerun()
                
# Different views based on session state
if st.session_state.view == "login":
    with st.container():
        st.header("Login")
        username = st.text_input("Username")
        
        login_button = st.button("Login", key="login_button")
        
        if login_button:
            if username:
                user = get_user_by_username(username)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.session_state.view = "main"
                    st.success(f"Welcome back, {username}!")
                    st.experimental_rerun()
                else:
                    st.error(f"User '{username}' not found. Please register first.")
            else:
                st.warning("Please enter a username.")
                
        if st.button("Back to Main", key="login_back"):
            st.session_state.view = "main"
            st.experimental_rerun()
            
elif st.session_state.view == "register":
    with st.container():
        st.header("Register New User")
        username = st.text_input("Username")
        email = st.text_input("Email")
        
        register_button = st.button("Register", key="register_button")
        
        if register_button:
            if username and email:
                existing_user = get_user_by_username(username)
                if existing_user:
                    st.error(f"Username '{username}' already exists. Please choose another username.")
                else:
                    user = create_user(username, email)
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.session_state.view = "main"
                    st.success(f"Welcome, {username}! Your account has been created successfully.")
                    st.experimental_rerun()
            else:
                st.warning("Please enter both username and email.")
                
        if st.button("Back to Main", key="register_back"):
            st.session_state.view = "main"
            st.experimental_rerun()
            
elif st.session_state.view == "history":
    with st.container():
        st.header("My Analysis History")
        
        if not st.session_state.logged_in:
            st.warning("Please login to view your analysis history.")
            if st.button("Go to Login", key="history_login"):
                st.session_state.view = "login"
                st.experimental_rerun()
        else:
            # Get analyses for the logged-in user
            analyses = get_analyses_by_user(st.session_state.user.id)
            
            if not analyses:
                st.info("You don't have any saved analyses yet.")
            else:
                # Display analyses in a table
                analysis_data = []
                for analysis in analyses:
                    analysis_data.append({
                        "ID": analysis.id,
                        "Title": analysis.title,
                        "Date": analysis.created_at.strftime("%Y-%m-%d %H:%M"),
                        "Duration (s)": f"{analysis.audio_duration:.2f}",
                        "Risk Score": f"{analysis.risk_score:.2f}" if analysis.risk_score is not None else "N/A"
                    })
                
                df = pd.DataFrame(analysis_data)
                st.dataframe(df)
                
                # Select analysis to view
                selected_id = st.selectbox("Select an analysis to view", [a.id for a in analyses], format_func=lambda x: f"Analysis #{x}")
                
                if st.button("View Selected Analysis", key="view_analysis"):
                    st.session_state.view = "view_analysis"
                    st.session_state.selected_analysis_id = selected_id
                    st.experimental_rerun()
        
        if st.button("Back to Main", key="history_back"):
            st.session_state.view = "main"
            st.experimental_rerun()
            
elif st.session_state.view == "view_analysis":
    with st.container():
        if not hasattr(st.session_state, 'selected_analysis_id'):
            st.error("No analysis selected.")
            if st.button("Back to History", key="no_analysis_back"):
                st.session_state.view = "history"
                st.experimental_rerun()
        else:
            # Get the selected analysis
            analysis = get_analysis_by_id(st.session_state.selected_analysis_id)
            
            if not analysis:
                st.error("Analysis not found.")
                if st.button("Back to History", key="not_found_back"):
                    st.session_state.view = "history"
                    st.experimental_rerun()
            else:
                st.header(f"Analysis: {analysis.title}")
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Date", analysis.created_at.strftime("%Y-%m-%d"))
                with col2:
                    st.metric("Duration", f"{analysis.audio_duration:.2f}s")
                with col3:
                    st.metric("Risk Score", f"{analysis.risk_score:.2f}" if analysis.risk_score is not None else "N/A")
                
                # Display report if available
                if analysis.report_html:
                    st.subheader("Report")
                    st.components.v1.html(analysis.report_html, height=600)
                    
                    # Download report button
                    report_bytes = BytesIO(analysis.report_html.encode('utf-8'))
                    st.download_button(
                        label="Download Report (HTML)",
                        data=report_bytes,
                        file_name=f"analysis_{analysis.id}_report.html",
                        mime="text/html"
                    )
                
                if st.button("Back to History", key="view_back"):
                    st.session_state.view = "history"
                    st.experimental_rerun()

# Sidebar for file upload and options
with st.sidebar:
    st.header("Upload Audio")
    
    supported_formats = get_supported_formats()
    formats_str = ", ".join(supported_formats)
    
    uploaded_file = st.file_uploader(
        f"Upload an audio file ({formats_str})",
        type=supported_formats
    )
    
    # Audio preprocessing options
    st.header("Preprocessing Options")
    apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=True)
    
    # Speech recognition options
    st.header("Speech Recognition")
    recognition_engine = st.selectbox(
        "Recognition Engine",
        options=["Whisper", "Vosk"],
        index=0
    )
    
    # Analysis options
    st.header("Analysis Options")
    st.subheader("Features to Extract")
    extract_linguistic = st.checkbox("Linguistic Features", value=True)
    extract_acoustic = st.checkbox("Acoustic Features", value=True)
    extract_temporal = st.checkbox("Temporal Features", value=True)
    
    st.subheader("Machine Learning")
    perform_clustering_option = st.checkbox("Perform Clustering", value=True)
    detect_anomalies_option = st.checkbox("Detect Anomalies", value=True)
    analyze_importance = st.checkbox("Analyze Feature Importance", value=True)
    
    # Process button
    process_button = st.button("Process Audio")

# Main content area
if uploaded_file is not None:
    # Save uploaded file to session state
    st.session_state.audio_file = uploaded_file
    
    # Display audio info
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name
    
    audio_info = get_audio_info(audio_path)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{audio_info['duration']:.2f}s")
    with col2:
        st.metric("Sample Rate", f"{audio_info['sample_rate']} Hz")
    with col3:
        st.metric("File Format", audio_info['format'])
    
    # Audio player
    st.audio(uploaded_file, format=f"audio/{audio_info['format']}")
    
    # Process the audio when button is clicked
    if process_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocess audio
        status_text.text("Preprocessing audio...")
        preprocessed_audio = preprocess_audio(
            audio_path, 
            apply_noise_reduction_param=apply_noise_reduction
        )
        st.session_state.audio_data = preprocessed_audio
        progress_bar.progress(20)
        
        # Step 2: Transcribe audio
        status_text.text("Transcribing audio...")
        transcription = transcribe_audio(
            preprocessed_audio,
            engine=recognition_engine.lower()
        )
        st.session_state.transcription = transcription
        progress_bar.progress(40)
        
        # Check if transcription has an error
        if transcription.get("transcription_error", False):
            status_text.text("Transcription failed. See details in the Transcription section.")
            progress_bar.progress(100)
            
            # Clean up temporary file
            os.unlink(audio_path)
            
            # Skip further processing by setting a flag and using else clause
            transcription_failed = True
        else:
            # Only execute the rest of the processing if transcription succeeded
            transcription_failed = False
            
            # Step 3: Extract features
            status_text.text("Extracting features...")
            features = {}
            
            if extract_linguistic:
                linguistic_features = extract_linguistic_features(transcription)
                features.update(linguistic_features)
            
            if extract_acoustic:
                acoustic_features = extract_acoustic_features(preprocessed_audio)
                features.update(acoustic_features)
            
            if extract_temporal:
                temporal_features = extract_temporal_features(
                    preprocessed_audio, 
                    transcription
                )
                features.update(temporal_features)
            
            st.session_state.features = features
            progress_bar.progress(60)
            
            # Step 4: Machine learning analysis
            status_text.text("Analyzing features...")
            analysis_results = {}
            
            if perform_clustering_option:
                clustering_results = perform_clustering(features)
                analysis_results['clustering'] = clustering_results
            
            if detect_anomalies_option:
                anomaly_results = detect_anomalies(features)
                analysis_results['anomalies'] = anomaly_results
            
            if analyze_importance:
                importance_results = analyze_feature_importance(features)
                analysis_results['importance'] = importance_results
            
            st.session_state.analysis_results = analysis_results
            progress_bar.progress(80)
            
            # Step 5: Generate report
            status_text.text("Generating report...")
            report_html = generate_report(
                audio_info,
                transcription,
                features,
                analysis_results
            )
            st.session_state.report_html = report_html
            progress_bar.progress(100)
            
            status_text.text("Analysis complete!")
            
            # Save to database if user is logged in
            if st.session_state.logged_in:
                # Ask if the user wants to save the analysis
                save_dialog = st.empty()
                with save_dialog.container():
                    st.success("Analysis complete! Would you like to save it to your account?")
                    save_cols = st.columns(2)
                    
                    with save_cols[0]:
                        analysis_title = st.text_input("Analysis Title", value=f"Analysis {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    
                    with save_cols[1]:
                        analysis_description = st.text_input("Description (optional)")
                    
                    save_button = st.button("Save Analysis")
                    if save_button:
                        if analysis_title:
                            # Save to database
                            analysis = save_analysis(
                                st.session_state.user.id,
                                analysis_title,
                                audio_info,
                                transcription,
                                features,
                                analysis_results,
                                report_html,
                                description=analysis_description
                            )
                            
                            st.session_state.save_dialog_shown = False
                            save_dialog.empty()
                            st.success(f"Analysis saved successfully with ID: {analysis.id}")
                        else:
                            st.warning("Please enter a title for your analysis.")
        
        # Clean up temporary file (whether transcription failed or not)
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")
                
        # Rerun to display results
        st.experimental_rerun()

# Display results if available
if st.session_state.transcription is not None:
    st.header("Transcription")
    
    # Check if transcription has an error
    if st.session_state.transcription.get("transcription_error", False):
        st.error("Transcription failed. See details below.")
        st.warning(st.session_state.transcription.get("text", "Unknown error"))
        
        # Display error details if available
        if "error_details" in st.session_state.transcription:
            with st.expander("Error Details"):
                st.code(st.session_state.transcription["error_details"])
        
        # Show troubleshooting tips
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            ### Common Solutions:
            
            1. **OpenAI API Key Issues**:
               - Check that your OpenAI API key is valid
               - Make sure you have credits available on your OpenAI account
               - Try using Vosk instead (select from the sidebar)
               
            2. **Audio File Issues**:
               - Try a different audio file format (e.g., WAV instead of MP3)
               - Ensure the audio file contains clear speech
               - Check that FFmpeg is installed if using non-WAV formats
               
            3. **Connection Issues**:
               - Check your internet connection if using Whisper API
               - Vosk works offline and doesn't require internet
            """)
    else:
        # Display normal transcription
        st.markdown(f"```\n{st.session_state.transcription['text']}\n```")
        
        # Only create tabs if transcription was successful and we have data to show
        # Create tabs for different sections
        tabs = st.tabs(["Features", "Visualizations", "Analysis Results", "Report"])
    
        # Features tab
        with tabs[0]:
            if st.session_state.features:
                st.header("Extracted Features")
                
                # Convert features to DataFrame for display
                features_df = pd.DataFrame({
                    "Feature": list(st.session_state.features.keys()),
                    "Value": list(st.session_state.features.values())
                })
                
                # Group features by category
                linguistic_features = features_df[features_df["Feature"].str.startswith("linguistic_")]
                acoustic_features = features_df[features_df["Feature"].str.startswith("acoustic_")]
                temporal_features = features_df[features_df["Feature"].str.startswith("temporal_")]
                
                # Create columns for different feature types
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Linguistic Features")
                    if not linguistic_features.empty:
                        st.dataframe(linguistic_features)
                    else:
                        st.info("No linguistic features extracted")
                
                with col2:
                    st.subheader("Acoustic Features")
                    if not acoustic_features.empty:
                        st.dataframe(acoustic_features)
                    else:
                        st.info("No acoustic features extracted")
                
                with col3:
                    st.subheader("Temporal Features")
                    if not temporal_features.empty:
                        st.dataframe(temporal_features)
                    else:
                        st.info("No temporal features extracted")
        
        # Visualizations tab
        with tabs[1]:
            if st.session_state.features and st.session_state.analysis_results:
                st.header("Visualizations")
                
                # Feature distributions
                st.subheader("Feature Distributions")
                feature_dist_fig = plot_feature_distributions(st.session_state.features)
                st.pyplot(feature_dist_fig)
                
                # Feature correlations
                st.subheader("Feature Correlations")
                corr_fig = plot_feature_correlation(st.session_state.features)
                st.pyplot(corr_fig)
                
                # Anomaly detection if available
                if 'anomalies' in st.session_state.analysis_results:
                    st.subheader("Anomaly Detection")
                    anomaly_fig = plot_anomaly_detection(
                        st.session_state.features,
                        st.session_state.analysis_results['anomalies']
                    )
                    st.pyplot(anomaly_fig)
                
                # Clustering results if available
                if 'clustering' in st.session_state.analysis_results:
                    st.subheader("Clustering Results")
                    cluster_fig = plot_clustering_results(
                        st.session_state.features,
                        st.session_state.analysis_results['clustering']
                    )
                    st.pyplot(cluster_fig)
        
        # Analysis Results tab
        with tabs[2]:
            if st.session_state.analysis_results:
                st.header("Analysis Results")
                
                # Display potential cognitive markers identified
                if 'anomalies' in st.session_state.analysis_results:
                    st.subheader("Potential Cognitive Markers")
                    
                    anomalies = st.session_state.analysis_results['anomalies']
                    if 'markers' in anomalies and anomalies['markers']:
                        # Create a properly structured DataFrame with consistent types
                        markers_data = []
                        for marker_name, marker_info in anomalies['markers'].items():
                            # Ensure consistent data types for each column
                            row = {
                                "Marker": marker_name,
                                "Anomaly Score": float(marker_info.get("anomaly_score", 0)),
                                "Is Anomaly": bool(marker_info.get("is_outlier", False)),
                                "Features Used": ", ".join(marker_info.get("features_used", []))
                            }
                            markers_data.append(row)
                        
                        # Create DataFrame with explicitly defined data types
                        markers_df = pd.DataFrame(markers_data)
                        st.dataframe(markers_df)
                    else:
                        st.info("No significant cognitive markers identified")
                
                # Display feature importance if available
                if 'importance' in st.session_state.analysis_results:
                    st.subheader("Feature Importance")
                    
                    importance = st.session_state.analysis_results['importance']
                    if 'scores' in importance:
                        importance_df = pd.DataFrame({
                            "Feature": list(importance['scores'].keys()),
                            "Importance Score": list(importance['scores'].values())
                        }).sort_values(by="Importance Score", ascending=False)
                        
                        st.dataframe(importance_df)
                        
                        # Bar chart of feature importance
                        st.bar_chart(importance_df.set_index("Feature"))
                    else:
                        st.info("Feature importance analysis not available")
        
        # Report tab
        with tabs[3]:
            if st.session_state.report_html:
                st.header("Cognitive Assessment Report")
                
                # Display HTML report
                st.components.v1.html(st.session_state.report_html, height=600)
                
                # Download report button
                report_bytes = BytesIO(st.session_state.report_html.encode('utf-8'))
                st.download_button(
                    label="Download Report (HTML)",
                    data=report_bytes,
                    file_name="cognitive_assessment_report.html",
                    mime="text/html"
                )

# About section
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    """
    This application is designed to analyze voice data for early detection 
    of cognitive impairment indicators. It extracts linguistic, acoustic, 
    and temporal features from audio samples and uses machine learning 
    techniques to identify potential markers of cognitive decline.
    
    **Features:**
    - Audio preprocessing and enhancement
    - Speech-to-text conversion
    - Comprehensive feature extraction
    - Unsupervised machine learning analysis
    - Visualization and reporting tools
    """
)

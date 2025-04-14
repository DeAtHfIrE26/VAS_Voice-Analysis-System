#!/usr/bin/env python
"""
Test script for Streamlit functionality.
This creates a minimal Streamlit app to verify that the basic setup is working.
"""

import streamlit as st
import os
import sys
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Run a minimal Streamlit app to test functionality."""
    # Page setup
    st.set_page_config(
        page_title="MemoTag Test App",
        page_icon="🧪",
        layout="wide"
    )
    
    # Header
    st.title("MemoTag Voice Analysis System - Test App")
    st.markdown("This is a test app to verify that Streamlit is working correctly.")
    
    # Environment information
    st.header("Environment Information")
    
    # System info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System")
        system_info = {
            "Python Version": sys.version.split()[0],
            "Operating System": platform.platform(),
            "Streamlit Version": st.__version__,
            "Working Directory": os.getcwd()
        }
        st.json(system_info)
    
    # Environment variables
    with col2:
        st.subheader("Environment Variables")
        env_vars = {
            "DATABASE_URL": os.environ.get("DATABASE_URL", "Not set"),
            "OPENAI_API_KEY": "Set" if os.environ.get("OPENAI_API_KEY") else "Not set"
        }
        st.json(env_vars)
    
    # Test data generation
    st.header("Data Generation Test")
    
    # Generate random data
    if st.button("Generate Test Data"):
        # Generate random data for testing
        data = pd.DataFrame({
            "Feature": [f"Feature_{i}" for i in range(10)],
            "Value": np.random.randn(10)
        })
        
        # Display as table
        st.subheader("Sample Data Table")
        st.dataframe(data)
        
        # Create a simple plot
        st.subheader("Sample Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(data["Feature"], data["Value"])
        ax.set_ylabel("Value")
        ax.set_xlabel("Feature")
        ax.set_title("Sample Feature Values")
        st.pyplot(fig)
        
        # Show success message
        st.success("Data generation and visualization test successful!")
    
    # File upload test
    st.header("File Upload Test")
    uploaded_file = st.file_uploader("Upload a test file", type=["txt", "csv", "wav", "mp3"])
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": uploaded_file.size
        }
        st.json(file_details)
        st.success("File upload test successful!")
        
    # Session state test
    st.header("Session State Test")
    if "count" not in st.session_state:
        st.session_state.count = 0
        
    if st.button("Increment Counter"):
        st.session_state.count += 1
        
    st.metric("Counter Value", st.session_state.count)
    
    if st.session_state.count > 0:
        st.success("Session state test successful!")
    
    # Footer
    st.markdown("---")
    st.info("""
    If all components on this page load correctly, your Streamlit setup is working properly.
    You can now run the full MemoTag application.
    """)

if __name__ == "__main__":
    main() 
import os
import tempfile
import numpy as np
import base64
from typing import List, Dict, Any
import streamlit as st

def get_supported_formats() -> List[str]:
    """
    Return a list of supported audio file formats.
    
    Returns:
        List[str]: List of supported audio file extensions
    """
    return ["wav", "mp3", "flac"]

def download_file(content: bytes, filename: str, mime_type: str) -> Dict[str, Any]:
    """
    Create a download button for a file.
    
    Args:
        content (bytes): File content
        filename (str): Name for the downloaded file
        mime_type (str): MIME type of the file
        
    Returns:
        Dict[str, Any]: Button HTML
    """
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def create_temp_audio_file(audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """
    Create a temporary audio file.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        format (str): Audio format
        
    Returns:
        str: Path to temporary file
    """
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp:
        sf.write(temp.name, audio_data, sample_rate)
        return temp.name

def display_audio_waveform(audio_data: np.ndarray, sample_rate: int):
    """
    Display audio waveform in Streamlit.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
    """
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    st.pyplot(fig)
    plt.close(fig)

def display_spectrogram(audio_data: np.ndarray, sample_rate: int):
    """
    Display spectrogram in Streamlit.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
    """
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    st.pyplot(fig)
    plt.close(fig)

def display_pitch_track(audio_data: np.ndarray, sample_rate: int):
    """
    Display pitch track in Streamlit.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
    """
    import librosa
    import matplotlib.pyplot as plt
    
    # Extract pitch using PYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    
    times = librosa.times_like(f0, sr=sample_rate)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, f0, label='f0', color='blue', alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Pitch Track (F0)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

def highlight_hesitations(transcription: Dict[str, Any]) -> str:
    """
    Create HTML that highlights hesitations in transcription.
    
    Args:
        transcription (Dict[str, Any]): Transcription data
        
    Returns:
        str: HTML with highlighted hesitations
    """
    if 'words' not in transcription:
        return transcription.get('text', '')
    
    words = transcription['words']
    highlighted_text = ""
    
    for word_info in words:
        word = word_info.get('word', '')
        is_hesitation = word_info.get('is_hesitation', False)
        
        if is_hesitation:
            highlighted_text += f'<span style="background-color: #ffeb3b;">{word}</span> '
        else:
            highlighted_text += f'{word} '
    
    return highlighted_text.strip()

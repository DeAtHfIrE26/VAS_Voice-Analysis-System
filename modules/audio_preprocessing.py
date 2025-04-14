import os
import tempfile
import librosa
import numpy as np
import soundfile as sf
import subprocess
import sys
from pydub import AudioSegment
import scipy.signal as signal
from typing import Dict, Tuple, List, Any

def check_ffmpeg():
    """
    Check if FFmpeg is available in the system PATH.
    If not, provide instructions to install it.
    """
    try:
        # Try to execute ffmpeg -version
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("\n" + "="*80)
        print("FFmpeg is not installed or not found in the system PATH.")
        print("FFmpeg is required for audio processing.")
        print("\nTo install FFmpeg:")
        if sys.platform.startswith('win'):
            print("1. Download FFmpeg from https://ffmpeg.org/download.html")
            print("2. Extract the downloaded zip file")
            print("3. Add the bin folder to your PATH environment variable")
            print("   - Search for 'Environment Variables' in Windows search")
            print("   - Click 'Edit the system environment variables'")
            print("   - Click 'Environment Variables'")
            print("   - Under 'System variables', find 'Path' and click 'Edit'")
            print("   - Click 'New' and add the path to the bin folder (e.g., C:\\ffmpeg\\bin)")
            print("   - Click 'OK' on all dialogs")
        elif sys.platform.startswith('darwin'):
            print("Run: brew install ffmpeg")
        else:
            print("Run: sudo apt update && sudo apt install ffmpeg")
        print("="*80 + "\n")
        return False

# Check for FFmpeg availability at module import time
ffmpeg_available = check_ffmpeg()

def get_supported_formats() -> List[str]:
    """
    Return a list of supported audio file formats.
    
    Returns:
        List[str]: List of supported audio file extensions
    """
    if ffmpeg_available:
        return ["wav", "mp3", "flac", "ogg", "m4a"]
    else:
        # Without FFmpeg, only wav files can be reliably processed
        return ["wav"]

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic information about an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        Dict[str, Any]: Dictionary containing audio information
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        file_format = file_path.split('.')[-1].lower()
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'channels': 1 if y.ndim == 1 else y.shape[0],
            'format': file_format,
            'num_samples': len(y)
        }
    except Exception as e:
        raise Exception(f"Error getting audio info: {str(e)}")

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        Tuple[np.ndarray, int]: Audio data as numpy array and sample rate
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        return y, sr
    except Exception as e:
        raise Exception(f"Error loading audio: {str(e)}")

def apply_noise_reduction(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply noise reduction to the audio signal.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        np.ndarray: Noise-reduced audio signal
    """
    # Estimate noise profile from the first 0.5 seconds (assumed to be noise/silence)
    noise_sample_duration = min(0.5, len(audio_data) / sr)
    noise_sample_length = int(noise_sample_duration * sr)
    noise_sample = audio_data[:noise_sample_length]
    
    # Compute noise profile
    noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
    
    # Apply spectral subtraction
    D = librosa.stft(audio_data)
    D_mag, D_phase = librosa.magphase(D)
    
    # Subtract noise profile from magnitude spectrogram
    D_mag_reduced = np.maximum(D_mag - noise_profile[:, np.newaxis] * 2, 0)
    
    # Reconstruct signal
    D_reduced = D_mag_reduced * D_phase
    audio_reduced = librosa.istft(D_reduced)
    
    return audio_reduced

def segment_audio(audio_data: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio into utterances using energy-based segmentation.
    
    Args:
        audio_data (np.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        List[Tuple[np.ndarray, float, float]]: List of tuples containing 
                                              (segment_data, start_time, end_time)
    """
    # Calculate energy of the signal
    energy = np.square(audio_data)
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.01 * sr)     # 10ms hop
    
    # Compute frame-wise energy
    energy_frames = librosa.util.frame(energy, frame_length=frame_length, hop_length=hop_length)
    energy_frames = np.mean(energy_frames, axis=0)
    
    # Calculate threshold as a fraction of the maximum energy
    threshold = 0.05 * np.max(energy_frames)
    
    # Find segments where energy is above threshold
    is_speech = energy_frames > threshold
    
    # Convert frame indices to sample indices
    speech_frames = np.where(is_speech)[0]
    
    segments = []
    if len(speech_frames) > 0:
        # Find boundaries of continuous speech segments
        boundaries = np.where(np.diff(speech_frames) > 1)[0] + 1
        segment_indices = np.split(speech_frames, boundaries)
        
        for indices in segment_indices:
            if len(indices) > 0:
                # Convert frame indices to time
                start_time = indices[0] * hop_length / sr
                end_time = (indices[-1] * hop_length + frame_length) / sr
                
                # Get corresponding audio samples
                start_sample = indices[0] * hop_length
                end_sample = min(len(audio_data), (indices[-1] * hop_length + frame_length))
                
                segment_data = audio_data[start_sample:end_sample]
                
                # Only add segments longer than 0.3 seconds
                if end_time - start_time > 0.3:
                    segments.append((segment_data, start_time, end_time))
    
    # If no segments were found or all were too short, use the entire audio
    if not segments:
        segments = [(audio_data, 0, len(audio_data)/sr)]
        
    return segments

def preprocess_audio(file_path: str, apply_noise_reduction_param: bool = True) -> Dict[str, Any]:
    """
    Preprocess an audio file for analysis.
    
    Args:
        file_path (str): Path to the audio file
        apply_noise_reduction_param (bool): Whether to apply noise reduction
        
    Returns:
        Dict[str, Any]: Dictionary containing preprocessed audio data
    """
    # Check if FFmpeg is required but not available
    file_ext = os.path.splitext(file_path)[1].lower().replace('.', '')
    if not ffmpeg_available and file_ext != 'wav':
        raise ValueError(f"FFmpeg is not available but required to process {file_ext} files. "
                        "Please install FFmpeg or convert your audio to WAV format.")
    
    # Load audio
    try:
        y, sr = load_audio(file_path)
    except Exception as e:
        if not ffmpeg_available:
            raise ValueError(f"Error loading audio: {str(e)}. This may be due to missing FFmpeg. "
                           "Please install FFmpeg as described in the documentation.") from e
        else:
            raise
    
    # Apply noise reduction if requested
    if apply_noise_reduction_param:
        y = apply_noise_reduction(y, sr)
    
    # Segment audio into utterances
    segments = segment_audio(y, sr)
    
    # Create temporary files for segments if needed
    segment_files = []
    for i, (segment_data, start_time, end_time) in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, segment_data, sr)
            segment_files.append({
                'path': tmp.name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
    
    # Extract raw audio features
    amplitude_envelope = np.mean(librosa.feature.rms(y=y))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Extract spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Extract mfccs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return {
        'raw_audio': y,
        'sample_rate': sr,
        'segments': segments,
        'segment_files': segment_files,
        'features': {
            'amplitude_envelope': amplitude_envelope,
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'mfccs': mfccs_mean
        }
    }

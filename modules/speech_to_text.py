import os
import tempfile
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI
import soundfile as sf

# Initialize OpenAI client if API key is available
try:
    client = OpenAI()
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Import external libraries based on availability
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

def transcribe_with_whisper(audio_data: Dict[str, Any], model_size: str = "base") -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI's Whisper API.
    
    Args:
        audio_data (Dict[str, Any]): Preprocessed audio data
        model_size (str): Whisper model size (ignored, API uses best available model)
        
    Returns:
        Dict[str, Any]: Transcription results with timestamps
    """
    if not OPENAI_AVAILABLE:
        raise ValueError("OpenAI client not available. Using Vosk as a fallback.")
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    
    # Create temporary file for audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio_data['raw_audio'], audio_data['sample_rate'])
        audio_path = tmp.name
    
    try:
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Call the OpenAI API
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language="en",
                timestamp_granularities=["word"]
            )
        
        # Process response to extract word-level information
        # Note: API returns a dictionary-like object, not a JSON string
        response_dict = response.model_dump()
        
        # Extract word-level information with timestamps
        words_with_timestamps = []
        
        # The structure might be different from the local whisper model output
        for segment in response_dict.get("segments", []):
            for word in segment.get("words", []):
                words_with_timestamps.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "confidence": word.get("confidence", 1.0)
                })
        
        # Identify hesitations and fillers
        hesitations = ["um", "uh", "er", "ah", "like", "you know"]
        for word_info in words_with_timestamps:
            word_info["is_hesitation"] = word_info["word"].lower().strip() in hesitations
        
        # Clean up temporary file
        os.unlink(audio_path)
        
        return {
            "text": response_dict.get("text", ""),
            "segments": response_dict.get("segments", []),
            "words": words_with_timestamps,
            "language": response_dict.get("language", "en")
        }
    
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        raise Exception(f"Error transcribing with Whisper API: {str(e)}")

def transcribe_with_vosk(audio_data: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe audio using Vosk offline speech recognition.
    
    Args:
        audio_data (Dict[str, Any]): Preprocessed audio data
        model_path (Optional[str]): Path to Vosk model directory
        
    Returns:
        Dict[str, Any]: Transcription results with timestamps
    """
    if not VOSK_AVAILABLE:
        raise ImportError("Vosk is not installed. Please install it with 'pip install vosk'")
    
    # Set up Vosk model - if model_path is None, use default model
    if model_path is None:
        model_path = os.path.join(os.path.expanduser("~"), ".cache", "vosk", "vosk-model-small-en-us-0.15")
        # If model doesn't exist, download a small one
        if not os.path.exists(model_path):
            import urllib.request
            import zipfile
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download small model if not present
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            zip_path = model_path + ".zip"
            
            print(f"Downloading Vosk model from {model_url}...")
            urllib.request.urlretrieve(model_url, zip_path)
            
            # Extract model
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_path))
            
            os.unlink(zip_path)
    
    # Load model
    model = vosk.Model(model_path)
    
    # Configure recognizer
    recognizer = vosk.KaldiRecognizer(model, audio_data['sample_rate'])
    recognizer.SetWords(True)  # Enable word timestamps
    
    # Process audio in chunks
    chunk_size = 4000  # Process 4000 samples at a time
    raw_audio = audio_data['raw_audio']
    
    results = []
    for i in range(0, len(raw_audio), chunk_size):
        chunk = raw_audio[i:i + chunk_size]
        
        # Convert to 16-bit integer format
        chunk_int16 = (chunk * 32767).astype(np.int16).tobytes()
        
        if recognizer.AcceptWaveform(chunk_int16):
            result = json.loads(recognizer.Result())
            results.append(result)
    
    # Get final result
    final_result = json.loads(recognizer.FinalResult())
    results.append(final_result)
    
    # Combine all results
    all_text = ""
    all_words = []
    
    for result in results:
        if "text" in result and result["text"].strip():
            all_text += " " + result["text"]
            
            if "result" in result:
                for word_info in result["result"]:
                    all_words.append({
                        "word": word_info["word"],
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "confidence": word_info["conf"]
                    })
    
    # Identify hesitations and fillers
    hesitations = ["um", "uh", "er", "ah", "like", "you know"]
    for word_info in all_words:
        word_info["is_hesitation"] = word_info["word"].lower().strip() in hesitations
    
    # Create segments (utterances separated by silences)
    segments = []
    
    if all_words:
        # Identify natural breaks (pauses longer than 0.5 seconds)
        silence_threshold = 0.5
        segment_start_idx = 0
        
        for i in range(1, len(all_words)):
            if all_words[i]["start"] - all_words[i-1]["end"] > silence_threshold:
                # End of a segment
                segment_words = all_words[segment_start_idx:i]
                segment_text = " ".join([w["word"] for w in segment_words])
                
                segments.append({
                    "text": segment_text,
                    "start": segment_words[0]["start"],
                    "end": segment_words[-1]["end"],
                    "words": segment_words
                })
                
                segment_start_idx = i
        
        # Add the last segment
        segment_words = all_words[segment_start_idx:]
        if segment_words:
            segment_text = " ".join([w["word"] for w in segment_words])
            segments.append({
                "text": segment_text,
                "start": segment_words[0]["start"],
                "end": segment_words[-1]["end"],
                "words": segment_words
            })
    
    return {
        "text": all_text.strip(),
        "segments": segments,
        "words": all_words,
        "language": "en"  # Vosk doesn't provide language detection
    }

def transcribe_audio(audio_data: Dict[str, Any], engine: str = "whisper") -> Dict[str, Any]:
    """
    Transcribe audio using the specified speech recognition engine.
    
    Args:
        audio_data (Dict[str, Any]): Preprocessed audio data
        engine (str): Recognition engine to use ('whisper' or 'vosk')
        
    Returns:
        Dict[str, Any]: Transcription results or a basic structure if all methods fail
    """
    # Default empty result in case all transcription methods fail
    empty_result = {
        "text": "Transcription failed. Please try again with a different audio file or check API keys.",
        "segments": [],
        "words": [],
        "language": "en",
        "transcription_error": True
    }
    
    # Check if OpenAI API key is available for Whisper
    openai_available = OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")
    
    try:
        if engine.lower() == "whisper":
            if not openai_available:
                print("OpenAI API key not available or not valid. Falling back to Vosk...")
                if VOSK_AVAILABLE:
                    try:
                        return transcribe_with_vosk(audio_data)
                    except Exception as vosk_e:
                        print(f"Error with Vosk transcription: {str(vosk_e)}")
                        return {
                            "text": "Transcription failed. Vosk fallback error: " + str(vosk_e),
                            "segments": [],
                            "words": [],
                            "language": "en",
                            "transcription_error": True,
                            "error_details": str(vosk_e)
                        }
                else:
                    return {
                        "text": "Transcription failed. OpenAI API key not available and Vosk is not installed.",
                        "segments": [],
                        "words": [],
                        "language": "en",
                        "transcription_error": True,
                        "error_details": "Missing API key and Vosk"
                    }
            
            # Try to use OpenAI Whisper API
            try:
                print("Using OpenAI Whisper API for transcription...")
                return transcribe_with_whisper(audio_data)
            except Exception as whisper_e:
                print(f"Error with Whisper API transcription: {str(whisper_e)}")
                print("Falling back to Vosk...")
                if VOSK_AVAILABLE:
                    try:
                        return transcribe_with_vosk(audio_data)
                    except Exception as vosk_e:
                        print(f"Error with Vosk transcription: {str(vosk_e)}")
                        return {
                            "text": "Both Whisper and Vosk transcription failed.",
                            "segments": [],
                            "words": [],
                            "language": "en",
                            "transcription_error": True,
                            "error_details": f"Whisper: {str(whisper_e)}, Vosk: {str(vosk_e)}"
                        }
                else:
                    return {
                        "text": f"Whisper API failed and Vosk is not installed. Error: {str(whisper_e)}",
                        "segments": [],
                        "words": [],
                        "language": "en",
                        "transcription_error": True,
                        "error_details": str(whisper_e)
                    }
                
        elif engine.lower() == "vosk":
            if not VOSK_AVAILABLE:
                if openai_available:
                    print("Vosk is not available. Falling back to Whisper API...")
                    try:
                        return transcribe_with_whisper(audio_data)
                    except Exception as whisper_e:
                        print(f"Error with Whisper transcription: {str(whisper_e)}")
                        return empty_result
                else:
                    return {
                        "text": "Transcription failed. Vosk is not installed and no OpenAI API key is available.",
                        "segments": [],
                        "words": [],
                        "language": "en",
                        "transcription_error": True,
                        "error_details": "Missing Vosk and API key"
                    }
            
            try:
                return transcribe_with_vosk(audio_data)
            except Exception as e:
                print(f"Error with Vosk transcription: {str(e)}")
                # Try Whisper as a fallback
                if openai_available:
                    try:
                        print("Vosk failed, trying Whisper API instead...")
                        return transcribe_with_whisper(audio_data)
                    except Exception as whisper_e:
                        print(f"Error with Whisper transcription: {str(whisper_e)}")
                        return empty_result
                else:
                    return {
                        "text": f"Vosk transcription failed and no OpenAI API key available. Error: {str(e)}",
                        "segments": [],
                        "words": [],
                        "language": "en",
                        "transcription_error": True,
                        "error_details": str(e)
                    }
        else:
            print(f"Unsupported speech recognition engine: {engine}")
            return {
                "text": f"Unsupported speech recognition engine: {engine}",
                "segments": [],
                "words": [],
                "language": "en",
                "transcription_error": True,
                "error_details": f"Invalid engine: {engine}"
            }
    except Exception as general_e:
        print(f"General transcription error: {str(general_e)}")
        return {
            "text": f"Transcription failed due to a general error: {str(general_e)}",
            "segments": [],
            "words": [],
            "language": "en",
            "transcription_error": True,
            "error_details": str(general_e)
        }

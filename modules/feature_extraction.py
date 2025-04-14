import numpy as np
import librosa
import nltk
import re  # Adding regex for fallback tokenization
from typing import Dict, List, Any, Optional, Tuple

# Custom tokenizers as fallbacks for when NLTK's tokenizers fail
def custom_word_tokenize(text):
    """Fallback word tokenizer that doesn't rely on punkt_tab"""
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except:
        # Simple fallback tokenization
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

def custom_sent_tokenize(text):
    """Fallback sentence tokenizer that doesn't rely on punkt_tab"""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except:
        # Simple fallback to split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def custom_pos_tag(tokens):
    """Fallback POS tagger"""
    try:
        return nltk.pos_tag(tokens)
    except:
        # Very simple fallback - just assume everything is a noun
        return [(token, 'NN') for token in tokens]

# Import stopwords with fallback
try:
    from nltk.corpus import stopwords
    STOPWORDS = stopwords.words('english')
except:
    # Basic stopwords list as fallback
    STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
                'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
                'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
                "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                "won't", 'wouldn', "wouldn't"]

# Ngrams with fallback
def get_ngrams(words, n=2):
    """Create n-grams from a list of words with fallback"""
    try:
        from nltk.util import ngrams
        return list(ngrams(words, n))
    except:
        # Simple fallback ngram implementation
        if len(words) < n:
            return []
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def extract_linguistic_features(transcription: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract linguistic features from transcribed text.
    
    Args:
        transcription (Dict[str, Any]): Transcription results
        
    Returns:
        Dict[str, float]: Dictionary of linguistic features
    """
    text = transcription.get("text", "")
    words = transcription.get("words", [])
    
    if not text.strip():
        return {
            "linguistic_word_count": 0,
            "linguistic_hesitation_frequency": 0,
            "linguistic_type_token_ratio": 0,
            "linguistic_avg_word_length": 0,
            "linguistic_sentence_count": 0,
            "linguistic_avg_words_per_sentence": 0,
            "linguistic_repeated_word_ratio": 0,
            "linguistic_filler_word_ratio": 0
        }
    
    # Tokenize text using custom functions to prevent punkt_tab errors
    word_tokens = custom_word_tokenize(text.lower())
    sentences = custom_sent_tokenize(text)
    
    # Basic counts
    word_count = len(word_tokens)
    sentence_count = len(sentences)
    
    # Hesitation markers
    hesitation_markers = ["um", "uh", "er", "ah", "like", "you know"]
    hesitation_count = sum(1 for word in word_tokens if word in hesitation_markers)
    
    # Calculate type-token ratio (vocabulary richness)
    unique_words = set(word_tokens)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0
    
    # Average word length
    word_lengths = [len(word) for word in word_tokens if word.isalpha()]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    
    # Words per sentence
    words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Repeated words analysis
    word_frequencies = {}
    for word in word_tokens:
        if word.isalpha() and word not in hesitation_markers:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    repeated_words = sum(1 for word, count in word_frequencies.items() if count > 1)
    repeated_word_ratio = repeated_words / len(word_frequencies) if word_frequencies else 0
    
    # Filler words ratio
    filler_words = ["well", "so", "basically", "actually", "literally", "anyway", "kind of"]
    filler_count = sum(1 for word in word_tokens if word in filler_words)
    filler_word_ratio = filler_count / word_count if word_count > 0 else 0
    
    # Part-of-speech analysis
    pos_tags = custom_pos_tag(word_tokens)
    
    # Count different parts of speech
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adjective_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    
    # Noun-to-verb ratio (can indicate syntactic complexity)
    noun_verb_ratio = noun_count / verb_count if verb_count > 0 else 0
    
    # Calculate word retrieval issues (hesitations before content words)
    word_retrieval_issues = 0
    for i in range(len(words) - 1):
        if i > 0 and words[i-1].get("is_hesitation", False):
            # Check if the current word is a content word (noun, verb, adjective)
            current_word = words[i]["word"].lower()
            pos_tags = custom_pos_tag([current_word])
            if pos_tags and pos_tags[0][1].startswith(('NN', 'VB', 'JJ')):
                word_retrieval_issues += 1
    
    # Word replacement or semantic error detection (difficult without a reference)
    # This is a simplified approximation based on uncommon word sequences
    uncommon_bigrams = 0
    word_bigrams = get_ngrams(word_tokens, 2)
    # Ideally, we would compare against a reference corpus
    # For now, we'll use a heuristic based on repeated bigrams
    bigram_counts = {}
    for bg in word_bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    
    uncommon_bigram_ratio = sum(1 for bg, count in bigram_counts.items() if count == 1) / len(bigram_counts) if bigram_counts else 0
    
    # Combine all features
    features = {
        "linguistic_word_count": word_count,
        "linguistic_sentence_count": sentence_count,
        "linguistic_hesitation_frequency": hesitation_count / word_count if word_count > 0 else 0,
        "linguistic_type_token_ratio": type_token_ratio,
        "linguistic_avg_word_length": avg_word_length,
        "linguistic_avg_words_per_sentence": words_per_sentence,
        "linguistic_noun_verb_ratio": noun_verb_ratio,
        "linguistic_repeated_word_ratio": repeated_word_ratio,
        "linguistic_filler_word_ratio": filler_word_ratio,
        "linguistic_word_retrieval_issues": word_retrieval_issues,
        "linguistic_uncommon_bigram_ratio": uncommon_bigram_ratio,
        "linguistic_noun_ratio": noun_count / word_count if word_count > 0 else 0,
        "linguistic_verb_ratio": verb_count / word_count if word_count > 0 else 0
    }
    
    return features

def extract_acoustic_features(audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract acoustic features from audio data.
    
    Args:
        audio_data (Dict[str, Any]): Preprocessed audio data
        
    Returns:
        Dict[str, Any]: Dictionary of acoustic features
    """
    y = audio_data['raw_audio']
    sr = audio_data['sample_rate']
    
    # Extract pause features
    # Detect silent regions
    silent_regions = librosa.effects.split(y, top_db=30)
    
    # Calculate pauses (gaps between silent regions)
    pauses = []
    for i in range(len(silent_regions)-1):
        pause_start = silent_regions[i][1] / sr
        pause_end = silent_regions[i+1][0] / sr
        pause_duration = pause_end - pause_start
        if pause_duration > 0.2:  # Only count pauses longer than 200ms
            pauses.append(pause_duration)
    
    # Pause statistics
    pause_count = len(pauses)
    avg_pause_duration = np.mean(pauses) if pauses else 0
    pause_rate = pause_count / (len(y) / sr) if len(y) > 0 else 0  # Pauses per second
    
    # Speech rate estimation (approximate)
    # This can be more accurate with word timestamps from the transcription
    if 'features' in audio_data and 'zero_crossing_rate' in audio_data['features']:
        speech_rate_indicator = audio_data['features']['zero_crossing_rate']
    else:
        speech_rate_indicator = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Extract pitch features (f0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    
    # Filter out NaN values for valid pitch statistics
    f0_valid = f0[~np.isnan(f0)]
    
    # Pitch statistics
    f0_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0
    f0_std = np.std(f0_valid) if len(f0_valid) > 0 else 0
    f0_range = np.ptp(f0_valid) if len(f0_valid) > 0 else 0
    
    # Extract voice quality features
    
    # Spectral centroid (indicates brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)
    
    # Spectral flatness (indicates noisiness vs. tonalness)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Harmonics-to-noise ratio (approximation)
    harmonic, percussive = librosa.effects.hpss(y)
    hnr_approx = np.sum(harmonic**2) / np.sum(percussive**2) if np.sum(percussive**2) > 0 else 0
    
    # Energy/amplitude features
    rms_energy = np.mean(librosa.feature.rms(y=y)[0])
    energy_std = np.std(librosa.feature.rms(y=y)[0])
    
    # Jitter approximation (pitch period perturbation)
    # This is a simplified approximation
    jitter = 0
    if len(f0_valid) > 1:
        f0_diff = np.diff(f0_valid)
        f0_diff_abs = np.abs(f0_diff)
        jitter = np.mean(f0_diff_abs) / f0_mean if f0_mean > 0 else 0
    
    # Combine all features
    features = {
        "acoustic_pause_count": pause_count,
        "acoustic_avg_pause_duration": avg_pause_duration,
        "acoustic_pause_rate": pause_rate,
        "acoustic_speech_rate_indicator": speech_rate_indicator,
        "acoustic_pitch_mean": f0_mean,
        "acoustic_pitch_std": f0_std,
        "acoustic_pitch_range": f0_range,
        "acoustic_spectral_centroid": spectral_centroid_mean,
        "acoustic_spectral_flatness": spectral_flatness,
        "acoustic_harmonics_to_noise_ratio": hnr_approx,
        "acoustic_rms_energy": rms_energy,
        "acoustic_energy_variability": energy_std,
        "acoustic_jitter": jitter
    }
    
    return features

def extract_temporal_features(audio_data: Dict[str, Any], transcription: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract temporal features combining audio and transcription.
    
    Args:
        audio_data (Dict[str, Any]): Preprocessed audio data
        transcription (Dict[str, Any]): Transcription results
        
    Returns:
        Dict[str, float]: Dictionary of temporal features
    """
    words = transcription.get("words", [])
    duration = len(audio_data['raw_audio']) / audio_data['sample_rate']
    
    if not words:
        return {
            "temporal_speech_rate_wpm": 0,
            "temporal_articulation_rate": 0,
            "temporal_avg_response_latency": 0,
            "temporal_speaking_time_ratio": 0,
            "temporal_avg_word_duration": 0
        }
    
    # Calculate speech rate in words per minute
    word_count = len(words)
    speech_rate_wpm = (word_count / duration) * 60 if duration > 0 else 0
    
    # Calculate articulation rate (speech rate excluding pauses)
    # Estimate speaking time (total duration of words)
    speaking_time = sum(word["end"] - word["start"] for word in words)
    articulation_rate = (word_count / speaking_time) * 60 if speaking_time > 0 else 0
    
    # Calculate response latency (time between words)
    response_latencies = []
    for i in range(1, len(words)):
        latency = words[i]["start"] - words[i-1]["end"]
        if latency > 0.1 and latency < 5:  # Filter out very short and very long pauses
            response_latencies.append(latency)
    
    avg_response_latency = np.mean(response_latencies) if response_latencies else 0
    
    # Calculate the ratio of speaking time to total time
    speaking_time_ratio = speaking_time / duration if duration > 0 else 0
    
    # Calculate average word duration
    word_durations = [word["end"] - word["start"] for word in words]
    avg_word_duration = np.mean(word_durations) if word_durations else 0
    
    # Calculate variability in word durations (could indicate inconsistent speech patterns)
    word_duration_std = np.std(word_durations) if word_durations else 0
    
    # Calculate number of long pauses (>1 second)
    long_pauses = sum(1 for latency in response_latencies if latency > 1.0)
    long_pause_rate = long_pauses / duration if duration > 0 else 0
    
    # Combine all features
    features = {
        "temporal_speech_rate_wpm": speech_rate_wpm,
        "temporal_articulation_rate": articulation_rate,
        "temporal_avg_response_latency": avg_response_latency,
        "temporal_speaking_time_ratio": speaking_time_ratio,
        "temporal_avg_word_duration": avg_word_duration,
        "temporal_word_duration_variability": word_duration_std,
        "temporal_long_pause_rate": long_pause_rate
    }
    
    return features

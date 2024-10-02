# src/audio_processor.py

import librosa
import numpy as np

def load_audio(audio_path, config):
    """
    Loads an audio file and applies trimming and normalization based on the configuration.
    
    Args:
        audio_path (str): Path to the audio file.
        config (dict): Audio processing configuration.
    
    Returns:
        np.ndarray: Processed audio signal.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=config['sample_rate'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading audio file at {audio_path}: {e}")

    if config.get('trim', True):
        audio, _ = librosa.effects.trim(audio)
    if config.get('normalize', True):
        audio = librosa.util.normalize(audio)

    return audio

def compute_mel_spectrogram(audio, config):
    """
    Computes the mel-spectrogram of an audio signal.
    
    Args:
        audio (np.ndarray): Audio signal.
        config (dict): Audio processing configuration.
    
    Returns:
        np.ndarray: Mel-spectrogram in decibels.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=config['sample_rate'],
        n_mels=config.get('n_mel_channels', 80),
        n_fft=config.get('filter_length', 1024),
        hop_length=config.get('hop_length', 256),
        win_length=config.get('win_length', 1024),
        fmin=config.get('mel_fmin', 0.0),
        fmax=config.get('mel_fmax', 8000.0)
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

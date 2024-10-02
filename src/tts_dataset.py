# src/tts_dataset.py

import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd
import os
import logging
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, config, augment=False):
        """
        Initializes the CustomDataset.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            tokenizer (TTSTokenizer): Tokenizer instance.
            config (dict): Configuration dictionary for audio processing.
            augment (bool): Whether to apply data augmentation.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.tokenizer = tokenizer
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.trim = config.get('trim', True)
        self.normalize = config.get('normalize', True)
        self.n_mel_channels = config.get('n_mel_channels', 80)
        self.filter_length = config.get('filter_length', 1024)
        self.hop_length = config.get('hop_length', 256)
        self.win_length = config.get('win_length', 1024)
        self.mel_fmin = config.get('mel_fmin', 0.0)
        self.mel_fmax = config.get('mel_fmax', 8000.0)
        self.augment = augment
        self.time_frames = config.get('time_frames', 5027)  # Fixed number of time frames

        self.logger = logging.getLogger("Train.Dataset")

        # Store the directory of metadata for resolving relative paths
        self.metadata_dir = os.path.dirname(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Check if 'audio_path' and 'text' exist
        if 'audio_path' not in row or 'text' not in row:
            self.logger.error(f"Missing 'audio_path' or 'text' in row {idx}")
            raise KeyError(f"Missing 'audio_path' or 'text' in row {idx}")

        audio_path = row['audio_path']
        text = row['text']

        # Resolve absolute path
        audio_path = os.path.join(self.metadata_dir, audio_path)

        # Encode text
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            self.logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise e

        # Trim silence
        if self.trim:
            audio, _ = librosa.effects.trim(audio)

        # Normalize audio
        if self.normalize:
            max_val = max(abs(audio))
            if max_val > 0:
                audio = audio / max_val

        # Data Augmentation
        if self.augment:
            audio = self.augment_audio(audio, self.sample_rate)

        # Compute mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )

        # Convert to log scale (dB)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize mel-spectrogram to [-1, 1]
        mel_spectrogram = (mel_spectrogram + 80) / 80  # Assuming dB ranges from -80 to 0
        mel_spectrogram = mel_spectrogram * 2 - 1  # Scale to [-1, 1]

        # Pad or Trim mel-spectrogram to fixed time_frames
        current_time_frames = mel_spectrogram.shape[1]
        if current_time_frames > self.time_frames:
            mel_spectrogram = mel_spectrogram[:, :self.time_frames]
        elif current_time_frames < self.time_frames:
            pad_amount = self.time_frames - current_time_frames
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)

        # Convert to tensor
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float)

        return {
            'tokens': tokens,           # Shape: (sequence_length,)
            'targets': mel_spectrogram # Shape: (n_mel_channels, time_frames)
        }

    def augment_audio(self, audio, sample_rate):
        """
        Applies data augmentation techniques to the audio.

        Args:
            audio (np.ndarray): Input audio signal.
            sample_rate (int): Sampling rate of the audio.

        Returns:
            np.ndarray: Augmented audio signal.
        """
        # Add Gaussian noise
        noise = np.random.normal(0, 0.005, audio.shape)
        audio_noisy = audio + noise

        # Pitch shifting
        audio_pitched = librosa.effects.pitch_shift(audio_noisy, sr=sample_rate, n_steps=2)

        # Time stretching
        audio_stretched = librosa.effects.time_stretch(audio_pitched, rate=1.1)

        return audio_stretched

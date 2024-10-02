# tests/test_audio.py

import unittest
from src.audio_processor import load_audio, compute_mel_spectrogram
import os
import numpy as np

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.audio_path = 'data/wavs/Second.wav'
        self.config = {
            "sample_rate": 22050,
            "trim": True,
            "normalize": True,
            "n_mel_channels": 80,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0
        }
    
    def test_load_audio(self):
        """
        Test that the audio is loaded correctly and is a numpy array.
        """
        if not os.path.exists(self.audio_path):
            self.skipTest(f"Audio file not found at {self.audio_path}")
        audio = load_audio(self.audio_path, self.config)
        self.assertIsInstance(audio, np.ndarray, "Loaded audio should be a numpy array.")
        # Additional checks can be added here, such as audio length, sample rate, etc.
    
    def test_compute_mel_spectrogram(self):
        """
        Test that mel-spectrograms are computed correctly and have the expected shape.
        """
        if not os.path.exists(self.audio_path):
            self.skipTest(f"Audio file not found at {self.audio_path}")
        audio = load_audio(self.audio_path, self.config)
        mel_spectrogram = compute_mel_spectrogram(audio, self.config)
        self.assertIsInstance(mel_spectrogram, np.ndarray, "Mel-spectrogram should be a numpy array.")
        self.assertEqual(mel_spectrogram.shape[0], self.config['n_mel_channels'], 
                         f"Mel-spectrogram should have {self.config['n_mel_channels']} mel channels.")
        # Additional checks can be added here, such as mel-spectrogram values range, etc.

if __name__ == '__main__':
    unittest.main()

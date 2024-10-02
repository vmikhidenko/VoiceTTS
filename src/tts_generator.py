# src/tts_generator.py

import torch

class TTSGenerator:
    def __init__(self, model, tokenizer, device):
        """
        Initializes the TTS Generator with the given model and tokenizer.
        
        Args:
            model (nn.Module): The trained VITS model.
            tokenizer (TTSTokenizer): Tokenizer instance for encoding text.
            device (torch.device): Device to run the model on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def synthesize(self, text):
        """
        Synthesizes speech from the given text.
        
        Args:
            text (str): Input text to synthesize.
        
        Returns:
            torch.Tensor: Generated mel-spectrogram.
        """
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)  # Shape: (1, sequence_length)
        
        with torch.no_grad():
            mel = self.model(tokens)  # Shape: (1, n_mels, time_frames)
        
        return mel.squeeze(0)  # Shape: (n_mels, time_frames)

# src/tts_data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict
from .my_tokenizer import TTSTokenizer

# Define your collate_fn
def collate_fn(batch, tokenizer: TTSTokenizer):
    tokens = [item['tokens'] for item in batch]
    targets = [item['targets'] for item in batch]
    
    # Pad tokens to the maximum length in the batch
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.char_to_id[tokenizer.pad])
    
    # Assuming mel-spectrograms have shape (n_mels, time_frames)
    # Pad along the time_frames dimension
    max_time_frames = max([mel.shape[1] for mel in targets]) if targets else 0
    mel_padded = [torch.nn.functional.pad(mel, (0, max_time_frames - mel.shape[1])) for mel in targets]
    mel_tensor = torch.stack(mel_padded)  # Shape: (batch_size, n_mels, max_time_frames)
    
    return {
        'tokens': tokens_padded,
        'targets': mel_tensor
    }

# Define your Dataset (this is just an example)
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# A function to load data and create DataLoader
def create_dataloader(data, batch_size, tokenizer: TTSTokenizer):
    dataset = MyDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=lambda x: collate_fn(x, tokenizer),
        shuffle=True
    )
    return dataloader

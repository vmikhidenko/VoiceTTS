# src/collate_fn.py

import torch
import logging
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    Pads mel-spectrograms and token sequences to the length of the longest in the batch.

    Args:
        batch (list of dict): Each dict contains 'tokens' and 'targets' tensors.

    Returns:
        dict: Padded 'tokens' and 'targets' tensors.
    """
    logger = logging.getLogger("Train.CollateFn")

    tokens = [item['tokens'] for item in batch]      # List of 1D tensors
    targets = [item['targets'] for item in batch]    # List of 2D tensors (n_mel_channels x time_frames)

    # Debugging: Log original shapes
    logger.debug(f"Original token lengths: {[t.shape[0] for t in tokens]}")
    logger.debug(f"Original target shapes: {[t.shape for t in targets]}")

    # Since all targets are already fixed to time_frames, no need to pad.
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)  # Shape: (batch_size, max_seq_length)

    # Debugging: Log padded shapes
    logger.debug(f"Padded tokens shape: {tokens_padded.shape}")
    logger.debug(f"Padded targets shape: {torch.stack(targets).shape}")

    targets_padded = torch.stack(targets)  # Shape: (batch_size, n_mel_channels, time_frames)

    return {
        'tokens': tokens_padded,
        'targets': targets_padded
    }

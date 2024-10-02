# src/model.py

import torch.nn as nn

class VITSModel(nn.Module):
    def __init__(
        self,
        embedding_dim=80,
        hidden_size=256,
        num_layers=2,
        vocab_size=73,
        n_mel_channels=80,
        time_frames=5027,
        upsampling_factor=8
    ):
        """
        Initializes the VITSModel with the specified parameters.

        Args:
            embedding_dim (int): Dimension of the embedding layer.
            hidden_size (int): Number of features in the hidden state of LSTM.
            num_layers (int): Number of recurrent layers in LSTM.
            vocab_size (int): Size of the vocabulary for the embedding layer.
            n_mel_channels (int): Number of mel frequency channels.
            time_frames (int): Number of time frames in the mel-spectrogram.
            upsampling_factor (int): Factor by which to upsample the sequence length.
        """
        super(VITSModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_mel_channels = n_mel_channels
        self.time_frames = time_frames
        self.upsampling_factor = upsampling_factor

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder Linear layer
        self.decoder_linear = nn.Linear(hidden_size, n_mel_channels * self.upsampling_factor)

        # Decoder BatchNorm1d
        self.decoder_bn = nn.BatchNorm1d(n_mel_channels * self.upsampling_factor)

        # Decoder ReLU
        self.decoder_relu = nn.ReLU()

        # Convolutional layers for better temporal mapping
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=n_mel_channels,
                out_channels=n_mel_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=n_mel_channels,
                out_channels=n_mel_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            # Add more layers if needed to reach or exceed desired time_frames
        )

        # Final activation and scaling
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing token indices. Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Output mel-spectrogram tensor. Shape: (batch_size, n_mel_channels, time_frames)
        """
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        encoded, _ = self.encoder(embedded)  # Shape: (batch_size, sequence_length, hidden_size)

        # Decode to mel-spectrogram frames
        decoded = self.decoder_linear(encoded)  # Shape: (batch_size, sequence_length, n_mel_channels * upsampling_factor)

        # Permute to (batch_size, n_mel_channels * upsampling_factor, sequence_length) for BatchNorm1d
        decoded = decoded.permute(0, 2, 1).contiguous()  # Shape: (batch_size, 640, sequence_length)

        # Apply BatchNorm1d
        decoded = self.decoder_bn(decoded)  # Shape: (batch_size, 640, sequence_length)

        # Apply ReLU
        decoded = self.decoder_relu(decoded)  # Shape: (batch_size, 640, sequence_length)

        # Permute back to (batch_size, sequence_length, 640)
        decoded = decoded.permute(0, 2, 1).contiguous()  # Shape: (batch_size, sequence_length, 640)

        # Reshape to (batch_size, sequence_length * upsampling_factor, n_mel_channels)
        decoded = decoded.view(decoded.size(0), decoded.size(1) * self.upsampling_factor, self.n_mel_channels)  # Shape: (batch_size, sequence_length * 8, 80)

        # Transpose to (batch_size, 80, sequence_length * 8)
        decoded = decoded.permute(0, 2, 1).contiguous()  # Shape: (batch_size, 80, sequence_length * 8)

        # Pass through convolutional layers
        mel_spectrogram = self.conv_layers(decoded)  # Shape: (batch_size, 80, sequence_length * 8 * 2^num_conv_layers)

        # Trim or pad to match desired time_frames
        if mel_spectrogram.size(2) > self.time_frames:
            mel_spectrogram = mel_spectrogram[:, :, :self.time_frames]
        elif mel_spectrogram.size(2) < self.time_frames:
            pad_amount = self.time_frames - mel_spectrogram.size(2)
            mel_spectrogram = nn.functional.pad(mel_spectrogram, (0, pad_amount), "constant", 0)

        # Apply activation and scaling
        mel_spectrogram = self.tanh(mel_spectrogram) * 40  # Adjust scaling based on mel-spectrogram range

        # Debugging: Log the shape
        print(f"[DEBUG] Mel-spectrogram shape: {mel_spectrogram.shape}")

        return mel_spectrogram  # Shape: (batch_size, n_mel_channels, time_frames)

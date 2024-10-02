# src/voice_trainer.py

import logging
import torch
import torch.nn as nn
from model import VITSModel

logger = logging.getLogger(__name__)

class VoiceTrainer:
    def __init__(self, model: VITSModel, device: torch.device, learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()  # Consider using nn.L1Loss() or a combination of losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        logger.info(f"Initialized VoiceTrainer with device: {self.device}")
    
    def train_step(self, tokens, targets):
        """
        Performs a single training step.
        
        Args:
            tokens (torch.Tensor): Input token IDs.
            targets (torch.Tensor): Target mel-spectrograms.
        
        Returns:
            float: Loss value.
        """
        self.model.train()
        tokens = tokens.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(tokens)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        
        Args:
            dataloader (DataLoader): Validation DataLoader.
        
        Returns:
            float: Average loss on validation set.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch['tokens'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(tokens)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss

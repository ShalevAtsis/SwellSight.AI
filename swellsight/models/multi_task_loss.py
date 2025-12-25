"""Multi-task loss function - placeholder implementation."""

import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining regression and classification losses.
    
    This is a placeholder implementation that will be fully implemented in task 2.
    """
    
    def __init__(self):
        """Initialize the multi-task loss function."""
        super().__init__()
        
        # Placeholder - will be implemented in task 2
        self.height_loss = nn.SmoothL1Loss()
        self.wave_type_loss = nn.CrossEntropyLoss()
        self.direction_loss = nn.CrossEntropyLoss()
        
        # Learnable loss weights
        self.height_weight = nn.Parameter(torch.tensor(1.0))
        self.wave_type_weight = nn.Parameter(torch.tensor(1.0))
        self.direction_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Combined loss tensor
        """
        # Placeholder implementation
        height_loss = self.height_loss(predictions['height'].squeeze(), targets['height'])
        wave_type_loss = self.wave_type_loss(predictions['wave_type'], targets['wave_type'])
        direction_loss = self.direction_loss(predictions['direction'], targets['direction'])
        
        total_loss = (
            self.height_weight * height_loss +
            self.wave_type_weight * wave_type_loss +
            self.direction_weight * direction_loss
        )
        
        return total_loss
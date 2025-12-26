"""Multi-task loss function for wave analysis model."""

import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining regression and classification losses with learnable weights.
    
    The loss function combines:
    - Height regression: SmoothL1Loss (robust to outliers)
    - Wave type classification: CrossEntropyLoss
    - Direction classification: CrossEntropyLoss
    
    Loss weights are learnable parameters that are optimized during training
    to automatically balance the contribution of each task.
    """
    
    def __init__(self):
        """Initialize the multi-task loss function with learnable weights."""
        super().__init__()
        
        # Individual loss functions for each task
        self.height_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.wave_type_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.direction_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # Learnable loss weights (initialized to 1.0)
        # These will be optimized during training to balance task contributions
        self.height_weight = nn.Parameter(torch.tensor(1.0))
        self.wave_type_weight = nn.Parameter(torch.tensor(1.0))
        self.direction_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions containing:
                - 'height': [batch_size, 1] - height regression outputs
                - 'wave_type': [batch_size, 4] - wave type classification logits
                - 'direction': [batch_size, 3] - direction classification logits
            targets: Ground truth targets containing:
                - 'height': [batch_size] - height values in meters
                - 'wave_type': [batch_size] - wave type class indices (0-3)
                - 'direction': [batch_size] - direction class indices (0-2)
        
        Returns:
            Dictionary containing:
                - 'total_loss': Combined weighted loss
                - 'height_loss': Individual height regression loss
                - 'wave_type_loss': Individual wave type classification loss
                - 'direction_loss': Individual direction classification loss
                - 'loss_weights': Current loss weight values
        """
        # Compute individual task losses
        height_loss = self.height_loss_fn(
            predictions['height'].squeeze(-1),  # Remove last dimension [batch_size, 1] -> [batch_size]
            targets['height']
        )
        
        wave_type_loss = self.wave_type_loss_fn(
            predictions['wave_type'],
            targets['wave_type']
        )
        
        direction_loss = self.direction_loss_fn(
            predictions['direction'],
            targets['direction']
        )
        
        # Apply learnable weights to combine losses
        # Use absolute values to ensure positive weights
        weighted_height_loss = torch.abs(self.height_weight) * height_loss
        weighted_wave_type_loss = torch.abs(self.wave_type_weight) * wave_type_loss
        weighted_direction_loss = torch.abs(self.direction_weight) * direction_loss
        
        # Total combined loss
        total_loss = weighted_height_loss + weighted_wave_type_loss + weighted_direction_loss
        
        return {
            'total_loss': total_loss,
            'height_loss': height_loss,
            'wave_type_loss': wave_type_loss,
            'direction_loss': direction_loss,
            'loss_weights': {
                'height_weight': torch.abs(self.height_weight).item(),
                'wave_type_weight': torch.abs(self.wave_type_weight).item(),
                'direction_weight': torch.abs(self.direction_weight).item()
            }
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weight values."""
        return {
            'height_weight': torch.abs(self.height_weight).item(),
            'wave_type_weight': torch.abs(self.wave_type_weight).item(),
            'direction_weight': torch.abs(self.direction_weight).item()
        }
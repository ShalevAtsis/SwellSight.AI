"""Wave Analysis Model architecture with ConvNeXt backbone."""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any

from ..config import ModelConfig


class WaveAnalysisModel(nn.Module):
    """
    Multi-task wave analysis model with shared ConvNeXt backbone and task-specific heads.
    
    Architecture:
    - Shared Feature Extractor: ConvNeXt-Base backbone (2048-dim features)
    - Height Regression Head: 2-layer MLP outputting single continuous value
    - Wave Type Classification Head: 2-layer MLP outputting 4-class probabilities
    - Direction Classification Head: 2-layer MLP outputting 3-class probabilities
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the wave analysis model.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__()
        self.config = config
        
        # Shared ConvNeXt backbone feature extractor
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Verify backbone output dimension matches config
        backbone_features = self.backbone.num_features
        if backbone_features != config.feature_dim:
            # Add projection layer if dimensions don't match
            self.feature_projection = nn.Linear(backbone_features, config.feature_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # Task-specific heads with 2-layer MLPs
        
        # Height regression head (outputs single continuous value in meters)
        self.height_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Wave type classification head (4 classes with softmax)
        self.wave_type_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_wave_types)
        )
        
        # Direction classification head (3 classes with softmax)
        self.direction_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_directions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for task-specific heads."""
        for module in [self.height_head, self.wave_type_head, self.direction_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, 3, 768, 768]
        
        Returns:
            Dictionary containing predictions for all tasks:
            - 'height': [batch_size, 1] - wave height in meters
            - 'wave_type': [batch_size, 4] - logits for wave type classification
            - 'direction': [batch_size, 3] - logits for direction classification
        """
        # Extract features using ConvNeXt backbone
        features = self.backbone(x)  # [batch_size, backbone_features]
        
        # Project to target feature dimension if needed
        features = self.feature_projection(features)  # [batch_size, feature_dim]
        
        # Generate predictions from task-specific heads
        height_pred = self.height_head(features)  # [batch_size, 1]
        wave_type_logits = self.wave_type_head(features)  # [batch_size, 4]
        direction_logits = self.direction_head(features)  # [batch_size, 3]
        
        return {
            'height': height_pred,
            'wave_type': wave_type_logits,
            'direction': direction_logits
        }
    
    def get_feature_extractor_output_dim(self) -> int:
        """Get the output dimension of the feature extractor."""
        return self.config.feature_dim
"""Model configuration for SwellSight Wave Analysis Model."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for the WaveAnalysisModel architecture."""
    
    # Model architecture
    backbone: str = 'convnext_base'
    input_size: Tuple[int, int] = (768, 768)
    feature_dim: int = 2048
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    # Task-specific configurations
    num_wave_types: int = 4  # A-frame, closeout, beach break, point break
    num_directions: int = 3  # left, right, both
    
    # Wave type classes
    wave_type_classes: Tuple[str, ...] = (
        'A_FRAME', 
        'CLOSEOUT', 
        'BEACH_BREAK', 
        'POINT_BREAK'
    )
    
    # Direction classes
    direction_classes: Tuple[str, ...] = (
        'LEFT', 
        'RIGHT', 
        'BOTH'
    )
    
    # Height regression bounds (meters)
    min_height: float = 0.3
    max_height: float = 4.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'backbone': self.backbone,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'num_wave_types': self.num_wave_types,
            'num_directions': self.num_directions,
            'wave_type_classes': self.wave_type_classes,
            'direction_classes': self.direction_classes,
            'min_height': self.min_height,
            'max_height': self.max_height
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
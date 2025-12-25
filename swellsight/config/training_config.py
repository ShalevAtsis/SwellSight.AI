"""Training configuration for SwellSight Wave Analysis Model."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for model training process."""
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Loss function weights (learnable)
    initial_height_weight: float = 1.0
    initial_type_weight: float = 1.0
    initial_direction_weight: float = 1.0
    
    # Optimization settings
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # Checkpointing
    checkpoint_frequency: int = 10  # Save every N epochs
    save_best_only: bool = False
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Validation
    validation_frequency: int = 1  # Validate every N epochs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'initial_height_weight': self.initial_height_weight,
            'initial_type_weight': self.initial_type_weight,
            'initial_direction_weight': self.initial_direction_weight,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'warmup_epochs': self.warmup_epochs,
            'checkpoint_frequency': self.checkpoint_frequency,
            'save_best_only': self.save_best_only,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'validation_frequency': self.validation_frequency
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
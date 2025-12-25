"""Data configuration for SwellSight Wave Analysis Model."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class DataConfig:
    """Configuration for data pipeline and augmentation."""
    
    # Dataset paths
    synthetic_data_path: str = "data/synthetic"
    real_data_path: str = "data/real"
    metadata_path: str = "data/metadata"
    
    # Dataset splits
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Data generation parameters
    num_synthetic_samples: int = 10000
    
    # Image preprocessing
    image_size: Tuple[int, int] = (768, 768)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet
    
    # Data augmentation parameters
    rotation_range: float = 15.0  # degrees
    brightness_range: float = 0.2  # ±20%
    contrast_range: float = 0.15   # ±15%
    gaussian_noise_std: float = 0.01
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'synthetic_data_path': self.synthetic_data_path,
            'real_data_path': self.real_data_path,
            'metadata_path': self.metadata_path,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'num_synthetic_samples': self.num_synthetic_samples,
            'image_size': self.image_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std,
            'rotation_range': self.rotation_range,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'gaussian_noise_std': self.gaussian_noise_std,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'shuffle_train': self.shuffle_train
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
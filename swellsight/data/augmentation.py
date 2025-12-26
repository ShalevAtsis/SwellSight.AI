"""Data augmentation transforms for wave analysis training data."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from torchvision import transforms
from PIL import Image
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)


class GaussianNoise(nn.Module):
    """Add Gaussian noise to images."""
    
    def __init__(self, std: float = 0.01, mean: float = 0.0):
        """
        Initialize Gaussian noise transform.
        
        Args:
            std: Standard deviation of noise
            mean: Mean of noise distribution
        """
        super().__init__()
        self.std = std
        self.mean = mean
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor (C, H, W)
            
        Returns:
            Tensor with added noise
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        
        # Clamp to valid range [0, 1] for normalized tensors
        return torch.clamp(noisy_tensor, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, mean={self.mean})"


class RandomBrightness(nn.Module):
    """Random brightness adjustment that preserves ground truth labels."""
    
    def __init__(self, brightness_range: float = 0.2):
        """
        Initialize random brightness transform.
        
        Args:
            brightness_range: Range for brightness adjustment (±range)
        """
        super().__init__()
        self.brightness_range = brightness_range
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random brightness adjustment.
        
        Args:
            tensor: Input tensor (C, H, W)
            
        Returns:
            Brightness-adjusted tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Generate random brightness factor
        brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        
        # Apply brightness adjustment
        adjusted_tensor = tensor * brightness_factor
        
        # Clamp to valid range [0, 1]
        return torch.clamp(adjusted_tensor, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(brightness_range={self.brightness_range})"


class RandomContrast(nn.Module):
    """Random contrast adjustment that preserves ground truth labels."""
    
    def __init__(self, contrast_range: float = 0.15):
        """
        Initialize random contrast transform.
        
        Args:
            contrast_range: Range for contrast adjustment (±range)
        """
        super().__init__()
        self.contrast_range = contrast_range
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random contrast adjustment.
        
        Args:
            tensor: Input tensor (C, H, W)
            
        Returns:
            Contrast-adjusted tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Generate random contrast factor
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        
        # Calculate mean for contrast adjustment
        mean = tensor.mean()
        
        # Apply contrast adjustment: (tensor - mean) * factor + mean
        adjusted_tensor = (tensor - mean) * contrast_factor + mean
        
        # Clamp to valid range [0, 1]
        return torch.clamp(adjusted_tensor, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(contrast_range={self.contrast_range})"


class WaveAugmentationPipeline:
    """
    Comprehensive augmentation pipeline for wave analysis training data.
    
    Implements rotation (±15°), brightness (±20%), contrast (±15%) transforms,
    Gaussian noise, and other realistic augmentations while preserving ground truth labels.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: float = 0.2,
        contrast_range: float = 0.15,
        gaussian_noise_std: float = 0.01,
        horizontal_flip_prob: float = 0.5,
        apply_noise_prob: float = 0.3,
        image_size: Tuple[int, int] = (768, 768),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation_range: Range for rotation in degrees (±range)
            brightness_range: Range for brightness adjustment (±range)
            contrast_range: Range for contrast adjustment (±range)
            gaussian_noise_std: Standard deviation for Gaussian noise
            horizontal_flip_prob: Probability of horizontal flip
            apply_noise_prob: Probability of applying Gaussian noise
            image_size: Target image size (height, width)
            normalize_mean: Mean values for normalization
            normalize_std: Standard deviation values for normalization
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_std = gaussian_noise_std
        self.horizontal_flip_prob = horizontal_flip_prob
        self.apply_noise_prob = apply_noise_prob
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        logger.info(f"Initialized WaveAugmentationPipeline with:")
        logger.info(f"  Rotation: ±{rotation_range}°")
        logger.info(f"  Brightness: ±{brightness_range*100}%")
        logger.info(f"  Contrast: ±{contrast_range*100}%")
        logger.info(f"  Gaussian noise std: {gaussian_noise_std}")
    
    def get_training_transforms(self) -> transforms.Compose:
        """
        Get training transforms with full augmentation pipeline.
        
        Returns:
            Composed transforms for training
        """
        # Define augmentation transforms (applied to PIL images)
        augmentation_transforms = [
            transforms.RandomRotation(
                degrees=self.rotation_range,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            ),
            transforms.ColorJitter(
                brightness=self.brightness_range,
                contrast=self.contrast_range,
                saturation=0.1,  # Slight saturation variation
                hue=0.05  # Slight hue variation for realism
            ),
            transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
        ]
        
        # Base preprocessing transforms
        base_transforms = [
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ]
        
        # Post-normalization transforms (applied to tensors)
        post_transforms = []
        if self.apply_noise_prob > 0:
            post_transforms.append(
                transforms.RandomApply([
                    GaussianNoise(std=self.gaussian_noise_std)
                ], p=self.apply_noise_prob)
            )
        
        # Combine all transforms
        all_transforms = augmentation_transforms + base_transforms + post_transforms
        
        return transforms.Compose(all_transforms)
    
    def get_validation_transforms(self) -> transforms.Compose:
        """
        Get validation transforms without augmentation.
        
        Returns:
            Composed transforms for validation
        """
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def validate_augmentation_preserves_labels(
        self, 
        sample_image: Union[Image.Image, torch.Tensor],
        ground_truth_labels: dict
    ) -> bool:
        """
        Validate that augmentation preserves ground truth labels.
        
        This is a conceptual validation - augmentations like rotation, brightness,
        and contrast should not change the physical wave properties.
        
        Args:
            sample_image: Sample image (PIL or tensor)
            ground_truth_labels: Ground truth labels dict
            
        Returns:
            True if labels are preserved (always True for geometric/photometric transforms)
        """
        # For geometric and photometric transforms, ground truth labels are preserved
        # This validation ensures the augmentation pipeline doesn't accidentally
        # modify label data structures
        
        required_keys = ['height_meters', 'wave_type', 'direction']
        
        if not all(key in ground_truth_labels for key in required_keys):
            logger.warning("Ground truth labels missing required keys")
            return False
        
        # Check data types
        if not isinstance(ground_truth_labels['height_meters'], (int, float)):
            logger.warning("Height should be numeric")
            return False
        
        if not isinstance(ground_truth_labels['wave_type'], str):
            logger.warning("Wave type should be string")
            return False
        
        if not isinstance(ground_truth_labels['direction'], str):
            logger.warning("Direction should be string")
            return False
        
        return True
    
    def apply_single_augmentation(
        self, 
        image: Union[Image.Image, torch.Tensor],
        augmentation_type: str,
        **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Apply a single augmentation transform for testing purposes.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation ('rotation', 'brightness', 'contrast', 'noise')
            **kwargs: Additional parameters for the augmentation
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'rotation':
            angle = kwargs.get('angle', random.uniform(-self.rotation_range, self.rotation_range))
            if isinstance(image, Image.Image):
                return transforms.functional.rotate(image, angle)
            else:
                raise ValueError("Rotation requires PIL Image input")
        
        elif augmentation_type == 'brightness':
            factor = kwargs.get('factor', 1.0 + random.uniform(-self.brightness_range, self.brightness_range))
            if isinstance(image, Image.Image):
                return transforms.functional.adjust_brightness(image, factor)
            elif isinstance(image, torch.Tensor):
                return RandomBrightness(self.brightness_range)(image)
            
        elif augmentation_type == 'contrast':
            factor = kwargs.get('factor', 1.0 + random.uniform(-self.contrast_range, self.contrast_range))
            if isinstance(image, Image.Image):
                return transforms.functional.adjust_contrast(image, factor)
            elif isinstance(image, torch.Tensor):
                return RandomContrast(self.contrast_range)(image)
            
        elif augmentation_type == 'noise':
            if isinstance(image, torch.Tensor):
                return GaussianNoise(self.gaussian_noise_std)(image)
            else:
                raise ValueError("Gaussian noise requires tensor input")
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def get_augmentation_info(self) -> dict:
        """
        Get information about the augmentation pipeline.
        
        Returns:
            Dictionary with augmentation parameters
        """
        return {
            'rotation_range_degrees': self.rotation_range,
            'brightness_range_percent': self.brightness_range * 100,
            'contrast_range_percent': self.contrast_range * 100,
            'gaussian_noise_std': self.gaussian_noise_std,
            'horizontal_flip_probability': self.horizontal_flip_prob,
            'noise_application_probability': self.apply_noise_prob,
            'target_image_size': self.image_size,
            'normalization_mean': self.normalize_mean,
            'normalization_std': self.normalize_std
        }


def create_augmentation_pipeline(config: Optional[dict] = None) -> WaveAugmentationPipeline:
    """
    Create augmentation pipeline from configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured augmentation pipeline
    """
    if config is None:
        config = {}
    
    return WaveAugmentationPipeline(
        rotation_range=config.get('rotation_range', 15.0),
        brightness_range=config.get('brightness_range', 0.2),
        contrast_range=config.get('contrast_range', 0.15),
        gaussian_noise_std=config.get('gaussian_noise_std', 0.01),
        horizontal_flip_prob=config.get('horizontal_flip_prob', 0.5),
        apply_noise_prob=config.get('apply_noise_prob', 0.3),
        image_size=config.get('image_size', (768, 768)),
        normalize_mean=config.get('normalize_mean', (0.485, 0.456, 0.406)),
        normalize_std=config.get('normalize_std', (0.229, 0.224, 0.225))
    )
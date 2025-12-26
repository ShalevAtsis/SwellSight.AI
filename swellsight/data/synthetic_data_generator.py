"""Synthetic data generator for wave analysis training data."""

import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import logging
from dataclasses import dataclass
import random
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class WaveParameters:
    """Wave generation parameters."""
    height_meters: float
    wave_type: str  # 'A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK'
    direction: str  # 'LEFT', 'RIGHT', 'BOTH'
    period_seconds: float = 10.0
    wavelength_meters: float = 100.0
    depth_meters: float = 5.0


class SyntheticDataGenerator:
    """
    Generator for synthetic wave training data from depth maps.
    
    Integrates with existing depth map generation code and converts depth maps
    to photorealistic training images using ControlNet.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Data generation configuration containing paths and parameters
        """
        self.config = config
        self.output_path = Path(config.get('synthetic_data_path', 'data/synthetic'))
        self.metadata_path = Path(config.get('metadata_path', 'data/metadata'))
        self.image_size = config.get('image_size', (768, 768))
        
        # Wave parameter ranges for realistic generation
        self.height_range = (0.3, 4.0)  # meters
        self.wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
        self.directions = ['LEFT', 'RIGHT', 'BOTH']
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SyntheticDataGenerator with output path: {self.output_path}")
    
    def generate_dataset(self, num_samples: int, output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic training dataset.
        
        Args:
            num_samples: Number of samples to generate
            output_path: Optional override for output path
        
        Returns:
            List of generated sample metadata
        """
        if output_path:
            self.output_path = output_path
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        samples_metadata = []
        
        for i in range(num_samples):
            # Generate random wave parameters within realistic ranges
            wave_params = self._generate_random_wave_params()
            
            try:
                # Generate single sample
                sample_metadata = self.generate_single_sample(wave_params, sample_id=i)
                samples_metadata.append(sample_metadata)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} samples")
                    
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                continue
        
        # Save dataset metadata
        metadata_file = self.metadata_path / "synthetic_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(samples_metadata, f, indent=2)
        
        logger.info(f"Successfully generated {len(samples_metadata)} samples")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return samples_metadata
    
    def generate_single_sample(self, wave_params: WaveParameters, sample_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a single synthetic sample from wave parameters.
        
        Args:
            wave_params: Wave generation parameters
            sample_id: Optional sample identifier
        
        Returns:
            Generated sample metadata with ground truth labels
        """
        if sample_id is None:
            sample_id = random.randint(0, 999999)
        
        # Generate depth map from wave parameters
        depth_map = self._generate_depth_map(wave_params)
        
        # Convert depth map to photorealistic image using ControlNet
        rgb_image = self._depth_to_image_controlnet(depth_map, wave_params)
        
        # Save image
        image_filename = f"sample_{sample_id:06d}.png"
        image_path = self.output_path / image_filename
        rgb_image.save(image_path)
        
        # Extract ground truth labels (these are preserved from generation parameters)
        ground_truth = self._extract_ground_truth_labels(wave_params)
        
        # Create sample metadata
        sample_metadata = {
            'sample_id': sample_id,
            'image_path': str(image_path),
            'image_filename': image_filename,
            'height_meters': ground_truth['height_meters'],
            'wave_type': ground_truth['wave_type'],
            'direction': ground_truth['direction'],
            'generation_params': {
                'period_seconds': wave_params.period_seconds,
                'wavelength_meters': wave_params.wavelength_meters,
                'depth_meters': wave_params.depth_meters
            },
            'image_size': self.image_size,
            'created_timestamp': datetime.now().isoformat()
        }
        
        return sample_metadata
    
    def _generate_random_wave_params(self) -> WaveParameters:
        """Generate random wave parameters within realistic ranges."""
        height = random.uniform(*self.height_range)
        wave_type = random.choice(self.wave_types)
        direction = random.choice(self.directions)
        period = random.uniform(8.0, 15.0)  # seconds
        wavelength = random.uniform(50.0, 200.0)  # meters
        depth = random.uniform(2.0, 10.0)  # meters
        
        return WaveParameters(
            height_meters=height,
            wave_type=wave_type,
            direction=direction,
            period_seconds=period,
            wavelength_meters=wavelength,
            depth_meters=depth
        )
    
    def _generate_depth_map(self, wave_params: WaveParameters) -> np.ndarray:
        """
        Generate depth map from wave parameters.
        
        This is a simplified implementation that creates a synthetic depth map
        based on wave parameters. In a full implementation, this would integrate
        with existing depth map generation code.
        
        Args:
            wave_params: Wave generation parameters
        
        Returns:
            Depth map as numpy array
        """
        height, width = self.image_size
        
        # Create base depth map
        x = np.linspace(0, wave_params.wavelength_meters, width)
        y = np.linspace(0, wave_params.wavelength_meters * height / width, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate wave pattern based on parameters
        wave_phase = 2 * np.pi * X / wave_params.wavelength_meters
        
        # Apply wave type characteristics
        if wave_params.wave_type == 'A_FRAME':
            # Peaked waves
            wave_height = wave_params.height_meters * np.abs(np.sin(wave_phase))
        elif wave_params.wave_type == 'CLOSEOUT':
            # Uniform breaking waves
            wave_height = wave_params.height_meters * np.ones_like(X)
        elif wave_params.wave_type == 'BEACH_BREAK':
            # Random breaking pattern
            noise = np.random.normal(0, 0.1, X.shape)
            wave_height = wave_params.height_meters * (0.8 + 0.4 * np.sin(wave_phase) + noise)
        else:  # POINT_BREAK
            # Directional breaking
            if wave_params.direction == 'LEFT':
                wave_height = wave_params.height_meters * np.exp(-X / (wave_params.wavelength_meters * 0.3))
            elif wave_params.direction == 'RIGHT':
                wave_height = wave_params.height_meters * np.exp(-(width - X) / (wave_params.wavelength_meters * 0.3))
            else:  # BOTH
                wave_height = wave_params.height_meters * np.sin(wave_phase)
        
        # Add base depth
        depth_map = wave_params.depth_meters - wave_height
        
        # Ensure positive depths
        depth_map = np.maximum(depth_map, 0.1)
        
        return depth_map.astype(np.float32)
    
    def _depth_to_image_controlnet(self, depth_map: np.ndarray, wave_params: WaveParameters) -> Image.Image:
        """
        Convert depth map to photorealistic image using ControlNet.
        
        This is a simplified implementation. In a full implementation, this would
        use ControlNet with Stable Diffusion to generate photorealistic beach images.
        
        Args:
            depth_map: Input depth map
            wave_params: Wave parameters for conditioning
        
        Returns:
            Generated RGB image
        """
        # For now, create a synthetic RGB image from the depth map
        # This simulates the ControlNet output
        
        # Normalize depth map to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        # Handle case where depth map has constant values
        if depth_max - depth_min < 1e-6:
            normalized_depth = np.full_like(depth_map, 128, dtype=np.uint8)
        else:
            normalized_depth = ((depth_map - depth_min) / 
                               (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Create RGB channels with wave-like appearance
        # Blue channel: water (inverse of depth)
        blue_channel = 255 - normalized_depth
        
        # Green channel: foam/wave activity
        green_channel = normalized_depth * 0.7
        
        # Red channel: sky/background
        red_channel = np.full_like(normalized_depth, 200)
        
        # Combine channels
        rgb_array = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        
        # Convert to PIL Image
        rgb_image = Image.fromarray(rgb_array.astype(np.uint8))
        
        # Resize to target size
        rgb_image = rgb_image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return rgb_image
    
    def _extract_ground_truth_labels(self, wave_params: WaveParameters) -> Dict[str, Any]:
        """
        Extract ground truth labels from wave generation parameters.
        
        This ensures perfect ground truth preservation as required by Property 4.
        
        Args:
            wave_params: Wave generation parameters
        
        Returns:
            Ground truth labels dictionary
        """
        return {
            'height_meters': float(wave_params.height_meters),
            'wave_type': wave_params.wave_type,
            'direction': wave_params.direction
        }
    
    def validate_sample_format(self, sample_metadata: Dict[str, Any]) -> bool:
        """
        Validate that a sample has the correct format.
        
        Args:
            sample_metadata: Sample metadata to validate
        
        Returns:
            True if format is valid, False otherwise
        """
        required_fields = [
            'sample_id', 'image_path', 'height_meters', 
            'wave_type', 'direction', 'image_size'
        ]
        
        # Check required fields exist
        for field in required_fields:
            if field not in sample_metadata:
                return False
        
        # Check data types
        if not isinstance(sample_metadata['height_meters'], (int, float)):
            return False
        
        if sample_metadata['wave_type'] not in self.wave_types:
            return False
        
        if sample_metadata['direction'] not in self.directions:
            return False
        
        # Check image file exists
        image_path = Path(sample_metadata['image_path'])
        if not image_path.exists():
            return False
        
        return True
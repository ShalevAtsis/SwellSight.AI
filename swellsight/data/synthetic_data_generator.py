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

# Import MiDaS and ControlNet components
from .midas_depth_extractor import MiDaSDepthExtractor
from .controlnet_generator import ControlNetSyntheticGenerator, AugmentationParameters, AugmentationParameterSystem
from .real_data_loader import RealDataLoader

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


@dataclass
class RealToSyntheticCorrespondence:
    """Tracks correspondence between real images and synthetic variants."""
    real_image_path: str
    real_image_labels: Dict[str, Any]
    depth_map_id: str
    synthetic_variants: List[Dict[str, Any]]
    augmentation_metadata: List[AugmentationParameters]
    correspondence_id: str
    created_timestamp: str


class SyntheticDataGenerator:
    """
    Generator for synthetic wave training data from depth maps.
    
    Updated to use MiDaS depth extraction and ControlNet generation with
    comprehensive augmentation parameter system. Integrates real-to-synthetic
    correspondence tracking and data quality validation.
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
        self.real_data_path = Path(config.get('real_data_path', 'data/real'))
        self.image_size = config.get('image_size', (768, 768))
        
        # Wave parameter ranges for realistic generation
        self.height_range = (0.3, 4.0)  # meters
        self.wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
        self.directions = ['LEFT', 'RIGHT', 'BOTH']
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MiDaS depth extractor
        self.depth_extractor = MiDaSDepthExtractor(
            model_name=config.get('midas_model', 'Intel/dpt-large'),
            storage_path=str(self.metadata_path / 'depth_maps')
        )
        
        # Initialize ControlNet synthetic generator
        self.controlnet_generator = ControlNetSyntheticGenerator(
            controlnet_model=config.get('controlnet_model', 'lllyasviel/sd-controlnet-depth'),
            batch_size=config.get('batch_size', 1)
        )
        
        # Initialize augmentation parameter system
        self.augmentation_system = AugmentationParameterSystem(
            seed=config.get('augmentation_seed', None)
        )
        
        # Initialize real data loader for correspondence tracking
        self.real_data_loader = RealDataLoader(str(self.real_data_path))
        
        # Data quality validation parameters
        self.min_quality_score = config.get('min_quality_score', 0.3)
        self.max_failed_attempts = config.get('max_failed_attempts', 3)
        
        # Real-to-synthetic correspondence tracking
        self.correspondence_data = []
        
        logger.info(f"Initialized SyntheticDataGenerator with:")
        logger.info(f"  Output path: {self.output_path}")
        logger.info(f"  MiDaS model: {config.get('midas_model', 'Intel/dpt-large')}")
        logger.info(f"  ControlNet model: {config.get('controlnet_model', 'lllyasviel/sd-controlnet-depth')}")
        logger.info(f"  Min quality score: {self.min_quality_score}")
    
    def generate_dataset(self, num_samples: int, output_path: Optional[Path] = None, 
                        use_real_images: bool = True) -> List[Dict[str, Any]]:
        """
        Generate synthetic training dataset using MiDaS and ControlNet.
        
        Args:
            num_samples: Number of samples to generate
            output_path: Optional override for output path
            use_real_images: Whether to use real images as source for depth extraction
        
        Returns:
            List of generated sample metadata
        """
        if output_path:
            self.output_path = output_path
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_samples} synthetic samples using MiDaS/ControlNet pipeline...")
        logger.info(f"Use real images: {use_real_images}")
        
        samples_metadata = []
        
        if use_real_images:
            # Generate from real images using MiDaS depth extraction
            samples_metadata = self._generate_from_real_images(num_samples)
        else:
            # Generate from synthetic depth maps (fallback method)
            samples_metadata = self._generate_from_synthetic_depth_maps(num_samples)
        
        # Save dataset metadata
        metadata_file = self.metadata_path / "synthetic_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(samples_metadata, f, indent=2)
        
        # Save correspondence data
        correspondence_file = self.metadata_path / "real_to_synthetic_correspondence.json"
        with open(correspondence_file, 'w') as f:
            json.dump([
                {
                    'real_image_path': corr.real_image_path,
                    'real_image_labels': corr.real_image_labels,
                    'depth_map_id': corr.depth_map_id,
                    'synthetic_variants': corr.synthetic_variants,
                    'correspondence_id': corr.correspondence_id,
                    'created_timestamp': corr.created_timestamp,
                    'augmentation_metadata': [
                        self._augmentation_params_to_dict(params) 
                        for params in corr.augmentation_metadata
                    ]
                }
                for corr in self.correspondence_data
            ], f, indent=2)
        
        logger.info(f"Successfully generated {len(samples_metadata)} samples")
        logger.info(f"Metadata saved to: {metadata_file}")
        logger.info(f"Correspondence data saved to: {correspondence_file}")
        
        return samples_metadata
    
    def _generate_from_real_images(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic samples from real images using MiDaS depth extraction."""
        # Load real image metadata
        real_metadata = self.real_data_loader.load_real_metadata()
        
        if not real_metadata:
            logger.warning("No real images found, falling back to synthetic depth map generation")
            return self._generate_from_synthetic_depth_maps(num_samples)
        
        logger.info(f"Found {len(real_metadata)} real images for depth extraction")
        
        samples_metadata = []
        samples_per_real_image = max(1, num_samples // len(real_metadata))
        
        for real_sample in real_metadata:
            try:
                # Extract depth map from real image
                depth_result = self.depth_extractor.extract_depth(real_sample['image_path'])
                
                # Validate depth quality
                if depth_result.depth_quality_score < self.min_quality_score:
                    logger.warning(f"Low quality depth map for {real_sample['image_path']}, skipping")
                    continue
                
                # Generate multiple synthetic variants from this real image
                variants_metadata = []
                augmentation_params_list = []
                
                for variant_idx in range(samples_per_real_image):
                    if len(samples_metadata) >= num_samples:
                        break
                    
                    # Generate augmentation parameters
                    augmentation_params = self.augmentation_system.generate_random_parameters()
                    
                    # Derive ground truth labels from augmentation parameters
                    ground_truth = self._derive_labels_from_augmentation(
                        augmentation_params, real_sample
                    )
                    
                    # Generate synthetic image using ControlNet
                    synthetic_result = self.controlnet_generator.generate_synthetic_image(
                        depth_result.depth_map, augmentation_params
                    )
                    
                    # Validate synthetic image quality
                    if not self._validate_synthetic_quality(synthetic_result):
                        logger.warning(f"Low quality synthetic image for variant {variant_idx}, retrying...")
                        continue
                    
                    # Save synthetic image
                    sample_id = len(samples_metadata)
                    image_filename = f"synthetic_{sample_id:06d}.jpg"
                    image_path = self.output_path / image_filename
                    
                    synthetic_image_pil = Image.fromarray(synthetic_result.synthetic_image)
                    synthetic_image_pil.save(image_path, 'JPEG', quality=95)
                    
                    # Create sample metadata
                    sample_metadata = {
                        'sample_id': sample_id,
                        'image_path': str(image_path),
                        'image_filename': image_filename,
                        'height_meters': ground_truth['height_meters'],
                        'wave_type': ground_truth['wave_type'],
                        'direction': ground_truth['direction'],
                        'data_source': 'synthetic_from_real',
                        'original_real_image': real_sample['image_path'],
                        'depth_map_id': depth_result.processing_metadata.get('depth_map_id', ''),
                        'depth_quality_score': depth_result.depth_quality_score,
                        'augmentation_params': self._augmentation_params_to_dict(augmentation_params),
                        'generation_metadata': synthetic_result.generation_metadata,
                        'image_size': self.image_size,
                        'created_timestamp': datetime.now().isoformat()
                    }
                    
                    samples_metadata.append(sample_metadata)
                    variants_metadata.append(sample_metadata)
                    augmentation_params_list.append(augmentation_params)
                
                # Track real-to-synthetic correspondence
                if variants_metadata:
                    correspondence = RealToSyntheticCorrespondence(
                        real_image_path=real_sample['image_path'],
                        real_image_labels=real_sample.get('labeler_info', {}),
                        depth_map_id=depth_result.processing_metadata.get('depth_map_id', ''),
                        synthetic_variants=variants_metadata,
                        augmentation_metadata=augmentation_params_list,
                        correspondence_id=f"corr_{len(self.correspondence_data):06d}",
                        created_timestamp=datetime.now().isoformat()
                    )
                    self.correspondence_data.append(correspondence)
                
            except Exception as e:
                logger.error(f"Failed to process real image {real_sample['image_path']}: {e}")
                continue
        
        return samples_metadata
    
    def _generate_from_synthetic_depth_maps(self, num_samples: int) -> List[Dict[str, Any]]:
        """Fallback method: generate from synthetic depth maps (original implementation)."""
        logger.info("Generating synthetic samples from synthetic depth maps (fallback method)")
        
        samples_metadata = []
        
        for i in range(num_samples):
            # Generate random wave parameters within realistic ranges
            wave_params = self._generate_random_wave_params()
            
            try:
                # Generate single sample using original method
                sample_metadata = self.generate_single_sample(wave_params, sample_id=i)
                samples_metadata.append(sample_metadata)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} samples")
                    
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                continue
        
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
        
        # Save image with error handling
        image_filename = f"sample_{sample_id:06d}.jpg"
        image_path = self.output_path / image_filename
        
        try:
            # Convert to RGB mode if needed (removes alpha channel issues)
            if rgb_image.mode != 'RGB':
                rgb_image = rgb_image.convert('RGB')
            rgb_image.save(image_path, 'JPEG', quality=95)
        except Exception as e:
            logger.warning(f"Failed to save as JPEG, trying PNG: {e}")
            # Fallback to PNG with error handling
            image_filename = f"sample_{sample_id:06d}.png"
            image_path = self.output_path / image_filename
            try:
                rgb_image.save(image_path, 'PNG')
            except Exception as png_error:
                logger.error(f"Failed to save image {sample_id}: {png_error}")
                raise
        
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
            'data_source': 'synthetic',  # Mark as synthetic data
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
    
    def _derive_labels_from_augmentation(self, augmentation_params: AugmentationParameters, 
                                       real_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive ground truth labels from augmentation parameters.
        
        This ensures ground truth labels are derived from augmentation parameters
        as required by the task specification.
        
        Args:
            augmentation_params: Augmentation parameters used for generation
            real_sample: Original real sample metadata for reference
        
        Returns:
            Ground truth labels dictionary
        """
        # Map augmentation parameters to wave analysis labels
        
        # Height: Use dominant wave height from augmentation parameters
        height_meters = augmentation_params.dominant_wave_height_m
        
        # Wave type: Derive from breaking behavior and wave characteristics
        if augmentation_params.breaking_type == "spilling":
            if augmentation_params.breaker_intensity < 0.3:
                wave_type = "BEACH_BREAK"
            else:
                wave_type = "A_FRAME"
        elif augmentation_params.breaking_type == "plunging":
            wave_type = "POINT_BREAK"
        elif augmentation_params.breaking_type == "collapsing":
            wave_type = "CLOSEOUT"
        else:  # surging
            wave_type = "BEACH_BREAK"
        
        # Direction: Derive from directional spread and wave fronts
        if augmentation_params.directional_spread_deg < 15.0:
            # Low spread indicates directional waves
            if augmentation_params.lateral_offset_m < -20:
                direction = "LEFT"
            elif augmentation_params.lateral_offset_m > 20:
                direction = "RIGHT"
            else:
                direction = "BOTH"
        else:
            # High spread indicates multi-directional
            direction = "BOTH"
        
        return {
            'height_meters': float(height_meters),
            'wave_type': wave_type,
            'direction': direction
        }
    
    def _validate_synthetic_quality(self, synthetic_result) -> bool:
        """
        Validate synthetic image quality and filter low-quality samples.
        
        Args:
            synthetic_result: SyntheticGenerationResult to validate
        
        Returns:
            True if quality is acceptable, False otherwise
        """
        try:
            # Check image properties
            if synthetic_result.synthetic_image is None:
                return False
            
            image = synthetic_result.synthetic_image
            
            # Check dimensions
            if len(image.shape) != 3 or image.shape[2] != 3:
                return False
            
            # Check value range
            if image.dtype != np.uint8 or np.any(image < 0) or np.any(image > 255):
                return False
            
            # Check for reasonable variation (not completely uniform)
            std_dev = np.std(image)
            if std_dev < 10:  # Too uniform
                return False
            
            # Check generation metadata for quality indicators
            metadata = synthetic_result.generation_metadata
            if 'prompt_quality_score' in metadata:
                if metadata['prompt_quality_score'] < 0.3:
                    return False
            
            # Check if ControlNet was used successfully
            if not metadata.get('use_controlnet', False):
                logger.debug("Using fallback generation, quality may be lower")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating synthetic quality: {e}")
            return False
    
    def _augmentation_params_to_dict(self, params: AugmentationParameters) -> Dict[str, Any]:
        """Convert AugmentationParameters to dictionary for JSON serialization."""
        return {
            # Camera View Geometry
            'camera_height_m': params.camera_height_m,
            'tilt_angle_deg': params.tilt_angle_deg,
            'horizontal_fov_deg': params.horizontal_fov_deg,
            'distance_to_breaking_m': params.distance_to_breaking_m,
            'lateral_offset_m': params.lateral_offset_m,
            
            # Wave Field Structure
            'dominant_wave_height_m': params.dominant_wave_height_m,
            'wavelength_m': params.wavelength_m,
            'wave_period_s': params.wave_period_s,
            'directional_spread_deg': params.directional_spread_deg,
            'visible_wave_fronts': params.visible_wave_fronts,
            
            # Breaking Behavior
            'breaking_type': params.breaking_type,
            'breaker_intensity': params.breaker_intensity,
            'crest_sharpness': params.crest_sharpness,
            'foam_coverage_pct': params.foam_coverage_pct,
            
            # Shore Interaction
            'beach_slope_deg': params.beach_slope_deg,
            'runup_distance_m': params.runup_distance_m,
            'backwash_visible': params.backwash_visible,
            'wet_sand_reflectivity': params.wet_sand_reflectivity,
            'shoreline_curvature': params.shoreline_curvature,
            
            # Water Surface Texture
            'surface_roughness': params.surface_roughness,
            'ripples_frequency_hz': params.ripples_frequency_hz,
            'wind_streak_visibility': params.wind_streak_visibility,
            'specular_highlight_intensity': params.specular_highlight_intensity,
            'micro_foam_density': params.micro_foam_density,
            
            # Lighting and Sun Position
            'sun_elevation_deg': params.sun_elevation_deg,
            'sun_azimuth_deg': params.sun_azimuth_deg,
            'light_intensity': params.light_intensity,
            'shadow_softness': params.shadow_softness,
            'sun_glare_probability': params.sun_glare_probability,
            
            # Atmospheric Conditions
            'haze_density': params.haze_density,
            'fog_layer_height_m': params.fog_layer_height_m,
            'humidity_level': params.humidity_level,
            'sky_clarity': params.sky_clarity,
            'contrast_attenuation': params.contrast_attenuation,
            
            # Weather State
            'cloud_coverage_pct': params.cloud_coverage_pct,
            'cloud_type': params.cloud_type,
            'rain_present': params.rain_present,
            'rain_streak_intensity': params.rain_streak_intensity,
            'storminess': params.storminess,
            
            # Optical and Sensor Artifacts
            'lens_distortion_coeff': params.lens_distortion_coeff,
            'motion_blur_kernel_size': params.motion_blur_kernel_size,
            'sensor_noise_level': params.sensor_noise_level,
            'compression_artifacts': params.compression_artifacts,
            'chromatic_aberration': params.chromatic_aberration,
            
            # Scene Occlusions and Noise Objects
            'people_count': params.people_count,
            'surfboard_present': params.surfboard_present,
            'birds_count': params.birds_count,
            'sea_spray_occlusion_prob': params.sea_spray_occlusion_prob,
            'foreground_blur_amount': params.foreground_blur_amount
        }
    
    def get_correspondence_data(self) -> List[RealToSyntheticCorrespondence]:
        """Get real-to-synthetic correspondence data for analysis."""
        return self.correspondence_data
    
    def validate_data_quality(self, samples_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data quality and provide statistics.
        
        Args:
            samples_metadata: List of sample metadata to validate
        
        Returns:
            Data quality validation results
        """
        validation_results = {
            'total_samples': len(samples_metadata),
            'valid_samples': 0,
            'invalid_samples': 0,
            'quality_issues': [],
            'parameter_coverage': {},
            'label_distribution': {
                'height_range': {'min': float('inf'), 'max': float('-inf')},
                'wave_types': {},
                'directions': {}
            }
        }
        
        heights = []
        wave_types = []
        directions = []
        
        for sample in samples_metadata:
            try:
                # Validate sample format
                if self.validate_sample_format(sample):
                    validation_results['valid_samples'] += 1
                    
                    # Collect statistics
                    height = sample['height_meters']
                    heights.append(height)
                    wave_types.append(sample['wave_type'])
                    directions.append(sample['direction'])
                    
                    # Update ranges
                    validation_results['label_distribution']['height_range']['min'] = min(
                        validation_results['label_distribution']['height_range']['min'], height
                    )
                    validation_results['label_distribution']['height_range']['max'] = max(
                        validation_results['label_distribution']['height_range']['max'], height
                    )
                    
                else:
                    validation_results['invalid_samples'] += 1
                    validation_results['quality_issues'].append(f"Invalid format: {sample.get('sample_id', 'unknown')}")
                    
            except Exception as e:
                validation_results['invalid_samples'] += 1
                validation_results['quality_issues'].append(f"Validation error: {e}")
        
        # Calculate distributions
        if wave_types:
            for wave_type in set(wave_types):
                validation_results['label_distribution']['wave_types'][wave_type] = wave_types.count(wave_type)
        
        if directions:
            for direction in set(directions):
                validation_results['label_distribution']['directions'][direction] = directions.count(direction)
        
        # Calculate parameter coverage if augmentation data is available
        if samples_metadata and 'augmentation_params' in samples_metadata[0]:
            validation_results['parameter_coverage'] = self._analyze_parameter_coverage(samples_metadata)
        
        # Calculate quality metrics
        validation_results['quality_score'] = validation_results['valid_samples'] / validation_results['total_samples'] if validation_results['total_samples'] > 0 else 0
        validation_results['height_diversity'] = len(set(heights)) / len(heights) if heights else 0
        validation_results['type_diversity'] = len(set(wave_types)) / len(wave_types) if wave_types else 0
        
        return validation_results
    
    def _analyze_parameter_coverage(self, samples_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze augmentation parameter coverage across samples."""
        coverage_stats = {}
        
        # Collect all augmentation parameters
        all_params = []
        for sample in samples_metadata:
            if 'augmentation_params' in sample:
                all_params.append(sample['augmentation_params'])
        
        if not all_params:
            return coverage_stats
        
        # Analyze coverage for key parameters
        key_params = [
            'dominant_wave_height_m', 'breaking_type', 'sun_elevation_deg',
            'sky_clarity', 'surface_roughness', 'foam_coverage_pct'
        ]
        
        for param_name in key_params:
            values = [params.get(param_name) for params in all_params if param_name in params]
            
            if values:
                if isinstance(values[0], (int, float)):
                    coverage_stats[param_name] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'unique_count': len(set(values))
                    }
                else:
                    # Categorical parameter
                    unique_values = set(values)
                    coverage_stats[param_name] = {
                        'unique_values': list(unique_values),
                        'counts': {val: values.count(val) for val in unique_values}
                    }
        
        return coverage_stats
    
    def generate_single_sample(self, wave_params: WaveParameters, sample_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a single synthetic sample from wave parameters (legacy method).
        
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
        
        # Save image with error handling
        image_filename = f"sample_{sample_id:06d}.jpg"
        image_path = self.output_path / image_filename
        
        try:
            # Convert to RGB mode if needed (removes alpha channel issues)
            if rgb_image.mode != 'RGB':
                rgb_image = rgb_image.convert('RGB')
            rgb_image.save(image_path, 'JPEG', quality=95)
        except Exception as e:
            logger.warning(f"Failed to save as JPEG, trying PNG: {e}")
            # Fallback to PNG with error handling
            image_filename = f"sample_{sample_id:06d}.png"
            image_path = self.output_path / image_filename
            try:
                rgb_image.save(image_path, 'PNG')
            except Exception as png_error:
                logger.error(f"Failed to save image {sample_id}: {png_error}")
                raise
        
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
            'data_source': 'synthetic',  # Mark as synthetic data
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
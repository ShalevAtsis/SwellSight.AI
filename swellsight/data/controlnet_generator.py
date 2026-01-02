"""ControlNet synthetic image generator for beach camera scenes."""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import logging
from dataclasses import dataclass, asdict
import random
from datetime import datetime
import cv2

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class AugmentationParameters:
    """Comprehensive augmentation parameters for beach camera scene generation."""
    
    # Camera View Geometry
    camera_height_m: float  # 1-50m
    tilt_angle_deg: float  # -10° to +30°
    horizontal_fov_deg: float  # 30-120°
    distance_to_breaking_m: float  # 10-500m
    lateral_offset_m: float  # -100m to +100m
    
    # Wave Field Structure
    dominant_wave_height_m: float  # 0.3-4.0m
    wavelength_m: float  # 5-200m
    wave_period_s: float  # 3-20s
    directional_spread_deg: float  # 0-45°
    visible_wave_fronts: int  # 1-10
    
    # Breaking Behavior
    breaking_type: str  # spilling, plunging, collapsing, surging
    breaker_intensity: float  # 0.0-1.0
    crest_sharpness: float  # 0.0-1.0
    foam_coverage_pct: float  # 0-100%
    
    # Shore Interaction
    beach_slope_deg: float  # 1-45°
    runup_distance_m: float  # 0-50m
    backwash_visible: bool
    wet_sand_reflectivity: float  # 0.0-1.0
    shoreline_curvature: float  # -0.1 to +0.1
    
    # Water Surface Texture
    surface_roughness: float  # 0.0-1.0
    ripples_frequency_hz: float  # 0-100 Hz
    wind_streak_visibility: float  # 0.0-1.0
    specular_highlight_intensity: float  # 0.0-1.0
    micro_foam_density: float  # 0.0-1.0
    
    # Lighting and Sun Position
    sun_elevation_deg: float  # 0-90°
    sun_azimuth_deg: float  # 0-360°
    light_intensity: float  # 0.0-2.0
    shadow_softness: float  # 0.0-1.0
    sun_glare_probability: float  # 0.0-1.0
    
    # Atmospheric Conditions
    haze_density: float  # 0.0-1.0
    fog_layer_height_m: float  # 0-100m
    humidity_level: float  # 0.0-1.0
    sky_clarity: str  # clear, partly_cloudy, overcast, stormy
    contrast_attenuation: float  # 0.0-1.0
    
    # Weather State
    cloud_coverage_pct: float  # 0-100%
    cloud_type: str  # cumulus, stratus, cirrus, cumulonimbus
    rain_present: bool
    rain_streak_intensity: float  # 0.0-1.0
    storminess: float  # 0.0-1.0
    
    # Optical and Sensor Artifacts
    lens_distortion_coeff: float  # -0.5 to +0.5
    motion_blur_kernel_size: int  # 0-20 pixels
    sensor_noise_level: float  # 0.0-0.1
    compression_artifacts: float  # 0.0-1.0
    chromatic_aberration: float  # 0.0-1.0
    
    # Scene Occlusions and Noise Objects
    people_count: int  # 0-20
    surfboard_present: bool
    birds_count: int  # 0-50
    sea_spray_occlusion_prob: float  # 0.0-1.0
    foreground_blur_amount: int  # 0-10 pixels


@dataclass
class SyntheticGenerationResult:
    """Result of synthetic image generation."""
    synthetic_image: np.ndarray
    depth_map: np.ndarray
    augmentation_params: AugmentationParameters
    generation_metadata: Dict[str, Any]
    quality_score: float


class AugmentationParameterSystem:
    """
    Comprehensive augmentation parameter system for beach camera scenes.
    
    Implements 10 categories of augmentation parameters with realistic
    distributions and plausible parameter combinations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize augmentation parameter system."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.parameter_ranges = self._define_parameter_ranges()
        self.parameter_correlations = self._define_parameter_correlations()
        
        logger.info("Augmentation parameter system initialized")
    
    def _define_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Define realistic parameter ranges for each category."""
        return {
            # Camera View Geometry
            'camera_height_m': {'min': 1.0, 'max': 50.0, 'distribution': 'log_normal'},
            'tilt_angle_deg': {'min': -10.0, 'max': 30.0, 'distribution': 'normal'},
            'horizontal_fov_deg': {'min': 30.0, 'max': 120.0, 'distribution': 'uniform'},
            'distance_to_breaking_m': {'min': 10.0, 'max': 500.0, 'distribution': 'log_normal'},
            'lateral_offset_m': {'min': -100.0, 'max': 100.0, 'distribution': 'normal'},
            
            # Wave Field Structure
            'dominant_wave_height_m': {'min': 0.3, 'max': 4.0, 'distribution': 'log_normal'},
            'wavelength_m': {'min': 5.0, 'max': 200.0, 'distribution': 'log_normal'},
            'wave_period_s': {'min': 3.0, 'max': 20.0, 'distribution': 'normal'},
            'directional_spread_deg': {'min': 0.0, 'max': 45.0, 'distribution': 'exponential'},
            'visible_wave_fronts': {'min': 1, 'max': 10, 'distribution': 'poisson'},
            
            # Breaking Behavior
            'breaking_type': {'values': ['spilling', 'plunging', 'collapsing', 'surging'], 'weights': [0.4, 0.3, 0.2, 0.1]},
            'breaker_intensity': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'crest_sharpness': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'foam_coverage_pct': {'min': 0.0, 'max': 100.0, 'distribution': 'beta'},
            
            # Shore Interaction
            'beach_slope_deg': {'min': 1.0, 'max': 45.0, 'distribution': 'log_normal'},
            'runup_distance_m': {'min': 0.0, 'max': 50.0, 'distribution': 'exponential'},
            'backwash_visible': {'probability': 0.6},
            'wet_sand_reflectivity': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'shoreline_curvature': {'min': -0.1, 'max': 0.1, 'distribution': 'normal'},
            
            # Water Surface Texture
            'surface_roughness': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'ripples_frequency_hz': {'min': 0.0, 'max': 100.0, 'distribution': 'exponential'},
            'wind_streak_visibility': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'specular_highlight_intensity': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'micro_foam_density': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            
            # Lighting and Sun Position
            'sun_elevation_deg': {'min': 0.0, 'max': 90.0, 'distribution': 'beta'},
            'sun_azimuth_deg': {'min': 0.0, 'max': 360.0, 'distribution': 'uniform'},
            'light_intensity': {'min': 0.0, 'max': 2.0, 'distribution': 'gamma'},
            'shadow_softness': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'sun_glare_probability': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            
            # Atmospheric Conditions
            'haze_density': {'min': 0.0, 'max': 1.0, 'distribution': 'exponential'},
            'fog_layer_height_m': {'min': 0.0, 'max': 100.0, 'distribution': 'exponential'},
            'humidity_level': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'sky_clarity': {'values': ['clear', 'partly_cloudy', 'overcast', 'stormy'], 'weights': [0.3, 0.4, 0.2, 0.1]},
            'contrast_attenuation': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            
            # Weather State
            'cloud_coverage_pct': {'min': 0.0, 'max': 100.0, 'distribution': 'beta'},
            'cloud_type': {'values': ['cumulus', 'stratus', 'cirrus', 'cumulonimbus'], 'weights': [0.4, 0.3, 0.2, 0.1]},
            'rain_present': {'probability': 0.15},
            'rain_streak_intensity': {'min': 0.0, 'max': 1.0, 'distribution': 'exponential'},
            'storminess': {'min': 0.0, 'max': 1.0, 'distribution': 'exponential'},
            
            # Optical and Sensor Artifacts
            'lens_distortion_coeff': {'min': -0.5, 'max': 0.5, 'distribution': 'normal'},
            'motion_blur_kernel_size': {'min': 0, 'max': 20, 'distribution': 'exponential'},
            'sensor_noise_level': {'min': 0.0, 'max': 0.1, 'distribution': 'exponential'},
            'compression_artifacts': {'min': 0.0, 'max': 1.0, 'distribution': 'exponential'},
            'chromatic_aberration': {'min': 0.0, 'max': 1.0, 'distribution': 'exponential'},
            
            # Scene Occlusions and Noise Objects
            'people_count': {'min': 0, 'max': 20, 'distribution': 'poisson'},
            'surfboard_present': {'probability': 0.3},
            'birds_count': {'min': 0, 'max': 50, 'distribution': 'poisson'},
            'sea_spray_occlusion_prob': {'min': 0.0, 'max': 1.0, 'distribution': 'beta'},
            'foreground_blur_amount': {'min': 0, 'max': 10, 'distribution': 'exponential'}
        }
    
    def _define_parameter_correlations(self) -> Dict[str, List[Tuple[str, float]]]:
        """Define correlations between parameters for realistic combinations."""
        return {
            # Storm conditions correlate with multiple parameters
            'storminess': [
                ('cloud_coverage_pct', 0.8),
                ('rain_present', 0.7),
                ('surface_roughness', 0.6),
                ('breaker_intensity', 0.5)
            ],
            
            # Sun elevation affects lighting and visibility
            'sun_elevation_deg': [
                ('light_intensity', 0.7),
                ('shadow_softness', -0.5),
                ('sun_glare_probability', 0.6)
            ],
            
            # Wave height correlates with breaking intensity
            'dominant_wave_height_m': [
                ('breaker_intensity', 0.6),
                ('foam_coverage_pct', 0.5),
                ('runup_distance_m', 0.4)
            ],
            
            # Weather conditions
            'rain_present': [
                ('cloud_coverage_pct', 0.8),
                ('humidity_level', 0.6),
                ('haze_density', 0.4)
            ]
        }
    
    def generate_random_parameters(self) -> AugmentationParameters:
        """Generate random augmentation parameters with realistic distributions."""
        params = {}
        
        # Generate base parameters
        for param_name, config in self.parameter_ranges.items():
            params[param_name] = self._sample_parameter(param_name, config)
        
        # Apply correlations for realistic combinations
        params = self._apply_correlations(params)
        
        # Validate and adjust parameters
        params = self._validate_parameters(params)
        
        return AugmentationParameters(**params)
    
    def _sample_parameter(self, param_name: str, config: Dict[str, Any]) -> Any:
        """Sample individual parameter from its distribution."""
        if 'values' in config:
            # Categorical parameter
            values = config['values']
            weights = config.get('weights', None)
            return np.random.choice(values, p=weights)
        
        elif 'probability' in config:
            # Boolean parameter
            return np.random.random() < config['probability']
        
        else:
            # Numerical parameter
            distribution = config.get('distribution', 'uniform')
            min_val = config['min']
            max_val = config['max']
            
            if distribution == 'uniform':
                value = np.random.uniform(min_val, max_val)
            elif distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6  # 3-sigma rule
                value = np.clip(np.random.normal(mean, std), min_val, max_val)
            elif distribution == 'log_normal':
                log_min = np.log(max(min_val, 1e-6))
                log_max = np.log(max_val)
                log_mean = (log_min + log_max) / 2
                log_std = (log_max - log_min) / 6
                value = np.exp(np.clip(np.random.normal(log_mean, log_std), log_min, log_max))
            elif distribution == 'exponential':
                # Exponential distribution scaled to range
                scale = (max_val - min_val) / 3
                value = min_val + np.clip(np.random.exponential(scale), 0, max_val - min_val)
            elif distribution == 'beta':
                # Beta distribution for 0-1 range, scaled to actual range
                alpha, beta = 2, 2  # Symmetric beta distribution
                unit_value = np.random.beta(alpha, beta)
                value = min_val + unit_value * (max_val - min_val)
            elif distribution == 'gamma':
                # Gamma distribution
                shape = 2
                scale = (max_val - min_val) / (shape * 2)
                value = min_val + np.clip(np.random.gamma(shape, scale), 0, max_val - min_val)
            elif distribution == 'poisson':
                # Poisson distribution for count data
                lam = (max_val - min_val) / 3
                value = min_val + np.clip(np.random.poisson(lam), 0, max_val - min_val)
            else:
                # Default to uniform
                value = np.random.uniform(min_val, max_val)
            
            # Convert to appropriate type
            if param_name.endswith('_count') or 'kernel_size' in param_name or 'wave_fronts' in param_name:
                return int(value)
            else:
                return float(value)
    
    def _apply_correlations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter correlations for realistic combinations."""
        for base_param, correlations in self.parameter_correlations.items():
            if base_param in params:
                base_value = params[base_param]
                
                for corr_param, correlation_strength in correlations:
                    if corr_param in params:
                        # Apply correlation influence
                        if isinstance(base_value, bool):
                            base_normalized = 1.0 if base_value else 0.0
                        elif isinstance(base_value, str):
                            # Skip string correlations for now
                            continue
                        else:
                            # Normalize base value to 0-1 range
                            param_config = self.parameter_ranges[base_param]
                            if 'min' in param_config and 'max' in param_config:
                                base_normalized = (base_value - param_config['min']) / (param_config['max'] - param_config['min'])
                            else:
                                continue
                        
                        # Apply correlation to target parameter
                        if isinstance(params[corr_param], bool):
                            # For boolean parameters, adjust probability
                            if correlation_strength > 0:
                                params[corr_param] = np.random.random() < (0.5 + 0.3 * base_normalized * correlation_strength)
                            else:
                                params[corr_param] = np.random.random() < (0.5 - 0.3 * base_normalized * abs(correlation_strength))
                        elif not isinstance(params[corr_param], str):
                            # For numerical parameters, adjust value
                            corr_config = self.parameter_ranges[corr_param]
                            if 'min' in corr_config and 'max' in corr_config:
                                current_normalized = (params[corr_param] - corr_config['min']) / (corr_config['max'] - corr_config['min'])
                                
                                # Apply correlation influence
                                influence = correlation_strength * base_normalized * 0.3  # Moderate influence
                                new_normalized = np.clip(current_normalized + influence, 0, 1)
                                
                                params[corr_param] = corr_config['min'] + new_normalized * (corr_config['max'] - corr_config['min'])
                                
                                # Convert to appropriate type
                                if corr_param.endswith('_count') or 'kernel_size' in corr_param:
                                    params[corr_param] = int(params[corr_param])
        
        return params
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust parameters for physical plausibility."""
        # Ensure wave period and wavelength are physically consistent
        if 'wave_period_s' in params and 'wavelength_m' in params:
            # Deep water wave speed: c = g*T/(2*pi) ≈ 1.56*T
            expected_wavelength = 1.56 * params['wave_period_s'] ** 2
            
            # Adjust if too far from expected
            ratio = params['wavelength_m'] / expected_wavelength
            if ratio < 0.5 or ratio > 2.0:
                params['wavelength_m'] = expected_wavelength * np.random.uniform(0.7, 1.3)
        
        # Ensure rain parameters are consistent
        if not params.get('rain_present', False):
            params['rain_streak_intensity'] = 0.0
        
        # Ensure storm parameters are consistent
        if params.get('storminess', 0) < 0.3:
            params['rain_present'] = False
            params['rain_streak_intensity'] = 0.0
        
        # Ensure sun glare is consistent with sun elevation
        if params.get('sun_elevation_deg', 0) < 10:
            params['sun_glare_probability'] = min(params.get('sun_glare_probability', 0), 0.2)
        
        return params


class ControlNetSyntheticGenerator:
    """
    ControlNet-based synthetic image generator for beach camera scenes.
    
    Uses Stable Diffusion with ControlNet depth conditioning to generate
    photorealistic synthetic beach images from depth maps with comprehensive
    augmentation parameter control.
    """
    
    def __init__(self, controlnet_model: str = "lllyasviel/sd-controlnet-depth", 
                 batch_size: int = 1, device: Optional[str] = None):
        """
        Initialize ControlNet synthetic generator.
        
        Args:
            controlnet_model: HuggingFace ControlNet model name
            batch_size: Batch size for generation
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.controlnet_model = controlnet_model
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components (placeholder for actual ControlNet implementation)
        self.use_controlnet = False  # Set to True when actual ControlNet is available
        
        logger.info(f"ControlNet generator initialized (model: {controlnet_model}, device: {self.device})")
        if not self.use_controlnet:
            logger.warning("ControlNet not available, using fallback synthetic generation")
    
    def generate_synthetic_image(self, depth_map: np.ndarray, 
                               augmentation_params: AugmentationParameters,
                               prompt: Optional[str] = None) -> SyntheticGenerationResult:
        """
        Generate synthetic beach image from depth map using ControlNet.
        
        Args:
            depth_map: Input depth map for structural guidance
            augmentation_params: Augmentation parameters for scene control
            prompt: Optional text prompt for generation
        
        Returns:
            SyntheticGenerationResult with generated image and metadata
        """
        try:
            if self.use_controlnet:
                # Use actual ControlNet implementation
                synthetic_image = self._generate_with_controlnet(
                    depth_map, augmentation_params, prompt
                )
            else:
                # Use fallback synthetic generation
                synthetic_image = self._generate_fallback_synthetic(
                    depth_map, augmentation_params
                )
            
            # Calculate quality score
            quality_score = self._assess_generation_quality(synthetic_image, depth_map)
            
            # Create generation metadata
            generation_metadata = {
                'controlnet_model': self.controlnet_model,
                'use_controlnet': self.use_controlnet,
                'device': self.device,
                'prompt': prompt or self._generate_prompt_from_params(augmentation_params),
                'depth_map_shape': depth_map.shape,
                'generation_timestamp': datetime.now().isoformat(),
                'prompt_quality_score': quality_score
            }
            
            return SyntheticGenerationResult(
                synthetic_image=synthetic_image,
                depth_map=depth_map,
                augmentation_params=augmentation_params,
                generation_metadata=generation_metadata,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic image: {e}")
            raise
    
    def _generate_with_controlnet(self, depth_map: np.ndarray, 
                                augmentation_params: AugmentationParameters,
                                prompt: str) -> np.ndarray:
        """Generate image using actual ControlNet (placeholder implementation)."""
        # This would be the actual ControlNet implementation
        # For now, return fallback generation
        return self._generate_fallback_synthetic(depth_map, augmentation_params)
    
    def _generate_fallback_synthetic(self, depth_map: np.ndarray, 
                                   augmentation_params: AugmentationParameters) -> np.ndarray:
        """
        Generate synthetic image using fallback method (enhanced depth-to-RGB conversion).
        
        This creates a more sophisticated synthetic image based on augmentation parameters
        while maintaining the depth structure.
        """
        height, width = depth_map.shape
        
        # Normalize depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min < 1e-6:
            normalized_depth = np.full_like(depth_map, 0.5)
        else:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Create base RGB channels
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky region (top portion, lighter depths)
        sky_mask = normalized_depth > 0.8
        sky_color = self._get_sky_color(augmentation_params)
        rgb_image[sky_mask] = sky_color
        
        # Water region (middle depths)
        water_mask = (normalized_depth >= 0.2) & (normalized_depth <= 0.8)
        water_color = self._get_water_color(augmentation_params, normalized_depth[water_mask])
        rgb_image[water_mask] = water_color
        
        # Beach/shore region (shallow depths)
        beach_mask = normalized_depth < 0.2
        beach_color = self._get_beach_color(augmentation_params)
        rgb_image[beach_mask] = beach_color
        
        # Apply wave effects
        rgb_image = self._apply_wave_effects(rgb_image, normalized_depth, augmentation_params)
        
        # Apply lighting effects
        rgb_image = self._apply_lighting_effects(rgb_image, augmentation_params)
        
        # Apply atmospheric effects
        rgb_image = self._apply_atmospheric_effects(rgb_image, augmentation_params)
        
        # Apply optical artifacts
        rgb_image = self._apply_optical_artifacts(rgb_image, augmentation_params)
        
        return rgb_image
    
    def _get_sky_color(self, params: AugmentationParameters) -> np.ndarray:
        """Generate sky color based on weather and lighting conditions."""
        base_blue = 135
        base_green = 206
        base_red = 235
        
        # Adjust for weather
        if params.sky_clarity == 'stormy':
            base_blue = 70
            base_green = 70
            base_red = 70
        elif params.sky_clarity == 'overcast':
            base_blue = 180
            base_green = 180
            base_red = 180
        elif params.sky_clarity == 'partly_cloudy':
            base_blue = 200
            base_green = 220
            base_red = 240
        
        # Adjust for sun elevation
        sun_factor = params.sun_elevation_deg / 90.0
        base_blue = int(base_blue * (0.5 + 0.5 * sun_factor))
        base_green = int(base_green * (0.5 + 0.5 * sun_factor))
        base_red = int(base_red * (0.5 + 0.5 * sun_factor))
        
        return np.array([base_red, base_green, base_blue], dtype=np.uint8)
    
    def _get_water_color(self, params: AugmentationParameters, depth_values: np.ndarray) -> np.ndarray:
        """Generate water color based on depth and conditions."""
        # Base water color (blue-green)
        base_colors = np.zeros((len(depth_values), 3), dtype=np.uint8)
        
        # Deeper water is darker blue
        blue_intensity = 50 + (1 - depth_values) * 150
        green_intensity = 30 + (1 - depth_values) * 100
        red_intensity = 10 + (1 - depth_values) * 50
        
        base_colors[:, 0] = np.clip(red_intensity, 0, 255)
        base_colors[:, 1] = np.clip(green_intensity, 0, 255)
        base_colors[:, 2] = np.clip(blue_intensity, 0, 255)
        
        # Adjust for surface roughness
        if params.surface_roughness > 0.5:
            # Add whitecaps/foam
            foam_mask = np.random.random(len(depth_values)) < params.surface_roughness * 0.3
            base_colors[foam_mask] = [200, 220, 240]  # Foam color
        
        return base_colors
    
    def _get_beach_color(self, params: AugmentationParameters) -> np.ndarray:
        """Generate beach/sand color."""
        # Base sand color
        base_red = 194
        base_green = 178
        base_blue = 128
        
        # Adjust for wetness
        if params.wet_sand_reflectivity > 0.5:
            # Wet sand is darker
            base_red = int(base_red * 0.7)
            base_green = int(base_green * 0.7)
            base_blue = int(base_blue * 0.7)
        
        return np.array([base_red, base_green, base_blue], dtype=np.uint8)
    
    def _apply_wave_effects(self, rgb_image: np.ndarray, depth: np.ndarray, 
                          params: AugmentationParameters) -> np.ndarray:
        """Apply wave-specific visual effects."""
        height, width = rgb_image.shape[:2]
        
        # Add foam based on breaking behavior
        if params.foam_coverage_pct > 0:
            foam_intensity = params.foam_coverage_pct / 100.0
            
            # Create foam pattern based on depth gradients
            grad_x = np.gradient(depth, axis=1)
            grad_y = np.gradient(depth, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradients indicate breaking waves
            foam_mask = gradient_magnitude > np.percentile(gradient_magnitude, 90 - foam_intensity * 30)
            
            # Apply foam color
            foam_color = np.array([240, 248, 255], dtype=np.uint8)  # White foam
            rgb_image[foam_mask] = foam_color
        
        return rgb_image
    
    def _apply_lighting_effects(self, rgb_image: np.ndarray, 
                              params: AugmentationParameters) -> np.ndarray:
        """Apply lighting and shadow effects."""
        # Adjust overall brightness based on sun elevation
        brightness_factor = 0.3 + 0.7 * (params.sun_elevation_deg / 90.0)
        brightness_factor *= params.light_intensity
        
        rgb_image = np.clip(rgb_image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # Add sun glare if probability is high
        if params.sun_glare_probability > 0.7:
            height, width = rgb_image.shape[:2]
            
            # Create glare pattern
            center_x = int(width * 0.7)  # Assume sun is towards right
            center_y = int(height * 0.3)  # Upper portion
            
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            glare_radius = min(width, height) * 0.3
            glare_mask = distance < glare_radius
            
            glare_intensity = (1 - distance / glare_radius) * params.sun_glare_probability
            glare_intensity = np.clip(glare_intensity, 0, 1)
            
            # Apply glare (brighten affected areas)
            rgb_image[glare_mask] = np.clip(
                rgb_image[glare_mask].astype(np.float32) * (1 + glare_intensity[glare_mask, np.newaxis]),
                0, 255
            ).astype(np.uint8)
        
        return rgb_image
    
    def _apply_atmospheric_effects(self, rgb_image: np.ndarray, 
                                 params: AugmentationParameters) -> np.ndarray:
        """Apply atmospheric effects like haze and fog."""
        if params.haze_density > 0.1:
            # Apply haze by blending with gray
            haze_color = np.array([200, 200, 200], dtype=np.uint8)
            haze_factor = params.haze_density * 0.5  # Moderate haze effect
            
            rgb_image = (rgb_image.astype(np.float32) * (1 - haze_factor) + 
                        haze_color * haze_factor).astype(np.uint8)
        
        # Apply contrast attenuation
        if params.contrast_attenuation > 0.1:
            # Reduce contrast
            mean_color = np.mean(rgb_image, axis=(0, 1))
            attenuation_factor = 1 - params.contrast_attenuation * 0.5
            
            rgb_image = (mean_color + (rgb_image - mean_color) * attenuation_factor).astype(np.uint8)
        
        return rgb_image
    
    def _apply_optical_artifacts(self, rgb_image: np.ndarray, 
                               params: AugmentationParameters) -> np.ndarray:
        """Apply optical and sensor artifacts."""
        # Apply motion blur
        if params.motion_blur_kernel_size > 0:
            kernel_size = int(params.motion_blur_kernel_size)
            if kernel_size > 1:
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
                rgb_image = cv2.filter2D(rgb_image, -1, kernel)
        
        # Apply sensor noise
        if params.sensor_noise_level > 0:
            noise = np.random.normal(0, params.sensor_noise_level * 255, rgb_image.shape)
            rgb_image = np.clip(rgb_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Apply compression artifacts (simplified)
        if params.compression_artifacts > 0.1:
            # Simulate JPEG compression by reducing quality
            quality = int(100 * (1 - params.compression_artifacts))
            quality = max(quality, 10)  # Minimum quality
            
            # This would normally involve actual JPEG compression/decompression
            # For now, just add some blockiness
            if quality < 50:
                block_size = 8
                h, w = rgb_image.shape[:2]
                for i in range(0, h, block_size):
                    for j in range(0, w, block_size):
                        block = rgb_image[i:i+block_size, j:j+block_size]
                        if block.size > 0:
                            mean_color = np.mean(block, axis=(0, 1))
                            rgb_image[i:i+block_size, j:j+block_size] = mean_color
        
        return rgb_image
    
    def _generate_prompt_from_params(self, params: AugmentationParameters) -> str:
        """Generate text prompt from augmentation parameters."""
        prompt_parts = ["beach camera view"]
        
        # Add wave description
        if params.dominant_wave_height_m > 2.0:
            prompt_parts.append("large waves")
        elif params.dominant_wave_height_m > 1.0:
            prompt_parts.append("medium waves")
        else:
            prompt_parts.append("small waves")
        
        # Add breaking type
        if params.breaking_type == "plunging":
            prompt_parts.append("plunging waves")
        elif params.breaking_type == "spilling":
            prompt_parts.append("spilling waves")
        
        # Add weather conditions
        if params.sky_clarity == "stormy":
            prompt_parts.append("stormy weather")
        elif params.sky_clarity == "overcast":
            prompt_parts.append("overcast sky")
        elif params.sky_clarity == "clear":
            prompt_parts.append("clear sky")
        
        # Add time of day based on sun elevation
        if params.sun_elevation_deg < 20:
            prompt_parts.append("golden hour lighting")
        elif params.sun_elevation_deg > 70:
            prompt_parts.append("bright daylight")
        
        return ", ".join(prompt_parts)
    
    def _assess_generation_quality(self, synthetic_image: np.ndarray, 
                                 depth_map: np.ndarray) -> float:
        """Assess quality of generated synthetic image."""
        try:
            # Check basic image properties
            if synthetic_image is None or synthetic_image.size == 0:
                return 0.0
            
            # Check for reasonable color variation
            color_std = np.std(synthetic_image)
            if color_std < 10:  # Too uniform
                return 0.3
            
            # Check for proper depth correlation
            gray_image = cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2GRAY)
            correlation = np.corrcoef(gray_image.flatten(), depth_map.flatten())[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            # Quality score based on variation and depth correlation
            variation_score = min(color_std / 50.0, 1.0)  # Normalize to 0-1
            correlation_score = abs(correlation)  # Absolute correlation
            
            overall_quality = (variation_score + correlation_score) / 2
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Failed to assess generation quality: {e}")
            return 0.5  # Default moderate quality
    
    def batch_generate(self, depth_maps: List[np.ndarray], 
                      param_sets: List[AugmentationParameters]) -> List[SyntheticGenerationResult]:
        """Generate multiple synthetic images in batch."""
        if len(depth_maps) != len(param_sets):
            raise ValueError("Number of depth maps must match number of parameter sets")
        
        results = []
        for depth_map, params in zip(depth_maps, param_sets):
            try:
                result = self.generate_synthetic_image(depth_map, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate synthetic image in batch: {e}")
                continue
        
        return results
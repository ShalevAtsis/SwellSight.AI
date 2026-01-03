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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ControlNet synthetic generator.
        
        Args:
            config: Configuration dictionary with ControlNet settings
        """
        self.controlnet_model = config.get('controlnet_model_name', "lllyasviel/sd-controlnet-depth")
        self.batch_size = config.get('batch_size', 1)
        self.guidance_scale = config.get('guidance_scale', 7.5)
        self.num_inference_steps = config.get('num_inference_steps', 20)
        self.controlnet_conditioning_scale = config.get('controlnet_conditioning_scale', 1.0)
        
        # Auto-detect device if not specified
        device = config.get('device')
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Try to initialize proper ControlNet
        self.controlnet_pipeline = None
        self.use_controlnet = False
        
        # Check if diffusers is available
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            self._initialize_controlnet()
        except ImportError:
            logger.warning("Diffusers not available. Install with: pip install diffusers transformers accelerate")
            logger.info("Using enhanced fallback generation")
        
        logger.info(f"ControlNet generator initialized (model: {self.controlnet_model}, device: {self.device})")
        if self.use_controlnet:
            logger.info("Using proper ControlNet pipeline for high-quality generation")
        else:
            logger.info("Using enhanced fallback generation with improved beach realism")
    
    def _initialize_controlnet(self):
        """Initialize proper ControlNet pipeline."""
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            
            logger.info("Loading ControlNet depth model...")
            
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load Stable Diffusion pipeline with ControlNet
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if self.device == "cuda":
                self.controlnet_pipeline = self.controlnet_pipeline.to("cuda")
                # Enable memory efficient attention
                try:
                    self.controlnet_pipeline.enable_model_cpu_offload()
                    self.controlnet_pipeline.enable_attention_slicing()
                except:
                    pass  # These optimizations are optional
            
            self.use_controlnet = True
            logger.info("ControlNet pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ControlNet: {e}")
            self.controlnet_pipeline = None
            self.use_controlnet = False
    
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
                synthetic_image, generation_metadata = self._generate_with_controlnet(
                    depth_map, augmentation_params, prompt
                )
            else:
                # Use enhanced fallback synthetic generation
                synthetic_image, generation_metadata = self._generate_enhanced_fallback(
                    depth_map, augmentation_params
                )
            
            # Calculate quality score
            quality_score = self._assess_generation_quality(synthetic_image, depth_map)
            
            # Update generation metadata
            generation_metadata.update({
                'controlnet_model': self.controlnet_model,
                'use_controlnet': self.use_controlnet,
                'device': self.device,
                'prompt': prompt or self._generate_detailed_prompt_from_params(augmentation_params),
                'depth_map_shape': depth_map.shape,
                'generation_timestamp': datetime.now().isoformat(),
                'quality_score': quality_score
            })
            
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
                                prompt: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate image using actual ControlNet with enhanced parameters for photorealism."""
        try:
            # Prepare depth map for ControlNet
            depth_image = self._prepare_depth_for_controlnet(depth_map)
            
            # Generate detailed surfer's perspective prompt
            if prompt is None:
                prompt = self._generate_detailed_prompt_from_params(augmentation_params)
            
            # Create comprehensive negative prompt
            negative_prompt = self._create_negative_prompt()
            
            logger.info(f"Generating surfer's perspective with ControlNet...")
            logger.info(f"Prompt: {prompt[:150]}...")
            
            # Enhanced generation parameters for photorealism
            enhanced_guidance_scale = 8.5  # Higher for better prompt following
            enhanced_steps = 25  # More steps for better quality
            controlnet_strength = 1.2  # Stronger depth conditioning
            
            # Generate image with ControlNet
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=enhanced_steps,
                guidance_scale=enhanced_guidance_scale,
                controlnet_conditioning_scale=controlnet_strength,
                generator=torch.Generator(device=self.device).manual_seed(42),
                # Additional quality parameters
                eta=0.0,  # Deterministic sampling for consistency
                cross_attention_kwargs={"scale": 1.0}  # Full attention scale
            )
            
            # Convert to numpy array
            synthetic_image = np.array(result.images[0])
            
            # Post-process for enhanced realism
            synthetic_image = self._post_process_for_realism(synthetic_image, augmentation_params)
            
            metadata = {
                'method': 'enhanced_controlnet',
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'guidance_scale': enhanced_guidance_scale,
                'num_inference_steps': enhanced_steps,
                'controlnet_conditioning_scale': controlnet_strength,
                'perspective': 'surfer_viewpoint',
                'quality_enhancements': [
                    'enhanced_guidance_scale',
                    'increased_inference_steps',
                    'stronger_depth_conditioning',
                    'comprehensive_negative_prompt',
                    'post_processing_realism'
                ]
            }
            
            return synthetic_image, metadata
            
        except Exception as e:
            logger.error(f"Enhanced ControlNet generation failed: {e}")
            # Fallback to enhanced generation
            return self._generate_enhanced_fallback(depth_map, augmentation_params)
    
    def _prepare_depth_for_controlnet(self, depth_map: np.ndarray) -> 'Image.Image':
        """Prepare depth map for ControlNet input."""
        # Normalize depth map to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min < 1e-6:
            normalized_depth = np.full_like(depth_map, 128, dtype=np.uint8)
        else:
            normalized_depth = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Convert to 3-channel image
        depth_rgb = np.stack([normalized_depth] * 3, axis=-1)
        
        # Convert to PIL Image
        depth_image = Image.fromarray(depth_rgb)
        
        # Resize to standard size if needed (ControlNet works best with 512x512)
        if depth_image.size != (512, 512):
            depth_image = depth_image.resize((512, 512), Image.LANCZOS)
        
        return depth_image
    
    def _create_negative_prompt(self) -> str:
        """Create comprehensive negative prompt to avoid unwanted elements and ensure photorealism."""
        return (
            # Avoid non-photorealistic styles
            "cartoon, anime, painting, drawing, sketch, illustration, "
            "digital art, CGI, 3D render, artificial, fake, "
            
            # Avoid low quality
            "low quality, blurry, distorted, unrealistic, pixelated, "
            "grainy, noisy, compressed, artifacts, low resolution, "
            "bad photography, amateur, overexposed, underexposed, "
            
            # Avoid unwanted objects and people
            "people, humans, person, surfer in water, swimmer, "
            "buildings, houses, cars, vehicles, boats, ships, "
            "surfboards in frame, equipment, gear, "
            
            # Avoid text and watermarks
            "text, watermark, logo, signature, copyright, "
            "writing, letters, numbers, signs, "
            
            # Avoid bad wave characteristics
            "bad anatomy, deformed waves, unnatural water, "
            "impossible wave physics, floating objects, "
            "weird water behavior, unrealistic foam, "
            
            # Avoid bad colors and lighting
            "oversaturated, artificial colors, neon colors, "
            "purple water, pink water, unnatural hues, "
            "harsh shadows, blown out highlights, "
            "flat lighting, indoor lighting, "
            
            # Avoid unwanted weather/conditions
            "snow, ice, frozen water, desert, mountains in water, "
            "swimming pool, bathtub, artificial water, "
            
            # Avoid composition issues
            "cropped waves, cut off horizon, tilted horizon, "
            "bad framing, cluttered composition, "
            "multiple horizons, floating elements, "
            
            # Avoid technical issues
            "motion blur, camera shake, out of focus background, "
            "lens flare, chromatic aberration, vignetting, "
            "fish eye distortion, wide angle distortion"
        )
    
    def _generate_enhanced_fallback(self, depth_map: np.ndarray, 
                                   augmentation_params: AugmentationParameters) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic image using enhanced fallback method with improved beach realism.
        
        This creates a sophisticated synthetic image based on augmentation parameters
        while maintaining the depth structure with much better quality than basic fallback.
        """
        height, width = depth_map.shape
        
        # Normalize depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min < 1e-6:
            normalized_depth = np.full_like(depth_map, 0.5)
        else:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Create base RGB channels with higher resolution processing
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Sky region (far/high depths) - more realistic sky
        sky_mask = normalized_depth > 0.7
        sky_color = self._get_realistic_sky_color(augmentation_params)
        rgb_image[sky_mask] = sky_color.astype(np.float32)
        
        # Water region with realistic wave patterns
        water_mask = (normalized_depth >= 0.3) & (normalized_depth <= 0.7)
        if np.any(water_mask):
            water_colors = self._generate_realistic_water(
                normalized_depth, 
                augmentation_params,
                water_mask
            )
            rgb_image[water_mask] = water_colors
        
        # Beach/shore region (close/low depths) - realistic sand
        beach_mask = normalized_depth < 0.3
        beach_color = self._get_realistic_beach_color(augmentation_params)
        rgb_image[beach_mask] = beach_color.astype(np.float32)
        
        # Add sophisticated wave foam and breaking patterns
        rgb_image = self._add_realistic_wave_foam(rgb_image, normalized_depth, augmentation_params)
        
        # Add depth-based lighting effects
        rgb_image = self._apply_depth_based_lighting(rgb_image, normalized_depth, augmentation_params)
        
        # Add realistic water surface effects
        rgb_image = self._add_water_surface_effects(rgb_image, normalized_depth, augmentation_params)
        
        # Apply atmospheric perspective
        rgb_image = self._apply_atmospheric_perspective(rgb_image, normalized_depth, augmentation_params)
        
        # Add realistic texture and noise
        rgb_image = self._add_realistic_texture(rgb_image, augmentation_params)
        
        # Convert to uint8
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        metadata = {
            'method': 'enhanced_fallback',
            'quality_enhancements': [
                'realistic_sky_colors',
                'wave_pattern_generation',
                'depth_based_lighting',
                'atmospheric_perspective',
                'surface_texture_simulation'
            ]
        }
        
        return rgb_image, metadata
    
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
    
    def _generate_detailed_prompt_from_params(self, params: AugmentationParameters) -> str:
        """Generate detailed surfer's perspective prompt (optimized for 77 token limit)."""
        
        # Core surfer's perspective (essential elements)
        prompt_parts = [
            "photorealistic surfer's perspective",
            "ocean waves from beach",
            "surf check viewpoint",
            "crystal clear water",
            "professional surf photography"
        ]
        
        # Wave characteristics (prioritized)
        wave_height = params.dominant_wave_height_m
        if wave_height < 1.0:
            prompt_parts.append("small clean waves")
        elif wave_height < 2.0:
            prompt_parts.append("perfect surfable waves")
        elif wave_height < 3.0:
            prompt_parts.append("large powerful waves")
        else:
            prompt_parts.append("massive waves")
        
        # Breaking type (key for wave analysis)
        breaking_type = params.breaking_type
        if breaking_type == "spilling":
            prompt_parts.append("spilling white foam")
        elif breaking_type == "plunging":
            prompt_parts.append("barreling waves")
        elif breaking_type == "collapsing":
            prompt_parts.append("explosive white water")
        elif breaking_type == "surging":
            prompt_parts.append("shore break waves")
        
        # Foam coverage (important visual element)
        if params.foam_coverage_pct > 50:
            prompt_parts.append("abundant white foam")
        elif params.foam_coverage_pct > 20:
            prompt_parts.append("clean wave crests")
        else:
            prompt_parts.append("glassy wave faces")
        
        # Lighting (affects image quality significantly)
        sun_elevation = params.sun_elevation_deg
        if sun_elevation > 60:
            prompt_parts.append("bright sunny day")
        elif sun_elevation > 30:
            prompt_parts.append("good natural lighting")
        else:
            prompt_parts.append("golden hour light")
        
        # Weather (affects overall scene)
        if params.rain_present:
            prompt_parts.append("stormy conditions")
        elif params.cloud_coverage_pct > 70:
            prompt_parts.append("overcast sky")
        else:
            prompt_parts.append("clear blue sky")
        
        # Camera perspective
        if params.camera_height_m > 20:
            prompt_parts.append("elevated viewpoint")
        else:
            prompt_parts.append("beach level view")
        
        # Quality enhancers (essential for realism)
        prompt_parts.extend([
            "high resolution",
            "sharp focus",
            "natural colors",
            "detailed wave structure"
        ])
        
        # Join and ensure we stay under token limit
        full_prompt = ", ".join(prompt_parts)
        
        # Rough token estimation (average 1.3 tokens per word)
        estimated_tokens = len(full_prompt.split()) * 1.3
        
        if estimated_tokens > 75:  # Leave some margin
            # Trim to essential elements
            essential_parts = [
                "photorealistic surfer's perspective",
                "ocean waves from beach",
                "surf check viewpoint",
                prompt_parts[5],  # Wave size
                prompt_parts[6],  # Breaking type
                prompt_parts[7],  # Foam
                prompt_parts[8],  # Lighting
                "high resolution",
                "sharp focus",
                "natural colors"
            ]
            full_prompt = ", ".join(essential_parts)
        
        return full_prompt
    
    def _get_realistic_sky_color(self, params: AugmentationParameters) -> np.ndarray:
        """Generate realistic sky colors based on weather and lighting conditions."""
        base_sky = np.array([135, 206, 235], dtype=np.float32)  # Sky blue
        
        # Adjust for weather conditions
        if params.rain_present:
            base_sky = np.array([100, 100, 120], dtype=np.float32)  # Dark gray
        elif params.cloud_coverage_pct > 70:
            base_sky = np.array([180, 180, 190], dtype=np.float32)  # Light gray
        elif params.sun_elevation_deg < 20:
            # Golden hour colors
            base_sky = np.array([255, 165, 100], dtype=np.float32)
        elif params.sun_elevation_deg < 40:
            # Warm daylight
            base_sky = np.array([200, 220, 255], dtype=np.float32)
        
        # Adjust for sun elevation (brightness)
        sun_factor = params.sun_elevation_deg / 90.0
        brightness_factor = 0.5 + 0.5 * sun_factor
        base_sky *= brightness_factor
        
        # Add atmospheric haze effect
        if params.haze_density > 0.1:
            haze_color = np.array([200, 200, 200], dtype=np.float32)
            base_sky = base_sky * (1 - params.haze_density * 0.3) + haze_color * params.haze_density * 0.3
        
        return np.clip(base_sky, 0, 255)
    
    def _generate_realistic_water(self, normalized_depth: np.ndarray, 
                                params: AugmentationParameters, 
                                water_mask: np.ndarray) -> np.ndarray:
        """Generate realistic water colors with wave patterns."""
        height, width = normalized_depth.shape
        
        # Get water region coordinates
        y_coords, x_coords = np.where(water_mask)
        
        if len(y_coords) == 0:
            return np.array([])
        
        # Base water color (deep ocean blue)
        base_water = np.array([64, 164, 223], dtype=np.float32)
        
        # Adjust base color for lighting conditions
        sun_factor = params.sun_elevation_deg / 90.0
        base_water *= (0.3 + 0.7 * sun_factor)  # Darker water in low light
        
        # Create wave patterns based on wave parameters
        wave_amplitude = params.dominant_wave_height_m * 20
        wave_frequency = 2 * np.pi / params.wavelength_m * 100  # Scale for pixel space
        
        # Generate wave pattern
        wave_pattern = np.sin(y_coords * wave_frequency) * wave_amplitude
        
        # Add directional spread
        if params.directional_spread_deg > 0:
            angle_variation = np.sin(x_coords * wave_frequency * 0.5) * params.directional_spread_deg / 45.0
            wave_pattern += angle_variation * wave_amplitude * 0.3
        
        # Create color variations based on wave pattern
        colors = np.tile(base_water, (len(y_coords), 1))
        
        # Wave crests are lighter (sun reflection)
        crest_mask = wave_pattern > wave_amplitude * 0.3
        colors[crest_mask] += 40  # Brighter
        
        # Wave troughs are darker
        trough_mask = wave_pattern < -wave_amplitude * 0.3
        colors[trough_mask] -= 30  # Darker
        
        # Add surface roughness effects
        if params.surface_roughness > 0.3:
            roughness_noise = np.random.normal(0, params.surface_roughness * 20, len(y_coords))
            colors += roughness_noise[:, np.newaxis]
        
        # Add specular highlights
        if params.specular_highlight_intensity > 0.5:
            # Simulate sun reflection on water
            sun_reflection_prob = params.specular_highlight_intensity * 0.1
            reflection_mask = np.random.random(len(y_coords)) < sun_reflection_prob
            colors[reflection_mask] = [200, 220, 255]  # Bright reflection
        
        return np.clip(colors, 0, 255).astype(np.float32)
    
    def _get_realistic_beach_color(self, params: AugmentationParameters) -> np.ndarray:
        """Generate realistic beach/sand colors."""
        # Base sand color
        base_sand = np.array([194, 178, 128], dtype=np.float32)
        
        # Adjust for wetness
        if params.wet_sand_reflectivity > 0.5:
            # Wet sand is darker and more reflective
            base_sand *= (0.6 + 0.2 * (1 - params.wet_sand_reflectivity))
        
        # Adjust for lighting
        sun_factor = params.sun_elevation_deg / 90.0
        base_sand *= (0.4 + 0.6 * sun_factor)
        
        return np.clip(base_sand, 0, 255)
    
    def _add_realistic_wave_foam(self, rgb_image: np.ndarray, 
                               normalized_depth: np.ndarray, 
                               params: AugmentationParameters) -> np.ndarray:
        """Add realistic wave foam patterns based on breaking behavior."""
        if params.foam_coverage_pct < 5:
            return rgb_image
        
        height, width = rgb_image.shape[:2]
        
        # Create foam mask based on depth gradients (wave breaking areas)
        grad_y, grad_x = np.gradient(normalized_depth)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient areas indicate breaking waves
        foam_threshold = np.percentile(gradient_magnitude, 98 - params.foam_coverage_pct * 0.5)
        foam_mask = gradient_magnitude > foam_threshold
        
        # Create realistic foam colors with variation
        foam_base = np.array([255, 255, 255], dtype=np.float32)  # Pure white
        foam_variation = np.array([240, 248, 255], dtype=np.float32)  # Slightly blue-white
        
        # Add foam with intensity variation based on breaker type
        if params.breaking_type == "plunging":
            # Intense white foam
            rgb_image[foam_mask] = foam_base
        elif params.breaking_type == "spilling":
            # More gradual foam with variation
            foam_intensity = np.random.uniform(0.7, 1.0, np.sum(foam_mask))
            rgb_image[foam_mask] = foam_base * foam_intensity[:, np.newaxis]
        else:
            # Mixed foam patterns
            rgb_image[foam_mask] = foam_variation
        
        # Add micro-foam density effects
        if params.micro_foam_density > 0.3:
            # Add small foam bubbles in water areas
            water_mask = (normalized_depth >= 0.3) & (normalized_depth <= 0.7)
            micro_foam_prob = params.micro_foam_density * 0.05
            
            y_water, x_water = np.where(water_mask)
            if len(y_water) > 0:
                micro_foam_mask = np.random.random(len(y_water)) < micro_foam_prob
                if np.any(micro_foam_mask):
                    foam_indices = (y_water[micro_foam_mask], x_water[micro_foam_mask])
                    rgb_image[foam_indices] = rgb_image[foam_indices] * 0.7 + foam_variation * 0.3
        
        return rgb_image
    
    def _apply_depth_based_lighting(self, rgb_image: np.ndarray, 
                                  normalized_depth: np.ndarray, 
                                  params: AugmentationParameters) -> np.ndarray:
        """Apply realistic lighting effects based on depth and sun position."""
        
        # Calculate lighting intensity based on sun elevation
        base_intensity = 0.3 + 0.7 * (params.sun_elevation_deg / 90.0)
        base_intensity *= params.light_intensity
        
        # Create depth-based lighting gradient
        # Closer objects (lower depth values) are brighter
        lighting_gradient = 0.7 + 0.3 * (1 - normalized_depth)
        
        # Apply sun azimuth effects (side lighting)
        height, width = rgb_image.shape[:2]
        x_coords = np.arange(width)
        
        # Convert sun azimuth to lighting direction
        sun_direction = np.cos(np.radians(params.sun_azimuth_deg))
        side_lighting = 0.9 + 0.1 * sun_direction * (x_coords / width - 0.5) * 2
        
        # Combine lighting effects
        total_lighting = base_intensity * lighting_gradient * side_lighting[np.newaxis, :]
        
        # Apply lighting
        rgb_image *= total_lighting[:, :, np.newaxis]
        
        # Add sun glare effects
        if params.sun_glare_probability > 0.6:
            # Create glare pattern
            center_x = int(width * (0.3 + 0.4 * (params.sun_azimuth_deg / 360.0)))
            center_y = int(height * (0.1 + 0.3 * (1 - params.sun_elevation_deg / 90.0)))
            
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            glare_radius = min(width, height) * 0.2
            glare_mask = distance < glare_radius
            
            if np.any(glare_mask):
                glare_intensity = (1 - distance / glare_radius) * params.sun_glare_probability * 0.5
                glare_intensity = np.clip(glare_intensity, 0, 1)
                
                # Apply glare (brighten affected areas)
                rgb_image[glare_mask] += glare_intensity[glare_mask, np.newaxis] * 100
        
        return rgb_image
    
    def _add_water_surface_effects(self, rgb_image: np.ndarray, 
                                 normalized_depth: np.ndarray, 
                                 params: AugmentationParameters) -> np.ndarray:
        """Add realistic water surface effects like ripples and wind streaks."""
        
        water_mask = (normalized_depth >= 0.3) & (normalized_depth <= 0.7)
        
        if not np.any(water_mask):
            return rgb_image
        
        # Add wind streaks
        if params.wind_streak_visibility > 0.3:
            height, width = rgb_image.shape[:2]
            
            # Create wind streak pattern
            streak_frequency = params.ripples_frequency_hz * 0.01
            y_coords, x_coords = np.where(water_mask)
            
            if len(y_coords) > 0:
                streak_pattern = np.sin(x_coords * streak_frequency) * params.wind_streak_visibility
                
                # Apply streaks as brightness variation
                streak_effect = 1.0 + streak_pattern * 0.1
                rgb_image[water_mask] *= streak_effect[:, np.newaxis]
        
        # Add surface ripples
        if params.ripples_frequency_hz > 10:
            y_coords, x_coords = np.where(water_mask)
            
            if len(y_coords) > 0:
                ripple_freq = params.ripples_frequency_hz * 0.001
                ripple_pattern = (np.sin(y_coords * ripple_freq) + 
                                np.sin(x_coords * ripple_freq * 0.7)) * 0.5
                
                # Apply ripples as subtle color variation
                ripple_effect = 1.0 + ripple_pattern * 0.05
                rgb_image[water_mask] *= ripple_effect[:, np.newaxis]
        
        return rgb_image
    
    def _apply_atmospheric_perspective(self, rgb_image: np.ndarray, 
                                     normalized_depth: np.ndarray, 
                                     params: AugmentationParameters) -> np.ndarray:
        """Apply atmospheric perspective effects (distant objects are hazier)."""
        
        # Distant objects (higher depth values) get more atmospheric haze
        if params.haze_density > 0.1:
            haze_color = np.array([200, 200, 200], dtype=np.float32)
            
            # Haze increases with distance
            haze_factor = params.haze_density * normalized_depth * 0.4
            
            # Apply haze
            rgb_image = (rgb_image * (1 - haze_factor[:, :, np.newaxis]) + 
                        haze_color * haze_factor[:, :, np.newaxis])
        
        # Apply contrast attenuation with distance
        if params.contrast_attenuation > 0.1:
            mean_color = np.mean(rgb_image, axis=(0, 1))
            
            # Reduce contrast for distant objects
            attenuation_factor = 1 - params.contrast_attenuation * normalized_depth * 0.3
            
            rgb_image = (mean_color + 
                        (rgb_image - mean_color) * attenuation_factor[:, :, np.newaxis])
        
        return rgb_image
    
    def _add_realistic_texture(self, rgb_image: np.ndarray, 
                             params: AugmentationParameters) -> np.ndarray:
        """Add realistic texture and sensor effects."""
        
        # Add subtle sensor noise
        if params.sensor_noise_level > 0:
            noise_std = params.sensor_noise_level * 10  # Reduced noise for better quality
            noise = np.random.normal(0, noise_std, rgb_image.shape).astype(np.float32)
            rgb_image += noise
        
        # Add motion blur if specified
        if params.motion_blur_kernel_size > 0:
            kernel_size = int(params.motion_blur_kernel_size)
            if kernel_size > 1:
                # Apply subtle motion blur
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
                
                # Convert to uint8 for cv2 processing
                temp_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
                blurred = cv2.filter2D(temp_image, -1, kernel)
                rgb_image = blurred.astype(np.float32)
        
        return rgb_image
    
    def _post_process_for_realism(self, synthetic_image: np.ndarray, 
                                params: AugmentationParameters) -> np.ndarray:
        """Post-process synthetic image for enhanced photorealism."""
        try:
            # Convert to float for processing
            image = synthetic_image.astype(np.float32)
            
            # Enhance water clarity and depth
            image = self._enhance_water_clarity(image, params)
            
            # Improve wave detail and texture
            image = self._enhance_wave_details(image, params)
            
            # Adjust color balance for natural beach tones
            image = self._adjust_natural_color_balance(image, params)
            
            # Add subtle film grain for photorealism
            image = self._add_photographic_grain(image, params)
            
            # Final contrast and saturation adjustment
            image = self._final_realism_adjustments(image, params)
            
            return np.clip(image, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Post-processing failed, returning original: {e}")
            return synthetic_image
    
    def _enhance_water_clarity(self, image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
        """Enhance water clarity and transparency effects."""
        height, width = image.shape[:2]
        
        # Create water mask (assume water is in blue-dominant areas)
        blue_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 0]
        
        # Water areas where blue > red and blue > green
        water_mask = (blue_channel > red_channel + 20) & (blue_channel > green_channel + 10)
        
        if np.any(water_mask):
            # Enhance blue saturation in water areas
            image[water_mask, 2] = np.clip(image[water_mask, 2] * 1.1, 0, 255)
            
            # Add subtle green tint for natural water color
            image[water_mask, 1] = np.clip(image[water_mask, 1] * 1.05, 0, 255)
            
            # Reduce red slightly for cleaner water
            image[water_mask, 0] = np.clip(image[water_mask, 0] * 0.95, 0, 255)
        
        return image
    
    def _enhance_wave_details(self, image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
        """Enhance wave structure and foam details."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect wave edges
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Enhance contrast along wave edges
        edge_mask = edges > 0
        if np.any(edge_mask):
            # Increase contrast at wave boundaries
            for channel in range(3):
                channel_data = image[:, :, channel]
                mean_val = np.mean(channel_data[edge_mask])
                
                # Brighten above-average pixels, darken below-average
                above_mean = channel_data[edge_mask] > mean_val
                below_mean = channel_data[edge_mask] <= mean_val
                
                edge_pixels = channel_data[edge_mask]
                edge_pixels[above_mean] = np.clip(edge_pixels[above_mean] * 1.1, 0, 255)
                edge_pixels[below_mean] = np.clip(edge_pixels[below_mean] * 0.9, 0, 255)
                
                channel_data[edge_mask] = edge_pixels
                image[:, :, channel] = channel_data
        
        return image
    
    def _adjust_natural_color_balance(self, image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
        """Adjust color balance for natural beach photography look."""
        
        # Warm up the image slightly for natural sunlight
        if params.sun_elevation_deg > 30:
            # Add warmth in highlights
            highlights = image > 180
            if np.any(highlights):
                # Apply color adjustment per channel
                image[:, :, 0][highlights[:, :, 0]] *= 1.02  # Red
                image[:, :, 1][highlights[:, :, 1]] *= 1.01  # Green  
                image[:, :, 2][highlights[:, :, 2]] *= 0.98  # Blue
                image = np.clip(image, 0, 255)
        
        # Cool down shadows for natural contrast
        shadows = image < 80
        if np.any(shadows):
            # Apply color adjustment per channel
            image[:, :, 0][shadows[:, :, 0]] *= 0.98  # Red
            image[:, :, 1][shadows[:, :, 1]] *= 0.99  # Green
            image[:, :, 2][shadows[:, :, 2]] *= 1.02  # Blue
            image = np.clip(image, 0, 255)
        
        # Enhance sky blues if present
        sky_mask = (image[:, :, 2] > 150) & (image[:, :, 1] > 120) & (image[:, :, 0] < 140)
        if np.any(sky_mask):
            image[:, :, 2][sky_mask] = np.clip(image[:, :, 2][sky_mask] * 1.05, 0, 255)  # More blue
            image[:, :, 0][sky_mask] = np.clip(image[:, :, 0][sky_mask] * 0.95, 0, 255)  # Less red
        
        return image
    
    def _add_photographic_grain(self, image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
        """Add subtle film grain for photographic realism."""
        # Very subtle grain - much less than sensor noise
        grain_strength = 0.5  # Very light grain
        
        # Generate grain pattern
        grain = np.random.normal(0, grain_strength, image.shape).astype(np.float32)
        
        # Apply grain more to mid-tones, less to highlights and shadows
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # Create grain mask (stronger in mid-tones)
        grain_mask = 1.0 - np.abs(luminance - 128) / 128.0
        grain_mask = np.clip(grain_mask, 0.2, 1.0)  # Minimum grain level
        
        # Apply grain
        for channel in range(3):
            image[:, :, channel] += grain[:, :, channel] * grain_mask
        
        return image
    
    def _final_realism_adjustments(self, image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
        """Final adjustments for photorealism."""
        
        # Subtle contrast enhancement
        image = np.clip((image - 128) * 1.05 + 128, 0, 255)
        
        # Slight saturation boost for vibrant but natural colors
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Boost saturation slightly
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
        
        # Convert back to RGB
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        return image
    
    def _assess_generation_quality(self, synthetic_image: np.ndarray, 
                                 depth_map: np.ndarray) -> float:
        """Assess quality of generated synthetic image with enhanced metrics."""
        try:
            # Check basic image properties
            if synthetic_image is None or synthetic_image.size == 0:
                return 0.0
            
            # Ensure image is in correct format
            if len(synthetic_image.shape) != 3 or synthetic_image.shape[2] != 3:
                return 0.2
            
            # Check for reasonable color variation (avoid uniform images)
            color_std = np.std(synthetic_image)
            if color_std < 5:  # Too uniform
                return 0.2
            elif color_std < 15:
                variation_score = 0.4
            else:
                variation_score = min(color_std / 40.0, 1.0)  # Normalize to 0-1
            
            # Check for proper depth correlation
            gray_image = cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2GRAY)
            
            # Resize if shapes don't match
            if gray_image.shape != depth_map.shape:
                gray_image = cv2.resize(gray_image, (depth_map.shape[1], depth_map.shape[0]))
            
            correlation = np.corrcoef(gray_image.flatten(), depth_map.flatten())[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            correlation_score = abs(correlation)  # Absolute correlation
            
            # Check color distribution (realistic beach colors)
            # Beach images should have blues (water), browns/yellows (sand), whites (foam)
            mean_colors = np.mean(synthetic_image, axis=(0, 1))
            
            # Check for presence of expected color ranges
            has_blue_water = mean_colors[2] > mean_colors[0] and mean_colors[2] > mean_colors[1]  # Blue channel dominant
            has_varied_colors = np.std(mean_colors) > 10  # Color variation across channels
            
            color_realism_score = 0.5
            if has_blue_water:
                color_realism_score += 0.3
            if has_varied_colors:
                color_realism_score += 0.2
            
            # Check for edge preservation (important for wave structure)
            edges_synthetic = cv2.Canny(gray_image, 50, 150)
            edges_depth = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
            
            edge_correlation = np.corrcoef(edges_synthetic.flatten(), edges_depth.flatten())[0, 1]
            if np.isnan(edge_correlation):
                edge_correlation = 0.0
            
            edge_score = abs(edge_correlation)
            
            # Combine all quality metrics
            overall_quality = (
                variation_score * 0.25 +      # Color variation
                correlation_score * 0.25 +    # Depth correlation
                color_realism_score * 0.25 +  # Color realism
                edge_score * 0.25             # Edge preservation
            )
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Failed to assess generation quality: {e}")
            return 0.4  # Default moderate quality for enhanced fallback
    
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
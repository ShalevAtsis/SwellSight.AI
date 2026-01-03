#!/usr/bin/env python3
"""
Improved synthetic data generation with better ControlNet integration and beach-specific prompts.

This script enhances the synthetic image quality by:
1. Using proper ControlNet models
2. Beach-specific prompts
3. Better depth map preprocessing
4. Quality filtering
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datetime import datetime

# Check if diffusers is available for proper ControlNet
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    logging.warning("Diffusers not available. Install with: pip install diffusers transformers accelerate")

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor
from swellsight.data.controlnet_generator import AugmentationParameterSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedControlNetGenerator:
    """Enhanced ControlNet generator with beach-specific optimizations."""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.controlnet_pipeline = None
        
        if CONTROLNET_AVAILABLE and self.device == "cuda":
            self._initialize_controlnet()
        else:
            logger.warning("Using enhanced fallback generation (install diffusers + use GPU for ControlNet)")
    
    def _initialize_controlnet(self):
        """Initialize proper ControlNet pipeline."""
        try:
            logger.info("Loading ControlNet depth model...")
            
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
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
                self.controlnet_pipeline.enable_model_cpu_offload()
            
            logger.info("ControlNet pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ControlNet: {e}")
            self.controlnet_pipeline = None
    
    def generate_beach_image(self, depth_map, augmentation_params):
        """Generate high-quality beach image from depth map."""
        
        if self.controlnet_pipeline:
            return self._generate_with_controlnet(depth_map, augmentation_params)
        else:
            return self._generate_enhanced_fallback(depth_map, augmentation_params)
    
    def _generate_with_controlnet(self, depth_map, augmentation_params):
        """Generate using actual ControlNet."""
        try:
            # Prepare depth map for ControlNet
            depth_image = self._prepare_depth_for_controlnet(depth_map)
            
            # Generate beach-specific prompt
            prompt = self._create_beach_prompt(augmentation_params)
            negative_prompt = self._create_negative_prompt()
            
            logger.info(f"Generating with prompt: {prompt[:100]}...")
            
            # Generate image
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            # Convert to numpy array
            synthetic_image = np.array(result.images[0])
            
            return synthetic_image, {
                'method': 'controlnet',
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'controlnet_conditioning_scale': 1.0
            }
            
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            return self._generate_enhanced_fallback(depth_map, augmentation_params)
    
    def _prepare_depth_for_controlnet(self, depth_map):
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
        
        # Resize to standard size if needed
        if depth_image.size != (512, 512):
            depth_image = depth_image.resize((512, 512), Image.LANCZOS)
        
        return depth_image
    
    def _create_beach_prompt(self, augmentation_params):
        """Create detailed beach-specific prompt based on augmentation parameters."""
        
        # Base beach scene
        prompt_parts = [
            "photorealistic beach camera view",
            "ocean waves breaking on shore",
            "natural beach scene"
        ]
        
        # Wave characteristics
        wave_height = augmentation_params.dominant_wave_height_m
        if wave_height < 1.0:
            prompt_parts.append("small gentle waves")
        elif wave_height < 2.0:
            prompt_parts.append("medium waves")
        else:
            prompt_parts.append("large powerful waves")
        
        # Breaking type
        breaking_type = augmentation_params.breaking_type
        if breaking_type == "spilling":
            prompt_parts.append("white foam spilling down wave faces")
        elif breaking_type == "plunging":
            prompt_parts.append("waves curling and plunging")
        elif breaking_type == "collapsing":
            prompt_parts.append("waves collapsing with white water")
        
        # Foam coverage
        if augmentation_params.foam_coverage_pct > 50:
            prompt_parts.append("lots of white foam and bubbles")
        elif augmentation_params.foam_coverage_pct > 20:
            prompt_parts.append("white foam on water surface")
        
        # Lighting conditions
        sun_elevation = augmentation_params.sun_elevation_deg
        if sun_elevation > 60:
            prompt_parts.append("bright sunny day, high sun")
        elif sun_elevation > 30:
            prompt_parts.append("good lighting, sun visible")
        else:
            prompt_parts.append("low sun angle, golden light")
        
        # Weather conditions
        if augmentation_params.rain_present:
            prompt_parts.append("overcast rainy conditions")
        elif augmentation_params.cloud_coverage_pct > 70:
            prompt_parts.append("cloudy sky")
        else:
            prompt_parts.append("clear blue sky")
        
        # Camera perspective
        if augmentation_params.camera_height_m > 20:
            prompt_parts.append("elevated camera view")
        else:
            prompt_parts.append("beach level camera view")
        
        # Quality enhancers
        prompt_parts.extend([
            "high resolution",
            "sharp focus",
            "natural colors",
            "realistic water texture",
            "detailed wave structure"
        ])
        
        return ", ".join(prompt_parts)
    
    def _create_negative_prompt(self):
        """Create negative prompt to avoid unwanted elements."""
        return (
            "cartoon, anime, painting, drawing, sketch, "
            "low quality, blurry, distorted, unrealistic, "
            "people, buildings, cars, text, watermark, "
            "oversaturated, artificial colors, "
            "bad anatomy, deformed waves"
        )
    
    def _generate_enhanced_fallback(self, depth_map, augmentation_params):
        """Enhanced fallback generation with better beach realism."""
        height, width = depth_map.shape
        
        # Normalize depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min < 1e-6:
            normalized_depth = np.full_like(depth_map, 0.5)
        else:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Create realistic beach scene
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky region (far/high depths)
        sky_mask = normalized_depth > 0.7
        sky_color = self._get_realistic_sky_color(augmentation_params)
        rgb_image[sky_mask] = sky_color
        
        # Water region with wave patterns
        water_mask = (normalized_depth >= 0.3) & (normalized_depth <= 0.7)
        water_colors = self._generate_realistic_water(
            normalized_depth[water_mask], 
            augmentation_params,
            np.where(water_mask)
        )
        rgb_image[water_mask] = water_colors
        
        # Beach/shore region (close/low depths)
        beach_mask = normalized_depth < 0.3
        beach_color = self._get_realistic_beach_color(augmentation_params)
        rgb_image[beach_mask] = beach_color
        
        # Add wave foam and texture
        rgb_image = self._add_wave_foam(rgb_image, normalized_depth, augmentation_params)
        
        # Add realistic noise and texture
        rgb_image = self._add_realistic_texture(rgb_image, augmentation_params)
        
        return rgb_image, {'method': 'enhanced_fallback'}
    
    def _get_realistic_sky_color(self, params):
        """Generate realistic sky colors based on conditions."""
        base_sky = np.array([135, 206, 235])  # Sky blue
        
        # Adjust for weather
        if params.rain_present:
            return np.array([100, 100, 120])  # Dark gray
        elif params.cloud_coverage_pct > 70:
            return np.array([180, 180, 190])  # Light gray
        elif params.sun_elevation_deg < 20:
            return np.array([255, 165, 100])  # Golden hour
        
        return base_sky
    
    def _generate_realistic_water(self, normalized_depths, params, positions):
        """Generate realistic water colors with wave patterns."""
        base_water = np.array([64, 164, 223])  # Ocean blue
        
        # Create wave patterns based on depth variation
        y_positions, x_positions = positions
        
        # Add wave height variation
        wave_pattern = np.sin(y_positions * 0.1) * params.dominant_wave_height_m * 10
        
        # Adjust colors based on wave pattern
        colors = np.tile(base_water, (len(normalized_depths), 1))
        
        # Lighter colors for wave crests
        crest_mask = wave_pattern > 0
        colors[crest_mask] = np.clip(colors[crest_mask] + 30, 0, 255)
        
        # Darker colors for troughs
        trough_mask = wave_pattern < -5
        colors[trough_mask] = np.clip(colors[trough_mask] - 20, 0, 255)
        
        return colors.astype(np.uint8)
    
    def _get_realistic_beach_color(self, params):
        """Generate realistic beach/sand colors."""
        base_sand = np.array([194, 178, 128])  # Sand color
        
        # Adjust for wetness
        if params.wet_sand_reflectivity > 0.5:
            return np.clip(base_sand * 0.7, 0, 255).astype(np.uint8)  # Darker wet sand
        
        return base_sand
    
    def _add_wave_foam(self, image, normalized_depth, params):
        """Add realistic wave foam patterns."""
        if params.foam_coverage_pct < 10:
            return image
        
        # Create foam mask based on depth gradients (wave breaking areas)
        grad_y, grad_x = np.gradient(normalized_depth)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient areas are likely breaking waves
        foam_threshold = np.percentile(gradient_magnitude, 95 - params.foam_coverage_pct)
        foam_mask = gradient_magnitude > foam_threshold
        
        # Add white foam
        foam_color = np.array([255, 255, 255])
        image[foam_mask] = foam_color
        
        return image
    
    def _add_realistic_texture(self, image, params):
        """Add realistic texture and noise."""
        # Add subtle noise
        noise_level = params.sensor_noise_level * 255
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        
        # Apply noise
        noisy_image = image.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image


def main():
    parser = argparse.ArgumentParser(description='Improved synthetic beach image generation')
    parser.add_argument('--input', '-i', required=True, help='Input directory with real beach images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for improved synthetic images')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum number of images to process')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(input_path.glob(ext))
    
    if not image_paths:
        logger.error(f"No images found in {input_path}")
        return 1
    
    # Limit images
    image_paths = image_paths[:args.max_images]
    logger.info(f"Processing {len(image_paths)} images with improved quality")
    
    # Initialize components
    depth_extractor = MiDaSDepthExtractor(device=args.device)
    param_system = AugmentationParameterSystem()
    improved_generator = ImprovedControlNetGenerator(device=args.device)
    
    # Process images
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            # Extract depth
            depth_result = depth_extractor.extract_depth(str(image_path))
            
            if depth_result.depth_quality_score < 0.3:
                logger.warning(f"Low depth quality, skipping: {image_path.name}")
                continue
            
            # Generate augmentation parameters
            augmentation_params = param_system.generate_random_parameters()
            
            # Generate improved synthetic image
            synthetic_image, metadata = improved_generator.generate_beach_image(
                depth_result.depth_map, 
                augmentation_params
            )
            
            # Save result
            output_filename = f"{image_path.stem}_improved_synthetic.jpg"
            output_file = output_path / output_filename
            
            Image.fromarray(synthetic_image).save(output_file, quality=95)
            
            logger.info(f"Saved improved synthetic image: {output_file}")
            logger.info(f"Generation method: {metadata['method']}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue
    
    logger.info("Improved synthetic generation complete!")
    return 0


if __name__ == '__main__':
    exit(main())
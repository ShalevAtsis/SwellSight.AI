# ControlNet Synthetic Generation Guide

This guide covers ControlNet-based synthetic image generation for beach camera scenes in the SwellSight Wave Analysis Model.

## Overview

The ControlNet synthetic generator uses Stable Diffusion with depth conditioning to create photorealistic synthetic beach images from MiDaS-extracted depth maps. This enables the generation of diverse training data with precise control over wave parameters and environmental conditions.

## Architecture

The synthetic generation system consists of three main components:

1. **AugmentationParameterSystem**: Generates realistic parameter combinations across 10 categories
2. **ControlNetSyntheticGenerator**: Creates synthetic images using depth conditioning
3. **Prompt Engineering**: Converts parameters to descriptive text prompts

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (recommended)
- Stable Diffusion and ControlNet models
- Sufficient GPU memory (8GB+ recommended)

### Installation

```bash
# Install SwellSight with ControlNet dependencies
pip install -e .

# Additional dependencies for ControlNet (if using actual implementation)
pip install diffusers transformers accelerate
```

## Basic Usage

### Simple Synthetic Generation

```python
from swellsight.data.controlnet_generator import (
    ControlNetSyntheticGenerator, 
    AugmentationParameterSystem
)
import numpy as np

# Initialize components
param_system = AugmentationParameterSystem(seed=42)
generator = ControlNetSyntheticGenerator(
    controlnet_model="lllyasviel/sd-controlnet-depth",
    device="cuda"
)

# Generate random parameters
params = param_system.generate_random_parameters()

# Create synthetic image from depth map
depth_map = np.load("depth_map.npy")  # Your depth map
result = generator.generate_synthetic_image(
    depth_map=depth_map,
    augmentation_params=params,
    prompt="beach camera view with breaking waves"
)

print(f"Generated image shape: {result.synthetic_image.shape}")
print(f"Quality score: {result.quality_score:.3f}")
```

### Batch Generation

```python
# Generate multiple parameter sets
param_sets = [param_system.generate_random_parameters() for _ in range(10)]
depth_maps = [np.load(f"depth_{i}.npy") for i in range(10)]

# Batch generate synthetic images
results = generator.batch_generate(depth_maps, param_sets)

for i, result in enumerate(results):
    print(f"Image {i}: Quality {result.quality_score:.3f}")
```

## Augmentation Parameter System

The system provides comprehensive control over beach camera scene variations through 10 categories:

### 1. Camera View Geometry

Controls the camera's physical position and orientation:

```python
# Example parameters
params.camera_height_m = 15.0        # 1-50m above sea level
params.tilt_angle_deg = 10.0         # -10° to +30° toward horizon
params.horizontal_fov_deg = 60.0     # 30-120° field of view
params.distance_to_breaking_m = 100.0 # 10-500m to breaking zone
params.lateral_offset_m = 20.0       # -100m to +100m from center
```

### 2. Wave Field Structure

Defines the wave characteristics:

```python
params.dominant_wave_height_m = 2.5   # 0.3-4.0m wave height
params.wavelength_m = 80.0           # 5-200m wavelength
params.wave_period_s = 12.0          # 3-20s wave period
params.directional_spread_deg = 15.0  # 0-45° directional spread
params.visible_wave_fronts = 3       # 1-10 visible wave fronts
```

### 3. Breaking Behavior

Controls wave breaking characteristics:

```python
params.breaking_type = "plunging"     # spilling, plunging, collapsing, surging
params.breaker_intensity = 0.8       # 0.0-1.0 breaking intensity
params.crest_sharpness = 0.6         # 0.0-1.0 crest sharpness
params.foam_coverage_pct = 25.0      # 0-100% foam coverage
```

### 4. Shore Interaction

Models wave-shore interaction:

```python
params.beach_slope_deg = 5.0         # 1-45° beach slope
params.runup_distance_m = 15.0       # 0-50m wave run-up
params.backwash_visible = True       # backwash visibility
params.wet_sand_reflectivity = 0.7   # 0.0-1.0 sand reflectivity
params.shoreline_curvature = 0.02    # -0.1 to +0.1 curvature
```

### 5. Water Surface Texture

Controls water surface appearance:

```python
params.surface_roughness = 0.4       # 0.0-1.0 surface roughness
params.ripples_frequency_hz = 20.0   # 0-100 Hz ripple frequency
params.wind_streak_visibility = 0.3  # 0.0-1.0 wind streak visibility
params.specular_highlight_intensity = 0.8 # 0.0-1.0 highlight intensity
params.micro_foam_density = 0.2      # 0.0-1.0 micro-foam density
```

### 6. Lighting and Sun Position

Controls lighting conditions:

```python
params.sun_elevation_deg = 45.0      # 0-90° sun elevation
params.sun_azimuth_deg = 180.0       # 0-360° sun azimuth
params.light_intensity = 1.2         # 0.0-2.0 light intensity
params.shadow_softness = 0.6         # 0.0-1.0 shadow softness
params.sun_glare_probability = 0.3   # 0.0-1.0 glare probability
```

### 7. Atmospheric Conditions

Models atmospheric effects:

```python
params.haze_density = 0.2            # 0.0-1.0 haze density
params.fog_layer_height_m = 0.0      # 0-100m fog height
params.humidity_level = 0.6          # 0.0-1.0 humidity level
params.sky_clarity = "partly_cloudy" # clear, partly_cloudy, overcast, stormy
params.contrast_attenuation = 0.1    # 0.0-1.0 contrast reduction
```

### 8. Weather State

Controls weather conditions:

```python
params.cloud_coverage_pct = 40.0     # 0-100% cloud coverage
params.cloud_type = "cumulus"        # cumulus, stratus, cirrus, cumulonimbus
params.rain_present = False          # rain presence
params.rain_streak_intensity = 0.0   # 0.0-1.0 rain streak intensity
params.storminess = 0.2              # 0.0-1.0 storm intensity
```

### 9. Optical and Sensor Artifacts

Simulates camera artifacts:

```python
params.lens_distortion_coeff = 0.05  # -0.5 to +0.5 lens distortion
params.motion_blur_kernel_size = 2   # 0-20 pixels motion blur
params.sensor_noise_level = 0.02     # 0.0-0.1 sensor noise
params.compression_artifacts = 0.1   # 0.0-1.0 compression artifacts
params.chromatic_aberration = 0.05   # 0.0-1.0 chromatic aberration
```

### 10. Scene Occlusions and Noise Objects

Adds realistic scene elements:

```python
params.people_count = 2              # 0-20 people in scene
params.surfboard_present = True      # surfboard presence
params.birds_count = 5               # 0-50 birds in scene
params.sea_spray_occlusion_prob = 0.3 # 0.0-1.0 spray occlusion
params.foreground_blur_amount = 3    # 0-10 pixels foreground blur
```

## Parameter Generation Strategies

### Random Generation with Correlations

The system includes realistic parameter correlations:

```python
# Storm conditions affect multiple parameters
if params.storminess > 0.7:
    # Automatically increases cloud coverage, surface roughness, etc.
    pass

# Sun elevation affects lighting
if params.sun_elevation_deg < 20:
    # Reduces glare probability, increases shadow softness
    pass
```

### Custom Parameter Sets

```python
# Create specific scenarios
def create_stormy_scene():
    params = param_system.generate_random_parameters()
    
    # Override for stormy conditions
    params.storminess = 0.9
    params.cloud_coverage_pct = 90.0
    params.sky_clarity = "stormy"
    params.surface_roughness = 0.8
    params.dominant_wave_height_m = 3.5
    
    return params

# Generate stormy scene
stormy_params = create_stormy_scene()
```

### Parameter Validation

```python
# The system automatically validates parameter combinations
def validate_scene_parameters(params):
    """Check if parameters represent a plausible scene."""
    
    # Wave physics validation
    expected_wavelength = 1.56 * params.wave_period_s ** 2
    wavelength_ratio = params.wavelength_m / expected_wavelength
    
    if not (0.5 <= wavelength_ratio <= 2.0):
        print("Warning: Wave period/wavelength mismatch")
    
    # Weather consistency
    if params.rain_present and params.sky_clarity == "clear":
        print("Warning: Rain with clear sky")
    
    return True
```

## Prompt Engineering

### Automatic Prompt Generation

The system automatically generates descriptive prompts from parameters:

```python
# Automatic prompt generation
prompt = generator._generate_prompt_from_params(params)
print(prompt)
# Output: "beach camera view, large waves, plunging waves, stormy weather"
```

### Custom Prompt Templates

```python
def create_detailed_prompt(params):
    """Create detailed prompt from parameters."""
    
    prompt_parts = ["photorealistic beach camera view"]
    
    # Wave description
    if params.dominant_wave_height_m > 2.5:
        prompt_parts.append("massive waves")
    elif params.dominant_wave_height_m > 1.5:
        prompt_parts.append("large waves")
    else:
        prompt_parts.append("moderate waves")
    
    # Breaking style
    prompt_parts.append(f"{params.breaking_type} waves")
    
    # Weather conditions
    if params.storminess > 0.6:
        prompt_parts.append("dramatic storm conditions")
    elif params.sky_clarity == "overcast":
        prompt_parts.append("overcast sky")
    
    # Lighting
    if params.sun_elevation_deg < 15:
        prompt_parts.append("golden hour lighting")
    elif params.sun_elevation_deg > 75:
        prompt_parts.append("bright midday sun")
    
    # Quality modifiers
    prompt_parts.extend([
        "high resolution",
        "detailed water texture",
        "realistic foam patterns",
        "professional photography"
    ])
    
    return ", ".join(prompt_parts)

# Use custom prompt
custom_prompt = create_detailed_prompt(params)
result = generator.generate_synthetic_image(
    depth_map, params, prompt=custom_prompt
)
```

## Quality Assessment

### Automatic Quality Scoring

The generator includes built-in quality assessment:

```python
# Quality metrics
result = generator.generate_synthetic_image(depth_map, params)

print(f"Overall quality: {result.quality_score:.3f}")
print(f"Generation metadata: {result.generation_metadata}")

# Quality interpretation:
# 0.0 - 0.3: Poor quality, regenerate recommended
# 0.3 - 0.6: Moderate quality, usable for training
# 0.6 - 0.8: Good quality, suitable for most purposes
# 0.8 - 1.0: Excellent quality, ideal for training
```

### Quality Factors

Quality assessment considers:

1. **Color Variation**: Sufficient color diversity in the image
2. **Depth Correlation**: Correlation between generated image and input depth map
3. **Structural Consistency**: Preservation of depth map structure
4. **Artifact Detection**: Presence of generation artifacts

### Quality Filtering

```python
def generate_high_quality_batch(depth_maps, param_sets, min_quality=0.6):
    """Generate batch with quality filtering."""
    
    high_quality_results = []
    
    for depth_map, params in zip(depth_maps, param_sets):
        max_attempts = 3
        
        for attempt in range(max_attempts):
            result = generator.generate_synthetic_image(depth_map, params)
            
            if result.quality_score >= min_quality:
                high_quality_results.append(result)
                break
            elif attempt == max_attempts - 1:
                print(f"Failed to generate high-quality image after {max_attempts} attempts")
    
    return high_quality_results
```

## Best Practices

### Parameter Selection

**For Training Data Diversity:**
```python
# Generate diverse parameter sets
def generate_diverse_parameters(count=1000):
    param_system = AugmentationParameterSystem()
    params_list = []
    
    for _ in range(count):
        params = param_system.generate_random_parameters()
        params_list.append(params)
    
    # Verify diversity
    heights = [p.dominant_wave_height_m for p in params_list]
    print(f"Height range: {min(heights):.1f} - {max(heights):.1f}m")
    
    return params_list
```

**For Specific Scenarios:**
```python
# Create scenario-specific generators
def create_calm_conditions():
    params = param_system.generate_random_parameters()
    params.dominant_wave_height_m = np.random.uniform(0.3, 1.0)
    params.surface_roughness = np.random.uniform(0.0, 0.3)
    params.breaker_intensity = np.random.uniform(0.0, 0.4)
    return params

def create_extreme_conditions():
    params = param_system.generate_random_parameters()
    params.dominant_wave_height_m = np.random.uniform(3.0, 4.0)
    params.surface_roughness = np.random.uniform(0.7, 1.0)
    params.breaker_intensity = np.random.uniform(0.8, 1.0)
    return params
```

### Performance Optimization

**GPU Memory Management:**
```python
# For large batches, process in chunks
def generate_large_batch(depth_maps, param_sets, chunk_size=5):
    results = []
    
    for i in range(0, len(depth_maps), chunk_size):
        chunk_depths = depth_maps[i:i+chunk_size]
        chunk_params = param_sets[i:i+chunk_size]
        
        chunk_results = generator.batch_generate(chunk_depths, chunk_params)
        results.extend(chunk_results)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

**Caching and Reuse:**
```python
# Cache generated images
import pickle
from pathlib import Path

def cache_generation_results(results, cache_dir="cache/synthetic"):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        # Save image
        image_path = cache_path / f"synthetic_{i:06d}.npy"
        np.save(image_path, result.synthetic_image)
        
        # Save metadata
        meta_path = cache_path / f"metadata_{i:06d}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'params': result.augmentation_params,
                'metadata': result.generation_metadata,
                'quality': result.quality_score
            }, f)
```

## Integration Examples

### With MiDaS Pipeline

```python
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor

# Complete MiDaS → ControlNet pipeline
def real_to_synthetic_pipeline(real_image_paths, synthetic_count_per_image=5):
    # Extract depth maps
    extractor = MiDaSDepthExtractor()
    depth_results = extractor.batch_extract(real_image_paths)
    
    # Generate synthetic variants
    generator = ControlNetSyntheticGenerator()
    param_system = AugmentationParameterSystem()
    
    all_synthetic_results = []
    
    for depth_result in depth_results:
        if depth_result.depth_quality_score > 0.4:  # Quality threshold
            
            # Generate multiple variants per real image
            for _ in range(synthetic_count_per_image):
                params = param_system.generate_random_parameters()
                
                synthetic_result = generator.generate_synthetic_image(
                    depth_result.depth_map,
                    params
                )
                
                # Add correspondence metadata
                synthetic_result.generation_metadata['source_image'] = depth_result.original_image_path
                synthetic_result.generation_metadata['depth_quality'] = depth_result.depth_quality_score
                
                all_synthetic_results.append(synthetic_result)
    
    return all_synthetic_results
```

### With Training Pipeline

```python
from swellsight.data.synthetic_data_generator import SyntheticDataGenerator

# Integration with training data generation
def generate_training_dataset(real_images_dir, output_dir, target_samples=10000):
    # Load real images
    real_image_paths = list(Path(real_images_dir).glob("*.jpg"))
    
    # Generate synthetic data
    synthetic_results = real_to_synthetic_pipeline(
        real_image_paths, 
        synthetic_count_per_image=target_samples // len(real_image_paths)
    )
    
    # Convert to training format
    training_samples = []
    for result in synthetic_results:
        # Extract ground truth from augmentation parameters
        ground_truth = {
            'wave_height_m': result.augmentation_params.dominant_wave_height_m,
            'wave_type': result.augmentation_params.breaking_type,
            'direction': 'LEFT'  # Derive from directional_spread_deg
        }
        
        training_sample = {
            'image': result.synthetic_image,
            'labels': ground_truth,
            'metadata': result.generation_metadata
        }
        
        training_samples.append(training_sample)
    
    return training_samples
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solutions:
# - Reduce batch size
generator = ControlNetSyntheticGenerator(batch_size=1)

# - Use CPU (slower)
generator = ControlNetSyntheticGenerator(device="cpu")

# - Process in smaller chunks
results = generate_large_batch(depth_maps, param_sets, chunk_size=2)
```

**2. Poor Generation Quality**
```python
# Check parameter validity
def debug_parameters(params):
    print(f"Wave height: {params.dominant_wave_height_m:.2f}m")
    print(f"Breaking type: {params.breaking_type}")
    print(f"Weather: {params.sky_clarity}")
    print(f"Storminess: {params.storminess:.2f}")

# Try different parameter combinations
if result.quality_score < 0.4:
    debug_parameters(params)
    # Generate new parameters
    new_params = param_system.generate_random_parameters()
```

**3. ControlNet Model Loading Issues**
```python
# Fallback to synthetic generation
generator = ControlNetSyntheticGenerator()
if not generator.use_controlnet:
    print("Using fallback synthetic generation")
    # This still creates useful training data
```

### Performance Issues

**Slow Generation:**
- Use GPU acceleration
- Reduce image resolution if acceptable
- Process in batches
- Cache frequently used depth maps

**Memory Issues:**
- Process smaller batches
- Clear GPU cache between batches
- Use gradient checkpointing
- Consider model quantization

## API Reference

### ControlNetSyntheticGenerator

#### Constructor
```python
ControlNetSyntheticGenerator(
    controlnet_model: str = "lllyasviel/sd-controlnet-depth",
    batch_size: int = 1,
    device: Optional[str] = None
)
```

#### Methods

**generate_synthetic_image(depth_map, augmentation_params, prompt=None)**
- Generate single synthetic image
- Returns: `SyntheticGenerationResult`

**batch_generate(depth_maps, param_sets)**
- Generate multiple synthetic images
- Returns: `List[SyntheticGenerationResult]`

### AugmentationParameterSystem

#### Constructor
```python
AugmentationParameterSystem(seed: Optional[int] = None)
```

#### Methods

**generate_random_parameters()**
- Generate random parameter set with correlations
- Returns: `AugmentationParameters`

### Data Classes

**AugmentationParameters**
- Contains all 10 categories of augmentation parameters
- Includes validation and correlation logic

**SyntheticGenerationResult**
- `synthetic_image`: Generated RGB image (numpy array)
- `depth_map`: Input depth map
- `augmentation_params`: Parameters used for generation
- `generation_metadata`: Generation details and settings
- `quality_score`: Quality assessment score (0.0-1.0)

## Examples

See the `examples/` directory for complete examples:
- `examples/basic_synthetic_generation.py`
- `examples/parameter_exploration.py`
- `examples/quality_assessment.py`
- `examples/batch_generation.py`
- `examples/midas_controlnet_pipeline.py`
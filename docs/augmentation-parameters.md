# Augmentation Parameter System

This document provides comprehensive documentation for SwellSight's 10-category augmentation parameter system, which enables systematic generation of diverse synthetic beach camera scenes for training data.

## Overview

The augmentation parameter system provides fine-grained control over synthetic image generation by organizing parameters into 10 distinct categories. Each category controls specific aspects of the beach scene, allowing for realistic and diverse training data generation.

## Parameter Categories

### 1. Camera View Geometry

Controls the camera's position and orientation relative to the beach scene.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `height` | 2.0 - 50.0 | meters | Camera height above sea level |
| `tilt` | -30.0 - 30.0 | degrees | Camera tilt angle (negative = down) |
| `fov` | 30.0 - 120.0 | degrees | Field of view |
| `distance` | 10.0 - 500.0 | meters | Distance from camera to shoreline |
| `offset` | -100.0 - 100.0 | meters | Lateral offset from beach center |

**Example Configuration:**
```json
{
  "camera_view_geometry": {
    "height": 15.0,
    "tilt": -10.0,
    "fov": 60.0,
    "distance": 100.0,
    "offset": 0.0
  }
}
```

### 2. Wave Field Structure

Defines the fundamental wave characteristics and patterns.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `height` | 0.1 - 8.0 | meters | Significant wave height |
| `wavelength` | 10.0 - 200.0 | meters | Dominant wavelength |
| `period` | 3.0 - 20.0 | seconds | Wave period |
| `spread` | 0.0 - 45.0 | degrees | Directional spreading |
| `fronts` | 1 - 5 | count | Number of wave fronts visible |

**Example Configuration:**
```json
{
  "wave_field_structure": {
    "height": 2.5,
    "wavelength": 80.0,
    "period": 8.0,
    "spread": 15.0,
    "fronts": 3
  }
}
```

### 3. Breaking Behavior

Controls wave breaking characteristics and patterns.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `type` | 0 - 3 | enum | Breaking type (0=spilling, 1=plunging, 2=collapsing, 3=surging) |
| `intensity` | 0.0 - 1.0 | ratio | Breaking intensity |
| `sharpness` | 0.0 - 1.0 | ratio | Breaking edge sharpness |
| `foam_coverage` | 0.0 - 0.8 | ratio | Foam coverage percentage |

**Breaking Types:**
- **Spilling (0)**: Gradual foam cascade down wave face
- **Plunging (1)**: Curling wave with air pocket
- **Collapsing (2)**: Steep wave collapsing forward
- **Surging (3)**: Wave rushing up beach without breaking

**Example Configuration:**
```json
{
  "breaking_behavior": {
    "type": 1,
    "intensity": 0.7,
    "sharpness": 0.8,
    "foam_coverage": 0.3
  }
}
```

### 4. Shore Interaction

Defines how waves interact with the shoreline and beach profile.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `slope` | 0.01 - 0.3 | ratio | Beach slope (rise/run) |
| `run_up` | 0.5 - 3.0 | meters | Maximum wave run-up distance |
| `backwash` | 0.0 - 1.0 | ratio | Backwash intensity |
| `reflectivity` | 0.0 - 0.5 | ratio | Wave reflection coefficient |
| `curvature` | -0.1 - 0.1 | 1/meters | Shoreline curvature |

**Example Configuration:**
```json
{
  "shore_interaction": {
    "slope": 0.05,
    "run_up": 1.5,
    "backwash": 0.3,
    "reflectivity": 0.1,
    "curvature": 0.02
  }
}
```

### 5. Water Surface Texture

Controls fine-scale water surface characteristics.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `roughness` | 0.0 - 1.0 | ratio | Overall surface roughness |
| `ripples` | 0.0 - 1.0 | ratio | Small-scale ripple intensity |
| `streaks` | 0.0 - 1.0 | ratio | Wind streak visibility |
| `highlights` | 0.0 - 1.0 | ratio | Specular highlight intensity |
| `micro_foam` | 0.0 - 0.5 | ratio | Micro-foam coverage |

**Example Configuration:**
```json
{
  "water_surface_texture": {
    "roughness": 0.4,
    "ripples": 0.6,
    "streaks": 0.2,
    "highlights": 0.7,
    "micro_foam": 0.1
  }
}
```

### 6. Lighting and Sun Position

Controls illumination conditions and sun position.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `elevation` | 5.0 - 90.0 | degrees | Sun elevation angle |
| `azimuth` | 0.0 - 360.0 | degrees | Sun azimuth angle |
| `intensity` | 0.3 - 1.0 | ratio | Sun intensity |
| `softness` | 0.0 - 1.0 | ratio | Light softness (cloud diffusion) |
| `glare` | 0.0 - 1.0 | ratio | Sun glare intensity |

**Example Configuration:**
```json
{
  "lighting_sun_position": {
    "elevation": 45.0,
    "azimuth": 180.0,
    "intensity": 0.8,
    "softness": 0.3,
    "glare": 0.4
  }
}
```

### 7. Atmospheric Conditions

Defines atmospheric effects and visibility conditions.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `haze` | 0.0 - 1.0 | ratio | Atmospheric haze intensity |
| `fog` | 0.0 - 0.8 | ratio | Fog density |
| `humidity` | 0.3 - 1.0 | ratio | Relative humidity |
| `clarity` | 0.2 - 1.0 | ratio | Atmospheric clarity |
| `attenuation` | 0.0 - 0.5 | ratio | Distance-based attenuation |

**Example Configuration:**
```json
{
  "atmospheric_conditions": {
    "haze": 0.2,
    "fog": 0.0,
    "humidity": 0.7,
    "clarity": 0.8,
    "attenuation": 0.1
  }
}
```

### 8. Weather State

Controls weather-related visual effects.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `clouds` | 0.0 - 1.0 | ratio | Cloud coverage |
| `type` | 0 - 4 | enum | Cloud type (0=clear, 1=cumulus, 2=stratus, 3=storm, 4=overcast) |
| `rain` | 0.0 - 1.0 | ratio | Rain intensity |
| `streaks` | 0.0 - 1.0 | ratio | Rain streak visibility |
| `storminess` | 0.0 - 1.0 | ratio | Overall storm intensity |

**Cloud Types:**
- **Clear (0)**: Minimal cloud coverage
- **Cumulus (1)**: Puffy white clouds
- **Stratus (2)**: Layered gray clouds
- **Storm (3)**: Dark storm clouds
- **Overcast (4)**: Complete cloud coverage

**Example Configuration:**
```json
{
  "weather_state": {
    "clouds": 0.4,
    "type": 1,
    "rain": 0.0,
    "streaks": 0.0,
    "storminess": 0.1
  }
}
```

### 9. Optical/Sensor Artifacts

Simulates camera and sensor-related effects.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `distortion` | 0.0 - 0.3 | ratio | Lens distortion amount |
| `blur` | 0.0 - 2.0 | pixels | Motion/focus blur radius |
| `noise` | 0.0 - 0.1 | ratio | Sensor noise level |
| `compression` | 0.0 - 0.5 | ratio | JPEG compression artifacts |
| `aberration` | 0.0 - 0.2 | ratio | Chromatic aberration |

**Example Configuration:**
```json
{
  "optical_sensor_artifacts": {
    "distortion": 0.05,
    "blur": 0.5,
    "noise": 0.02,
    "compression": 0.1,
    "aberration": 0.03
  }
}
```

### 10. Scene Occlusions/Noise

Controls foreground objects and scene distractions.

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `people` | 0.0 - 0.3 | ratio | People visibility/density |
| `surfboards` | 0.0 - 0.2 | ratio | Surfboard presence |
| `birds` | 0.0 - 0.4 | ratio | Bird activity level |
| `spray` | 0.0 - 1.0 | ratio | Water spray intensity |
| `foreground_blur` | 0.0 - 3.0 | pixels | Foreground blur radius |

**Example Configuration:**
```json
{
  "scene_occlusions_noise": {
    "people": 0.1,
    "surfboards": 0.05,
    "birds": 0.2,
    "spray": 0.4,
    "foreground_blur": 1.0
  }
}
```

## Parameter Sampling Strategies

### Uniform Sampling
```python
from swellsight.data.augmentation_parameter_system import AugmentationParameterSystem

# Create parameter system
param_system = AugmentationParameterSystem()

# Generate uniform random parameters
params = param_system.generate_uniform_parameters()
```

### Gaussian Sampling
```python
# Generate parameters with Gaussian distribution around defaults
params = param_system.generate_gaussian_parameters(
    std_factor=0.3  # Standard deviation as fraction of range
)
```

### Beta Distribution Sampling
```python
# Generate parameters with beta distribution (good for bounded ranges)
params = param_system.generate_beta_parameters(
    alpha=2.0,  # Shape parameter
    beta=2.0    # Shape parameter
)
```

### Custom Sampling
```python
# Generate parameters with custom constraints
constraints = {
    'wave_field_structure.height': (1.0, 4.0),  # Limit wave height
    'lighting_sun_position.elevation': (20.0, 70.0),  # Daytime only
    'weather_state.rain': (0.0, 0.2)  # Light rain only
}

params = param_system.generate_constrained_parameters(constraints)
```

## Parameter Correlation Modeling

The system includes correlation modeling to ensure realistic parameter combinations:

### Physical Correlations
- Higher wave heights correlate with increased breaking intensity
- Storm conditions correlate with increased cloud coverage and rain
- Low sun elevation correlates with increased glare and atmospheric effects

### Example Correlation Rules
```python
# Wave height affects breaking behavior
if params['wave_field_structure']['height'] > 3.0:
    params['breaking_behavior']['intensity'] *= 1.2
    params['breaking_behavior']['foam_coverage'] *= 1.3

# Storm conditions affect multiple categories
if params['weather_state']['storminess'] > 0.7:
    params['atmospheric_conditions']['haze'] *= 1.5
    params['lighting_sun_position']['intensity'] *= 0.7
    params['water_surface_texture']['roughness'] *= 1.4
```

## Parameter Validation

### Range Validation
All parameters are automatically validated against their specified ranges:

```python
# Validation example
def validate_parameters(params):
    for category, category_params in params.items():
        for param_name, value in category_params.items():
            min_val, max_val = get_parameter_range(category, param_name)
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {category}.{param_name} = {value} outside range [{min_val}, {max_val}]")
```

### Physical Plausibility
The system checks for physically plausible parameter combinations:

```python
# Example plausibility checks
def check_plausibility(params):
    # Wave period should be consistent with wavelength
    wavelength = params['wave_field_structure']['wavelength']
    period = params['wave_field_structure']['period']
    expected_period = np.sqrt(wavelength / 1.56)  # Deep water approximation
    
    if abs(period - expected_period) > expected_period * 0.5:
        logger.warning("Wave period inconsistent with wavelength")
```

## Integration with ControlNet

Parameters are automatically translated to ControlNet prompts for image generation:

### Prompt Engineering
```python
def parameters_to_prompt(params):
    prompt_parts = []
    
    # Wave characteristics
    wave_height = params['wave_field_structure']['height']
    if wave_height > 3.0:
        prompt_parts.append("large waves")
    elif wave_height > 1.5:
        prompt_parts.append("medium waves")
    else:
        prompt_parts.append("small waves")
    
    # Breaking behavior
    breaking_type = params['breaking_behavior']['type']
    breaking_names = ['spilling', 'plunging', 'collapsing', 'surging']
    prompt_parts.append(f"{breaking_names[breaking_type]} waves")
    
    # Weather conditions
    cloud_coverage = params['weather_state']['clouds']
    if cloud_coverage > 0.7:
        prompt_parts.append("overcast sky")
    elif cloud_coverage > 0.3:
        prompt_parts.append("partly cloudy")
    else:
        prompt_parts.append("clear sky")
    
    return ", ".join(prompt_parts)
```

## Best Practices

### Parameter Space Exploration
1. **Start with default ranges** for initial experiments
2. **Gradually expand ranges** as model performance improves
3. **Monitor parameter distribution** in training data
4. **Use stratified sampling** to ensure coverage of all parameter combinations

### Quality Control
1. **Validate generated images** against parameter specifications
2. **Monitor correlation preservation** in synthetic data
3. **Check for parameter drift** over time
4. **Implement automated quality scoring**

### Performance Optimization
1. **Cache parameter sets** for repeated use
2. **Batch parameter generation** for efficiency
3. **Use parameter templates** for common scenarios
4. **Implement parameter interpolation** for smooth transitions

## Troubleshooting

### Common Issues

**Issue**: Generated images don't match parameter specifications
- **Solution**: Check prompt engineering logic and ControlNet conditioning strength

**Issue**: Parameter combinations produce unrealistic scenes
- **Solution**: Review correlation rules and add additional plausibility checks

**Issue**: Limited parameter diversity in training data
- **Solution**: Increase sampling ranges and use stratified sampling strategies

**Issue**: Slow parameter generation
- **Solution**: Implement parameter caching and batch processing

### Debugging Tools

```python
# Parameter visualization
param_system.visualize_parameter_distribution(params_list)

# Parameter correlation analysis
param_system.analyze_correlations(params_list)

# Parameter coverage analysis
param_system.check_coverage(params_list, target_coverage=0.95)
```

## API Reference

### AugmentationParameterSystem Class

```python
class AugmentationParameterSystem:
    def __init__(self, config: Optional[Dict] = None)
    def generate_uniform_parameters(self) -> Dict[str, Dict[str, float]]
    def generate_gaussian_parameters(self, std_factor: float = 0.3) -> Dict[str, Dict[str, float]]
    def generate_beta_parameters(self, alpha: float = 2.0, beta: float = 2.0) -> Dict[str, Dict[str, float]]
    def generate_constrained_parameters(self, constraints: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]
    def validate_parameters(self, params: Dict[str, Dict[str, float]]) -> bool
    def check_plausibility(self, params: Dict[str, Dict[str, float]]) -> float
    def parameters_to_prompt(self, params: Dict[str, Dict[str, float]]) -> str
    def visualize_parameters(self, params: Dict[str, Dict[str, float]]) -> None
```

For complete API documentation and examples, see the source code in `swellsight/data/augmentation_parameter_system.py`.
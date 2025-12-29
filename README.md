# SwellSight Wave Analysis Model

A multi-task deep learning system that extracts objective physical wave parameters from single beach camera images.

## Overview

SwellSight uses a shared backbone architecture with specialized task heads to simultaneously predict:
- **Wave Height**: Continuous value in meters (regression)
- **Wave Type**: A-frame, closeout, beach break, point break (classification)  
- **Wave Direction**: Left, right, both (classification)

The model is trained primarily on synthetic data generated from depth maps and validated on real-world beach images.

## Features

- Multi-task learning with shared ConvNeXt backbone
- Synthetic data generation using ControlNet and Stable Diffusion
- Property-based testing for robust validation
- Comprehensive evaluation metrics (MAE, RMSE, accuracy, F1-score)
- Real-world validation on beach camera images
- Easy-to-use inference API

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Install from source
```bash
git clone https://github.com/swellsight/wave-analysis-model.git
cd wave-analysis-model
pip install -e .
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## Quick Start

### Training
```python
from swellsight import WaveAnalysisModel, Trainer, ModelConfig
from swellsight.data import SyntheticDataGenerator, DatasetManager

# Initialize configuration
config = ModelConfig(
    backbone='convnext_base',
    input_size=(768, 768),
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100
)

# Generate synthetic training data
data_generator = SyntheticDataGenerator(config)
data_generator.generate_dataset(num_samples=10000, output_dir="data/synthetic")

# Set up data management
dataset_manager = DatasetManager(config)
train_loader, val_loader = dataset_manager.get_data_loaders("data/synthetic")

# Initialize model and trainer
model = WaveAnalysisModel(config)
trainer = Trainer(model, config)

# Train the model
trainer.train(train_loader, val_loader)
```

### Inference
```python
from swellsight.inference import InferenceEngine

# Load trained model
engine = InferenceEngine.from_checkpoint("checkpoints/best_model.pth")

# Single image prediction
result = engine.predict("path/to/wave_image.jpg")
print(f"Wave height: {result.height_meters:.1f}m")
print(f"Wave type: {result.wave_type}")
print(f"Direction: {result.direction}")
print(f"Confidence: {result.confidence_scores}")

# Batch prediction
results = engine.predict_batch([
    "image1.jpg", 
    "image2.png", 
    "image3.jpeg"
])

# Get model information
info = engine.get_model_info()
print(f"Model has {info['model_parameters']:,} parameters")
```

## Project Structure

```
swellsight/
├── config/          # Configuration management
├── models/          # Model architectures
├── data/            # Data pipeline components
├── training/        # Training utilities
├── inference/       # Inference engine
├── evaluation/      # Metrics and evaluation
└── utils/           # Utility functions
```

## Model Architecture

The SwellSight model uses a multi-task architecture:

1. **Shared Feature Extractor**: ConvNeXt-Base backbone (2048-dim features)
2. **Task-Specific Heads**:
   - Height regression: 2-layer MLP → scalar output
   - Wave type classification: 2-layer MLP → 4-class softmax
   - Direction classification: 2-layer MLP → 3-class softmax

## Data Pipeline

### Synthetic Data Generation
- Depth maps define precise wave geometry
- ControlNet converts depth maps to photorealistic images
- Perfect ground truth labels from generation parameters
- 10,000+ training samples with parameter diversity

### Real-World Validation
- Beach camera images for domain adaptation testing
- Manual labeling for validation set
- Separate test set (never used in training)

## API Reference

### InferenceEngine

The main interface for wave analysis predictions.

#### Methods

**`InferenceEngine.from_checkpoint(checkpoint_path, device="auto")`**
- Load a trained model from checkpoint
- `checkpoint_path`: Path to model checkpoint file
- `device`: Device to run inference on ("auto", "cpu", "cuda")
- Returns: Initialized InferenceEngine instance

**`predict(image_path)`**
- Predict wave parameters from a single image
- `image_path`: Path to image file (JPEG, PNG supported)
- Returns: `WavePrediction` object
- Raises: `InferenceError` for invalid inputs

**`predict_batch(image_paths)`**
- Predict wave parameters for multiple images
- `image_paths`: List of image file paths
- Returns: List of `WavePrediction` objects

**`predict_from_tensor(image_tensor)`**
- Predict from preprocessed tensor
- `image_tensor`: Torch tensor [1,3,H,W] or [3,H,W]
- Returns: `WavePrediction` object

**`get_model_info()`**
- Get model metadata and configuration
- Returns: Dictionary with model information

### WavePrediction

Structured prediction result containing wave parameters and confidence scores.

#### Attributes
- `height_meters`: Wave height in meters (float)
- `wave_type`: Predicted wave type (str)
- `direction`: Wave direction (str)
- `wave_type_probs`: Probability distribution for wave types (dict)
- `direction_probs`: Probability distribution for directions (dict)
- `confidence_scores`: Confidence scores for each prediction (dict)

#### Methods
- `to_dict()`: Convert to dictionary for JSON serialization
- `to_json()`: Convert to JSON string

### Response Format

```json
{
  "height_meters": 2.3,
  "wave_type": "A_FRAME",
  "direction": "LEFT",
  "wave_type_probabilities": {
    "A_FRAME": 0.85,
    "CLOSEOUT": 0.10,
    "BEACH_BREAK": 0.03,
    "POINT_BREAK": 0.02
  },
  "direction_probabilities": {
    "LEFT": 0.78,
    "RIGHT": 0.15,
    "BOTH": 0.07
  },
  "confidence_scores": {
    "height": 0.82,
    "wave_type": 0.85,
    "direction": 0.78
  }
}
```

### Error Handling

The API includes comprehensive error handling:

- **`InferenceError`**: Raised for prediction failures
- **File validation**: Checks for file existence and supported formats
- **Image validation**: Validates image dimensions and format
- **Tensor validation**: Ensures correct tensor shapes and types

### Supported Formats

- **Image formats**: JPEG (.jpg, .jpeg), PNG (.png)
- **Input sizes**: 32x32 to 4096x4096 pixels
- **Color modes**: RGB (3 channels)

## Training and Evaluation

### Command Line Interface

```bash
# Generate synthetic training data
python -m swellsight.scripts.generate_training_data \
    --num_samples 10000 \
    --output_dir data/synthetic \
    --config_path config/model_config.yaml

# Train the model
python -m swellsight.scripts.train_end_to_end \
    --data_dir data/synthetic \
    --checkpoint_dir checkpoints \
    --config_path config/model_config.yaml \
    --num_epochs 100

# Evaluate on test set
python -m swellsight.scripts.evaluate_model \
    --checkpoint_path checkpoints/best_model.pth \
    --test_data_dir data/real_test \
    --output_dir results
```

### Configuration

Model configuration is managed through YAML files:

```yaml
# config/model_config.yaml
model:
  backbone: "convnext_base"
  input_size: [768, 768]
  feature_dim: 2048
  hidden_dim: 512
  dropout_rate: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  weight_decay: 1e-5
  
data:
  min_height: 0.3
  max_height: 4.0
  wave_types: ["A_FRAME", "CLOSEOUT", "BEACH_BREAK", "POINT_BREAK"]
  directions: ["LEFT", "RIGHT", "BOTH"]
```

## Testing

The project uses comprehensive testing including:
- Unit tests for individual components
- Property-based tests using Hypothesis
- Integration tests for end-to-end pipeline

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swellsight

# Run property-based tests only
pytest tests/property/

# Run integration tests
pytest tests/integration/

# Run with verbose output
pytest -v
```

### Test Categories

**Property-Based Tests**: Validate universal properties across randomly generated inputs
- Model architecture consistency
- Data pipeline integrity
- API response structure
- Mathematical properties of loss functions

**Integration Tests**: Test complete workflows
- End-to-end training pipeline
- Real image processing
- API integration with various input formats
- Model checkpoint persistence

**Unit Tests**: Test individual components
- Model forward pass
- Loss function computation
- Data loading and preprocessing
- Configuration management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SwellSight in your research, please cite:

```bibtex
@software{swellsight2024,
  title={SwellSight: Multi-Task Deep Learning for Wave Analysis},
  author={SwellSight Team},
  year={2024},
  url={https://github.com/swellsight/wave-analysis-model}
}
```
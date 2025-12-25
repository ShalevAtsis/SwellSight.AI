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

# Initialize model and trainer
config = ModelConfig()
model = WaveAnalysisModel(config)
trainer = Trainer(model, config)

# Train the model
trainer.train()
```

### Inference
```python
from swellsight import InferenceEngine

# Load trained model
engine = InferenceEngine.from_checkpoint("path/to/checkpoint.pth")

# Analyze wave image
result = engine.predict("path/to/wave_image.jpg")
print(f"Wave height: {result.height_meters:.1f}m")
print(f"Wave type: {result.wave_type}")
print(f"Direction: {result.direction}")
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
```

## Performance Metrics

### Regression (Wave Height)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Classification (Wave Type & Direction)
- Accuracy
- F1-Score
- Confusion Matrix

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
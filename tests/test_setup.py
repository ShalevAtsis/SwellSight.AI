"""Test basic package setup and imports."""

import pytest
import torch
from swellsight import ModelConfig, WaveAnalysisModel, MultiTaskLoss


def test_package_import():
    """Test that the package can be imported."""
    import swellsight
    assert swellsight.__version__ == "0.1.0"


def test_model_config():
    """Test model configuration creation."""
    config = ModelConfig()
    assert config.backbone == 'convnext_base'
    assert config.input_size == (768, 768)
    assert config.feature_dim == 2048
    assert config.num_wave_types == 4
    assert config.num_directions == 3


def test_model_creation():
    """Test model creation with config."""
    config = ModelConfig()
    model = WaveAnalysisModel(config)
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass."""
    config = ModelConfig()
    model = WaveAnalysisModel(config)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, *config.input_size)
    
    # Forward pass
    outputs = model(x)
    
    # Check outputs
    assert 'height' in outputs
    assert 'wave_type' in outputs
    assert 'direction' in outputs
    
    assert outputs['height'].shape == (batch_size, 1)
    assert outputs['wave_type'].shape == (batch_size, config.num_wave_types)
    assert outputs['direction'].shape == (batch_size, config.num_directions)


def test_loss_function():
    """Test multi-task loss function."""
    loss_fn = MultiTaskLoss()
    
    # Create dummy predictions and targets
    batch_size = 2
    predictions = {
        'height': torch.randn(batch_size, 1),
        'wave_type': torch.randn(batch_size, 4),
        'direction': torch.randn(batch_size, 3)
    }
    targets = {
        'height': torch.randn(batch_size),
        'wave_type': torch.randint(0, 4, (batch_size,)),
        'direction': torch.randint(0, 3, (batch_size,))
    }
    
    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    assert isinstance(loss_dict, dict)
    assert 'total_loss' in loss_dict
    assert isinstance(loss_dict['total_loss'], torch.Tensor)
    assert loss_dict['total_loss'].dim() == 0  # Scalar loss
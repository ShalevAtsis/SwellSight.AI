"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from swellsight.config import ModelConfig, TrainingConfig, DataConfig


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return ModelConfig()


@pytest.fixture
def training_config():
    """Default training configuration for testing."""
    return TrainingConfig(
        batch_size=4,  # Small batch for testing
        num_epochs=2,  # Few epochs for testing
    )


@pytest.fixture
def data_config():
    """Default data configuration for testing."""
    return DataConfig(
        num_synthetic_samples=100,  # Small dataset for testing
        num_workers=0,  # No multiprocessing in tests
    )


@pytest.fixture
def sample_image_tensor(model_config):
    """Create a sample image tensor for testing."""
    batch_size = 2
    channels = 3
    height, width = model_config.input_size
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_labels(model_config):
    """Create sample labels for testing."""
    batch_size = 2
    return {
        'height': torch.tensor([1.5, 2.3], dtype=torch.float32),
        'wave_type': torch.tensor([0, 2], dtype=torch.long),
        'direction': torch.tensor([1, 0], dtype=torch.long)
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "synthetic").mkdir()
    (data_dir / "real").mkdir()
    (data_dir / "metadata").mkdir()
    (data_dir / "checkpoints").mkdir()
    
    return data_dir


@pytest.fixture
def sample_wave_parameters():
    """Sample wave generation parameters."""
    return {
        'height_meters': 1.8,
        'wave_type': 'A_FRAME',
        'direction': 'RIGHT',
        'depth_map_path': 'path/to/depth_map.png',
        'generation_seed': 42
    }
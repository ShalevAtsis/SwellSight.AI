"""
SwellSight Wave Analysis Model

A multi-task deep learning system that extracts objective physical wave parameters
from single beach camera images.
"""

__version__ = "0.1.0"
__author__ = "SwellSight Team"
__description__ = "Multi-task deep learning model for wave analysis"

from .config import ModelConfig
from .models import WaveAnalysisModel, MultiTaskLoss
from .data import SyntheticDataGenerator, DatasetManager, RealDataLoader
from .training import Trainer
from .inference import InferenceEngine
from .evaluation import MetricsCalculator

__all__ = [
    "ModelConfig",
    "WaveAnalysisModel", 
    "MultiTaskLoss",
    "SyntheticDataGenerator",
    "DatasetManager", 
    "RealDataLoader",
    "Trainer",
    "InferenceEngine",
    "MetricsCalculator"
]
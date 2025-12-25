"""Model architectures for SwellSight Wave Analysis."""

from .wave_analysis_model import WaveAnalysisModel
from .multi_task_loss import MultiTaskLoss

__all__ = ["WaveAnalysisModel", "MultiTaskLoss"]
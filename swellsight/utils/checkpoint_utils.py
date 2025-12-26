"""Checkpoint utilities for SwellSight Wave Analysis Model."""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    filepath: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with complete state and metadata.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of evaluation metrics
        config: Model and training configuration
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add custom metadata if provided
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    # Save human-readable metadata
    metadata_file = filepath.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp'],
            'config': config,
            'metadata': metadata or {}
        }, f, indent=2)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore state.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
    
    Returns:
        Dictionary containing checkpoint information
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint with weights_only=False to handle metadata
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'metadata': checkpoint.get('metadata', {})
    }


def validate_checkpoint_integrity(filepath: Path) -> bool:
    """
    Validate checkpoint file integrity.
    
    Args:
        filepath: Path to checkpoint file
    
    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        # Load with weights_only=False to handle metadata
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Check required fields
        required_fields = ['model_state_dict', 'epoch', 'loss']
        for field in required_fields:
            if field not in checkpoint:
                return False
        
        # Check if model state dict is valid
        if not isinstance(checkpoint['model_state_dict'], dict):
            return False
        
        return True
    
    except Exception:
        return False


def get_checkpoint_info(filepath: Path) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.
    
    Args:
        filepath: Path to checkpoint file
    
    Returns:
        Dictionary containing checkpoint information
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Try to load metadata file first (faster)
    metadata_file = filepath.with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    # Fall back to loading checkpoint with weights_only=False
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'config': checkpoint.get('config', {}),
        'metadata': checkpoint.get('metadata', {})
    }
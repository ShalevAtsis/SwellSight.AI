"""Device utilities for SwellSight Wave Analysis Model."""

import torch
from typing import Union, Any


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ('cpu', 'cuda', 'auto', or torch.device)
                If None or 'auto', automatically selects best available device
    
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(device, str):
        return torch.device(device)
    
    return device


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Move data to the specified device.
    
    Args:
        data: Data to move (tensor, dict, list, or other)
        device: Target device
    
    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Device information including CUDA availability and GPU details
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
        info["memory_info"] = [
            {
                "device": i,
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(i),
                "cached_memory": torch.cuda.memory_reserved(i),
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info
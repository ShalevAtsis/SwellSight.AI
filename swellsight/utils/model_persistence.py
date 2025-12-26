"""Model persistence utilities for SwellSight Wave Analysis Model."""

import torch
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .device_utils import get_device, move_to_device
from .checkpoint_utils import validate_checkpoint_integrity
from ..models.wave_analysis_model import WaveAnalysisModel
from ..config import ModelConfig


class ModelPersistence:
    """
    Comprehensive model persistence system with serialization, integrity validation,
    and device compatibility support.
    """
    
    def __init__(self):
        """Initialize the model persistence system."""
        self.supported_formats = ['.pth', '.pt']
    
    def save_model(
        self,
        model: WaveAnalysisModel,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        include_config: bool = True,
        compress: bool = False
    ) -> Dict[str, Any]:
        """
        Save a trained model with complete state and metadata.
        
        Args:
            model: The trained WaveAnalysisModel to save
            filepath: Path to save the model
            metadata: Additional metadata to include
            include_config: Whether to include model configuration
            compress: Whether to compress the saved file
        
        Returns:
            Dictionary containing save information and integrity hash
        """
        filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'pytorch_version': torch.__version__,
            'timestamp': datetime.now().isoformat(),
            'device': str(next(model.parameters()).device),
        }
        
        # Include model configuration if requested
        if include_config and hasattr(model, 'config'):
            model_data['config'] = model.config.__dict__
        
        # Add custom metadata
        if metadata:
            model_data['metadata'] = metadata
        
        # Save the model (without integrity hash first)
        if compress:
            # Use torch.save with compression
            torch.save(model_data, filepath, _use_new_zipfile_serialization=True)
        else:
            torch.save(model_data, filepath)
        
        # Calculate integrity hash of the saved file
        integrity_hash = self._calculate_file_hash(filepath)
        
        # Save human-readable metadata
        metadata_file = filepath.with_suffix('.json')
        metadata_info = {
            'model_class': model_data['model_class'],
            'pytorch_version': model_data['pytorch_version'],
            'timestamp': model_data['timestamp'],
            'device': model_data['device'],
            'integrity_hash': integrity_hash,
            'file_size_bytes': filepath.stat().st_size,
            'config': model_data.get('config', {}),
            'metadata': metadata or {}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        return {
            'filepath': str(filepath),
            'integrity_hash': integrity_hash,
            'file_size': filepath.stat().st_size,
            'timestamp': model_data['timestamp']
        }
    
    def load_model(
        self,
        filepath: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        validate_integrity: bool = True
    ) -> WaveAnalysisModel:
        """
        Load a saved model with integrity validation and device compatibility.
        
        Args:
            filepath: Path to the saved model
            device: Target device ('cpu', 'cuda', 'auto', or torch.device)
            strict: Whether to strictly enforce state dict loading
            validate_integrity: Whether to validate file integrity
        
        Returns:
            Loaded WaveAnalysisModel instance
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model data is invalid or corrupted
            RuntimeError: If model loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Validate file integrity if requested
        if validate_integrity and not self.validate_model_integrity(filepath):
            raise ValueError(f"Model file integrity validation failed: {filepath}")
        
        # Determine target device
        target_device = get_device(device)
        
        try:
            # Load model data
            model_data = torch.load(filepath, map_location=target_device, weights_only=False)
            
            # Validate required fields
            if 'model_state_dict' not in model_data:
                raise ValueError("Model file missing 'model_state_dict'")
            
            # Create model instance
            if 'config' in model_data:
                # Reconstruct config from saved data
                config = ModelConfig(**model_data['config'])
                model = WaveAnalysisModel(config)
            else:
                # Use default config (may not work for all models)
                config = ModelConfig()
                model = WaveAnalysisModel(config)
            
            # Load state dict
            model.load_state_dict(model_data['model_state_dict'], strict=strict)
            
            # Move model to target device
            model = model.to(target_device)
            
            # Set to evaluation mode by default
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {str(e)}")
    
    def validate_model_integrity(self, filepath: Union[str, Path]) -> bool:
        """
        Validate the integrity of a saved model file.
        
        Args:
            filepath: Path to the model file
        
        Returns:
            True if the model file is valid and uncorrupted
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return False
        
        try:
            # Load and validate model data structure
            model_data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Check for required fields for model persistence (not checkpoint)
            required_fields = ['model_state_dict', 'pytorch_version', 'timestamp']
            for field in required_fields:
                if field not in model_data:
                    return False
            
            # Validate state dict structure
            state_dict = model_data['model_state_dict']
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                return False
            
            # Check if integrity hash matches (if available in metadata file)
            metadata_file = filepath.with_suffix('.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata_info = json.load(f)
                    if 'integrity_hash' in metadata_info:
                        current_hash = self._calculate_file_hash(filepath)
                        if current_hash != metadata_info['integrity_hash']:
                            return False
                except Exception:
                    # If we can't read metadata, continue with other validation
                    pass
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a saved model without loading it completely.
        
        Args:
            filepath: Path to the model file
        
        Returns:
            Dictionary containing model information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Try to load metadata file first (faster)
        metadata_file = filepath.with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        # Fall back to loading model data
        try:
            model_data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            info = {
                'model_class': model_data.get('model_class', 'Unknown'),
                'pytorch_version': model_data.get('pytorch_version', 'Unknown'),
                'timestamp': model_data.get('timestamp', 'Unknown'),
                'device': model_data.get('device', 'Unknown'),
                'file_size_bytes': filepath.stat().st_size,
                'config': model_data.get('config', {}),
                'metadata': model_data.get('metadata', {})
            }
            
            if 'integrity_hash' in model_data:
                info['integrity_hash'] = model_data['integrity_hash']
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to read model info from {filepath}: {str(e)}")
    
    def test_device_compatibility(
        self,
        model: WaveAnalysisModel,
        test_input_shape: tuple = (1, 3, 768, 768)
    ) -> Dict[str, bool]:
        """
        Test model compatibility across different devices.
        
        Args:
            model: The model to test
            test_input_shape: Shape of test input tensor
        
        Returns:
            Dictionary indicating device compatibility
        """
        compatibility = {
            'cpu': False,
            'cuda': False
        }
        
        # Test CPU compatibility
        try:
            model_cpu = model.to('cpu')
            test_input = torch.randn(test_input_shape)
            with torch.no_grad():
                output = model_cpu(test_input)
            compatibility['cpu'] = True
        except Exception:
            compatibility['cpu'] = False
        
        # Test CUDA compatibility (if available)
        if torch.cuda.is_available():
            try:
                model_cuda = model.to('cuda')
                test_input = torch.randn(test_input_shape).cuda()
                with torch.no_grad():
                    output = model_cuda(test_input)
                compatibility['cuda'] = True
            except Exception:
                compatibility['cuda'] = False
        
        return compatibility
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


# Convenience functions for easy access
def save_model(
    model: WaveAnalysisModel,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to save a model.
    
    Args:
        model: The model to save
        filepath: Path to save the model
        metadata: Additional metadata
        **kwargs: Additional arguments for ModelPersistence.save_model
    
    Returns:
        Save information dictionary
    """
    persistence = ModelPersistence()
    return persistence.save_model(model, filepath, metadata, **kwargs)


def load_model(
    filepath: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> WaveAnalysisModel:
    """
    Convenience function to load a model.
    
    Args:
        filepath: Path to the model file
        device: Target device
        **kwargs: Additional arguments for ModelPersistence.load_model
    
    Returns:
        Loaded model instance
    """
    persistence = ModelPersistence()
    return persistence.load_model(filepath, device, **kwargs)


def validate_model(filepath: Union[str, Path]) -> bool:
    """
    Convenience function to validate a model file.
    
    Args:
        filepath: Path to the model file
    
    Returns:
        True if model is valid
    """
    persistence = ModelPersistence()
    return persistence.validate_model_integrity(filepath)


def get_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to get model information.
    
    Args:
        filepath: Path to the model file
    
    Returns:
        Model information dictionary
    """
    persistence = ModelPersistence()
    return persistence.get_model_info(filepath)
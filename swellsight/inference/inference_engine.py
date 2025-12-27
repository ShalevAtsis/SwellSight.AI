"""Inference engine for wave analysis model."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import json
import logging
from torchvision import transforms

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class WavePrediction:
    """Wave prediction result."""
    height_meters: float
    wave_type: str
    direction: str
    wave_type_probs: Dict[str, float]
    direction_probs: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary for JSON serialization."""
        return {
            'height_meters': float(self.height_meters),
            'wave_type': self.wave_type,
            'direction': self.direction,
            'wave_type_probabilities': self.wave_type_probs,
            'direction_probabilities': self.direction_probs,
            'confidence_scores': self.confidence_scores
        }
    
    def to_json(self) -> str:
        """Convert prediction to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class InferenceError(Exception):
    """Custom exception for inference errors."""
    pass


class InferenceEngine:
    """
    Inference engine for wave analysis model.
    
    Supports JPEG and PNG image formats, returns structured predictions with confidence scores,
    and includes comprehensive error handling for invalid inputs.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    def __init__(self, model: torch.nn.Module, config: 'ModelConfig', device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained wave analysis model
            config: Model configuration
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device and set to evaluation mode
        self.model = model.to(self.device)
        self.model.eval()
        
        # Create preprocessing transforms (validation transforms without augmentation)
        self.preprocess_transforms = transforms.Compose([
            transforms.Resize(config.input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet normalization
                std=(0.229, 0.224, 0.225)
            )
        ])
        
        logger.info(f"InferenceEngine initialized on device: {self.device}")
        logger.info(f"Model input size: {config.input_size}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "auto") -> 'InferenceEngine':
        """
        Load inference engine from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        
        Returns:
            Initialized inference engine
            
        Raises:
            InferenceError: If checkpoint loading fails
        """
        try:
            from ..models import WaveAnalysisModel
            from ..config import ModelConfig
            from ..utils.model_persistence import load_model
            
            # Load model and config from checkpoint
            model, config, metadata = load_model(checkpoint_path, device)
            
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
            logger.info(f"Model metadata: {metadata}")
            
            return cls(model, config, device)
            
        except Exception as e:
            raise InferenceError(f"Failed to load model from checkpoint {checkpoint_path}: {str(e)}")
    
    def _validate_image_path(self, image_path: Union[str, Path]) -> Path:
        """
        Validate image path and format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Validated Path object
            
        Raises:
            InferenceError: If path is invalid or format unsupported
        """
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            raise InferenceError(f"Image file not found: {image_path}")
        
        # Check if it's a file
        if not path.is_file():
            raise InferenceError(f"Path is not a file: {image_path}")
        
        # Check file format
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise InferenceError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        return path
    
    def _load_and_preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor [1, 3, H, W]
            
        Raises:
            InferenceError: If image loading or preprocessing fails
        """
        try:
            # Validate path
            path = self._validate_image_path(image_path)
            
            # Load image
            image = Image.open(path).convert('RGB')
            
            # Check image dimensions
            width, height = image.size
            if width < 32 or height < 32:
                raise InferenceError(f"Image too small: {width}x{height}. Minimum size: 32x32")
            
            if width > 4096 or height > 4096:
                raise InferenceError(f"Image too large: {width}x{height}. Maximum size: 4096x4096")
            
            # Apply preprocessing transforms
            tensor = self.preprocess_transforms(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
            
            return tensor
            
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Failed to load and preprocess image {image_path}: {str(e)}")
    
    def _compute_confidence_scores(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute confidence scores for predictions.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Dictionary of confidence scores
        """
        confidence_scores = {}
        
        # Height confidence: inverse of normalized prediction uncertainty
        # For regression, we use a simple heuristic based on the prediction value
        height_pred = predictions['height'].item()
        height_normalized = (height_pred - self.config.min_height) / (self.config.max_height - self.config.min_height)
        height_confidence = 1.0 - abs(height_normalized - 0.5) * 0.4  # Higher confidence for mid-range values
        confidence_scores['height'] = max(0.5, min(1.0, height_confidence))
        
        # Wave type confidence: maximum probability
        wave_type_probs = F.softmax(predictions['wave_type'], dim=1)
        wave_type_confidence = torch.max(wave_type_probs).item()
        confidence_scores['wave_type'] = wave_type_confidence
        
        # Direction confidence: maximum probability
        direction_probs = F.softmax(predictions['direction'], dim=1)
        direction_confidence = torch.max(direction_probs).item()
        confidence_scores['direction'] = direction_confidence
        
        return confidence_scores
    
    def _postprocess_predictions(self, predictions: Dict[str, torch.Tensor]) -> WavePrediction:
        """
        Postprocess raw model predictions into structured output.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Structured wave prediction
        """
        # Extract height prediction
        height_meters = predictions['height'].squeeze().item()
        
        # Clamp height to valid range
        height_meters = max(self.config.min_height, min(self.config.max_height, height_meters))
        
        # Process wave type classification
        wave_type_logits = predictions['wave_type']
        wave_type_probs = F.softmax(wave_type_logits, dim=1).squeeze()
        wave_type_idx = torch.argmax(wave_type_probs).item()
        wave_type = self.config.wave_type_classes[wave_type_idx]
        
        # Create wave type probability dictionary
        wave_type_probs_dict = {
            class_name: prob.item() 
            for class_name, prob in zip(self.config.wave_type_classes, wave_type_probs)
        }
        
        # Process direction classification
        direction_logits = predictions['direction']
        direction_probs = F.softmax(direction_logits, dim=1).squeeze()
        direction_idx = torch.argmax(direction_probs).item()
        direction = self.config.direction_classes[direction_idx]
        
        # Create direction probability dictionary
        direction_probs_dict = {
            class_name: prob.item() 
            for class_name, prob in zip(self.config.direction_classes, direction_probs)
        }
        
        # Compute confidence scores
        confidence_scores = self._compute_confidence_scores(predictions)
        
        return WavePrediction(
            height_meters=height_meters,
            wave_type=wave_type,
            direction=direction,
            wave_type_probs=wave_type_probs_dict,
            direction_probs=direction_probs_dict,
            confidence_scores=confidence_scores
        )
    
    def predict(self, image_path: Union[str, Path]) -> WavePrediction:
        """
        Predict wave parameters from image.
        
        Args:
            image_path: Path to input image (JPEG or PNG)
        
        Returns:
            Wave prediction result with confidence scores
            
        Raises:
            InferenceError: If prediction fails
        """
        try:
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Postprocess predictions
            result = self._postprocess_predictions(predictions)
            
            logger.debug(f"Prediction completed for {image_path}: {result.wave_type}, {result.height_meters}m")
            
            return result
            
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Prediction failed for {image_path}: {str(e)}")
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[WavePrediction]:
        """
        Predict wave parameters for batch of images.
        
        Args:
            image_paths: List of paths to input images
        
        Returns:
            List of wave prediction results
            
        Raises:
            InferenceError: If batch prediction fails
        """
        if not image_paths:
            return []
        
        try:
            # Load and preprocess all images
            image_tensors = []
            valid_paths = []
            
            for image_path in image_paths:
                try:
                    tensor = self._load_and_preprocess_image(image_path)
                    image_tensors.append(tensor)
                    valid_paths.append(image_path)
                except InferenceError as e:
                    logger.warning(f"Skipping invalid image {image_path}: {e}")
                    continue
            
            if not image_tensors:
                raise InferenceError("No valid images found in batch")
            
            # Concatenate into batch
            batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)
            
            # Run batch inference
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor)
            
            # Postprocess each prediction
            results = []
            for i in range(batch_tensor.size(0)):
                # Extract predictions for single image
                single_predictions = {
                    'height': batch_predictions['height'][i:i+1],
                    'wave_type': batch_predictions['wave_type'][i:i+1],
                    'direction': batch_predictions['direction'][i:i+1]
                }
                
                result = self._postprocess_predictions(single_predictions)
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(results)} images")
            
            return results
            
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Batch prediction failed: {str(e)}")
    
    def predict_from_tensor(self, image_tensor: torch.Tensor) -> WavePrediction:
        """
        Predict wave parameters from preprocessed tensor.
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, H, W] or [3, H, W]
        
        Returns:
            Wave prediction result
            
        Raises:
            InferenceError: If tensor prediction fails
        """
        try:
            # Validate tensor
            if not isinstance(image_tensor, torch.Tensor):
                raise InferenceError(f"Expected torch.Tensor, got {type(image_tensor)}")
            
            # Add batch dimension if needed
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.dim() != 4:
                raise InferenceError(f"Expected 3D or 4D tensor, got {image_tensor.dim()}D")
            
            # Check tensor shape
            if image_tensor.size(1) != 3:
                raise InferenceError(f"Expected 3 channels, got {image_tensor.size(1)}")
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Postprocess predictions
            result = self._postprocess_predictions(predictions)
            
            return result
            
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Tensor prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'device': str(self.device),
            'input_size': self.config.input_size,
            'wave_type_classes': list(self.config.wave_type_classes),
            'direction_classes': list(self.config.direction_classes),
            'height_range': [self.config.min_height, self.config.max_height],
            'supported_formats': list(self.SUPPORTED_FORMATS),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
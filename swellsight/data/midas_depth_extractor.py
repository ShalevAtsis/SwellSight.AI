"""MiDaS depth extraction for real beach camera images."""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import logging
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DepthExtractionResult:
    """Result of depth extraction from an image."""
    depth_map: np.ndarray
    original_image_path: str
    depth_quality_score: float
    processing_metadata: Dict[str, Any]


class MiDaSDepthExtractor:
    """
    MiDaS depth extraction from HuggingFace for real beach camera images.
    
    Uses MiDaS model to extract depth maps from real beach images in data/real/images
    with corresponding labels from data/real/labels/labels.json.
    """
    
    def __init__(self, model_name: str = "Intel/dpt-large", device: Optional[str] = None):
        """
        Initialize the MiDaS depth extractor.
        
        Args:
            model_name: HuggingFace model name for MiDaS (Intel/dpt-large or Intel/dpt-hybrid-midas)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing MiDaS depth extractor with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load MiDaS model and processor from HuggingFace
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Successfully loaded MiDaS model and processor")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise
    
    def extract_depth(self, image_path: str) -> DepthExtractionResult:
        """
        Extract depth map from a single beach camera image.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            DepthExtractionResult containing depth map and metadata
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            logger.debug(f"Processing image: {image_path} (size: {original_size})")
            
            # Prepare image for MiDaS
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract depth map
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Post-process depth map
            depth_map = self._postprocess_depth_map(
                predicted_depth, 
                original_size, 
                image_path
            )
            
            # Validate depth quality
            quality_score = self.validate_depth_quality(depth_map)
            
            # Create processing metadata
            processing_metadata = {
                'model_name': self.model_name,
                'device': self.device,
                'original_image_size': original_size,
                'depth_map_shape': depth_map.shape,
                'depth_range': (float(depth_map.min()), float(depth_map.max())),
                'processing_timestamp': torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
            }
            
            result = DepthExtractionResult(
                depth_map=depth_map,
                original_image_path=str(image_path),
                depth_quality_score=quality_score,
                processing_metadata=processing_metadata
            )
            
            logger.debug(f"Successfully extracted depth map with quality score: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract depth from {image_path}: {e}")
            raise
    
    def batch_extract(self, image_paths: List[str]) -> List[DepthExtractionResult]:
        """
        Extract depth maps from multiple images in batch.
        
        Args:
            image_paths: List of paths to input images
        
        Returns:
            List of DepthExtractionResult objects
        """
        logger.info(f"Starting batch depth extraction for {len(image_paths)} images")
        
        results = []
        failed_count = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.extract_depth(image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Batch extraction completed: {len(results)} successful, {failed_count} failed")
        
        return results
    
    def validate_depth_quality(self, depth_map: np.ndarray) -> float:
        """
        Validate depth map quality for beach scene analysis.
        
        Args:
            depth_map: Input depth map to validate
        
        Returns:
            Quality score between 0.0 and 1.0 (higher is better)
        """
        try:
            # Check for valid depth values
            if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
                logger.warning("Depth map contains NaN or infinite values")
                return 0.0
            
            # Check depth range (should have reasonable variation for beach scenes)
            depth_range = depth_map.max() - depth_map.min()
            if depth_range < 1e-6:
                logger.warning("Depth map has insufficient variation")
                return 0.1
            
            # Calculate quality metrics
            quality_metrics = []
            
            # 1. Depth variation (good beach scenes should have depth variation)
            depth_std = np.std(depth_map)
            depth_mean = np.mean(depth_map)
            variation_score = min(depth_std / (depth_mean + 1e-6), 1.0)
            quality_metrics.append(variation_score)
            
            # 2. Gradient consistency (smooth transitions expected in beach scenes)
            grad_x = np.gradient(depth_map, axis=1)
            grad_y = np.gradient(depth_map, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_score = 1.0 - min(np.mean(gradient_magnitude) / depth_range, 1.0)
            quality_metrics.append(gradient_score)
            
            # 3. Spatial coherence (neighboring pixels should have similar depths)
            kernel = np.ones((3, 3)) / 9
            smoothed = cv2.filter2D(depth_map, -1, kernel)
            coherence_error = np.mean(np.abs(depth_map - smoothed))
            coherence_score = 1.0 - min(coherence_error / depth_range, 1.0)
            quality_metrics.append(coherence_score)
            
            # 4. Dynamic range utilization
            depth_normalized = (depth_map - depth_map.min()) / depth_range
            hist, _ = np.histogram(depth_normalized, bins=50)
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
            range_score = min(entropy / np.log(50), 1.0)  # Normalize by max entropy
            quality_metrics.append(range_score)
            
            # Combine metrics with weights
            weights = [0.3, 0.2, 0.3, 0.2]  # Emphasize variation and coherence
            overall_quality = sum(w * score for w, score in zip(weights, quality_metrics))
            
            logger.debug(f"Depth quality metrics: variation={variation_score:.3f}, "
                        f"gradient={gradient_score:.3f}, coherence={coherence_score:.3f}, "
                        f"range={range_score:.3f}, overall={overall_quality:.3f}")
            
            return float(overall_quality)
            
        except Exception as e:
            logger.error(f"Error validating depth quality: {e}")
            return 0.0
    
    def _postprocess_depth_map(self, predicted_depth: torch.Tensor, 
                              original_size: Tuple[int, int], 
                              image_path: Path) -> np.ndarray:
        """
        Post-process MiDaS depth prediction to create final depth map.
        
        Args:
            predicted_depth: Raw depth prediction from MiDaS
            original_size: Original image size (width, height)
            image_path: Path to original image for debugging
        
        Returns:
            Post-processed depth map as numpy array
        """
        try:
            # Convert to numpy and remove batch dimension
            depth_map = predicted_depth.squeeze().cpu().numpy()
            
            # Resize to original image dimensions
            depth_map = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LINEAR)
            
            # MiDaS outputs inverse depth, so invert to get actual depth
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            depth_map = 1.0 / (depth_map + epsilon)
            
            # Normalize depth values to reasonable range for beach scenes
            # Typical beach camera depth range: 1-100 meters
            depth_min = np.percentile(depth_map, 5)  # Use percentiles to handle outliers
            depth_max = np.percentile(depth_map, 95)
            
            # Clip extreme values
            depth_map = np.clip(depth_map, depth_min, depth_max)
            
            # Normalize to 1-100 meter range (typical for beach scenes)
            depth_map = 1.0 + (depth_map - depth_min) / (depth_max - depth_min) * 99.0
            
            # Apply Gaussian smoothing to reduce noise
            depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
            
            logger.debug(f"Post-processed depth map: shape={depth_map.shape}, "
                        f"range=({depth_map.min():.2f}, {depth_map.max():.2f})")
            
            return depth_map.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error post-processing depth map for {image_path}: {e}")
            raise
    
    def save_depth_map(self, depth_map: np.ndarray, output_path: str, 
                      format: str = 'npy') -> None:
        """
        Save depth map to file.
        
        Args:
            depth_map: Depth map to save
            output_path: Output file path
            format: Save format ('npy', 'png', 'tiff')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'npy':
                np.save(output_path, depth_map)
            elif format == 'png':
                # Normalize to 0-65535 for 16-bit PNG
                depth_normalized = ((depth_map - depth_map.min()) / 
                                  (depth_map.max() - depth_map.min()) * 65535).astype(np.uint16)
                cv2.imwrite(str(output_path), depth_normalized)
            elif format == 'tiff':
                # Save as 32-bit float TIFF
                cv2.imwrite(str(output_path), depth_map.astype(np.float32))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Saved depth map to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save depth map to {output_path}: {e}")
            raise
    
    def load_depth_map(self, depth_path: str) -> np.ndarray:
        """
        Load depth map from file.
        
        Args:
            depth_path: Path to depth map file
        
        Returns:
            Loaded depth map as numpy array
        """
        depth_path = Path(depth_path)
        
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        
        try:
            if depth_path.suffix == '.npy':
                depth_map = np.load(depth_path)
            elif depth_path.suffix in ['.png', '.tiff', '.tif']:
                depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_map is None:
                    raise ValueError(f"Failed to load image: {depth_path}")
                # Convert back to float if needed
                if depth_map.dtype == np.uint16:
                    depth_map = depth_map.astype(np.float32) / 65535.0 * 100.0  # Assume 0-100m range
            else:
                raise ValueError(f"Unsupported depth map format: {depth_path.suffix}")
            
            logger.debug(f"Loaded depth map from: {depth_path}, shape: {depth_map.shape}")
            
            return depth_map.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to load depth map from {depth_path}: {e}")
            raise
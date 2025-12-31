"""Depth-based wave analysis from MiDaS depth maps."""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WaveDetectionResult:
    """Result of wave detection from depth map."""
    wave_crests: List[Tuple[int, int]]  # (x, y) coordinates of wave crests
    wave_heights: List[float]  # Heights in meters for each detected wave
    dominant_direction: str  # 'left', 'right', 'both'
    breaking_regions: np.ndarray  # Binary mask of breaking wave regions
    wave_type: str  # Estimated wave type
    confidence_score: float  # Overall confidence in detection


class DepthWaveAnalyzer:
    """
    Analyzes wave parameters from MiDaS depth maps.
    
    Implements depth-based wave height estimation, breaking detection,
    and wave direction analysis for beach scenes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the depth wave analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Analysis parameters
        self.min_wave_height = self.config.get('min_wave_height', 0.1)  # meters
        self.max_wave_height = self.config.get('max_wave_height', 5.0)  # meters
        self.depth_scale_factor = self.config.get('depth_scale_factor', 0.1)
        self.breaking_gradient_threshold = self.config.get('breaking_gradient_threshold', 0.15)
        self.wave_detection_sensitivity = self.config.get('wave_detection_sensitivity', 0.8)
        
        logger.info("Initialized DepthWaveAnalyzer")
    
    def analyze_waves_from_depth(self, depth_map: np.ndarray, 
                                image_filename: Optional[str] = None) -> WaveDetectionResult:
        """
        Analyze wave parameters from a depth map.
        
        Args:
            depth_map: Input depth map from MiDaS
            image_filename: Optional filename for logging
            
        Returns:
            WaveDetectionResult with detected wave parameters
        """
        try:
            # Preprocess depth map
            processed_depth = self._preprocess_depth_map(depth_map)
            
            # Detect wave crests
            wave_crests = self._detect_wave_crests(processed_depth)
            
            # Estimate wave heights
            wave_heights = self._estimate_wave_heights(processed_depth, wave_crests)
            
            # Analyze wave direction
            dominant_direction = self._analyze_wave_direction(processed_depth, wave_crests)
            
            # Detect breaking regions
            breaking_regions = self._detect_breaking_regions(processed_depth)
            
            # Classify wave type
            wave_type = self._classify_wave_type(processed_depth, wave_crests, breaking_regions)
            
            # Compute confidence score
            confidence_score = self._compute_confidence_score(
                processed_depth, wave_crests, wave_heights, breaking_regions
            )
            
            result = WaveDetectionResult(
                wave_crests=wave_crests,
                wave_heights=wave_heights,
                dominant_direction=dominant_direction,
                breaking_regions=breaking_regions,
                wave_type=wave_type,
                confidence_score=confidence_score
            )
            
            logger.info(f"Analyzed waves for {image_filename or 'unknown'}: "
                       f"{len(wave_crests)} crests, avg height {np.mean(wave_heights):.2f}m")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing waves from depth map: {e}")
            # Return empty result
            return WaveDetectionResult(
                wave_crests=[],
                wave_heights=[],
                dominant_direction='both',
                breaking_regions=np.zeros_like(depth_map, dtype=bool),
                wave_type='unknown',
                confidence_score=0.0
            )
    
    def _preprocess_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Preprocess depth map for wave analysis.
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Processed depth map
        """
        # Normalize depth values
        depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        
        # Apply Gaussian smoothing to reduce noise
        depth_smooth = ndimage.gaussian_filter(depth_normalized, sigma=2.0)
        
        # Focus on water region (bottom 70% of image)
        water_mask = np.zeros_like(depth_smooth)
        water_start = int(0.3 * depth_smooth.shape[0])
        water_mask[water_start:, :] = 1.0
        
        # Apply water mask
        depth_processed = depth_smooth * water_mask
        
        return depth_processed
    
    def _detect_wave_crests(self, depth_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect wave crests from depth map using local maxima detection.
        
        Args:
            depth_map: Processed depth map
            
        Returns:
            List of (x, y) coordinates of wave crests
        """
        # Compute horizontal gradients to find wave patterns
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        
        # Find local maxima in depth (wave crests)
        local_maxima = ndimage.maximum_filter(depth_map, size=10) == depth_map
        
        # Filter by minimum depth threshold (avoid detecting noise)
        depth_threshold = np.percentile(depth_map[depth_map > 0], 70)
        valid_maxima = local_maxima & (depth_map > depth_threshold)
        
        # Get coordinates of wave crests
        crest_coords = np.where(valid_maxima)
        wave_crests = list(zip(crest_coords[1], crest_coords[0]))  # (x, y) format
        
        # Cluster nearby crests to avoid duplicates
        if len(wave_crests) > 1:
            wave_crests = self._cluster_wave_crests(wave_crests)
        
        return wave_crests
    
    def _cluster_wave_crests(self, wave_crests: List[Tuple[int, int]], 
                           min_distance: int = 20) -> List[Tuple[int, int]]:
        """
        Cluster nearby wave crests to avoid duplicates.
        
        Args:
            wave_crests: List of crest coordinates
            min_distance: Minimum distance between crests
            
        Returns:
            Filtered list of wave crests
        """
        if len(wave_crests) <= 1:
            return wave_crests
        
        # Convert to numpy array for clustering
        crests_array = np.array(wave_crests)
        
        # Use DBSCAN clustering
        clustering = DBSCAN(eps=min_distance, min_samples=1)
        cluster_labels = clustering.fit_predict(crests_array)
        
        # Take centroid of each cluster
        clustered_crests = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_points = crests_array[cluster_labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0).astype(int)
            clustered_crests.append((centroid[0], centroid[1]))
        
        return clustered_crests
    
    def _estimate_wave_heights(self, depth_map: np.ndarray, 
                              wave_crests: List[Tuple[int, int]]) -> List[float]:
        """
        Estimate wave heights from depth variations around crests.
        
        Args:
            depth_map: Processed depth map
            wave_crests: List of wave crest coordinates
            
        Returns:
            List of estimated wave heights in meters
        """
        wave_heights = []
        
        for crest_x, crest_y in wave_crests:
            try:
                # Define analysis window around crest
                window_size = 30
                y_start = max(0, crest_y - window_size)
                y_end = min(depth_map.shape[0], crest_y + window_size)
                x_start = max(0, crest_x - window_size)
                x_end = min(depth_map.shape[1], crest_x + window_size)
                
                # Extract local depth region
                local_depth = depth_map[y_start:y_end, x_start:x_end]
                
                if local_depth.size == 0:
                    wave_heights.append(0.0)
                    continue
                
                # Estimate height from depth variation
                crest_depth = depth_map[crest_y, crest_x]
                trough_depth = np.min(local_depth[local_depth > 0])
                
                # Convert depth difference to wave height
                depth_diff = crest_depth - trough_depth
                estimated_height = depth_diff * self.depth_scale_factor
                
                # Clamp to reasonable range
                estimated_height = np.clip(estimated_height, self.min_wave_height, self.max_wave_height)
                wave_heights.append(float(estimated_height))
                
            except Exception as e:
                logger.warning(f"Error estimating height for crest at ({crest_x}, {crest_y}): {e}")
                wave_heights.append(0.0)
        
        return wave_heights
    
    def _analyze_wave_direction(self, depth_map: np.ndarray, 
                               wave_crests: List[Tuple[int, int]]) -> str:
        """
        Analyze dominant wave direction from depth flow patterns.
        
        Args:
            depth_map: Processed depth map
            wave_crests: List of wave crest coordinates
            
        Returns:
            Dominant direction: 'left', 'right', or 'both'
        """
        if len(wave_crests) == 0:
            return 'both'
        
        # Compute horizontal gradients
        grad_x = np.gradient(depth_map, axis=1)
        
        # Analyze flow direction around wave crests
        left_flow_strength = 0.0
        right_flow_strength = 0.0
        
        for crest_x, crest_y in wave_crests:
            # Sample gradient in region around crest
            window_size = 20
            y_start = max(0, crest_y - window_size)
            y_end = min(depth_map.shape[0], crest_y + window_size)
            x_start = max(0, crest_x - window_size)
            x_end = min(depth_map.shape[1], crest_x + window_size)
            
            local_grad_x = grad_x[y_start:y_end, x_start:x_end]
            
            # Accumulate flow strengths
            left_flow_strength += np.sum(local_grad_x[local_grad_x < 0])
            right_flow_strength += np.sum(local_grad_x[local_grad_x > 0])
        
        # Determine dominant direction
        left_magnitude = abs(left_flow_strength)
        right_magnitude = abs(right_flow_strength)
        
        if left_magnitude > right_magnitude * 1.5:
            return 'left'
        elif right_magnitude > left_magnitude * 1.5:
            return 'right'
        else:
            return 'both'
    
    def _detect_breaking_regions(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Detect breaking wave regions from depth gradients.
        
        Args:
            depth_map: Processed depth map
            
        Returns:
            Binary mask of breaking regions
        """
        # Compute gradient magnitude
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold for breaking detection
        breaking_threshold = np.percentile(gradient_magnitude, 85)
        breaking_regions = gradient_magnitude > breaking_threshold
        
        # Apply morphological operations to clean up regions
        kernel = np.ones((5, 5), np.uint8)
        breaking_regions = cv2.morphologyEx(
            breaking_regions.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        ).astype(bool)
        
        return breaking_regions
    
    def _classify_wave_type(self, depth_map: np.ndarray, 
                           wave_crests: List[Tuple[int, int]], 
                           breaking_regions: np.ndarray) -> str:
        """
        Classify wave type based on depth patterns and breaking behavior.
        
        Args:
            depth_map: Processed depth map
            wave_crests: List of wave crest coordinates
            breaking_regions: Binary mask of breaking regions
            
        Returns:
            Wave type classification
        """
        if len(wave_crests) == 0:
            return 'flat'
        
        # Analyze breaking intensity
        breaking_intensity = np.sum(breaking_regions) / breaking_regions.size
        
        # Analyze wave organization
        if len(wave_crests) >= 3:
            # Check if crests are organized in lines (beach break pattern)
            crest_coords = np.array(wave_crests)
            y_coords = crest_coords[:, 1]
            y_std = np.std(y_coords)
            
            if y_std < 20:  # Crests are aligned horizontally
                if breaking_intensity > 0.15:
                    return 'beach_break'
                else:
                    return 'point_break'
            else:
                return 'a_frame'
        else:
            # Few crests - analyze individual characteristics
            if breaking_intensity > 0.2:
                return 'closeout'
            else:
                return 'beach_break'
    
    def _compute_confidence_score(self, depth_map: np.ndarray, 
                                 wave_crests: List[Tuple[int, int]], 
                                 wave_heights: List[float], 
                                 breaking_regions: np.ndarray) -> float:
        """
        Compute confidence score for wave detection results.
        
        Args:
            depth_map: Processed depth map
            wave_crests: List of wave crest coordinates
            wave_heights: List of estimated wave heights
            breaking_regions: Binary mask of breaking regions
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []
        
        # Factor 1: Depth map quality
        depth_range = np.max(depth_map) - np.min(depth_map)
        if depth_range > 0.1:
            depth_quality = min(depth_range / 0.5, 1.0)
        else:
            depth_quality = 0.0
        confidence_factors.append(depth_quality)
        
        # Factor 2: Number of detected crests
        crest_factor = min(len(wave_crests) / 5.0, 1.0)
        confidence_factors.append(crest_factor)
        
        # Factor 3: Wave height consistency
        if wave_heights:
            height_std = np.std(wave_heights)
            height_mean = np.mean(wave_heights)
            if height_mean > 0:
                height_consistency = 1.0 - min(height_std / height_mean, 1.0)
            else:
                height_consistency = 0.0
        else:
            height_consistency = 0.0
        confidence_factors.append(height_consistency)
        
        # Factor 4: Breaking region coherence
        breaking_intensity = np.sum(breaking_regions) / breaking_regions.size
        breaking_factor = 1.0 if 0.05 <= breaking_intensity <= 0.3 else 0.5
        confidence_factors.append(breaking_factor)
        
        # Compute overall confidence
        overall_confidence = np.mean(confidence_factors)
        return float(overall_confidence)
    
    def extract_wave_parameters(self, detection_result: WaveDetectionResult) -> Dict[str, Any]:
        """
        Extract structured wave parameters from detection result.
        
        Args:
            detection_result: Result from wave detection
            
        Returns:
            Dictionary with structured wave parameters
        """
        parameters = {
            'wave_count': len(detection_result.wave_crests),
            'average_height_meters': float(np.mean(detection_result.wave_heights)) if detection_result.wave_heights else 0.0,
            'max_height_meters': float(np.max(detection_result.wave_heights)) if detection_result.wave_heights else 0.0,
            'min_height_meters': float(np.min(detection_result.wave_heights)) if detection_result.wave_heights else 0.0,
            'height_std_meters': float(np.std(detection_result.wave_heights)) if detection_result.wave_heights else 0.0,
            'dominant_direction': detection_result.dominant_direction,
            'wave_type': detection_result.wave_type,
            'breaking_intensity': float(np.sum(detection_result.breaking_regions) / detection_result.breaking_regions.size),
            'confidence_score': detection_result.confidence_score,
            'crest_coordinates': [(int(x), int(y)) for x, y in detection_result.wave_crests]  # Convert to int
        }
        
        return parameters
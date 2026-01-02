"""Depth-based wave analysis for MiDaS extracted depth maps."""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class WaveAnalysisResult:
    """Result of depth-based wave analysis."""
    estimated_height: float
    height_confidence: float
    breaking_patterns: Dict[str, float]
    wave_direction: str
    direction_confidence: float
    depth_features: np.ndarray
    analysis_metadata: Dict[str, Any]


class DepthAnalyzer:
    """
    Analyzes MiDaS-extracted depth maps to estimate wave parameters.
    
    Provides alternative wave parameter estimation method for validation
    and comparison with the main deep learning model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize depth analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Analysis parameters
        self.min_wave_height = self.config.get('min_wave_height', 0.1)  # meters
        self.max_wave_height = self.config.get('max_wave_height', 10.0)  # meters
        self.crest_detection_threshold = self.config.get('crest_detection_threshold', 0.1)
        self.smoothing_kernel_size = self.config.get('smoothing_kernel_size', 5)
        self.gradient_threshold = self.config.get('gradient_threshold', 0.05)
        
        logger.info("Depth analyzer initialized")
    
    def analyze_wave_parameters(self, depth_map: np.ndarray) -> WaveAnalysisResult:
        """
        Analyze wave parameters from depth map.
        
        Args:
            depth_map: Input depth map from MiDaS
        
        Returns:
            WaveAnalysisResult with estimated parameters
        """
        try:
            # Validate and preprocess depth map
            processed_depth = self._preprocess_depth_map(depth_map)
            
            # Estimate wave height
            height, height_confidence = self.estimate_wave_height(processed_depth)
            
            # Detect breaking patterns
            breaking_patterns = self.detect_breaking_patterns(processed_depth)
            
            # Analyze wave direction
            direction, direction_confidence = self.analyze_wave_direction(processed_depth)
            
            # Generate depth features
            depth_features = self.generate_depth_features(processed_depth)
            
            # Create analysis metadata
            analysis_metadata = {
                'depth_map_shape': depth_map.shape,
                'depth_range': (float(depth_map.min()), float(depth_map.max())),
                'processing_parameters': {
                    'smoothing_kernel_size': self.smoothing_kernel_size,
                    'gradient_threshold': self.gradient_threshold,
                    'crest_detection_threshold': self.crest_detection_threshold
                },
                'quality_indicators': self._assess_depth_quality(processed_depth)
            }
            
            result = WaveAnalysisResult(
                estimated_height=height,
                height_confidence=height_confidence,
                breaking_patterns=breaking_patterns,
                wave_direction=direction,
                direction_confidence=direction_confidence,
                depth_features=depth_features,
                analysis_metadata=analysis_metadata
            )
            
            logger.debug(f"Wave analysis complete: height={height:.2f}m, direction={direction}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze wave parameters: {e}")
            raise
    
    def estimate_wave_height(self, depth_map: np.ndarray) -> Tuple[float, float]:
        """
        Estimate wave height from depth map using crest detection.
        
        Args:
            depth_map: Preprocessed depth map
        
        Returns:
            Tuple of (height, confidence)
        """
        try:
            # Extract wave crests
            crests = self.extract_wave_crests(depth_map)
            
            if len(crests) == 0:
                logger.warning("No wave crests detected")
                return 0.0, 0.0
            
            # Calculate height statistics from crests
            crest_heights = []
            for crest in crests:
                # Get depth values along crest
                crest_depths = depth_map[crest[:, 0], crest[:, 1]]
                
                # Find local minima (troughs) around crest
                trough_depth = self._find_nearby_trough(depth_map, crest)
                
                # Calculate wave height as difference
                if trough_depth is not None:
                    wave_height = trough_depth - np.min(crest_depths)
                    if self.min_wave_height <= wave_height <= self.max_wave_height:
                        crest_heights.append(wave_height)
            
            if not crest_heights:
                logger.warning("No valid wave heights calculated")
                return 0.0, 0.0
            
            # Calculate statistics
            estimated_height = np.median(crest_heights)  # Use median for robustness
            height_std = np.std(crest_heights)
            
            # Calculate confidence based on consistency
            confidence = self._calculate_height_confidence(crest_heights, estimated_height)
            
            logger.debug(f"Estimated wave height: {estimated_height:.2f}m ± {height_std:.2f}m")
            
            return float(estimated_height), float(confidence)
            
        except Exception as e:
            logger.error(f"Failed to estimate wave height: {e}")
            return 0.0, 0.0
    
    def detect_breaking_patterns(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        Detect wave breaking patterns from depth map gradients and discontinuities.
        
        Args:
            depth_map: Preprocessed depth map
        
        Returns:
            Dictionary with breaking pattern scores
        """
        try:
            # Calculate gradients
            grad_x = np.gradient(depth_map, axis=1)
            grad_y = np.gradient(depth_map, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate second derivatives (curvature)
            grad_xx = np.gradient(grad_x, axis=1)
            grad_yy = np.gradient(grad_y, axis=0)
            curvature = np.abs(grad_xx) + np.abs(grad_yy)
            
            # Detect discontinuities
            discontinuities = self._detect_depth_discontinuities(depth_map)
            
            # Analyze breaking patterns
            breaking_patterns = {}
            
            # Spilling breakers: gradual depth changes with moderate gradients
            spilling_score = self._analyze_spilling_pattern(gradient_magnitude, curvature)
            breaking_patterns['spilling'] = spilling_score
            
            # Plunging breakers: sharp depth changes with high gradients
            plunging_score = self._analyze_plunging_pattern(gradient_magnitude, discontinuities)
            breaking_patterns['plunging'] = plunging_score
            
            # Collapsing breakers: irregular patterns with high curvature
            collapsing_score = self._analyze_collapsing_pattern(curvature, discontinuities)
            breaking_patterns['collapsing'] = collapsing_score
            
            # Surging breakers: minimal breaking with low gradients
            surging_score = self._analyze_surging_pattern(gradient_magnitude, depth_map)
            breaking_patterns['surging'] = surging_score
            
            # Normalize scores
            total_score = sum(breaking_patterns.values())
            if total_score > 0:
                breaking_patterns = {k: v / total_score for k, v in breaking_patterns.items()}
            
            logger.debug(f"Breaking patterns: {breaking_patterns}")
            
            return breaking_patterns
            
        except Exception as e:
            logger.error(f"Failed to detect breaking patterns: {e}")
            return {'spilling': 0.25, 'plunging': 0.25, 'collapsing': 0.25, 'surging': 0.25}
    
    def analyze_wave_direction(self, depth_map: np.ndarray) -> Tuple[str, float]:
        """
        Determine wave direction from depth map flow analysis.
        
        Args:
            depth_map: Preprocessed depth map
        
        Returns:
            Tuple of (direction, confidence)
        """
        try:
            # Calculate optical flow on depth map
            flow_vectors = self._calculate_depth_flow(depth_map)
            
            if flow_vectors is None or len(flow_vectors) == 0:
                return 'BOTH', 0.0
            
            # Analyze flow directions
            angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
            angles_deg = np.degrees(angles)
            
            # Normalize angles to 0-360 range
            angles_deg = (angles_deg + 360) % 360
            
            # Analyze directional consistency
            direction, confidence = self._classify_wave_direction(angles_deg)
            
            logger.debug(f"Wave direction: {direction} (confidence: {confidence:.3f})")
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Failed to analyze wave direction: {e}")
            return 'BOTH', 0.0
    
    def generate_depth_features(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Generate feature vector from depth map for analysis.
        
        Args:
            depth_map: Preprocessed depth map
        
        Returns:
            Feature vector
        """
        try:
            features = []
            
            # Basic statistics
            features.extend([
                np.mean(depth_map),
                np.std(depth_map),
                np.min(depth_map),
                np.max(depth_map),
                np.median(depth_map)
            ])
            
            # Gradient statistics
            grad_x = np.gradient(depth_map, axis=1)
            grad_y = np.gradient(depth_map, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
            # Texture features
            features.extend(self._calculate_texture_features(depth_map))
            
            # Spatial frequency features
            features.extend(self._calculate_frequency_features(depth_map))
            
            # Morphological features
            features.extend(self._calculate_morphological_features(depth_map))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate depth features: {e}")
            return np.zeros(20, dtype=np.float32)  # Return default feature vector
    
    def validate_depth_quality(self, depth_map: np.ndarray) -> float:
        """
        Validate depth map quality for beach scene analysis.
        
        Args:
            depth_map: Input depth map to validate
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            quality_indicators = self._assess_depth_quality(depth_map)
            
            # Weighted combination of quality indicators
            weights = {
                'variation_score': 0.3,
                'gradient_consistency': 0.2,
                'spatial_coherence': 0.3,
                'range_utilization': 0.2
            }
            
            overall_quality = sum(
                weights.get(key, 0) * value 
                for key, value in quality_indicators.items()
            )
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Failed to validate depth quality: {e}")
            return 0.0
    
    def extract_wave_crests(self, depth_map: np.ndarray) -> List[np.ndarray]:
        """
        Extract wave crest lines from depth map.
        
        Args:
            depth_map: Input depth map
        
        Returns:
            List of crest line coordinates
        """
        try:
            # Smooth depth map to reduce noise
            smoothed = cv2.GaussianBlur(depth_map, (self.smoothing_kernel_size, self.smoothing_kernel_size), 1.0)
            
            # Find local maxima (shallow areas indicating crests)
            local_maxima = self._find_local_maxima(smoothed)
            
            # Connect nearby maxima into crest lines
            crests = self._connect_crest_points(local_maxima, smoothed)
            
            # Filter crests by length and consistency
            filtered_crests = self._filter_crests(crests, smoothed)
            
            logger.debug(f"Extracted {len(filtered_crests)} wave crests")
            
            return filtered_crests
            
        except Exception as e:
            logger.error(f"Failed to extract wave crests: {e}")
            return []
    
    def _preprocess_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Preprocess depth map for analysis."""
        # Handle invalid values
        processed = np.copy(depth_map)
        processed[np.isnan(processed)] = np.nanmean(processed)
        processed[np.isinf(processed)] = np.nanmean(processed)
        
        # Apply median filter to reduce noise
        processed = ndimage.median_filter(processed, size=3)
        
        return processed
    
    def _find_nearby_trough(self, depth_map: np.ndarray, crest: np.ndarray) -> Optional[float]:
        """Find nearby trough depth for wave height calculation."""
        try:
            # Get crest center
            crest_center = np.mean(crest, axis=0).astype(int)
            
            # Search in expanding circles around crest
            max_radius = min(depth_map.shape) // 4
            
            for radius in range(5, max_radius, 5):
                # Create circular mask
                y, x = np.ogrid[:depth_map.shape[0], :depth_map.shape[1]]
                mask = (x - crest_center[1])**2 + (y - crest_center[0])**2 <= radius**2
                
                if np.any(mask):
                    # Find minimum depth in this region
                    masked_depths = depth_map[mask]
                    trough_depth = np.max(masked_depths)  # Max depth (deepest water)
                    
                    # Check if this is significantly deeper than crest
                    crest_depth = np.min(depth_map[crest[:, 0], crest[:, 1]])
                    if trough_depth - crest_depth > self.min_wave_height:
                        return trough_depth
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to find nearby trough: {e}")
            return None
    
    def _calculate_height_confidence(self, heights: List[float], estimated_height: float) -> float:
        """Calculate confidence score for height estimation."""
        if not heights or len(heights) < 2:
            return 0.0
        
        # Coefficient of variation (lower is better)
        cv = np.std(heights) / (np.mean(heights) + 1e-6)
        cv_score = np.exp(-cv * 2)  # Exponential decay
        
        # Number of measurements (more is better)
        count_score = min(len(heights) / 10.0, 1.0)
        
        # Range check (heights should be reasonable)
        range_score = 1.0 if self.min_wave_height <= estimated_height <= self.max_wave_height else 0.5
        
        # Combined confidence
        confidence = (cv_score + count_score + range_score) / 3
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _detect_depth_discontinuities(self, depth_map: np.ndarray) -> np.ndarray:
        """Detect sharp depth discontinuities."""
        # Use Laplacian to detect discontinuities
        laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
        
        # Threshold to find significant discontinuities
        threshold = np.std(laplacian) * 2
        discontinuities = np.abs(laplacian) > threshold
        
        return discontinuities.astype(np.float32)
    
    def _analyze_spilling_pattern(self, gradient_magnitude: np.ndarray, curvature: np.ndarray) -> float:
        """Analyze spilling breaker pattern."""
        # Spilling: moderate gradients, low curvature
        moderate_gradient = (gradient_magnitude > np.percentile(gradient_magnitude, 30)) & \
                           (gradient_magnitude < np.percentile(gradient_magnitude, 70))
        low_curvature = curvature < np.percentile(curvature, 50)
        
        spilling_regions = moderate_gradient & low_curvature
        return float(np.mean(spilling_regions))
    
    def _analyze_plunging_pattern(self, gradient_magnitude: np.ndarray, discontinuities: np.ndarray) -> float:
        """Analyze plunging breaker pattern."""
        # Plunging: high gradients with discontinuities
        high_gradient = gradient_magnitude > np.percentile(gradient_magnitude, 80)
        high_discontinuity = discontinuities > 0.5
        
        plunging_regions = high_gradient & high_discontinuity
        return float(np.mean(plunging_regions))
    
    def _analyze_collapsing_pattern(self, curvature: np.ndarray, discontinuities: np.ndarray) -> float:
        """Analyze collapsing breaker pattern."""
        # Collapsing: high curvature with some discontinuities
        high_curvature = curvature > np.percentile(curvature, 75)
        some_discontinuity = discontinuities > 0.3
        
        collapsing_regions = high_curvature & some_discontinuity
        return float(np.mean(collapsing_regions))
    
    def _analyze_surging_pattern(self, gradient_magnitude: np.ndarray, depth_map: np.ndarray) -> float:
        """Analyze surging breaker pattern."""
        # Surging: low gradients, minimal breaking
        low_gradient = gradient_magnitude < np.percentile(gradient_magnitude, 40)
        
        # Check for smooth depth transitions
        smoothness = 1.0 - np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)
        
        surging_score = np.mean(low_gradient) * smoothness
        return float(np.clip(surging_score, 0.0, 1.0))
    
    def _calculate_depth_flow(self, depth_map: np.ndarray) -> Optional[np.ndarray]:
        """Calculate optical flow on depth map."""
        try:
            # Convert to uint8 for optical flow
            depth_uint8 = ((depth_map - depth_map.min()) / 
                          (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # Create shifted versions for flow calculation
            shifted_x = np.roll(depth_uint8, 1, axis=1)
            shifted_y = np.roll(depth_uint8, 1, axis=0)
            
            # Calculate flow using Lucas-Kanade method
            flow = cv2.calcOpticalFlowPyrLK(
                depth_uint8, shifted_x, 
                None, None,
                winSize=(15, 15),
                maxLevel=2
            )
            
            if flow[0] is not None:
                # Extract valid flow vectors
                good_points = flow[1].ravel() == 1
                if np.any(good_points):
                    flow_vectors = flow[0][good_points] - np.arange(len(flow[0]))[good_points, np.newaxis]
                    return flow_vectors
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to calculate depth flow: {e}")
            return None
    
    def _classify_wave_direction(self, angles_deg: np.ndarray) -> Tuple[str, float]:
        """Classify wave direction from flow angles."""
        if len(angles_deg) == 0:
            return 'BOTH', 0.0
        
        # Analyze angle distribution
        # Convert to unit vectors for circular statistics
        unit_vectors = np.column_stack([np.cos(np.radians(angles_deg)), 
                                       np.sin(np.radians(angles_deg))])
        
        # Calculate mean direction
        mean_vector = np.mean(unit_vectors, axis=0)
        mean_angle = np.degrees(np.arctan2(mean_vector[1], mean_vector[0]))
        mean_angle = (mean_angle + 360) % 360
        
        # Calculate directional consistency
        consistency = np.linalg.norm(mean_vector)
        
        # Classify direction based on mean angle
        if consistency < 0.3:
            return 'BOTH', consistency
        elif 315 <= mean_angle or mean_angle < 45:
            return 'RIGHT', consistency
        elif 135 <= mean_angle < 225:
            return 'LEFT', consistency
        else:
            return 'BOTH', consistency * 0.5  # Reduce confidence for ambiguous directions
    
    def _calculate_texture_features(self, depth_map: np.ndarray) -> List[float]:
        """Calculate texture features from depth map."""
        # Gray-Level Co-occurrence Matrix (GLCM) features
        # Simplified implementation
        
        # Normalize depth map to 0-255 range
        normalized = ((depth_map - depth_map.min()) / 
                     (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        # Calculate local binary patterns
        lbp = self._calculate_lbp(normalized)
        
        # Texture statistics
        features = [
            np.std(lbp),  # LBP variance
            np.mean(np.abs(np.diff(normalized, axis=0))),  # Vertical roughness
            np.mean(np.abs(np.diff(normalized, axis=1))),  # Horizontal roughness
        ]
        
        return features
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        # Simplified LBP calculation
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def _calculate_frequency_features(self, depth_map: np.ndarray) -> List[float]:
        """Calculate frequency domain features."""
        # FFT-based features
        fft = np.fft.fft2(depth_map)
        fft_magnitude = np.abs(fft)
        
        # Frequency statistics
        features = [
            np.mean(fft_magnitude),
            np.std(fft_magnitude),
            np.sum(fft_magnitude > np.percentile(fft_magnitude, 90))  # High frequency content
        ]
        
        return features
    
    def _calculate_morphological_features(self, depth_map: np.ndarray) -> List[float]:
        """Calculate morphological features."""
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        
        # Convert to uint8 for morphological operations
        depth_uint8 = ((depth_map - depth_map.min()) / 
                      (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        opening = cv2.morphologyEx(depth_uint8, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(depth_uint8, cv2.MORPH_CLOSE, kernel)
        
        features = [
            np.mean(np.abs(depth_uint8 - opening)),  # Opening difference
            np.mean(np.abs(depth_uint8 - closing)),  # Closing difference
        ]
        
        return features
    
    def _assess_depth_quality(self, depth_map: np.ndarray) -> Dict[str, float]:
        """Assess depth map quality indicators."""
        # Variation score
        depth_std = np.std(depth_map)
        depth_mean = np.mean(depth_map)
        variation_score = min(depth_std / (depth_mean + 1e-6), 1.0)
        
        # Gradient consistency
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_consistency = 1.0 - min(np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6), 1.0)
        
        # Spatial coherence
        kernel = np.ones((3, 3)) / 9
        smoothed = cv2.filter2D(depth_map, -1, kernel)
        coherence_error = np.mean(np.abs(depth_map - smoothed))
        depth_range = depth_map.max() - depth_map.min()
        spatial_coherence = 1.0 - min(coherence_error / (depth_range + 1e-6), 1.0)
        
        # Range utilization
        depth_normalized = (depth_map - depth_map.min()) / (depth_range + 1e-6)
        hist, _ = np.histogram(depth_normalized, bins=50)
        hist_normalized = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        range_utilization = min(entropy / np.log(50), 1.0)
        
        return {
            'variation_score': float(variation_score),
            'gradient_consistency': float(gradient_consistency),
            'spatial_coherence': float(spatial_coherence),
            'range_utilization': float(range_utilization)
        }
    
    def _find_local_maxima(self, depth_map: np.ndarray) -> np.ndarray:
        """Find local maxima in depth map."""
        # Use morphological operations to find local maxima
        kernel = np.ones((5, 5))
        dilated = ndimage.maximum_filter(depth_map, footprint=kernel)
        
        # Local maxima are points where original equals dilated
        local_maxima = (depth_map == dilated) & (depth_map > np.percentile(depth_map, 70))
        
        # Get coordinates
        coords = np.column_stack(np.where(local_maxima))
        
        return coords
    
    def _connect_crest_points(self, maxima_coords: np.ndarray, depth_map: np.ndarray) -> List[np.ndarray]:
        """Connect nearby maxima points into crest lines."""
        if len(maxima_coords) < 2:
            return []
        
        # Use DBSCAN clustering to group nearby points
        clustering = DBSCAN(eps=10, min_samples=3)
        labels = clustering.fit_predict(maxima_coords)
        
        crests = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            
            cluster_points = maxima_coords[labels == label]
            if len(cluster_points) >= 3:  # Minimum points for a crest
                # Sort points to form a line
                sorted_points = self._sort_points_into_line(cluster_points)
                crests.append(sorted_points)
        
        return crests
    
    def _sort_points_into_line(self, points: np.ndarray) -> np.ndarray:
        """Sort points to form a continuous line."""
        if len(points) <= 2:
            return points
        
        # Start with leftmost point
        sorted_points = [points[np.argmin(points[:, 1])]]
        remaining_points = list(points)
        remaining_points.remove(sorted_points[0])
        
        # Greedily add nearest points
        while remaining_points:
            current_point = sorted_points[-1]
            distances = [np.linalg.norm(current_point - p) for p in remaining_points]
            nearest_idx = np.argmin(distances)
            
            sorted_points.append(remaining_points[nearest_idx])
            remaining_points.pop(nearest_idx)
        
        return np.array(sorted_points)
    
    def _filter_crests(self, crests: List[np.ndarray], depth_map: np.ndarray) -> List[np.ndarray]:
        """Filter crests by length and consistency."""
        filtered_crests = []
        
        for crest in crests:
            # Check minimum length
            if len(crest) < 5:
                continue
            
            # Check depth consistency along crest
            crest_depths = depth_map[crest[:, 0], crest[:, 1]]
            depth_variation = np.std(crest_depths) / (np.mean(crest_depths) + 1e-6)
            
            if depth_variation < 0.5:  # Reasonable consistency
                filtered_crests.append(crest)
        
        return filtered_crests
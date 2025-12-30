"""Integration between MiDaS depth extraction and real data labels."""

import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime
import csv
from dataclasses import dataclass, asdict

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor, DepthExtractionResult
from swellsight.data.real_data_loader import RealDataLoader, ManualLabelingUtility

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DepthValidationResult:
    """Result of depth-based wave parameter validation."""
    image_filename: str
    manual_labels: Dict[str, Any]
    depth_analysis: Dict[str, Any]
    validation_metrics: Dict[str, float]
    quality_assessment: Dict[str, Any]


class MiDaSRealDataIntegrator:
    """
    Integrates MiDaS depth extraction with real data labels and validation.
    
    Creates correspondence between real images, depth maps, and manual labels.
    Implements validation pipeline for depth-based wave parameter estimation.
    """
    
    def __init__(self, real_data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MiDaS real data integrator.
        
        Args:
            real_data_path: Path to real data directory
            config: Optional configuration dictionary
        """
        self.real_data_path = Path(real_data_path)
        self.config = config or {}
        
        # Initialize components
        self.midas_extractor = MiDaSDepthExtractor(
            model_name=self.config.get('midas_model', 'Intel/dpt-large'),
            device=self.config.get('device', None)
        )
        
        self.real_data_loader = RealDataLoader(str(self.real_data_path), self.config)
        self.labeling_utility = ManualLabelingUtility(str(self.real_data_path))
        
        # Paths
        self.depth_maps_path = self.real_data_path / 'depth_maps'
        self.validation_results_path = self.real_data_path / 'validation'
        
        # Create directories
        self.depth_maps_path.mkdir(parents=True, exist_ok=True)
        self.validation_results_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MiDaSRealDataIntegrator for: {self.real_data_path}")
    
    def extract_depth_maps_for_real_images(self, force_recompute: bool = False) -> List[DepthExtractionResult]:
        """
        Extract depth maps for all real images with manual labels.
        
        Args:
            force_recompute: Whether to recompute existing depth maps
            
        Returns:
            List of depth extraction results
        """
        # Load manual labels
        labels = self.real_data_loader.load_manual_labels()
        
        if not labels:
            logger.warning("No manual labels found. Cannot extract depth maps.")
            return []
        
        # Get image paths
        image_paths = []
        for label in labels:
            image_path = self.real_data_path / 'images' / label['image_filename']
            if image_path.exists():
                image_paths.append(str(image_path))
            else:
                logger.warning(f"Image not found: {image_path}")
        
        if not image_paths:
            logger.error("No valid image paths found for depth extraction")
            return []
        
        # Check for existing depth maps
        if not force_recompute:
            existing_depth_maps = list(self.depth_maps_path.glob('*.npy'))
            if len(existing_depth_maps) >= len(image_paths):
                logger.info(f"Found {len(existing_depth_maps)} existing depth maps. Use force_recompute=True to regenerate.")
        
        # Extract depth maps
        logger.info(f"Extracting depth maps for {len(image_paths)} real images...")
        depth_results = self.midas_extractor.batch_extract(image_paths)
        
        # Save depth maps
        for result in depth_results:
            image_filename = Path(result.original_image_path).stem
            depth_save_path = self.depth_maps_path / f"{image_filename}_depth.npy"
            
            self.midas_extractor.save_depth_map(
                result.depth_map, 
                str(depth_save_path), 
                format='npy'
            )
            
            # Also save metadata
            metadata_path = self.depth_maps_path / f"{image_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result.processing_metadata, f, indent=2)
        
        logger.info(f"Successfully extracted and saved {len(depth_results)} depth maps")
        return depth_results
    
    def create_real_depth_correspondence(self) -> List[Dict[str, Any]]:
        """
        Create correspondence between real images, depth maps, and manual labels.
        
        Returns:
            List of correspondence dictionaries
        """
        # Load manual labels
        labels = self.real_data_loader.load_manual_labels()
        
        if not labels:
            logger.warning("No manual labels found")
            return []
        
        correspondences = []
        
        for label in labels:
            image_filename = label['image_filename']
            image_stem = Path(image_filename).stem
            
            # Check for image file
            image_path = self.real_data_path / 'images' / image_filename
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Check for depth map
            depth_map_path = self.depth_maps_path / f"{image_stem}_depth.npy"
            depth_metadata_path = self.depth_maps_path / f"{image_stem}_metadata.json"
            
            if not depth_map_path.exists():
                logger.warning(f"Depth map not found for {image_filename}")
                continue
            
            # Load depth metadata if available
            depth_metadata = {}
            if depth_metadata_path.exists():
                with open(depth_metadata_path, 'r') as f:
                    depth_metadata = json.load(f)
            
            # Create correspondence
            correspondence = {
                'image_filename': image_filename,
                'image_path': str(image_path),
                'depth_map_path': str(depth_map_path),
                'depth_metadata_path': str(depth_metadata_path) if depth_metadata_path.exists() else None,
                'manual_labels': label,
                'depth_metadata': depth_metadata,
                'correspondence_timestamp': datetime.now().isoformat()
            }
            
            correspondences.append(correspondence)
        
        # Save correspondences
        correspondence_file = self.validation_results_path / 'real_depth_correspondence.json'
        with open(correspondence_file, 'w') as f:
            json.dump(correspondences, f, indent=2)
        
        logger.info(f"Created {len(correspondences)} real-depth correspondences")
        return correspondences
    
    def validate_depth_based_estimation(self, correspondences: Optional[List[Dict[str, Any]]] = None) -> List[DepthValidationResult]:
        """
        Implement validation pipeline for depth-based wave parameter estimation.
        
        Args:
            correspondences: Optional list of correspondences, loads from file if None
            
        Returns:
            List of depth validation results
        """
        if correspondences is None:
            correspondence_file = self.validation_results_path / 'real_depth_correspondence.json'
            if correspondence_file.exists():
                with open(correspondence_file, 'r') as f:
                    correspondences = json.load(f)
            else:
                logger.error("No correspondences found. Run create_real_depth_correspondence() first.")
                return []
        
        validation_results = []
        
        for correspondence in correspondences:
            try:
                # Load depth map
                depth_map = self.midas_extractor.load_depth_map(correspondence['depth_map_path'])
                
                # Perform depth-based analysis
                depth_analysis = self._analyze_depth_map(depth_map, correspondence['image_filename'])
                
                # Compare with manual labels
                manual_labels = correspondence['manual_labels']
                validation_metrics = self._compute_validation_metrics(depth_analysis, manual_labels)
                
                # Assess quality
                quality_assessment = self._assess_depth_quality(
                    depth_map, 
                    depth_analysis, 
                    correspondence['depth_metadata']
                )
                
                # Create validation result
                result = DepthValidationResult(
                    image_filename=correspondence['image_filename'],
                    manual_labels=manual_labels,
                    depth_analysis=depth_analysis,
                    validation_metrics=validation_metrics,
                    quality_assessment=quality_assessment
                )
                
                validation_results.append(result)
                
            except Exception as e:
                logger.error(f"Error validating {correspondence['image_filename']}: {e}")
                continue
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        logger.info(f"Completed depth validation for {len(validation_results)} images")
        return validation_results
    
    def _analyze_depth_map(self, depth_map: np.ndarray, image_filename: str) -> Dict[str, Any]:
        """
        Analyze depth map to estimate wave parameters.
        
        Args:
            depth_map: Input depth map
            image_filename: Name of the image file
            
        Returns:
            Dictionary with depth-based analysis results
        """
        try:
            # Basic depth statistics
            depth_stats = {
                'min_depth': float(np.min(depth_map)),
                'max_depth': float(np.max(depth_map)),
                'mean_depth': float(np.mean(depth_map)),
                'std_depth': float(np.std(depth_map)),
                'depth_range': float(np.max(depth_map) - np.min(depth_map))
            }
            
            # Estimate wave height from depth variation
            # Simple approach: use depth range in water region (bottom 70% of image)
            water_region = depth_map[int(0.3 * depth_map.shape[0]):, :]
            water_depth_range = np.max(water_region) - np.min(water_region)
            estimated_height = min(water_depth_range * 0.1, 5.0)  # Scale factor and cap
            
            # Detect breaking patterns from depth gradients
            grad_x = np.gradient(depth_map, axis=1)
            grad_y = np.gradient(depth_map, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradient areas might indicate breaking waves
            high_gradient_threshold = np.percentile(gradient_magnitude, 90)
            breaking_regions = gradient_magnitude > high_gradient_threshold
            breaking_intensity = np.sum(breaking_regions) / breaking_regions.size
            
            # Estimate wave direction from depth flow patterns
            # Simplified approach: analyze horizontal gradients
            horizontal_flow = np.mean(grad_x, axis=0)
            left_flow = np.sum(horizontal_flow < -np.std(horizontal_flow))
            right_flow = np.sum(horizontal_flow > np.std(horizontal_flow))
            
            if left_flow > right_flow * 1.5:
                estimated_direction = 'LEFT'
            elif right_flow > left_flow * 1.5:
                estimated_direction = 'RIGHT'
            else:
                estimated_direction = 'BOTH'
            
            # Estimate wave type from breaking patterns and depth structure
            if breaking_intensity > 0.15:
                if water_depth_range > 2.0:
                    estimated_wave_type = 'CLOSEOUT'
                else:
                    estimated_wave_type = 'BEACH_BREAK'
            else:
                if estimated_height > 1.5:
                    estimated_wave_type = 'A_FRAME'
                else:
                    estimated_wave_type = 'POINT_BREAK'
            
            analysis = {
                'depth_statistics': depth_stats,
                'estimated_height_meters': float(estimated_height),
                'estimated_wave_type': estimated_wave_type,
                'estimated_direction': estimated_direction,
                'breaking_analysis': {
                    'breaking_intensity': float(breaking_intensity),
                    'high_gradient_threshold': float(high_gradient_threshold),
                    'breaking_region_percentage': float(breaking_intensity * 100)
                },
                'flow_analysis': {
                    'left_flow_strength': float(left_flow),
                    'right_flow_strength': float(right_flow),
                    'dominant_flow': estimated_direction
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing depth map for {image_filename}: {e}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _compute_validation_metrics(self, depth_analysis: Dict[str, Any], manual_labels: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute validation metrics comparing depth analysis with manual labels.
        
        Args:
            depth_analysis: Results from depth-based analysis
            manual_labels: Manual ground truth labels
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {}
        
        try:
            # Height validation
            if 'estimated_height_meters' in depth_analysis and 'height_meters' in manual_labels:
                estimated_height = depth_analysis['estimated_height_meters']
                manual_height = manual_labels['height_meters']
                
                height_error = abs(estimated_height - manual_height)
                height_relative_error = height_error / max(manual_height, 0.1)  # Avoid division by zero
                
                metrics['height_absolute_error'] = float(height_error)
                metrics['height_relative_error'] = float(height_relative_error)
                metrics['height_accuracy'] = float(1.0 - min(height_relative_error, 1.0))
            
            # Wave type validation
            if 'estimated_wave_type' in depth_analysis and 'wave_type' in manual_labels:
                estimated_type = depth_analysis['estimated_wave_type']
                manual_type = manual_labels['wave_type']
                
                metrics['wave_type_match'] = float(1.0 if estimated_type == manual_type else 0.0)
            
            # Direction validation
            if 'estimated_direction' in depth_analysis and 'direction' in manual_labels:
                estimated_direction = depth_analysis['estimated_direction']
                manual_direction = manual_labels['direction']
                
                metrics['direction_match'] = float(1.0 if estimated_direction == manual_direction else 0.0)
            
            # Overall accuracy (weighted combination)
            accuracy_components = []
            if 'height_accuracy' in metrics:
                accuracy_components.append(metrics['height_accuracy'] * 0.5)  # 50% weight
            if 'wave_type_match' in metrics:
                accuracy_components.append(metrics['wave_type_match'] * 0.3)  # 30% weight
            if 'direction_match' in metrics:
                accuracy_components.append(metrics['direction_match'] * 0.2)  # 20% weight
            
            if accuracy_components:
                metrics['overall_accuracy'] = float(sum(accuracy_components) / len(accuracy_components))
            
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _assess_depth_quality(self, depth_map: np.ndarray, depth_analysis: Dict[str, Any], 
                             depth_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess depth map quality for beach scene analysis.
        
        Args:
            depth_map: Input depth map
            depth_analysis: Results from depth analysis
            depth_metadata: Metadata from depth extraction
            
        Returns:
            Dictionary with quality assessment
        """
        try:
            # Use MiDaS quality validation
            midas_quality_score = self.midas_extractor.validate_depth_quality(depth_map)
            
            # Additional quality metrics specific to beach scenes
            quality_metrics = {
                'midas_quality_score': float(midas_quality_score),
                'depth_range_adequacy': self._assess_depth_range(depth_map),
                'spatial_coherence': self._assess_spatial_coherence(depth_map),
                'beach_scene_suitability': self._assess_beach_scene_suitability(depth_map, depth_analysis)
            }
            
            # Overall quality assessment
            quality_scores = [
                quality_metrics['midas_quality_score'],
                quality_metrics['depth_range_adequacy'],
                quality_metrics['spatial_coherence'],
                quality_metrics['beach_scene_suitability']
            ]
            
            overall_quality = sum(quality_scores) / len(quality_scores)
            quality_metrics['overall_quality'] = float(overall_quality)
            
            # Quality classification
            if overall_quality >= 0.8:
                quality_class = 'excellent'
            elif overall_quality >= 0.6:
                quality_class = 'good'
            elif overall_quality >= 0.4:
                quality_class = 'fair'
            else:
                quality_class = 'poor'
            
            quality_metrics['quality_class'] = quality_class
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing depth quality: {e}")
            return {'error': str(e)}
    
    def _assess_depth_range(self, depth_map: np.ndarray) -> float:
        """Assess if depth range is adequate for beach scene analysis."""
        depth_range = np.max(depth_map) - np.min(depth_map)
        # Good beach scenes should have depth range of 5-50 meters
        if 5.0 <= depth_range <= 50.0:
            return 1.0
        elif 2.0 <= depth_range <= 100.0:
            return 0.7
        else:
            return 0.3
    
    def _assess_spatial_coherence(self, depth_map: np.ndarray) -> float:
        """Assess spatial coherence of depth map."""
        # Compute local variance to assess smoothness
        from scipy import ndimage
        local_variance = ndimage.generic_filter(depth_map, np.var, size=5)
        mean_local_variance = np.mean(local_variance)
        
        # Lower variance indicates better spatial coherence
        coherence_score = 1.0 / (1.0 + mean_local_variance / np.mean(depth_map))
        return float(min(coherence_score, 1.0))
    
    def _assess_beach_scene_suitability(self, depth_map: np.ndarray, depth_analysis: Dict[str, Any]) -> float:
        """Assess if depth map is suitable for beach scene analysis."""
        try:
            # Check if depth increases from top to bottom (typical for beach scenes)
            top_region = depth_map[:depth_map.shape[0]//3, :]
            bottom_region = depth_map[2*depth_map.shape[0]//3:, :]
            
            top_mean = np.mean(top_region)
            bottom_mean = np.mean(bottom_region)
            
            depth_gradient_score = 1.0 if bottom_mean > top_mean else 0.5
            
            # Check for reasonable breaking intensity
            breaking_intensity = depth_analysis.get('breaking_analysis', {}).get('breaking_intensity', 0)
            breaking_score = 1.0 if 0.05 <= breaking_intensity <= 0.3 else 0.7
            
            # Combine scores
            suitability_score = (depth_gradient_score + breaking_score) / 2.0
            return float(suitability_score)
            
        except Exception:
            return 0.5  # Default moderate score if assessment fails
    
    def _save_validation_results(self, validation_results: List[DepthValidationResult]) -> None:
        """Save validation results to file."""
        results_file = self.validation_results_path / 'depth_validation_results.json'
        
        # Convert dataclass objects to dictionaries
        results_data = [asdict(result) for result in validation_results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved validation results to: {results_file}")
    
    def generate_statistical_analysis(self, validation_results: Optional[List[DepthValidationResult]] = None) -> Dict[str, Any]:
        """
        Generate statistical analysis of depth quality across different image conditions.
        
        Args:
            validation_results: Optional validation results, loads from file if None
            
        Returns:
            Dictionary with statistical analysis
        """
        if validation_results is None:
            results_file = self.validation_results_path / 'depth_validation_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                validation_results = [DepthValidationResult(**data) for data in results_data]
            else:
                logger.error("No validation results found")
                return {}
        
        if not validation_results:
            return {}
        
        # Extract metrics
        height_errors = []
        height_accuracies = []
        wave_type_matches = []
        direction_matches = []
        overall_accuracies = []
        quality_scores = []
        
        for result in validation_results:
            metrics = result.validation_metrics
            quality = result.quality_assessment
            
            if 'height_absolute_error' in metrics:
                height_errors.append(metrics['height_absolute_error'])
            if 'height_accuracy' in metrics:
                height_accuracies.append(metrics['height_accuracy'])
            if 'wave_type_match' in metrics:
                wave_type_matches.append(metrics['wave_type_match'])
            if 'direction_match' in metrics:
                direction_matches.append(metrics['direction_match'])
            if 'overall_accuracy' in metrics:
                overall_accuracies.append(metrics['overall_accuracy'])
            if 'overall_quality' in quality:
                quality_scores.append(quality['overall_quality'])
        
        # Compute statistics
        stats = {
            'total_images': len(validation_results),
            'height_analysis': self._compute_array_stats(height_errors, 'Height Error (m)'),
            'height_accuracy_analysis': self._compute_array_stats(height_accuracies, 'Height Accuracy'),
            'wave_type_accuracy': np.mean(wave_type_matches) if wave_type_matches else 0.0,
            'direction_accuracy': np.mean(direction_matches) if direction_matches else 0.0,
            'overall_accuracy_analysis': self._compute_array_stats(overall_accuracies, 'Overall Accuracy'),
            'quality_analysis': self._compute_array_stats(quality_scores, 'Quality Score'),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save statistical analysis
        stats_file = self.validation_results_path / 'statistical_analysis.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Generated statistical analysis for {len(validation_results)} images")
        return stats
    
    def _compute_array_stats(self, values: List[float], name: str) -> Dict[str, Any]:
        """Compute statistics for an array of values."""
        if not values:
            return {'name': name, 'count': 0}
        
        values_array = np.array(values)
        return {
            'name': name,
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'percentile_25': float(np.percentile(values_array, 25)),
            'percentile_75': float(np.percentile(values_array, 75))
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of MiDaS-real data integration.
        
        Returns:
            Dictionary with integration status
        """
        status = {
            'real_data_path': str(self.real_data_path),
            'midas_model': self.midas_extractor.model_name,
            'device': self.midas_extractor.device,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for manual labels
        labels = self.real_data_loader.load_manual_labels()
        status['manual_labels_count'] = len(labels)
        
        # Check for depth maps
        depth_maps = list(self.depth_maps_path.glob('*_depth.npy'))
        status['depth_maps_count'] = len(depth_maps)
        
        # Check for correspondences
        correspondence_file = self.validation_results_path / 'real_depth_correspondence.json'
        if correspondence_file.exists():
            with open(correspondence_file, 'r') as f:
                correspondences = json.load(f)
            status['correspondences_count'] = len(correspondences)
        else:
            status['correspondences_count'] = 0
        
        # Check for validation results
        results_file = self.validation_results_path / 'depth_validation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            status['validation_results_count'] = len(results)
        else:
            status['validation_results_count'] = 0
        
        # Integration completeness
        status['integration_complete'] = (
            status['manual_labels_count'] > 0 and
            status['depth_maps_count'] > 0 and
            status['correspondences_count'] > 0 and
            status['validation_results_count'] > 0
        )
        
        return status
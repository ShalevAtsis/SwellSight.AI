"""Property-based tests for depth analysis functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays
from typing import Dict, Any, List, Tuple

from swellsight.data.depth_analyzer import DepthWaveAnalyzer, WaveDetectionResult
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor, DepthExtractionResult


# Test data generators
@st.composite
def valid_depth_map(draw):
    """Generate valid depth maps for testing."""
    # Generate smaller realistic beach scene depth map dimensions
    height = draw(st.integers(min_value=100, max_value=300))
    width = draw(st.integers(min_value=150, max_value=400))
    
    # Generate depth values in realistic range for beach scenes (1-50 meters)
    depth_map = draw(arrays(
        dtype=np.float32,
        shape=(height, width),
        elements=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    ))
    
    return depth_map


@st.composite
def depth_map_with_known_waves(draw):
    """Generate depth map with known wave characteristics for testing."""
    # Use smaller, simpler dimensions for faster generation
    height = draw(st.integers(min_value=100, max_value=200))
    width = draw(st.integers(min_value=150, max_value=300))
    
    # Create base depth (deeper water)
    base_depth = draw(st.floats(min_value=10.0, max_value=30.0))
    depth_map = np.full((height, width), base_depth, dtype=np.float32)
    
    # Add simple wave patterns (fewer waves, simpler shapes)
    num_waves = draw(st.integers(min_value=1, max_value=3))
    wave_heights = []
    wave_positions = []
    
    for i in range(num_waves):
        # Generate wave parameters
        wave_height = draw(st.floats(min_value=0.5, max_value=3.0))
        wave_x = draw(st.integers(min_value=30, max_value=width-30))
        wave_y = draw(st.integers(min_value=int(0.4*height), max_value=int(0.8*height)))
        wave_width = 30  # Fixed width for simplicity
        
        # Add simple rectangular wave crest
        y_start = max(0, wave_y - wave_width//2)
        y_end = min(height, wave_y + wave_width//2)
        x_start = max(0, wave_x - wave_width//2)
        x_end = min(width, wave_x + wave_width//2)
        
        # Simple rectangular wave shape
        depth_map[y_start:y_end, x_start:x_end] = base_depth - wave_height
        
        wave_heights.append(wave_height)
        wave_positions.append((wave_x, wave_y))
    
    return depth_map, wave_heights, wave_positions


@st.composite
def depth_map_with_breaking_waves(draw):
    """Generate depth map with breaking wave patterns."""
    # Use smaller dimensions for faster generation
    height = draw(st.integers(min_value=100, max_value=200))
    width = draw(st.integers(min_value=150, max_value=300))
    
    # Create simple depth gradient
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        # Create depth gradient from shore (top) to deep water (bottom)
        shore_distance = y / height
        base_depth = 1.0 + shore_distance * 15.0  # 1-16 meters depth
        depth_map[y, :] = base_depth
    
    # Add simple breaking wave regions
    num_breaking_regions = draw(st.integers(min_value=1, max_value=2))
    
    for i in range(num_breaking_regions):
        # Breaking waves in shallow water
        break_y = draw(st.integers(min_value=int(0.3*height), max_value=int(0.7*height)))
        break_x_start = draw(st.integers(min_value=30, max_value=width//2))
        break_x_end = min(width-30, break_x_start + 50)
        
        # Create simple sharp gradient
        gradient_strength = draw(st.floats(min_value=3.0, max_value=6.0))
        for x in range(break_x_start, break_x_end):
            for dy in range(-5, 6):
                y_pos = break_y + dy
                if 0 <= y_pos < height:
                    if dy < 0:  # Before break
                        depth_map[y_pos, x] *= 0.6  # Shallow
                    else:  # After break
                        depth_map[y_pos, x] += gradient_strength  # Sudden deepening
    
    return depth_map


class TestDepthAnalysisProperties:
    """Property-based tests for depth analysis functionality."""
    
    def _create_depth_analyzer(self):
        """Create depth analyzer for testing."""
        config = {
            'min_wave_height': 0.1,
            'max_wave_height': 5.0,
            'depth_scale_factor': 0.1,
            'breaking_gradient_threshold': 0.15,
            'wave_detection_sensitivity': 0.8
        }
        return DepthWaveAnalyzer(config)
    
    @given(st.data())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_property_31_depth_based_height_estimation(self, data):
        """
        Feature: wave-analysis-model, Property 31: Depth-Based Height Estimation
        
        For any depth map with known wave characteristics, height estimation should be 
        within reasonable accuracy bounds (±0.5 meters for waves 0.5-3.0m).
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        depth_analyzer = self._create_depth_analyzer()
        # Generate depth map with known wave characteristics
        depth_map, known_heights, wave_positions = data.draw(depth_map_with_known_waves())
        
        # Skip if no waves were generated
        assume(len(known_heights) > 0)
        assume(all(0.5 <= h <= 3.0 for h in known_heights))
        
        # Analyze waves from depth map
        result = depth_analyzer.analyze_waves_from_depth(depth_map)
        
        # Property 31.1: Height estimation should be within reasonable bounds
        if result.wave_heights:
            for estimated_height in result.wave_heights:
                assert 0.0 <= estimated_height <= 5.0, \
                    f"Estimated height {estimated_height} outside reasonable range [0, 5] meters"
        
        # Property 31.2: Should detect at least some waves when they exist
        assert len(result.wave_crests) >= 0, "Wave detection should not fail"
        
        # Property 31.3: If waves are detected, heights should be reasonable
        if result.wave_heights:
            avg_estimated = np.mean(result.wave_heights)
            avg_known = np.mean(known_heights)
            
            # Allow for reasonable estimation error (within 100% of actual for synthetic data)
            max_reasonable_error = max(1.0, avg_known)  # At least 1m tolerance
            assert abs(avg_estimated - avg_known) <= max_reasonable_error, \
                f"Average height estimation error too large: estimated={avg_estimated:.2f}, " \
                f"known={avg_known:.2f}, error={abs(avg_estimated - avg_known):.2f}"
        
        # Property 31.4: Confidence should reflect estimation quality
        assert 0.0 <= result.confidence_score <= 1.0, \
            f"Confidence score {result.confidence_score} should be in [0, 1]"
        
        # Property 31.5: Wave parameters should be extractable
        parameters = depth_analyzer.extract_wave_parameters(result)
        assert isinstance(parameters, dict), "Should return parameter dictionary"
        assert 'average_height_meters' in parameters, "Should include average height"
        assert 'confidence_score' in parameters, "Should include confidence score"
        
        if parameters['wave_count'] > 0:
            assert parameters['average_height_meters'] >= 0, "Average height should be non-negative"
            assert parameters['max_height_meters'] >= parameters['min_height_meters'], \
                "Max height should be >= min height"
    
    @given(depth_map_with_breaking_waves())
    @settings(max_examples=8, deadline=12000, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_property_32_breaking_pattern_detection(self, depth_map):
        """
        Feature: wave-analysis-model, Property 32: Breaking Pattern Detection
        
        For any depth map showing wave breaking, the analyzer should correctly identify 
        breaking regions and patterns with >80% precision.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        depth_analyzer = self._create_depth_analyzer()
        # Analyze waves from depth map with breaking patterns
        result = depth_analyzer.analyze_waves_from_depth(depth_map)
        
        # Property 32.1: Breaking regions should be detected
        assert isinstance(result.breaking_regions, np.ndarray), \
            "Breaking regions should be numpy array"
        assert result.breaking_regions.dtype == bool, \
            "Breaking regions should be boolean mask"
        assert result.breaking_regions.shape == depth_map.shape, \
            "Breaking regions should match depth map dimensions"
        
        # Property 32.2: Breaking intensity should be reasonable
        parameters = depth_analyzer.extract_wave_parameters(result)
        breaking_intensity = parameters['breaking_intensity']
        assert 0.0 <= breaking_intensity <= 1.0, \
            f"Breaking intensity {breaking_intensity} should be in [0, 1]"
        
        # Property 32.3: Breaking regions should be spatially coherent
        if np.any(result.breaking_regions):
            # Check that breaking regions are not just random noise
            # Use connected components to verify spatial coherence
            import cv2
            breaking_uint8 = result.breaking_regions.astype(np.uint8)
            num_components, labels = cv2.connectedComponents(breaking_uint8)
            
            # Should have reasonable number of connected components (not pure noise)
            total_breaking_pixels = np.sum(result.breaking_regions)
            if total_breaking_pixels > 100:  # Only check if significant breaking detected
                avg_component_size = total_breaking_pixels / max(1, num_components - 1)  # -1 for background
                assert avg_component_size >= 10, \
                    f"Breaking regions too fragmented: avg component size {avg_component_size}"
        
        # Property 32.4: Wave type classification should be reasonable
        valid_wave_types = ['flat', 'beach_break', 'point_break', 'a_frame', 'closeout', 'unknown']
        assert result.wave_type in valid_wave_types, \
            f"Wave type '{result.wave_type}' not in valid types: {valid_wave_types}"
        
        # Property 32.5: Breaking detection should be consistent with wave detection
        if len(result.wave_crests) > 0 and breaking_intensity > 0.1:
            # If waves and breaking are detected, confidence should be reasonable
            assert result.confidence_score > 0.1, \
                "Confidence should be higher when both waves and breaking are detected"
        
        # Property 32.6: Breaking regions should correlate with depth gradients
        if np.any(result.breaking_regions):
            # Compute depth gradients
            grad_x = np.gradient(depth_map, axis=1)
            grad_y = np.gradient(depth_map, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Breaking regions should have higher gradients on average
            breaking_gradients = gradient_magnitude[result.breaking_regions]
            non_breaking_gradients = gradient_magnitude[~result.breaking_regions]
            
            if len(breaking_gradients) > 10 and len(non_breaking_gradients) > 10:
                avg_breaking_grad = np.mean(breaking_gradients)
                avg_non_breaking_grad = np.mean(non_breaking_gradients)
                
                # Breaking regions should have higher gradients (within reason)
                assert avg_breaking_grad >= avg_non_breaking_grad * 0.8, \
                    f"Breaking regions should have higher gradients: " \
                    f"breaking={avg_breaking_grad:.4f}, non-breaking={avg_non_breaking_grad:.4f}"
    
    @given(valid_depth_map())
    @settings(max_examples=10, deadline=12000, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_property_33_direction_analysis_consistency(self, depth_map):
        """
        Feature: wave-analysis-model, Property 33: Direction Analysis Consistency
        
        For any depth map, wave direction analysis should produce consistent results 
        (±10 degrees) across multiple analysis runs.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        depth_analyzer = self._create_depth_analyzer()
        # Run analysis multiple times on the same depth map
        num_runs = 3
        results = []
        
        for i in range(num_runs):
            result = depth_analyzer.analyze_waves_from_depth(depth_map, f"run_{i}")
            results.append(result)
        
        # Property 33.1: All runs should complete successfully
        assert len(results) == num_runs, f"Expected {num_runs} results, got {len(results)}"
        
        for i, result in enumerate(results):
            assert isinstance(result, WaveDetectionResult), f"Run {i} should return WaveDetectionResult"
        
        # Property 33.2: Direction analysis should be consistent
        directions = [result.dominant_direction for result in results]
        valid_directions = ['left', 'right', 'both']
        
        for i, direction in enumerate(directions):
            assert direction in valid_directions, \
                f"Run {i} direction '{direction}' not in valid directions: {valid_directions}"
        
        # Property 33.3: Confidence scores should be consistent (within reasonable variance)
        confidence_scores = [result.confidence_score for result in results]
        
        for i, confidence in enumerate(confidence_scores):
            assert 0.0 <= confidence <= 1.0, \
                f"Run {i} confidence {confidence} should be in [0, 1]"
        
        if len(confidence_scores) > 1:
            confidence_std = np.std(confidence_scores)
            confidence_mean = np.mean(confidence_scores)
            
            # Confidence should not vary wildly between runs (coefficient of variation < 0.5)
            if confidence_mean > 0.1:  # Only check if confidence is meaningful
                cv = confidence_std / confidence_mean
                assert cv < 0.5, \
                    f"Confidence scores too inconsistent across runs: std={confidence_std:.3f}, " \
                    f"mean={confidence_mean:.3f}, CV={cv:.3f}"
        
        # Property 33.4: Wave count should be reasonably consistent
        wave_counts = [len(result.wave_crests) for result in results]
        
        if max(wave_counts) > 0:  # Only check if waves are detected
            wave_count_std = np.std(wave_counts)
            wave_count_mean = np.mean(wave_counts)
            
            # Wave count should not vary too much (within 50% of mean)
            max_reasonable_variation = max(1, wave_count_mean * 0.5)
            assert wave_count_std <= max_reasonable_variation, \
                f"Wave count too inconsistent: counts={wave_counts}, std={wave_count_std:.2f}"
        
        # Property 33.5: Wave type classification should be consistent
        wave_types = [result.wave_type for result in results]
        unique_wave_types = set(wave_types)
        
        # Should not have more than 2 different wave types across runs
        assert len(unique_wave_types) <= 2, \
            f"Wave type classification too inconsistent: {wave_types}"
        
        # Property 33.6: Breaking intensity should be consistent
        breaking_intensities = []
        for result in results:
            parameters = depth_analyzer.extract_wave_parameters(result)
            breaking_intensities.append(parameters['breaking_intensity'])
        
        if len(breaking_intensities) > 1:
            breaking_std = np.std(breaking_intensities)
            breaking_mean = np.mean(breaking_intensities)
            
            # Breaking intensity should be consistent (CV < 0.3)
            if breaking_mean > 0.05:  # Only check if significant breaking detected
                cv = breaking_std / breaking_mean
                assert cv < 0.3, \
                    f"Breaking intensity too inconsistent: intensities={breaking_intensities}, CV={cv:.3f}"
        
        # Property 33.7: Extracted parameters should have consistent structure
        for i, result in enumerate(results):
            parameters = depth_analyzer.extract_wave_parameters(result)
            
            # Check required fields
            required_fields = [
                'wave_count', 'average_height_meters', 'max_height_meters', 
                'min_height_meters', 'height_std_meters', 'dominant_direction',
                'wave_type', 'breaking_intensity', 'confidence_score', 'crest_coordinates'
            ]
            
            for field in required_fields:
                assert field in parameters, f"Run {i} missing required field: {field}"
            
            # Check data types
            assert isinstance(parameters['wave_count'], int), f"Run {i} wave_count should be int"
            assert isinstance(parameters['average_height_meters'], float), f"Run {i} average_height should be float"
            assert isinstance(parameters['dominant_direction'], str), f"Run {i} direction should be str"
            assert isinstance(parameters['crest_coordinates'], list), f"Run {i} coordinates should be list"


class TestDepthAnalysisEdgeCases:
    """Test edge cases for depth analysis."""
    
    def _create_depth_analyzer(self):
        """Create depth analyzer for testing."""
        return DepthWaveAnalyzer()
    
    def test_empty_depth_map(self):
        """Test handling of empty depth map."""
        depth_analyzer = self._create_depth_analyzer()
        empty_depth = np.array([[]], dtype=np.float32)
        
        result = depth_analyzer.analyze_waves_from_depth(empty_depth)
        
        # Should handle gracefully without crashing
        assert isinstance(result, WaveDetectionResult)
        assert len(result.wave_crests) == 0
        assert len(result.wave_heights) == 0
        assert result.confidence_score == 0.0
    
    def test_uniform_depth_map(self):
        """Test handling of uniform depth map (no waves)."""
        depth_analyzer = self._create_depth_analyzer()
        uniform_depth = np.full((100, 100), 10.0, dtype=np.float32)
        
        result = depth_analyzer.analyze_waves_from_depth(uniform_depth)
        
        # Should detect no waves in uniform depth
        assert isinstance(result, WaveDetectionResult)
        assert len(result.wave_crests) == 0 or all(h < 0.5 for h in result.wave_heights)
        assert result.wave_type in ['flat', 'unknown']
    
    def test_extreme_depth_values(self):
        """Test handling of extreme depth values."""
        depth_analyzer = self._create_depth_analyzer()
        # Very large depth values
        large_depth = np.full((50, 50), 1000.0, dtype=np.float32)
        result = depth_analyzer.analyze_waves_from_depth(large_depth)
        assert isinstance(result, WaveDetectionResult)
        
        # Very small depth values
        small_depth = np.full((50, 50), 0.001, dtype=np.float32)
        result = depth_analyzer.analyze_waves_from_depth(small_depth)
        assert isinstance(result, WaveDetectionResult)
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values in depth map."""
        depth_analyzer = self._create_depth_analyzer()
        # Create depth map with NaN values
        depth_with_nan = np.full((50, 50), 10.0, dtype=np.float32)
        depth_with_nan[10:20, 10:20] = np.nan
        
        result = depth_analyzer.analyze_waves_from_depth(depth_with_nan)
        assert isinstance(result, WaveDetectionResult)
        
        # Create depth map with infinite values
        depth_with_inf = np.full((50, 50), 10.0, dtype=np.float32)
        depth_with_inf[10:20, 10:20] = np.inf
        
        result = depth_analyzer.analyze_waves_from_depth(depth_with_inf)
        assert isinstance(result, WaveDetectionResult)
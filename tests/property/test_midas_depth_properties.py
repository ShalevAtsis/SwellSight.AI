"""Property-based tests for MiDaS depth extraction components."""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor, DepthExtractionResult


# Test data strategies
@st.composite
def beach_image_strategy(draw):
    """Generate valid beach camera images for testing."""
    # Create realistic beach scene image
    width = draw(st.integers(min_value=512, max_value=1024))
    height = draw(st.integers(min_value=384, max_value=768))
    
    # Create a simple beach scene with gradient (sky to water)
    image = Image.new('RGB', (width, height))
    pixels = []
    
    for y in range(height):
        for x in range(width):
            # Create gradient from sky (light blue) to water (dark blue)
            sky_ratio = y / height
            
            # Sky color (light blue to white)
            if sky_ratio < 0.3:
                r = int(135 + (255 - 135) * (1 - sky_ratio / 0.3))
                g = int(206 + (255 - 206) * (1 - sky_ratio / 0.3))
                b = int(235 + (255 - 235) * (1 - sky_ratio / 0.3))
            # Water color (blue to dark blue)
            else:
                water_ratio = (sky_ratio - 0.3) / 0.7
                r = int(65 * (1 - water_ratio) + 25 * water_ratio)
                g = int(105 * (1 - water_ratio) + 55 * water_ratio)
                b = int(225 * (1 - water_ratio) + 155 * water_ratio)
            
            # Add some horizontal variation for waves
            wave_offset = int(10 * np.sin(x * 0.02 + y * 0.01))
            r = max(0, min(255, r + wave_offset))
            g = max(0, min(255, g + wave_offset))
            b = max(0, min(255, b + wave_offset))
            
            pixels.append((r, g, b))
    
    image.putdata(pixels)
    return image


@st.composite
def batch_images_strategy(draw):
    """Generate a batch of beach images for testing."""
    batch_size = draw(st.integers(min_value=2, max_value=5))
    images = []
    
    for _ in range(batch_size):
        image = draw(beach_image_strategy())
        images.append(image)
    
    return images


class TestMiDaSDepthExtractionProperties:
    """Property-based tests for MiDaS depth extraction."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / 'images').mkdir(parents=True)
        (self.temp_path / 'depth_maps').mkdir(parents=True)
        
        # Initialize MiDaS extractor with CPU for testing
        self.extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            device="cpu"  # Use CPU for consistent testing
        )
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(beach_image_strategy())
    @settings(max_examples=3, deadline=None)
    def test_depth_map_generation_consistency(self, beach_image):
        """
        Feature: wave-analysis-model, Property 22: Depth Map Generation Consistency
        
        For any valid beach image, MiDaS should produce a depth map with same spatial 
        dimensions and valid depth values within expected ranges for beach scenes.
        """
        # Save test image
        image_path = self.temp_path / 'images' / 'test_beach.jpg'
        beach_image.save(image_path, 'JPEG')
        
        # Extract depth map
        result = self.extractor.extract_depth(str(image_path))
        
        # Verify result structure
        assert isinstance(result, DepthExtractionResult)
        assert isinstance(result.depth_map, np.ndarray)
        assert isinstance(result.depth_quality_score, float)
        assert isinstance(result.processing_metadata, dict)
        
        # Verify depth map properties
        depth_map = result.depth_map
        original_size = beach_image.size  # (width, height)
        
        # Check spatial dimensions match original image
        assert depth_map.shape == (original_size[1], original_size[0])  # (height, width)
        
        # Check data type
        assert depth_map.dtype == np.float32
        
        # Check depth values are valid (no NaN, inf, negative values)
        assert not np.any(np.isnan(depth_map))
        assert not np.any(np.isinf(depth_map))
        assert np.all(depth_map > 0)  # Positive depths only
        
        # Check depth range is reasonable for beach scenes (1-100 meters)
        assert np.min(depth_map) >= 0.5  # Minimum reasonable depth
        assert np.max(depth_map) <= 150.0  # Maximum reasonable depth
        
        # Check depth variation (beach scenes should have depth variation)
        depth_std = np.std(depth_map)
        depth_mean = np.mean(depth_map)
        variation_coefficient = depth_std / depth_mean
        assert variation_coefficient > 0.01  # Some variation expected
        
        # Verify processing metadata
        metadata = result.processing_metadata
        assert 'model_name' in metadata
        assert 'device' in metadata
        assert 'original_image_size' in metadata
        assert 'depth_map_shape' in metadata
        assert 'depth_range' in metadata
        
        assert metadata['model_name'] == "Intel/dpt-large"
        assert metadata['original_image_size'] == original_size
        assert metadata['depth_map_shape'] == depth_map.shape
        
        depth_range = metadata['depth_range']
        assert len(depth_range) == 2
        assert depth_range[0] == float(depth_map.min())
        assert depth_range[1] == float(depth_map.max())
        
        # Verify original image path is preserved
        assert result.original_image_path == str(image_path)
    
    @given(beach_image_strategy())
    @settings(max_examples=3, deadline=None)
    def test_depth_quality_validation(self, beach_image):
        """
        Feature: wave-analysis-model, Property 23: Depth Quality Validation
        
        For any generated depth map, quality metrics should be within acceptable 
        ranges for beach scene analysis, with scores between 0.0 and 1.0.
        """
        # Save test image
        image_path = self.temp_path / 'images' / 'test_beach_quality.jpg'
        beach_image.save(image_path, 'JPEG')
        
        # Extract depth map
        result = self.extractor.extract_depth(str(image_path))
        depth_map = result.depth_map
        quality_score = result.depth_quality_score
        
        # Test 1: Quality score is in valid range
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)
        
        # Test 2: Quality validation method works independently
        independent_quality = self.extractor.validate_depth_quality(depth_map)
        assert 0.0 <= independent_quality <= 1.0
        assert abs(quality_score - independent_quality) < 1e-6  # Should be identical
        
        # Test 3: Quality metrics are reasonable for beach scenes
        # Beach scenes should have moderate to good quality scores
        assert quality_score > 0.1  # Not completely invalid
        
        # Test 4: Quality validation handles edge cases
        # Test with uniform depth map (should have low quality)
        uniform_depth = np.ones_like(depth_map) * 10.0
        uniform_quality = self.extractor.validate_depth_quality(uniform_depth)
        assert uniform_quality < 0.5  # Low quality for uniform depth
        
        # Test with NaN values (should return 0.0)
        nan_depth = depth_map.copy()
        nan_depth[0, 0] = np.nan
        nan_quality = self.extractor.validate_depth_quality(nan_depth)
        assert nan_quality == 0.0
        
        # Test with infinite values (should return 0.0)
        inf_depth = depth_map.copy()
        inf_depth[0, 0] = np.inf
        inf_quality = self.extractor.validate_depth_quality(inf_depth)
        assert inf_quality == 0.0
        
        # Test 5: Quality components are computed correctly
        # Verify depth variation component
        depth_std = np.std(depth_map)
        depth_mean = np.mean(depth_map)
        variation_score = min(depth_std / (depth_mean + 1e-6), 1.0)
        assert 0.0 <= variation_score <= 1.0
        
        # Verify gradient consistency component
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        depth_range = depth_map.max() - depth_map.min()
        gradient_score = 1.0 - min(np.mean(gradient_magnitude) / depth_range, 1.0)
        assert 0.0 <= gradient_score <= 1.0
        
        # Verify spatial coherence component
        kernel = np.ones((3, 3)) / 9
        smoothed = cv2.filter2D(depth_map, -1, kernel)
        coherence_error = np.mean(np.abs(depth_map - smoothed))
        coherence_score = 1.0 - min(coherence_error / depth_range, 1.0)
        assert 0.0 <= coherence_score <= 1.0
        
        # Verify dynamic range utilization component
        depth_normalized = (depth_map - depth_map.min()) / depth_range
        hist, _ = np.histogram(depth_normalized, bins=50)
        hist_normalized = hist / hist.sum()
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        range_score = min(entropy / np.log(50), 1.0)
        assert 0.0 <= range_score <= 1.0
    
    @given(batch_images_strategy())
    @settings(max_examples=2, deadline=None)
    def test_batch_processing_consistency(self, batch_images):
        """
        Feature: wave-analysis-model, Property 24: Batch Processing Consistency
        
        For any batch of images, individual vs batch processing should produce 
        identical results with consistent depth map properties across all images.
        """
        # Save batch images
        image_paths = []
        for i, image in enumerate(batch_images):
            image_path = self.temp_path / 'images' / f'batch_test_{i}.jpg'
            image.save(image_path, 'JPEG')
            image_paths.append(str(image_path))
        
        # Test 1: Individual processing
        individual_results = []
        for image_path in image_paths:
            result = self.extractor.extract_depth(image_path)
            individual_results.append(result)
        
        # Test 2: Batch processing
        batch_results = self.extractor.batch_extract(image_paths)
        
        # Test 3: Compare results
        assert len(individual_results) == len(batch_results)
        assert len(batch_results) == len(batch_images)
        
        for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
            # Verify both are DepthExtractionResult objects
            assert isinstance(individual, DepthExtractionResult)
            assert isinstance(batch, DepthExtractionResult)
            
            # Verify image paths match
            assert individual.original_image_path == batch.original_image_path
            assert individual.original_image_path == image_paths[i]
            
            # Verify depth maps are identical (or very close due to floating point)
            np.testing.assert_allclose(
                individual.depth_map, 
                batch.depth_map, 
                rtol=1e-6, 
                atol=1e-6,
                err_msg=f"Depth maps differ for image {i}"
            )
            
            # Verify quality scores are identical
            assert abs(individual.depth_quality_score - batch.depth_quality_score) < 1e-6
            
            # Verify metadata consistency
            individual_meta = individual.processing_metadata
            batch_meta = batch.processing_metadata
            
            assert individual_meta['model_name'] == batch_meta['model_name']
            assert individual_meta['device'] == batch_meta['device']
            assert individual_meta['original_image_size'] == batch_meta['original_image_size']
            assert individual_meta['depth_map_shape'] == batch_meta['depth_map_shape']
            
            # Depth ranges should be very close
            ind_range = individual_meta['depth_range']
            batch_range = batch_meta['depth_range']
            assert abs(ind_range[0] - batch_range[0]) < 1e-6
            assert abs(ind_range[1] - batch_range[1]) < 1e-6
        
        # Test 4: Verify batch processing efficiency
        # Batch processing should handle all images without errors
        assert len(batch_results) == len(image_paths)
        
        # All results should have valid depth maps
        for result in batch_results:
            assert isinstance(result.depth_map, np.ndarray)
            assert result.depth_map.dtype == np.float32
            assert not np.any(np.isnan(result.depth_map))
            assert not np.any(np.isinf(result.depth_map))
            assert np.all(result.depth_map > 0)
        
        # Test 5: Verify consistent processing across batch
        # All images should use same model and device
        model_names = [result.processing_metadata['model_name'] for result in batch_results]
        devices = [result.processing_metadata['device'] for result in batch_results]
        
        assert all(name == model_names[0] for name in model_names)
        assert all(device == devices[0] for device in devices)
        
        # Quality scores should all be valid
        quality_scores = [result.depth_quality_score for result in batch_results]
        assert all(0.0 <= score <= 1.0 for score in quality_scores)


class TestMiDaSDepthStorageProperties:
    """Property-based tests for MiDaS depth map storage and retrieval."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / 'images').mkdir(parents=True)
        (self.temp_path / 'depth_maps').mkdir(parents=True)
        
        # Initialize MiDaS extractor
        self.extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            device="cpu"
        )
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(beach_image_strategy())
    @settings(max_examples=2, deadline=None)
    def test_depth_map_storage_round_trip(self, beach_image):
        """
        Test that depth maps can be saved and loaded without data loss.
        
        For any depth map, saving and loading should preserve the data
        with acceptable precision for different storage formats.
        """
        # Save test image and extract depth
        image_path = self.temp_path / 'images' / 'test_storage.jpg'
        beach_image.save(image_path, 'JPEG')
        
        result = self.extractor.extract_depth(str(image_path))
        original_depth = result.depth_map
        
        # Test different storage formats
        formats = ['npy', 'png', 'tiff']
        
        for fmt in formats:
            # Save depth map
            depth_save_path = self.temp_path / 'depth_maps' / f'test_depth.{fmt}'
            self.extractor.save_depth_map(original_depth, str(depth_save_path), format=fmt)
            
            # Verify file was created
            assert depth_save_path.exists()
            
            # Load depth map
            loaded_depth = self.extractor.load_depth_map(str(depth_save_path))
            
            # Verify loaded depth map properties
            assert isinstance(loaded_depth, np.ndarray)
            assert loaded_depth.dtype == np.float32
            assert loaded_depth.shape == original_depth.shape
            
            # Verify data preservation (with format-specific tolerances)
            if fmt == 'npy':
                # NumPy format should preserve exact values
                np.testing.assert_allclose(
                    original_depth, loaded_depth, 
                    rtol=1e-6, atol=1e-6
                )
            elif fmt == 'png':
                # PNG format has quantization, allow larger tolerance
                np.testing.assert_allclose(
                    original_depth, loaded_depth, 
                    rtol=0.01, atol=0.1
                )
            elif fmt == 'tiff':
                # TIFF format should preserve float32 precision
                np.testing.assert_allclose(
                    original_depth, loaded_depth, 
                    rtol=1e-5, atol=1e-5
                )
            
            # Verify depth values are still valid
            assert not np.any(np.isnan(loaded_depth))
            assert not np.any(np.isinf(loaded_depth))
            assert np.all(loaded_depth > 0)
    
    @given(st.lists(beach_image_strategy(), min_size=2, max_size=4))
    @settings(max_examples=1, deadline=None)
    def test_multiple_depth_maps_storage(self, beach_images):
        """
        Test storage and retrieval of multiple depth maps.
        
        For any collection of depth maps, each should be stored and loaded
        independently without interference.
        """
        depth_maps = []
        original_paths = []
        
        # Extract depth maps from all images
        for i, image in enumerate(beach_images):
            image_path = self.temp_path / 'images' / f'multi_test_{i}.jpg'
            image.save(image_path, 'JPEG')
            original_paths.append(str(image_path))
            
            result = self.extractor.extract_depth(str(image_path))
            depth_maps.append(result.depth_map)
        
        # Save all depth maps
        saved_paths = []
        for i, depth_map in enumerate(depth_maps):
            depth_save_path = self.temp_path / 'depth_maps' / f'multi_depth_{i}.npy'
            self.extractor.save_depth_map(depth_map, str(depth_save_path), format='npy')
            saved_paths.append(str(depth_save_path))
        
        # Load all depth maps
        loaded_depth_maps = []
        for save_path in saved_paths:
            loaded_depth = self.extractor.load_depth_map(save_path)
            loaded_depth_maps.append(loaded_depth)
        
        # Verify all depth maps were preserved correctly
        assert len(loaded_depth_maps) == len(depth_maps)
        
        for original, loaded in zip(depth_maps, loaded_depth_maps):
            assert original.shape == loaded.shape
            assert original.dtype == loaded.dtype
            np.testing.assert_allclose(original, loaded, rtol=1e-6, atol=1e-6)
        
        # Verify no cross-contamination between depth maps
        for i in range(len(depth_maps)):
            for j in range(i + 1, len(depth_maps)):
                # Different depth maps should not be identical
                assert not np.allclose(loaded_depth_maps[i], loaded_depth_maps[j], rtol=1e-3)
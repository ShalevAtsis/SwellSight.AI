"""Integration tests for MiDaS depth extraction with storage system."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import json

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor
from swellsight.data.depth_map_storage import DepthMapStorage, DepthMapVisualizer


class TestMiDaSStorageIntegration:
    """Integration tests for MiDaS extractor with depth map storage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.images_dir = self.temp_dir / 'images'
        self.storage_dir = self.temp_dir / 'storage'
        self.images_dir.mkdir(parents=True)
        self.storage_dir.mkdir(parents=True)
        
        # Create test images
        self.test_images = []
        for i in range(3):
            # Create synthetic beach-like image
            image = Image.new('RGB', (320, 240), color=(135, 206, 235))  # Sky blue
            
            # Add some variation to make it more realistic
            pixels = np.array(image)
            # Add beach (sand color at bottom)
            pixels[180:, :] = [194, 178, 128]  # Sandy color
            # Add water (blue-green in middle)
            pixels[120:180, :] = [64, 164, 223]  # Water color
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(pixels)
            image_path = self.images_dir / f'beach_test_{i:03d}.jpg'
            image.save(image_path, 'JPEG')
            self.test_images.append(image_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_depth_extraction_and_storage(self):
        """Test complete workflow from image to stored depth map."""
        # Initialize MiDaS extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Process first test image
        image_path = self.test_images[0]
        result = extractor.extract_depth(str(image_path), store_result=True)
        
        # Verify extraction result
        assert result.depth_map is not None
        assert result.depth_map.shape[0] > 0
        assert result.depth_map.shape[1] > 0
        assert result.depth_quality_score >= 0.0
        assert result.depth_quality_score <= 1.0
        assert result.original_image_path == str(image_path)
        
        # Verify depth map was stored
        storage = extractor.storage
        assert storage is not None
        
        # Find stored depth map by image filename
        stored_metadata = extractor.find_stored_depth_maps_by_image(image_path.name)
        assert len(stored_metadata) == 1
        
        stored_meta = stored_metadata[0]
        assert stored_meta.image_filename == image_path.name
        assert stored_meta.midas_model_version == "Intel/dpt-large"
        
        # Retrieve stored depth map
        retrieved_depth, retrieved_metadata = extractor.retrieve_stored_depth_map(stored_meta.depth_map_id)
        
        # Verify retrieved data matches original
        np.testing.assert_allclose(retrieved_depth, result.depth_map, rtol=1e-5)
        assert retrieved_metadata.depth_map_id == stored_meta.depth_map_id
    
    def test_batch_processing_with_storage(self):
        """Test batch processing multiple images with storage."""
        # Initialize extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Process all test images
        image_paths = [str(img_path) for img_path in self.test_images]
        results = extractor.batch_extract(image_paths)
        
        # Verify all images were processed
        assert len(results) == len(self.test_images)
        
        # Store all results
        stored_ids = []
        for result in results:
            if extractor.storage:
                stored_id = extractor._store_depth_map_result(result)
                stored_ids.append(stored_id)
        
        assert len(stored_ids) == len(results)
        
        # Verify all depth maps can be retrieved
        for i, stored_id in enumerate(stored_ids):
            retrieved_depth, metadata = extractor.retrieve_stored_depth_map(stored_id)
            original_depth = results[i].depth_map
            
            np.testing.assert_allclose(retrieved_depth, original_depth, rtol=1e-5)
            assert metadata.image_filename == self.test_images[i].name
    
    def test_storage_statistics_and_management(self):
        """Test storage statistics and management features."""
        # Initialize extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Process and store multiple images
        for image_path in self.test_images:
            result = extractor.extract_depth(str(image_path), store_result=True)
        
        # Get storage statistics
        stats = extractor.get_storage_statistics()
        
        # Verify statistics structure and content
        assert 'total_depth_maps' in stats
        assert 'model_statistics' in stats
        assert 'quality_distribution' in stats
        assert 'storage_efficiency' in stats
        
        assert stats['total_depth_maps'] == len(self.test_images)
        assert len(stats['model_statistics']) == 1
        assert stats['model_statistics'][0]['model_version'] == "Intel/dpt-large"
        assert stats['model_statistics'][0]['count'] == len(self.test_images)
        
        # Test cleanup functionality
        storage = extractor.storage
        
        # Store additional versions of first image
        first_image = self.test_images[0]
        for i in range(2):
            result = extractor.extract_depth(str(first_image), store_result=True)
        
        # Should now have 5 total depth maps (3 original + 2 additional)
        updated_stats = extractor.get_storage_statistics()
        assert updated_stats['total_depth_maps'] == len(self.test_images) + 2
        
        # Find all versions of first image
        versions = extractor.find_stored_depth_maps_by_image(first_image.name)
        assert len(versions) == 3  # Original + 2 additional
        
        # Cleanup old versions, keeping only latest 1
        deleted_count = storage.cleanup_old_versions(keep_latest_n=1)
        assert deleted_count >= 2  # Should delete at least 2 old versions
        
        # Verify cleanup worked
        remaining_versions = extractor.find_stored_depth_maps_by_image(first_image.name)
        assert len(remaining_versions) == 1
    
    def test_visualization_integration(self):
        """Test depth map visualization integration."""
        # Initialize extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Process and store image
        image_path = self.test_images[0]
        result = extractor.extract_depth(str(image_path), store_result=True)
        
        # Find stored depth map
        stored_metadata = extractor.find_stored_depth_maps_by_image(image_path.name)
        assert len(stored_metadata) == 1
        
        depth_map_id = stored_metadata[0].depth_map_id
        
        # Create visualizer
        visualizer = DepthMapVisualizer(extractor.storage)
        
        # Create single depth map visualization
        viz_path = self.temp_dir / 'single_visualization.png'
        visualizer.create_depth_visualization(depth_map_id, str(viz_path))
        
        # Verify visualization was created
        assert viz_path.exists()
        assert viz_path.stat().st_size > 0
        
        # Process second image for comparison
        second_image = self.test_images[1]
        result2 = extractor.extract_depth(str(second_image), store_result=True)
        
        stored_metadata2 = extractor.find_stored_depth_maps_by_image(second_image.name)
        depth_map_id2 = stored_metadata2[0].depth_map_id
        
        # Create comparison visualization
        comparison_path = self.temp_dir / 'comparison_visualization.png'
        visualizer.create_comparison_visualization(
            [depth_map_id, depth_map_id2], 
            str(comparison_path)
        )
        
        # Verify comparison visualization was created
        assert comparison_path.exists()
        assert comparison_path.stat().st_size > 0
    
    def test_different_storage_formats(self):
        """Test different storage formats work correctly."""
        formats = ['compressed_npy', 'gzip_pickle', 'npy']
        
        for storage_format in formats:
            # Create separate storage for each format
            format_storage_dir = self.storage_dir / storage_format
            format_storage_dir.mkdir(exist_ok=True)
            
            # Initialize extractor with specific storage format
            extractor = MiDaSDepthExtractor(
                model_name="Intel/dpt-large",
                storage_path=str(format_storage_dir)
            )
            
            # Configure storage format
            extractor.storage.storage_format = storage_format
            
            # Process and store image
            image_path = self.test_images[0]
            result = extractor.extract_depth(str(image_path), store_result=True)
            
            # Verify storage worked
            stored_metadata = extractor.find_stored_depth_maps_by_image(image_path.name)
            assert len(stored_metadata) == 1
            assert stored_metadata[0].storage_format == storage_format
            
            # Verify retrieval works
            retrieved_depth, _ = extractor.retrieve_stored_depth_map(stored_metadata[0].depth_map_id)
            
            # Verify data integrity based on format
            if storage_format in ['npy', 'gzip_pickle']:
                np.testing.assert_allclose(retrieved_depth, result.depth_map, rtol=1e-6)
            else:  # compressed_npy
                np.testing.assert_allclose(retrieved_depth, result.depth_map, rtol=1e-5)
    
    def test_metadata_export_and_analysis(self):
        """Test metadata export and analysis capabilities."""
        # Initialize extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Process all test images
        for image_path in self.test_images:
            result = extractor.extract_depth(str(image_path), store_result=True)
        
        # Export metadata
        export_path = self.temp_dir / 'exported_metadata.json'
        extractor.storage.export_metadata(str(export_path))
        
        # Verify export file exists and has content
        assert export_path.exists()
        
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == len(self.test_images)
        
        # Verify exported data structure
        for item in exported_data:
            assert 'depth_map_id' in item
            assert 'image_filename' in item
            assert 'midas_model_version' in item
            assert 'quality_score' in item
            assert 'depth_map_shape' in item
            assert 'depth_range' in item
            
            # Verify values are reasonable
            assert item['midas_model_version'] == "Intel/dpt-large"
            assert 0.0 <= item['quality_score'] <= 1.0
            assert len(item['depth_map_shape']) == 2
            assert len(item['depth_range']) == 2
            assert item['depth_range'][0] < item['depth_range'][1]
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in integrated system."""
        # Initialize extractor with storage
        extractor = MiDaSDepthExtractor(
            model_name="Intel/dpt-large",
            storage_path=str(self.storage_dir)
        )
        
        # Test with non-existent image
        with pytest.raises(FileNotFoundError):
            extractor.extract_depth("/nonexistent/image.jpg")
        
        # Test with invalid depth map ID
        with pytest.raises(ValueError):
            extractor.retrieve_stored_depth_map("nonexistent_id")
        
        # Test storage without initialization
        extractor_no_storage = MiDaSDepthExtractor(model_name="Intel/dpt-large")
        
        with pytest.raises(ValueError, match="Storage system not initialized"):
            extractor_no_storage.get_storage_statistics()
        
        # Test successful processing after errors
        image_path = self.test_images[0]
        result = extractor.extract_depth(str(image_path), store_result=True)
        
        # Verify normal operation continues after errors
        assert result.depth_map is not None
        assert result.depth_quality_score >= 0.0
        
        stored_metadata = extractor.find_stored_depth_maps_by_image(image_path.name)
        assert len(stored_metadata) == 1


if __name__ == "__main__":
    pytest.main([__file__])
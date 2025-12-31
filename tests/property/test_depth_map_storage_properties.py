"""Property-based tests for depth map storage system."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from swellsight.data.depth_map_storage import (
    DepthMapStorage, 
    DepthMapMetadata, 
    DepthMapVisualizer
)


# Hypothesis strategies
@st.composite
def depth_map_strategy(draw):
    """Generate valid depth maps for testing."""
    height = draw(st.integers(min_value=50, max_value=200))  # Reduced max size
    width = draw(st.integers(min_value=50, max_value=200))   # Reduced max size
    
    # Generate depth values in realistic range for beach scenes (1-100 meters)
    depth_map = draw(arrays(
        dtype=np.float32,
        shape=(height, width),
        elements=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    ))
    
    return depth_map


@st.composite
def metadata_strategy(draw, depth_map_shape=None):
    """Generate valid DepthMapMetadata for testing."""
    if depth_map_shape is None:
        depth_map_shape = (100, 150)
    
    depth_map_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))))
    image_filename = draw(st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))) + ".jpg"
    model_version = draw(st.sampled_from(["Intel/dpt-large", "Intel/dpt-hybrid-midas", "Intel/dpt-small"]))
    quality_score = draw(st.floats(min_value=0.0, max_value=1.0))
    depth_range = draw(st.tuples(
        st.floats(min_value=1.0, max_value=50.0),
        st.floats(min_value=50.0, max_value=100.0)
    ))
    storage_format = draw(st.sampled_from(["compressed_npy", "gzip_pickle", "npy"]))
    
    return DepthMapMetadata(
        depth_map_id=depth_map_id,
        original_image_path=f"/test/images/{image_filename}",
        image_filename=image_filename,
        midas_model_version=model_version,
        extraction_timestamp=datetime.now().isoformat(),
        depth_map_shape=depth_map_shape,
        depth_range=depth_range,
        quality_score=quality_score,
        storage_format=storage_format
    )


class TestDepthMapStorageProperties:
    """Property-based tests for DepthMapStorage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(depth_map_strategy(), st.sampled_from(["compressed_npy", "gzip_pickle", "npy"]))
    @settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_storage_round_trip_preserves_data(self, depth_map, storage_format):
        """
        Property: For any depth map and storage format, storing and retrieving 
        should preserve the data within acceptable precision.
        
        Validates: Requirements 8.1, 8.2 (model persistence and metadata tracking)
        """
        # Create storage with specified format
        storage = DepthMapStorage(str(self.temp_dir), {'storage_format': storage_format})
        
        # Create metadata
        metadata = DepthMapMetadata(
            depth_map_id=f"test_{hash(depth_map.tobytes()) % 10000:04d}",
            original_image_path="/test/images/test.jpg",
            image_filename="test.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=depth_map.shape,
            depth_range=(float(depth_map.min()), float(depth_map.max())),
            quality_score=0.8,
            storage_format=storage_format
        )
        
        # Store depth map
        depth_map_id = storage.store_depth_map(depth_map, metadata)
        
        # Retrieve depth map
        retrieved_depth, retrieved_metadata = storage.retrieve_depth_map(depth_map_id)
        
        # Verify data preservation based on format
        if storage_format == "npy":
            # NumPy format should preserve exact values
            np.testing.assert_allclose(depth_map, retrieved_depth, rtol=1e-6, atol=1e-6)
        elif storage_format == "gzip_pickle":
            # Pickle format should preserve exact values
            np.testing.assert_allclose(depth_map, retrieved_depth, rtol=1e-6, atol=1e-6)
        elif storage_format == "compressed_npy":
            # Compressed format should preserve values with minimal loss
            np.testing.assert_allclose(depth_map, retrieved_depth, rtol=1e-5, atol=1e-5)
        
        # Verify metadata preservation
        assert retrieved_metadata.depth_map_id == depth_map_id
        assert retrieved_metadata.image_filename == metadata.image_filename
        assert retrieved_metadata.midas_model_version == metadata.midas_model_version
        assert retrieved_metadata.storage_format == storage_format
        
        # Verify shape and range preservation
        assert retrieved_depth.shape == depth_map.shape
        assert retrieved_depth.dtype == np.float32
        assert not np.any(np.isnan(retrieved_depth))
        assert not np.any(np.isinf(retrieved_depth))
    
    @given(st.lists(depth_map_strategy(), min_size=1, max_size=5))
    @settings(max_examples=3, deadline=None)
    def test_batch_operations_consistency(self, depth_maps):
        """
        Property: For any list of depth maps, batch operations should be 
        equivalent to individual operations.
        
        Validates: Requirements 8.1, 8.2 (efficient storage and retrieval)
        """
        storage = DepthMapStorage(str(self.temp_dir))
        
        # Prepare batch data
        batch_data = []
        individual_ids = []
        
        for i, depth_map in enumerate(depth_maps):
            metadata = DepthMapMetadata(
                depth_map_id=f"batch_test_{i:03d}",
                original_image_path=f"/test/images/batch_{i:03d}.jpg",
                image_filename=f"batch_{i:03d}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=depth_map.shape,
                depth_range=(float(depth_map.min()), float(depth_map.max())),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            batch_data.append((depth_map, metadata))
        
        # Test batch store
        batch_ids = storage.batch_store(batch_data)
        assert len(batch_ids) == len(depth_maps)
        
        # Test batch retrieve
        batch_results = storage.batch_retrieve(batch_ids)
        assert len(batch_results) == len(depth_maps)
        
        # Verify each result matches original
        for i, (retrieved_depth, retrieved_metadata) in enumerate(batch_results):
            original_depth = depth_maps[i]
            np.testing.assert_allclose(retrieved_depth, original_depth, rtol=1e-5, atol=1e-5)
            assert retrieved_metadata.image_filename == f"batch_{i:03d}.jpg"
    
    @given(depth_map_strategy())
    @settings(max_examples=3, deadline=None)
    def test_compression_efficiency(self, depth_map):
        """
        Property: For any depth map, compressed storage should achieve 
        reasonable compression ratios.
        
        Validates: Requirements 8.1, 8.2 (efficient storage)
        """
        storage = DepthMapStorage(str(self.temp_dir), {'storage_format': 'compressed_npy'})
        
        metadata = DepthMapMetadata(
            depth_map_id=f"compression_test_{hash(depth_map.tobytes()) % 10000:04d}",
            original_image_path="/test/images/compression_test.jpg",
            image_filename="compression_test.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=depth_map.shape,
            depth_range=(float(depth_map.min()), float(depth_map.max())),
            quality_score=0.8,
            storage_format="compressed_npy"
        )
        
        # Store depth map
        depth_map_id = storage.store_depth_map(depth_map, metadata)
        
        # Retrieve metadata to check compression
        _, stored_metadata = storage.retrieve_depth_map(depth_map_id)
        
        # Verify compression ratio is reasonable
        assert stored_metadata.compression_ratio is not None
        assert stored_metadata.compression_ratio > 1.0  # Should achieve some compression
        assert stored_metadata.file_size_bytes is not None
        assert stored_metadata.file_size_bytes > 0
        
        # Original size should be larger than compressed size
        original_size = depth_map.nbytes
        compressed_size = stored_metadata.file_size_bytes
        assert original_size > compressed_size
    
    @given(st.lists(metadata_strategy(), min_size=2, max_size=10))
    @settings(max_examples=3, deadline=None)
    def test_search_operations_correctness(self, metadatas):
        """
        Property: For any set of metadata, search operations should return 
        correct and complete results.
        
        Validates: Requirements 8.2, 10.2 (metadata tracking and searchability)
        """
        storage = DepthMapStorage(str(self.temp_dir))
        
        # Store depth maps with given metadata
        stored_ids = []
        test_depth_map = np.random.rand(50, 75).astype(np.float32) * 50 + 1
        
        for i, metadata in enumerate(metadatas):
            # Ensure unique IDs
            metadata.depth_map_id = f"search_test_{i:03d}"
            depth_map_id = storage.store_depth_map(test_depth_map, metadata)
            stored_ids.append(depth_map_id)
        
        # Test search by image filename
        unique_filenames = list(set(meta.image_filename for meta in metadatas))
        for filename in unique_filenames:
            found_metadata = storage.find_by_image(filename)
            expected_count = sum(1 for meta in metadatas if meta.image_filename == filename)
            assert len(found_metadata) == expected_count
            assert all(meta.image_filename == filename for meta in found_metadata)
        
        # Test search by model version
        unique_models = list(set(meta.midas_model_version for meta in metadatas))
        for model_version in unique_models:
            found_metadata = storage.find_by_model_version(model_version)
            expected_count = sum(1 for meta in metadatas if meta.midas_model_version == model_version)
            assert len(found_metadata) == expected_count
            assert all(meta.midas_model_version == model_version for meta in found_metadata)
        
        # Test search by quality range
        all_qualities = [meta.quality_score for meta in metadatas]
        if all_qualities:
            min_quality = min(all_qualities)
            max_quality = max(all_qualities)
            
            found_metadata = storage.find_by_quality_range(min_quality, max_quality)
            assert len(found_metadata) == len(metadatas)
            
            # Test partial range
            mid_quality = (min_quality + max_quality) / 2
            high_quality_metadata = storage.find_by_quality_range(mid_quality, max_quality)
            expected_high_count = sum(1 for q in all_qualities if q >= mid_quality)
            assert len(high_quality_metadata) == expected_high_count
    
    @given(depth_map_strategy())
    @settings(max_examples=3, deadline=None)
    def test_checksum_integrity_validation(self, depth_map):
        """
        Property: For any depth map, checksum validation should detect 
        data corruption and preserve integrity.
        
        Validates: Requirements 8.1, 8.2 (data integrity and validation)
        """
        storage = DepthMapStorage(str(self.temp_dir), {'enable_checksums': True})
        
        metadata = DepthMapMetadata(
            depth_map_id=f"checksum_test_{hash(depth_map.tobytes()) % 10000:04d}",
            original_image_path="/test/images/checksum_test.jpg",
            image_filename="checksum_test.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=depth_map.shape,
            depth_range=(float(depth_map.min()), float(depth_map.max())),
            quality_score=0.8,
            storage_format="compressed_npy"
        )
        
        # Store depth map
        depth_map_id = storage.store_depth_map(depth_map, metadata)
        
        # Retrieve and verify checksum was calculated
        retrieved_depth, retrieved_metadata = storage.retrieve_depth_map(depth_map_id)
        
        assert retrieved_metadata.checksum is not None
        assert len(retrieved_metadata.checksum) == 32  # MD5 hash length
        
        # Verify data integrity
        np.testing.assert_allclose(retrieved_depth, depth_map, rtol=1e-5, atol=1e-5)
        
        # Verify checksum consistency
        expected_checksum = storage._calculate_checksum(depth_map)
        assert retrieved_metadata.checksum == expected_checksum
    
    @given(st.lists(depth_map_strategy(), min_size=3, max_size=8))
    @settings(max_examples=2, deadline=None)
    def test_cache_behavior_consistency(self, depth_maps):
        """
        Property: For any sequence of depth maps, cache behavior should not 
        affect data retrieval correctness.
        
        Validates: Requirements 8.1, 8.2 (efficient retrieval with caching)
        """
        # Create storage with small cache to force evictions
        cache_size = max(2, len(depth_maps) // 2)
        storage = DepthMapStorage(str(self.temp_dir), {'max_cache_size': cache_size})
        
        # Store all depth maps
        stored_data = []
        for i, depth_map in enumerate(depth_maps):
            metadata = DepthMapMetadata(
                depth_map_id=f"cache_test_{i:03d}",
                original_image_path=f"/test/images/cache_{i:03d}.jpg",
                image_filename=f"cache_{i:03d}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=depth_map.shape,
                depth_range=(float(depth_map.min()), float(depth_map.max())),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            depth_map_id = storage.store_depth_map(depth_map, metadata)
            stored_data.append((depth_map_id, depth_map))
        
        # Retrieve all depth maps multiple times to test cache behavior
        for _ in range(2):
            for depth_map_id, original_depth in stored_data:
                retrieved_depth, _ = storage.retrieve_depth_map(depth_map_id)
                np.testing.assert_allclose(retrieved_depth, original_depth, rtol=1e-5, atol=1e-5)
        
        # Verify cache size constraint is respected
        assert len(storage._cache) <= cache_size
    
    @given(depth_map_strategy())
    @settings(max_examples=3, deadline=None)
    def test_versioning_and_cleanup_behavior(self, depth_map):
        """
        Property: For any depth map, version management and cleanup should 
        preserve data integrity while managing storage efficiently.
        
        Validates: Requirements 8.2, 10.2 (versioning and storage management)
        """
        storage = DepthMapStorage(str(self.temp_dir))
        
        image_filename = "version_test.jpg"
        stored_ids = []
        
        # Store multiple versions of same image
        for i in range(4):
            metadata = DepthMapMetadata(
                depth_map_id=f"version_test_{i:03d}",
                original_image_path=f"/test/images/{image_filename}",
                image_filename=image_filename,
                midas_model_version=f"Intel/dpt-large-v{i}",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=depth_map.shape,
                depth_range=(float(depth_map.min()), float(depth_map.max())),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            depth_map_id = storage.store_depth_map(depth_map, metadata)
            stored_ids.append(depth_map_id)
        
        # Verify all versions exist
        found_versions = storage.find_by_image(image_filename)
        assert len(found_versions) == 4
        
        # Cleanup old versions, keeping latest 2
        deleted_count = storage.cleanup_old_versions(keep_latest_n=2)
        assert deleted_count == 2
        
        # Verify only 2 versions remain
        remaining_versions = storage.find_by_image(image_filename)
        assert len(remaining_versions) == 2
        
        # Verify remaining versions are still accessible
        for metadata in remaining_versions:
            retrieved_depth, _ = storage.retrieve_depth_map(metadata.depth_map_id)
            np.testing.assert_allclose(retrieved_depth, depth_map, rtol=1e-5, atol=1e-5)


class TestDepthMapVisualizerProperties:
    """Property-based tests for DepthMapVisualizer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = DepthMapStorage(str(self.temp_dir))
        self.visualizer = DepthMapVisualizer(self.storage)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(depth_map_strategy())
    @settings(max_examples=3, deadline=None)
    def test_visualization_generation_robustness(self, depth_map):
        """
        Property: For any valid depth map, visualization generation should 
        succeed and produce valid output files.
        
        Validates: Requirements 10.2 (debugging and visualization utilities)
        """
        # Store depth map
        metadata = DepthMapMetadata(
            depth_map_id=f"viz_test_{hash(depth_map.tobytes()) % 10000:04d}",
            original_image_path="/test/images/viz_test.jpg",
            image_filename="viz_test.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=depth_map.shape,
            depth_range=(float(depth_map.min()), float(depth_map.max())),
            quality_score=0.8,
            storage_format="compressed_npy"
        )
        depth_map_id = self.storage.store_depth_map(depth_map, metadata)
        
        # Create visualization
        output_path = self.temp_dir / f"viz_{depth_map_id}.png"
        self.visualizer.create_depth_visualization(depth_map_id, str(output_path))
        
        # Verify output file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify file is a valid image (basic check)
        try:
            import cv2
            img = cv2.imread(str(output_path))
            assert img is not None
            assert len(img.shape) == 3  # Should be color image
            assert img.shape[2] == 3   # Should have 3 channels (BGR)
        except ImportError:
            # If OpenCV not available, just check file exists and has content
            pass


if __name__ == "__main__":
    pytest.main([__file__])
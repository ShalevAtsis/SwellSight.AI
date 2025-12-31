"""Unit tests for depth map storage and retrieval system."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sqlite3
import json

from swellsight.data.depth_map_storage import (
    DepthMapStorage, 
    DepthMapMetadata, 
    DepthMapVisualizer
)


class TestDepthMapStorage:
    """Test cases for DepthMapStorage class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = DepthMapStorage(str(self.temp_dir))
        
        # Create test depth map
        self.test_depth_map = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        
        # Create test metadata
        self.test_metadata = DepthMapMetadata(
            depth_map_id="test_depth_001",
            original_image_path="/test/images/beach_001.jpg",
            image_filename="beach_001.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=self.test_depth_map.shape,
            depth_range=(float(self.test_depth_map.min()), float(self.test_depth_map.max())),
            quality_score=0.85,
            storage_format="compressed_npy"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_storage_initialization(self):
        """Test storage system initialization."""
        # Check directory structure
        assert (self.temp_dir / 'depth_maps').exists()
        assert (self.temp_dir / 'metadata').exists()
        assert (self.temp_dir / 'versions').exists()
        
        # Check database initialization
        db_path = self.temp_dir / 'metadata' / 'depth_maps.db'
        assert db_path.exists()
        
        # Verify database schema
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert 'depth_maps' in tables
    
    def test_store_depth_map_compressed_npy(self):
        """Test storing depth map in compressed numpy format."""
        # Store depth map
        depth_map_id = self.storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Verify ID was generated
        assert depth_map_id == self.test_metadata.depth_map_id
        
        # Check file exists
        depth_file = self.temp_dir / 'depth_maps' / f'{depth_map_id}.npz'
        assert depth_file.exists()
        
        # Verify metadata in database
        metadata = self.storage._get_metadata(depth_map_id)
        assert metadata.image_filename == "beach_001.jpg"
        assert metadata.midas_model_version == "Intel/dpt-large"
        assert metadata.quality_score == 0.85
    
    def test_store_depth_map_gzip_pickle(self):
        """Test storing depth map in gzip+pickle format."""
        # Configure for gzip+pickle format
        storage = DepthMapStorage(str(self.temp_dir), {'storage_format': 'gzip_pickle'})
        self.test_metadata.storage_format = 'gzip_pickle'
        
        # Store depth map
        depth_map_id = storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Check file exists with correct extension
        depth_file = self.temp_dir / 'depth_maps' / f'{depth_map_id}.pkl.gz'
        assert depth_file.exists()
    
    def test_retrieve_depth_map(self):
        """Test retrieving stored depth map."""
        # Store depth map first
        depth_map_id = self.storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Retrieve depth map
        retrieved_depth, retrieved_metadata = self.storage.retrieve_depth_map(depth_map_id)
        
        # Verify depth map data
        np.testing.assert_allclose(retrieved_depth, self.test_depth_map, rtol=1e-6)
        assert retrieved_depth.dtype == np.float32
        assert retrieved_depth.shape == self.test_depth_map.shape
        
        # Verify metadata
        assert retrieved_metadata.depth_map_id == depth_map_id
        assert retrieved_metadata.image_filename == "beach_001.jpg"
        assert retrieved_metadata.quality_score == 0.85
    
    def test_batch_operations(self):
        """Test batch store and retrieve operations."""
        # Create multiple test depth maps
        depth_maps = []
        metadatas = []
        
        for i in range(3):
            depth_map = np.random.rand(80, 120).astype(np.float32) * 30 + 5
            metadata = DepthMapMetadata(
                depth_map_id=f"batch_test_{i:03d}",
                original_image_path=f"/test/images/batch_{i:03d}.jpg",
                image_filename=f"batch_{i:03d}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=depth_map.shape,
                depth_range=(float(depth_map.min()), float(depth_map.max())),
                quality_score=0.7 + i * 0.1,
                storage_format="compressed_npy"
            )
            depth_maps.append((depth_map, metadata))
            metadatas.append(metadata)
        
        # Batch store
        stored_ids = self.storage.batch_store(depth_maps)
        assert len(stored_ids) == 3
        
        # Batch retrieve
        retrieved_results = self.storage.batch_retrieve(stored_ids)
        assert len(retrieved_results) == 3
        
        # Verify each result
        for i, (retrieved_depth, retrieved_metadata) in enumerate(retrieved_results):
            original_depth = depth_maps[i][0]
            np.testing.assert_allclose(retrieved_depth, original_depth, rtol=1e-6)
            assert retrieved_metadata.image_filename == f"batch_{i:03d}.jpg"
    
    def test_find_by_image(self):
        """Test finding depth maps by image filename."""
        # Store multiple versions of same image
        image_filename = "test_beach.jpg"
        
        for i in range(2):
            metadata = DepthMapMetadata(
                depth_map_id=f"version_{i}",
                original_image_path=f"/test/images/{image_filename}",
                image_filename=image_filename,
                midas_model_version=f"Intel/dpt-large-v{i}",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=0.8 + i * 0.1,
                storage_format="compressed_npy"
            )
            self.storage.store_depth_map(self.test_depth_map, metadata)
        
        # Find by image filename
        found_metadata = self.storage.find_by_image(image_filename)
        assert len(found_metadata) == 2
        
        # Verify results are sorted by extraction timestamp (newest first)
        assert all(meta.image_filename == image_filename for meta in found_metadata)
    
    def test_find_by_model_version(self):
        """Test finding depth maps by MiDaS model version."""
        model_version = "Intel/dpt-hybrid-midas"
        
        # Store depth maps with specific model version
        for i in range(2):
            metadata = DepthMapMetadata(
                depth_map_id=f"model_test_{i}",
                original_image_path=f"/test/images/model_test_{i}.jpg",
                image_filename=f"model_test_{i}.jpg",
                midas_model_version=model_version,
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=0.75,
                storage_format="compressed_npy"
            )
            self.storage.store_depth_map(self.test_depth_map, metadata)
        
        # Find by model version
        found_metadata = self.storage.find_by_model_version(model_version)
        assert len(found_metadata) == 2
        assert all(meta.midas_model_version == model_version for meta in found_metadata)
    
    def test_find_by_quality_range(self):
        """Test finding depth maps by quality score range."""
        # Store depth maps with different quality scores
        quality_scores = [0.3, 0.6, 0.8, 0.95]
        
        for i, quality in enumerate(quality_scores):
            metadata = DepthMapMetadata(
                depth_map_id=f"quality_test_{i}",
                original_image_path=f"/test/images/quality_test_{i}.jpg",
                image_filename=f"quality_test_{i}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=quality,
                storage_format="compressed_npy"
            )
            self.storage.store_depth_map(self.test_depth_map, metadata)
        
        # Find high quality depth maps (>= 0.7)
        high_quality = self.storage.find_by_quality_range(0.7)
        assert len(high_quality) == 2
        assert all(meta.quality_score >= 0.7 for meta in high_quality)
        
        # Find medium quality depth maps (0.5 - 0.7)
        medium_quality = self.storage.find_by_quality_range(0.5, 0.7)
        assert len(medium_quality) == 1
        assert medium_quality[0].quality_score == 0.6
    
    def test_delete_depth_map(self):
        """Test deleting depth map and metadata."""
        # Store depth map
        depth_map_id = self.storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Verify it exists
        depth_file = self.temp_dir / 'depth_maps' / f'{depth_map_id}.npz'
        assert depth_file.exists()
        
        # Delete depth map
        success = self.storage.delete_depth_map(depth_map_id)
        assert success
        
        # Verify file and metadata are deleted
        assert not depth_file.exists()
        
        with pytest.raises(ValueError, match="Metadata not found"):
            self.storage._get_metadata(depth_map_id)
    
    def test_storage_statistics(self):
        """Test storage system statistics."""
        # Store some test data
        for i in range(3):
            metadata = DepthMapMetadata(
                depth_map_id=f"stats_test_{i}",
                original_image_path=f"/test/images/stats_test_{i}.jpg",
                image_filename=f"stats_test_{i}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            self.storage.store_depth_map(self.test_depth_map, metadata)
        
        # Get statistics
        stats = self.storage.get_storage_statistics()
        
        # Verify statistics structure
        assert 'total_depth_maps' in stats
        assert 'storage_path' in stats
        assert 'model_statistics' in stats
        assert 'quality_distribution' in stats
        assert 'storage_efficiency' in stats
        
        # Verify values
        assert stats['total_depth_maps'] == 3
        assert len(stats['model_statistics']) == 1
        assert stats['model_statistics'][0]['model_version'] == "Intel/dpt-large"
        assert stats['model_statistics'][0]['count'] == 3
    
    def test_cleanup_old_versions(self):
        """Test cleanup of old depth map versions."""
        image_filename = "cleanup_test.jpg"
        
        # Store 5 versions of same image
        for i in range(5):
            metadata = DepthMapMetadata(
                depth_map_id=f"cleanup_test_{i}",
                original_image_path=f"/test/images/{image_filename}",
                image_filename=image_filename,
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            self.storage.store_depth_map(self.test_depth_map, metadata)
        
        # Cleanup, keeping only latest 2
        deleted_count = self.storage.cleanup_old_versions(keep_latest_n=2)
        assert deleted_count == 3
        
        # Verify only 2 remain
        remaining = self.storage.find_by_image(image_filename)
        assert len(remaining) == 2
    
    def test_export_metadata(self):
        """Test metadata export functionality."""
        # Store test data
        depth_map_id = self.storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Export metadata
        export_path = self.temp_dir / "exported_metadata.json"
        self.storage.export_metadata(str(export_path))
        
        # Verify export file exists
        assert export_path.exists()
        
        # Verify export content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 1
        assert exported_data[0]['depth_map_id'] == depth_map_id
        assert exported_data[0]['image_filename'] == "beach_001.jpg"
    
    def test_cache_functionality(self):
        """Test in-memory cache functionality."""
        # Configure small cache size
        storage = DepthMapStorage(str(self.temp_dir), {'max_cache_size': 2})
        
        # Store multiple depth maps
        depth_map_ids = []
        for i in range(3):
            metadata = DepthMapMetadata(
                depth_map_id=f"cache_test_{i}",
                original_image_path=f"/test/images/cache_test_{i}.jpg",
                image_filename=f"cache_test_{i}.jpg",
                midas_model_version="Intel/dpt-large",
                extraction_timestamp=datetime.now().isoformat(),
                depth_map_shape=self.test_depth_map.shape,
                depth_range=(1.0, 50.0),
                quality_score=0.8,
                storage_format="compressed_npy"
            )
            depth_map_id = storage.store_depth_map(self.test_depth_map, metadata)
            depth_map_ids.append(depth_map_id)
        
        # Verify cache size limit
        assert len(storage._cache) <= 2
        
        # Access first depth map (should load from disk)
        retrieved_depth, _ = storage.retrieve_depth_map(depth_map_ids[0])
        np.testing.assert_allclose(retrieved_depth, self.test_depth_map, rtol=1e-6)
    
    def test_checksum_validation(self):
        """Test checksum validation functionality."""
        # Store depth map with checksums enabled
        storage = DepthMapStorage(str(self.temp_dir), {'enable_checksums': True})
        
        depth_map_id = storage.store_depth_map(self.test_depth_map, self.test_metadata)
        
        # Retrieve and verify checksum validation works
        retrieved_depth, metadata = storage.retrieve_depth_map(depth_map_id)
        
        # Verify checksum was calculated and stored
        assert metadata.checksum is not None
        assert len(metadata.checksum) == 32  # MD5 hash length
        
        # Verify data integrity
        np.testing.assert_allclose(retrieved_depth, self.test_depth_map, rtol=1e-6)


class TestDepthMapVisualizer:
    """Test cases for DepthMapVisualizer class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = DepthMapStorage(str(self.temp_dir))
        self.visualizer = DepthMapVisualizer(self.storage)
        
        # Create and store test depth map
        self.test_depth_map = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        self.test_metadata = DepthMapMetadata(
            depth_map_id="viz_test_001",
            original_image_path="/test/images/viz_test.jpg",
            image_filename="viz_test.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=self.test_depth_map.shape,
            depth_range=(float(self.test_depth_map.min()), float(self.test_depth_map.max())),
            quality_score=0.85,
            storage_format="compressed_npy"
        )
        self.depth_map_id = self.storage.store_depth_map(self.test_depth_map, self.test_metadata)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_depth_visualization(self):
        """Test creating depth map visualization."""
        output_path = self.temp_dir / "visualization.png"
        
        # Create visualization
        self.visualizer.create_depth_visualization(self.depth_map_id, str(output_path))
        
        # Verify output file exists
        assert output_path.exists()
        
        # Verify file is not empty
        assert output_path.stat().st_size > 0
    
    def test_create_comparison_visualization(self):
        """Test creating comparison visualization."""
        # Store another depth map for comparison
        test_metadata_2 = DepthMapMetadata(
            depth_map_id="viz_test_002",
            original_image_path="/test/images/viz_test_2.jpg",
            image_filename="viz_test_2.jpg",
            midas_model_version="Intel/dpt-hybrid-midas",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=self.test_depth_map.shape,
            depth_range=(float(self.test_depth_map.min()), float(self.test_depth_map.max())),
            quality_score=0.75,
            storage_format="compressed_npy"
        )
        depth_map_id_2 = self.storage.store_depth_map(self.test_depth_map, test_metadata_2)
        
        output_path = self.temp_dir / "comparison.png"
        
        # Create comparison visualization
        self.visualizer.create_comparison_visualization(
            [self.depth_map_id, depth_map_id_2], 
            str(output_path)
        )
        
        # Verify output file exists
        assert output_path.exists()
        
        # Verify file is not empty
        assert output_path.stat().st_size > 0


class TestDepthMapMetadata:
    """Test cases for DepthMapMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating DepthMapMetadata instance."""
        metadata = DepthMapMetadata(
            depth_map_id="test_001",
            original_image_path="/test/image.jpg",
            image_filename="image.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp="2024-01-01T12:00:00",
            depth_map_shape=(100, 150),
            depth_range=(1.0, 50.0),
            quality_score=0.85,
            storage_format="compressed_npy"
        )
        
        assert metadata.depth_map_id == "test_001"
        assert metadata.image_filename == "image.jpg"
        assert metadata.quality_score == 0.85
        assert metadata.depth_map_shape == (100, 150)
    
    def test_metadata_serialization(self):
        """Test metadata serialization to dictionary."""
        from dataclasses import asdict
        
        metadata = DepthMapMetadata(
            depth_map_id="test_001",
            original_image_path="/test/image.jpg",
            image_filename="image.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp="2024-01-01T12:00:00",
            depth_map_shape=(100, 150),
            depth_range=(1.0, 50.0),
            quality_score=0.85,
            storage_format="compressed_npy",
            compression_ratio=3.2,
            file_size_bytes=1024,
            checksum="abc123"
        )
        
        metadata_dict = asdict(metadata)
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['depth_map_id'] == "test_001"
        assert metadata_dict['compression_ratio'] == 3.2
        assert metadata_dict['checksum'] == "abc123"


if __name__ == "__main__":
    pytest.main([__file__])
"""Depth map storage and retrieval system for MiDaS depth maps."""

import json
import numpy as np
import gzip
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import cv2
from dataclasses import dataclass, asdict
import sqlite3

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DepthMapMetadata:
    """Metadata for stored depth maps."""
    depth_map_id: str
    original_image_path: str
    image_filename: str
    midas_model_version: str
    extraction_timestamp: str
    depth_map_shape: Tuple[int, int]
    depth_range: Tuple[float, float]
    quality_score: float
    storage_format: str
    compression_ratio: Optional[float] = None
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None


class DepthMapStorage:
    """
    Efficient storage and retrieval system for MiDaS depth maps.
    
    Features:
    - Compressed numpy array storage
    - Metadata tracking with SQLite database
    - Version control for different MiDaS models
    - Integrity validation with checksums
    - Efficient batch operations
    """
    
    def __init__(self, storage_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize depth map storage system.
        
        Args:
            storage_path: Base path for depth map storage
            config: Optional configuration dictionary
        """
        self.storage_path = Path(storage_path)
        self.config = config or {}
        
        # Configuration
        self.compression_level = self.config.get('compression_level', 6)
        self.storage_format = self.config.get('storage_format', 'compressed_npy')
        self.enable_checksums = self.config.get('enable_checksums', True)
        self.max_cache_size = self.config.get('max_cache_size', 100)
        
        # Create directory structure
        self.depth_maps_path = self.storage_path / 'depth_maps'
        self.metadata_path = self.storage_path / 'metadata'
        self.versions_path = self.storage_path / 'versions'
        
        for path in [self.depth_maps_path, self.metadata_path, self.versions_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.metadata_path / 'depth_maps.db'
        self._init_database()
        
        # In-memory cache
        self._cache = {}
        self._cache_order = []
        
        logger.info(f"Initialized DepthMapStorage at: {self.storage_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS depth_maps (
                    depth_map_id TEXT PRIMARY KEY,
                    original_image_path TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    midas_model_version TEXT NOT NULL,
                    extraction_timestamp TEXT NOT NULL,
                    depth_map_shape_height INTEGER NOT NULL,
                    depth_map_shape_width INTEGER NOT NULL,
                    depth_range_min REAL NOT NULL,
                    depth_range_max REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    storage_format TEXT NOT NULL,
                    compression_ratio REAL,
                    file_size_bytes INTEGER,
                    checksum TEXT,
                    created_timestamp TEXT NOT NULL,
                    updated_timestamp TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_filename 
                ON depth_maps(image_filename)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_midas_model 
                ON depth_maps(midas_model_version)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_quality_score 
                ON depth_maps(quality_score)
            ''')
            
            conn.commit()
    
    def store_depth_map(self, depth_map: np.ndarray, metadata: DepthMapMetadata) -> str:
        """
        Store depth map with metadata.
        
        Args:
            depth_map: Depth map array to store
            metadata: Metadata for the depth map
            
        Returns:
            Depth map ID
        """
        try:
            # Generate unique ID if not provided
            if not metadata.depth_map_id:
                metadata.depth_map_id = self._generate_depth_map_id(
                    metadata.image_filename, 
                    metadata.midas_model_version
                )
            
            # Store depth map file
            depth_file_path = self._get_depth_file_path(metadata.depth_map_id)
            original_size = depth_map.nbytes
            
            if self.storage_format == 'compressed_npy':
                # Compress and save as .npz
                depth_file_path = depth_file_path.with_suffix('.npz')
                np.savez_compressed(depth_file_path, depth_map=depth_map)
            elif self.storage_format == 'gzip_pickle':
                # Use gzip + pickle for maximum compression
                depth_file_path = depth_file_path.with_suffix('.pkl.gz')
                with gzip.open(depth_file_path, 'wb', compresslevel=self.compression_level) as f:
                    pickle.dump(depth_map, f)
            elif self.storage_format == 'npy':
                # Uncompressed numpy
                depth_file_path = depth_file_path.with_suffix('.npy')
                np.save(depth_file_path, depth_map)
            else:
                raise ValueError(f"Unsupported storage format: {self.storage_format}")
            
            # Calculate compression ratio and file size
            if depth_file_path.exists():
                compressed_size = depth_file_path.stat().st_size
                metadata.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                metadata.file_size_bytes = compressed_size
            else:
                # Fallback if file doesn't exist
                metadata.compression_ratio = 1.0
                metadata.file_size_bytes = original_size
            
            # Calculate checksum if enabled
            if self.enable_checksums:
                metadata.checksum = self._calculate_checksum(depth_map)
            
            # Store metadata in database
            self._store_metadata(metadata)
            
            # Add to cache
            self._add_to_cache(metadata.depth_map_id, depth_map)
            
            logger.debug(f"Stored depth map: {metadata.depth_map_id}")
            return metadata.depth_map_id
            
        except Exception as e:
            logger.error(f"Error storing depth map {metadata.depth_map_id}: {e}")
            raise
    
    def retrieve_depth_map(self, depth_map_id: str) -> Tuple[np.ndarray, DepthMapMetadata]:
        """
        Retrieve depth map and metadata by ID.
        
        Args:
            depth_map_id: Unique depth map identifier
            
        Returns:
            Tuple of (depth_map, metadata)
        """
        try:
            # Check cache first
            if depth_map_id in self._cache:
                depth_map = self._cache[depth_map_id]
                metadata = self._get_metadata(depth_map_id)
                self._update_cache_order(depth_map_id)
                return depth_map, metadata
            
            # Load from storage
            depth_file_path = self._get_depth_file_path(depth_map_id)
            
            if not depth_file_path.exists():
                # Try alternative extensions
                for ext in ['.npz', '.pkl.gz', '.npy']:
                    alt_path = depth_file_path.with_suffix(ext)
                    if alt_path.exists():
                        depth_file_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Depth map file not found: {depth_map_id}")
            
            # Load based on file extension
            if depth_file_path.suffix == '.npz':
                with np.load(depth_file_path) as data:
                    depth_map = data['depth_map']
            elif depth_file_path.suffix == '.gz':
                with gzip.open(depth_file_path, 'rb') as f:
                    depth_map = pickle.load(f)
            elif depth_file_path.suffix == '.npy':
                depth_map = np.load(depth_file_path)
            else:
                raise ValueError(f"Unsupported file format: {depth_file_path.suffix}")
            
            # Get metadata
            metadata = self._get_metadata(depth_map_id)
            
            # Validate checksum if enabled
            if self.enable_checksums and metadata.checksum:
                calculated_checksum = self._calculate_checksum(depth_map)
                if calculated_checksum != metadata.checksum:
                    logger.warning(f"Checksum mismatch for {depth_map_id}")
            
            # Add to cache
            self._add_to_cache(depth_map_id, depth_map)
            
            logger.debug(f"Retrieved depth map: {depth_map_id}")
            return depth_map, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving depth map {depth_map_id}: {e}")
            raise
    
    def batch_store(self, depth_maps: List[Tuple[np.ndarray, DepthMapMetadata]]) -> List[str]:
        """
        Store multiple depth maps in batch.
        
        Args:
            depth_maps: List of (depth_map, metadata) tuples
            
        Returns:
            List of depth map IDs
        """
        stored_ids = []
        
        for depth_map, metadata in depth_maps:
            try:
                depth_map_id = self.store_depth_map(depth_map, metadata)
                stored_ids.append(depth_map_id)
            except Exception as e:
                logger.error(f"Error in batch store for {metadata.image_filename}: {e}")
                continue
        
        logger.info(f"Batch stored {len(stored_ids)}/{len(depth_maps)} depth maps")
        return stored_ids
    
    def batch_retrieve(self, depth_map_ids: List[str]) -> List[Tuple[np.ndarray, DepthMapMetadata]]:
        """
        Retrieve multiple depth maps in batch.
        
        Args:
            depth_map_ids: List of depth map IDs
            
        Returns:
            List of (depth_map, metadata) tuples
        """
        results = []
        
        for depth_map_id in depth_map_ids:
            try:
                depth_map, metadata = self.retrieve_depth_map(depth_map_id)
                results.append((depth_map, metadata))
            except Exception as e:
                logger.error(f"Error in batch retrieve for {depth_map_id}: {e}")
                continue
        
        logger.info(f"Batch retrieved {len(results)}/{len(depth_map_ids)} depth maps")
        return results
    
    def find_by_image(self, image_filename: str) -> List[DepthMapMetadata]:
        """
        Find depth maps by image filename.
        
        Args:
            image_filename: Name of the original image file
            
        Returns:
            List of matching metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM depth_maps WHERE image_filename = ? ORDER BY extraction_timestamp DESC',
                (image_filename,)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_metadata(row) for row in rows]
    
    def find_by_model_version(self, model_version: str) -> List[DepthMapMetadata]:
        """
        Find depth maps by MiDaS model version.
        
        Args:
            model_version: MiDaS model version string
            
        Returns:
            List of matching metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM depth_maps WHERE midas_model_version = ? ORDER BY extraction_timestamp DESC',
                (model_version,)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_metadata(row) for row in rows]
    
    def find_by_quality_range(self, min_quality: float, max_quality: float = 1.0) -> List[DepthMapMetadata]:
        """
        Find depth maps by quality score range.
        
        Args:
            min_quality: Minimum quality score
            max_quality: Maximum quality score
            
        Returns:
            List of matching metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM depth_maps WHERE quality_score BETWEEN ? AND ? ORDER BY quality_score DESC',
                (min_quality, max_quality)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_metadata(row) for row in rows]
    
    def delete_depth_map(self, depth_map_id: str) -> bool:
        """
        Delete depth map and metadata.
        
        Args:
            depth_map_id: Depth map ID to delete
            
        Returns:
            True if successfully deleted
        """
        try:
            # Remove from cache
            if depth_map_id in self._cache:
                del self._cache[depth_map_id]
                self._cache_order.remove(depth_map_id)
            
            # Delete file
            depth_file_path = self._get_depth_file_path(depth_map_id)
            for ext in ['.npz', '.pkl.gz', '.npy']:
                file_path = depth_file_path.with_suffix(ext)
                if file_path.exists():
                    file_path.unlink()
                    break
            
            # Delete metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM depth_maps WHERE depth_map_id = ?', (depth_map_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            
            logger.debug(f"Deleted depth map: {depth_map_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting depth map {depth_map_id}: {e}")
            return False
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage system statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total count
            cursor = conn.execute('SELECT COUNT(*) FROM depth_maps')
            total_count = cursor.fetchone()[0]
            
            # Storage by model version
            cursor = conn.execute('''
                SELECT midas_model_version, COUNT(*), AVG(quality_score), AVG(compression_ratio)
                FROM depth_maps 
                GROUP BY midas_model_version
            ''')
            model_stats = cursor.fetchall()
            
            # Quality distribution
            cursor = conn.execute('''
                SELECT 
                    COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as excellent,
                    COUNT(CASE WHEN quality_score >= 0.6 AND quality_score < 0.8 THEN 1 END) as good,
                    COUNT(CASE WHEN quality_score >= 0.4 AND quality_score < 0.6 THEN 1 END) as fair,
                    COUNT(CASE WHEN quality_score < 0.4 THEN 1 END) as poor
                FROM depth_maps
            ''')
            quality_dist = cursor.fetchone()
            
            # Storage efficiency
            cursor = conn.execute('SELECT AVG(compression_ratio), SUM(file_size_bytes) FROM depth_maps')
            efficiency = cursor.fetchone()
        
        stats = {
            'total_depth_maps': total_count,
            'storage_path': str(self.storage_path),
            'storage_format': self.storage_format,
            'model_statistics': [
                {
                    'model_version': row[0],
                    'count': row[1],
                    'avg_quality': row[2],
                    'avg_compression_ratio': row[3]
                }
                for row in model_stats
            ],
            'quality_distribution': {
                'excellent': quality_dist[0],
                'good': quality_dist[1],
                'fair': quality_dist[2],
                'poor': quality_dist[3]
            },
            'storage_efficiency': {
                'avg_compression_ratio': efficiency[0],
                'total_storage_bytes': efficiency[1]
            },
            'cache_statistics': {
                'cache_size': len(self._cache),
                'max_cache_size': self.max_cache_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def cleanup_old_versions(self, keep_latest_n: int = 3) -> int:
        """
        Clean up old depth map versions, keeping only the latest N per image.
        
        Args:
            keep_latest_n: Number of latest versions to keep per image
            
        Returns:
            Number of depth maps deleted
        """
        deleted_count = 0
        
        # Get all unique image filenames
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT image_filename FROM depth_maps')
            image_filenames = [row[0] for row in cursor.fetchall()]
        
        for image_filename in image_filenames:
            # Get all depth maps for this image, ordered by extraction time
            metadata_list = self.find_by_image(image_filename)
            
            # Delete older versions if more than keep_latest_n
            if len(metadata_list) > keep_latest_n:
                to_delete = metadata_list[keep_latest_n:]
                
                for metadata in to_delete:
                    if self.delete_depth_map(metadata.depth_map_id):
                        deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old depth map versions")
        return deleted_count
    
    def export_metadata(self, output_path: str) -> None:
        """
        Export all metadata to JSON file.
        
        Args:
            output_path: Path for output JSON file
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM depth_maps ORDER BY extraction_timestamp')
            rows = cursor.fetchall()
        
        metadata_list = [asdict(self._row_to_metadata(row)) for row in rows]
        
        with open(output_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Exported {len(metadata_list)} metadata records to {output_path}")
    
    def _generate_depth_map_id(self, image_filename: str, model_version: str) -> str:
        """Generate unique depth map ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = Path(image_filename).stem
        return f"{base_name}_{model_version.replace('/', '_')}_{timestamp}"
    
    def _get_depth_file_path(self, depth_map_id: str) -> Path:
        """Get file path for depth map storage."""
        return self.depth_maps_path / depth_map_id
    
    def _calculate_checksum(self, depth_map: np.ndarray) -> str:
        """Calculate MD5 checksum for depth map."""
        return hashlib.md5(depth_map.tobytes()).hexdigest()
    
    def _store_metadata(self, metadata: DepthMapMetadata) -> None:
        """Store metadata in database."""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO depth_maps (
                    depth_map_id, original_image_path, image_filename, midas_model_version,
                    extraction_timestamp, depth_map_shape_height, depth_map_shape_width,
                    depth_range_min, depth_range_max, quality_score, storage_format,
                    compression_ratio, file_size_bytes, checksum, created_timestamp, updated_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.depth_map_id, metadata.original_image_path, metadata.image_filename,
                metadata.midas_model_version, metadata.extraction_timestamp,
                metadata.depth_map_shape[0], metadata.depth_map_shape[1],
                metadata.depth_range[0], metadata.depth_range[1], metadata.quality_score,
                metadata.storage_format, metadata.compression_ratio, metadata.file_size_bytes,
                metadata.checksum, timestamp, timestamp
            ))
            conn.commit()
    
    def _get_metadata(self, depth_map_id: str) -> DepthMapMetadata:
        """Get metadata from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM depth_maps WHERE depth_map_id = ?', (depth_map_id,))
            row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Metadata not found for depth map: {depth_map_id}")
        
        return self._row_to_metadata(row)
    
    def _row_to_metadata(self, row: Tuple) -> DepthMapMetadata:
        """Convert database row to DepthMapMetadata."""
        return DepthMapMetadata(
            depth_map_id=row[0],
            original_image_path=row[1],
            image_filename=row[2],
            midas_model_version=row[3],
            extraction_timestamp=row[4],
            depth_map_shape=(row[5], row[6]),
            depth_range=(row[7], row[8]),
            quality_score=row[9],
            storage_format=row[10],
            compression_ratio=row[11],
            file_size_bytes=row[12],
            checksum=row[13]
        )
    
    def _add_to_cache(self, depth_map_id: str, depth_map: np.ndarray) -> None:
        """Add depth map to cache with LRU eviction."""
        # Remove if already exists
        if depth_map_id in self._cache:
            self._cache_order.remove(depth_map_id)
        
        # Add to cache
        self._cache[depth_map_id] = depth_map
        self._cache_order.append(depth_map_id)
        
        # Evict oldest if cache is full
        while len(self._cache) > self.max_cache_size:
            oldest_id = self._cache_order.pop(0)
            del self._cache[oldest_id]
    
    def _update_cache_order(self, depth_map_id: str) -> None:
        """Update cache order for LRU."""
        if depth_map_id in self._cache_order:
            self._cache_order.remove(depth_map_id)
            self._cache_order.append(depth_map_id)


class DepthMapVisualizer:
    """
    Utilities for depth map visualization and debugging.
    """
    
    def __init__(self, storage: DepthMapStorage):
        """
        Initialize visualizer with storage system.
        
        Args:
            storage: DepthMapStorage instance
        """
        self.storage = storage
    
    def create_depth_visualization(self, depth_map_id: str, output_path: str) -> None:
        """
        Create visualization of depth map.
        
        Args:
            depth_map_id: Depth map ID to visualize
            output_path: Output path for visualization image
        """
        try:
            depth_map, metadata = self.storage.retrieve_depth_map(depth_map_id)
            
            # Normalize depth map for visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Add metadata overlay
            height, width = depth_map.shape
            overlay = np.zeros((height + 100, width, 3), dtype=np.uint8)
            overlay[:height, :] = depth_colored
            
            # Add text information
            info_text = [
                f"ID: {metadata.depth_map_id}",
                f"Model: {metadata.midas_model_version}",
                f"Quality: {metadata.quality_score:.3f}",
                f"Range: {metadata.depth_range[0]:.1f}-{metadata.depth_range[1]:.1f}m"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(overlay, text, (10, height + 20 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save visualization
            cv2.imwrite(output_path, overlay)
            logger.info(f"Created depth visualization: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization for {depth_map_id}: {e}")
            raise
    
    def create_comparison_visualization(self, depth_map_ids: List[str], output_path: str) -> None:
        """
        Create side-by-side comparison of multiple depth maps.
        
        Args:
            depth_map_ids: List of depth map IDs to compare
            output_path: Output path for comparison image
        """
        try:
            depth_maps = []
            metadatas = []
            
            for depth_map_id in depth_map_ids:
                depth_map, metadata = self.storage.retrieve_depth_map(depth_map_id)
                depth_maps.append(depth_map)
                metadatas.append(metadata)
            
            # Create side-by-side visualization
            if not depth_maps:
                return
            
            # Resize all depth maps to same size
            target_height, target_width = depth_maps[0].shape
            resized_maps = []
            
            for depth_map in depth_maps:
                if depth_map.shape != (target_height, target_width):
                    resized = cv2.resize(depth_map, (target_width, target_height))
                    resized_maps.append(resized)
                else:
                    resized_maps.append(depth_map)
            
            # Create comparison image
            comparison_width = target_width * len(resized_maps)
            comparison = np.zeros((target_height + 60, comparison_width, 3), dtype=np.uint8)
            
            for i, (depth_map, metadata) in enumerate(zip(resized_maps, metadatas)):
                # Normalize and colorize
                depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Place in comparison image
                x_start = i * target_width
                comparison[:target_height, x_start:x_start + target_width] = depth_colored
                
                # Add labels
                label = f"Q:{metadata.quality_score:.2f}"
                cv2.putText(comparison, label, (x_start + 10, target_height + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                model_short = metadata.midas_model_version.split('/')[-1][:10]
                cv2.putText(comparison, model_short, (x_start + 10, target_height + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Save comparison
            cv2.imwrite(output_path, comparison)
            logger.info(f"Created comparison visualization: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")
            raise
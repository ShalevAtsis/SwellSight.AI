"""Depth map storage system for MiDaS extracted depth maps."""

import json
import numpy as np
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import pickle
import gzip

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
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None


class DepthMapStorage:
    """
    Storage system for MiDaS extracted depth maps.
    
    Provides efficient storage, retrieval, and management of depth maps
    with metadata tracking and integrity validation.
    """
    
    def __init__(self, storage_path: str, storage_format: str = 'compressed_numpy'):
        """
        Initialize depth map storage system.
        
        Args:
            storage_path: Base path for storage
            storage_format: Storage format ('numpy', 'compressed_numpy', 'pickle')
        """
        self.storage_path = Path(storage_path)
        self.storage_format = storage_format
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.depth_maps_path = self.storage_path / 'depth_maps'
        self.depth_maps_path.mkdir(exist_ok=True)
        
        # Initialize metadata database
        self.db_path = self.storage_path / 'metadata.db'
        self._initialize_database()
        
        logger.info(f"Depth map storage initialized at: {self.storage_path}")
        logger.info(f"Storage format: {self.storage_format}")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS depth_maps (
                    depth_map_id TEXT PRIMARY KEY,
                    original_image_path TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    midas_model_version TEXT NOT NULL,
                    extraction_timestamp TEXT NOT NULL,
                    depth_map_shape_h INTEGER NOT NULL,
                    depth_map_shape_w INTEGER NOT NULL,
                    depth_range_min REAL NOT NULL,
                    depth_range_max REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    storage_format TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    checksum TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_image_filename ON depth_maps(image_filename)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_extraction_timestamp ON depth_maps(extraction_timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_score ON depth_maps(quality_score)')
            
            conn.commit()
        
        logger.debug("Database initialized successfully")
    
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
            # Generate unique ID
            depth_map_id = self._generate_depth_map_id(metadata.original_image_path, metadata.extraction_timestamp)
            metadata.depth_map_id = depth_map_id
            
            # Store depth map file
            file_path = self._store_depth_map_file(depth_map, depth_map_id)
            
            # Calculate file size and checksum
            metadata.file_size_bytes = file_path.stat().st_size
            metadata.checksum = self._calculate_checksum(file_path)
            
            # Store metadata in database
            self._store_metadata(metadata)
            
            logger.debug(f"Stored depth map: {depth_map_id}")
            return depth_map_id
            
        except Exception as e:
            logger.error(f"Failed to store depth map: {e}")
            raise
    
    def retrieve_depth_map(self, depth_map_id: str) -> Tuple[np.ndarray, DepthMapMetadata]:
        """
        Retrieve depth map and metadata by ID.
        
        Args:
            depth_map_id: Depth map ID to retrieve
        
        Returns:
            Tuple of (depth_map, metadata)
        """
        try:
            # Retrieve metadata
            metadata = self._retrieve_metadata(depth_map_id)
            if not metadata:
                raise ValueError(f"Depth map not found: {depth_map_id}")
            
            # Load depth map file
            depth_map = self._load_depth_map_file(depth_map_id, metadata.storage_format)
            
            # Validate integrity
            if not self._validate_integrity(depth_map, metadata):
                logger.warning(f"Integrity validation failed for depth map: {depth_map_id}")
            
            return depth_map, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve depth map {depth_map_id}: {e}")
            raise
    
    def find_by_image(self, image_filename: str) -> List[DepthMapMetadata]:
        """
        Find depth maps by original image filename.
        
        Args:
            image_filename: Original image filename
        
        Returns:
            List of matching metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM depth_maps 
                    WHERE image_filename = ? 
                    ORDER BY extraction_timestamp DESC
                ''', (image_filename,))
                
                results = []
                for row in cursor.fetchall():
                    metadata = self._row_to_metadata(row)
                    results.append(metadata)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to find depth maps for image {image_filename}: {e}")
            return []
    
    def find_by_quality_range(self, min_quality: float, max_quality: float = 1.0) -> List[DepthMapMetadata]:
        """
        Find depth maps by quality score range.
        
        Args:
            min_quality: Minimum quality score
            max_quality: Maximum quality score
        
        Returns:
            List of matching metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM depth_maps 
                    WHERE quality_score >= ? AND quality_score <= ?
                    ORDER BY quality_score DESC
                ''', (min_quality, max_quality))
                
                results = []
                for row in cursor.fetchall():
                    metadata = self._row_to_metadata(row)
                    results.append(metadata)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to find depth maps by quality range: {e}")
            return []
    
    def delete_depth_map(self, depth_map_id: str) -> bool:
        """
        Delete depth map and its metadata.
        
        Args:
            depth_map_id: Depth map ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete file
            file_path = self._get_depth_map_file_path(depth_map_id)
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM depth_maps WHERE depth_map_id = ?', (depth_map_id,))
                deleted = cursor.rowcount > 0
            
            if deleted:
                logger.debug(f"Deleted depth map: {depth_map_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete depth map {depth_map_id}: {e}")
            return False
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage system statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total depth maps
                cursor = conn.execute('SELECT COUNT(*) FROM depth_maps')
                total_count = cursor.fetchone()[0]
                
                # Calculate total storage size
                cursor = conn.execute('SELECT SUM(file_size_bytes) FROM depth_maps')
                total_size = cursor.fetchone()[0] or 0
                
                # Get quality statistics
                cursor = conn.execute('''
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality
                    FROM depth_maps
                ''')
                quality_stats = cursor.fetchone()
                
                # Get model version distribution
                cursor = conn.execute('''
                    SELECT midas_model_version, COUNT(*) 
                    FROM depth_maps 
                    GROUP BY midas_model_version
                ''')
                model_distribution = dict(cursor.fetchall())
                
                # Get storage format distribution
                cursor = conn.execute('''
                    SELECT storage_format, COUNT(*) 
                    FROM depth_maps 
                    GROUP BY storage_format
                ''')
                format_distribution = dict(cursor.fetchall())
                
                return {
                    'total_depth_maps': total_count,
                    'total_storage_bytes': total_size,
                    'total_storage_mb': total_size / (1024 * 1024) if total_size else 0,
                    'average_quality_score': quality_stats[0] if quality_stats[0] else 0,
                    'min_quality_score': quality_stats[1] if quality_stats[1] else 0,
                    'max_quality_score': quality_stats[2] if quality_stats[2] else 0,
                    'model_version_distribution': model_distribution,
                    'storage_format_distribution': format_distribution,
                    'storage_path': str(self.storage_path)
                }
                
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return {}
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up orphaned depth map files without metadata entries.
        
        Returns:
            Number of files cleaned up
        """
        try:
            # Get all depth map IDs from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT depth_map_id FROM depth_maps')
                valid_ids = {row[0] for row in cursor.fetchall()}
            
            # Find all files in storage directory
            cleanup_count = 0
            for file_path in self.depth_maps_path.iterdir():
                if file_path.is_file():
                    # Extract ID from filename
                    file_id = file_path.stem
                    if file_id not in valid_ids:
                        file_path.unlink()
                        cleanup_count += 1
                        logger.debug(f"Cleaned up orphaned file: {file_path}")
            
            logger.info(f"Cleaned up {cleanup_count} orphaned files")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return 0
    
    def _generate_depth_map_id(self, image_path: str, timestamp: str) -> str:
        """Generate unique depth map ID."""
        # Create hash from image path and timestamp
        content = f"{image_path}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        return f"depth_{hash_obj.hexdigest()[:16]}"
    
    def _store_depth_map_file(self, depth_map: np.ndarray, depth_map_id: str) -> Path:
        """Store depth map to file."""
        if self.storage_format == 'numpy':
            file_path = self.depth_maps_path / f"{depth_map_id}.npy"
            np.save(file_path, depth_map)
            
        elif self.storage_format == 'compressed_numpy':
            file_path = self.depth_maps_path / f"{depth_map_id}.npz"
            np.savez_compressed(file_path, depth_map=depth_map)
            
        elif self.storage_format == 'pickle':
            file_path = self.depth_maps_path / f"{depth_map_id}.pkl.gz"
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(depth_map, f)
                
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
        
        return file_path
    
    def _load_depth_map_file(self, depth_map_id: str, storage_format: str) -> np.ndarray:
        """Load depth map from file."""
        if storage_format == 'numpy':
            file_path = self.depth_maps_path / f"{depth_map_id}.npy"
            return np.load(file_path)
            
        elif storage_format == 'compressed_numpy':
            file_path = self.depth_maps_path / f"{depth_map_id}.npz"
            data = np.load(file_path)
            return data['depth_map']
            
        elif storage_format == 'pickle':
            file_path = self.depth_maps_path / f"{depth_map_id}.pkl.gz"
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError(f"Unsupported storage format: {storage_format}")
    
    def _get_depth_map_file_path(self, depth_map_id: str) -> Path:
        """Get file path for depth map ID."""
        # Try different extensions based on storage format
        extensions = {
            'numpy': '.npy',
            'compressed_numpy': '.npz',
            'pickle': '.pkl.gz'
        }
        
        for format_name, ext in extensions.items():
            file_path = self.depth_maps_path / f"{depth_map_id}{ext}"
            if file_path.exists():
                return file_path
        
        # Default to current storage format
        ext = extensions.get(self.storage_format, '.npy')
        return self.depth_maps_path / f"{depth_map_id}{ext}"
    
    def _store_metadata(self, metadata: DepthMapMetadata) -> None:
        """Store metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO depth_maps (
                    depth_map_id, original_image_path, image_filename,
                    midas_model_version, extraction_timestamp,
                    depth_map_shape_h, depth_map_shape_w,
                    depth_range_min, depth_range_max,
                    quality_score, storage_format,
                    file_size_bytes, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.depth_map_id,
                metadata.original_image_path,
                metadata.image_filename,
                metadata.midas_model_version,
                metadata.extraction_timestamp,
                metadata.depth_map_shape[0],
                metadata.depth_map_shape[1],
                metadata.depth_range[0],
                metadata.depth_range[1],
                metadata.quality_score,
                metadata.storage_format,
                metadata.file_size_bytes,
                metadata.checksum
            ))
            conn.commit()
    
    def _retrieve_metadata(self, depth_map_id: str) -> Optional[DepthMapMetadata]:
        """Retrieve metadata from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM depth_maps WHERE depth_map_id = ?
            ''', (depth_map_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_metadata(row)
            
            return None
    
    def _row_to_metadata(self, row: Tuple) -> DepthMapMetadata:
        """Convert database row to metadata object."""
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
            file_size_bytes=row[11],
            checksum=row[12]
        )
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _validate_integrity(self, depth_map: np.ndarray, metadata: DepthMapMetadata) -> bool:
        """Validate depth map integrity against metadata."""
        try:
            # Check shape
            if depth_map.shape != metadata.depth_map_shape:
                logger.warning(f"Shape mismatch: expected {metadata.depth_map_shape}, got {depth_map.shape}")
                return False
            
            # Check depth range (allow some tolerance)
            actual_min = float(depth_map.min())
            actual_max = float(depth_map.max())
            expected_min, expected_max = metadata.depth_range
            
            tolerance = 0.01  # 1% tolerance
            if (abs(actual_min - expected_min) > tolerance * abs(expected_min) or
                abs(actual_max - expected_max) > tolerance * abs(expected_max)):
                logger.warning(f"Depth range mismatch: expected {metadata.depth_range}, got ({actual_min}, {actual_max})")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Integrity validation error: {e}")
            return False
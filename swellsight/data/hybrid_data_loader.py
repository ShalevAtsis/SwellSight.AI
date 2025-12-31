"""Hybrid data loader for MiDaS depth maps, ControlNet synthetic images, and augmentation metadata."""

import json
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import logging
from datetime import datetime
import random

# Import existing data loaders and components
from .real_data_loader import RealDataLoader, RealWaveDataset
from .synthetic_data_generator import SyntheticDataGenerator
from .augmentation import WaveAugmentationPipeline, create_augmentation_pipeline
from .midas_depth_extractor import MiDaSDepthExtractor

# Set up logging
logger = logging.getLogger(__name__)


class HybridWaveDataset(Dataset):
    """
    Hybrid dataset that combines real images, synthetic images, depth maps, and augmentation metadata.
    
    Handles MiDaS depth maps, ControlNet synthetic images, and comprehensive augmentation metadata
    while ensuring proper train/validation splits with real data isolation.
    """
    
    def __init__(self, samples_metadata: List[Dict[str, Any]], 
                 transform: Optional[transforms.Compose] = None,
                 include_depth_maps: bool = True,
                 include_augmentation_metadata: bool = True,
                 data_source_filter: Optional[str] = None):
        """
        Initialize hybrid wave dataset.
        
        Args:
            samples_metadata: List of sample metadata dictionaries
            transform: Optional image transforms
            include_depth_maps: Whether to load and include depth maps
            include_augmentation_metadata: Whether to include augmentation parameters
            data_source_filter: Optional filter for data source ('real', 'synthetic', 'synthetic_from_real')
        """
        self.samples_metadata = samples_metadata
        self.transform = transform
        self.include_depth_maps = include_depth_maps
        self.include_augmentation_metadata = include_augmentation_metadata
        
        # Filter by data source if specified
        if data_source_filter:
            self.samples_metadata = [
                sample for sample in samples_metadata 
                if sample.get('data_source', '').startswith(data_source_filter)
            ]
        
        # Wave type and direction mappings
        self.wave_type_to_idx = {
            'A_FRAME': 0,
            'CLOSEOUT': 1, 
            'BEACH_BREAK': 2,
            'POINT_BREAK': 3
        }
        
        self.direction_to_idx = {
            'LEFT': 0,
            'RIGHT': 1,
            'BOTH': 2
        }
        
        # Initialize depth extractor if needed
        self.depth_extractor = None
        if include_depth_maps:
            try:
                self.depth_extractor = MiDaSDepthExtractor()
            except Exception as e:
                logger.warning(f"Failed to initialize depth extractor: {e}")
                self.include_depth_maps = False
        
        logger.info(f"Initialized HybridWaveDataset with {len(self.samples_metadata)} samples")
        logger.info(f"Include depth maps: {self.include_depth_maps}")
        logger.info(f"Include augmentation metadata: {self.include_augmentation_metadata}")
        if data_source_filter:
            logger.info(f"Data source filter: {data_source_filter}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples_metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image tensor, labels, and optional depth/augmentation data
        """
        sample_meta = self.samples_metadata[idx]
        
        # Load image
        image_path = Path(sample_meta['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Prepare labels
        height = torch.tensor(sample_meta['height_meters'], dtype=torch.float32)
        wave_type = torch.tensor(self.wave_type_to_idx[sample_meta['wave_type']], dtype=torch.long)
        direction = torch.tensor(self.direction_to_idx[sample_meta['direction']], dtype=torch.long)
        
        # Create base sample dictionary
        sample = {
            'image': image,
            'height': height,
            'wave_type': wave_type,
            'direction': direction,
            'sample_id': sample_meta['sample_id'],
            'data_source': sample_meta.get('data_source', 'unknown')
        }
        
        # Add depth map if available and requested
        if self.include_depth_maps:
            depth_map = self._load_depth_map(sample_meta)
            if depth_map is not None:
                sample['depth_map'] = torch.tensor(depth_map, dtype=torch.float32)
        
        # Add augmentation metadata if available and requested
        if self.include_augmentation_metadata and 'augmentation_params' in sample_meta:
            aug_params = self._process_augmentation_metadata(sample_meta['augmentation_params'])
            sample['augmentation_metadata'] = aug_params
        
        # Add correspondence information for synthetic samples
        if sample_meta.get('data_source') == 'synthetic_from_real':
            sample['original_real_image'] = sample_meta.get('original_real_image', '')
            sample['depth_map_id'] = sample_meta.get('depth_map_id', '')
            sample['depth_quality_score'] = torch.tensor(
                sample_meta.get('depth_quality_score', 0.0), dtype=torch.float32
            )
        
        return sample
    
    def _load_depth_map(self, sample_meta: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load depth map for a sample if available."""
        try:
            # Check if depth map ID is available (for synthetic_from_real samples)
            if 'depth_map_id' in sample_meta and sample_meta['depth_map_id']:
                if self.depth_extractor and self.depth_extractor.storage:
                    depth_map, _ = self.depth_extractor.retrieve_stored_depth_map(
                        sample_meta['depth_map_id']
                    )
                    return depth_map
            
            # Check if original real image is available for depth extraction
            if 'original_real_image' in sample_meta:
                original_image_path = sample_meta['original_real_image']
                if Path(original_image_path).exists() and self.depth_extractor:
                    depth_result = self.depth_extractor.extract_depth(original_image_path)
                    return depth_result.depth_map
            
            # For pure synthetic samples, try to reconstruct depth map from generation params
            if sample_meta.get('data_source') == 'synthetic' and 'generation_params' in sample_meta:
                # This would require the original depth map generation logic
                # For now, return None
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to load depth map for sample {sample_meta.get('sample_id', 'unknown')}: {e}")
            return None
    
    def _process_augmentation_metadata(self, aug_params: Dict[str, Any]) -> torch.Tensor:
        """Process augmentation metadata into tensor format."""
        # Convert key augmentation parameters to tensor
        # Select most important parameters for model training
        key_params = [
            aug_params.get('dominant_wave_height_m', 1.0),
            aug_params.get('breaker_intensity', 0.5),
            aug_params.get('surface_roughness', 0.3),
            aug_params.get('sun_elevation_deg', 45.0) / 90.0,  # Normalize to [0,1]
            aug_params.get('haze_density', 0.1),
            aug_params.get('foam_coverage_pct', 20.0) / 100.0,  # Normalize to [0,1]
            aug_params.get('sensor_noise_level', 0.02),
            aug_params.get('light_intensity', 1.0) / 2.0,  # Normalize to [0,1]
        ]
        
        return torch.tensor(key_params, dtype=torch.float32)
    
    def get_data_source_distribution(self) -> Dict[str, int]:
        """Get distribution of data sources in the dataset."""
        distribution = {}
        for sample in self.samples_metadata:
            source = sample.get('data_source', 'unknown')
            distribution[source] = distribution.get(source, 0) + 1
        return distribution
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """Get statistics about labels in the dataset."""
        heights = [sample['height_meters'] for sample in self.samples_metadata]
        wave_types = [sample['wave_type'] for sample in self.samples_metadata]
        directions = [sample['direction'] for sample in self.samples_metadata]
        
        return {
            'height_stats': {
                'min': min(heights),
                'max': max(heights),
                'mean': sum(heights) / len(heights),
                'count': len(heights)
            },
            'wave_type_distribution': {
                wt: wave_types.count(wt) for wt in set(wave_types)
            },
            'direction_distribution': {
                d: directions.count(d) for d in set(directions)
            }
        }


class HybridDataLoader:
    """
    Hybrid data loader for MiDaS depth maps, ControlNet synthetic images, and augmentation metadata.
    
    Handles data loading with proper train/validation splits, real data isolation,
    domain adaptation training strategies, and data streaming for large-scale synthetic datasets.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid data loader.
        
        Args:
            data_path: Path to data directory
            config: Optional configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # Configuration parameters
        self.image_size = self.config.get('image_size', (768, 768))
        self.batch_size = self.config.get('batch_size', 32)
        self.num_workers = self.config.get('num_workers', 4)
        self.pin_memory = self.config.get('pin_memory', True)
        
        # Data split configuration
        self.train_split = self.config.get('train_split', 0.8)
        self.val_split = self.config.get('val_split', 0.2)
        self.ensure_real_data_isolation = self.config.get('ensure_real_data_isolation', True)
        
        # Domain adaptation configuration
        self.enable_domain_adaptation = self.config.get('enable_domain_adaptation', False)
        self.synthetic_real_ratio = self.config.get('synthetic_real_ratio', 10.0)  # 10:1 synthetic to real
        
        # Data streaming configuration
        self.enable_streaming = self.config.get('enable_streaming', False)
        self.streaming_buffer_size = self.config.get('streaming_buffer_size', 1000)
        
        # Paths
        self.synthetic_path = self.data_path / 'synthetic'
        self.real_path = self.data_path / 'real'
        self.metadata_path = self.data_path / 'metadata'
        
        # Initialize components
        self.real_data_loader = RealDataLoader(str(self.real_path), config)
        self.augmentation_pipeline = create_augmentation_pipeline(config.get('augmentation', {}))
        
        # Cached datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        logger.info(f"Initialized HybridDataLoader with data path: {self.data_path}")
        logger.info(f"Real data isolation: {self.ensure_real_data_isolation}")
        logger.info(f"Domain adaptation: {self.enable_domain_adaptation}")
        logger.info(f"Data streaming: {self.enable_streaming}")
    
    def load_all_metadata(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load all synthetic and real metadata.
        
        Returns:
            Tuple of (synthetic_metadata, real_metadata)
        """
        # Load synthetic metadata
        synthetic_metadata = []
        synthetic_metadata_file = self.metadata_path / 'synthetic_dataset_metadata.json'
        if synthetic_metadata_file.exists():
            with open(synthetic_metadata_file, 'r') as f:
                synthetic_metadata = json.load(f)
        
        # Load real metadata
        real_metadata = self.real_data_loader.load_real_metadata()
        
        logger.info(f"Loaded {len(synthetic_metadata)} synthetic samples")
        logger.info(f"Loaded {len(real_metadata)} real samples")
        
        return synthetic_metadata, real_metadata
    
    def create_train_val_splits(self, synthetic_metadata: List[Dict[str, Any]], 
                               real_metadata: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], 
                                                                           List[Dict[str, Any]], 
                                                                           List[Dict[str, Any]]]:
        """
        Create train/validation/test splits with proper real data isolation.
        
        Args:
            synthetic_metadata: List of synthetic sample metadata
            real_metadata: List of real sample metadata
        
        Returns:
            Tuple of (train_metadata, val_metadata, test_metadata)
        """
        # Ensure real data isolation: all real data goes to test set only
        if self.ensure_real_data_isolation:
            test_metadata = real_metadata.copy()
            
            # Split synthetic data for train/val
            random.shuffle(synthetic_metadata)
            split_idx = int(len(synthetic_metadata) * self.train_split)
            
            train_metadata = synthetic_metadata[:split_idx]
            val_metadata = synthetic_metadata[split_idx:]
            
            logger.info(f"Real data isolation enabled:")
            logger.info(f"  Train: {len(train_metadata)} synthetic samples")
            logger.info(f"  Val: {len(val_metadata)} synthetic samples")
            logger.info(f"  Test: {len(test_metadata)} real samples")
        
        else:
            # Mix real and synthetic data (not recommended for domain adaptation)
            all_metadata = synthetic_metadata + real_metadata
            random.shuffle(all_metadata)
            
            train_split_idx = int(len(all_metadata) * self.train_split)
            val_split_idx = int(len(all_metadata) * (self.train_split + self.val_split))
            
            train_metadata = all_metadata[:train_split_idx]
            val_metadata = all_metadata[train_split_idx:val_split_idx]
            test_metadata = all_metadata[val_split_idx:]
            
            logger.info(f"Mixed data splits:")
            logger.info(f"  Train: {len(train_metadata)} samples")
            logger.info(f"  Val: {len(val_metadata)} samples")
            logger.info(f"  Test: {len(test_metadata)} samples")
        
        return train_metadata, val_metadata, test_metadata
    
    def get_training_loader(self, batch_size: Optional[int] = None, 
                           include_depth_maps: bool = True,
                           include_augmentation_metadata: bool = True) -> DataLoader:
        """
        Get training data loader with augmentation.
        
        Args:
            batch_size: Override default batch size
            include_depth_maps: Whether to include depth maps
            include_augmentation_metadata: Whether to include augmentation metadata
        
        Returns:
            Training data loader
        """
        if self._train_dataset is None:
            synthetic_metadata, real_metadata = self.load_all_metadata()
            train_metadata, _, _ = self.create_train_val_splits(synthetic_metadata, real_metadata)
            
            # Create training dataset with augmentation
            train_transform = self.augmentation_pipeline.get_training_transforms()
            
            if self.enable_domain_adaptation:
                # Create separate datasets for domain adaptation
                self._train_dataset = self._create_domain_adaptation_dataset(
                    train_metadata, train_transform, include_depth_maps, include_augmentation_metadata
                )
            else:
                # Standard training dataset
                self._train_dataset = HybridWaveDataset(
                    train_metadata, train_transform, include_depth_maps, include_augmentation_metadata
                )
        
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def get_validation_loader(self, batch_size: Optional[int] = None,
                             include_depth_maps: bool = True,
                             include_augmentation_metadata: bool = True) -> DataLoader:
        """
        Get validation data loader without augmentation.
        
        Args:
            batch_size: Override default batch size
            include_depth_maps: Whether to include depth maps
            include_augmentation_metadata: Whether to include augmentation metadata
        
        Returns:
            Validation data loader
        """
        if self._val_dataset is None:
            synthetic_metadata, real_metadata = self.load_all_metadata()
            _, val_metadata, _ = self.create_train_val_splits(synthetic_metadata, real_metadata)
            
            # Create validation dataset without augmentation
            val_transform = self.augmentation_pipeline.get_validation_transforms()
            self._val_dataset = HybridWaveDataset(
                val_metadata, val_transform, include_depth_maps, include_augmentation_metadata
            )
        
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_test_loader(self, batch_size: Optional[int] = None,
                       include_depth_maps: bool = True,
                       include_augmentation_metadata: bool = False) -> DataLoader:
        """
        Get test data loader (real data only if isolation is enabled).
        
        Args:
            batch_size: Override default batch size
            include_depth_maps: Whether to include depth maps
            include_augmentation_metadata: Whether to include augmentation metadata
        
        Returns:
            Test data loader
        """
        if self._test_dataset is None:
            synthetic_metadata, real_metadata = self.load_all_metadata()
            _, _, test_metadata = self.create_train_val_splits(synthetic_metadata, real_metadata)
            
            # Create test dataset without augmentation
            test_transform = self.augmentation_pipeline.get_validation_transforms()
            self._test_dataset = HybridWaveDataset(
                test_metadata, test_transform, include_depth_maps, include_augmentation_metadata
            )
        
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def _create_domain_adaptation_dataset(self, train_metadata: List[Dict[str, Any]],
                                        transform: transforms.Compose,
                                        include_depth_maps: bool,
                                        include_augmentation_metadata: bool) -> Dataset:
        """Create dataset for domain adaptation training."""
        # Separate synthetic and real samples
        synthetic_samples = [s for s in train_metadata if s.get('data_source', '').startswith('synthetic')]
        real_samples = [s for s in train_metadata if s.get('data_source') == 'real']
        
        # Create separate datasets
        synthetic_dataset = HybridWaveDataset(
            synthetic_samples, transform, include_depth_maps, include_augmentation_metadata,
            data_source_filter='synthetic'
        )
        
        if real_samples:
            real_dataset = HybridWaveDataset(
                real_samples, transform, include_depth_maps, include_augmentation_metadata,
                data_source_filter='real'
            )
            
            # Combine datasets with specified ratio
            # This is a simplified approach - more sophisticated domain adaptation
            # would require custom sampling strategies
            return ConcatDataset([synthetic_dataset, real_dataset])
        else:
            return synthetic_dataset
    
    def validate_data_isolation(self) -> bool:
        """
        Validate that real data is properly isolated from training data.
        
        Returns:
            True if data isolation is maintained
        """
        synthetic_metadata, real_metadata = self.load_all_metadata()
        train_metadata, val_metadata, test_metadata = self.create_train_val_splits(
            synthetic_metadata, real_metadata
        )
        
        # Check that no real data appears in training or validation sets
        train_real_samples = [s for s in train_metadata if s.get('data_source') == 'real']
        val_real_samples = [s for s in val_metadata if s.get('data_source') == 'real']
        
        if train_real_samples or val_real_samples:
            logger.error(f"Data isolation violation: {len(train_real_samples)} real samples in train, "
                        f"{len(val_real_samples)} real samples in val")
            return False
        
        # Check that all real samples are in test set
        test_real_samples = [s for s in test_metadata if s.get('data_source') == 'real']
        if len(test_real_samples) != len(real_metadata):
            logger.error(f"Data isolation violation: {len(test_real_samples)} real samples in test, "
                        f"expected {len(real_metadata)}")
            return False
        
        logger.info("Data isolation validation passed")
        return True
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        synthetic_metadata, real_metadata = self.load_all_metadata()
        train_metadata, val_metadata, test_metadata = self.create_train_val_splits(
            synthetic_metadata, real_metadata
        )
        
        def get_split_stats(metadata: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
            if not metadata:
                return {'count': 0}
            
            heights = [s['height_meters'] for s in metadata]
            wave_types = [s['wave_type'] for s in metadata]
            directions = [s['direction'] for s in metadata]
            data_sources = [s.get('data_source', 'unknown') for s in metadata]
            
            return {
                'count': len(metadata),
                'height_stats': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': sum(heights) / len(heights)
                },
                'wave_type_distribution': {wt: wave_types.count(wt) for wt in set(wave_types)},
                'direction_distribution': {d: directions.count(d) for d in set(directions)},
                'data_source_distribution': {ds: data_sources.count(ds) for ds in set(data_sources)}
            }
        
        return {
            'total_synthetic': len(synthetic_metadata),
            'total_real': len(real_metadata),
            'train_stats': get_split_stats(train_metadata, 'train'),
            'val_stats': get_split_stats(val_metadata, 'val'),
            'test_stats': get_split_stats(test_metadata, 'test'),
            'data_isolation_enabled': self.ensure_real_data_isolation,
            'domain_adaptation_enabled': self.enable_domain_adaptation,
            'streaming_enabled': self.enable_streaming
        }
    
    def clear_cache(self):
        """Clear cached datasets to free memory."""
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        logger.info("Cleared dataset cache")
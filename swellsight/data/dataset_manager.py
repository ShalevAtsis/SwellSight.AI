"""Dataset manager for wave analysis training data."""

import json
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import logging

from .augmentation import WaveAugmentationPipeline, create_augmentation_pipeline

# Set up logging
logger = logging.getLogger(__name__)


class WaveDataset(Dataset):
    """PyTorch Dataset for wave analysis data."""
    
    def __init__(self, samples_metadata: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None):
        """
        Initialize wave dataset.
        
        Args:
            samples_metadata: List of sample metadata dictionaries
            transform: Optional image transforms
        """
        self.samples_metadata = samples_metadata
        self.transform = transform
        
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
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples_metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image tensor and labels
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
        
        return {
            'image': image,
            'height': height,
            'wave_type': wave_type,
            'direction': direction,
            'sample_id': sample_meta['sample_id']
        }


class DatasetManager:
    """
    Manager for dataset loading and splitting.
    
    Handles train/validation splits for synthetic data and manages
    real-world test data separately to prevent data leakage.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset manager.
        
        Args:
            data_path: Path to dataset directory
            config: Optional configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # Configuration parameters
        self.train_split = self.config.get('train_split', 0.8)
        self.val_split = self.config.get('val_split', 0.2)
        self.image_size = self.config.get('image_size', (768, 768))
        self.normalize_mean = self.config.get('normalize_mean', (0.485, 0.456, 0.406))
        self.normalize_std = self.config.get('normalize_std', (0.229, 0.224, 0.225))
        
        # Data augmentation parameters
        self.rotation_range = self.config.get('rotation_range', 15.0)
        self.brightness_range = self.config.get('brightness_range', 0.2)
        self.contrast_range = self.config.get('contrast_range', 0.15)
        self.gaussian_noise_std = self.config.get('gaussian_noise_std', 0.01)
        
        # Data loading parameters
        self.num_workers = self.config.get('num_workers', 4)
        self.pin_memory = self.config.get('pin_memory', True)
        self.shuffle_train = self.config.get('shuffle_train', True)
        
        # Paths
        self.synthetic_path = self.data_path / 'synthetic'
        self.real_path = self.data_path / 'real'
        self.metadata_path = self.data_path / 'metadata'
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = create_augmentation_pipeline(self.config)
        
        # Cached datasets
        self._train_dataset = None
        self._val_dataset = None
        self._real_test_dataset = None
        
        logger.info(f"Initialized DatasetManager with data path: {self.data_path}")
        logger.info(f"Train/Val split: {self.train_split:.1%}/{self.val_split:.1%}")
        logger.info(f"Augmentation pipeline: {self.augmentation_pipeline.get_augmentation_info()}")
    
    def _get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        Get image transforms for training or validation.
        
        Args:
            is_training: Whether to apply training augmentations
            
        Returns:
            Composed transforms
        """
        if is_training:
            return self.augmentation_pipeline.get_training_transforms()
        else:
            return self.augmentation_pipeline.get_validation_transforms()
    
    def _load_synthetic_metadata(self) -> List[Dict[str, Any]]:
        """
        Load synthetic dataset metadata.
        
        Returns:
            List of sample metadata
        """
        metadata_file = self.metadata_path / 'synthetic_dataset_metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Synthetic metadata file not found: {metadata_file}")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {len(metadata)} synthetic samples from metadata")
        return metadata
    
    def _load_real_metadata(self) -> List[Dict[str, Any]]:
        """
        Load real dataset metadata.
        
        Returns:
            List of sample metadata
        """
        metadata_file = self.metadata_path / 'real_dataset_metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Real metadata file not found: {metadata_file}")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {len(metadata)} real samples from metadata")
        return metadata
    
    def _split_synthetic_data(self, metadata: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split synthetic data into train and validation sets.
        
        Args:
            metadata: List of sample metadata
            
        Returns:
            Tuple of (train_metadata, val_metadata)
        """
        if not metadata:
            return [], []
        
        # Calculate split sizes
        total_samples = len(metadata)
        train_size = int(total_samples * self.train_split)
        val_size = total_samples - train_size
        
        # Create indices for splitting
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_metadata = [metadata[i] for i in train_indices]
        val_metadata = [metadata[i] for i in val_indices]
        
        logger.info(f"Split synthetic data: {len(train_metadata)} train, {len(val_metadata)} validation")
        
        return train_metadata, val_metadata
    
    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        """
        Get training data loader.
        
        Args:
            batch_size: Batch size for data loading
            
        Returns:
            Training data loader
        """
        if self._train_dataset is None:
            # Load and split synthetic data
            synthetic_metadata = self._load_synthetic_metadata()
            train_metadata, _ = self._split_synthetic_data(synthetic_metadata)
            
            # Create dataset with training transforms
            train_transform = self._get_transforms(is_training=True)
            self._train_dataset = WaveDataset(train_metadata, transform=train_transform)
        
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Ensure consistent batch sizes
        )
    
    def get_validation_loader(self, batch_size: int = 32) -> DataLoader:
        """
        Get validation data loader.
        
        Args:
            batch_size: Batch size for data loading
            
        Returns:
            Validation data loader
        """
        if self._val_dataset is None:
            # Load and split synthetic data
            synthetic_metadata = self._load_synthetic_metadata()
            _, val_metadata = self._split_synthetic_data(synthetic_metadata)
            
            # Create dataset with validation transforms (no augmentation)
            val_transform = self._get_transforms(is_training=False)
            self._val_dataset = WaveDataset(val_metadata, transform=val_transform)
        
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_real_test_loader(self, batch_size: int = 32) -> DataLoader:
        """
        Get real-world test data loader.
        
        Args:
            batch_size: Batch size for data loading
            
        Returns:
            Real test data loader
        """
        if self._real_test_dataset is None:
            # Load real data metadata
            real_metadata = self._load_real_metadata()
            
            # Create dataset with validation transforms (no augmentation)
            test_transform = self._get_transforms(is_training=False)
            self._real_test_dataset = WaveDataset(real_metadata, transform=test_transform)
        
        return DataLoader(
            self._real_test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        synthetic_metadata = self._load_synthetic_metadata()
        real_metadata = self._load_real_metadata()
        
        train_metadata, val_metadata = self._split_synthetic_data(synthetic_metadata)
        
        info = {
            'synthetic_total': len(synthetic_metadata),
            'synthetic_train': len(train_metadata),
            'synthetic_val': len(val_metadata),
            'real_test': len(real_metadata),
            'train_split_ratio': self.train_split,
            'val_split_ratio': self.val_split,
            'image_size': self.image_size,
            'data_path': str(self.data_path)
        }
        
        # Add class distribution if data exists
        if train_metadata:
            wave_types = [sample['wave_type'] for sample in train_metadata]
            directions = [sample['direction'] for sample in train_metadata]
            
            info['train_wave_type_distribution'] = {
                wave_type: wave_types.count(wave_type) 
                for wave_type in set(wave_types)
            }
            info['train_direction_distribution'] = {
                direction: directions.count(direction) 
                for direction in set(directions)
            }
        
        return info
    
    def validate_dataset_integrity(self) -> Dict[str, bool]:
        """
        Validate dataset integrity and splits.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'synthetic_metadata_exists': False,
            'real_metadata_exists': False,
            'train_val_split_valid': False,
            'no_data_leakage': True,
            'all_images_exist': True
        }
        
        # Check metadata files exist
        synthetic_meta_file = self.metadata_path / 'synthetic_dataset_metadata.json'
        real_meta_file = self.metadata_path / 'real_dataset_metadata.json'
        
        results['synthetic_metadata_exists'] = synthetic_meta_file.exists()
        results['real_metadata_exists'] = real_meta_file.exists()
        
        if results['synthetic_metadata_exists']:
            synthetic_metadata = self._load_synthetic_metadata()
            train_metadata, val_metadata = self._split_synthetic_data(synthetic_metadata)
            
            # Check split ratios
            total = len(synthetic_metadata)
            if total > 0:
                actual_train_ratio = len(train_metadata) / total
                expected_ratio = self.train_split
                ratio_diff = abs(actual_train_ratio - expected_ratio)
                # For small datasets, allow larger tolerance
                tolerance = 0.15 if total < 20 else 0.05
                results['train_val_split_valid'] = ratio_diff <= tolerance
            
            # Check for data leakage (no overlap between train and val)
            train_ids = {sample['sample_id'] for sample in train_metadata}
            val_ids = {sample['sample_id'] for sample in val_metadata}
            results['no_data_leakage'] = len(train_ids.intersection(val_ids)) == 0
            
            # Check if all image files exist
            all_samples = train_metadata + val_metadata
            for sample in all_samples:
                image_path = Path(sample['image_path'])
                if not image_path.exists():
                    results['all_images_exist'] = False
                    break
        
        return results
    
    def validate_augmentation_preserves_labels(self, sample_metadata: Dict[str, Any]) -> bool:
        """
        Validate that augmentation preserves ground truth labels.
        
        Args:
            sample_metadata: Sample metadata with ground truth labels
            
        Returns:
            True if augmentation preserves labels
        """
        # Load sample image
        image_path = Path(sample_metadata['image_path'])
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False
        
        image = Image.open(image_path).convert('RGB')
        
        # Extract ground truth labels
        ground_truth = {
            'height_meters': sample_metadata['height_meters'],
            'wave_type': sample_metadata['wave_type'],
            'direction': sample_metadata['direction']
        }
        
        # Validate using augmentation pipeline
        return self.augmentation_pipeline.validate_augmentation_preserves_labels(
            image, ground_truth
        )
"""Real data loader for beach camera images."""

import json
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
from datetime import datetime
import csv

# Set up logging
logger = logging.getLogger(__name__)


class RealWaveDataset(Dataset):
    """PyTorch Dataset for real beach camera images."""
    
    def __init__(self, samples_metadata: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None):
        """
        Initialize real wave dataset.
        
        Args:
            samples_metadata: List of sample metadata dictionaries
            transform: Optional image transforms
        """
        self.samples_metadata = samples_metadata
        self.transform = transform
        
        # Wave type and direction mappings (same as synthetic data)
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
            'sample_id': sample_meta['sample_id'],
            'data_source': 'real'  # Mark as real data
        }


class RealDataLoader:
    """
    Loader for real beach camera images.
    
    Handles real-world beach camera images with manual labels.
    Ensures real images are isolated in test set only to prevent data leakage.
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real data loader.
        
        Args:
            data_path: Path to real data directory
            config: Optional configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # Configuration parameters
        self.image_size = self.config.get('image_size', (768, 768))
        self.normalize_mean = self.config.get('normalize_mean', (0.485, 0.456, 0.406))
        self.normalize_std = self.config.get('normalize_std', (0.229, 0.224, 0.225))
        
        # Data loading parameters
        self.num_workers = self.config.get('num_workers', 4)
        self.pin_memory = self.config.get('pin_memory', True)
        
        # Paths
        self.images_path = self.data_path / 'images'
        self.labels_path = self.data_path / 'labels'
        self.metadata_path = self.data_path.parent / 'metadata'
        
        # Ensure directories exist
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Cached dataset
        self._real_dataset = None
        
        logger.info(f"Initialized RealDataLoader with data path: {self.data_path}")
    
    def _get_transforms(self) -> transforms.Compose:
        """
        Get image transforms for real data (no augmentation).
        
        Returns:
            Composed transforms for validation/test
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def load_manual_labels(self, labels_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load manual labels from JSON file.
        
        Args:
            labels_file: Optional path to labels file
            
        Returns:
            List of label dictionaries
        """
        if labels_file is None:
            # Try JSON file first, then CSV as fallback
            json_file = self.labels_path / 'labels.json'
            csv_file = self.labels_path / 'manual_labels.csv'
            
            if json_file.exists():
                labels_file = json_file
            elif csv_file.exists():
                labels_file = csv_file
            else:
                logger.warning(f"Labels file not found: {json_file} or {csv_file}")
                return []
        else:
            labels_file = Path(labels_file)
        
        if not labels_file.exists():
            logger.warning(f"Labels file not found: {labels_file}")
            return []
        
        labels = []
        
        # Handle JSON format
        if labels_file.suffix == '.json':
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
            
            for image_filename, label_data in labels_data.items():
                label = {
                    'image_filename': image_filename,
                    'height_meters': float(label_data['height_meters']),
                    'wave_type': label_data['wave_type'],
                    'direction': label_data['direction'],
                    'confidence': label_data.get('confidence', 'high'),
                    'notes': label_data.get('notes', ''),
                    'data_key': label_data.get('data_key', 0),
                    'label_timestamp': datetime.now().isoformat()
                }
                labels.append(label)
        
        # Handle CSV format (fallback)
        elif labels_file.suffix == '.csv':
            with open(labels_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert string values to appropriate types
                    label = {
                        'image_filename': row['image_filename'],
                        'height_meters': float(row['height_meters']),
                        'wave_type': row['wave_type'],
                        'direction': row['direction'],
                        'labeler_id': row.get('labeler_id', 'unknown'),
                        'confidence': row.get('confidence', 'high'),
                        'notes': row.get('notes', ''),
                        'label_timestamp': row.get('label_timestamp', '')
                    }
                    labels.append(label)
        else:
            logger.error(f"Unsupported labels file format: {labels_file}")
            return []
        
        logger.info(f"Loaded {len(labels)} manual labels from {labels_file}")
        return labels
    
    def create_metadata_from_labels(self, labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create metadata from manual labels.
        
        Args:
            labels: List of manual label dictionaries
            
        Returns:
            List of sample metadata
        """
        metadata = []
        
        for i, label in enumerate(labels):
            image_path = self.images_path / label['image_filename']
            
            # Skip if image doesn't exist
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                continue
            
            sample_metadata = {
                'sample_id': f"real_{i:06d}",
                'image_path': str(image_path),
                'image_filename': label['image_filename'],
                'height_meters': label['height_meters'],
                'wave_type': label['wave_type'],
                'direction': label['direction'],
                'data_source': 'real',
                'labeler_info': {
                    'labeler_id': label['labeler_id'],
                    'confidence': label['confidence'],
                    'notes': label['notes'],
                    'label_timestamp': label['label_timestamp']
                },
                'image_size': self.image_size,
                'created_timestamp': datetime.now().isoformat()
            }
            
            metadata.append(sample_metadata)
        
        logger.info(f"Created metadata for {len(metadata)} real samples")
        return metadata
    
    def save_real_metadata(self, metadata: List[Dict[str, Any]], filename: str = 'real_dataset_metadata.json'):
        """
        Save real dataset metadata to file.
        
        Args:
            metadata: List of sample metadata
            filename: Output filename
        """
        metadata_file = self.metadata_path / filename
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved real dataset metadata to: {metadata_file}")
    
    def load_real_metadata(self, filename: str = 'real_dataset_metadata.json') -> List[Dict[str, Any]]:
        """
        Load real dataset metadata from file.
        
        Args:
            filename: Metadata filename
            
        Returns:
            List of sample metadata
        """
        metadata_file = self.metadata_path / filename
        
        if not metadata_file.exists():
            logger.warning(f"Real metadata file not found: {metadata_file}")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {len(metadata)} real samples from metadata")
        return metadata
    
    def get_test_loader(self, batch_size: int = 32, labels_file: Optional[str] = None) -> DataLoader:
        """
        Get test data loader for real images.
        
        Args:
            batch_size: Batch size for data loading
            labels_file: Optional path to labels file
            
        Returns:
            Test data loader
        """
        if self._real_dataset is None:
            # Load or create metadata
            metadata = self.load_real_metadata()
            
            if not metadata and labels_file:
                # Create metadata from labels file
                labels = self.load_manual_labels(labels_file)
                metadata = self.create_metadata_from_labels(labels)
                self.save_real_metadata(metadata)
            
            # Create dataset with test transforms (no augmentation)
            test_transform = self._get_transforms()
            self._real_dataset = RealWaveDataset(metadata, transform=test_transform)
        
        return DataLoader(
            self._real_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def validate_real_data_isolation(self, synthetic_metadata: List[Dict[str, Any]]) -> bool:
        """
        Validate that real data is isolated from synthetic training data.
        
        Args:
            synthetic_metadata: List of synthetic sample metadata
            
        Returns:
            True if real data is properly isolated
        """
        real_metadata = self.load_real_metadata()
        
        if not real_metadata:
            return True  # No real data to check
        
        # Get all image paths
        real_image_paths = {Path(sample['image_path']).name for sample in real_metadata}
        synthetic_image_paths = {Path(sample['image_path']).name for sample in synthetic_metadata}
        
        # Check for overlap
        overlap = real_image_paths.intersection(synthetic_image_paths)
        
        if overlap:
            logger.error(f"Found {len(overlap)} overlapping images between real and synthetic data")
            return False
        
        # Check that all real samples are marked as real data source
        for sample in real_metadata:
            if sample.get('data_source') != 'real':
                logger.error(f"Real sample {sample['sample_id']} not marked as real data source")
                return False
        
        logger.info("Real data isolation validation passed")
        return True
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the real dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        metadata = self.load_real_metadata()
        
        info = {
            'total_samples': len(metadata),
            'data_source': 'real',
            'image_size': self.image_size,
            'data_path': str(self.data_path)
        }
        
        # Add class distribution if data exists
        if metadata:
            wave_types = [sample['wave_type'] for sample in metadata]
            directions = [sample['direction'] for sample in metadata]
            heights = [sample['height_meters'] for sample in metadata]
            
            info['wave_type_distribution'] = {
                wave_type: wave_types.count(wave_type) 
                for wave_type in set(wave_types)
            }
            info['direction_distribution'] = {
                direction: directions.count(direction) 
                for direction in set(directions)
            }
            info['height_statistics'] = {
                'min': min(heights),
                'max': max(heights),
                'mean': sum(heights) / len(heights),
                'count': len(heights)
            }
        
        return info


class ManualLabelingUtility:
    """
    Utility for manual labeling of real beach camera images.
    
    Provides tools for creating and validating manual labels for real data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the manual labeling utility.
        
        Args:
            data_path: Path to real data directory
        """
        self.data_path = Path(data_path)
        self.images_path = self.data_path / 'images'
        self.labels_path = self.data_path / 'labels'
        
        # Ensure directories exist
        self.labels_path.mkdir(parents=True, exist_ok=True)
        
        # Valid values for validation
        self.valid_wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
        self.valid_directions = ['LEFT', 'RIGHT', 'BOTH']
        self.height_range = (0.1, 10.0)  # meters
        
        logger.info(f"Initialized ManualLabelingUtility for: {self.data_path}")
    
    def create_labels_template(self, output_file: str = 'manual_labels.csv') -> str:
        """
        Create a CSV template for manual labeling.
        
        Args:
            output_file: Output CSV filename
            
        Returns:
            Path to created template file
        """
        template_path = self.labels_path / output_file
        
        # Get list of image files
        image_files = []
        if self.images_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(self.images_path.glob(ext))
        
        # Create CSV with headers and empty rows for each image
        with open(template_path, 'w', newline='') as csvfile:
            fieldnames = [
                'image_filename', 'height_meters', 'wave_type', 'direction',
                'labeler_id', 'confidence', 'notes', 'label_timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Add empty rows for each image
            for image_file in sorted(image_files):
                writer.writerow({
                    'image_filename': image_file.name,
                    'height_meters': '',
                    'wave_type': '',
                    'direction': '',
                    'labeler_id': '',
                    'confidence': '1.0',
                    'notes': '',
                    'label_timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Created labeling template with {len(image_files)} images: {template_path}")
        return str(template_path)
    
    def validate_labels(self, labels_file: str = 'manual_labels.csv') -> Dict[str, Any]:
        """
        Validate manual labels for correctness.
        
        Args:
            labels_file: Path to labels CSV file
            
        Returns:
            Validation results dictionary
        """
        labels_path = self.labels_path / labels_file
        
        if not labels_path.exists():
            return {'valid': False, 'error': f'Labels file not found: {labels_path}'}
        
        validation_results = {
            'valid': True,
            'total_labels': 0,
            'valid_labels': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            with open(labels_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                    validation_results['total_labels'] += 1
                    row_valid = True
                    
                    # Check required fields
                    if not row.get('image_filename'):
                        validation_results['errors'].append(f"Row {row_num}: Missing image_filename")
                        row_valid = False
                    
                    # Validate height
                    try:
                        height = float(row.get('height_meters', ''))
                        if not (self.height_range[0] <= height <= self.height_range[1]):
                            validation_results['warnings'].append(
                                f"Row {row_num}: Height {height}m outside typical range {self.height_range}"
                            )
                    except (ValueError, TypeError):
                        validation_results['errors'].append(f"Row {row_num}: Invalid height_meters value")
                        row_valid = False
                    
                    # Validate wave type
                    wave_type = row.get('wave_type', '').strip()
                    if wave_type not in self.valid_wave_types:
                        validation_results['errors'].append(
                            f"Row {row_num}: Invalid wave_type '{wave_type}'. Must be one of {self.valid_wave_types}"
                        )
                        row_valid = False
                    
                    # Validate direction
                    direction = row.get('direction', '').strip()
                    if direction not in self.valid_directions:
                        validation_results['errors'].append(
                            f"Row {row_num}: Invalid direction '{direction}'. Must be one of {self.valid_directions}"
                        )
                        row_valid = False
                    
                    # Validate confidence
                    try:
                        confidence = float(row.get('confidence', '1.0'))
                        if not (0.0 <= confidence <= 1.0):
                            validation_results['warnings'].append(
                                f"Row {row_num}: Confidence {confidence} outside range [0.0, 1.0]"
                            )
                    except (ValueError, TypeError):
                        validation_results['warnings'].append(f"Row {row_num}: Invalid confidence value")
                    
                    # Check if image file exists
                    if row.get('image_filename'):
                        image_path = self.images_path / row['image_filename']
                        if not image_path.exists():
                            validation_results['errors'].append(
                                f"Row {row_num}: Image file not found: {row['image_filename']}"
                            )
                            row_valid = False
                    
                    if row_valid:
                        validation_results['valid_labels'] += 1
        
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Error reading labels file: {e}")
        
        # Overall validation result
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        logger.info(f"Label validation: {validation_results['valid_labels']}/{validation_results['total_labels']} valid")
        if validation_results['errors']:
            logger.error(f"Found {len(validation_results['errors'])} validation errors")
        if validation_results['warnings']:
            logger.warning(f"Found {len(validation_results['warnings'])} validation warnings")
        
        return validation_results
    
    def get_labeling_statistics(self, labels_file: str = 'manual_labels.csv') -> Dict[str, Any]:
        """
        Get statistics about manual labels.
        
        Args:
            labels_file: Path to labels CSV file
            
        Returns:
            Statistics dictionary
        """
        labels_path = self.labels_path / labels_file
        
        if not labels_path.exists():
            return {'error': f'Labels file not found: {labels_path}'}
        
        stats = {
            'total_images': 0,
            'labeled_images': 0,
            'unlabeled_images': 0,
            'wave_type_distribution': {},
            'direction_distribution': {},
            'height_statistics': {},
            'labeler_distribution': {}
        }
        
        heights = []
        wave_types = []
        directions = []
        labelers = []
        
        try:
            with open(labels_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    stats['total_images'] += 1
                    
                    # Check if row is labeled (has required fields)
                    if (row.get('height_meters') and 
                        row.get('wave_type') and 
                        row.get('direction')):
                        
                        stats['labeled_images'] += 1
                        
                        try:
                            heights.append(float(row['height_meters']))
                            wave_types.append(row['wave_type'])
                            directions.append(row['direction'])
                            labelers.append(row.get('labeler_id', 'unknown'))
                        except ValueError:
                            pass  # Skip invalid numeric values
                    else:
                        stats['unlabeled_images'] += 1
        
        except Exception as e:
            stats['error'] = f"Error reading labels file: {e}"
            return stats
        
        # Calculate distributions
        if wave_types:
            stats['wave_type_distribution'] = {
                wt: wave_types.count(wt) for wt in set(wave_types)
            }
        
        if directions:
            stats['direction_distribution'] = {
                d: directions.count(d) for d in set(directions)
            }
        
        if heights:
            stats['height_statistics'] = {
                'min': min(heights),
                'max': max(heights),
                'mean': sum(heights) / len(heights),
                'count': len(heights)
            }
        
        if labelers:
            stats['labeler_distribution'] = {
                labeler: labelers.count(labeler) for labeler in set(labelers)
            }
        
        return stats
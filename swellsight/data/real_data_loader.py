"""
Real data loader for beach camera images with manual labels.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RealBeachDataset(Dataset):
    """Dataset for real beach images with manual labels."""
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize real beach dataset.
        
        Args:
            data_dir: Path to data/real directory
            transform: Optional image transforms
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_file = self.data_dir / "labels" / "labels.json"
        self.transform = transform or self._default_transform()
        
        # Load labels
        with open(self.labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Get list of images that have labels
        self.image_files = [
            img for img in os.listdir(self.images_dir)
            if img in self.labels and img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Define class mappings
        self.wave_type_to_idx = {
            'a_frame': 0,
            'closeout': 1, 
            'beach_break': 2,
            'point_break': 3
        }
        
        self.direction_to_idx = {
            'left': 0,
            'right': 1,
            'both': 2
        }
    
    def _default_transform(self) -> transforms.Compose:
        """Default image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get image and labels.
        
        Returns:
            image: Preprocessed image tensor
            labels: Dictionary with height, wave_type, direction
        """
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        label_data = self.labels[img_name]
        
        labels = {
            'height': torch.tensor(label_data['height_meters'], dtype=torch.float32),
            'wave_type': torch.tensor(self.wave_type_to_idx[label_data['wave_type']], dtype=torch.long),
            'direction': torch.tensor(self.direction_to_idx[label_data['direction']], dtype=torch.long),
            'filename': img_name
        }
        
        return image, labels
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        heights = [self.labels[img]['height_meters'] for img in self.image_files]
        wave_types = [self.labels[img]['wave_type'] for img in self.image_files]
        directions = [self.labels[img]['direction'] for img in self.image_files]
        
        return {
            'total_images': len(self.image_files),
            'height_range': (min(heights), max(heights)),
            'height_mean': sum(heights) / len(heights),
            'wave_type_distribution': {wt: wave_types.count(wt) for wt in set(wave_types)},
            'direction_distribution': {d: directions.count(d) for d in set(directions)}
        }


def load_real_data(data_dir: str, batch_size: int = 16) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for real beach data.
    
    Args:
        data_dir: Path to data/real directory
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader for real beach images
    """
    dataset = RealBeachDataset(data_dir)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=2,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the real data loader
    data_dir = "data/real"
    
    if os.path.exists(os.path.join(data_dir, "labels", "labels.json")):
        dataset = RealBeachDataset(data_dir)
        print(f"Loaded {len(dataset)} real beach images")
        print("Dataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test loading first sample
        if len(dataset) > 0:
            image, labels = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Image shape: {image.shape}")
            print(f"  Height: {labels['height'].item():.2f}m")
            print(f"  Wave type: {labels['wave_type'].item()}")
            print(f"  Direction: {labels['direction'].item()}")
    else:
        print(f"Labels file not found at {data_dir}/labels/labels.json")
        print("Please collect real data first!")
#!/usr/bin/env python3
"""Generate synthetic training dataset for SwellSight Wave Analysis Model."""

import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from swellsight.data.synthetic_data_generator import SyntheticDataGenerator
from swellsight.data.dataset_manager import DatasetManager
from swellsight.config.data_config import DataConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_dataset(
    num_samples: int = 10000,
    output_dir: str = "data",
    train_split: float = 0.8,
    val_split: float = 0.2
) -> Dict[str, Any]:
    """
    Generate synthetic training dataset with proper train/validation splits.
    
    Args:
        num_samples: Number of synthetic samples to generate
        output_dir: Base output directory for data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
    
    Returns:
        Dataset generation statistics
    """
    logger.info(f"Starting synthetic dataset generation...")
    logger.info(f"Target samples: {num_samples}")
    logger.info(f"Train/Val split: {train_split:.1%}/{val_split:.1%}")
    
    # Set up paths
    base_path = Path(output_dir)
    synthetic_path = base_path / "synthetic"
    metadata_path = base_path / "metadata"
    
    # Create directories
    base_path.mkdir(parents=True, exist_ok=True)
    synthetic_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    # Configure data generation
    config = {
        'synthetic_data_path': str(synthetic_path),
        'metadata_path': str(metadata_path),
        'image_size': (768, 768)  # Full resolution for training
    }
    
    # Initialize generator
    generator = SyntheticDataGenerator(config)
    
    # Generate dataset
    logger.info("Generating synthetic samples...")
    samples_metadata = generator.generate_dataset(num_samples)
    
    if not samples_metadata:
        raise RuntimeError("Failed to generate any samples")
    
    logger.info(f"Successfully generated {len(samples_metadata)} samples")
    
    # Create dataset manager for train/val splits
    data_config = DataConfig()
    data_config.synthetic_data_path = str(synthetic_path)
    data_config.metadata_path = str(metadata_path)
    data_config.train_split = train_split
    data_config.val_split = val_split
    
    dataset_manager = DatasetManager(str(base_path), data_config.to_dict())
    
    # Get dataset info and splits
    dataset_info = dataset_manager.get_dataset_info()
    
    # Analyze parameter diversity
    parameter_stats = analyze_parameter_diversity(samples_metadata)
    
    # Create comprehensive dataset report
    dataset_report = {
        'generation_info': {
            'total_samples_requested': num_samples,
            'total_samples_generated': len(samples_metadata),
            'success_rate': len(samples_metadata) / num_samples,
            'output_directory': str(base_path),
            'image_size': config['image_size']
        },
        'dataset_splits': dataset_info,
        'parameter_diversity': parameter_stats,
        'file_paths': {
            'synthetic_data': str(synthetic_path),
            'metadata': str(metadata_path),
            'dataset_metadata': str(metadata_path / "synthetic_dataset_metadata.json")
        }
    }
    
    # Save dataset report
    report_path = metadata_path / "dataset_generation_report.json"
    with open(report_path, 'w') as f:
        json.dump(dataset_report, f, indent=2)
    
    logger.info(f"Dataset generation complete!")
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SYNTHETIC DATASET GENERATION SUMMARY")
    print("="*60)
    print(f"Total samples generated: {len(samples_metadata):,}")
    print(f"Training samples: {dataset_info.get('synthetic_train', 0):,}")
    print(f"Validation samples: {dataset_info.get('synthetic_val', 0):,}")
    print(f"Success rate: {dataset_report['generation_info']['success_rate']:.1%}")
    print(f"\nParameter Diversity:")
    print(f"  Height range: {parameter_stats['height_range']['min']:.1f}m - {parameter_stats['height_range']['max']:.1f}m")
    print(f"  Wave types: {len(parameter_stats['wave_types'])} types")
    print(f"  Directions: {len(parameter_stats['directions'])} directions")
    print(f"\nOutput directory: {base_path}")
    print("="*60)
    
    return dataset_report


def analyze_parameter_diversity(samples_metadata: list) -> Dict[str, Any]:
    """
    Analyze parameter diversity across generated samples.
    
    Args:
        samples_metadata: List of sample metadata dictionaries
    
    Returns:
        Parameter diversity statistics
    """
    heights = [sample['height_meters'] for sample in samples_metadata]
    wave_types = [sample['wave_type'] for sample in samples_metadata]
    directions = [sample['direction'] for sample in samples_metadata]
    
    # Count occurrences
    wave_type_counts = {}
    direction_counts = {}
    
    for wave_type in wave_types:
        wave_type_counts[wave_type] = wave_type_counts.get(wave_type, 0) + 1
    
    for direction in directions:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    
    return {
        'height_range': {
            'min': min(heights),
            'max': max(heights),
            'mean': sum(heights) / len(heights),
            'std': (sum((h - sum(heights) / len(heights))**2 for h in heights) / len(heights))**0.5
        },
        'wave_types': wave_type_counts,
        'directions': direction_counts,
        'total_samples': len(samples_metadata)
    }


def main():
    """Main entry point for dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training dataset for SwellSight Wave Analysis Model"
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=10000,
        help='Number of synthetic samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated data (default: data)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Fraction of data for validation (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Validate splits
    if abs(args.train_split + args.val_split - 1.0) > 1e-6:
        raise ValueError("Train and validation splits must sum to 1.0")
    
    try:
        # Generate dataset
        dataset_report = generate_training_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        logger.info("Dataset generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
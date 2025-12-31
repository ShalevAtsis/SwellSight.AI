#!/usr/bin/env python3
"""
Demonstration script for Task 14.4: Depth Map Storage and Retrieval System

This script demonstrates the complete implementation of the depth map storage system
including:
- Efficient storage formats (compressed numpy arrays)
- Metadata tracking for depth extraction parameters  
- Depth map versioning for different MiDaS model versions
- Utilities for depth map visualization and debugging

Requirements validated: 8.1, 8.2, 10.2
"""

import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor
from swellsight.data.depth_map_storage import DepthMapStorage, DepthMapMetadata, DepthMapVisualizer


def create_demo_images(output_dir: Path) -> list[Path]:
    """Create demo beach images for testing."""
    images = []
    
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
        image_path = output_dir / f'demo_beach_{i:03d}.jpg'
        image.save(image_path, 'JPEG')
        images.append(image_path)
        
        print(f"Created demo image: {image_path.name}")
    
    return images


def demonstrate_storage_formats(storage_dir: Path, test_depth_map: np.ndarray):
    """Demonstrate different storage formats and their efficiency."""
    print("\n=== Storage Format Comparison ===")
    
    formats = ['compressed_npy', 'gzip_pickle', 'npy']
    results = {}
    
    for fmt in formats:
        format_dir = storage_dir / fmt
        format_dir.mkdir(exist_ok=True)
        
        storage = DepthMapStorage(str(format_dir), {'storage_format': fmt})
        
        metadata = DepthMapMetadata(
            depth_map_id=f"format_demo_{fmt}",
            original_image_path=f"/demo/format_test_{fmt}.jpg",
            image_filename=f"format_test_{fmt}.jpg",
            midas_model_version="Intel/dpt-large",
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=test_depth_map.shape,
            depth_range=(float(test_depth_map.min()), float(test_depth_map.max())),
            quality_score=0.8,
            storage_format=fmt
        )
        
        # Store depth map
        depth_map_id = storage.store_depth_map(test_depth_map, metadata)
        
        # Retrieve and check integrity
        retrieved_depth, retrieved_metadata = storage.retrieve_depth_map(depth_map_id)
        
        # Calculate metrics
        original_size = test_depth_map.nbytes
        compressed_size = retrieved_metadata.file_size_bytes or 0
        compression_ratio = retrieved_metadata.compression_ratio or 1.0
        
        results[fmt] = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'integrity_check': np.allclose(test_depth_map, retrieved_depth, rtol=1e-5)
        }
        
        print(f"Format: {fmt}")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Data integrity: {'✓' if results[fmt]['integrity_check'] else '✗'}")
    
    return results


def demonstrate_versioning_and_metadata(storage: DepthMapStorage, test_depth_map: np.ndarray):
    """Demonstrate versioning and metadata tracking."""
    print("\n=== Versioning and Metadata Tracking ===")
    
    image_filename = "versioning_demo.jpg"
    model_versions = ["Intel/dpt-large", "Intel/dpt-hybrid-midas", "Intel/dpt-small"]
    
    stored_ids = []
    
    # Store multiple versions
    for i, model_version in enumerate(model_versions):
        metadata = DepthMapMetadata(
            depth_map_id=f"version_demo_{i:03d}",
            original_image_path=f"/demo/{image_filename}",
            image_filename=image_filename,
            midas_model_version=model_version,
            extraction_timestamp=datetime.now().isoformat(),
            depth_map_shape=test_depth_map.shape,
            depth_range=(float(test_depth_map.min()), float(test_depth_map.max())),
            quality_score=0.7 + i * 0.1,
            storage_format="compressed_npy"
        )
        
        depth_map_id = storage.store_depth_map(test_depth_map, metadata)
        stored_ids.append(depth_map_id)
        
        print(f"Stored version {i+1}: {model_version} (ID: {depth_map_id})")
    
    # Demonstrate search capabilities
    print(f"\nSearch by image filename '{image_filename}':")
    found_by_image = storage.find_by_image(image_filename)
    for meta in found_by_image:
        print(f"  Found: {meta.depth_map_id} - {meta.midas_model_version} (Quality: {meta.quality_score:.2f})")
    
    print(f"\nSearch by model version 'Intel/dpt-large':")
    found_by_model = storage.find_by_model_version("Intel/dpt-large")
    for meta in found_by_model:
        print(f"  Found: {meta.depth_map_id} - {meta.image_filename}")
    
    print(f"\nSearch by quality range (>= 0.8):")
    found_by_quality = storage.find_by_quality_range(0.8)
    for meta in found_by_quality:
        print(f"  Found: {meta.depth_map_id} - Quality: {meta.quality_score:.2f}")
    
    return stored_ids


def demonstrate_visualization(storage: DepthMapStorage, depth_map_ids: list[str], output_dir: Path):
    """Demonstrate visualization utilities."""
    print("\n=== Visualization and Debugging Utilities ===")
    
    visualizer = DepthMapVisualizer(storage)
    
    # Create single depth map visualization
    if depth_map_ids:
        single_viz_path = output_dir / "single_depth_visualization.png"
        visualizer.create_depth_visualization(depth_map_ids[0], str(single_viz_path))
        print(f"Created single visualization: {single_viz_path}")
        
        # Create comparison visualization if multiple depth maps
        if len(depth_map_ids) > 1:
            comparison_viz_path = output_dir / "comparison_visualization.png"
            visualizer.create_comparison_visualization(depth_map_ids[:3], str(comparison_viz_path))
            print(f"Created comparison visualization: {comparison_viz_path}")


def demonstrate_midas_integration(images: list[Path], storage_dir: Path):
    """Demonstrate MiDaS integration with storage system."""
    print("\n=== MiDaS Integration with Storage ===")
    
    # Initialize MiDaS extractor with storage
    extractor = MiDaSDepthExtractor(
        model_name="Intel/dpt-large",
        storage_path=str(storage_dir / "midas_integration")
    )
    
    print("Processing images with MiDaS and storing depth maps...")
    
    # Process each image
    for i, image_path in enumerate(images):
        print(f"\nProcessing image {i+1}: {image_path.name}")
        
        # Extract depth with storage
        result = extractor.extract_depth(str(image_path), store_result=True)
        
        print(f"  Depth map shape: {result.depth_map.shape}")
        print(f"  Quality score: {result.depth_quality_score:.3f}")
        print(f"  Depth range: {result.processing_metadata['depth_range'][0]:.1f} - {result.processing_metadata['depth_range'][1]:.1f}m")
    
    # Show storage statistics
    stats = extractor.get_storage_statistics()
    print(f"\nStorage Statistics:")
    print(f"  Total depth maps: {stats['total_depth_maps']}")
    print(f"  Storage format: {stats['storage_format']}")
    
    if stats['model_statistics']:
        for model_stat in stats['model_statistics']:
            print(f"  Model {model_stat['model_version']}: {model_stat['count']} depth maps, avg quality: {model_stat['avg_quality']:.3f}")
    
    return extractor


def demonstrate_export_and_analysis(storage: DepthMapStorage, output_dir: Path):
    """Demonstrate metadata export and analysis."""
    print("\n=== Metadata Export and Analysis ===")
    
    # Export metadata
    export_path = output_dir / "depth_map_metadata.json"
    storage.export_metadata(str(export_path))
    
    print(f"Exported metadata to: {export_path}")
    
    # Load and analyze exported data
    with open(export_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"Exported {len(metadata_list)} depth map records")
    
    # Analyze quality distribution
    qualities = [meta['quality_score'] for meta in metadata_list]
    if qualities:
        print(f"Quality scores - Min: {min(qualities):.3f}, Max: {max(qualities):.3f}, Avg: {sum(qualities)/len(qualities):.3f}")
    
    # Analyze model versions
    models = [meta['midas_model_version'] for meta in metadata_list]
    model_counts = {}
    for model in models:
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print("Model version distribution:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} depth maps")


def main():
    """Main demonstration function."""
    print("=== Task 14.4: Depth Map Storage and Retrieval System Demo ===")
    print("Demonstrating comprehensive depth map storage with:")
    print("- Efficient storage formats (compressed numpy arrays)")
    print("- Metadata tracking for depth extraction parameters")
    print("- Depth map versioning for different MiDaS model versions")
    print("- Utilities for depth map visualization and debugging")
    print("- Requirements: 8.1, 8.2, 10.2")
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    images_dir = temp_dir / "images"
    storage_dir = temp_dir / "storage"
    output_dir = temp_dir / "output"
    
    for dir_path in [images_dir, storage_dir, output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # Create demo images
        images = create_demo_images(images_dir)
        
        # Create test depth map for demonstrations
        test_depth_map = np.random.rand(100, 150).astype(np.float32) * 50 + 1
        
        # Demonstrate storage formats
        format_results = demonstrate_storage_formats(storage_dir, test_depth_map)
        
        # Initialize main storage system
        main_storage = DepthMapStorage(str(storage_dir / "main"))
        
        # Demonstrate versioning and metadata
        stored_ids = demonstrate_versioning_and_metadata(main_storage, test_depth_map)
        
        # Demonstrate visualization
        demonstrate_visualization(main_storage, stored_ids, output_dir)
        
        # Demonstrate MiDaS integration
        extractor = demonstrate_midas_integration(images, storage_dir)
        
        # Demonstrate export and analysis
        demonstrate_export_and_analysis(main_storage, output_dir)
        
        print(f"\n=== Demo Complete ===")
        print(f"All files created in: {temp_dir}")
        print("Task 14.4 implementation successfully demonstrated!")
        
        # Show final statistics
        final_stats = main_storage.get_storage_statistics()
        print(f"\nFinal Storage Statistics:")
        print(f"  Total depth maps stored: {final_stats['total_depth_maps']}")
        print(f"  Storage efficiency: {final_stats['storage_efficiency']['avg_compression_ratio']:.2f}x compression")
        print(f"  Total storage used: {final_stats['storage_efficiency']['total_storage_bytes']:,} bytes")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
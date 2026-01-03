#!/usr/bin/env python3
"""
Simple script to generate depth maps from real beach images using MiDaS.

Usage:
    python generate_depth_maps.py --input data/real/images --output data/depth_maps
"""

import argparse
import logging
from pathlib import Path
import glob
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate depth maps from real beach images')
    parser.add_argument('--input', '-i', required=True, help='Input directory with beach images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for depth maps')
    parser.add_argument('--model', default='Intel/dpt-large', 
                       choices=['Intel/dpt-large', 'Intel/dpt-hybrid-midas', 'Intel/dpt-base'],
                       help='MiDaS model to use')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use (auto-detect if not specified)')
    parser.add_argument('--quality-threshold', type=float, default=0.3, 
                       help='Minimum quality threshold (0.0-1.0)')
    parser.add_argument('--format', choices=['npy', 'png', 'tiff'], default='npy',
                       help='Output format for depth maps')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return 1
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(ext))
        image_paths.extend(input_path.glob(ext.upper()))
    
    if not image_paths:
        logger.error(f"No image files found in {input_path}")
        return 1
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Initialize MiDaS depth extractor
    logger.info(f"Initializing MiDaS depth extractor with model: {args.model}")
    try:
        extractor = MiDaSDepthExtractor(
            model_name=args.model,
            device=args.device,
            storage_path=str(output_path) if args.format == 'npy' else None
        )
    except Exception as e:
        logger.error(f"Failed to initialize MiDaS extractor: {e}")
        return 1
    
    # Process images
    successful = 0
    failed = 0
    low_quality = 0
    
    for i, image_path in enumerate(image_paths):
        try:
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            
            # Extract depth map
            result = extractor.extract_depth(str(image_path), store_result=False)
            
            # Check quality
            if result.depth_quality_score < args.quality_threshold:
                logger.warning(f"Low quality depth map (score: {result.depth_quality_score:.3f}), skipping")
                low_quality += 1
                continue
            
            # Save depth map
            output_filename = f"{image_path.stem}_depth.{args.format}"
            output_file = output_path / output_filename
            
            extractor.save_depth_map(result.depth_map, str(output_file), format=args.format)
            
            logger.info(f"Saved depth map: {output_file} (quality: {result.depth_quality_score:.3f})")
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            failed += 1
            continue
    
    # Summary
    logger.info(f"Processing complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Low quality: {low_quality}")
    logger.info(f"  Total processed: {successful + failed + low_quality}")
    
    return 0


if __name__ == '__main__':
    exit(main())
#!/usr/bin/env python3
"""
Real Images to Synthetic Images Pipeline

This script performs exactly what you requested:
1. Use MiDaS model to extract depth maps from real images in data/real/images
2. Perform data augmentation on these depth maps
3. Produce synthetic images from these depth maps using ControlNet

Usage:
    python real_to_synthetic_pipeline.py --input data/real/images --output data/synthetic
"""

import argparse
import logging
from pathlib import Path
import glob
import json
import numpy as np
from datetime import datetime

from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor
from swellsight.data.controlnet_generator import ControlNetSyntheticGenerator, AugmentationParameterSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Real Images to Synthetic Images Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input directory with real beach images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for synthetic images')
    parser.add_argument('--depth-output', default='data/depth_maps', help='Output directory for depth maps')
    parser.add_argument('--midas-model', default='Intel/dpt-large', 
                       choices=['Intel/dpt-large', 'Intel/dpt-hybrid-midas', 'Intel/dpt-base'],
                       help='MiDaS model to use')
    parser.add_argument('--controlnet-model', default='lllyasviel/sd-controlnet-depth',
                       help='ControlNet model to use')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use (auto-detect if not specified)')
    parser.add_argument('--quality-threshold', type=float, default=0.3, 
                       help='Minimum depth quality threshold (0.0-1.0)')
    parser.add_argument('--synthetic-per-real', type=int, default=3,
                       help='Number of synthetic images to generate per real image')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images to process (default: 50)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    depth_output_path = Path(args.depth_output)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    depth_output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'labels').mkdir(exist_ok=True)
    
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
    
    # Limit number of images to process
    if len(image_paths) > args.max_images:
        logger.info(f"Limiting processing to {args.max_images} images (found {len(image_paths)})")
        image_paths = image_paths[:args.max_images]
    
    logger.info(f"Processing {len(image_paths)} real images")
    
    # STEP 1: Initialize MiDaS depth extractor
    logger.info("Step 1: Initializing MiDaS depth extractor...")
    try:
        depth_extractor = MiDaSDepthExtractor(
            model_name=args.midas_model,
            device=args.device,
            storage_path=str(depth_output_path)
        )
        logger.info(f"MiDaS depth extractor initialized with model: {args.midas_model}")
    except Exception as e:
        logger.error(f"Failed to initialize MiDaS extractor: {e}")
        return 1
    
    # STEP 2: Initialize augmentation parameter system
    logger.info("Step 2: Initializing augmentation parameter system...")
    try:
        param_system = AugmentationParameterSystem()
        logger.info("Augmentation parameter system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize augmentation system: {e}")
        return 1
    
    # STEP 3: Initialize ControlNet generator
    logger.info("Step 3: Initializing ControlNet synthetic generator...")
    try:
        controlnet_config = {
            'controlnet_model_name': args.controlnet_model,
            'device': args.device,
            'batch_size': args.batch_size,
            'guidance_scale': 7.5,
            'num_inference_steps': 20,
            'controlnet_conditioning_scale': 1.0
        }
        
        controlnet_generator = ControlNetSyntheticGenerator(controlnet_config)
        logger.info(f"ControlNet generator initialized with model: {args.controlnet_model}")
    except Exception as e:
        logger.error(f"Failed to initialize ControlNet generator: {e}")
        return 1
    
    # Process each real image
    total_synthetic_generated = 0
    successful_real_images = 0
    failed_real_images = 0
    
    synthetic_metadata = []
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"\nProcessing real image {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            # STEP 1: Extract depth map from real image
            logger.info("  Step 1: Extracting depth map with MiDaS...")
            depth_result = depth_extractor.extract_depth(str(image_path), store_result=True)
            
            # Check depth quality
            if depth_result.depth_quality_score < args.quality_threshold:
                logger.warning(f"  Low quality depth map (score: {depth_result.depth_quality_score:.3f}), skipping")
                failed_real_images += 1
                continue
            
            logger.info(f"  Depth extraction successful (quality: {depth_result.depth_quality_score:.3f})")
            
            # STEP 2: Generate augmented parameters for synthetic variations
            logger.info(f"  Step 2: Generating {args.synthetic_per_real} augmented parameter sets...")
            
            for j in range(args.synthetic_per_real):
                # Generate augmented parameters
                augmentation_params = param_system.generate_random_parameters()
                
                logger.info(f"    Generated augmentation parameters for variation {j+1}")
                
                # STEP 3: Generate synthetic image using ControlNet
                logger.info(f"  Step 3: Generating synthetic image {j+1}/{args.synthetic_per_real}...")
                
                try:
                    synthetic_result = controlnet_generator.generate_synthetic_image(
                        depth_map=depth_result.depth_map,
                        augmentation_params=augmentation_params
                    )
                    
                    # Save synthetic image
                    synthetic_filename = f"{image_path.stem}_synthetic_{j+1:03d}.jpg"
                    synthetic_image_path = output_path / 'images' / synthetic_filename
                    
                    # Convert numpy array to PIL Image and save
                    if isinstance(synthetic_result.synthetic_image, np.ndarray):
                        # Ensure the array is in the right format (0-255, uint8)
                        if synthetic_result.synthetic_image.max() <= 1.0:
                            synthetic_image_array = (synthetic_result.synthetic_image * 255).astype(np.uint8)
                        else:
                            synthetic_image_array = synthetic_result.synthetic_image.astype(np.uint8)
                        
                        from PIL import Image
                        synthetic_image_pil = Image.fromarray(synthetic_image_array)
                        synthetic_image_pil.save(synthetic_image_path)
                    else:
                        # If it's already a PIL Image
                        synthetic_result.synthetic_image.save(synthetic_image_path)
                    
                    # Create metadata
                    sample_metadata = {
                        'sample_id': f"real_to_synthetic_{total_synthetic_generated:06d}",
                        'original_real_image': str(image_path),
                        'synthetic_image_path': str(synthetic_image_path),
                        'depth_map_quality': depth_result.depth_quality_score,
                        'augmentation_params': augmentation_params.__dict__ if hasattr(augmentation_params, '__dict__') else augmentation_params,
                        'generation_metadata': synthetic_result.generation_metadata,
                        'quality_score': synthetic_result.quality_score,
                        'created_timestamp': datetime.now().isoformat(),
                        'pipeline_version': '1.0'
                    }
                    
                    synthetic_metadata.append(sample_metadata)
                    total_synthetic_generated += 1
                    
                    logger.info(f"    Synthetic image saved: {synthetic_image_path}")
                
                except Exception as e:
                    logger.error(f"    Error generating synthetic image {j+1}: {e}")
                    continue
            
            successful_real_images += 1
            logger.info(f"  Successfully processed real image: {image_path.name}")
            
        except Exception as e:
            logger.error(f"  Failed to process real image {image_path}: {e}")
            failed_real_images += 1
            continue
    
    # Save metadata
    metadata_file = output_path / 'synthetic_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(synthetic_metadata, f, indent=2)
    
    # Generate summary report
    summary = {
        'pipeline_execution': {
            'timestamp': datetime.now().isoformat(),
            'input_path': str(input_path),
            'output_path': str(output_path),
            'midas_model': args.midas_model,
            'controlnet_model': args.controlnet_model
        },
        'processing_results': {
            'total_real_images': len(image_paths),
            'successful_real_images': successful_real_images,
            'failed_real_images': failed_real_images,
            'total_synthetic_generated': total_synthetic_generated,
            'synthetic_per_real_target': args.synthetic_per_real
        },
        'quality_metrics': {
            'depth_quality_threshold': args.quality_threshold,
            'success_rate': successful_real_images / len(image_paths) if image_paths else 0,
            'synthetic_generation_rate': total_synthetic_generated / (successful_real_images * args.synthetic_per_real) if successful_real_images > 0 else 0
        }
    }
    
    summary_file = output_path / 'pipeline_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("REAL TO SYNTHETIC PIPELINE COMPLETED")
    print("="*60)
    print(f"Real images processed: {successful_real_images}/{len(image_paths)}")
    print(f"Synthetic images generated: {total_synthetic_generated}")
    print(f"Success rate: {successful_real_images/len(image_paths)*100:.1f}%")
    print(f"Output directory: {output_path}")
    print(f"Depth maps saved to: {depth_output_path}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Summary saved to: {summary_file}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
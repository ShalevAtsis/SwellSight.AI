"""Inference script for SwellSight Wave Analysis Model."""

import argparse
import json
import logging
from pathlib import Path

from swellsight.inference import InferenceEngine


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Run inference with SwellSight Wave Analysis Model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info(f"Processing image: {image_path}")
    
    # Load inference engine
    engine = InferenceEngine.from_checkpoint(str(checkpoint_path), device=args.device)
    
    # Run inference
    result = engine.predict(str(image_path))
    
    # Format output
    output_data = {
        "image_path": str(image_path),
        "predictions": {
            "wave_height_m": result.height_meters,
            "wave_type": result.wave_type,
            "direction": result.direction,
            "probabilities": {
                "wave_type": result.wave_type_probs,
                "direction": result.direction_probs
            },
            "confidence_scores": result.confidence_scores
        }
    }
    
    # Save or print results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    else:
        print(json.dumps(output_data, indent=2))
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
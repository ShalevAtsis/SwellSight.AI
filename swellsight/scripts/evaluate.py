"""Evaluation script for SwellSight Wave Analysis Model."""

import argparse
import json
import logging
from pathlib import Path

from swellsight.inference import InferenceEngine
from swellsight.evaluation import MetricsCalculator
from swellsight.data import DatasetManager


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate SwellSight Wave Analysis Model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--dataset-type", choices=["synthetic", "real"], default="synthetic", 
                       help="Type of dataset to evaluate on")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info(f"Evaluating on {args.dataset_type} dataset: {data_path}")
    
    # Load inference engine
    engine = InferenceEngine.from_checkpoint(str(checkpoint_path), device=args.device)
    
    # Load dataset
    dataset_manager = DatasetManager(str(data_path))
    if args.dataset_type == "synthetic":
        dataloader = dataset_manager.get_validation_loader(batch_size=args.batch_size)
    else:
        dataloader = dataset_manager.get_real_test_loader(batch_size=args.batch_size)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    logger.info("Running evaluation...")
    
    # Collect predictions and ground truth
    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        
        # Run inference on batch
        predictions = engine.predict_batch(images)
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = metrics_calc.calculate_all_metrics(all_predictions, all_targets)
    
    # Save detailed results
    results = {
        "dataset_type": args.dataset_type,
        "dataset_path": str(data_path),
        "checkpoint_path": str(checkpoint_path),
        "num_samples": len(all_predictions),
        "metrics": metrics
    }
    
    # Save results
    results_file = output_dir / f"evaluation_results_{args.dataset_type}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    metrics_calc.generate_report(
        all_predictions, 
        all_targets, 
        output_dir / f"evaluation_report_{args.dataset_type}.html"
    )
    
    # Print summary
    print(f"\nEvaluation Results ({args.dataset_type} dataset):")
    print(f"Number of samples: {len(all_predictions)}")
    print(f"\nHeight Regression:")
    print(f"  MAE: {metrics['height']['mae']:.3f} meters")
    print(f"  RMSE: {metrics['height']['rmse']:.3f} meters")
    print(f"\nWave Type Classification:")
    print(f"  Accuracy: {metrics['wave_type']['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['wave_type']['f1_score']:.3f}")
    print(f"\nDirection Classification:")
    print(f"  Accuracy: {metrics['direction']['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['direction']['f1_score']:.3f}")
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
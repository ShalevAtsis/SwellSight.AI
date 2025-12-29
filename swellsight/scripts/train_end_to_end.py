#!/usr/bin/env python3
"""
End-to-end training script for SwellSight Wave Analysis Model.

This script wires together all components: data generation, training, and evaluation
to provide a complete training pipeline with command-line interface, progress tracking,
and comprehensive logging.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# SwellSight imports
from swellsight.config import ModelConfig, TrainingConfig, DataConfig
from swellsight.models import WaveAnalysisModel
from swellsight.data import SyntheticDataGenerator, DatasetManager
from swellsight.training import Trainer
from swellsight.evaluation import MetricsCalculator
from swellsight.utils.device_utils import get_device


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Set up comprehensive logging for the training pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger('swellsight').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def generate_synthetic_data(
    data_config: DataConfig,
    output_dir: Path,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Generate synthetic training dataset if needed.
    
    Args:
        data_config: Data configuration
        output_dir: Output directory for data
        force_regenerate: Force regeneration even if data exists
    
    Returns:
        Dataset generation statistics
    """
    logger = logging.getLogger(__name__)
    
    synthetic_path = output_dir / "synthetic"
    metadata_path = output_dir / "metadata"
    
    # Check if data already exists
    dataset_report_path = metadata_path / "dataset_generation_report.json"
    if dataset_report_path.exists() and not force_regenerate:
        logger.info("Synthetic dataset already exists, loading existing data...")
        with open(dataset_report_path, 'r') as f:
            return json.load(f)
    
    logger.info("Generating synthetic training dataset...")
    
    # Create directories
    synthetic_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    # Configure generator
    generator_config = {
        'synthetic_data_path': str(synthetic_path),
        'metadata_path': str(metadata_path),
        'image_size': data_config.image_size
    }
    
    # Generate dataset
    generator = SyntheticDataGenerator(generator_config)
    samples_metadata = generator.generate_dataset(data_config.num_synthetic_samples)
    
    if not samples_metadata:
        raise RuntimeError("Failed to generate synthetic dataset")
    
    logger.info(f"Generated {len(samples_metadata)} synthetic samples")
    
    # Analyze parameter diversity
    heights = [sample['height_meters'] for sample in samples_metadata]
    wave_types = [sample['wave_type'] for sample in samples_metadata]
    directions = [sample['direction'] for sample in samples_metadata]
    
    wave_type_counts = {}
    direction_counts = {}
    
    for wave_type in wave_types:
        wave_type_counts[wave_type] = wave_type_counts.get(wave_type, 0) + 1
    
    for direction in directions:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    
    # Create dataset report
    dataset_report = {
        'generation_info': {
            'total_samples_requested': data_config.num_synthetic_samples,
            'total_samples_generated': len(samples_metadata),
            'success_rate': len(samples_metadata) / data_config.num_synthetic_samples,
            'generation_time': datetime.now().isoformat()
        },
        'parameter_diversity': {
            'height_range': {
                'min': min(heights),
                'max': max(heights),
                'mean': sum(heights) / len(heights)
            },
            'wave_types': wave_type_counts,
            'directions': direction_counts
        }
    }
    
    # Save report
    with open(dataset_report_path, 'w') as f:
        json.dump(dataset_report, f, indent=2)
    
    return dataset_report


def create_model(model_config: ModelConfig) -> WaveAnalysisModel:
    """
    Create and initialize the wave analysis model.
    
    Args:
        model_config: Model configuration
    
    Returns:
        Initialized model
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Creating WaveAnalysisModel...")
    logger.info(f"Backbone: {model_config.backbone}")
    logger.info(f"Input size: {model_config.input_size}")
    logger.info(f"Feature dim: {model_config.feature_dim}")
    
    model = WaveAnalysisModel(model_config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def setup_training(
    model: WaveAnalysisModel,
    training_config: TrainingConfig,
    data_config: DataConfig,
    output_dir: Path
) -> Trainer:
    """
    Set up the training pipeline.
    
    Args:
        model: Wave analysis model
        training_config: Training configuration
        data_config: Data configuration
        output_dir: Output directory for checkpoints
    
    Returns:
        Configured trainer
    """
    logger = logging.getLogger(__name__)
    
    # Create dataset manager
    dataset_manager = DatasetManager(
        data_path=str(output_dir),
        config=data_config.to_dict()
    )
    
    # Get dataset info
    dataset_info = dataset_manager.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        training_config=training_config,
        data_config=data_config,
        output_dir=output_dir / "checkpoints",
        dataset_manager=dataset_manager
    )
    
    return trainer


def run_evaluation(
    model: WaveAnalysisModel,
    dataset_manager: DatasetManager,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on the trained model.
    
    Args:
        model: Trained model
        dataset_manager: Dataset manager
        output_dir: Output directory for evaluation results
    
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Running model evaluation...")
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator()
    
    # Get validation data loader
    val_loader = dataset_manager.get_validation_loader(batch_size=32)
    
    # Run evaluation
    device = get_device()
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    import torch
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = {
                'height': batch['height'].to(device),
                'wave_type': batch['wave_type'].to(device),
                'direction': batch['direction'].to(device)
            }
            
            predictions = model(images)
            
            # Convert to CPU for metrics calculation
            pred_dict = {
                'height': predictions['height'].cpu().numpy(),
                'wave_type': torch.argmax(predictions['wave_type'], dim=1).cpu().numpy(),
                'direction': torch.argmax(predictions['direction'], dim=1).cpu().numpy()
            }
            
            target_dict = {
                'height': targets['height'].cpu().numpy(),
                'wave_type': targets['wave_type'].cpu().numpy(),
                'direction': targets['direction'].cpu().numpy()
            }
            
            all_predictions.append(pred_dict)
            all_targets.append(target_dict)
    
    # Combine all predictions and targets
    import numpy as np
    combined_predictions = {
        'height': np.concatenate([p['height'] for p in all_predictions]),
        'wave_type': np.concatenate([p['wave_type'] for p in all_predictions]),
        'direction': np.concatenate([p['direction'] for p in all_predictions])
    }
    
    combined_targets = {
        'height': np.concatenate([t['height'] for t in all_targets]),
        'wave_type': np.concatenate([t['wave_type'] for t in all_targets]),
        'direction': np.concatenate([t['direction'] for t in all_targets])
    }
    
    # Calculate metrics
    evaluation_results = metrics_calculator.calculate_comprehensive_metrics(
        predictions=combined_predictions,
        targets=combined_targets,
        dataset_type='synthetic_validation'
    )
    
    # Save evaluation results
    eval_results_path = output_dir / "evaluation_results.json"
    with open(eval_results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {eval_results_path}")
    
    # Log key metrics
    logger.info("Evaluation Results:")
    logger.info(f"  Height MAE: {evaluation_results['height_metrics']['mae']:.3f}m")
    logger.info(f"  Height RMSE: {evaluation_results['height_metrics']['rmse']:.3f}m")
    logger.info(f"  Wave Type Accuracy: {evaluation_results['wave_type_metrics']['accuracy']:.3f}")
    logger.info(f"  Direction Accuracy: {evaluation_results['direction_metrics']['accuracy']:.3f}")
    
    return evaluation_results


def save_training_summary(
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    dataset_report: Dict[str, Any],
    configs: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save comprehensive training summary.
    
    Args:
        training_results: Results from training
        evaluation_results: Results from evaluation
        dataset_report: Dataset generation report
        configs: All configuration objects
        output_dir: Output directory
    """
    summary = {
        'training_summary': {
            'status': training_results['status'],
            'epochs_trained': training_results['epochs_trained'],
            'early_stopped': training_results['early_stopped'],
            'total_training_time': training_results['total_training_time'],
            'best_validation_loss': training_results['best_val_loss']
        },
        'final_metrics': {
            'training': training_results['final_train_metrics'],
            'validation': training_results['final_val_metrics'],
            'evaluation': evaluation_results
        },
        'dataset_info': dataset_report,
        'configurations': configs,
        'completion_time': datetime.now().isoformat()
    }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training summary saved to: {summary_path}")


def main():
    """Main entry point for end-to-end training."""
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline for SwellSight Wave Analysis Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation arguments
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=10000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--force-regenerate-data',
        action='store_true',
        help='Force regeneration of synthetic data even if it exists'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='convnext_base',
        choices=['convnext_base', 'resnet50', 'efficientnet_b0'],
        help='Model backbone architecture'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Base output directory for all results'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated timestamp)'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable logging to file'
    )
    
    # Resume training
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Create experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"swellsight_training_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = None if args.no_log_file else output_dir / "training.log"
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("SWELLSIGHT WAVE ANALYSIS MODEL - END-TO-END TRAINING")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {get_device()}")
    
    try:
        # Create configurations
        model_config = ModelConfig(
            backbone=args.backbone,
            input_size=(768, 768),
            feature_dim=2048,
            hidden_dim=512
        )
        
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            checkpoint_frequency=10,
            early_stopping_patience=20
        )
        
        data_config = DataConfig(
            num_synthetic_samples=args.num_samples,
            synthetic_data_path=str(output_dir / "data" / "synthetic"),
            metadata_path=str(output_dir / "data" / "metadata"),
            train_split=0.8,
            val_split=0.2
        )
        
        # Save configurations
        configs = {
            'model_config': model_config.to_dict(),
            'training_config': training_config.to_dict(),
            'data_config': data_config.to_dict(),
            'command_line_args': vars(args)
        }
        
        config_path = output_dir / "configs.json"
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Configurations saved to: {config_path}")
        
        # Step 1: Generate synthetic data
        logger.info("\n" + "="*60)
        logger.info("STEP 1: SYNTHETIC DATA GENERATION")
        logger.info("="*60)
        
        dataset_report = generate_synthetic_data(
            data_config=data_config,
            output_dir=output_dir / "data",
            force_regenerate=args.force_regenerate_data
        )
        
        # Step 2: Create model
        logger.info("\n" + "="*60)
        logger.info("STEP 2: MODEL CREATION")
        logger.info("="*60)
        
        model = create_model(model_config)
        
        # Step 3: Set up training
        logger.info("\n" + "="*60)
        logger.info("STEP 3: TRAINING SETUP")
        logger.info("="*60)
        
        trainer = setup_training(
            model=model,
            training_config=training_config,
            data_config=data_config,
            output_dir=output_dir
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from: {args.resume_from}")
            trainer.load_checkpoint(Path(args.resume_from))
        
        # Step 4: Train model
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*60)
        
        start_time = time.time()
        training_results = trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.1f} seconds")
        logger.info(f"Status: {training_results['status']}")
        logger.info(f"Epochs trained: {training_results['epochs_trained']}")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        
        # Step 5: Evaluate model
        logger.info("\n" + "="*60)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("="*60)
        
        evaluation_results = run_evaluation(
            model=model,
            dataset_manager=trainer.dataset_manager,
            output_dir=output_dir
        )
        
        # Step 6: Save comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("="*60)
        
        save_training_summary(
            training_results=training_results,
            evaluation_results=evaluation_results,
            dataset_report=dataset_report,
            configs=configs,
            output_dir=output_dir
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Total time: {time.time() - start_time:.1f} seconds")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Best model: {output_dir / 'checkpoints' / 'best_model.pth'}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
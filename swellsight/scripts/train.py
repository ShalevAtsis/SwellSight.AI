"""Training script for SwellSight Wave Analysis Model."""

import argparse
import logging
from pathlib import Path

from swellsight.config import ModelConfig, TrainingConfig, DataConfig
from swellsight.models import WaveAnalysisModel
from swellsight.training import Trainer


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train SwellSight Wave Analysis Model")
    
    # Model configuration
    parser.add_argument("--backbone", default="convnext_base", help="Backbone architecture")
    parser.add_argument("--input-size", nargs=2, type=int, default=[768, 768], help="Input image size")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Data configuration
    parser.add_argument("--data-path", type=str, default="data", help="Data directory path")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic samples")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configurations
    model_config = ModelConfig(
        backbone=args.backbone,
        input_size=tuple(args.input_size)
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    data_config = DataConfig(
        synthetic_data_path=str(Path(args.data_path) / "synthetic"),
        real_data_path=str(Path(args.data_path) / "real"),
        num_synthetic_samples=args.num_samples
    )
    
    logger.info("Starting SwellSight training...")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    logger.info(f"Data config: {data_config}")
    
    # Initialize model and trainer
    model = WaveAnalysisModel(model_config)
    trainer = Trainer(model, training_config, data_config, output_dir)
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
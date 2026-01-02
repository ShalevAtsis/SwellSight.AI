#!/usr/bin/env python3
"""
End-to-end MiDaS/ControlNet pipeline script for SwellSight wave analysis.

This script wires together MiDaS depth extraction, ControlNet generation, and 
augmentation systems to create a complete pipeline from real beach images to 
synthetic training datasets with comprehensive quality validation.

Task 20.1: Create end-to-end MiDaS/ControlNet pipeline script
- Wire together MiDaS depth extraction, ControlNet generation, and augmentation systems
- Add command-line interface for processing real images and generating synthetic datasets
- Include progress tracking and quality validation throughout the pipeline
- Add comprehensive error handling and recovery mechanisms
- Create pipeline configuration management and optimization
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback
import time

# Import SwellSight components
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor, DepthExtractionResult
from swellsight.data.synthetic_data_generator import SyntheticDataGenerator
from swellsight.data.real_data_loader import RealDataLoader
from swellsight.training.data_quality_monitor import DataQualityMonitor
from swellsight.utils.model_persistence import ModelPersistence

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Pipeline configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline configuration."""
        self.config = self._load_default_config()
        
        if config_path:
            self._load_config_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default pipeline configuration."""
        return {
            # Input/Output paths
            'real_data_path': 'data/real',
            'synthetic_data_path': 'data/synthetic',
            'metadata_path': 'data/metadata',
            'output_path': 'data/pipeline_output',
            
            # MiDaS configuration
            'midas_model': 'Intel/dpt-large',
            'midas_device': None,  # Auto-detect
            'min_depth_quality': 0.3,
            
            # ControlNet configuration
            'controlnet_model': 'lllyasviel/sd-controlnet-depth',
            'batch_size': 1,
            'max_synthetic_per_real': 5,
            
            # Data generation parameters
            'target_dataset_size': 1000,
            'use_real_images': True,
            'augmentation_seed': 42,
            
            # Quality validation
            'enable_quality_monitoring': True,
            'quality_threshold': 0.5,
            'max_failed_attempts': 3,
            
            # Pipeline optimization
            'enable_caching': True,
            'parallel_processing': False,
            'checkpoint_interval': 100,
            
            # Error handling
            'continue_on_error': True,
            'max_errors': 10,
            'retry_attempts': 2
        }
    
    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with default config
            self.config.update(file_config)
            logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
            logger.info("Using default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def save(self, output_path: str) -> None:
        """Save current configuration to file."""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to: {output_path}")


class PipelineProgress:
    """Progress tracking for pipeline execution."""
    
    def __init__(self, total_steps: int):
        """Initialize progress tracker."""
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.errors = []
        
    def update(self, step_name: str, increment: int = 1) -> None:
        """Update progress."""
        self.current_step += increment
        current_time = time.time()
        self.step_times.append(current_time)
        
        elapsed = current_time - self.start_time
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            eta = avg_time_per_step * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        progress_pct = (self.current_step / self.total_steps) * 100
        
        logger.info(f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - "
                   f"{step_name} - ETA: {eta:.1f}s")
    
    def add_error(self, error_msg: str) -> None:
        """Add error to tracking."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'step': self.current_step
        })
        logger.error(f"Pipeline error at step {self.current_step}: {error_msg}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total_time = time.time() - self.start_time
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'completion_rate': self.current_step / self.total_steps if self.total_steps > 0 else 0,
            'total_time_seconds': total_time,
            'average_time_per_step': total_time / self.current_step if self.current_step > 0 else 0,
            'error_count': len(self.errors),
            'errors': self.errors
        }


class EndToEndPipeline:
    """
    End-to-end MiDaS/ControlNet pipeline for synthetic data generation.
    
    Integrates all components: MiDaS depth extraction, ControlNet synthetic generation,
    augmentation systems, and quality validation into a single cohesive pipeline.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize the end-to-end pipeline."""
        self.config = config
        self.progress = None
        
        # Create output directories
        self.output_path = Path(config.get('output_path'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.pipeline_state = {
            'initialized': True,
            'start_time': None,
            'end_time': None,
            'total_processed': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'checkpoints': []
        }
        
        logger.info("End-to-end pipeline initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize MiDaS depth extractor
            self.depth_extractor = MiDaSDepthExtractor(
                model_name=self.config.get('midas_model'),
                device=self.config.get('midas_device'),
                storage_path=str(self.output_path / 'depth_maps')
            )
            logger.info("MiDaS depth extractor initialized")
            
            # Initialize synthetic data generator
            generator_config = {
                'synthetic_data_path': self.config.get('synthetic_data_path'),
                'metadata_path': self.config.get('metadata_path'),
                'real_data_path': self.config.get('real_data_path'),
                'midas_model': self.config.get('midas_model'),
                'controlnet_model': self.config.get('controlnet_model'),
                'batch_size': self.config.get('batch_size'),
                'min_quality_score': self.config.get('min_depth_quality'),
                'max_failed_attempts': self.config.get('max_failed_attempts'),
                'augmentation_seed': self.config.get('augmentation_seed')
            }
            
            self.synthetic_generator = SyntheticDataGenerator(generator_config)
            logger.info("Synthetic data generator initialized")
            
            # Initialize real data loader
            self.real_data_loader = RealDataLoader(self.config.get('real_data_path'))
            logger.info("Real data loader initialized")
            
            # Initialize quality monitor if enabled
            if self.config.get('enable_quality_monitoring'):
                quality_config = {
                    'output_path': str(self.output_path / 'quality_reports'),
                    'drift_threshold': 0.1,
                    'quality_thresholds': {
                        'diversity_min': 0.3,
                        'balance_min': 0.2,
                        'overall_min': self.config.get('quality_threshold')
                    }
                }
                self.quality_monitor = DataQualityMonitor(quality_config)
                logger.info("Data quality monitor initialized")
            else:
                self.quality_monitor = None
            
            # Initialize model persistence for checkpointing
            self.model_persistence = ModelPersistence()
            logger.info("Model persistence initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def run_full_pipeline(self, target_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            target_samples: Number of synthetic samples to generate (overrides config)
        
        Returns:
            Pipeline execution results
        """
        logger.info("Starting end-to-end pipeline execution")
        
        # Set target samples
        if target_samples is None:
            target_samples = self.config.get('target_dataset_size')
        
        # Initialize progress tracking
        estimated_steps = self._estimate_pipeline_steps(target_samples)
        self.progress = PipelineProgress(estimated_steps)
        
        # Update pipeline state
        self.pipeline_state['start_time'] = datetime.now().isoformat()
        
        try:
            # Step 1: Load and validate real data
            self.progress.update("Loading real data")
            real_metadata = self._load_real_data()
            
            # Step 2: Extract depth maps from real images
            self.progress.update("Extracting depth maps")
            depth_results = self._extract_depth_maps(real_metadata)
            
            # Step 3: Generate synthetic dataset
            self.progress.update("Generating synthetic data")
            synthetic_metadata = self._generate_synthetic_dataset(
                depth_results, target_samples
            )
            
            # Step 4: Validate data quality
            if self.quality_monitor:
                self.progress.update("Validating data quality")
                quality_results = self._validate_data_quality(synthetic_metadata)
            else:
                quality_results = None
            
            # Step 5: Generate final reports
            self.progress.update("Generating reports")
            pipeline_results = self._generate_pipeline_results(
                synthetic_metadata, quality_results
            )
            
            # Update final state
            self.pipeline_state['end_time'] = datetime.now().isoformat()
            self.pipeline_state['successful_samples'] = len(synthetic_metadata)
            
            logger.info(f"Pipeline completed successfully: {len(synthetic_metadata)} samples generated")
            
            return pipeline_results
            
        except Exception as e:
            self.progress.add_error(f"Pipeline execution failed: {e}")
            self.pipeline_state['end_time'] = datetime.now().isoformat()
            
            # Save error state
            error_report = self._generate_error_report(e)
            
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(f"Error report saved to: {error_report}")
            
            raise
    
    def _estimate_pipeline_steps(self, target_samples: int) -> int:
        """Estimate total pipeline steps for progress tracking."""
        # Base steps: load data, extract depths, generate synthetic, validate, report
        base_steps = 5
        
        # Add steps for depth extraction (one per real image)
        try:
            real_metadata = self.real_data_loader.load_real_metadata()
            depth_steps = len(real_metadata) if real_metadata else 10  # Estimate
        except:
            depth_steps = 10  # Default estimate
        
        # Add steps for synthetic generation (one per target sample)
        synthetic_steps = target_samples
        
        # Add steps for quality validation
        quality_steps = 3 if self.quality_monitor else 0
        
        total_steps = base_steps + depth_steps + synthetic_steps + quality_steps
        
        logger.info(f"Estimated pipeline steps: {total_steps}")
        return total_steps
    
    def _load_real_data(self) -> List[Dict[str, Any]]:
        """Load and validate real data."""
        try:
            real_metadata = self.real_data_loader.load_real_metadata()
            
            if not real_metadata:
                if self.config.get('use_real_images'):
                    logger.warning("No real images found, will use synthetic depth generation")
                    return []
                else:
                    logger.info("Real images disabled in configuration")
                    return []
            
            logger.info(f"Loaded {len(real_metadata)} real images for processing")
            
            # Validate real data
            valid_samples = []
            for sample in real_metadata:
                if self._validate_real_sample(sample):
                    valid_samples.append(sample)
                else:
                    logger.warning(f"Invalid real sample: {sample.get('image_path', 'unknown')}")
            
            logger.info(f"Validated {len(valid_samples)} real samples")
            return valid_samples
            
        except Exception as e:
            if self.config.get('continue_on_error'):
                logger.warning(f"Failed to load real data: {e}")
                return []
            else:
                raise
    
    def _validate_real_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate individual real sample."""
        required_fields = ['image_path']
        
        for field in required_fields:
            if field not in sample:
                return False
        
        # Check if image file exists
        image_path = Path(sample['image_path'])
        if not image_path.exists():
            return False
        
        return True
    
    def _extract_depth_maps(self, real_metadata: List[Dict[str, Any]]) -> List[DepthExtractionResult]:
        """Extract depth maps from real images."""
        depth_results = []
        error_count = 0
        max_errors = self.config.get('max_errors')
        
        for sample in real_metadata:
            try:
                # Extract depth map
                depth_result = self.depth_extractor.extract_depth(
                    sample['image_path'], 
                    store_result=True
                )
                
                # Validate depth quality
                if depth_result.depth_quality_score >= self.config.get('min_depth_quality'):
                    depth_results.append(depth_result)
                    self.progress.update(f"Extracted depth: {sample['image_path']}")
                else:
                    logger.warning(f"Low quality depth map: {sample['image_path']} "
                                 f"(score: {depth_result.depth_quality_score:.3f})")
                    
            except Exception as e:
                error_count += 1
                error_msg = f"Failed to extract depth from {sample['image_path']}: {e}"
                self.progress.add_error(error_msg)
                
                if error_count >= max_errors and not self.config.get('continue_on_error'):
                    raise RuntimeError(f"Too many depth extraction errors: {error_count}")
        
        logger.info(f"Successfully extracted {len(depth_results)} depth maps")
        return depth_results
    
    def _generate_synthetic_dataset(self, depth_results: List[DepthExtractionResult], 
                                  target_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic dataset using ControlNet."""
        try:
            # Use synthetic generator with real depth maps if available
            if depth_results and self.config.get('use_real_images'):
                logger.info(f"Generating {target_samples} synthetic samples from {len(depth_results)} real images")
                
                # Generate synthetic samples from real images
                synthetic_metadata = self.synthetic_generator.generate_dataset(
                    num_samples=target_samples,
                    output_path=Path(self.config.get('synthetic_data_path')),
                    use_real_images=True
                )
            else:
                logger.info(f"Generating {target_samples} synthetic samples from synthetic depth maps")
                
                # Fallback to synthetic depth generation
                synthetic_metadata = self.synthetic_generator.generate_dataset(
                    num_samples=target_samples,
                    output_path=Path(self.config.get('synthetic_data_path')),
                    use_real_images=False
                )
            
            # Update progress for each generated sample
            for i, sample in enumerate(synthetic_metadata):
                self.progress.update(f"Generated sample {i+1}")
                
                # Create checkpoint periodically
                if (i + 1) % self.config.get('checkpoint_interval') == 0:
                    self._create_checkpoint(synthetic_metadata[:i+1])
            
            return synthetic_metadata
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            raise
    
    def _validate_data_quality(self, synthetic_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data quality using quality monitor."""
        try:
            # Analyze dataset quality
            self.progress.update("Analyzing dataset quality")
            quality_metrics = self.quality_monitor.analyze_dataset_quality(
                synthetic_metadata, "pipeline_synthetic_dataset"
            )
            
            # Validate individual samples
            self.progress.update("Validating sample quality")
            validation_results = self.quality_monitor.validate_data_quality(synthetic_metadata)
            
            # Detect data drift if baseline exists
            self.progress.update("Detecting data drift")
            drift_results = self.quality_monitor.detect_data_drift(synthetic_metadata)
            
            # Generate quality dashboard
            dashboard_path = self.quality_monitor.generate_quality_dashboard(quality_metrics)
            
            # Generate quality report
            report_path = self.quality_monitor.generate_quality_report(
                quality_metrics, drift_results, validation_results
            )
            
            quality_results = {
                'metrics': quality_metrics,
                'validation': validation_results,
                'drift': drift_results,
                'dashboard_path': str(dashboard_path),
                'report_path': str(report_path)
            }
            
            logger.info(f"Data quality validation completed")
            logger.info(f"Overall quality score: {quality_metrics.overall_quality_score:.3f}")
            logger.info(f"Dashboard: {dashboard_path}")
            logger.info(f"Report: {report_path}")
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Failed to validate data quality: {e}")
            if self.config.get('continue_on_error'):
                return {'error': str(e)}
            else:
                raise
    
    def _generate_pipeline_results(self, synthetic_metadata: List[Dict[str, Any]], 
                                 quality_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final pipeline results."""
        # Compile pipeline results
        results = {
            'pipeline_metadata': {
                'execution_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'config': self.config.config,
                'start_time': self.pipeline_state['start_time'],
                'end_time': self.pipeline_state['end_time'],
                'total_samples_generated': len(synthetic_metadata),
                'pipeline_version': '1.0'
            },
            'synthetic_dataset': {
                'sample_count': len(synthetic_metadata),
                'output_path': self.config.get('synthetic_data_path'),
                'metadata_path': self.config.get('metadata_path'),
                'samples': synthetic_metadata
            },
            'progress_summary': self.progress.get_summary() if self.progress else {},
            'pipeline_state': self.pipeline_state
        }
        
        if quality_results:
            results['quality_analysis'] = quality_results
        
        # Save results to file
        results_file = self.output_path / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to: {results_file}")
        
        return results
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def _create_checkpoint(self, synthetic_metadata: List[Dict[str, Any]]) -> None:
        """Create pipeline checkpoint."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'samples_generated': len(synthetic_metadata),
            'pipeline_state': self.pipeline_state,
            'progress': self.progress.get_summary() if self.progress else {}
        }
        
        checkpoint_file = self.output_path / f"checkpoint_{len(synthetic_metadata)}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.pipeline_state['checkpoints'].append(str(checkpoint_file))
        logger.info(f"Checkpoint created: {checkpoint_file}")
    
    def _generate_error_report(self, error: Exception) -> Path:
        """Generate error report for failed pipeline execution."""
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'pipeline_state': self.pipeline_state,
            'progress_summary': self.progress.get_summary() if self.progress else {},
            'config': self.config.config
        }
        
        error_file = self.output_path / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        return error_file


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="End-to-end MiDaS/ControlNet pipeline for SwellSight wave analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python end_to_end_pipeline.py

  # Generate 500 samples with custom config
  python end_to_end_pipeline.py --samples 500 --config my_config.json

  # Run without real images (synthetic depth only)
  python end_to_end_pipeline.py --no-real-images --samples 1000

  # Run with custom output path
  python end_to_end_pipeline.py --output-path /path/to/output --samples 200
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        help='Number of synthetic samples to generate'
    )
    
    parser.add_argument(
        '--output-path', '-o',
        type=str,
        help='Output directory path'
    )
    
    parser.add_argument(
        '--real-data-path',
        type=str,
        help='Path to real data directory'
    )
    
    parser.add_argument(
        '--no-real-images',
        action='store_true',
        help='Disable real image processing (use synthetic depth only)'
    )
    
    parser.add_argument(
        '--midas-model',
        type=str,
        choices=['Intel/dpt-large', 'Intel/dpt-hybrid-midas'],
        help='MiDaS model to use for depth extraction'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float,
        help='Minimum quality threshold for generated samples'
    )
    
    parser.add_argument(
        '--no-quality-monitoring',
        action='store_true',
        help='Disable data quality monitoring'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution on non-critical errors'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for the pipeline script."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = PipelineConfig(args.config)
        
        # Override config with command-line arguments
        if args.samples:
            config.config['target_dataset_size'] = args.samples
        
        if args.output_path:
            config.config['output_path'] = args.output_path
        
        if args.real_data_path:
            config.config['real_data_path'] = args.real_data_path
        
        if args.no_real_images:
            config.config['use_real_images'] = False
        
        if args.midas_model:
            config.config['midas_model'] = args.midas_model
        
        if args.batch_size:
            config.config['batch_size'] = args.batch_size
        
        if args.quality_threshold:
            config.config['quality_threshold'] = args.quality_threshold
        
        if args.no_quality_monitoring:
            config.config['enable_quality_monitoring'] = False
        
        if args.continue_on_error:
            config.config['continue_on_error'] = True
        
        # Save effective configuration
        output_path = Path(config.get('output_path'))
        output_path.mkdir(parents=True, exist_ok=True)
        config.save(str(output_path / 'pipeline_config.json'))
        
        # Initialize and run pipeline
        logger.info("Initializing end-to-end pipeline...")
        pipeline = EndToEndPipeline(config)
        
        logger.info("Starting pipeline execution...")
        results = pipeline.run_full_pipeline(args.samples)
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Samples generated: {results['synthetic_dataset']['sample_count']}")
        print(f"Output path: {results['synthetic_dataset']['output_path']}")
        print(f"Execution time: {results['progress_summary'].get('total_time_seconds', 0):.1f}s")
        
        if 'quality_analysis' in results:
            quality_score = results['quality_analysis']['metrics'].overall_quality_score
            print(f"Overall quality score: {quality_score:.3f}")
        
        print(f"Results saved to: {config.get('output_path')}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
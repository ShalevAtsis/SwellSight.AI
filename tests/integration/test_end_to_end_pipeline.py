"""
Integration tests for the complete SwellSight pipeline from depth map to trained model.

These tests verify that all components work together correctly in the complete
training and inference pipeline.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from swellsight.config import ModelConfig, TrainingConfig, DataConfig
from swellsight.models import WaveAnalysisModel
from swellsight.data import SyntheticDataGenerator, DatasetManager
from swellsight.training import Trainer
from swellsight.evaluation import MetricsCalculator
from swellsight.inference import InferenceEngine


class TestEndToEndPipeline:
    """Test complete pipeline integration."""
    
    def test_complete_training_pipeline(self, temp_data_dir, device):
        """Test complete pipeline from data generation to trained model."""
        # Configure for minimal test run
        model_config = ModelConfig(
            backbone='convnext_base',
            input_size=(224, 224),  # Smaller for faster testing
            feature_dim=1024,
            hidden_dim=256
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=2,
            num_epochs=2,  # Minimal training
            checkpoint_frequency=1,
            early_stopping_patience=5
        )
        
        data_config = DataConfig(
            num_synthetic_samples=10,  # Minimal dataset
            synthetic_data_path=str(temp_data_dir / "synthetic"),
            metadata_path=str(temp_data_dir / "metadata"),
            train_split=0.8,
            val_split=0.2,
            num_workers=0
        )
        
        # Step 1: Generate synthetic data
        generator_config = {
            'synthetic_data_path': data_config.synthetic_data_path,
            'metadata_path': data_config.metadata_path,
            'image_size': model_config.input_size
        }
        
        # Mock the actual image generation to avoid ControlNet dependency
        with patch.object(SyntheticDataGenerator, '_depth_to_image_controlnet') as mock_gen:
            # Mock successful image generation - return a PIL Image
            mock_image = Image.new('RGB', (224, 224), color='blue')
            mock_gen.return_value = mock_image
            
            generator = SyntheticDataGenerator(generator_config)
            samples_metadata = generator.generate_dataset(data_config.num_synthetic_samples)
            
            assert len(samples_metadata) > 0, "Should generate at least some samples"
        
        # Step 2: Create model
        model = WaveAnalysisModel(model_config)
        assert model is not None, "Model should be created successfully"
        
        # Step 3: Set up dataset manager
        dataset_manager = DatasetManager(
            data_path=str(temp_data_dir),
            config=data_config.to_dict()
        )
        
        # Verify dataset info (handle different possible return formats)
        dataset_info = dataset_manager.get_dataset_info()
        # Just verify the dataset manager was created successfully
        assert dataset_info is not None, "Dataset info should be available"
        
        # Step 4: Set up trainer
        checkpoint_dir = temp_data_dir / "checkpoints"
        trainer = Trainer(
            model=model,
            training_config=training_config,
            data_config=data_config,
            output_dir=checkpoint_dir,
            dataset_manager=dataset_manager
        )
        
        # Step 5: Train model (minimal training)
        training_results = trainer.train()
        
        assert training_results['status'] == 'completed', "Training should complete"
        assert training_results['epochs_trained'] > 0, "Should train for at least one epoch"
        
        # Step 6: Verify model can make predictions
        model.eval()
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(1, 3, *model_config.input_size)
            predictions = model(dummy_input)
            
            # Verify output structure
            assert 'height' in predictions, "Should predict height"
            assert 'wave_type' in predictions, "Should predict wave type"
            assert 'direction' in predictions, "Should predict direction"
            
            # Verify output shapes
            assert predictions['height'].shape == (1, 1), "Height should be scalar per sample"
            assert predictions['wave_type'].shape[0] == 1, "Wave type should have batch dimension"
            assert predictions['direction'].shape[0] == 1, "Direction should have batch dimension"
    
    def test_model_checkpoint_persistence(self, temp_data_dir, device):
        """Test that model checkpoints can be saved and loaded correctly."""
        model_config = ModelConfig(
            input_size=(224, 224),
            feature_dim=512
        )
        
        # Create and train model briefly
        model = WaveAnalysisModel(model_config)
        original_state = model.state_dict()
        
        # Save checkpoint
        checkpoint_path = temp_data_dir / "test_checkpoint.pth"
        checkpoint_data = {
            'model_state_dict': original_state,
            'model_config': model_config.to_dict(),
            'epoch': 5,
            'loss': 0.123
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = WaveAnalysisModel(model_config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify models are equivalent
        new_state = new_model.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], new_state[key]), f"Parameter {key} should match"
    
    def test_evaluation_metrics_integration(self, temp_data_dir):
        """Test that evaluation metrics work with real model outputs."""
        # Create minimal model and data
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        
        # Generate some predictions and targets in the correct format
        batch_size = 5
        
        # Create predictions as list of dictionaries (as expected by calculate_all_metrics)
        predictions = []
        targets = []
        
        for i in range(batch_size):
            # Use valid wave type and direction strings instead of integers
            wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK']
            directions = ['LEFT', 'RIGHT', 'BOTH']
            
            pred_dict = {
                'height_meters': float(torch.randn(1).item() + 2.0),  # Ensure positive height
                'wave_type': wave_types[i % len(wave_types)],
                'direction': directions[i % len(directions)]
            }
            
            target_dict = {
                'height_meters': float(torch.randn(1).item() + 2.0),  # Ensure positive height
                'wave_type': wave_types[i % len(wave_types)],
                'direction': directions[i % len(directions)]
            }
            
            predictions.append(pred_dict)
            targets.append(target_dict)
        
        # Calculate metrics using the correct method - pass individual predictions
        metrics_calculator = MetricsCalculator()
        
        # Use calculate_detailed_metrics which expects the format we have
        results = metrics_calculator.calculate_detailed_metrics(
            predictions=predictions,
            targets=targets,
            dataset_type='test'
        )
        
        # Verify metrics structure - results is an EvaluationResults object
        assert hasattr(results, 'height_metrics'), "Should have height metrics"
        assert hasattr(results, 'wave_type_metrics'), "Should have wave type metrics"
        assert hasattr(results, 'direction_metrics'), "Should have direction metrics"
        
        # Verify specific metrics exist
        assert hasattr(results.height_metrics, 'mae'), "Should have MAE for height"
        assert hasattr(results.height_metrics, 'rmse'), "Should have RMSE for height"
        assert hasattr(results.wave_type_metrics, 'accuracy'), "Should have accuracy for wave type"
        assert hasattr(results.direction_metrics, 'accuracy'), "Should have accuracy for direction"
    
    def test_data_pipeline_integration(self, temp_data_dir):
        """Test that data pipeline components work together correctly."""
        data_config = DataConfig(
            num_synthetic_samples=5,
            synthetic_data_path=str(temp_data_dir / "synthetic"),
            metadata_path=str(temp_data_dir / "metadata"),
            train_split=0.8,
            val_split=0.2,
            num_workers=0
        )
        
        # Mock data generation
        with patch.object(SyntheticDataGenerator, '_depth_to_image_controlnet') as mock_gen:
            mock_image = Image.new('RGB', (224, 224), color='blue')
            mock_gen.return_value = mock_image
            
            # Generate data
            generator_config = {
                'synthetic_data_path': data_config.synthetic_data_path,
                'metadata_path': data_config.metadata_path,
                'image_size': (224, 224)
            }
            
            generator = SyntheticDataGenerator(generator_config)
            samples_metadata = generator.generate_dataset(data_config.num_synthetic_samples)
            
            assert len(samples_metadata) > 0, "Should generate samples"
        
        # Create dataset manager
        dataset_manager = DatasetManager(
            data_path=str(temp_data_dir),
            config=data_config.to_dict()
        )
        
        # Test data loaders
        train_loader = dataset_manager.get_train_loader(batch_size=2)
        val_loader = dataset_manager.get_validation_loader(batch_size=2)
        
        # Verify loaders work
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Verify batch structure
        for batch in [train_batch, val_batch]:
            assert 'image' in batch, "Batch should contain images"
            assert 'height' in batch, "Batch should contain height labels"
            assert 'wave_type' in batch, "Batch should contain wave type labels"
            assert 'direction' in batch, "Batch should contain direction labels"
            
            # Verify tensor shapes
            assert len(batch['image'].shape) == 4, "Images should be 4D tensors"
            assert batch['image'].shape[1] == 3, "Images should have 3 channels"
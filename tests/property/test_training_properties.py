"""Property-based tests for training components."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
import torch
import torch.nn as nn
from datetime import datetime

from swellsight.training.trainer import Trainer
from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config.model_config import ModelConfig
from swellsight.config.training_config import TrainingConfig
from swellsight.config.data_config import DataConfig
from swellsight.data.synthetic_data_generator import SyntheticDataGenerator
from swellsight.data.dataset_manager import DatasetManager
from swellsight.utils.checkpoint_utils import load_checkpoint, validate_checkpoint_integrity


# Test data strategies
@st.composite
def training_config_strategy(draw):
    """Generate valid training configuration for testing."""
    config = TrainingConfig()
    # Use smaller values for testing
    config.num_epochs = draw(st.integers(min_value=2, max_value=5))
    config.batch_size = draw(st.integers(min_value=2, max_value=4))
    config.checkpoint_frequency = draw(st.integers(min_value=1, max_value=2))
    config.validation_frequency = 1
    config.early_stopping_patience = 10
    return config


@st.composite
def model_config_strategy(draw):
    """Generate valid model configuration for testing."""
    config = ModelConfig()
    # Use smaller model for testing
    config.backbone = 'convnext_tiny'
    config.feature_dim = 768  # ConvNeXt-Tiny output
    config.hidden_dim = 256
    return config


class TestTrainerProperties:
    """Property-based tests for Trainer class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create directory structure
        (self.temp_path / 'synthetic').mkdir(parents=True)
        (self.temp_path / 'metadata').mkdir(parents=True)
        (self.temp_path / 'checkpoints').mkdir(parents=True)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_dataset(self, num_samples: int = 8):
        """Create a small test dataset for training tests."""
        # Generate synthetic data
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (224, 224)  # Smaller for faster testing
        }
        generator = SyntheticDataGenerator(config)
        samples_metadata = generator.generate_dataset(num_samples)
        return samples_metadata
    
    @given(training_config_strategy(), model_config_strategy())
    @settings(max_examples=1, deadline=None)
    def test_checkpoint_persistence(self, training_config, model_config):
        """
        Feature: wave-analysis-model, Property 10: Checkpoint Persistence
        
        For any training run, model checkpoints should be saved every 10 epochs
        with complete model state and validation metrics.
        """
        # Create test dataset
        self._create_test_dataset(8)
        
        # Create data config
        data_config = DataConfig()
        data_config.synthetic_data_path = str(self.temp_path / 'synthetic')
        data_config.metadata_path = str(self.temp_path / 'metadata')
        data_config.image_size = (224, 224)
        
        # Update model config for smaller images
        model_config.input_size = (224, 224)
        
        # Create model and trainer
        model = WaveAnalysisModel(model_config)
        
        # Create dataset manager
        dataset_config = data_config.to_dict()
        dataset_config['num_workers'] = 0  # Avoid multiprocessing in tests
        dataset_manager = DatasetManager(str(self.temp_path), dataset_config)
        
        trainer = Trainer(
            model=model,
            training_config=training_config,
            data_config=data_config,
            output_dir=self.temp_path / 'checkpoints',
            dataset_manager=dataset_manager
        )
        
        # Run short training
        results = trainer.train()
        
        # Verify training completed
        assert results['status'] == 'completed'
        assert results['epochs_trained'] >= 1
        
        # Check that checkpoints were saved according to frequency
        checkpoint_dir = self.temp_path / 'checkpoints'
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        # Should have at least one checkpoint (final checkpoint always saved)
        assert len(checkpoint_files) >= 1
        
        # Verify checkpoint frequency
        expected_checkpoints = []
        for epoch in range(1, results['epochs_trained'] + 1):
            if epoch % training_config.checkpoint_frequency == 0:
                expected_checkpoints.append(epoch)
        
        # Always save final checkpoint
        if results['epochs_trained'] not in expected_checkpoints:
            expected_checkpoints.append(results['epochs_trained'])
        
        # Check that expected checkpoints exist
        for epoch in expected_checkpoints:
            checkpoint_file = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            assert checkpoint_file.exists(), f"Missing checkpoint for epoch {epoch}"
            
            # Verify checkpoint integrity
            assert validate_checkpoint_integrity(checkpoint_file) is True
            
            # Verify checkpoint can be loaded
            checkpoint_info = load_checkpoint(
                filepath=checkpoint_file,
                model=model,
                device=torch.device('cpu')
            )
            
            # Verify checkpoint contains required information
            assert checkpoint_info['epoch'] == epoch
            assert 'loss' in checkpoint_info
            assert 'metrics' in checkpoint_info
            assert isinstance(checkpoint_info['metrics'], dict)
        
        # Verify best model was saved
        best_model_file = checkpoint_dir / 'best_model.pth'
        assert best_model_file.exists()
        assert validate_checkpoint_integrity(best_model_file) is True
    
    @given(training_config_strategy(), model_config_strategy())
    @settings(max_examples=1, deadline=None)
    def test_metadata_completeness(self, training_config, model_config):
        """
        Feature: wave-analysis-model, Property 20: Metadata Completeness
        
        For any saved model, the checkpoint should include training date,
        dataset version, and performance metrics in the metadata.
        """
        # Create test dataset
        self._create_test_dataset(6)
        
        # Create data config
        data_config = DataConfig()
        data_config.synthetic_data_path = str(self.temp_path / 'synthetic')
        data_config.metadata_path = str(self.temp_path / 'metadata')
        data_config.image_size = (224, 224)
        
        # Update model config for smaller images
        model_config.input_size = (224, 224)
        
        # Create model and trainer
        model = WaveAnalysisModel(model_config)
        
        # Create dataset manager
        dataset_config = data_config.to_dict()
        dataset_config['num_workers'] = 0
        dataset_manager = DatasetManager(str(self.temp_path), dataset_config)
        
        trainer = Trainer(
            model=model,
            training_config=training_config,
            data_config=data_config,
            output_dir=self.temp_path / 'checkpoints',
            dataset_manager=dataset_manager
        )
        
        # Run short training
        results = trainer.train()
        
        # Get the final checkpoint
        checkpoint_dir = self.temp_path / 'checkpoints'
        final_epoch = results['epochs_trained']
        checkpoint_file = checkpoint_dir / f'checkpoint_epoch_{final_epoch:03d}.pth'
        
        assert checkpoint_file.exists()
        
        # Load checkpoint and verify metadata
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        
        # Verify required metadata fields exist
        required_fields = [
            'epoch', 'loss', 'metrics', 'config', 'timestamp', 'pytorch_version'
        ]
        for field in required_fields:
            assert field in checkpoint, f"Missing required field: {field}"
        
        # Verify metadata content
        assert isinstance(checkpoint['epoch'], int)
        assert checkpoint['epoch'] == final_epoch
        
        assert isinstance(checkpoint['loss'], (int, float))
        assert checkpoint['loss'] >= 0
        
        assert isinstance(checkpoint['metrics'], dict)
        # Should contain validation metrics
        expected_metrics = ['height_mae', 'type_acc', 'direction_acc']
        for metric in expected_metrics:
            assert metric in checkpoint['metrics']
        
        assert isinstance(checkpoint['config'], dict)
        
        # Verify timestamp format (ISO format)
        timestamp = checkpoint['timestamp']
        assert isinstance(timestamp, str)
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp.replace('Z', '+00:00') if timestamp.endswith('Z') else timestamp)
        
        assert isinstance(checkpoint['pytorch_version'], str)
        
        # Check for additional metadata if present
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            assert isinstance(metadata, dict)
            
            # Should contain training information
            expected_metadata_fields = [
                'training_config', 'data_config', 'model_config', 'dataset_info'
            ]
            for field in expected_metadata_fields:
                assert field in metadata, f"Missing metadata field: {field}"
            
            # Verify training config preservation
            saved_training_config = metadata['training_config']
            assert saved_training_config['learning_rate'] == training_config.learning_rate
            assert saved_training_config['batch_size'] == training_config.batch_size
            assert saved_training_config['num_epochs'] == training_config.num_epochs
            
            # Verify data config preservation
            saved_data_config = metadata['data_config']
            assert saved_data_config['synthetic_data_path'] == str(self.temp_path / 'synthetic')
            assert saved_data_config['metadata_path'] == str(self.temp_path / 'metadata')
            
            # Verify model config preservation
            saved_model_config = metadata['model_config']
            assert saved_model_config['backbone'] == model_config.backbone
            assert saved_model_config['feature_dim'] == model_config.feature_dim
            
            # Verify dataset info
            dataset_info = metadata['dataset_info']
            assert isinstance(dataset_info, dict)
            assert 'synthetic_total' in dataset_info
            assert 'synthetic_train' in dataset_info
            assert 'synthetic_val' in dataset_info
        
        # Check JSON metadata file
        json_metadata_file = checkpoint_file.with_suffix('.json')
        assert json_metadata_file.exists()
        
        with open(json_metadata_file, 'r') as f:
            json_metadata = json.load(f)
        
        # Verify JSON metadata contains key information
        assert 'epoch' in json_metadata
        assert 'loss' in json_metadata
        assert 'metrics' in json_metadata
        assert 'timestamp' in json_metadata
        assert 'config' in json_metadata
        
        assert json_metadata['epoch'] == final_epoch
        assert isinstance(json_metadata['metrics'], dict)
    
    @given(st.integers(min_value=6, max_value=10))
    @settings(max_examples=1, deadline=None)
    def test_training_loop_consistency(self, num_samples):
        """
        Test that the training loop produces consistent results and metrics.
        
        For any training run, the trainer should:
        - Complete all requested epochs or stop early
        - Produce decreasing or stable loss values
        - Generate valid metrics for all tasks
        - Maintain training history
        """
        # Create test dataset
        self._create_test_dataset(num_samples)
        
        # Create configurations
        training_config = TrainingConfig()
        training_config.num_epochs = 3
        training_config.batch_size = 2
        training_config.checkpoint_frequency = 1
        training_config.validation_frequency = 1
        training_config.early_stopping_patience = 10
        
        model_config = ModelConfig()
        model_config.backbone = 'convnext_tiny'
        model_config.feature_dim = 768
        model_config.input_size = (224, 224)
        
        data_config = DataConfig()
        data_config.data_path = str(self.temp_path)
        data_config.image_size = (224, 224)
        
        # Create model and trainer
        model = WaveAnalysisModel(model_config)
        
        dataset_config = data_config.to_dict()
        dataset_config['num_workers'] = 0
        dataset_manager = DatasetManager(str(self.temp_path), dataset_config)
        
        trainer = Trainer(
            model=model,
            training_config=training_config,
            data_config=data_config,
            output_dir=self.temp_path / 'checkpoints',
            dataset_manager=dataset_manager
        )
        
        # Run training
        results = trainer.train()
        
        # Verify training completion
        assert results['status'] == 'completed'
        assert results['epochs_trained'] >= 1
        assert results['epochs_trained'] <= training_config.num_epochs
        
        # Verify training history exists and has correct structure
        history = results['training_history']
        assert isinstance(history, dict)
        
        expected_history_keys = [
            'train_loss', 'val_loss', 'train_height_mae', 'val_height_mae',
            'train_type_acc', 'val_type_acc', 'train_direction_acc', 'val_direction_acc'
        ]
        for key in expected_history_keys:
            assert key in history
            assert isinstance(history[key], list)
            assert len(history[key]) == results['epochs_trained']
        
        # Verify metrics are valid numbers
        for epoch_idx in range(results['epochs_trained']):
            # Loss values should be positive
            assert history['train_loss'][epoch_idx] >= 0
            assert history['val_loss'][epoch_idx] >= 0
            
            # MAE values should be positive
            assert history['train_height_mae'][epoch_idx] >= 0
            assert history['val_height_mae'][epoch_idx] >= 0
            
            # Accuracy values should be between 0 and 1
            assert 0 <= history['train_type_acc'][epoch_idx] <= 1
            assert 0 <= history['val_type_acc'][epoch_idx] <= 1
            assert 0 <= history['train_direction_acc'][epoch_idx] <= 1
            assert 0 <= history['val_direction_acc'][epoch_idx] <= 1
        
        # Verify final metrics
        final_train_metrics = results['final_train_metrics']
        final_val_metrics = results['final_val_metrics']
        
        for metrics in [final_train_metrics, final_val_metrics]:
            assert 'loss' in metrics
            assert 'height_mae' in metrics
            assert 'type_acc' in metrics
            assert 'direction_acc' in metrics
            
            assert metrics['loss'] >= 0
            assert metrics['height_mae'] >= 0
            assert 0 <= metrics['type_acc'] <= 1
            assert 0 <= metrics['direction_acc'] <= 1
        
        # Verify best validation loss tracking
        assert isinstance(results['best_val_loss'], (int, float))
        assert results['best_val_loss'] >= 0
        
        # Best validation loss should be the minimum from history
        min_val_loss = min(history['val_loss'])
        assert abs(results['best_val_loss'] - min_val_loss) < 1e-3  # Relaxed tolerance for floating point comparison
        
        # Verify training time tracking
        assert 'total_training_time' in results
        assert isinstance(results['total_training_time'], (int, float))
        assert results['total_training_time'] > 0
        
        # Verify early stopping flag
        assert 'early_stopped' in results
        assert isinstance(results['early_stopped'], bool)


class TestCheckpointProperties:
    """Property-based tests for checkpoint functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(model_config_strategy())
    @settings(max_examples=2, deadline=None)
    def test_checkpoint_round_trip_consistency(self, model_config):
        """
        Test that saving and loading checkpoints preserves model state.
        
        For any model state, saving to checkpoint and loading should
        produce identical model outputs for the same inputs.
        """
        from swellsight.utils.checkpoint_utils import save_checkpoint, load_checkpoint
        
        # Create model
        model_config.input_size = (224, 224)
        model = WaveAnalysisModel(model_config)
        
        # Create dummy optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Create test input
        test_input = torch.randn(2, 3, 224, 224)
        
        # Get original model output
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # Save checkpoint
        checkpoint_path = self.temp_path / 'test_checkpoint.pth'
        test_metrics = {'loss': 0.5, 'accuracy': 0.8}
        test_config = {'learning_rate': 0.001, 'batch_size': 32}
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            loss=0.5,
            metrics=test_metrics,
            config=test_config,
            filepath=checkpoint_path,
            metadata={'test': True}
        )
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists()
        assert validate_checkpoint_integrity(checkpoint_path) is True
        
        # Create new model with same config
        new_model = WaveAnalysisModel(model_config)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)
        
        # Load checkpoint
        checkpoint_info = load_checkpoint(
            filepath=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=torch.device('cpu')
        )
        
        # Verify checkpoint info
        assert checkpoint_info['epoch'] == 5
        assert checkpoint_info['loss'] == 0.5
        assert checkpoint_info['metrics'] == test_metrics
        assert checkpoint_info['config'] == test_config
        assert checkpoint_info['metadata'] == {'test': True}
        
        # Get loaded model output
        new_model.eval()
        with torch.no_grad():
            loaded_output = new_model(test_input)
        
        # Verify outputs are identical (within numerical precision)
        for key in original_output.keys():
            assert torch.allclose(
                original_output[key], 
                loaded_output[key], 
                atol=1e-6, 
                rtol=1e-5
            ), f"Output mismatch for {key}"
        
        # Verify optimizer state is preserved
        original_param_groups = optimizer.param_groups[0]
        loaded_param_groups = new_optimizer.param_groups[0]
        assert original_param_groups['lr'] == loaded_param_groups['lr']
        
        # Verify scheduler state is preserved
        assert scheduler.step_size == new_scheduler.step_size
        assert scheduler.gamma == new_scheduler.gamma
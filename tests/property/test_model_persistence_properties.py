"""Property-based tests for model persistence system."""

import torch
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings
import pytest

from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config import ModelConfig
from swellsight.utils.model_persistence import (
    ModelPersistence, 
    save_model, 
    load_model, 
    validate_model,
    get_model_info
)


class TestModelPersistenceProperties:
    """Property-based tests for model persistence system."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.persistence = ModelPersistence()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        include_metadata=st.booleans(),
        compress=st.booleans()
    )
    @settings(max_examples=10, deadline=None)
    def test_model_serialization_round_trip(self, batch_size, include_metadata, compress):
        """
        Feature: wave-analysis-model, Property 19: Model Serialization Round-Trip
        
        For any trained model, saving and then loading the model should produce 
        identical inference outputs for the same input.
        
        **Validates: Requirements 8.1, 8.3**
        """
        # Create and initialize model
        config = ModelConfig()
        original_model = WaveAnalysisModel(config)
        original_model.eval()
        
        # Create test input
        test_input = torch.randn(batch_size, 3, 768, 768)
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = original_model(test_input)
        
        # Prepare metadata if requested
        metadata = None
        if include_metadata:
            metadata = {
                'test_batch_size': batch_size,
                'test_timestamp': 'test_run',
                'test_compress': compress
            }
        
        # Save model
        model_path = self.temp_dir / f"test_model_{batch_size}.pth"
        save_info = save_model(
            original_model, 
            model_path, 
            metadata=metadata,
            compress=compress
        )
        
        # Verify save info
        assert 'filepath' in save_info
        assert 'integrity_hash' in save_info
        assert 'file_size' in save_info
        assert 'timestamp' in save_info
        
        # Load model
        loaded_model = load_model(model_path)
        loaded_model.eval()
        
        # Get loaded model predictions
        with torch.no_grad():
            loaded_outputs = loaded_model(test_input)
        
        # Verify outputs are identical (within numerical precision)
        tolerance = 1e-4  # More realistic tolerance for model serialization
        
        # Check height predictions
        height_diff = torch.abs(original_outputs['height'] - loaded_outputs['height'])
        assert torch.all(height_diff < tolerance), \
            f"Height predictions differ by more than {tolerance}"
        
        # Check wave type predictions
        wave_type_diff = torch.abs(original_outputs['wave_type'] - loaded_outputs['wave_type'])
        assert torch.all(wave_type_diff < tolerance), \
            f"Wave type predictions differ by more than {tolerance}"
        
        # Check direction predictions
        direction_diff = torch.abs(original_outputs['direction'] - loaded_outputs['direction'])
        assert torch.all(direction_diff < tolerance), \
            f"Direction predictions differ by more than {tolerance}"
        
        # Verify model integrity
        assert validate_model(model_path), "Loaded model failed integrity validation"
        
        # Verify metadata if included
        if include_metadata:
            model_info = get_model_info(model_path)
            assert 'metadata' in model_info
            assert model_info['metadata']['test_batch_size'] == batch_size
            assert model_info['metadata']['test_compress'] == compress
    
    @given(
        device_str=st.sampled_from(['cpu', 'cuda', 'auto'])
    )
    @settings(max_examples=5, deadline=None)
    def test_device_compatibility(self, device_str):
        """
        Feature: wave-analysis-model, Property 21: Device Compatibility
        
        For any trained model, it should run successfully on both CPU and GPU devices 
        with identical outputs (within numerical precision).
        
        **Validates: Requirements 8.5**
        """
        # Skip CUDA tests if not available
        if device_str == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create model
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Test device compatibility
        compatibility = self.persistence.test_device_compatibility(model)
        
        # CPU should always be compatible
        assert compatibility['cpu'], "Model should be compatible with CPU"
        
        # CUDA compatibility depends on availability
        if torch.cuda.is_available():
            assert compatibility['cuda'], "Model should be compatible with CUDA when available"
        
        # Test save/load with specific device
        model_path = self.temp_dir / f"test_device_{device_str}.pth"
        
        # Save model
        save_model(model, model_path)
        
        # Load model on specified device
        loaded_model = load_model(model_path, device=device_str)
        
        # Verify model is on correct device
        expected_device = 'cuda' if device_str == 'cuda' and torch.cuda.is_available() else 'cpu'
        model_device = str(next(loaded_model.parameters()).device).split(':')[0]
        assert model_device == expected_device, \
            f"Model should be on {expected_device}, but is on {model_device}"
        
        # Test inference on the device
        test_input = torch.randn(1, 3, 768, 768)
        if expected_device == 'cuda':
            test_input = test_input.cuda()
        
        with torch.no_grad():
            outputs = loaded_model(test_input)
        
        # Verify outputs are on correct device
        for key, output in outputs.items():
            output_device = str(output.device).split(':')[0]
            assert output_device == expected_device, \
                f"Output {key} should be on {expected_device}, but is on {output_device}"
    
    @given(
        num_models=st.integers(min_value=2, max_value=5),
        use_different_configs=st.booleans()
    )
    @settings(max_examples=5, deadline=None)
    def test_multiple_model_persistence(self, num_models, use_different_configs):
        """
        Test that multiple models can be saved and loaded independently.
        
        This ensures the persistence system can handle multiple models without
        interference or corruption.
        """
        models = []
        model_paths = []
        original_outputs = []
        
        # Create and save multiple models
        for i in range(num_models):
            if use_different_configs:
                # Use slightly different configurations
                config = ModelConfig(
                    hidden_dim=512 + i * 64,
                    dropout_rate=0.1 + i * 0.05
                )
            else:
                config = ModelConfig()
            
            model = WaveAnalysisModel(config)
            model.eval()
            models.append(model)
            
            # Create consistent test input for this model
            torch.manual_seed(42 + i)  # Ensure reproducible test input
            test_input = torch.randn(1, 3, 768, 768)
            
            # Get original predictions
            with torch.no_grad():
                outputs = model(test_input)
            original_outputs.append((outputs, test_input))
            
            # Save model
            model_path = self.temp_dir / f"model_{i}.pth"
            save_model(model, model_path, metadata={'model_index': i})
            model_paths.append(model_path)
        
        # Load and verify all models
        for i, model_path in enumerate(model_paths):
            loaded_model = load_model(model_path)
            loaded_model.eval()
            
            # Verify model info
            model_info = get_model_info(model_path)
            assert model_info['metadata']['model_index'] == i
            
            # Use the same test input that was used for original predictions
            original_output, test_input = original_outputs[i]
            
            # Test inference with the same input
            with torch.no_grad():
                loaded_outputs = loaded_model(test_input)
            
            # Verify outputs match original (within tolerance)
            tolerance = 1e-4  # More realistic tolerance for model serialization
            for key in ['height', 'wave_type', 'direction']:
                diff = torch.abs(original_output[key] - loaded_outputs[key])
                assert torch.all(diff < tolerance), \
                    f"Model {i} output {key} differs after save/load: max diff = {torch.max(diff).item()}"
    
    @given(
        corrupt_type=st.sampled_from(['truncate', 'random_bytes', 'empty'])
    )
    @settings(max_examples=3, deadline=None)
    def test_integrity_validation_with_corruption(self, corrupt_type):
        """
        Test that integrity validation correctly detects corrupted model files.
        
        This ensures the persistence system can detect and reject corrupted files.
        """
        # Create and save a valid model
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model_path = self.temp_dir / "test_model.pth"
        save_model(model, model_path)
        
        # Verify model is initially valid
        assert validate_model(model_path), "Model should be valid before corruption"
        
        # Corrupt the file
        if corrupt_type == 'truncate':
            # Truncate file to half size
            with open(model_path, 'r+b') as f:
                f.seek(0, 2)  # Go to end
                size = f.tell()
                f.truncate(size // 2)
        
        elif corrupt_type == 'random_bytes':
            # Overwrite part of file with random bytes
            with open(model_path, 'r+b') as f:
                f.seek(100)  # Skip header
                f.write(b'\x00' * 1000)  # Write zeros
        
        elif corrupt_type == 'empty':
            # Make file empty
            with open(model_path, 'w') as f:
                pass
        
        # Verify corruption is detected
        assert not validate_model(model_path), \
            f"Corrupted model ({corrupt_type}) should fail validation"
        
        # Verify loading fails gracefully
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            load_model(model_path)
    
    @given(
        metadata_keys=st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            min_size=1,
            max_size=5,
            unique=True
        ),
        metadata_values=st.lists(
            st.one_of(
                st.text(min_size=1, max_size=50),
                st.integers(min_value=0, max_value=1000),
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=5, deadline=None)
    def test_metadata_preservation(self, metadata_keys, metadata_values):
        """
        Test that custom metadata is correctly preserved during save/load cycles.
        
        This ensures that additional information can be reliably stored with models.
        """
        # Ensure we have matching keys and values
        metadata = dict(zip(metadata_keys[:len(metadata_values)], metadata_values))
        
        # Create model
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        
        # Save with metadata
        model_path = self.temp_dir / "metadata_test.pth"
        save_model(model, model_path, metadata=metadata)
        
        # Load model info
        model_info = get_model_info(model_path)
        
        # Verify metadata is preserved
        assert 'metadata' in model_info
        saved_metadata = model_info['metadata']
        
        for key, expected_value in metadata.items():
            assert key in saved_metadata, f"Metadata key '{key}' not found"
            assert saved_metadata[key] == expected_value, \
                f"Metadata value for '{key}' changed: expected {expected_value}, got {saved_metadata[key]}"
        
        # Verify model still loads correctly
        loaded_model = load_model(model_path)
        assert isinstance(loaded_model, WaveAnalysisModel)
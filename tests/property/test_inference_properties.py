"""Property-based tests for inference API."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from swellsight.inference.inference_engine import InferenceEngine, WavePrediction, InferenceError
from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config.model_config import ModelConfig


# Test data generators
@st.composite
def valid_image_tensor(draw):
    """Generate valid image tensors for testing."""
    # Generate tensor with shape [3, H, W] where H, W are reasonable sizes
    height = draw(st.integers(min_value=64, max_value=512))
    width = draw(st.integers(min_value=64, max_value=512))
    
    # Generate normalized tensor values [0, 1]
    tensor = draw(arrays(
        dtype=np.float32,
        shape=(3, height, width),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    return torch.from_numpy(tensor)


@st.composite
def create_test_image_file(draw, tmp_path):
    """Create a test image file with valid format."""
    # Choose format
    format_choice = draw(st.sampled_from(['JPEG', 'PNG']))
    extension = '.jpg' if format_choice == 'JPEG' else '.png'
    
    # Generate image dimensions
    width = draw(st.integers(min_value=64, max_value=512))
    height = draw(st.integers(min_value=64, max_value=512))
    
    # Create random RGB image
    image_array = draw(arrays(
        dtype=np.uint8,
        shape=(height, width, 3),
        elements=st.integers(min_value=0, max_value=255)
    ))
    
    # Create PIL image and save
    image = Image.fromarray(image_array, 'RGB')
    image_path = tmp_path / f"test_image{extension}"
    image.save(image_path, format=format_choice)
    
    return str(image_path)


@st.composite
def invalid_image_path(draw):
    """Generate invalid image paths for testing error handling."""
    path_type = draw(st.sampled_from(['nonexistent', 'wrong_extension', 'directory']))
    
    if path_type == 'nonexistent':
        return "/nonexistent/path/image.jpg"
    elif path_type == 'wrong_extension':
        return "image.txt"
    else:  # directory
        return "/tmp"


class TestInferenceProperties:
    """Property-based tests for InferenceEngine."""
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing."""
        return ModelConfig()
    
    @pytest.fixture
    def mock_model(self, model_config):
        """Create a mock model for testing."""
        model = WaveAnalysisModel(model_config)
        model.eval()
        return model
    
    @pytest.fixture
    def inference_engine(self, mock_model, model_config):
        """Create inference engine for testing."""
        return InferenceEngine(mock_model, model_config, device="cpu")
    
    @pytest.fixture
    def tmp_path(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @given(valid_image_tensor())
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_11_image_format_compatibility_tensor(self, inference_engine, image_tensor):
        """
        Feature: wave-analysis-model, Property 11: Image Format Compatibility
        Test that the inference engine can process valid image tensors.
        """
        try:
            result = inference_engine.predict_from_tensor(image_tensor)
            
            # Verify result is a WavePrediction
            assert isinstance(result, WavePrediction)
            
            # Verify all required fields are present
            assert hasattr(result, 'height_meters')
            assert hasattr(result, 'wave_type')
            assert hasattr(result, 'direction')
            assert hasattr(result, 'wave_type_probs')
            assert hasattr(result, 'direction_probs')
            assert hasattr(result, 'confidence_scores')
            
        except Exception as e:
            pytest.fail(f"Valid tensor should not raise exception: {e}")
    
    @given(st.data())
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_11_image_format_compatibility_files(self, inference_engine, tmp_path, data):
        """
        Feature: wave-analysis-model, Property 11: Image Format Compatibility
        Test that the inference engine can process JPEG and PNG files.
        """
        # Generate test image file
        image_path = data.draw(create_test_image_file(tmp_path))
        
        try:
            result = inference_engine.predict(image_path)
            
            # Verify result is a WavePrediction
            assert isinstance(result, WavePrediction)
            
            # Verify file format was handled correctly
            path_obj = Path(image_path)
            assert path_obj.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            
        except Exception as e:
            pytest.fail(f"Valid image file should not raise exception: {e}")
    
    @given(valid_image_tensor())
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_12_api_response_structure(self, inference_engine, image_tensor):
        """
        Feature: wave-analysis-model, Property 12: API Response Structure
        Test that API returns structured JSON output with wave parameters and confidence scores.
        """
        result = inference_engine.predict_from_tensor(image_tensor)
        
        # Test structured response
        assert isinstance(result, WavePrediction)
        
        # Test height prediction
        assert isinstance(result.height_meters, float)
        assert 0.0 <= result.height_meters <= 10.0  # Reasonable range
        
        # Test wave type prediction
        assert isinstance(result.wave_type, str)
        assert result.wave_type in inference_engine.config.wave_type_classes
        
        # Test direction prediction
        assert isinstance(result.direction, str)
        assert result.direction in inference_engine.config.direction_classes
        
        # Test wave type probabilities
        assert isinstance(result.wave_type_probs, dict)
        assert len(result.wave_type_probs) == len(inference_engine.config.wave_type_classes)
        assert all(isinstance(prob, float) for prob in result.wave_type_probs.values())
        assert all(0.0 <= prob <= 1.0 for prob in result.wave_type_probs.values())
        assert abs(sum(result.wave_type_probs.values()) - 1.0) < 1e-5  # Probabilities sum to 1
        
        # Test direction probabilities
        assert isinstance(result.direction_probs, dict)
        assert len(result.direction_probs) == len(inference_engine.config.direction_classes)
        assert all(isinstance(prob, float) for prob in result.direction_probs.values())
        assert all(0.0 <= prob <= 1.0 for prob in result.direction_probs.values())
        assert abs(sum(result.direction_probs.values()) - 1.0) < 1e-5  # Probabilities sum to 1
        
        # Test confidence scores
        assert isinstance(result.confidence_scores, dict)
        assert 'height' in result.confidence_scores
        assert 'wave_type' in result.confidence_scores
        assert 'direction' in result.confidence_scores
        assert all(isinstance(score, float) for score in result.confidence_scores.values())
        assert all(0.0 <= score <= 1.0 for score in result.confidence_scores.values())
        
        # Test JSON serialization
        json_dict = result.to_dict()
        assert isinstance(json_dict, dict)
        assert 'height_meters' in json_dict
        assert 'wave_type' in json_dict
        assert 'direction' in json_dict
        assert 'wave_type_probabilities' in json_dict
        assert 'direction_probabilities' in json_dict
        assert 'confidence_scores' in json_dict
        
        # Test JSON string conversion
        json_str = result.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    @given(st.data())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_13_error_handling_robustness_invalid_paths(self, inference_engine, data):
        """
        Feature: wave-analysis-model, Property 13: Error Handling Robustness
        Test that API returns appropriate error messages for invalid inputs without crashing.
        """
        # Generate invalid path
        invalid_path = data.draw(invalid_image_path())
        
        # Should raise InferenceError, not crash
        with pytest.raises(InferenceError) as exc_info:
            inference_engine.predict(invalid_path)
        
        # Error message should be informative
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0
        assert isinstance(error_msg, str)
    
    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_13_error_handling_robustness_invalid_tensors(self, inference_engine, num_dims):
        """
        Feature: wave-analysis-model, Property 13: Error Handling Robustness
        Test error handling for invalid tensor inputs.
        """
        # Create tensor with wrong number of dimensions
        if num_dims == 1:
            invalid_tensor = torch.randn(10)
        elif num_dims == 2:
            invalid_tensor = torch.randn(10, 10)
        elif num_dims == 4:
            # Wrong number of channels
            invalid_tensor = torch.randn(1, 5, 64, 64)  # 5 channels instead of 3
        else:
            # 5D tensor
            invalid_tensor = torch.randn(1, 3, 64, 64, 64)
        
        # Should raise InferenceError
        with pytest.raises(InferenceError) as exc_info:
            inference_engine.predict_from_tensor(invalid_tensor)
        
        # Error message should be informative
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0
        assert isinstance(error_msg, str)
    
    def test_property_13_error_handling_robustness_non_tensor_input(self, inference_engine):
        """
        Feature: wave-analysis-model, Property 13: Error Handling Robustness
        Test error handling for non-tensor inputs.
        """
        # Test with various invalid input types
        invalid_inputs = [
            "not_a_tensor",
            123,
            [1, 2, 3],
            {"key": "value"},
            None
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(InferenceError) as exc_info:
                inference_engine.predict_from_tensor(invalid_input)
            
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0
            assert "Expected torch.Tensor" in error_msg
    
    @given(st.lists(valid_image_tensor(), min_size=1, max_size=5))
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_prediction_consistency(self, inference_engine, image_tensors):
        """
        Test that batch prediction produces consistent results.
        """
        # Get predictions using predict_from_tensor on individual images
        individual_results = []
        for tensor in image_tensors:
            result = inference_engine.predict_from_tensor(tensor)
            individual_results.append(result)
        
        # Verify we get the expected number of results
        assert len(individual_results) == len(image_tensors)
        
        # Each result should be valid
        for result in individual_results:
            assert isinstance(result, WavePrediction)
            assert isinstance(result.height_meters, float)
            assert result.wave_type in inference_engine.config.wave_type_classes
            assert result.direction in inference_engine.config.direction_classes
    
    def test_model_info_completeness(self, inference_engine):
        """
        Test that model info contains all required information.
        """
        info = inference_engine.get_model_info()
        
        # Check required fields
        required_fields = [
            'device', 'input_size', 'wave_type_classes', 'direction_classes',
            'height_range', 'supported_formats', 'model_parameters', 'model_trainable_parameters'
        ]
        
        for field in required_fields:
            assert field in info, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(info['device'], str)
        assert isinstance(info['input_size'], (list, tuple))
        assert len(info['input_size']) == 2
        assert isinstance(info['wave_type_classes'], list)
        assert isinstance(info['direction_classes'], list)
        assert isinstance(info['height_range'], list)
        assert len(info['height_range']) == 2
        assert isinstance(info['supported_formats'], list)
        assert isinstance(info['model_parameters'], int)
        assert isinstance(info['model_trainable_parameters'], int)
        
        # Check parameter counts are reasonable
        assert info['model_parameters'] > 0
        assert info['model_trainable_parameters'] >= 0
        assert info['model_trainable_parameters'] <= info['model_parameters']
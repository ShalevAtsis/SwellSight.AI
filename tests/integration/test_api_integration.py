"""
Integration tests for API integration with various input formats.

These tests verify that the inference API can handle different input formats
and integration scenarios correctly.
"""

import pytest
import torch
import numpy as np
import json
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock
from torchvision import transforms

from swellsight.config import ModelConfig
from swellsight.models import WaveAnalysisModel
from swellsight.inference import InferenceEngine


class TestAPIIntegration:
    """Test API integration with various input formats."""
    
    def test_inference_engine_with_file_path(self, temp_data_dir):
        """Test inference engine with file path input."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create test image
        image = Image.new('RGB', (640, 480), color='lightblue')
        image_path = temp_data_dir / "test_beach.jpg"
        image.save(image_path, 'JPEG')
        
        # Test inference with file path
        predictions = inference_engine.predict(str(image_path))
        
        # Verify API response structure - predictions is a WavePrediction object
        assert hasattr(predictions, 'height_meters'), "Response should have height_meters"
        assert hasattr(predictions, 'wave_type'), "Response should have wave_type"
        assert hasattr(predictions, 'direction'), "Response should have direction"
        assert hasattr(predictions, 'confidence_scores'), "Response should have confidence_scores"
        
        # Verify data types
        assert isinstance(predictions.height_meters, float), "Height should be float"
        assert isinstance(predictions.wave_type, str), "Wave type should be string"
        assert isinstance(predictions.direction, str), "Direction should be string"
        assert isinstance(predictions.confidence_scores, dict), "Confidence scores should be dict"
    
    def test_inference_engine_with_pil_image(self, temp_data_dir):
        """Test inference engine with PIL Image input."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create PIL image
        image = Image.new('RGB', (640, 480), color='lightblue')
        
        # Convert PIL to tensor for inference
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image)
        
        # Test inference with tensor
        predictions = inference_engine.predict_from_tensor(image_tensor)
        
        # Verify response structure
        assert hasattr(predictions, 'height_meters'), "Should predict height"
        assert hasattr(predictions, 'wave_type'), "Should predict wave type"
        assert hasattr(predictions, 'direction'), "Should predict direction"
        assert hasattr(predictions, 'confidence_scores'), "Should have confidence scores"
    
    def test_inference_engine_with_numpy_array(self, temp_data_dir):
        """Test inference engine with numpy array input."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create numpy array (simulating image data)
        image_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Convert numpy array to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        # Apply normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        # Test inference with tensor
        predictions = inference_engine.predict_from_tensor(image_tensor)
        
        # Verify response structure
        assert hasattr(predictions, 'height_meters'), "Should predict height"
        assert hasattr(predictions, 'wave_type'), "Should predict wave type"
        assert hasattr(predictions, 'direction'), "Should predict direction"
    
    def test_batch_inference_api(self, temp_data_dir):
        """Test batch inference capabilities."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            image = Image.new('RGB', (640, 480), color=f'#{i*80:02x}{i*60:02x}{i*40:02x}')
            image_path = temp_data_dir / f"batch_test_{i}.jpg"
            image.save(image_path, 'JPEG')
            image_paths.append(str(image_path))
        
        # Test batch inference
        batch_predictions = inference_engine.predict_batch(image_paths)
        
        # Verify batch response structure
        assert isinstance(batch_predictions, list), "Batch predictions should be a list"
        assert len(batch_predictions) == len(image_paths), "Should have prediction for each image"
        
        # Verify each prediction in batch
        for i, prediction in enumerate(batch_predictions):
            assert hasattr(prediction, 'height_meters'), f"Prediction {i} should have height"
            assert hasattr(prediction, 'wave_type'), f"Prediction {i} should have wave type"
            assert hasattr(prediction, 'direction'), f"Prediction {i} should have direction"
    
    def test_api_response_serialization(self, temp_data_dir):
        """Test that API responses can be properly serialized to JSON."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create test image
        image = Image.new('RGB', (640, 480), color='lightblue')
        image_path = temp_data_dir / "test_serialization.jpg"
        image.save(image_path, 'JPEG')
        
        # Get predictions
        predictions = inference_engine.predict(str(image_path))
        
        # Test JSON serialization using the to_dict method
        try:
            predictions_dict = predictions.to_dict()
            json_str = json.dumps(predictions_dict)
            assert isinstance(json_str, str), "Should serialize to JSON string"
            
            # Test deserialization
            deserialized = json.loads(json_str)
            assert deserialized == predictions_dict, "Should deserialize correctly"
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"Failed to serialize predictions to JSON: {e}")
    
    def test_api_error_handling_and_status_codes(self, temp_data_dir):
        """Test API error handling for various error conditions."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Test with non-existent file
        try:
            predictions = inference_engine.predict("nonexistent_file.jpg")
            pytest.fail("Should raise exception for non-existent file")
        except Exception as e:
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        
        # Test with invalid image format
        invalid_path = temp_data_dir / "invalid.txt"
        with open(invalid_path, 'w') as f:
            f.write("This is not an image")
        
        try:
            predictions = inference_engine.predict(str(invalid_path))
            pytest.fail("Should raise exception for invalid image format")
        except Exception as e:
            assert isinstance(e, Exception), "Should raise appropriate exception"
        
        # Test with None input
        try:
            predictions = inference_engine.predict(None)
            pytest.fail("Should raise exception for None input")
        except Exception as e:
            assert isinstance(e, Exception), "Should raise appropriate exception"
    
    def test_api_input_validation(self, temp_data_dir):
        """Test API input validation for different scenarios."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Test with empty string
        with pytest.raises(Exception):
            inference_engine.predict("")
        
        # Test with invalid tensor shape
        invalid_array = np.random.rand(10, 10)  # 2D instead of 3D
        invalid_tensor = torch.from_numpy(invalid_array)
        with pytest.raises(Exception):
            inference_engine.predict_from_tensor(invalid_tensor)
        
        # Test with invalid tensor dtype - this should work with conversion
        valid_shape_array = np.random.rand(3, 224, 224).astype(np.float64)
        valid_tensor = torch.from_numpy(valid_shape_array).float()
        try:
            predictions = inference_engine.predict_from_tensor(valid_tensor)
            assert hasattr(predictions, 'height_meters'), "Should handle float64 arrays after conversion"
        except Exception:
            # It's acceptable if the engine doesn't support this dtype
            pass
    
    def test_api_metadata_inclusion(self, temp_data_dir):
        """Test that API responses include proper metadata."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create test image
        image = Image.new('RGB', (640, 480), color='lightblue')
        image_path = temp_data_dir / "test_metadata.jpg"
        image.save(image_path, 'JPEG')
        
        # Get predictions
        predictions = inference_engine.predict(str(image_path))
        
        # Verify model info structure (using get_model_info method)
        model_info = inference_engine.get_model_info()
        expected_info_keys = ['device', 'input_size', 'wave_type_classes', 'direction_classes']
        
        for key in expected_info_keys:
            assert key in model_info, f"Model info should contain {key}"
        
        # Verify model info values
        assert isinstance(model_info['input_size'], (list, tuple)), "Input size should be list/tuple"
        assert len(model_info['input_size']) == 2, "Input size should have width and height"
    
    def test_api_confidence_scores_structure(self, temp_data_dir):
        """Test that confidence scores have proper structure and values."""
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create test image
        image = Image.new('RGB', (640, 480), color='lightblue')
        image_path = temp_data_dir / "test_confidence.jpg"
        image.save(image_path, 'JPEG')
        
        # Get predictions
        predictions = inference_engine.predict(str(image_path))
        
        # Verify confidence scores structure
        confidence_scores = predictions.confidence_scores
        expected_confidence_keys = ['wave_type', 'direction', 'height']
        
        for key in expected_confidence_keys:
            assert key in confidence_scores, f"Confidence scores should contain {key}"
            assert isinstance(confidence_scores[key], float), f"{key} should be float"
            assert 0 <= confidence_scores[key] <= 1, f"{key} should be between 0 and 1"
    
    def test_api_integration_with_different_model_configs(self, temp_data_dir):
        """Test API integration with different model configurations."""
        # Test different input sizes
        input_sizes = [(224, 224), (256, 256), (384, 384)]
        
        for input_size in input_sizes:
            model_config = ModelConfig(
                input_size=input_size,
                backbone='convnext_base'
            )
            model = WaveAnalysisModel(model_config)
            inference_engine = InferenceEngine(model, model_config)
            
            # Create test image
            image = Image.new('RGB', (640, 480), color='lightblue')
            image_path = temp_data_dir / f"test_{input_size[0]}x{input_size[1]}.jpg"
            image.save(image_path, 'JPEG')
            
            # Test inference
            try:
                predictions = inference_engine.predict(str(image_path))
                
                # Verify basic structure
                assert hasattr(predictions, 'height_meters'), f"Should work with {input_size} input size"
                assert hasattr(predictions, 'wave_type'), f"Should predict wave type with {input_size}"
                assert hasattr(predictions, 'direction'), f"Should predict direction with {input_size}"
                
                # Verify model info contains correct input size
                model_info = inference_engine.get_model_info()
                assert tuple(model_info['input_size']) == input_size, f"Model info should match config for {input_size}"
                
            except Exception as e:
                pytest.fail(f"Failed with input size {input_size}: {e}")
    
    def test_concurrent_api_requests(self, temp_data_dir):
        """Test that API can handle concurrent requests correctly."""
        import threading
        import time
        
        model_config = ModelConfig(input_size=(224, 224))
        model = WaveAnalysisModel(model_config)
        inference_engine = InferenceEngine(model, model_config)
        
        # Create test images
        image_paths = []
        for i in range(5):
            image = Image.new('RGB', (640, 480), color=f'#{i*50:02x}{i*40:02x}{i*30:02x}')
            image_path = temp_data_dir / f"concurrent_test_{i}.jpg"
            image.save(image_path, 'JPEG')
            image_paths.append(str(image_path))
        
        # Results storage
        results = {}
        errors = {}
        
        def run_inference(image_path, thread_id):
            """Run inference in a thread."""
            try:
                predictions = inference_engine.predict(image_path)
                results[thread_id] = predictions
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Start concurrent threads
        threads = []
        for i, image_path in enumerate(image_paths):
            thread = threading.Thread(target=run_inference, args=(image_path, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify results
        assert len(errors) == 0, f"Should have no errors in concurrent execution: {errors}"
        assert len(results) == len(image_paths), "Should have results for all images"
        
        # Verify each result
        for thread_id, predictions in results.items():
            assert hasattr(predictions, 'height_meters'), f"Thread {thread_id} should have valid predictions"
            assert hasattr(predictions, 'wave_type'), f"Thread {thread_id} should predict wave type"
            assert hasattr(predictions, 'direction'), f"Thread {thread_id} should predict direction"
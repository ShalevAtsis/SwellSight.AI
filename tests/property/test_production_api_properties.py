"""Property-based tests for production API."""

import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np

from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config import ModelConfig
from swellsight.utils.model_versioning import ModelVersionManager
from swellsight.api.production_api import create_app


class TestProductionAPIProperties:
    """Property-based tests for production API."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create model and version manager
        self.config = ModelConfig()
        self.model = WaveAnalysisModel(self.config)
        self.model.eval()
        
        # Create version manager and register model
        self.version_manager = ModelVersionManager(self.temp_dir / "registry")
        self.version_manager.register_model(
            model=self.model,
            version="1.0.0",
            performance_metrics={"accuracy": 0.85},
            training_metadata={"test": True}
        )
        
        # Create FastAPI app
        self.app = create_app(
            registry_path=str(self.temp_dir / "registry"),
            enable_auth=False,
            cors_origins=["*"]
        )
        
        # Create test client
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, width: int = 256, height: int = 256, format: str = "JPEG") -> bytes:
        """Create a test image for API testing."""
        # Create random RGB image
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    @given(
        image_width=st.integers(min_value=64, max_value=512),
        image_height=st.integers(min_value=64, max_value=512),
        image_format=st.sampled_from(["JPEG", "PNG"])
    )
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_property_35_api_response_time(self, image_width, image_height, image_format):
        """
        Feature: wave-analysis-model, Property 35: API Response Time
        
        For any valid inference request, the API should respond within 2 seconds.
        
        **Validates: Requirements 5.3, 11.3**
        """
        # Create test image
        image_data = self.create_test_image(image_width, image_height, image_format)
        
        # Prepare file upload
        files = {
            "image": ("test_image.jpg", image_data, "image/jpeg")
        }
        
        # Measure response time
        start_time = time.time()
        
        response = self.client.post("/predict", files=files)
        
        response_time = time.time() - start_time
        
        # Verify response time is within 2 seconds
        assert response_time < 2.0, f"API response time {response_time:.2f}s exceeds 2 second threshold"
        
        # Verify successful response
        assert response.status_code == 200, f"API request failed with status {response.status_code}"
        
        # Verify response structure
        response_data = response.json()
        assert "request_id" in response_data
        assert "height_meters" in response_data
        assert "wave_type" in response_data
        assert "direction" in response_data
        assert "processing_time_ms" in response_data
        
        # Verify processing time is reasonable
        processing_time_ms = response_data["processing_time_ms"]
        assert processing_time_ms < 2000, f"Processing time {processing_time_ms}ms exceeds 2 second threshold"
        
        # Verify confidence scores are present and valid
        assert "confidence_scores" in response_data
        confidence_scores = response_data["confidence_scores"]
        assert isinstance(confidence_scores, dict)
        
        for score_name, score_value in confidence_scores.items():
            assert isinstance(score_value, (int, float))
            assert 0.0 <= score_value <= 1.0, f"Confidence score {score_name} = {score_value} not in [0,1] range"
        
        print(f"✅ API response time test passed: {response_time:.3f}s (processing: {processing_time_ms:.1f}ms)")
    
    @given(
        corruption_type=st.sampled_from(["empty", "truncated", "invalid_format", "too_large"])
    )
    @settings(max_examples=4, deadline=None)
    def test_property_36_data_quality_validation(self, corruption_type):
        """
        Feature: wave-analysis-model, Property 36: Data Quality Validation
        
        For any input image, the system should correctly identify corrupted or invalid images.
        
        **Validates: Requirements 5.3, 11.3, 12.1, 12.2**
        """
        # Create corrupted image data based on corruption type
        if corruption_type == "empty":
            image_data = b""
            expected_status = 400
            
        elif corruption_type == "truncated":
            # Create valid image then truncate it
            valid_image = self.create_test_image(256, 256, "JPEG")
            image_data = valid_image[:len(valid_image)//2]  # Truncate to half
            expected_status = 400
            
        elif corruption_type == "invalid_format":
            # Create random bytes that aren't a valid image
            image_data = b"This is not an image file" * 100
            expected_status = 400
            
        elif corruption_type == "too_large":
            # Create an image that's too large (simulate by creating large random data)
            image_data = b"x" * (11 * 1024 * 1024)  # 11MB (exceeds 10MB limit)
            expected_status = 400
        
        # Prepare file upload
        files = {
            "image": ("corrupted_image.jpg", image_data, "image/jpeg")
        }
        
        # Send request
        response = self.client.post("/predict", files=files)
        
        # Verify that corrupted/invalid images are properly rejected
        assert response.status_code == expected_status, \
            f"Expected status {expected_status} for {corruption_type} image, got {response.status_code}"
        
        # Verify error response structure
        if response.status_code != 200:
            response_data = response.json()
            assert "detail" in response_data, "Error response should contain 'detail' field"
            assert isinstance(response_data["detail"], str), "Error detail should be a string"
            assert len(response_data["detail"]) > 0, "Error detail should not be empty"
        
        print(f"✅ Data quality validation test passed for {corruption_type} image")
    
    def test_api_health_check_functionality(self):
        """Test API health check endpoint functionality."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "message" in health_data
        assert "metrics" in health_data
        assert "timestamp" in health_data
        
        # Status should be one of the expected values
        assert health_data["status"] in ["healthy", "degraded", "unhealthy", "unknown"]
        
        print("✅ Health check functionality test passed")
    
    def test_api_metrics_endpoint(self):
        """Test API metrics collection and reporting."""
        # Make a few requests to generate metrics
        image_data = self.create_test_image()
        files = {"image": ("test.jpg", image_data, "image/jpeg")}
        
        for _ in range(3):
            self.client.post("/predict", files=files)
        
        # Check metrics endpoint
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert "metrics" in metrics_data
        assert "recent_alerts" in metrics_data
        assert "timestamp" in metrics_data
        
        # Verify metrics structure
        metrics = metrics_data["metrics"]
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "average_response_time" in metrics
        
        # Should have recorded our test requests
        assert metrics["total_requests"] >= 3
        
        print("✅ Metrics endpoint test passed")
    
    def test_api_model_info_endpoint(self):
        """Test model information endpoint."""
        response = self.client.get("/model/info")
        
        assert response.status_code == 200
        
        info_data = response.json()
        assert "model_info" in info_data
        assert "timestamp" in info_data
        
        model_info = info_data["model_info"]
        assert "device" in model_info
        assert "input_size" in model_info
        assert "wave_type_classes" in model_info
        assert "direction_classes" in model_info
        assert "supported_formats" in model_info
        
        print("✅ Model info endpoint test passed")
    
    def test_api_rate_limiting(self):
        """Test API rate limiting functionality."""
        image_data = self.create_test_image()
        files = {"image": ("test.jpg", image_data, "image/jpeg")}
        
        # Make many requests quickly to trigger rate limiting
        # Note: This test might be flaky depending on rate limit settings
        responses = []
        for i in range(10):
            response = self.client.post("/predict", files=files)
            responses.append(response.status_code)
        
        # At least some requests should succeed
        successful_requests = sum(1 for status in responses if status == 200)
        assert successful_requests > 0, "At least some requests should succeed"
        
        print(f"✅ Rate limiting test passed: {successful_requests}/10 requests succeeded")
    
    def test_api_error_handling_robustness(self):
        """Test API error handling for various invalid inputs."""
        # Test missing file
        response = self.client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity
        
        # Test invalid file type
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = self.client.post("/predict", files=files)
        assert response.status_code == 400
        
        # Test non-existent endpoint
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
        
        print("✅ Error handling robustness test passed")
    
    @given(
        num_images=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_prediction_consistency(self, num_images):
        """Test batch prediction functionality and consistency."""
        # Create multiple test images
        images = []
        for i in range(num_images):
            image_data = self.create_test_image(256, 256, "JPEG")
            images.append(("image", (f"test_{i}.jpg", image_data, "image/jpeg")))
        
        # Send batch request
        response = self.client.post("/predict/batch", files=images)
        
        # Verify response
        assert response.status_code == 200
        
        batch_data = response.json()
        assert "request_id" in batch_data
        assert "results" in batch_data
        assert "total_images" in batch_data
        assert "successful_predictions" in batch_data
        
        # Verify results structure
        results = batch_data["results"]
        assert len(results) == num_images
        
        for i, result in enumerate(results):
            assert "image_index" in result
            assert result["image_index"] == i
            assert "filename" in result
            assert "processing_time_ms" in result
            
            # Should have either prediction or error
            assert "prediction" in result or "error" in result
        
        print(f"✅ Batch prediction test passed for {num_images} images")
    
    def test_concurrent_request_handling(self):
        """Test API handling of concurrent requests."""
        import threading
        import queue
        
        image_data = self.create_test_image()
        files = {"image": ("test.jpg", image_data, "image/jpeg")}
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = self.client.post("/predict", files=files)
                results.put(("success", response.status_code, response.json()))
            except Exception as e:
                results.put(("error", str(e), None))
        
        # Create multiple threads for concurrent requests
        threads = []
        num_threads = 3
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        successful_requests = 0
        while not results.empty():
            result_type, status_or_error, data = results.get()
            if result_type == "success" and status_or_error == 200:
                successful_requests += 1
        
        # At least some concurrent requests should succeed
        assert successful_requests > 0, f"No concurrent requests succeeded out of {num_threads}"
        
        print(f"✅ Concurrent request handling test passed: {successful_requests}/{num_threads} succeeded")
    
    def test_api_response_format_consistency(self):
        """Test that API responses maintain consistent format."""
        image_data = self.create_test_image()
        files = {"image": ("test.jpg", image_data, "image/jpeg")}
        
        # Make multiple requests
        responses = []
        for _ in range(3):
            response = self.client.post("/predict", files=files)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Verify all responses have the same structure
        required_fields = [
            "request_id", "height_meters", "wave_type", "direction",
            "wave_type_probabilities", "direction_probabilities",
            "confidence_scores", "processing_time_ms", "model_version", "timestamp"
        ]
        
        for response_data in responses:
            for field in required_fields:
                assert field in response_data, f"Missing required field: {field}"
            
            # Verify data types
            assert isinstance(response_data["height_meters"], (int, float))
            assert isinstance(response_data["wave_type"], str)
            assert isinstance(response_data["direction"], str)
            assert isinstance(response_data["wave_type_probabilities"], dict)
            assert isinstance(response_data["direction_probabilities"], dict)
            assert isinstance(response_data["confidence_scores"], dict)
            assert isinstance(response_data["processing_time_ms"], (int, float))
            assert isinstance(response_data["model_version"], str)
            assert isinstance(response_data["timestamp"], str)
        
        print("✅ API response format consistency test passed")
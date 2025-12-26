"""Property-based tests for model architecture."""

import torch
import torch.nn.functional as F
from hypothesis import given, strategies as st, settings
import pytest

from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config import ModelConfig


class TestModelArchitectureProperties:
    """Property-based tests for WaveAnalysisModel architecture."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        height=st.just(768),  # Fixed input size as per requirements
        width=st.just(768)
    )
    @settings(max_examples=5, deadline=None)  # Reduced examples, no deadline for model loading
    def test_model_input_output_consistency(self, batch_size, height, width):
        """
        Feature: wave-analysis-model, Property 1: Model Input/Output Consistency
        
        For any RGB image of size 768x768 pixels, the Wave_Analysis_Model should produce 
        exactly three outputs: a height scalar, a 4-dimensional wave type probability vector, 
        and a 3-dimensional direction probability vector.
        
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6**
        """
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Create input tensor with specified dimensions
        input_tensor = torch.randn(batch_size, 3, height, width)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Verify exactly three outputs exist
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        assert 'height' in outputs, "Missing 'height' output"
        assert 'wave_type' in outputs, "Missing 'wave_type' output"
        assert 'direction' in outputs, "Missing 'direction' output"
        
        # Verify height output shape (scalar per batch item)
        height_output = outputs['height']
        assert height_output.shape == (batch_size, 1), \
            f"Height output shape should be ({batch_size}, 1), got {height_output.shape}"
        
        # Verify wave type output shape (4-dimensional vector per batch item)
        wave_type_output = outputs['wave_type']
        assert wave_type_output.shape == (batch_size, 4), \
            f"Wave type output shape should be ({batch_size}, 4), got {wave_type_output.shape}"
        
        # Verify direction output shape (3-dimensional vector per batch item)
        direction_output = outputs['direction']
        assert direction_output.shape == (batch_size, 3), \
            f"Direction output shape should be ({batch_size}, 3), got {direction_output.shape}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=5, deadline=None)
    def test_feature_extractor_dimensionality(self, batch_size):
        """
        Feature: wave-analysis-model, Property 2: Feature Extractor Dimensionality
        
        For any valid input image, the Feature_Extractor should output a feature vector 
        of exactly 2048 dimensions.
        
        **Validates: Requirements 1.2**
        """
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Create input tensor with correct dimensions
        input_tensor = torch.randn(batch_size, 3, 768, 768)
        
        with torch.no_grad():
            # Extract features using backbone
            features = model.backbone(input_tensor)
            features = model.feature_projection(features)
        
        # Verify feature dimension is exactly 2048
        assert features.shape == (batch_size, 2048), \
            f"Feature extractor should output ({batch_size}, 2048), got {features.shape}"
        
        # Verify the model reports correct feature dimension
        assert model.get_feature_extractor_output_dim() == 2048, \
            f"Model should report feature dim as 2048, got {model.get_feature_extractor_output_dim()}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=5, deadline=None)
    def test_probability_vector_validity(self, batch_size):
        """
        Feature: wave-analysis-model, Property 3: Probability Vector Validity
        
        For any model output, both classification heads should produce probability vectors 
        that sum to 1.0 (±1e-6 tolerance) after applying softmax.
        
        **Validates: Requirements 1.5, 1.6**
        """
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, 3, 768, 768)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Apply softmax to get probabilities
            wave_type_probs = F.softmax(outputs['wave_type'], dim=1)
            direction_probs = F.softmax(outputs['direction'], dim=1)
        
        # Verify wave type probabilities sum to 1.0
        wave_type_sums = wave_type_probs.sum(dim=1)
        for i, prob_sum in enumerate(wave_type_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-6, \
                f"Wave type probabilities for batch item {i} sum to {prob_sum.item()}, expected 1.0"
        
        # Verify direction probabilities sum to 1.0
        direction_sums = direction_probs.sum(dim=1)
        for i, prob_sum in enumerate(direction_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-6, \
                f"Direction probabilities for batch item {i} sum to {prob_sum.item()}, expected 1.0"
        
        # Verify all probabilities are non-negative
        assert (wave_type_probs >= 0).all(), "Wave type probabilities should be non-negative"
        assert (direction_probs >= 0).all(), "Direction probabilities should be non-negative"
        
        # Verify all probabilities are <= 1.0
        assert (wave_type_probs <= 1.0).all(), "Wave type probabilities should be <= 1.0"
        assert (direction_probs <= 1.0).all(), "Direction probabilities should be <= 1.0"


# Additional test for model instantiation with different configurations
class TestModelConfigurationProperties:
    """Property-based tests for model configuration handling."""
    
    @given(
        feature_dim=st.integers(min_value=512, max_value=4096),
        hidden_dim=st.integers(min_value=128, max_value=1024),
        dropout_rate=st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=3, deadline=None)
    def test_model_configuration_flexibility(self, feature_dim, hidden_dim, dropout_rate):
        """
        Test that model can be instantiated with various valid configurations.
        
        This ensures the model architecture is flexible and can handle different
        hyperparameter settings.
        """
        config = ModelConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Model should instantiate without errors
        model = WaveAnalysisModel(config)
        
        # Verify configuration is stored correctly
        assert model.config.feature_dim == feature_dim
        assert model.config.hidden_dim == hidden_dim
        assert model.config.dropout_rate == dropout_rate
        
        # Verify model can perform forward pass
        input_tensor = torch.randn(1, 3, 768, 768)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Basic output validation
        assert len(outputs) == 3
        assert outputs['height'].shape == (1, 1)
        assert outputs['wave_type'].shape == (1, 4)
        assert outputs['direction'].shape == (1, 3)
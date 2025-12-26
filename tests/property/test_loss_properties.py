"""Property-based tests for multi-task loss function."""

import torch
from hypothesis import given, strategies as st, settings
import pytest

from swellsight.models.multi_task_loss import MultiTaskLoss


class TestMultiTaskLossProperties:
    """Property-based tests for MultiTaskLoss."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        height_values=st.lists(
            st.floats(min_value=0.3, max_value=4.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=16
        ),
        wave_type_indices=st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=1, max_size=16
        ),
        direction_indices=st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=1, max_size=16
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_multi_task_loss_composition(self, batch_size, height_values, wave_type_indices, direction_indices):
        """
        Feature: wave-analysis-model, Property 8: Multi-Task Loss Composition
        
        For any training batch, the total loss should be computed as a weighted sum 
        of height regression loss, wave type classification loss, and direction 
        classification loss.
        
        **Validates: Requirements 3.1, 3.2**
        """
        # Ensure all lists have the same length as batch_size
        height_values = height_values[:batch_size] + [1.5] * max(0, batch_size - len(height_values))
        wave_type_indices = wave_type_indices[:batch_size] + [0] * max(0, batch_size - len(wave_type_indices))
        direction_indices = direction_indices[:batch_size] + [0] * max(0, batch_size - len(direction_indices))
        
        # Create loss function
        loss_fn = MultiTaskLoss()
        
        # Create mock predictions (random logits)
        predictions = {
            'height': torch.randn(batch_size, 1),
            'wave_type': torch.randn(batch_size, 4),
            'direction': torch.randn(batch_size, 3)
        }
        
        # Create targets
        targets = {
            'height': torch.tensor(height_values[:batch_size], dtype=torch.float32),
            'wave_type': torch.tensor(wave_type_indices[:batch_size], dtype=torch.long),
            'direction': torch.tensor(direction_indices[:batch_size], dtype=torch.long)
        }
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        
        # Verify all required components are present
        assert 'total_loss' in loss_dict, "Missing 'total_loss' in output"
        assert 'height_loss' in loss_dict, "Missing 'height_loss' in output"
        assert 'wave_type_loss' in loss_dict, "Missing 'wave_type_loss' in output"
        assert 'direction_loss' in loss_dict, "Missing 'direction_loss' in output"
        assert 'loss_weights' in loss_dict, "Missing 'loss_weights' in output"
        
        # Verify individual losses are non-negative
        assert loss_dict['height_loss'].item() >= 0, "Height loss should be non-negative"
        assert loss_dict['wave_type_loss'].item() >= 0, "Wave type loss should be non-negative"
        assert loss_dict['direction_loss'].item() >= 0, "Direction loss should be non-negative"
        
        # Verify total loss is non-negative
        assert loss_dict['total_loss'].item() >= 0, "Total loss should be non-negative"
        
        # Verify loss weights are positive (we use absolute values)
        weights = loss_fn.get_loss_weights()
        assert weights['height_weight'] > 0, "Height weight should be positive"
        assert weights['wave_type_weight'] > 0, "Wave type weight should be positive"
        assert weights['direction_weight'] > 0, "Direction weight should be positive"
        
        # Verify the composition property: total loss should be approximately equal to
        # the weighted sum of individual losses (within numerical precision)
        expected_total = (
            weights['height_weight'] * loss_dict['height_loss'].item() +
            weights['wave_type_weight'] * loss_dict['wave_type_loss'].item() +
            weights['direction_weight'] * loss_dict['direction_loss'].item()
        )
        
        actual_total = loss_dict['total_loss'].item()
        
        # Allow for small numerical differences
        assert abs(actual_total - expected_total) < 1e-6, \
            f"Total loss {actual_total} should equal weighted sum {expected_total}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=5, deadline=None)
    def test_loss_function_types(self, batch_size):
        """
        Test that the correct loss function types are used for each task.
        
        - Height regression should use SmoothL1Loss
        - Classification tasks should use CrossEntropyLoss
        """
        loss_fn = MultiTaskLoss()
        
        # Create predictions and targets
        predictions = {
            'height': torch.randn(batch_size, 1),
            'wave_type': torch.randn(batch_size, 4),
            'direction': torch.randn(batch_size, 3)
        }
        
        targets = {
            'height': torch.rand(batch_size) * 3.7 + 0.3,  # Random heights in valid range
            'wave_type': torch.randint(0, 4, (batch_size,)),
            'direction': torch.randint(0, 3, (batch_size,))
        }
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        
        # Verify that loss functions are of correct types
        assert isinstance(loss_fn.height_loss_fn, torch.nn.SmoothL1Loss), \
            "Height loss should use SmoothL1Loss"
        assert isinstance(loss_fn.wave_type_loss_fn, torch.nn.CrossEntropyLoss), \
            "Wave type loss should use CrossEntropyLoss"
        assert isinstance(loss_fn.direction_loss_fn, torch.nn.CrossEntropyLoss), \
            "Direction loss should use CrossEntropyLoss"
        
        # Verify loss values are reasonable (finite and non-negative)
        assert torch.isfinite(loss_dict['total_loss']), "Total loss should be finite"
        assert torch.isfinite(loss_dict['height_loss']), "Height loss should be finite"
        assert torch.isfinite(loss_dict['wave_type_loss']), "Wave type loss should be finite"
        assert torch.isfinite(loss_dict['direction_loss']), "Direction loss should be finite"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=5, deadline=None)
    def test_learnable_weights_optimization(self, batch_size):
        """
        Test that loss weights are learnable parameters that can be optimized.
        """
        loss_fn = MultiTaskLoss()
        
        # Verify weights are parameters
        assert isinstance(loss_fn.height_weight, torch.nn.Parameter), \
            "Height weight should be a learnable parameter"
        assert isinstance(loss_fn.wave_type_weight, torch.nn.Parameter), \
            "Wave type weight should be a learnable parameter"
        assert isinstance(loss_fn.direction_weight, torch.nn.Parameter), \
            "Direction weight should be a learnable parameter"
        
        # Verify weights require gradients
        assert loss_fn.height_weight.requires_grad, "Height weight should require gradients"
        assert loss_fn.wave_type_weight.requires_grad, "Wave type weight should require gradients"
        assert loss_fn.direction_weight.requires_grad, "Direction weight should require gradients"
        
        # Create predictions and targets
        predictions = {
            'height': torch.randn(batch_size, 1, requires_grad=True),
            'wave_type': torch.randn(batch_size, 4, requires_grad=True),
            'direction': torch.randn(batch_size, 3, requires_grad=True)
        }
        
        targets = {
            'height': torch.rand(batch_size) * 3.7 + 0.3,
            'wave_type': torch.randint(0, 4, (batch_size,)),
            'direction': torch.randint(0, 3, (batch_size,))
        }
        
        # Compute loss and backpropagate
        loss_dict = loss_fn(predictions, targets)
        loss_dict['total_loss'].backward()
        
        # Verify that weights have gradients after backpropagation
        assert loss_fn.height_weight.grad is not None, "Height weight should have gradients"
        assert loss_fn.wave_type_weight.grad is not None, "Wave type weight should have gradients"
        assert loss_fn.direction_weight.grad is not None, "Direction weight should have gradients"
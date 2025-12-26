"""Property-based tests for evaluation metrics."""

import numpy as np
import torch
from hypothesis import given, strategies as st, settings
import pytest
from typing import List, Dict, Any, Union

from swellsight.evaluation.metrics_calculator import MetricsCalculator


class TestEvaluationMetricsProperties:
    """Property-based tests for evaluation metrics computation."""
    
    @given(
        num_samples=st.integers(min_value=10, max_value=100),
        height_range=st.tuples(
            st.floats(min_value=0.3, max_value=2.0),
            st.floats(min_value=2.1, max_value=4.0)
        ),
        wave_type_classes=st.integers(min_value=0, max_value=3),
        direction_classes=st.integers(min_value=0, max_value=2)
    )
    @settings(max_examples=10, deadline=None)
    def test_metrics_computation_correctness(self, num_samples, height_range, wave_type_classes, direction_classes):
        """
        Feature: wave-analysis-model, Property 14: Metrics Computation Correctness
        
        For any set of predictions and ground truth values, the evaluation system should 
        compute MAE, RMSE for regression and accuracy, F1-score for classification tasks 
        using standard formulas.
        
        **Validates: Requirements 6.1, 6.2**
        """
        calculator = MetricsCalculator()
        
        # Generate synthetic predictions and targets
        min_height, max_height = height_range
        
        # Get class names from calculator
        wave_type_classes = calculator.wave_type_classes
        direction_classes = calculator.direction_classes
        
        # Create predictions in expected format
        predictions = []
        targets = []
        
        for i in range(num_samples):
            # Generate random but valid predictions
            pred_height = np.random.uniform(min_height, max_height)
            pred_wave_type_idx = np.random.randint(0, 4)
            pred_direction_idx = np.random.randint(0, 3)
            
            # Generate corresponding targets (with some noise for realistic testing)
            target_height = pred_height + np.random.normal(0, 0.1)  # Small noise
            target_wave_type_idx = np.random.randint(0, 4)
            target_direction_idx = np.random.randint(0, 3)
            
            # Format as expected by MetricsCalculator (predictions as indices, targets as strings)
            predictions.append({
                'height_meters': pred_height,
                'wave_type': pred_wave_type_idx,
                'direction': pred_direction_idx
            })
            
            targets.append({
                'height_meters': target_height,
                'wave_type': wave_type_classes[target_wave_type_idx],
                'direction': direction_classes[target_direction_idx]
            })
        
        # Calculate metrics using our implementation
        metrics = calculator.calculate_all_metrics(predictions, targets, "test")
        
        # Verify metrics structure and validity
        assert 'height' in metrics, "Height metrics missing"
        assert 'wave_type' in metrics, "Wave type metrics missing"
        assert 'direction' in metrics, "Direction metrics missing"
        
        # Verify height regression metrics
        height_metrics = metrics['height']
        assert 'mae' in height_metrics, "MAE missing from height metrics"
        assert 'rmse' in height_metrics, "RMSE missing from height metrics"
        
        # MAE and RMSE should be non-negative
        assert height_metrics['mae'] >= 0, f"MAE should be non-negative, got {height_metrics['mae']}"
        assert height_metrics['rmse'] >= 0, f"RMSE should be non-negative, got {height_metrics['rmse']}"
        
        # RMSE should be >= MAE (mathematical property)
        assert height_metrics['rmse'] >= height_metrics['mae'], \
            f"RMSE ({height_metrics['rmse']}) should be >= MAE ({height_metrics['mae']})"
        
        # Verify classification metrics
        for task in ['wave_type', 'direction']:
            task_metrics = metrics[task]
            assert 'accuracy' in task_metrics, f"Accuracy missing from {task} metrics"
            assert 'f1_score' in task_metrics, f"F1-score missing from {task} metrics"
            
            # Accuracy and F1-score should be in [0, 1]
            assert 0 <= task_metrics['accuracy'] <= 1, \
                f"{task} accuracy should be in [0,1], got {task_metrics['accuracy']}"
            assert 0 <= task_metrics['f1_score'] <= 1, \
                f"{task} F1-score should be in [0,1], got {task_metrics['f1_score']}"
        
        # Verify manual computation of MAE matches
        pred_heights = np.array([p['height_meters'] for p in predictions])
        target_heights = np.array([t['height_meters'] for t in targets])
        expected_mae = np.mean(np.abs(pred_heights - target_heights))
        
        # Allow small numerical differences
        assert abs(height_metrics['mae'] - expected_mae) < 1e-6, \
            f"Computed MAE ({height_metrics['mae']}) doesn't match expected ({expected_mae})"
        
        # Verify manual computation of RMSE matches
        expected_rmse = np.sqrt(np.mean((pred_heights - target_heights) ** 2))
        assert abs(height_metrics['rmse'] - expected_rmse) < 1e-6, \
            f"Computed RMSE ({height_metrics['rmse']}) doesn't match expected ({expected_rmse})"
    
    @given(
        synthetic_samples=st.integers(min_value=5, max_value=50),
        real_samples=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=10, deadline=None)
    def test_dataset_separation(self, synthetic_samples, real_samples):
        """
        Feature: wave-analysis-model, Property 15: Dataset Separation
        
        For any evaluation run, metrics should be computed separately for synthetic 
        validation data and real-world test data with no data leakage between sets.
        
        **Validates: Requirements 6.3**
        """
        calculator = MetricsCalculator()
        
        # Get class names from calculator
        wave_type_classes = calculator.wave_type_classes
        direction_classes = calculator.direction_classes
        
        # Generate synthetic dataset
        synthetic_predictions = []
        synthetic_targets = []
        
        for i in range(synthetic_samples):
            pred_wave_type_idx = np.random.randint(0, 4)
            pred_direction_idx = np.random.randint(0, 3)
            target_wave_type_idx = np.random.randint(0, 4)
            target_direction_idx = np.random.randint(0, 3)
            
            synthetic_predictions.append({
                'height_meters': np.random.uniform(0.5, 3.0),
                'wave_type': pred_wave_type_idx,
                'direction': pred_direction_idx
            })
            
            synthetic_targets.append({
                'height_meters': np.random.uniform(0.5, 3.0),
                'wave_type': wave_type_classes[target_wave_type_idx],
                'direction': direction_classes[target_direction_idx]
            })
        
        # Generate real dataset (different characteristics to ensure separation)
        real_predictions = []
        real_targets = []
        
        for i in range(real_samples):
            pred_wave_type_idx = np.random.randint(0, 4)
            pred_direction_idx = np.random.randint(0, 3)
            target_wave_type_idx = np.random.randint(0, 4)
            target_direction_idx = np.random.randint(0, 3)
            
            # Use slightly different ranges to verify separation
            real_predictions.append({
                'height_meters': np.random.uniform(1.0, 4.0),
                'wave_type': pred_wave_type_idx,
                'direction': pred_direction_idx
            })
            
            real_targets.append({
                'height_meters': np.random.uniform(1.0, 4.0),
                'wave_type': wave_type_classes[target_wave_type_idx],
                'direction': direction_classes[target_direction_idx]
            })
        
        # Calculate metrics separately for each dataset
        synthetic_metrics = calculator.calculate_all_metrics(
            synthetic_predictions, synthetic_targets, "synthetic"
        )
        
        real_metrics = calculator.calculate_all_metrics(
            real_predictions, real_targets, "real"
        )
        
        # Verify that metrics are computed independently
        # (they should be different since we used different data)
        assert synthetic_metrics != real_metrics, \
            "Synthetic and real metrics should be computed independently"
        
        # Verify both metric sets have the same structure
        for dataset_name, metrics in [("synthetic", synthetic_metrics), ("real", real_metrics)]:
            assert 'height' in metrics, f"Height metrics missing from {dataset_name} dataset"
            assert 'wave_type' in metrics, f"Wave type metrics missing from {dataset_name} dataset"
            assert 'direction' in metrics, f"Direction metrics missing from {dataset_name} dataset"
            
            # Verify all metrics are valid numbers
            for task_name, task_metrics in metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    assert isinstance(metric_value, (int, float)), \
                        f"{dataset_name} {task_name} {metric_name} should be numeric, got {type(metric_value)}"
                    assert not np.isnan(metric_value), \
                        f"{dataset_name} {task_name} {metric_name} should not be NaN"
                    assert not np.isinf(metric_value), \
                        f"{dataset_name} {task_name} {metric_name} should not be infinite"
        
        # Test the detailed metrics calculation as well
        synthetic_detailed = calculator.calculate_detailed_metrics(
            synthetic_predictions, synthetic_targets, "synthetic"
        )
        
        real_detailed = calculator.calculate_detailed_metrics(
            real_predictions, real_targets, "real"
        )
        
        # Verify dataset type is correctly tracked
        assert synthetic_detailed.dataset_type == "synthetic", \
            f"Synthetic dataset type should be 'synthetic', got '{synthetic_detailed.dataset_type}'"
        assert real_detailed.dataset_type == "real", \
            f"Real dataset type should be 'real', got '{real_detailed.dataset_type}'"
        
        # Verify sample counts are correct
        assert synthetic_detailed.num_samples == synthetic_samples, \
            f"Synthetic sample count should be {synthetic_samples}, got {synthetic_detailed.num_samples}"
        assert real_detailed.num_samples == real_samples, \
            f"Real sample count should be {real_samples}, got {real_detailed.num_samples}"
        
        # Verify no data leakage by ensuring different results
        # (this is a probabilistic test, but with different ranges it should almost always pass)
        synthetic_mae = synthetic_detailed.height_metrics.mae
        real_mae = real_detailed.height_metrics.mae
        
        # The MAEs should be different (with high probability) since we used different data ranges
        # We allow them to be equal only if both are very small (edge case)
        if synthetic_mae > 0.01 and real_mae > 0.01:
            assert abs(synthetic_mae - real_mae) > 1e-10, \
                "Synthetic and real MAE should be different, indicating proper dataset separation"


class TestEvaluationEdgeCases:
    """Test edge cases for evaluation metrics."""
    
    def test_empty_predictions_handling(self):
        """Test that empty predictions are handled gracefully."""
        calculator = MetricsCalculator()
        
        # Test with empty lists
        metrics = calculator.calculate_all_metrics([], [], "empty")
        
        # Should return zero metrics without crashing
        assert metrics['height']['mae'] == 0.0
        assert metrics['height']['rmse'] == 0.0
        assert metrics['wave_type']['accuracy'] == 0.0
        assert metrics['wave_type']['f1_score'] == 0.0
        assert metrics['direction']['accuracy'] == 0.0
        assert metrics['direction']['f1_score'] == 0.0
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        calculator = MetricsCalculator()
        
        # Get class names from calculator
        wave_type_classes = calculator.wave_type_classes
        direction_classes = calculator.direction_classes
        
        # Create perfect predictions (identical to targets)
        predictions = [
            {'height_meters': 1.5, 'wave_type': 0, 'direction': 1},
            {'height_meters': 2.0, 'wave_type': 1, 'direction': 2},
            {'height_meters': 1.0, 'wave_type': 2, 'direction': 0}
        ]
        
        targets = [
            {'height_meters': 1.5, 'wave_type': wave_type_classes[0], 'direction': direction_classes[1]},
            {'height_meters': 2.0, 'wave_type': wave_type_classes[1], 'direction': direction_classes[2]},
            {'height_meters': 1.0, 'wave_type': wave_type_classes[2], 'direction': direction_classes[0]}
        ]
        
        metrics = calculator.calculate_all_metrics(predictions, targets, "perfect")
        
        # Perfect predictions should have zero error and 100% accuracy
        assert abs(metrics['height']['mae']) < 1e-10, "Perfect height predictions should have MAE ≈ 0"
        assert abs(metrics['height']['rmse']) < 1e-10, "Perfect height predictions should have RMSE ≈ 0"
        assert abs(metrics['wave_type']['accuracy'] - 1.0) < 1e-10, "Perfect wave type predictions should have accuracy = 1.0"
        assert abs(metrics['direction']['accuracy'] - 1.0) < 1e-10, "Perfect direction predictions should have accuracy = 1.0"
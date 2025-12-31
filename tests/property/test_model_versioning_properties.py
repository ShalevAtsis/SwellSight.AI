"""Property-based tests for model versioning system."""

import torch
import tempfile
import shutil
import uuid
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
import pytest
import semantic_version

from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config import ModelConfig
from swellsight.utils.model_versioning import (
    ModelVersionManager, 
    ABTestManager,
    ModelVersion,
    ModelComparison
)
from swellsight.utils.performance_benchmarks import PerformanceBenchmark


class TestModelVersioningProperties:
    """Property-based tests for model versioning system."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create unique registry path for each test to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        self.version_manager = ModelVersionManager(self.temp_dir / f"registry_{unique_id}")
        self.benchmark = PerformanceBenchmark(self.temp_dir / f"benchmarks_{unique_id}")
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(
        major=st.integers(min_value=1, max_value=2),
        minor=st.integers(min_value=0, max_value=2),
        patch=st.integers(min_value=0, max_value=2),
        batch_size=st.integers(min_value=1, max_value=2)
    )
    @settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_property_34_model_version_consistency(self, major, minor, patch, batch_size):
        """
        Feature: wave-analysis-model, Property 34: Model Version Consistency
        
        For any model version, loading and inference should produce identical results 
        across different environments.
        
        **Validates: Requirements 10.1, 10.2, 11.1**
        """
        # Create semantic version
        version = f"{major}.{minor}.{patch}"
        
        # Create and train model
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Create test input
        test_input = torch.randn(batch_size, 3, 768, 768)
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = model(test_input)
        
        # Register model version
        performance_metrics = {
            "accuracy": 0.85,
            "mae": 0.15,
            "f1_score": 0.82
        }
        
        training_metadata = {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "dataset_size": 10000
        }
        
        model_version = self.version_manager.register_model(
            model=model,
            version=version,
            performance_metrics=performance_metrics,
            training_metadata=training_metadata,
            description=f"Test model version {version}"
        )
        
        # Verify registration
        assert model_version.version == version
        assert model_version.performance_metrics == performance_metrics
        assert model_version.training_metadata == training_metadata
        
        # Load model from registry (simulating different environment)
        loaded_model, loaded_version = self.version_manager.get_model(version)
        loaded_model.eval()
        
        # Verify version metadata consistency
        assert loaded_version.version == version
        assert loaded_version.performance_metrics == performance_metrics
        assert loaded_version.training_metadata == training_metadata
        
        # Test inference consistency across environments
        with torch.no_grad():
            loaded_outputs = loaded_model(test_input)
        
        # Verify outputs are identical (within numerical precision)
        tolerance = 1e-4
        
        # Check height predictions
        height_diff = torch.abs(original_outputs['height'] - loaded_outputs['height'])
        assert torch.all(height_diff < tolerance), \
            f"Height predictions differ across environments: max diff = {torch.max(height_diff).item()}"
        
        # Check wave type predictions
        wave_type_diff = torch.abs(original_outputs['wave_type'] - loaded_outputs['wave_type'])
        assert torch.all(wave_type_diff < tolerance), \
            f"Wave type predictions differ across environments: max diff = {torch.max(wave_type_diff).item()}"
        
        # Check direction predictions
        direction_diff = torch.abs(original_outputs['direction'] - loaded_outputs['direction'])
        assert torch.all(direction_diff < tolerance), \
            f"Direction predictions differ across environments: max diff = {torch.max(direction_diff).item()}"
        
        # Test model loading with different device specifications
        cpu_model, _ = self.version_manager.get_model(version)
        cpu_model = cpu_model.to('cpu')
        cpu_model.eval()
        
        # Test inference on CPU
        cpu_input = test_input.to('cpu')
        with torch.no_grad():
            cpu_outputs = cpu_model(cpu_input)
        
        # Verify CPU outputs match original (within tolerance)
        original_cpu = {k: v.to('cpu') for k, v in original_outputs.items()}
        
        for key in ['height', 'wave_type', 'direction']:
            diff = torch.abs(original_cpu[key] - cpu_outputs[key])
            assert torch.all(diff < tolerance), \
                f"CPU inference differs from original: {key} max diff = {torch.max(diff).item()}"
        
        # Verify model can be retrieved by version
        retrieved_model, retrieved_version = self.version_manager.get_model(version)
        assert retrieved_version.version == version
        
        # Test version listing
        versions = self.version_manager.list_versions()
        assert len(versions) >= 1
        assert any(v.version == version for v in versions)
        
        # Test registry statistics
        stats = self.version_manager.get_registry_stats()
        assert stats["total_versions"] >= 1
        assert stats["latest_version"] == version
    
    @given(
        num_versions=st.integers(min_value=2, max_value=5),
        use_lineage=st.booleans()
    )
    @settings(max_examples=5, deadline=None)
    def test_model_lineage_tracking(self, num_versions, use_lineage):
        """
        Test model lineage tracking and version relationships.
        
        This ensures version history and relationships are properly maintained.
        """
        versions = []
        models = []
        
        # Create a chain of model versions
        for i in range(num_versions):
            version = f"1.{i}.0"
            
            # Create model with slight variations
            config = ModelConfig(hidden_dim=512 + i * 32)
            model = WaveAnalysisModel(config)
            model.eval()
            models.append(model)
            
            # Determine parent version
            parent_version = None
            if use_lineage and i > 0:
                parent_version = versions[-1]
            
            # Register model
            performance_metrics = {
                "accuracy": 0.80 + i * 0.02,
                "mae": 0.20 - i * 0.01
            }
            
            training_metadata = {
                "epochs": 50 + i * 10,
                "iteration": i
            }
            
            model_version = self.version_manager.register_model(
                model=model,
                version=version,
                performance_metrics=performance_metrics,
                training_metadata=training_metadata,
                parent_version=parent_version,
                description=f"Model iteration {i}"
            )
            
            versions.append(version)
        
        # Test lineage tracking
        if use_lineage:
            # Get lineage for the last version
            lineage = self.version_manager.get_lineage(versions[-1])
            
            # Verify lineage is correct
            assert len(lineage) == num_versions
            assert lineage == versions
            
            # Test lineage for middle version
            if num_versions > 2:
                middle_version = versions[num_versions // 2]
                middle_lineage = self.version_manager.get_lineage(middle_version)
                expected_length = (num_versions // 2) + 1
                assert len(middle_lineage) == expected_length
        
        # Test version comparison
        if num_versions >= 2:
            comparison = self.version_manager.compare_models(versions[0], versions[-1])
            
            assert isinstance(comparison, ModelComparison)
            assert comparison.version_a == versions[0]
            assert comparison.version_b == versions[-1]
            assert isinstance(comparison.performance_diff, dict)
            assert isinstance(comparison.config_diff, dict)
            assert isinstance(comparison.recommendation, str)
            assert 0.0 <= comparison.confidence <= 1.0
            
            # Verify performance differences are calculated correctly
            expected_acc_diff = (0.80 + (num_versions - 1) * 0.02) - 0.80
            actual_acc_diff = comparison.performance_diff.get("accuracy", 0.0)
            assert abs(actual_acc_diff - expected_acc_diff) < 0.001
        
        # Test version listing and filtering
        all_versions = self.version_manager.list_versions()
        assert len(all_versions) == num_versions
        
        # Verify versions are sorted by semantic version
        version_strings = [v.version for v in all_versions]
        sorted_versions = sorted(version_strings, key=lambda v: semantic_version.Version(v))
        assert version_strings == sorted_versions
        
        # Test latest version tracking
        latest_version = self.version_manager.registry["latest"]
        assert latest_version == versions[-1]
    
    @given(
        traffic_split=st.floats(min_value=0.1, max_value=0.9),
        num_requests=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=5, deadline=None)
    def test_ab_testing_functionality(self, traffic_split, num_requests):
        """
        Test A/B testing functionality for model versions.
        
        This ensures A/B testing works correctly for comparing model versions.
        """
        # Create two model versions
        config_a = ModelConfig(hidden_dim=512)
        model_a = WaveAnalysisModel(config_a)
        model_a.eval()
        
        config_b = ModelConfig(hidden_dim=768)
        model_b = WaveAnalysisModel(config_b)
        model_b.eval()
        
        # Register both versions
        version_a = "1.0.0"
        version_b = "1.1.0"
        
        self.version_manager.register_model(
            model=model_a,
            version=version_a,
            performance_metrics={"accuracy": 0.80},
            training_metadata={"config": "A"}
        )
        
        self.version_manager.register_model(
            model=model_b,
            version=version_b,
            performance_metrics={"accuracy": 0.82},
            training_metadata={"config": "B"},
            parent_version=version_a
        )
        
        # Create A/B test
        ab_manager = ABTestManager(self.version_manager)
        
        test_config = ab_manager.create_ab_test(
            test_name="accuracy_test",
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            success_metrics=["accuracy", "latency"]
        )
        
        # Verify test configuration
        assert test_config["test_name"] == "accuracy_test"
        assert test_config["version_a"] == version_a
        assert test_config["version_b"] == version_b
        assert test_config["traffic_split"] == traffic_split
        assert test_config["status"] == "active"
        
        # Simulate requests and routing
        version_a_count = 0
        version_b_count = 0
        
        for i in range(num_requests):
            request_id = f"request_{i}"
            routed_version = ab_manager.route_request("accuracy_test", request_id)
            
            assert routed_version in [version_a, version_b]
            
            if routed_version == version_a:
                version_a_count += 1
            else:
                version_b_count += 1
            
            # Record mock results
            mock_metrics = {
                "accuracy": 0.80 + (0.02 if routed_version == version_b else 0.0),
                "latency": 50.0 + (5.0 if routed_version == version_b else 0.0)
            }
            
            ab_manager.record_result("accuracy_test", routed_version, mock_metrics)
        
        # Verify traffic split is approximately correct (within reasonable tolerance)
        expected_a_count = num_requests * traffic_split
        actual_split = version_a_count / num_requests
        
        # Allow for some variance due to hash-based routing
        assert abs(actual_split - traffic_split) < 0.3, \
            f"Traffic split deviation too large: expected {traffic_split}, got {actual_split}"
        
        # Get test results
        test_results = ab_manager.get_test_results("accuracy_test")
        
        assert test_results["results"]["version_a"]["requests"] == version_a_count
        assert test_results["results"]["version_b"]["requests"] == version_b_count
        
        # Verify metrics are recorded
        if version_a_count > 0:
            assert "accuracy" in test_results["results"]["version_a"]["metrics"]
            assert "latency" in test_results["results"]["version_a"]["metrics"]
        
        if version_b_count > 0:
            assert "accuracy" in test_results["results"]["version_b"]["metrics"]
            assert "latency" in test_results["results"]["version_b"]["metrics"]
        
        # Stop test
        final_results = ab_manager.stop_test("accuracy_test", winner=version_b)
        assert final_results["status"] == "completed"
        assert final_results["winner"] == version_b
    
    @given(
        num_benchmarks=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=3, deadline=None)
    def test_performance_regression_detection(self, num_benchmarks):
        """
        Test performance regression detection across model versions.
        
        This ensures performance regressions are properly detected and reported.
        """
        versions = []
        
        # Create multiple model versions with varying performance
        for i in range(num_benchmarks):
            version = f"2.{i}.0"
            
            config = ModelConfig()
            model = WaveAnalysisModel(config)
            model.eval()
            
            # Register model
            performance_metrics = {
                "accuracy": 0.85 - i * 0.02,  # Decreasing accuracy (regression)
                "mae": 0.10 + i * 0.01        # Increasing error (regression)
            }
            
            self.version_manager.register_model(
                model=model,
                version=version,
                performance_metrics=performance_metrics,
                training_metadata={"iteration": i}
            )
            
            # Run benchmark
            benchmark_suite = self.benchmark.run_full_benchmark(
                model=model,
                model_version=version,
                accuracy_metrics=performance_metrics
            )
            
            versions.append(version)
        
        # Create regression test using first version as baseline
        baseline_version = versions[0]
        
        regression_test = self.benchmark.create_regression_test(
            test_name="performance_regression",
            baseline_version=baseline_version,
            thresholds={
                "latency_increase_pct": 20.0,
                "memory_increase_pct": 15.0,
                "throughput_decrease_pct": 10.0
            },
            description="Test for performance regressions"
        )
        
        assert regression_test["test_name"] == "performance_regression"
        assert regression_test["baseline_version"] == baseline_version
        
        # Run regression tests on subsequent versions
        for version in versions[1:]:
            test_result = self.benchmark.run_regression_test("performance_regression", version)
            
            assert test_result["candidate_version"] == version
            assert isinstance(test_result["passed"], bool)
            assert isinstance(test_result["failures"], list)
            assert isinstance(test_result["metrics"], dict)
        
        # Test benchmark comparison
        if len(versions) >= 2:
            comparison = self.benchmark.compare_benchmarks(versions[0], versions[-1])
            
            assert comparison["version_a"] == versions[0]
            assert comparison["version_b"] == versions[-1]
            assert isinstance(comparison["latency_diff_ms"], float)
            assert isinstance(comparison["memory_diff_mb"], float)
            assert isinstance(comparison["throughput_diff_sps"], float)
        
        # Test performance trends
        trends = self.benchmark.get_performance_trends()
        
        assert len(trends["versions"]) == num_benchmarks
        assert len(trends["latency_ms"]) == num_benchmarks
        assert len(trends["memory_mb"]) == num_benchmarks
        assert len(trends["throughput_sps"]) == num_benchmarks
        
        # Verify all versions are included in trends
        for version in versions:
            assert version in trends["versions"]
    
    def test_model_rollback_functionality(self):
        """
        Test model rollback functionality.
        
        This ensures models can be safely rolled back to previous versions.
        """
        # Create multiple versions
        versions = ["1.0.0", "1.1.0", "1.2.0"]
        
        for i, version in enumerate(versions):
            config = ModelConfig()
            model = WaveAnalysisModel(config)
            
            self.version_manager.register_model(
                model=model,
                version=version,
                performance_metrics={"accuracy": 0.80 + i * 0.01},
                training_metadata={"iteration": i}
            )
        
        # Verify latest version
        assert self.version_manager.registry["latest"] == "1.2.0"
        
        # Rollback to previous version
        rollback_version = self.version_manager.rollback_to_version("1.1.0")
        
        assert rollback_version.version == "1.1.0"
        assert self.version_manager.registry["latest"] == "1.1.0"
        
        # Verify rollback model can be loaded
        model, version_info = self.version_manager.get_model()  # Should get latest (rolled back)
        assert version_info.version == "1.1.0"
        
        # Test rollback to non-existent version
        with pytest.raises(ValueError):
            self.version_manager.rollback_to_version("2.0.0")
    
    def test_model_registry_integrity(self):
        """
        Test model registry integrity and error handling.
        
        This ensures the registry maintains integrity under various conditions.
        """
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        
        # Test duplicate version registration
        version = "1.0.0"
        
        self.version_manager.register_model(
            model=model,
            version=version,
            performance_metrics={"accuracy": 0.85},
            training_metadata={"test": True}
        )
        
        # Should raise error for duplicate version
        with pytest.raises(ValueError, match="already exists"):
            self.version_manager.register_model(
                model=model,
                version=version,
                performance_metrics={"accuracy": 0.86},
                training_metadata={"test": True}
            )
        
        # Test invalid semantic version
        with pytest.raises(ValueError, match="Invalid semantic version"):
            self.version_manager.register_model(
                model=model,
                version="invalid.version",
                performance_metrics={"accuracy": 0.85},
                training_metadata={"test": True}
            )
        
        # Test non-existent parent version
        with pytest.raises(ValueError, match="Parent version .* does not exist"):
            self.version_manager.register_model(
                model=model,
                version="1.1.0",
                performance_metrics={"accuracy": 0.85},
                training_metadata={"test": True},
                parent_version="0.9.0"
            )
        
        # Test loading non-existent version
        with pytest.raises(ValueError, match="Version .* not found"):
            self.version_manager.get_model("2.0.0")
        
        # Test registry statistics
        stats = self.version_manager.get_registry_stats()
        assert stats["total_versions"] == 1
        assert stats["latest_version"] == version
        assert stats["oldest_version"] == version
        assert stats["newest_version"] == version
        assert isinstance(stats["total_size_bytes"], int)
        assert stats["total_size_bytes"] > 0
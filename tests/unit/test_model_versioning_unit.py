"""Unit tests for model versioning system."""

import torch
import tempfile
import shutil
from pathlib import Path
import pytest

from swellsight.models.wave_analysis_model import WaveAnalysisModel
from swellsight.config import ModelConfig
from swellsight.utils.model_versioning import ModelVersionManager


class TestModelVersioningUnit:
    """Unit tests for model versioning system."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_manager = ModelVersionManager(self.temp_dir / "registry")
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_model_version_consistency_basic(self):
        """
        Test basic model version consistency - that loading and inference 
        produce identical results.
        """
        # Create semantic version
        version = "1.0.0"
        
        # Create and train model
        config = ModelConfig()
        model = WaveAnalysisModel(config)
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, 3, 768, 768)
        
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
        
        print("✅ Model version consistency test passed")
    
    def test_model_lineage_tracking(self):
        """Test model lineage tracking and version relationships."""
        # Create a chain of model versions
        versions = ["1.0.0", "1.1.0", "1.2.0"]
        
        for i, version in enumerate(versions):
            # Create model with slight variations
            config = ModelConfig(hidden_dim=512 + i * 32)
            model = WaveAnalysisModel(config)
            model.eval()
            
            # Determine parent version
            parent_version = None
            if i > 0:
                parent_version = versions[i-1]
            
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
        
        # Test lineage tracking
        lineage = self.version_manager.get_lineage(versions[-1])
        assert len(lineage) == len(versions)
        assert lineage == versions
        
        # Test version comparison
        comparison = self.version_manager.compare_models(versions[0], versions[-1])
        assert comparison.version_a == versions[0]
        assert comparison.version_b == versions[-1]
        
        print("✅ Model lineage tracking test passed")
    
    def test_model_rollback(self):
        """Test model rollback functionality."""
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
        
        print("✅ Model rollback test passed")
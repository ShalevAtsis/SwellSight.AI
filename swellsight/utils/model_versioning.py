"""Model versioning and registry system for SwellSight Wave Analysis Model."""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import semantic_version
import torch

from .model_persistence import ModelPersistence, get_model_info
from ..models.wave_analysis_model import WaveAnalysisModel
from ..config import ModelConfig


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_path: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    created_at: str = ""
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ModelComparison:
    """Model comparison result."""
    version_a: str
    version_b: str
    performance_diff: Dict[str, float]
    config_diff: Dict[str, Any]
    recommendation: str
    confidence: float


class ModelVersionManager:
    """
    Comprehensive model versioning system with semantic versioning,
    lineage tracking, and performance comparison utilities.
    """
    
    def __init__(self, registry_path: Union[str, Path] = "models/registry"):
        """
        Initialize model version manager.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.persistence = ModelPersistence()
        self._load_registry()
    
    def _load_registry(self):
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "versions": {},
                "latest": None,
                "created_at": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save registry metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _validate_version(self, version: str) -> semantic_version.Version:
        """
        Validate semantic version string.
        
        Args:
            version: Version string to validate
            
        Returns:
            Parsed semantic version
            
        Raises:
            ValueError: If version is invalid
        """
        try:
            return semantic_version.Version(version)
        except ValueError as e:
            raise ValueError(f"Invalid semantic version '{version}': {e}")
    
    def _generate_model_hash(self, model: WaveAnalysisModel) -> str:
        """Generate unique hash for model architecture and weights."""
        # Get model state dict
        state_dict = model.state_dict()
        
        # Create hash from model structure and weights
        model_str = ""
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            model_str += f"{key}:{tensor.shape}:{tensor.sum().item():.6f};"
        
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def register_model(
        self,
        model: WaveAnalysisModel,
        version: str,
        performance_metrics: Dict[str, float],
        training_metadata: Dict[str, Any],
        description: str = "",
        tags: List[str] = None,
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a new model version in the registry.
        
        Args:
            model: Trained model to register
            version: Semantic version string (e.g., "1.0.0")
            performance_metrics: Model performance metrics
            training_metadata: Training configuration and metadata
            description: Human-readable description
            tags: List of tags for categorization
            parent_version: Parent version for lineage tracking
            
        Returns:
            ModelVersion object with registration details
            
        Raises:
            ValueError: If version is invalid or already exists
        """
        # Validate version
        sem_version = self._validate_version(version)
        
        # Check if version already exists
        if version in self.registry["versions"]:
            raise ValueError(f"Version {version} already exists")
        
        # Validate parent version if specified
        if parent_version and parent_version not in self.registry["versions"]:
            raise ValueError(f"Parent version {parent_version} does not exist")
        
        # Generate model hash for uniqueness check
        model_hash = self._generate_model_hash(model)
        
        # Check for duplicate models
        for existing_version, metadata in self.registry["versions"].items():
            if metadata.get("model_hash") == model_hash:
                raise ValueError(f"Model with identical weights already exists as version {existing_version}")
        
        # Create model file path
        model_filename = f"model_v{version}.pth"
        model_path = self.models_dir / model_filename
        
        # Save model
        save_info = self.persistence.save_model(
            model,
            model_path,
            metadata={
                "version": version,
                "performance_metrics": performance_metrics,
                "training_metadata": training_metadata,
                "model_hash": model_hash
            }
        )
        
        # Create version metadata
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            config=model.config.__dict__ if hasattr(model, 'config') else {},
            performance_metrics=performance_metrics,
            training_metadata=training_metadata,
            parent_version=parent_version,
            description=description,
            tags=tags or []
        )
        
        # Add to registry
        version_data = asdict(model_version)
        version_data["model_hash"] = model_hash
        version_data["file_size"] = save_info["file_size"]
        version_data["integrity_hash"] = save_info["integrity_hash"]
        
        self.registry["versions"][version] = version_data
        
        # Update latest version if this is newer
        if not self.registry["latest"] or sem_version > semantic_version.Version(self.registry["latest"]):
            self.registry["latest"] = version
        
        self._save_registry()
        
        return model_version
    
    def get_model(self, version: str = None) -> Tuple[WaveAnalysisModel, ModelVersion]:
        """
        Load a model by version.
        
        Args:
            version: Version to load (defaults to latest)
            
        Returns:
            Tuple of (model, version_metadata)
            
        Raises:
            ValueError: If version doesn't exist
        """
        if version is None:
            version = self.registry["latest"]
            if version is None:
                raise ValueError("No models registered")
        
        if version not in self.registry["versions"]:
            raise ValueError(f"Version {version} not found")
        
        version_data = self.registry["versions"][version]
        model_path = Path(version_data["model_path"])
        
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model
        model = self.persistence.load_model(model_path)
        
        # Create version metadata
        model_version = ModelVersion(**{k: v for k, v in version_data.items() 
                                      if k not in ["model_hash", "file_size", "integrity_hash"]})
        
        return model, model_version
    
    def list_versions(self, tags: List[str] = None) -> List[ModelVersion]:
        """
        List all registered model versions.
        
        Args:
            tags: Filter by tags (optional)
            
        Returns:
            List of model versions sorted by semantic version
        """
        versions = []
        
        for version_str, version_data in self.registry["versions"].items():
            # Filter by tags if specified
            if tags:
                version_tags = version_data.get("tags", [])
                if not any(tag in version_tags for tag in tags):
                    continue
            
            model_version = ModelVersion(**{k: v for k, v in version_data.items() 
                                          if k not in ["model_hash", "file_size", "integrity_hash"]})
            versions.append(model_version)
        
        # Sort by semantic version
        versions.sort(key=lambda v: semantic_version.Version(v.version))
        
        return versions
    
    def compare_models(self, version_a: str, version_b: str) -> ModelComparison:
        """
        Compare two model versions.
        
        Args:
            version_a: First version to compare
            version_b: Second version to compare
            
        Returns:
            ModelComparison with detailed comparison results
        """
        if version_a not in self.registry["versions"]:
            raise ValueError(f"Version {version_a} not found")
        if version_b not in self.registry["versions"]:
            raise ValueError(f"Version {version_b} not found")
        
        data_a = self.registry["versions"][version_a]
        data_b = self.registry["versions"][version_b]
        
        # Compare performance metrics
        metrics_a = data_a["performance_metrics"]
        metrics_b = data_b["performance_metrics"]
        
        performance_diff = {}
        for metric in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(metric, 0.0)
            val_b = metrics_b.get(metric, 0.0)
            performance_diff[metric] = val_b - val_a
        
        # Compare configurations
        config_a = data_a["config"]
        config_b = data_b["config"]
        
        config_diff = {}
        for key in set(config_a.keys()) | set(config_b.keys()):
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                config_diff[key] = {"from": val_a, "to": val_b}
        
        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(performance_diff, config_diff)
        
        return ModelComparison(
            version_a=version_a,
            version_b=version_b,
            performance_diff=performance_diff,
            config_diff=config_diff,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _generate_recommendation(self, performance_diff: Dict[str, float], config_diff: Dict[str, Any]) -> Tuple[str, float]:
        """Generate recommendation based on comparison results."""
        # Simple heuristic for recommendation
        positive_changes = 0
        negative_changes = 0
        total_changes = 0
        
        # Analyze performance changes
        for metric, diff in performance_diff.items():
            if abs(diff) > 0.001:  # Significant change threshold
                total_changes += 1
                if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:
                    # Higher is better for these metrics
                    if diff > 0:
                        positive_changes += 1
                    else:
                        negative_changes += 1
                elif metric.lower() in ['loss', 'mae', 'rmse', 'error']:
                    # Lower is better for these metrics
                    if diff < 0:
                        positive_changes += 1
                    else:
                        negative_changes += 1
        
        if total_changes == 0:
            return "No significant performance difference", 0.5
        
        improvement_ratio = positive_changes / total_changes
        
        if improvement_ratio >= 0.7:
            return "Newer version recommended - significant improvements", improvement_ratio
        elif improvement_ratio >= 0.5:
            return "Newer version slightly better - consider upgrade", improvement_ratio
        elif improvement_ratio >= 0.3:
            return "Mixed results - evaluate based on specific needs", improvement_ratio
        else:
            return "Older version may be better - regression detected", 1.0 - improvement_ratio
    
    def get_lineage(self, version: str) -> List[str]:
        """
        Get the lineage (ancestry) of a model version.
        
        Args:
            version: Version to trace lineage for
            
        Returns:
            List of versions from root to specified version
        """
        if version not in self.registry["versions"]:
            raise ValueError(f"Version {version} not found")
        
        lineage = []
        current = version
        
        while current:
            lineage.append(current)
            version_data = self.registry["versions"][current]
            current = version_data.get("parent_version")
        
        return list(reversed(lineage))
    
    def rollback_to_version(self, version: str) -> ModelVersion:
        """
        Rollback to a specific version by setting it as latest.
        
        Args:
            version: Version to rollback to
            
        Returns:
            ModelVersion metadata for the rollback version
        """
        if version not in self.registry["versions"]:
            raise ValueError(f"Version {version} not found")
        
        # Update latest version
        self.registry["latest"] = version
        self._save_registry()
        
        version_data = self.registry["versions"][version]
        return ModelVersion(**{k: v for k, v in version_data.items() 
                             if k not in ["model_hash", "file_size", "integrity_hash"]})
    
    def delete_version(self, version: str, force: bool = False):
        """
        Delete a model version from the registry.
        
        Args:
            version: Version to delete
            force: Force deletion even if it has dependents
            
        Raises:
            ValueError: If version doesn't exist or has dependents
        """
        if version not in self.registry["versions"]:
            raise ValueError(f"Version {version} not found")
        
        # Check for dependent versions
        if not force:
            dependents = [v for v, data in self.registry["versions"].items() 
                         if data.get("parent_version") == version]
            if dependents:
                raise ValueError(f"Cannot delete version {version}: has dependent versions {dependents}")
        
        # Get version data
        version_data = self.registry["versions"][version]
        model_path = Path(version_data["model_path"])
        
        # Delete model file
        if model_path.exists():
            model_path.unlink()
        
        # Delete metadata file if exists
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from registry
        del self.registry["versions"][version]
        
        # Update latest if this was the latest
        if self.registry["latest"] == version:
            remaining_versions = list(self.registry["versions"].keys())
            if remaining_versions:
                # Set latest to highest semantic version
                latest_version = max(remaining_versions, key=lambda v: semantic_version.Version(v))
                self.registry["latest"] = latest_version
            else:
                self.registry["latest"] = None
        
        self._save_registry()
    
    def export_version(self, version: str, export_path: Union[str, Path]) -> Path:
        """
        Export a model version to a standalone file.
        
        Args:
            version: Version to export
            export_path: Path to export to
            
        Returns:
            Path to exported file
        """
        if version not in self.registry["versions"]:
            raise ValueError(f"Version {version} not found")
        
        version_data = self.registry["versions"][version]
        model_path = Path(version_data["model_path"])
        export_path = Path(export_path)
        
        # Copy model file
        shutil.copy2(model_path, export_path)
        
        # Copy metadata file if exists
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            export_metadata_path = export_path.with_suffix('.json')
            shutil.copy2(metadata_path, export_metadata_path)
        
        return export_path
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry."""
        versions = list(self.registry["versions"].keys())
        
        if not versions:
            return {
                "total_versions": 0,
                "latest_version": None,
                "total_size_bytes": 0,
                "created_at": self.registry["created_at"]
            }
        
        # Calculate total size
        total_size = 0
        for version_data in self.registry["versions"].values():
            total_size += version_data.get("file_size", 0)
        
        # Get version range
        sem_versions = [semantic_version.Version(v) for v in versions]
        oldest_version = str(min(sem_versions))
        newest_version = str(max(sem_versions))
        
        return {
            "total_versions": len(versions),
            "latest_version": self.registry["latest"],
            "oldest_version": oldest_version,
            "newest_version": newest_version,
            "total_size_bytes": total_size,
            "created_at": self.registry["created_at"]
        }


class ABTestManager:
    """A/B testing manager for model versions."""
    
    def __init__(self, version_manager: ModelVersionManager):
        """
        Initialize A/B test manager.
        
        Args:
            version_manager: Model version manager instance
        """
        self.version_manager = version_manager
        self.active_tests = {}
    
    def create_ab_test(
        self,
        test_name: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        success_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new A/B test between two model versions.
        
        Args:
            test_name: Name of the A/B test
            version_a: First version to test
            version_b: Second version to test
            traffic_split: Fraction of traffic for version A (0.0-1.0)
            success_metrics: List of metrics to track for success
            
        Returns:
            A/B test configuration
        """
        if test_name in self.active_tests:
            raise ValueError(f"A/B test '{test_name}' already exists")
        
        # Validate versions exist
        if version_a not in self.version_manager.registry["versions"]:
            raise ValueError(f"Version {version_a} not found")
        if version_b not in self.version_manager.registry["versions"]:
            raise ValueError(f"Version {version_b} not found")
        
        test_config = {
            "test_name": test_name,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "success_metrics": success_metrics or ["accuracy"],
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "results": {
                "version_a": {"requests": 0, "metrics": {}},
                "version_b": {"requests": 0, "metrics": {}}
            }
        }
        
        self.active_tests[test_name] = test_config
        return test_config
    
    def route_request(self, test_name: str, request_id: str = None) -> str:
        """
        Route a request to appropriate model version based on A/B test configuration.
        
        Args:
            test_name: Name of the A/B test
            request_id: Optional request ID for consistent routing
            
        Returns:
            Version to use for this request
        """
        if test_name not in self.active_tests:
            raise ValueError(f"A/B test '{test_name}' not found")
        
        test_config = self.active_tests[test_name]
        
        if test_config["status"] != "active":
            raise ValueError(f"A/B test '{test_name}' is not active")
        
        # Simple hash-based routing for consistency
        if request_id:
            hash_value = hash(request_id) % 100
            use_version_a = hash_value < (test_config["traffic_split"] * 100)
        else:
            # Random routing
            import random
            use_version_a = random.random() < test_config["traffic_split"]
        
        return test_config["version_a"] if use_version_a else test_config["version_b"]
    
    def record_result(self, test_name: str, version: str, metrics: Dict[str, float]):
        """
        Record results for an A/B test.
        
        Args:
            test_name: Name of the A/B test
            version: Version that was used
            metrics: Performance metrics to record
        """
        if test_name not in self.active_tests:
            raise ValueError(f"A/B test '{test_name}' not found")
        
        test_config = self.active_tests[test_name]
        
        # Determine which version group
        if version == test_config["version_a"]:
            results = test_config["results"]["version_a"]
        elif version == test_config["version_b"]:
            results = test_config["results"]["version_b"]
        else:
            raise ValueError(f"Version {version} not part of A/B test {test_name}")
        
        # Update request count
        results["requests"] += 1
        
        # Update metrics (running average)
        for metric, value in metrics.items():
            if metric not in results["metrics"]:
                results["metrics"][metric] = value
            else:
                # Running average
                n = results["requests"]
                current_avg = results["metrics"][metric]
                results["metrics"][metric] = ((n - 1) * current_avg + value) / n
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get current results for an A/B test."""
        if test_name not in self.active_tests:
            raise ValueError(f"A/B test '{test_name}' not found")
        
        return self.active_tests[test_name]
    
    def stop_test(self, test_name: str, winner: str = None) -> Dict[str, Any]:
        """
        Stop an A/B test and optionally declare a winner.
        
        Args:
            test_name: Name of the A/B test
            winner: Optional winner version
            
        Returns:
            Final test results
        """
        if test_name not in self.active_tests:
            raise ValueError(f"A/B test '{test_name}' not found")
        
        test_config = self.active_tests[test_name]
        test_config["status"] = "completed"
        test_config["completed_at"] = datetime.now().isoformat()
        
        if winner:
            test_config["winner"] = winner
        
        return test_config
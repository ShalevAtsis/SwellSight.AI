"""Performance benchmarking and regression testing for SwellSight models."""

import time
import torch
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from ..models.wave_analysis_model import WaveAnalysisModel
from ..config import ModelConfig


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result."""
    metric_name: str
    value: float
    unit: str
    timestamp: str
    device: str
    model_version: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    model_version: str
    device: str
    timestamp: str
    inference_latency_ms: float
    memory_usage_mb: float
    throughput_samples_per_sec: float
    model_size_mb: float
    accuracy_metrics: Dict[str, float]
    system_info: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking system for model evaluation.
    
    Measures inference latency, memory usage, throughput, and tracks
    performance regression across model versions.
    """
    
    def __init__(self, results_dir: Union[str, Path] = "benchmarks"):
        """
        Initialize performance benchmark system.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.results_dir / "benchmark_results.json"
        self.regression_file = self.results_dir / "regression_tests.json"
        
        self._load_results()
    
    def _load_results(self):
        """Load existing benchmark results."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {"benchmarks": [], "created_at": datetime.now().isoformat()}
        
        if self.regression_file.exists():
            with open(self.regression_file, 'r') as f:
                self.regression_tests = json.load(f)
        else:
            self.regression_tests = {"tests": [], "created_at": datetime.now().isoformat()}
    
    def _save_results(self):
        """Save benchmark results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        with open(self.regression_file, 'w') as f:
            json.dump(self.regression_tests, f, indent=2)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{torch.__version__}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def measure_inference_latency(
        self,
        model: WaveAnalysisModel,
        input_shape: Tuple[int, int, int, int] = (1, 3, 768, 768),
        num_warmup: int = 10,
        num_iterations: int = 100,
        device: str = "auto"
    ) -> BenchmarkResult:
        """
        Measure model inference latency.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            device: Device to run benchmark on
            
        Returns:
            BenchmarkResult with latency measurement
        """
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        model.eval()
        
        # Create test input
        test_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(test_input)
        
        # Synchronize for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(test_input)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        
        return BenchmarkResult(
            metric_name="inference_latency",
            value=mean_latency,
            unit="ms",
            device=device,
            timestamp=datetime.now().isoformat()
        )
    
    def measure_memory_usage(
        self,
        model: WaveAnalysisModel,
        input_shape: Tuple[int, int, int, int] = (1, 3, 768, 768),
        device: str = "auto"
    ) -> BenchmarkResult:
        """
        Measure model memory usage during inference.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            device: Device to run benchmark on
            
        Returns:
            BenchmarkResult with memory usage measurement
        """
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        model.eval()
        
        if device == "cuda":
            # GPU memory measurement
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            test_input = torch.randn(input_shape, device=device)
            
            with torch.no_grad():
                _ = model(test_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
            torch.cuda.empty_cache()
            
        else:
            # CPU memory measurement
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)
            
            test_input = torch.randn(input_shape, device=device)
            
            with torch.no_grad():
                _ = model(test_input)
            
            peak_memory = process.memory_info().rss / (1024**2) - initial_memory
        
        return BenchmarkResult(
            metric_name="memory_usage",
            value=peak_memory,
            unit="MB",
            device=device,
            timestamp=datetime.now().isoformat()
        )
    
    def measure_throughput(
        self,
        model: WaveAnalysisModel,
        input_shape: Tuple[int, int, int, int] = (1, 3, 768, 768),
        duration_seconds: float = 10.0,
        device: str = "auto"
    ) -> BenchmarkResult:
        """
        Measure model throughput (samples per second).
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            duration_seconds: How long to run the benchmark
            device: Device to run benchmark on
            
        Returns:
            BenchmarkResult with throughput measurement
        """
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        model.eval()
        
        # Create test input
        test_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Measure throughput
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        sample_count = 0
        
        with torch.no_grad():
            while time.perf_counter() < end_time:
                _ = model(test_input)
                sample_count += input_shape[0]  # Batch size
        
        actual_duration = time.perf_counter() - start_time
        throughput = sample_count / actual_duration
        
        return BenchmarkResult(
            metric_name="throughput",
            value=throughput,
            unit="samples/sec",
            device=device,
            timestamp=datetime.now().isoformat()
        )
    
    def measure_model_size(self, model: WaveAnalysisModel) -> BenchmarkResult:
        """
        Measure model size in memory.
        
        Args:
            model: Model to measure
            
        Returns:
            BenchmarkResult with model size measurement
        """
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024**2)
        
        return BenchmarkResult(
            metric_name="model_size",
            value=total_size_mb,
            unit="MB",
            device="cpu",
            timestamp=datetime.now().isoformat()
        )
    
    def run_full_benchmark(
        self,
        model: WaveAnalysisModel,
        model_version: str = "unknown",
        input_shape: Tuple[int, int, int, int] = (1, 3, 768, 768),
        device: str = "auto",
        accuracy_metrics: Dict[str, float] = None
    ) -> BenchmarkSuite:
        """
        Run complete benchmark suite on a model.
        
        Args:
            model: Model to benchmark
            model_version: Version identifier for the model
            input_shape: Input tensor shape
            device: Device to run benchmark on
            accuracy_metrics: Optional accuracy metrics to include
            
        Returns:
            BenchmarkSuite with all measurements
        """
        print(f"Running benchmark suite for model version {model_version}...")
        
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Run individual benchmarks
        latency_result = self.measure_inference_latency(model, input_shape, device=device)
        memory_result = self.measure_memory_usage(model, input_shape, device=device)
        throughput_result = self.measure_throughput(model, input_shape, device=device)
        size_result = self.measure_model_size(model)
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            model_version=model_version,
            device=device,
            timestamp=datetime.now().isoformat(),
            inference_latency_ms=latency_result.value,
            memory_usage_mb=memory_result.value,
            throughput_samples_per_sec=throughput_result.value,
            model_size_mb=size_result.value,
            accuracy_metrics=accuracy_metrics or {},
            system_info=self._get_system_info()
        )
        
        # Store results
        self.results["benchmarks"].append(asdict(suite))
        self._save_results()
        
        print(f"Benchmark completed:")
        print(f"  Inference Latency: {latency_result.value:.2f} ms")
        print(f"  Memory Usage: {memory_result.value:.2f} MB")
        print(f"  Throughput: {throughput_result.value:.2f} samples/sec")
        print(f"  Model Size: {size_result.value:.2f} MB")
        
        return suite
    
    def compare_benchmarks(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """
        Compare benchmark results between two model versions.
        
        Args:
            version_a: First version to compare
            version_b: Second version to compare
            
        Returns:
            Comparison results with performance differences
        """
        # Find benchmark results for each version
        results_a = None
        results_b = None
        
        for benchmark in self.results["benchmarks"]:
            if benchmark["model_version"] == version_a:
                results_a = benchmark
            elif benchmark["model_version"] == version_b:
                results_b = benchmark
        
        if not results_a:
            raise ValueError(f"No benchmark results found for version {version_a}")
        if not results_b:
            raise ValueError(f"No benchmark results found for version {version_b}")
        
        # Calculate differences
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "latency_diff_ms": results_b["inference_latency_ms"] - results_a["inference_latency_ms"],
            "memory_diff_mb": results_b["memory_usage_mb"] - results_a["memory_usage_mb"],
            "throughput_diff_sps": results_b["throughput_samples_per_sec"] - results_a["throughput_samples_per_sec"],
            "size_diff_mb": results_b["model_size_mb"] - results_a["model_size_mb"],
            "latency_change_pct": ((results_b["inference_latency_ms"] - results_a["inference_latency_ms"]) / results_a["inference_latency_ms"]) * 100,
            "memory_change_pct": ((results_b["memory_usage_mb"] - results_a["memory_usage_mb"]) / results_a["memory_usage_mb"]) * 100,
            "throughput_change_pct": ((results_b["throughput_samples_per_sec"] - results_a["throughput_samples_per_sec"]) / results_a["throughput_samples_per_sec"]) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison
    
    def create_regression_test(
        self,
        test_name: str,
        baseline_version: str,
        thresholds: Dict[str, float],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a performance regression test.
        
        Args:
            test_name: Name of the regression test
            baseline_version: Baseline version to compare against
            thresholds: Performance thresholds (e.g., {"latency_increase_pct": 10.0})
            description: Test description
            
        Returns:
            Regression test configuration
        """
        # Find baseline benchmark
        baseline_benchmark = None
        for benchmark in self.results["benchmarks"]:
            if benchmark["model_version"] == baseline_version:
                baseline_benchmark = benchmark
                break
        
        if not baseline_benchmark:
            raise ValueError(f"No benchmark results found for baseline version {baseline_version}")
        
        regression_test = {
            "test_name": test_name,
            "baseline_version": baseline_version,
            "baseline_benchmark": baseline_benchmark,
            "thresholds": thresholds,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "test_results": []
        }
        
        self.regression_tests["tests"].append(regression_test)
        self._save_results()
        
        return regression_test
    
    def run_regression_test(self, test_name: str, candidate_version: str) -> Dict[str, Any]:
        """
        Run a regression test against a candidate model version.
        
        Args:
            test_name: Name of the regression test
            candidate_version: Version to test
            
        Returns:
            Test results with pass/fail status
        """
        # Find regression test
        regression_test = None
        for test in self.regression_tests["tests"]:
            if test["test_name"] == test_name:
                regression_test = test
                break
        
        if not regression_test:
            raise ValueError(f"Regression test '{test_name}' not found")
        
        # Find candidate benchmark
        candidate_benchmark = None
        for benchmark in self.results["benchmarks"]:
            if benchmark["model_version"] == candidate_version:
                candidate_benchmark = benchmark
                break
        
        if not candidate_benchmark:
            raise ValueError(f"No benchmark results found for candidate version {candidate_version}")
        
        baseline = regression_test["baseline_benchmark"]
        thresholds = regression_test["thresholds"]
        
        # Run regression checks
        test_result = {
            "candidate_version": candidate_version,
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "failures": [],
            "metrics": {}
        }
        
        # Check latency regression
        if "latency_increase_pct" in thresholds:
            latency_increase = ((candidate_benchmark["inference_latency_ms"] - baseline["inference_latency_ms"]) / baseline["inference_latency_ms"]) * 100
            test_result["metrics"]["latency_increase_pct"] = latency_increase
            
            if latency_increase > thresholds["latency_increase_pct"]:
                test_result["passed"] = False
                test_result["failures"].append(f"Latency increased by {latency_increase:.2f}%, threshold: {thresholds['latency_increase_pct']}%")
        
        # Check memory regression
        if "memory_increase_pct" in thresholds:
            memory_increase = ((candidate_benchmark["memory_usage_mb"] - baseline["memory_usage_mb"]) / baseline["memory_usage_mb"]) * 100
            test_result["metrics"]["memory_increase_pct"] = memory_increase
            
            if memory_increase > thresholds["memory_increase_pct"]:
                test_result["passed"] = False
                test_result["failures"].append(f"Memory usage increased by {memory_increase:.2f}%, threshold: {thresholds['memory_increase_pct']}%")
        
        # Check throughput regression
        if "throughput_decrease_pct" in thresholds:
            throughput_decrease = ((baseline["throughput_samples_per_sec"] - candidate_benchmark["throughput_samples_per_sec"]) / baseline["throughput_samples_per_sec"]) * 100
            test_result["metrics"]["throughput_decrease_pct"] = throughput_decrease
            
            if throughput_decrease > thresholds["throughput_decrease_pct"]:
                test_result["passed"] = False
                test_result["failures"].append(f"Throughput decreased by {throughput_decrease:.2f}%, threshold: {thresholds['throughput_decrease_pct']}%")
        
        # Store test result
        regression_test["test_results"].append(test_result)
        self._save_results()
        
        return test_result
    
    def get_benchmark_history(self, model_version: str = None) -> List[Dict[str, Any]]:
        """
        Get benchmark history for a specific version or all versions.
        
        Args:
            model_version: Optional version filter
            
        Returns:
            List of benchmark results
        """
        if model_version:
            return [b for b in self.results["benchmarks"] if b["model_version"] == model_version]
        else:
            return self.results["benchmarks"]
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """
        Get performance trends across all benchmarked versions.
        
        Returns:
            Dictionary with performance metrics over time
        """
        benchmarks = sorted(self.results["benchmarks"], key=lambda x: x["timestamp"])
        
        trends = {
            "versions": [b["model_version"] for b in benchmarks],
            "timestamps": [b["timestamp"] for b in benchmarks],
            "latency_ms": [b["inference_latency_ms"] for b in benchmarks],
            "memory_mb": [b["memory_usage_mb"] for b in benchmarks],
            "throughput_sps": [b["throughput_samples_per_sec"] for b in benchmarks],
            "model_size_mb": [b["model_size_mb"] for b in benchmarks]
        }
        
        return trends
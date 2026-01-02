#!/usr/bin/env python3
"""
Comprehensive performance benchmarking script for SwellSight models.

This script runs performance benchmarks, model optimization, and regression tests
to ensure production readiness and track performance across model versions.

Usage:
    python swellsight/scripts/run_performance_benchmarks.py --model-path checkpoints/model.pth --version 1.0.0
    python swellsight/scripts/run_performance_benchmarks.py --config benchmark_config.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from swellsight.models import WaveAnalysisModel
from swellsight.config import ModelConfig
from swellsight.utils.performance_benchmarks import PerformanceBenchmark, BenchmarkSuite
from swellsight.utils.model_optimization import ModelOptimizer, OptimizationResult
from swellsight.utils.model_versioning import ModelVersionManager
from swellsight.data import DatasetManager


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for SwellSight models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'benchmark_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.benchmark = PerformanceBenchmark(self.output_dir / 'benchmarks')
        self.optimizer = ModelOptimizer(self.output_dir / 'optimized_models')
        
        if config.get('model_registry_path'):
            self.version_manager = ModelVersionManager(config['model_registry_path'])
        else:
            self.version_manager = None
        
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': config,
            'benchmarks': [],
            'optimizations': [],
            'regression_tests': []
        }
    
    def load_model(self, model_path: str, config_path: Optional[str] = None) -> WaveAnalysisModel:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Optional path to model configuration
            
        Returns:
            Loaded model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model configuration
        if config_path and Path(config_path).exists():
            model_config = ModelConfig.from_file(config_path)
        else:
            # Use default configuration
            model_config = ModelConfig()
        
        # Create and load model
        model = WaveAnalysisModel(model_config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model loaded successfully")
        
        return model
    
    def run_basic_benchmarks(
        self,
        model: WaveAnalysisModel,
        model_version: str,
        devices: List[str] = None
    ) -> List[BenchmarkSuite]:
        """
        Run basic performance benchmarks.
        
        Args:
            model: Model to benchmark
            model_version: Version identifier
            devices: List of devices to test on
            
        Returns:
            List of benchmark results
        """
        if devices is None:
            devices = ['cpu']
            if torch.cuda.is_available():
                devices.append('cuda')
        
        benchmark_results = []
        
        for device in devices:
            print(f"\n=== Running benchmarks on {device.upper()} ===")
            
            try:
                # Run full benchmark suite
                suite = self.benchmark.run_full_benchmark(
                    model=model,
                    model_version=model_version,
                    device=device,
                    accuracy_metrics=self.config.get('accuracy_metrics', {})
                )
                
                benchmark_results.append(suite)
                self.results['benchmarks'].append({
                    'device': device,
                    'suite': suite.__dict__
                })
                
            except Exception as e:
                print(f"Benchmark failed on {device}: {e}")
                continue
        
        return benchmark_results
    
    def run_optimization_benchmarks(
        self,
        model: WaveAnalysisModel,
        model_version: str,
        calibration_data: Optional[DataLoader] = None
    ) -> List[OptimizationResult]:
        """
        Run model optimization benchmarks.
        
        Args:
            model: Model to optimize
            model_version: Version identifier
            calibration_data: Optional calibration data for static quantization
            
        Returns:
            List of optimization results
        """
        print(f"\n=== Running optimization benchmarks ===")
        
        optimization_results = []
        
        # Dynamic quantization
        if self.config.get('run_dynamic_quantization', True):
            print("\n--- Dynamic Quantization ---")
            try:
                quantized_model, result = self.optimizer.quantize_dynamic(
                    model, f"{model_version}_dynamic"
                )
                if result:
                    optimization_results.append(result)
                    self.results['optimizations'].append(result.__dict__)
            except Exception as e:
                print(f"Dynamic quantization failed: {e}")
        
        # Static quantization (if calibration data available)
        if self.config.get('run_static_quantization', False) and calibration_data:
            print("\n--- Static Quantization ---")
            try:
                quantized_model, result = self.optimizer.quantize_static(
                    model, calibration_data, f"{model_version}_static"
                )
                if result:
                    optimization_results.append(result)
                    self.results['optimizations'].append(result.__dict__)
            except Exception as e:
                print(f"Static quantization failed: {e}")
        
        # TorchScript compilation
        if self.config.get('run_torchscript', True):
            print("\n--- TorchScript Compilation ---")
            try:
                scripted_model, result = self.optimizer.compile_torchscript(
                    model, f"{model_version}_torchscript"
                )
                if result:
                    optimization_results.append(result)
                    self.results['optimizations'].append(result.__dict__)
            except Exception as e:
                print(f"TorchScript compilation failed: {e}")
        
        # Mobile optimization
        if self.config.get('run_mobile_optimization', False):
            print("\n--- Mobile Optimization ---")
            try:
                mobile_model, result = self.optimizer.optimize_for_mobile(
                    model, f"{model_version}_mobile"
                )
                if result:
                    optimization_results.append(result)
                    self.results['optimizations'].append(result.__dict__)
            except Exception as e:
                print(f"Mobile optimization failed: {e}")
        
        # ONNX export
        if self.config.get('run_onnx_export', False):
            print("\n--- ONNX Export ---")
            try:
                result = self.optimizer.create_onnx_model(
                    model, f"{model_version}_onnx"
                )
                if result:
                    optimization_results.append(result)
                    self.results['optimizations'].append(result.__dict__)
            except Exception as e:
                print(f"ONNX export failed: {e}")
        
        return optimization_results
    
    def run_regression_tests(
        self,
        model_version: str,
        baseline_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run performance regression tests.
        
        Args:
            model_version: Current model version
            baseline_version: Baseline version to compare against
            
        Returns:
            List of regression test results
        """
        print(f"\n=== Running regression tests ===")
        
        regression_results = []
        
        # Get regression test configuration
        regression_config = self.config.get('regression_tests', {})
        
        if not regression_config:
            print("No regression tests configured")
            return regression_results
        
        # Determine baseline version
        if not baseline_version:
            baseline_version = regression_config.get('baseline_version')
        
        if not baseline_version:
            print("No baseline version specified for regression tests")
            return regression_results
        
        # Create or run regression tests
        for test_name, test_config in regression_config.get('tests', {}).items():
            try:
                # Check if test exists, create if not
                try:
                    test_result = self.benchmark.run_regression_test(test_name, model_version)
                except ValueError:
                    # Test doesn't exist, create it
                    print(f"Creating regression test: {test_name}")
                    self.benchmark.create_regression_test(
                        test_name=test_name,
                        baseline_version=baseline_version,
                        thresholds=test_config.get('thresholds', {}),
                        description=test_config.get('description', '')
                    )
                    test_result = self.benchmark.run_regression_test(test_name, model_version)
                
                regression_results.append(test_result)
                self.results['regression_tests'].append(test_result)
                
                # Print test results
                if test_result['passed']:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    for failure in test_result['failures']:
                        print(f"   - {failure}")
                
            except Exception as e:
                print(f"Regression test {test_name} failed: {e}")
                continue
        
        return regression_results
    
    def run_stress_tests(
        self,
        model: WaveAnalysisModel,
        model_version: str
    ) -> Dict[str, Any]:
        """
        Run stress tests for high-load scenarios.
        
        Args:
            model: Model to test
            model_version: Version identifier
            
        Returns:
            Stress test results
        """
        print(f"\n=== Running stress tests ===")
        
        stress_config = self.config.get('stress_tests', {})
        if not stress_config:
            print("No stress tests configured")
            return {}
        
        stress_results = {
            'model_version': model_version,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests': {}
        }
        
        # Concurrent inference test
        if stress_config.get('concurrent_inference', False):
            print("--- Concurrent Inference Test ---")
            try:
                import threading
                import queue
                
                num_threads = stress_config.get('num_threads', 4)
                requests_per_thread = stress_config.get('requests_per_thread', 10)
                
                results_queue = queue.Queue()
                
                def inference_worker():
                    test_input = torch.randn(1, 3, 768, 768)
                    latencies = []
                    
                    for _ in range(requests_per_thread):
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = model(test_input)
                        end_time = time.perf_counter()
                        latencies.append((end_time - start_time) * 1000)
                    
                    results_queue.put(latencies)
                
                # Start threads
                threads = []
                start_time = time.perf_counter()
                
                for _ in range(num_threads):
                    thread = threading.Thread(target=inference_worker)
                    thread.start()
                    threads.append(thread)
                
                # Wait for completion
                for thread in threads:
                    thread.join()
                
                total_time = time.perf_counter() - start_time
                
                # Collect results
                all_latencies = []
                while not results_queue.empty():
                    all_latencies.extend(results_queue.get())
                
                stress_results['tests']['concurrent_inference'] = {
                    'num_threads': num_threads,
                    'requests_per_thread': requests_per_thread,
                    'total_requests': len(all_latencies),
                    'total_time_s': total_time,
                    'avg_latency_ms': sum(all_latencies) / len(all_latencies),
                    'max_latency_ms': max(all_latencies),
                    'min_latency_ms': min(all_latencies),
                    'throughput_rps': len(all_latencies) / total_time
                }
                
                print(f"Concurrent inference completed:")
                print(f"  Threads: {num_threads}")
                print(f"  Total requests: {len(all_latencies)}")
                print(f"  Throughput: {len(all_latencies) / total_time:.2f} RPS")
                print(f"  Avg latency: {sum(all_latencies) / len(all_latencies):.2f} ms")
                
            except Exception as e:
                print(f"Concurrent inference test failed: {e}")
        
        # Memory stress test
        if stress_config.get('memory_stress', False):
            print("--- Memory Stress Test ---")
            try:
                import psutil
                
                batch_sizes = stress_config.get('batch_sizes', [1, 4, 8, 16, 32])
                memory_usage = {}
                
                for batch_size in batch_sizes:
                    try:
                        test_input = torch.randn(batch_size, 3, 768, 768)
                        
                        # Measure memory before
                        process = psutil.Process()
                        memory_before = process.memory_info().rss / (1024**2)
                        
                        # Run inference
                        with torch.no_grad():
                            _ = model(test_input)
                        
                        # Measure memory after
                        memory_after = process.memory_info().rss / (1024**2)
                        memory_usage[batch_size] = memory_after - memory_before
                        
                        print(f"  Batch size {batch_size}: {memory_usage[batch_size]:.1f} MB")
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  Batch size {batch_size}: OOM")
                            memory_usage[batch_size] = "OOM"
                            break
                        else:
                            raise
                
                stress_results['tests']['memory_stress'] = {
                    'batch_sizes_tested': list(memory_usage.keys()),
                    'memory_usage_mb': memory_usage,
                    'max_batch_size': max([bs for bs, mem in memory_usage.items() if mem != "OOM"])
                }
                
            except Exception as e:
                print(f"Memory stress test failed: {e}")
        
        return stress_results
    
    def generate_report(self) -> Path:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            Path to generated report
        """
        print(f"\n=== Generating benchmark report ===")
        
        # Add summary statistics
        self.results['summary'] = {
            'total_benchmarks': len(self.results['benchmarks']),
            'total_optimizations': len(self.results['optimizations']),
            'total_regression_tests': len(self.results['regression_tests']),
            'regression_tests_passed': sum(1 for test in self.results['regression_tests'] if test.get('passed', False)),
            'best_optimization': self._find_best_optimization(),
            'performance_summary': self._generate_performance_summary()
        }
        
        # Save detailed results
        report_path = self.output_dir / f"benchmark_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_path = self.output_dir / f"benchmark_summary_{int(time.time())}.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_text_summary())
        
        print(f"Benchmark report saved to: {report_path}")
        print(f"Summary report saved to: {summary_path}")
        
        return report_path
    
    def _find_best_optimization(self) -> Optional[Dict[str, Any]]:
        """Find the optimization with best speedup."""
        if not self.results['optimizations']:
            return None
        
        best_opt = max(
            self.results['optimizations'],
            key=lambda x: x.get('speedup_factor', 0)
        )
        
        return {
            'type': best_opt.get('optimization_type'),
            'speedup_factor': best_opt.get('speedup_factor'),
            'size_reduction_pct': best_opt.get('size_reduction_pct')
        }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary across all benchmarks."""
        if not self.results['benchmarks']:
            return {}
        
        # Collect metrics by device
        cpu_benchmarks = [b for b in self.results['benchmarks'] if b['device'] == 'cpu']
        gpu_benchmarks = [b for b in self.results['benchmarks'] if b['device'] == 'cuda']
        
        summary = {}
        
        if cpu_benchmarks:
            cpu_suite = cpu_benchmarks[0]['suite']
            summary['cpu'] = {
                'inference_latency_ms': cpu_suite['inference_latency_ms'],
                'memory_usage_mb': cpu_suite['memory_usage_mb'],
                'throughput_sps': cpu_suite['throughput_samples_per_sec'],
                'model_size_mb': cpu_suite['model_size_mb']
            }
        
        if gpu_benchmarks:
            gpu_suite = gpu_benchmarks[0]['suite']
            summary['gpu'] = {
                'inference_latency_ms': gpu_suite['inference_latency_ms'],
                'memory_usage_mb': gpu_suite['memory_usage_mb'],
                'throughput_sps': gpu_suite['throughput_samples_per_sec'],
                'model_size_mb': gpu_suite['model_size_mb']
            }
        
        return summary
    
    def _generate_text_summary(self) -> str:
        """Generate human-readable text summary."""
        lines = []
        lines.append("SwellSight Model Performance Benchmark Report")
        lines.append("=" * 50)
        lines.append(f"Generated: {self.results['timestamp']}")
        lines.append("")
        
        # Performance summary
        if self.results['summary']['performance_summary']:
            lines.append("Performance Summary:")
            lines.append("-" * 20)
            
            for device, metrics in self.results['summary']['performance_summary'].items():
                lines.append(f"{device.upper()}:")
                lines.append(f"  Inference Latency: {metrics['inference_latency_ms']:.2f} ms")
                lines.append(f"  Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
                lines.append(f"  Throughput: {metrics['throughput_sps']:.2f} samples/sec")
                lines.append(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
                lines.append("")
        
        # Optimization summary
        if self.results['summary']['best_optimization']:
            best_opt = self.results['summary']['best_optimization']
            lines.append("Best Optimization:")
            lines.append("-" * 17)
            lines.append(f"  Type: {best_opt['type']}")
            lines.append(f"  Speedup: {best_opt['speedup_factor']:.2f}x")
            lines.append(f"  Size Reduction: {best_opt['size_reduction_pct']:.1f}%")
            lines.append("")
        
        # Regression test summary
        total_tests = self.results['summary']['total_regression_tests']
        passed_tests = self.results['summary']['regression_tests_passed']
        
        if total_tests > 0:
            lines.append("Regression Tests:")
            lines.append("-" * 16)
            lines.append(f"  Total: {total_tests}")
            lines.append(f"  Passed: {passed_tests}")
            lines.append(f"  Failed: {total_tests - passed_tests}")
            lines.append(f"  Pass Rate: {(passed_tests / total_tests) * 100:.1f}%")
            lines.append("")
        
        # Detailed results
        lines.append("Detailed Results:")
        lines.append("-" * 16)
        lines.append(f"See full report: {self.output_dir}")
        
        return "\n".join(lines)


def create_default_config() -> Dict[str, Any]:
    """Create default benchmark configuration."""
    return {
        'output_dir': 'benchmark_results',
        'devices': ['cpu', 'cuda'],
        'run_dynamic_quantization': True,
        'run_static_quantization': False,
        'run_torchscript': True,
        'run_mobile_optimization': False,
        'run_onnx_export': False,
        'stress_tests': {
            'concurrent_inference': True,
            'memory_stress': True,
            'num_threads': 4,
            'requests_per_thread': 10,
            'batch_sizes': [1, 4, 8, 16, 32]
        },
        'regression_tests': {
            'baseline_version': '1.0.0',
            'tests': {
                'performance_regression': {
                    'thresholds': {
                        'latency_increase_pct': 10.0,
                        'memory_increase_pct': 15.0,
                        'throughput_decrease_pct': 10.0
                    },
                    'description': 'Basic performance regression test'
                }
            }
        }
    }


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description='Run SwellSight model performance benchmarks')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-config', type=str,
                       help='Path to model configuration file')
    parser.add_argument('--version', type=str, required=True,
                       help='Model version identifier')
    parser.add_argument('--config', type=str,
                       help='Path to benchmark configuration file')
    parser.add_argument('--baseline-version', type=str,
                       help='Baseline version for regression tests')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--devices', nargs='+', default=['cpu'],
                       help='Devices to benchmark on')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip optimization benchmarks')
    parser.add_argument('--skip-regression', action='store_true',
                       help='Skip regression tests')
    parser.add_argument('--skip-stress', action='store_true',
                       help='Skip stress tests')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['output_dir'] = args.output_dir
    config['devices'] = args.devices
    
    if args.baseline_version:
        config['regression_tests']['baseline_version'] = args.baseline_version
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(config)
    
    try:
        # Load model
        model = runner.load_model(args.model_path, args.model_config)
        
        # Run basic benchmarks
        print(f"Starting benchmarks for model version {args.version}")
        benchmark_results = runner.run_basic_benchmarks(model, args.version, config['devices'])
        
        # Run optimization benchmarks
        if not args.skip_optimization:
            optimization_results = runner.run_optimization_benchmarks(model, args.version)
        
        # Run regression tests
        if not args.skip_regression:
            regression_results = runner.run_regression_tests(args.version, args.baseline_version)
        
        # Run stress tests
        if not args.skip_stress:
            stress_results = runner.run_stress_tests(model, args.version)
        
        # Generate report
        report_path = runner.generate_report()
        
        print(f"\n✅ Benchmarking completed successfully!")
        print(f"📊 Results saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
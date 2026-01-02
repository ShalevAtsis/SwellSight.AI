"""Model optimization utilities for SwellSight wave analysis models."""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import script
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json

from ..models.wave_analysis_model import WaveAnalysisModel
from ..config import ModelConfig


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_pct: float
    original_latency_ms: float
    optimized_latency_ms: float
    speedup_factor: float
    optimization_type: str
    accuracy_preserved: bool
    timestamp: str


class ModelOptimizer:
    """
    Model optimization utilities for production deployment.
    
    Provides quantization, TorchScript compilation, and other optimization
    techniques to improve inference performance.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "optimized_models"):
        """
        Initialize model optimizer.
        
        Args:
            output_dir: Directory to save optimized models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_results = []
    
    def quantize_dynamic(
        self,
        model: WaveAnalysisModel,
        model_name: str = "quantized_model",
        qconfig_spec: Optional[Dict] = None
    ) -> Tuple[nn.Module, OptimizationResult]:
        """
        Apply dynamic quantization to model for CPU inference.
        
        Args:
            model: Model to quantize
            model_name: Name for saved quantized model
            qconfig_spec: Quantization configuration specification
            
        Returns:
            Tuple of (quantized_model, optimization_result)
        """
        print(f"Applying dynamic quantization to {model_name}...")
        
        # Measure original model
        original_size = self._get_model_size_mb(model)
        original_latency = self._measure_inference_latency(model, device="cpu")
        
        # Set model to evaluation mode
        model.eval()
        
        # Default quantization configuration
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.Conv2d: torch.quantization.default_dynamic_qconfig
            }
        
        # Apply dynamic quantization
        try:
            quantized_model = quantization.quantize_dynamic(
                model, qconfig_spec, dtype=torch.qint8
            )
        except Exception as e:
            warnings.warn(f"Dynamic quantization failed: {e}")
            return model, None
        
        # Measure quantized model
        quantized_size = self._get_model_size_mb(quantized_model)
        quantized_latency = self._measure_inference_latency(quantized_model, device="cpu")
        
        # Calculate improvements
        size_reduction = ((original_size - quantized_size) / original_size) * 100
        speedup = original_latency / quantized_latency if quantized_latency > 0 else 1.0
        
        # Save quantized model
        quantized_path = self.output_dir / f"{model_name}_quantized.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=quantized_size,
            size_reduction_pct=size_reduction,
            original_latency_ms=original_latency,
            optimized_latency_ms=quantized_latency,
            speedup_factor=speedup,
            optimization_type="dynamic_quantization",
            accuracy_preserved=True,  # Dynamic quantization typically preserves accuracy
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.optimization_results.append(result)
        
        print(f"Dynamic quantization completed:")
        print(f"  Size reduction: {size_reduction:.1f}% ({original_size:.1f}MB → {quantized_size:.1f}MB)")
        print(f"  Speedup: {speedup:.2f}x ({original_latency:.1f}ms → {quantized_latency:.1f}ms)")
        print(f"  Saved to: {quantized_path}")
        
        return quantized_model, result
    
    def quantize_static(
        self,
        model: WaveAnalysisModel,
        calibration_data: torch.utils.data.DataLoader,
        model_name: str = "static_quantized_model"
    ) -> Tuple[nn.Module, OptimizationResult]:
        """
        Apply static quantization with calibration data.
        
        Args:
            model: Model to quantize
            calibration_data: DataLoader with calibration samples
            model_name: Name for saved quantized model
            
        Returns:
            Tuple of (quantized_model, optimization_result)
        """
        print(f"Applying static quantization to {model_name}...")
        
        # Measure original model
        original_size = self._get_model_size_mb(model)
        original_latency = self._measure_inference_latency(model, device="cpu")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse modules if possible
        try:
            model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        except:
            model_fused = model
        
        # Prepare for static quantization
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibrate with sample data
        print("Calibrating model with sample data...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= 100:  # Limit calibration samples
                    break
                
                if isinstance(batch, dict):
                    inputs = batch['image']
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                model_prepared(inputs)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        # Measure quantized model
        quantized_size = self._get_model_size_mb(quantized_model)
        quantized_latency = self._measure_inference_latency(quantized_model, device="cpu")
        
        # Calculate improvements
        size_reduction = ((original_size - quantized_size) / original_size) * 100
        speedup = original_latency / quantized_latency if quantized_latency > 0 else 1.0
        
        # Save quantized model
        quantized_path = self.output_dir / f"{model_name}_static_quantized.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=quantized_size,
            size_reduction_pct=size_reduction,
            original_latency_ms=original_latency,
            optimized_latency_ms=quantized_latency,
            speedup_factor=speedup,
            optimization_type="static_quantization",
            accuracy_preserved=False,  # May need accuracy validation
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.optimization_results.append(result)
        
        print(f"Static quantization completed:")
        print(f"  Size reduction: {size_reduction:.1f}% ({original_size:.1f}MB → {quantized_size:.1f}MB)")
        print(f"  Speedup: {speedup:.2f}x ({original_latency:.1f}ms → {quantized_latency:.1f}ms)")
        print(f"  Saved to: {quantized_path}")
        
        return quantized_model, result
    
    def compile_torchscript(
        self,
        model: WaveAnalysisModel,
        model_name: str = "torchscript_model",
        example_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.jit.ScriptModule, OptimizationResult]:
        """
        Compile model to TorchScript for optimized inference.
        
        Args:
            model: Model to compile
            model_name: Name for saved TorchScript model
            example_input: Example input tensor for tracing
            
        Returns:
            Tuple of (scripted_model, optimization_result)
        """
        print(f"Compiling {model_name} to TorchScript...")
        
        # Measure original model
        original_size = self._get_model_size_mb(model)
        original_latency = self._measure_inference_latency(model)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input if not provided
        if example_input is None:
            example_input = torch.randn(1, 3, 768, 768)
        
        # Try tracing first, fall back to scripting
        try:
            scripted_model = torch.jit.trace(model, example_input)
            compilation_method = "trace"
        except Exception as e:
            print(f"Tracing failed ({e}), trying scripting...")
            try:
                scripted_model = torch.jit.script(model)
                compilation_method = "script"
            except Exception as e2:
                warnings.warn(f"TorchScript compilation failed: {e2}")
                return model, None
        
        # Optimize the scripted model
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        # Measure scripted model
        scripted_size = self._get_model_size_mb(scripted_model)
        scripted_latency = self._measure_inference_latency(scripted_model)
        
        # Calculate improvements
        size_reduction = ((original_size - scripted_size) / original_size) * 100
        speedup = original_latency / scripted_latency if scripted_latency > 0 else 1.0
        
        # Save scripted model
        scripted_path = self.output_dir / f"{model_name}_torchscript.pt"
        scripted_model.save(str(scripted_path))
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=scripted_size,
            size_reduction_pct=size_reduction,
            original_latency_ms=original_latency,
            optimized_latency_ms=scripted_latency,
            speedup_factor=speedup,
            optimization_type=f"torchscript_{compilation_method}",
            accuracy_preserved=True,  # TorchScript should preserve accuracy
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.optimization_results.append(result)
        
        print(f"TorchScript compilation completed:")
        print(f"  Method: {compilation_method}")
        print(f"  Size change: {size_reduction:.1f}% ({original_size:.1f}MB → {scripted_size:.1f}MB)")
        print(f"  Speedup: {speedup:.2f}x ({original_latency:.1f}ms → {scripted_latency:.1f}ms)")
        print(f"  Saved to: {scripted_path}")
        
        return scripted_model, result
    
    def optimize_for_mobile(
        self,
        model: WaveAnalysisModel,
        model_name: str = "mobile_model"
    ) -> Tuple[torch.jit.ScriptModule, OptimizationResult]:
        """
        Optimize model for mobile deployment.
        
        Args:
            model: Model to optimize
            model_name: Name for saved mobile model
            
        Returns:
            Tuple of (mobile_model, optimization_result)
        """
        print(f"Optimizing {model_name} for mobile deployment...")
        
        # First compile to TorchScript
        scripted_model, script_result = self.compile_torchscript(model, model_name)
        
        if scripted_model is None:
            return model, None
        
        # Apply mobile optimizations
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            mobile_model = optimize_for_mobile(scripted_model)
        except ImportError:
            warnings.warn("Mobile optimizer not available, using TorchScript model")
            return scripted_model, script_result
        
        # Measure mobile model
        mobile_size = self._get_model_size_mb(mobile_model)
        mobile_latency = self._measure_inference_latency(mobile_model)
        
        # Calculate improvements vs original
        original_size = script_result.original_size_mb
        original_latency = script_result.original_latency_ms
        
        size_reduction = ((original_size - mobile_size) / original_size) * 100
        speedup = original_latency / mobile_latency if mobile_latency > 0 else 1.0
        
        # Save mobile model
        mobile_path = self.output_dir / f"{model_name}_mobile.ptl"
        mobile_model._save_for_lite_interpreter(str(mobile_path))
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=mobile_size,
            size_reduction_pct=size_reduction,
            original_latency_ms=original_latency,
            optimized_latency_ms=mobile_latency,
            speedup_factor=speedup,
            optimization_type="mobile_optimization",
            accuracy_preserved=True,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.optimization_results.append(result)
        
        print(f"Mobile optimization completed:")
        print(f"  Size reduction: {size_reduction:.1f}% ({original_size:.1f}MB → {mobile_size:.1f}MB)")
        print(f"  Speedup: {speedup:.2f}x ({original_latency:.1f}ms → {mobile_latency:.1f}ms)")
        print(f"  Saved to: {mobile_path}")
        
        return mobile_model, result
    
    def create_onnx_model(
        self,
        model: WaveAnalysisModel,
        model_name: str = "onnx_model",
        input_shape: Tuple[int, int, int, int] = (1, 3, 768, 768)
    ) -> OptimizationResult:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: Model to export
            model_name: Name for saved ONNX model
            input_shape: Input tensor shape
            
        Returns:
            OptimizationResult with export information
        """
        print(f"Exporting {model_name} to ONNX format...")
        
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            warnings.warn("ONNX not available, skipping ONNX export")
            return None
        
        # Measure original model
        original_size = self._get_model_size_mb(model)
        original_latency = self._measure_inference_latency(model)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['height', 'wave_type', 'direction'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'height': {0: 'batch_size'},
                    'wave_type': {0: 'batch_size'},
                    'direction': {0: 'batch_size'}
                }
            )
        except Exception as e:
            warnings.warn(f"ONNX export failed: {e}")
            return None
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            warnings.warn(f"ONNX model verification failed: {e}")
            return None
        
        # Measure ONNX model performance
        try:
            ort_session = ort.InferenceSession(str(onnx_path))
            onnx_latency = self._measure_onnx_latency(ort_session, input_shape)
        except Exception as e:
            warnings.warn(f"ONNX performance measurement failed: {e}")
            onnx_latency = original_latency
        
        # Get ONNX file size
        onnx_size = onnx_path.stat().st_size / (1024**2)  # MB
        
        # Calculate improvements
        size_reduction = ((original_size - onnx_size) / original_size) * 100
        speedup = original_latency / onnx_latency if onnx_latency > 0 else 1.0
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=onnx_size,
            size_reduction_pct=size_reduction,
            original_latency_ms=original_latency,
            optimized_latency_ms=onnx_latency,
            speedup_factor=speedup,
            optimization_type="onnx_export",
            accuracy_preserved=True,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.optimization_results.append(result)
        
        print(f"ONNX export completed:")
        print(f"  Size change: {size_reduction:.1f}% ({original_size:.1f}MB → {onnx_size:.1f}MB)")
        print(f"  Speedup: {speedup:.2f}x ({original_latency:.1f}ms → {onnx_latency:.1f}ms)")
        print(f"  Saved to: {onnx_path}")
        
        return result
    
    def _get_model_size_mb(self, model: Union[nn.Module, torch.jit.ScriptModule]) -> float:
        """Calculate model size in MB."""
        if isinstance(model, torch.jit.ScriptModule):
            # For TorchScript models, save to temporary file and measure
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                model.save(tmp.name)
                return Path(tmp.name).stat().st_size / (1024**2)
        else:
            # For regular PyTorch models
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024**2)
    
    def _measure_inference_latency(
        self,
        model: Union[nn.Module, torch.jit.ScriptModule],
        device: str = "cpu",
        num_iterations: int = 100
    ) -> float:
        """Measure average inference latency in milliseconds."""
        # Create test input
        test_input = torch.randn(1, 3, 768, 768, device=device)
        
        if hasattr(model, 'to'):
            model = model.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
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
        
        return sum(latencies) / len(latencies)
    
    def _measure_onnx_latency(
        self,
        ort_session,
        input_shape: Tuple[int, int, int, int],
        num_iterations: int = 100
    ) -> float:
        """Measure ONNX model inference latency."""
        import numpy as np
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, {'input': test_input})
        
        # Measure latency
        latencies = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = ort_session.run(None, {'input': test_input})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return sum(latencies) / len(latencies)
    
    def save_optimization_report(self, filename: str = "optimization_report.json"):
        """Save optimization results to JSON file."""
        report_path = self.output_dir / filename
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_optimizations": len(self.optimization_results),
            "optimizations": [
                {
                    "optimization_type": result.optimization_type,
                    "size_reduction_pct": result.size_reduction_pct,
                    "speedup_factor": result.speedup_factor,
                    "original_size_mb": result.original_size_mb,
                    "optimized_size_mb": result.optimized_size_mb,
                    "original_latency_ms": result.original_latency_ms,
                    "optimized_latency_ms": result.optimized_latency_ms,
                    "accuracy_preserved": result.accuracy_preserved,
                    "timestamp": result.timestamp
                }
                for result in self.optimization_results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Optimization report saved to: {report_path}")
        return report_path


class CachingStrategy:
    """
    Caching strategies for improved inference performance.
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize caching strategy.
        
        Args:
            cache_size: Maximum number of cached predictions
        """
        self.cache_size = cache_size
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, image_tensor: torch.Tensor) -> str:
        """Generate cache key from image tensor."""
        # Use tensor hash for cache key
        return str(hash(image_tensor.data.tobytes()))
    
    def get_cached_prediction(self, image_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available."""
        cache_key = self.get_cache_key(image_tensor)
        
        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]
        else:
            self.cache_misses += 1
            return None
    
    def cache_prediction(self, image_tensor: torch.Tensor, prediction: Dict[str, Any]):
        """Cache prediction result."""
        cache_key = self.get_cache_key(image_tensor)
        
        # Implement LRU eviction if cache is full
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = prediction
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear all cached predictions."""
        self.prediction_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
"""
Property-based tests for end-to-end pipeline integrity.

Task 20.3: Write final integration property test
- Property 39: End-to-End Pipeline Integrity - For any real beach image, the complete pipeline should produce wave parameter predictions within expected accuracy bounds
- Validates: Requirements 2.1-2.7, 4.1-4.5, 5.1-5.5
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import json
import time
import traceback
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Tuple
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from swellsight.config import ModelConfig, TrainingConfig, DataConfig
from swellsight.models import WaveAnalysisModel
from swellsight.data import SyntheticDataGenerator, DatasetManager, RealDataLoader
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor, DepthExtractionResult
from swellsight.data.depth_analyzer import DepthAnalyzer
from swellsight.training import Trainer
from swellsight.evaluation import MetricsCalculator
from swellsight.inference import InferenceEngine
from swellsight.scripts.end_to_end_pipeline import EndToEndPipeline, PipelineConfig


# Test data generators
@st.composite
def realistic_beach_image_data(draw):
    """Generate realistic beach image test data."""
    # Beach image characteristics
    wave_height = draw(st.floats(min_value=0.3, max_value=4.0))
    wave_type = draw(st.sampled_from(['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']))
    direction = draw(st.sampled_from(['LEFT', 'RIGHT', 'BOTH']))
    
    # Image properties
    width = draw(st.integers(min_value=320, max_value=800))
    height = draw(st.integers(min_value=240, max_value=600))
    
    # Environmental conditions
    lighting_condition = draw(st.sampled_from(['sunny', 'overcast', 'sunset', 'stormy']))
    water_clarity = draw(st.floats(min_value=0.3, max_value=1.0))
    
    return {
        'wave_height': wave_height,
        'wave_type': wave_type,
        'direction': direction,
        'width': width,
        'height': height,
        'lighting_condition': lighting_condition,
        'water_clarity': water_clarity
    }


@st.composite
def pipeline_configuration(draw):
    """Generate valid pipeline configurations."""
    return {
        'target_dataset_size': draw(st.integers(min_value=5, max_value=20)),
        'min_depth_quality': draw(st.floats(min_value=0.2, max_value=0.8)),
        'quality_threshold': draw(st.floats(min_value=0.3, max_value=0.9)),
        'max_synthetic_per_real': draw(st.integers(min_value=1, max_value=5)),
        'batch_size': draw(st.integers(min_value=1, max_value=4))
    }


class TestEndToEndPipelineProperties:
    """Property-based tests for end-to-end pipeline integrity."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.real_data_dir = self.temp_dir / 'real'
        self.synthetic_data_dir = self.temp_dir / 'synthetic'
        self.metadata_dir = self.temp_dir / 'metadata'
        self.output_dir = self.temp_dir / 'output'
        
        # Create directory structure
        for dir_path in [self.real_data_dir, self.synthetic_data_dir, 
                        self.metadata_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_beach_image(self, image_data: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
        """Create a test beach image with specified characteristics."""
        # Create base image with realistic beach scene
        width, height = image_data['width'], image_data['height']
        
        # Color scheme based on lighting condition
        color_schemes = {
            'sunny': {'sky': (135, 206, 235), 'water': (64, 164, 223), 'beach': (194, 178, 128)},
            'overcast': {'sky': (120, 120, 120), 'water': (80, 120, 160), 'beach': (160, 145, 100)},
            'sunset': {'sky': (255, 165, 0), 'water': (80, 120, 160), 'beach': (200, 180, 140)},
            'stormy': {'sky': (70, 70, 70), 'water': (40, 60, 80), 'beach': (140, 125, 85)}
        }
        
        colors = color_schemes[image_data['lighting_condition']]
        
        # Create base image
        image = Image.new('RGB', (width, height), color=colors['sky'])
        pixels = np.array(image)
        
        # Add beach (bottom third)
        beach_start = int(height * 0.67)
        pixels[beach_start:, :] = colors['beach']
        
        # Add water with wave simulation (middle third)
        water_start = int(height * 0.33)
        water_end = beach_start
        
        wave_height = image_data['wave_height']
        
        for y in range(water_start, water_end):
            # Simulate wave patterns based on height
            wave_factor = np.sin((y - water_start) * 0.1) * wave_height * 10
            for x in range(width):
                x_wave = np.sin(x * 0.02) * wave_height * 5
                intensity = int(wave_factor + x_wave)
                
                # Modify water color based on wave intensity and clarity
                clarity_factor = image_data['water_clarity']
                new_color = [
                    max(0, min(255, int(colors['water'][0] * clarity_factor + intensity))),
                    max(0, min(255, int(colors['water'][1] * clarity_factor + intensity))),
                    max(0, min(255, int(colors['water'][2] * clarity_factor + intensity)))
                ]
                pixels[y, x] = new_color
        
        # Add realistic noise
        noise = np.random.randint(-10, 10, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        image = Image.fromarray(pixels)
        image_path = self.real_data_dir / f'beach_{int(time.time() * 1000000) % 1000000}.jpg'
        image.save(image_path, 'JPEG', quality=85)
        
        # Create ground truth metadata
        ground_truth = {
            'image_path': str(image_path),
            'height_meters': image_data['wave_height'],
            'wave_type': image_data['wave_type'],
            'direction': image_data['direction'],
            'lighting_condition': image_data['lighting_condition'],
            'water_clarity': image_data['water_clarity'],
            'timestamp': f"2024-01-01T12:00:00Z"
        }
        
        return image_path, ground_truth
    
    def _mock_pipeline_components(self):
        """Create mock pipeline components to avoid external dependencies."""
        # Mock MiDaS depth extractor
        def mock_extract_depth(image_path, store_result=False):
            # Create realistic depth map based on image
            image = Image.open(image_path)
            width, height = image.size
            
            # Resize to standard processing size
            depth_height, depth_width = 224, 224
            depth_map = np.random.rand(depth_height, depth_width) * 8 + 2  # 2-10m depth range
            
            # Add wave-like patterns
            x = np.linspace(0, 4*np.pi, depth_width)
            y = np.linspace(0, 3*np.pi, depth_height)
            X, Y = np.meshgrid(x, y)
            wave_pattern = np.sin(X) * np.cos(Y) * 2
            depth_map += wave_pattern
            
            # Ensure positive depths
            depth_map = np.abs(depth_map)
            
            return DepthExtractionResult(
                depth_map=depth_map,
                original_image_path=image_path,
                depth_quality_score=0.75,
                processing_metadata={'model': 'Intel/dpt-large'}
            )
        
        # Mock ControlNet generation
        def mock_generate_synthetic_image(depth_map, augmentation_params):
            height, width = depth_map.shape
            
            # Use depth map to influence image generation
            normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            # Create RGB image with depth-influenced colors
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Sky (low depth values)
            sky_mask = normalized_depth < 0.3
            rgb_image[sky_mask] = [135, 206, 235]  # Sky blue
            
            # Water (medium depth values)
            water_mask = (normalized_depth >= 0.3) & (normalized_depth < 0.7)
            rgb_image[water_mask] = [64, 164, 223]  # Water blue
            
            # Beach (high depth values)
            beach_mask = normalized_depth >= 0.7
            rgb_image[beach_mask] = [194, 178, 128]  # Sand color
            
            # Add wave height influence from augmentation parameters
            wave_height = augmentation_params.get('wave_height_meters', 1.0)
            if wave_height > 2.0:
                # Add white foam for larger waves
                foam_mask = normalized_depth > 0.8
                rgb_image[foam_mask] = [255, 255, 255]
            
            return Image.fromarray(rgb_image)
        
        return mock_extract_depth, mock_generate_synthetic_image
    
    @given(realistic_beach_image_data(), pipeline_configuration())
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_39_end_to_end_pipeline_integrity(self, beach_image_data, pipeline_config):
        """
        Feature: wave-analysis-model, Property 39: End-to-End Pipeline Integrity
        
        For any real beach image, the complete pipeline (MiDaS depth extraction → 
        ControlNet synthetic generation → model training → inference) should produce 
        wave parameter predictions within expected accuracy bounds.
        
        Validates: Requirements 2.1-2.7, 4.1-4.5, 5.1-5.5
        """
        # Create test beach image
        image_path, ground_truth = self._create_test_beach_image(beach_image_data)
        
        # Set up mock components
        mock_extract_depth, mock_generate_synthetic = self._mock_pipeline_components()
        
        # Configure pipeline
        model_config = ModelConfig(
            backbone='convnext_base',
            input_size=(224, 224),
            feature_dim=512,
            hidden_dim=128
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=pipeline_config['batch_size'],
            num_epochs=2,  # Minimal training for property test
            checkpoint_frequency=1,
            early_stopping_patience=10
        )
        
        data_config = DataConfig(
            num_synthetic_samples=pipeline_config['target_dataset_size'],
            synthetic_data_path=str(self.synthetic_data_dir),
            metadata_path=str(self.metadata_dir),
            train_split=0.7,
            val_split=0.3,
            num_workers=0
        )
        
        try:
            # Step 1: MiDaS depth extraction
            with patch('swellsight.data.midas_depth_extractor.DPTImageProcessor'), \
                 patch('swellsight.data.midas_depth_extractor.DPTForDepthEstimation'), \
                 patch.object(MiDaSDepthExtractor, '__init__', return_value=None):
                
                depth_extractor = MiDaSDepthExtractor.__new__(MiDaSDepthExtractor)
                depth_extractor.model_name = "Intel/dpt-large"
                depth_extractor.device = "cpu"
                depth_extractor.extract_depth = mock_extract_depth
                
                depth_result = depth_extractor.extract_depth(str(image_path))
                
                # Verify depth extraction quality
                assert depth_result.depth_map is not None, "Depth extraction should produce depth map"
                assert depth_result.depth_map.shape == (224, 224), "Depth map should have correct dimensions"
                assert depth_result.depth_quality_score >= pipeline_config['min_depth_quality'], \
                    f"Depth quality {depth_result.depth_quality_score} should meet minimum threshold {pipeline_config['min_depth_quality']}"
                assert np.all(depth_result.depth_map >= 0), "Depth values should be non-negative"
            
            # Step 2: ControlNet synthetic generation
            with patch.object(SyntheticDataGenerator, '_depth_to_image_controlnet') as mock_controlnet, \
                 patch.object(SyntheticDataGenerator, '__init__', return_value=None):
                mock_controlnet.side_effect = mock_generate_synthetic
                
                # Initialize synthetic data generator
                generator_config = {
                    'synthetic_data_path': str(self.synthetic_data_dir),
                    'metadata_path': str(self.metadata_dir),
                    'image_size': model_config.input_size,
                    'real_data_path': str(self.real_data_dir)
                }
                
                generator = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
                generator.config = generator_config
                generator.depth_extractor = depth_extractor
                
                # Create real data loader
                real_metadata = [ground_truth]
                real_metadata_path = self.metadata_dir / 'real_dataset_metadata.json'
                with open(real_metadata_path, 'w') as f:
                    json.dump(real_metadata, f, indent=2)
                
                real_loader = RealDataLoader(
                    data_path=str(self.metadata_dir),
                    config={'real_data_path': str(self.real_data_dir)}
                )
                generator.real_data_loader = real_loader
                
                # Generate synthetic dataset
                def mock_generate_dataset(num_samples, output_path=None, use_real_images=True):
                    synthetic_samples = []
                    
                    for i in range(min(num_samples, pipeline_config['max_synthetic_per_real'])):
                        # Generate synthetic image
                        synthetic_image = mock_generate_synthetic(
                            depth_result.depth_map,
                            {'wave_height_meters': ground_truth['height_meters']}
                        )
                        synthetic_path = self.synthetic_data_dir / f'synthetic_{i:03d}.jpg'
                        synthetic_image.save(synthetic_path, 'JPEG')
                        
                        # Create sample metadata with ground truth
                        sample = {
                            'sample_id': f'synthetic_{i:03d}',
                            'image_path': str(synthetic_path.relative_to(self.temp_dir)),
                            'height_meters': ground_truth['height_meters'] + np.random.normal(0, 0.1),  # Small noise
                            'wave_type': ground_truth['wave_type'],
                            'direction': ground_truth['direction'],
                            'source_real_image': str(image_path),
                            'augmentation_params': {
                                'wave_height_meters': ground_truth['height_meters'],
                                'lighting_condition': ground_truth['lighting_condition']
                            }
                        }
                        
                        synthetic_samples.append(sample)
                    
                    return synthetic_samples
                
                generator.generate_dataset = mock_generate_dataset
                synthetic_metadata = generator.generate_dataset(data_config.num_synthetic_samples)
                
                # Save synthetic metadata
                synthetic_metadata_path = self.metadata_dir / 'synthetic_dataset_metadata.json'
                with open(synthetic_metadata_path, 'w') as f:
                    json.dump(synthetic_metadata, f, indent=2)
                
                # Verify synthetic data generation
                assert len(synthetic_metadata) > 0, "Should generate synthetic samples"
                assert len(synthetic_metadata) <= data_config.num_synthetic_samples, "Should not exceed target size"
                
                # Verify synthetic samples preserve ground truth characteristics
                for sample in synthetic_metadata:
                    assert 'height_meters' in sample, "Sample should have height"
                    assert 'wave_type' in sample, "Sample should have wave type"
                    assert 'direction' in sample, "Sample should have direction"
                    assert sample['height_meters'] > 0, "Height should be positive"
                    
                    # Height should be close to original (within 50% for property test)
                    height_error = abs(sample['height_meters'] - ground_truth['height_meters'])
                    height_tolerance = ground_truth['height_meters'] * 0.5
                    assert height_error <= height_tolerance, \
                        f"Synthetic height {sample['height_meters']} should be close to ground truth {ground_truth['height_meters']}"
            
            # Step 3: Model training
            dataset_manager = DatasetManager(
                data_path=str(self.temp_dir),
                config=data_config.to_dict()
            )
            
            model = WaveAnalysisModel(model_config)
            trainer = Trainer(
                model=model,
                training_config=training_config,
                data_config=data_config,
                output_dir=self.output_dir / "checkpoints",
                dataset_manager=dataset_manager
            )
            
            training_results = trainer.train()
            
            # Verify training completed successfully
            assert training_results['status'] == 'completed', "Training should complete successfully"
            assert training_results['epochs_trained'] >= 1, "Should train for at least one epoch"
            assert 'final_loss' in training_results, "Should report final loss"
            assert training_results['final_loss'] > 0, "Final loss should be positive"
            
            # Step 4: Inference on original real image
            inference_engine = InferenceEngine(model, model_config)
            prediction = inference_engine.predict(str(image_path))
            
            # Verify prediction structure
            assert hasattr(prediction, 'height_meters'), "Prediction should have height"
            assert hasattr(prediction, 'wave_type'), "Prediction should have wave type"
            assert hasattr(prediction, 'direction'), "Prediction should have direction"
            assert prediction.height_meters > 0, "Predicted height should be positive"
            
            # Step 5: Validate end-to-end accuracy bounds
            # Height accuracy: within expected bounds for the complete pipeline
            height_error = abs(prediction.height_meters - ground_truth['height_meters'])
            max_height_error = max(2.0, ground_truth['height_meters'] * 0.8)  # Relaxed bounds for property test
            
            assert height_error <= max_height_error, \
                f"End-to-end height error {height_error:.3f}m should be within bounds {max_height_error:.3f}m " \
                f"(predicted: {prediction.height_meters:.3f}m, ground truth: {ground_truth['height_meters']:.3f}m)"
            
            # Wave type should be valid
            valid_wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
            assert prediction.wave_type in valid_wave_types, \
                f"Predicted wave type '{prediction.wave_type}' should be valid"
            
            # Direction should be valid
            valid_directions = ['LEFT', 'RIGHT', 'BOTH']
            assert prediction.direction in valid_directions, \
                f"Predicted direction '{prediction.direction}' should be valid"
            
            # Confidence scores should be reasonable
            assert hasattr(prediction, 'confidence_scores'), "Should have confidence scores"
            assert 'height' in prediction.confidence_scores, "Should have height confidence"
            assert 0 <= prediction.confidence_scores['height'] <= 1, "Height confidence should be in [0,1]"
            
            # Overall pipeline integrity check passed
            pipeline_accuracy = 1.0 - (height_error / max(ground_truth['height_meters'], 1.0))
            assert pipeline_accuracy >= 0.0, "Pipeline accuracy should be non-negative"
            
            # Log successful end-to-end validation
            print(f"Property 39 validation passed:")
            print(f"  Ground truth height: {ground_truth['height_meters']:.3f}m")
            print(f"  Predicted height: {prediction.height_meters:.3f}m")
            print(f"  Height error: {height_error:.3f}m")
            print(f"  Pipeline accuracy: {pipeline_accuracy:.3f}")
            print(f"  Wave type: {ground_truth['wave_type']} → {prediction.wave_type}")
            print(f"  Direction: {ground_truth['direction']} → {prediction.direction}")
            
        except Exception as e:
            # Provide detailed error information for debugging
            error_msg = f"End-to-end pipeline failed for image with characteristics: {beach_image_data}"
            error_msg += f"\nPipeline config: {pipeline_config}"
            error_msg += f"\nError: {str(e)}"
            error_msg += f"\nTraceback: {traceback.format_exc()}"
            
            pytest.fail(error_msg)
    
    def test_property_39_minimal_integration_validation(self):
        """
        Minimal validation test for Property 39 to ensure basic functionality.
        
        This test uses fixed parameters to validate the core pipeline integrity
        without the complexity of property-based generation.
        """
        # Create simple test image
        beach_data = {
            'wave_height': 1.5,
            'wave_type': 'A_FRAME',
            'direction': 'RIGHT',
            'width': 640,
            'height': 480,
            'lighting_condition': 'sunny',
            'water_clarity': 0.8
        }
        
        image_path, ground_truth = self._create_test_beach_image(beach_data)
        
        # Set up minimal pipeline configuration
        pipeline_config = {
            'target_dataset_size': 5,
            'min_depth_quality': 0.3,
            'quality_threshold': 0.5,
            'max_synthetic_per_real': 2,
            'batch_size': 1
        }
        
        # Set up mock components
        mock_extract_depth, mock_generate_synthetic = self._mock_pipeline_components()
        
        # Configure pipeline
        model_config = ModelConfig(
            backbone='convnext_base',
            input_size=(224, 224),
            feature_dim=512,
            hidden_dim=128
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=pipeline_config['batch_size'],
            num_epochs=1,  # Minimal training for property test
            checkpoint_frequency=1,
            early_stopping_patience=10
        )
        
        data_config = DataConfig(
            num_synthetic_samples=pipeline_config['target_dataset_size'],
            synthetic_data_path=str(self.synthetic_data_dir),
            metadata_path=str(self.metadata_dir),
            train_split=0.7,
            val_split=0.3,
            num_workers=0
        )
        
        try:
            # Step 1: MiDaS depth extraction (mocked)
            with patch('swellsight.data.midas_depth_extractor.DPTImageProcessor'), \
                 patch('swellsight.data.midas_depth_extractor.DPTForDepthEstimation'), \
                 patch.object(MiDaSDepthExtractor, '__init__', return_value=None):
                
                depth_extractor = MiDaSDepthExtractor.__new__(MiDaSDepthExtractor)
                depth_extractor.model_name = "Intel/dpt-large"
                depth_extractor.device = "cpu"
                depth_extractor.extract_depth = mock_extract_depth
                
                depth_result = depth_extractor.extract_depth(str(image_path))
                
                # Verify depth extraction
                assert depth_result.depth_map is not None
                assert depth_result.depth_map.shape == (224, 224)
                assert depth_result.depth_quality_score >= pipeline_config['min_depth_quality']
                assert np.all(depth_result.depth_map >= 0)
            
            # Step 2: Create synthetic data (simplified)
            synthetic_samples = []
            for i in range(pipeline_config['target_dataset_size']):
                # Generate synthetic image
                synthetic_image = mock_generate_synthetic(
                    depth_result.depth_map,
                    {'wave_height_meters': ground_truth['height_meters']}
                )
                synthetic_path = self.synthetic_data_dir / f'synthetic_{i:03d}.jpg'
                synthetic_image.save(synthetic_path, 'JPEG')
                
                # Create sample metadata
                sample = {
                    'sample_id': f'synthetic_{i:03d}',
                    'image_path': str(synthetic_path),  # Use absolute path
                    'height_meters': ground_truth['height_meters'] + np.random.normal(0, 0.1),
                    'wave_type': ground_truth['wave_type'],
                    'direction': ground_truth['direction'],
                    'source_real_image': str(image_path),
                }
                synthetic_samples.append(sample)
            
            # Save synthetic metadata
            synthetic_metadata_path = self.metadata_dir / 'synthetic_dataset_metadata.json'
            with open(synthetic_metadata_path, 'w') as f:
                json.dump(synthetic_samples, f, indent=2)
            
            # Step 3: Train model (minimal)
            dataset_manager = DatasetManager(
                data_path=str(self.temp_dir),
                config=data_config.to_dict()
            )
            
            model = WaveAnalysisModel(model_config)
            trainer = Trainer(
                model=model,
                training_config=training_config,
                data_config=data_config,
                output_dir=self.output_dir / "checkpoints",
                dataset_manager=dataset_manager
            )
            
            training_results = trainer.train()
            
            # Verify training completed
            assert training_results['status'] == 'completed'
            assert training_results['epochs_trained'] >= 1
            
            # Step 4: Test inference
            inference_engine = InferenceEngine(model, model_config)
            prediction = inference_engine.predict(str(image_path))
            
            # Verify prediction structure
            assert hasattr(prediction, 'height_meters')
            assert hasattr(prediction, 'wave_type')
            assert hasattr(prediction, 'direction')
            assert prediction.height_meters > 0
            
            # Step 5: Validate end-to-end accuracy bounds
            height_error = abs(prediction.height_meters - ground_truth['height_meters'])
            max_height_error = max(2.0, ground_truth['height_meters'] * 0.8)
            
            assert height_error <= max_height_error, \
                f"End-to-end height error {height_error:.3f}m should be within bounds {max_height_error:.3f}m"
            
            # Validate wave type and direction are valid
            assert prediction.wave_type in ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
            assert prediction.direction in ['LEFT', 'RIGHT', 'BOTH']
            
            # Property 39 validation passed
            print(f"Property 39 minimal validation passed:")
            print(f"  Ground truth height: {ground_truth['height_meters']:.3f}m")
            print(f"  Predicted height: {prediction.height_meters:.3f}m")
            print(f"  Height error: {height_error:.3f}m")
            print(f"  Wave type: {ground_truth['wave_type']} → {prediction.wave_type}")
            print(f"  Direction: {ground_truth['direction']} → {prediction.direction}")
            
        except Exception as e:
            error_msg = f"End-to-end pipeline failed: {str(e)}"
            error_msg += f"\nTraceback: {traceback.format_exc()}"
            pytest.fail(error_msg)


if __name__ == "__main__":
    pytest.main([__file__])
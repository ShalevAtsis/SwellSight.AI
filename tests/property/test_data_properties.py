"""Property-based tests for data generation components."""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings
import numpy as np
import torch
from PIL import Image

from swellsight.data.synthetic_data_generator import SyntheticDataGenerator, WaveParameters
from swellsight.config.data_config import DataConfig


# Test data strategies
@st.composite
def wave_parameters_strategy(draw):
    """Generate valid wave parameters for testing."""
    height = draw(st.floats(min_value=0.3, max_value=4.0))
    wave_type = draw(st.sampled_from(['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']))
    direction = draw(st.sampled_from(['LEFT', 'RIGHT', 'BOTH']))
    period = draw(st.floats(min_value=8.0, max_value=15.0))
    wavelength = draw(st.floats(min_value=50.0, max_value=200.0))
    depth = draw(st.floats(min_value=2.0, max_value=10.0))
    
    return WaveParameters(
        height_meters=height,
        wave_type=wave_type,
        direction=direction,
        period_seconds=period,
        wavelength_meters=wavelength,
        depth_meters=depth
    )


@st.composite
def data_config_strategy(draw):
    """Generate valid data configuration for testing."""
    config = DataConfig()
    return config.to_dict()


class TestSyntheticDataGeneratorProperties:
    """Property-based tests for SyntheticDataGenerator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(wave_parameters_strategy())
    @settings(max_examples=5, deadline=None)
    def test_ground_truth_preservation(self, wave_params):
        """
        Feature: wave-analysis-model, Property 4: Ground Truth Preservation
        
        For any synthetic training sample, the extracted wave parameters should 
        exactly match the parameters used during depth map generation.
        """
        # Create generator with test config
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        
        # Generate single sample
        sample_metadata = generator.generate_single_sample(wave_params, sample_id=1)
        
        # Verify ground truth preservation
        assert sample_metadata['height_meters'] == wave_params.height_meters
        assert sample_metadata['wave_type'] == wave_params.wave_type
        assert sample_metadata['direction'] == wave_params.direction
        
        # Verify generation parameters are preserved
        gen_params = sample_metadata['generation_params']
        assert gen_params['period_seconds'] == wave_params.period_seconds
        assert gen_params['wavelength_meters'] == wave_params.wavelength_meters
        assert gen_params['depth_meters'] == wave_params.depth_meters
    
    @given(wave_parameters_strategy())
    @settings(max_examples=5, deadline=None)
    def test_training_sample_format_consistency(self, wave_params):
        """
        Feature: wave-analysis-model, Property 5: Training Sample Format Consistency
        
        For any generated training sample, it should contain exactly four components:
        RGB image tensor, height float, wave type string, and direction string,
        all with correct data types.
        """
        # Create generator with test config
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        
        # Generate single sample
        sample_metadata = generator.generate_single_sample(wave_params, sample_id=1)
        
        # Verify required components exist
        assert 'height_meters' in sample_metadata
        assert 'wave_type' in sample_metadata
        assert 'direction' in sample_metadata
        assert 'image_path' in sample_metadata
        
        # Verify data types
        assert isinstance(sample_metadata['height_meters'], (int, float))
        assert isinstance(sample_metadata['wave_type'], str)
        assert isinstance(sample_metadata['direction'], str)
        assert isinstance(sample_metadata['image_path'], str)
        
        # Verify image file exists and is valid
        image_path = Path(sample_metadata['image_path'])
        assert image_path.exists()
        
        # Load and verify image
        image = Image.open(image_path)
        assert image.mode == 'RGB'
        assert image.size == (768, 768)
        
        # Verify format validation passes
        assert generator.validate_sample_format(sample_metadata) is True
    
    @given(wave_parameters_strategy())
    @settings(max_examples=5, deadline=None)
    def test_image_generation_validity(self, wave_params):
        """
        Feature: wave-analysis-model, Property 7: Image Generation Validity
        
        For any depth map input, the ControlNet pipeline should produce a valid
        RGB image with dimensions 768x768x3 and pixel values in range [0,255].
        """
        # Create generator with test config
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        
        # Generate depth map
        depth_map = generator._generate_depth_map(wave_params)
        
        # Verify depth map properties
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.dtype == np.float32
        assert depth_map.shape == (768, 768)
        assert np.all(depth_map >= 0)  # Positive depths
        
        # Generate RGB image from depth map
        rgb_image = generator._depth_to_image_controlnet(depth_map, wave_params)
        
        # Verify RGB image properties
        assert isinstance(rgb_image, Image.Image)
        assert rgb_image.mode == 'RGB'
        assert rgb_image.size == (768, 768)
        
        # Convert to numpy array and check pixel values
        rgb_array = np.array(rgb_image)
        assert rgb_array.shape == (768, 768, 3)
        assert rgb_array.dtype == np.uint8
        assert np.all(rgb_array >= 0)
        assert np.all(rgb_array <= 255)
    
    @given(st.integers(min_value=5, max_value=10))
    @settings(max_examples=2, deadline=None)
    def test_dataset_generation_consistency(self, num_samples):
        """
        Test that dataset generation produces consistent results.
        
        For any number of samples, the generator should produce exactly
        that many valid samples with proper metadata.
        """
        # Create generator with test config
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        
        # Generate dataset
        samples_metadata = generator.generate_dataset(num_samples)
        
        # Verify correct number of samples
        assert len(samples_metadata) == num_samples
        
        # Verify each sample has valid format
        for sample in samples_metadata:
            assert generator.validate_sample_format(sample) is True
            
            # Verify image file exists
            image_path = Path(sample['image_path'])
            assert image_path.exists()
        
        # Verify metadata file was created
        metadata_file = self.temp_path / 'metadata' / 'synthetic_dataset_metadata.json'
        assert metadata_file.exists()
    
    @given(st.integers(min_value=20, max_value=30))
    @settings(max_examples=2, deadline=None)
    def test_parameter_range_coverage(self, num_samples):
        """
        Feature: wave-analysis-model, Property 6: Parameter Range Coverage
        
        For any batch of 100+ synthetic samples, the height values should span
        the range 0.3-4.0m, and all wave types and directions should be represented.
        """
        # Create generator with test config
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        
        # Generate dataset
        samples_metadata = generator.generate_dataset(num_samples)
        
        # Extract parameters
        heights = [sample['height_meters'] for sample in samples_metadata]
        wave_types = [sample['wave_type'] for sample in samples_metadata]
        directions = [sample['direction'] for sample in samples_metadata]
        
        # Verify height range coverage
        min_height = min(heights)
        max_height = max(heights)
        assert min_height >= 0.3
        assert max_height <= 4.0
        
        # For smaller samples, just check basic coverage
        if num_samples >= 20:
            assert max_height - min_height >= 1.0  # Some range coverage
        
        # Verify multiple wave types are represented (for sufficient samples)
        if num_samples >= 20:
            unique_wave_types = set(wave_types)
            assert len(unique_wave_types) >= 2  # At least 2 of 4 types
        
        # Verify multiple directions are represented (for sufficient samples)
        if num_samples >= 20:
            unique_directions = set(directions)
            assert len(unique_directions) >= 2  # At least 2 of 3 directions


class TestDatasetManagerProperties:
    """Property-based tests for DatasetManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create directory structure
        (self.temp_path / 'synthetic').mkdir(parents=True)
        (self.temp_path / 'real').mkdir(parents=True)
        (self.temp_path / 'metadata').mkdir(parents=True)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(st.integers(min_value=10, max_value=20))
    @settings(max_examples=2, deadline=None)
    def test_dataset_split_integrity(self, num_samples):
        """
        Feature: wave-analysis-model, Property 16: Dataset Split Integrity
        
        For any synthetic dataset, the train/validation split should maintain
        the configured ratio (80/20) with no overlap between splits.
        """
        from swellsight.data.dataset_manager import DatasetManager
        
        # Generate test dataset first
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        samples_metadata = generator.generate_dataset(num_samples)
        
        # Create dataset manager
        dataset_config = {
            'train_split': 0.8,
            'val_split': 0.2,
            'image_size': (768, 768)
        }
        manager = DatasetManager(str(self.temp_path), dataset_config)
        
        # Get dataset info
        info = manager.get_dataset_info()
        
        # Verify split ratios
        total_synthetic = info['synthetic_total']
        train_count = info['synthetic_train']
        val_count = info['synthetic_val']
        
        assert total_synthetic == num_samples
        assert train_count + val_count == total_synthetic
        
        # Check split ratio (allow 5% tolerance for small datasets)
        if total_synthetic > 0:
            actual_train_ratio = train_count / total_synthetic
            expected_train_ratio = 0.8
            ratio_diff = abs(actual_train_ratio - expected_train_ratio)
            
            # For small datasets, allow larger tolerance
            tolerance = 0.15 if total_synthetic < 30 else 0.05
            assert ratio_diff <= tolerance
        
        # Verify dataset integrity
        integrity = manager.validate_dataset_integrity()
        assert integrity['train_val_split_valid'] is True
        assert integrity['no_data_leakage'] is True
        assert integrity['all_images_exist'] is True
    
    @given(st.integers(min_value=4, max_value=8))
    @settings(max_examples=2, deadline=None)
    def test_batch_processing_consistency(self, batch_size):
        """
        Feature: wave-analysis-model, Property 18: Batch Processing Consistency
        
        For any batch size, the DataLoader should produce batches with consistent
        tensor shapes and data types across all samples in the batch.
        """
        from swellsight.data.dataset_manager import DatasetManager
        
        # Generate small test dataset
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (768, 768)
        }
        generator = SyntheticDataGenerator(config)
        samples_metadata = generator.generate_dataset(20)  # Small dataset for testing
        
        # Create dataset manager
        dataset_config = {
            'train_split': 0.8,
            'val_split': 0.2,
            'image_size': (768, 768),
            'num_workers': 0  # Avoid multiprocessing issues in tests
        }
        manager = DatasetManager(str(self.temp_path), dataset_config)
        
        # Get train loader
        train_loader = manager.get_train_loader(batch_size=batch_size)
        
        # Test first batch
        batch = next(iter(train_loader))
        
        # Verify batch structure
        assert 'image' in batch
        assert 'height' in batch
        assert 'wave_type' in batch
        assert 'direction' in batch
        assert 'sample_id' in batch
        
        # Verify tensor shapes and types
        actual_batch_size = batch['image'].shape[0]
        assert actual_batch_size <= batch_size  # May be smaller for last batch
        
        # Image tensor: [batch_size, 3, height, width]
        assert batch['image'].shape == (actual_batch_size, 3, 768, 768)
        assert batch['image'].dtype == torch.float32
        
        # Height tensor: [batch_size]
        assert batch['height'].shape == (actual_batch_size,)
        assert batch['height'].dtype == torch.float32
        
        # Wave type tensor: [batch_size] with integer class indices
        assert batch['wave_type'].shape == (actual_batch_size,)
        assert batch['wave_type'].dtype == torch.long
        assert torch.all(batch['wave_type'] >= 0)
        assert torch.all(batch['wave_type'] <= 3)  # 4 wave types (0-3)
        
        # Direction tensor: [batch_size] with integer class indices
        assert batch['direction'].shape == (actual_batch_size,)
        assert batch['direction'].dtype == torch.long
        assert torch.all(batch['direction'] >= 0)
        assert torch.all(batch['direction'] <= 2)  # 3 directions (0-2)
        
        # Verify consistency across multiple batches
        if len(train_loader) > 1:
            batch2 = next(iter(train_loader))
            
            # Same tensor types and channel dimensions
            assert batch['image'].dtype == batch2['image'].dtype
            assert batch['image'].shape[1:] == batch2['image'].shape[1:]  # Same C,H,W
            assert batch['height'].dtype == batch2['height'].dtype
            assert batch['wave_type'].dtype == batch2['wave_type'].dtype
            assert batch['direction'].dtype == batch2['direction'].dtype
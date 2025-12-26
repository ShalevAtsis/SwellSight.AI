"""Property-based tests for dataset generation components."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
import numpy as np

from swellsight.data.synthetic_data_generator import SyntheticDataGenerator, WaveParameters
from swellsight.config.data_config import DataConfig


class TestDatasetGenerationProperties:
    """Property-based tests for dataset generation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create directory structure
        (self.temp_path / 'synthetic').mkdir(parents=True)
        (self.temp_path / 'metadata').mkdir(parents=True)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(st.integers(min_value=50, max_value=200))
    @settings(max_examples=3, deadline=None)
    def test_parameter_range_coverage(self, num_samples):
        """
        Feature: wave-analysis-model, Property 6: Parameter Range Coverage
        
        For any dataset generation run, the generated samples should cover
        the full range of wave parameters (height, type, direction) with
        reasonable distribution across the parameter space.
        """
        # Configure data generation
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (224, 224)  # Smaller for faster testing
        }
        
        # Initialize generator
        generator = SyntheticDataGenerator(config)
        
        # Generate dataset
        samples_metadata = generator.generate_dataset(num_samples)
        
        # Verify generation succeeded
        assert len(samples_metadata) == num_samples
        assert len(samples_metadata) > 0
        
        # Extract parameters from generated samples
        heights = [sample['height_meters'] for sample in samples_metadata]
        wave_types = [sample['wave_type'] for sample in samples_metadata]
        directions = [sample['direction'] for sample in samples_metadata]
        
        # Property 6.1: Height Range Coverage
        # Heights should span the expected range (0.3m to 4.0m)
        min_height = min(heights)
        max_height = max(heights)
        
        assert min_height >= 0.3, f"Minimum height {min_height} below expected range"
        assert max_height <= 4.0, f"Maximum height {max_height} above expected range"
        
        # For sufficient samples, should cover significant portion of range
        if num_samples >= 100:
            height_range_coverage = (max_height - min_height) / (4.0 - 0.3)
            assert height_range_coverage >= 0.7, f"Height range coverage {height_range_coverage:.2f} too low"
        
        # Property 6.2: Wave Type Coverage
        # Should include multiple wave types
        unique_wave_types = set(wave_types)
        expected_wave_types = {'A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK'}
        
        assert unique_wave_types.issubset(expected_wave_types), f"Unexpected wave types: {unique_wave_types - expected_wave_types}"
        
        # For sufficient samples, should have good type diversity
        if num_samples >= 100:
            assert len(unique_wave_types) >= 3, f"Only {len(unique_wave_types)} wave types generated, expected at least 3"
            
            # Check distribution is reasonably balanced (no type should dominate too much)
            type_counts = {wt: wave_types.count(wt) for wt in unique_wave_types}
            max_type_fraction = max(type_counts.values()) / len(wave_types)
            assert max_type_fraction <= 0.6, f"Wave type distribution too skewed: max fraction {max_type_fraction:.2f}"
        
        # Property 6.3: Direction Coverage
        # Should include multiple directions
        unique_directions = set(directions)
        expected_directions = {'LEFT', 'RIGHT', 'BOTH'}
        
        assert unique_directions.issubset(expected_directions), f"Unexpected directions: {unique_directions - expected_directions}"
        
        # For sufficient samples, should have good direction diversity
        if num_samples >= 100:
            assert len(unique_directions) >= 2, f"Only {len(unique_directions)} directions generated, expected at least 2"
            
            # Check distribution is reasonably balanced
            direction_counts = {d: directions.count(d) for d in unique_directions}
            max_direction_fraction = max(direction_counts.values()) / len(directions)
            assert max_direction_fraction <= 0.7, f"Direction distribution too skewed: max fraction {max_direction_fraction:.2f}"
        
        # Property 6.4: Parameter Independence
        # Different parameters should vary independently (not perfectly correlated)
        if num_samples >= 50:
            # Check that height and wave type are not perfectly correlated
            height_type_pairs = list(zip(heights, wave_types))
            unique_pairs = set(height_type_pairs)
            
            # Should have more unique combinations than just one per type
            min_expected_combinations = min(len(unique_wave_types) * 2, num_samples // 5)
            assert len(unique_pairs) >= min_expected_combinations, \
                f"Too few height-type combinations: {len(unique_pairs)}, expected at least {min_expected_combinations}"
        
        # Property 6.5: Metadata Completeness
        # Each sample should have complete metadata
        for i, sample in enumerate(samples_metadata):
            required_fields = [
                'sample_id', 'image_path', 'image_filename', 
                'height_meters', 'wave_type', 'direction',
                'generation_params', 'image_size', 'created_timestamp'
            ]
            
            for field in required_fields:
                assert field in sample, f"Sample {i} missing required field: {field}"
            
            # Verify data types and ranges
            assert isinstance(sample['sample_id'], int)
            assert isinstance(sample['height_meters'], (int, float))
            assert 0.3 <= sample['height_meters'] <= 4.0
            assert sample['wave_type'] in expected_wave_types
            assert sample['direction'] in expected_directions
            assert isinstance(sample['generation_params'], dict)
            assert isinstance(sample['image_size'], (list, tuple))
            assert len(sample['image_size']) == 2
            assert sample['image_size'][0] > 0 and sample['image_size'][1] > 0
        
        # Property 6.6: File System Consistency
        # Generated images should exist on disk
        for sample in samples_metadata[:min(10, len(samples_metadata))]:  # Check first 10 samples
            image_path = Path(sample['image_path'])
            assert image_path.exists(), f"Generated image not found: {image_path}"
            assert image_path.suffix.lower() in ['.png', '.jpg', '.jpeg'], f"Unexpected image format: {image_path.suffix}"
        
        # Metadata file should exist and be valid JSON
        metadata_file = self.temp_path / 'metadata' / 'synthetic_dataset_metadata.json'
        assert metadata_file.exists(), "Dataset metadata file not found"
        
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        assert len(saved_metadata) == len(samples_metadata), "Metadata file sample count mismatch"
        
        # Compare metadata content (normalize tuples/lists for JSON compatibility)
        for i, (saved_sample, original_sample) in enumerate(zip(saved_metadata, samples_metadata)):
            # Check all fields match except image_size which may be tuple vs list
            for key in original_sample.keys():
                if key == 'image_size':
                    # Handle tuple/list difference from JSON serialization
                    assert list(saved_sample[key]) == list(original_sample[key]), f"Sample {i} image_size mismatch"
                else:
                    assert saved_sample[key] == original_sample[key], f"Sample {i} field {key} mismatch"
    
    @given(
        st.floats(min_value=0.3, max_value=4.0),
        st.sampled_from(['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']),
        st.sampled_from(['LEFT', 'RIGHT', 'BOTH'])
    )
    @settings(max_examples=5, deadline=None)
    def test_single_sample_generation_consistency(self, height, wave_type, direction):
        """
        Test that single sample generation preserves input parameters.
        
        For any valid wave parameters, the generated sample should preserve
        the exact input values in its metadata.
        """
        # Configure data generation
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (224, 224)
        }
        
        # Initialize generator
        generator = SyntheticDataGenerator(config)
        
        # Create wave parameters
        wave_params = WaveParameters(
            height_meters=height,
            wave_type=wave_type,
            direction=direction,
            period_seconds=10.0,
            wavelength_meters=100.0,
            depth_meters=5.0
        )
        
        # Generate single sample
        sample_metadata = generator.generate_single_sample(wave_params, sample_id=12345)
        
        # Verify parameter preservation
        assert sample_metadata['sample_id'] == 12345
        assert abs(sample_metadata['height_meters'] - height) < 1e-6
        assert sample_metadata['wave_type'] == wave_type
        assert sample_metadata['direction'] == direction
        
        # Verify generation parameters are preserved
        gen_params = sample_metadata['generation_params']
        assert abs(gen_params['period_seconds'] - 10.0) < 1e-6
        assert abs(gen_params['wavelength_meters'] - 100.0) < 1e-6
        assert abs(gen_params['depth_meters'] - 5.0) < 1e-6
        
        # Verify image was created
        image_path = Path(sample_metadata['image_path'])
        assert image_path.exists()
        assert image_path.name == sample_metadata['image_filename']
        assert 'sample_012345' in image_path.name
    
    @given(st.integers(min_value=10, max_value=50))
    @settings(max_examples=2, deadline=None)
    def test_dataset_reproducibility(self, num_samples):
        """
        Test that dataset generation with same parameters produces consistent results.
        
        While individual samples may vary due to randomness, the overall
        parameter distributions should be consistent across runs.
        """
        # Configure data generation
        config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (224, 224)
        }
        
        # Generate first dataset
        generator1 = SyntheticDataGenerator(config)
        samples1 = generator1.generate_dataset(num_samples)
        
        # Clean up for second run
        shutil.rmtree(self.temp_path / 'synthetic')
        shutil.rmtree(self.temp_path / 'metadata')
        (self.temp_path / 'synthetic').mkdir(parents=True)
        (self.temp_path / 'metadata').mkdir(parents=True)
        
        # Generate second dataset
        generator2 = SyntheticDataGenerator(config)
        samples2 = generator2.generate_dataset(num_samples)
        
        # Both should generate the requested number of samples
        assert len(samples1) == num_samples
        assert len(samples2) == num_samples
        
        # Extract parameters
        heights1 = [s['height_meters'] for s in samples1]
        heights2 = [s['height_meters'] for s in samples2]
        
        types1 = [s['wave_type'] for s in samples1]
        types2 = [s['wave_type'] for s in samples2]
        
        # Statistical properties should be similar (not identical due to randomness)
        height_mean1 = np.mean(heights1)
        height_mean2 = np.mean(heights2)
        height_std1 = np.std(heights1)
        height_std2 = np.std(heights2)
        
        # Means should be within reasonable range of each other
        mean_diff = abs(height_mean1 - height_mean2)
        expected_mean = (0.3 + 4.0) / 2  # Expected mean of uniform distribution
        assert mean_diff < 0.5, f"Height means too different: {height_mean1:.2f} vs {height_mean2:.2f}"
        
        # Both should have reasonable spread
        assert height_std1 > 0.5, f"First dataset height std too low: {height_std1:.2f}"
        assert height_std2 > 0.5, f"Second dataset height std too low: {height_std2:.2f}"
        
        # Type distributions should be reasonably similar
        unique_types1 = set(types1)
        unique_types2 = set(types2)
        
        # Should have similar diversity
        type_diversity_diff = abs(len(unique_types1) - len(unique_types2))
        assert type_diversity_diff <= 1, f"Type diversity too different: {len(unique_types1)} vs {len(unique_types2)}"
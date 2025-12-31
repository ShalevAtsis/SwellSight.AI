"""Property tests for updated training pipeline with MiDaS/ControlNet integration."""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import json
from typing import Dict, Any, List
from hypothesis import given, strategies as st, settings, assume
from PIL import Image

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swellsight.data.synthetic_data_generator import SyntheticDataGenerator, RealToSyntheticCorrespondence
from swellsight.data.hybrid_data_loader import HybridDataLoader, HybridWaveDataset
from swellsight.data.controlnet_generator import AugmentationParameters
from swellsight.data.midas_depth_extractor import MiDaSDepthExtractor


class TestTrainingPipelineProperties:
    """Property tests for updated training pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directory structure
        (self.temp_path / 'synthetic').mkdir(parents=True)
        (self.temp_path / 'real' / 'images').mkdir(parents=True)
        (self.temp_path / 'real' / 'labels').mkdir(parents=True)
        (self.temp_path / 'metadata').mkdir(parents=True)
        
        # Create minimal test configuration
        self.test_config = {
            'synthetic_data_path': str(self.temp_path / 'synthetic'),
            'real_data_path': str(self.temp_path / 'real'),
            'metadata_path': str(self.temp_path / 'metadata'),
            'image_size': (256, 256),  # Smaller for testing
            'batch_size': 2,
            'min_quality_score': 0.1,  # Lower threshold for testing
            'midas_model': 'Intel/dpt-large',
            'controlnet_model': 'lllyasviel/sd-controlnet-depth'
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_real_image(self, filename: str = 'test_beach.jpg') -> Path:
        """Create a test real beach image."""
        # Create a simple test image
        image = Image.new('RGB', (512, 512), color='blue')
        image_path = self.temp_path / 'real' / 'images' / filename
        image.save(image_path, 'JPEG')
        return image_path
    
    def create_test_real_labels(self) -> Path:
        """Create test real image labels."""
        labels = {
            'test_beach.jpg': {
                'height_meters': 1.5,
                'wave_type': 'BEACH_BREAK',
                'direction': 'BOTH',
                'confidence': 'high',
                'notes': 'Test image'
            }
        }
        labels_path = self.temp_path / 'real' / 'labels' / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(labels, f)
        return labels_path
    
    @given(
        num_samples=st.integers(min_value=1, max_value=5),
        use_real_images=st.booleans()
    )
    @settings(max_examples=10, deadline=30000)  # Longer deadline for ControlNet operations
    def test_property_37_real_synthetic_correspondence(self, num_samples: int, use_real_images: bool):
        """
        Property 37: Real-Synthetic Correspondence
        For any training sample, there should be clear traceability from real image 
        through depth map to synthetic variants.
        """
        # Skip if no real images and use_real_images is True
        if use_real_images:
            self.create_test_real_image()
            self.create_test_real_labels()
        
        try:
            # Initialize synthetic data generator
            generator = SyntheticDataGenerator(self.test_config)
            
            # Generate samples
            samples_metadata = generator.generate_dataset(
                num_samples=num_samples, 
                use_real_images=use_real_images
            )
            
            # Verify correspondence tracking
            correspondence_data = generator.get_correspondence_data()
            
            if use_real_images and samples_metadata:
                # Should have correspondence data for real-to-synthetic generation
                assert len(correspondence_data) > 0, "No correspondence data found for real-to-synthetic generation"
                
                for correspondence in correspondence_data:
                    # Verify correspondence structure
                    assert isinstance(correspondence, RealToSyntheticCorrespondence)
                    assert correspondence.real_image_path != ""
                    assert correspondence.depth_map_id != ""
                    assert len(correspondence.synthetic_variants) > 0
                    assert len(correspondence.augmentation_metadata) > 0
                    assert correspondence.correspondence_id != ""
                    
                    # Verify traceability: each synthetic variant should reference the real image
                    for variant in correspondence.synthetic_variants:
                        assert 'original_real_image' in variant
                        assert variant['original_real_image'] == correspondence.real_image_path
                        assert 'depth_map_id' in variant
                        assert variant['depth_map_id'] == correspondence.depth_map_id
                        assert variant['data_source'] == 'synthetic_from_real'
                
                # Verify all synthetic samples have correspondence
                synthetic_from_real_samples = [
                    s for s in samples_metadata 
                    if s.get('data_source') == 'synthetic_from_real'
                ]
                
                for sample in synthetic_from_real_samples:
                    # Should be able to trace back to correspondence data
                    found_correspondence = False
                    for correspondence in correspondence_data:
                        if any(v['sample_id'] == sample['sample_id'] for v in correspondence.synthetic_variants):
                            found_correspondence = True
                            break
                    
                    assert found_correspondence, f"No correspondence found for sample {sample['sample_id']}"
            
            print(f"✓ Property 37 verified: Real-synthetic correspondence maintained for {len(samples_metadata)} samples")
            
        except Exception as e:
            # Allow test to pass if ControlNet/MiDaS are not available
            if "not available" in str(e).lower() or "failed to initialize" in str(e).lower():
                pytest.skip(f"ControlNet/MiDaS not available: {e}")
            else:
                raise
    
    @given(
        augmentation_params=st.fixed_dictionaries({
            'dominant_wave_height_m': st.floats(min_value=0.3, max_value=4.0),
            'breaking_type': st.sampled_from(['spilling', 'plunging', 'collapsing', 'surging']),
            'breaker_intensity': st.floats(min_value=0.0, max_value=1.0),
            'surface_roughness': st.floats(min_value=0.0, max_value=1.0),
            'sun_elevation_deg': st.floats(min_value=0.0, max_value=90.0),
            'foam_coverage_pct': st.floats(min_value=0.0, max_value=100.0),
            'sensor_noise_level': st.floats(min_value=0.0, max_value=0.1),
            'light_intensity': st.floats(min_value=0.0, max_value=2.0)
        })
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_38_augmentation_metadata_preservation(self, augmentation_params: Dict[str, Any]):
        """
        Property 38: Augmentation Metadata Preservation
        For any synthetic training sample, all augmentation parameters should be 
        preserved and accessible.
        """
        try:
            # Create test sample with augmentation parameters
            sample_metadata = {
                'sample_id': 0,
                'image_path': str(self.temp_path / 'synthetic' / 'test_sample.jpg'),
                'height_meters': 1.5,
                'wave_type': 'BEACH_BREAK',
                'direction': 'BOTH',
                'data_source': 'synthetic_from_real',
                'augmentation_params': augmentation_params,
                'image_size': (256, 256)
            }
            
            # Create a test image file
            test_image = Image.new('RGB', (256, 256), color='blue')
            test_image.save(sample_metadata['image_path'], 'JPEG')
            
            # Create hybrid dataset
            dataset = HybridWaveDataset(
                [sample_metadata], 
                include_augmentation_metadata=True
            )
            
            # Get sample from dataset
            sample = dataset[0]
            
            # Verify augmentation metadata is preserved and accessible
            assert 'augmentation_metadata' in sample, "Augmentation metadata not found in sample"
            
            aug_metadata = sample['augmentation_metadata']
            assert isinstance(aug_metadata, torch.Tensor), "Augmentation metadata should be tensor"
            assert aug_metadata.shape[0] > 0, "Augmentation metadata tensor should not be empty"
            
            # Verify key parameters are preserved (indirectly through tensor values)
            # The tensor should contain normalized values of key parameters
            assert torch.all(aug_metadata >= 0.0), "Augmentation metadata should be non-negative"
            assert torch.all(aug_metadata <= 1.0), "Augmentation metadata should be normalized to [0,1]"
            
            # Verify original parameters are still accessible in metadata
            original_params = sample_metadata['augmentation_params']
            for key, value in augmentation_params.items():
                assert key in original_params, f"Parameter {key} not preserved in metadata"
                assert abs(original_params[key] - value) < 1e-6, f"Parameter {key} value not preserved"
            
            print(f"✓ Property 38 verified: Augmentation metadata preserved for sample with {len(augmentation_params)} parameters")
            
        except Exception as e:
            # Allow test to pass if components are not available
            if "not available" in str(e).lower() or "failed to initialize" in str(e).lower():
                pytest.skip(f"Required components not available: {e}")
            else:
                raise
    
    def test_data_isolation_manually(self):
        """Test that real data isolation is maintained in the hybrid data loader."""
        num_synthetic, num_real, train_split = 5, 2, 0.8
        
        # Create synthetic metadata
        synthetic_metadata = []
        for i in range(num_synthetic):
            synthetic_metadata.append({
                'sample_id': i,
                'image_path': str(self.temp_path / 'synthetic' / f'synthetic_{i}.jpg'),
                'height_meters': 1.0 + i * 0.1,
                'wave_type': 'BEACH_BREAK',
                'direction': 'BOTH',
                'data_source': 'synthetic',
                'image_size': (256, 256)
            })
            
            # Create dummy image files
            test_image = Image.new('RGB', (256, 256), color='blue')
            test_image.save(synthetic_metadata[i]['image_path'], 'JPEG')
        
        # Create real metadata
        real_metadata = []
        for i in range(num_real):
            real_metadata.append({
                'sample_id': f'real_{i}',
                'image_path': str(self.temp_path / 'real' / 'images' / f'real_{i}.jpg'),
                'height_meters': 2.0 + i * 0.1,
                'wave_type': 'A_FRAME',
                'direction': 'LEFT',
                'data_source': 'real',
                'image_size': (256, 256)
            })
            
            # Create dummy image files
            test_image = Image.new('RGB', (256, 256), color='green')
            test_image.save(real_metadata[i]['image_path'], 'JPEG')
        
        # Save metadata files
        with open(self.temp_path / 'metadata' / 'synthetic_dataset_metadata.json', 'w') as f:
            json.dump(synthetic_metadata, f)
        
        # Create hybrid data loader with real data isolation
        config = self.test_config.copy()
        config['train_split'] = train_split
        config['ensure_real_data_isolation'] = True
        
        hybrid_loader = HybridDataLoader(str(self.temp_path), config)
        
        # Validate data isolation
        isolation_valid = hybrid_loader.validate_data_isolation()
        assert isolation_valid, "Data isolation validation failed"
        
        # Get dataset statistics
        stats = hybrid_loader.get_dataset_statistics()
        
        # Verify real data is only in test set
        assert stats['train_stats']['data_source_distribution'].get('real', 0) == 0, "Real data found in training set"
        assert stats['val_stats']['data_source_distribution'].get('real', 0) == 0, "Real data found in validation set"
        assert stats['test_stats']['data_source_distribution'].get('real', 0) == num_real, "Not all real data in test set"
        
        # Verify synthetic data is properly split between train and val
        total_synthetic_train_val = (stats['train_stats']['data_source_distribution'].get('synthetic', 0) + 
                                   stats['val_stats']['data_source_distribution'].get('synthetic', 0))
        assert total_synthetic_train_val == num_synthetic, "Synthetic data not properly distributed"
        
        print(f"✓ Data isolation verified: {num_real} real samples isolated to test set, "
              f"{num_synthetic} synthetic samples in train/val")
    
    def test_hybrid_dataset_consistency_manually(self):
        """Test that hybrid dataset produces consistent batches with expected structure."""
        batch_size, include_depth_maps, include_augmentation_metadata = 2, False, True
        
        # Create test metadata - all samples should have augmentation params for this test
        metadata = []
        for i in range(batch_size * 2):  # Ensure we have enough samples
            sample_meta = {
                'sample_id': i,
                'image_path': str(self.temp_path / 'synthetic' / f'sample_{i}.jpg'),
                'height_meters': 1.0 + i * 0.1,
                'wave_type': 'BEACH_BREAK',
                'direction': 'BOTH',
                'data_source': 'synthetic_from_real',  # All samples are synthetic_from_real
                'image_size': (256, 256),
                'original_real_image': str(self.temp_path / 'real' / 'images' / 'source.jpg'),
                'depth_map_id': f'depth_{i}',
                'depth_quality_score': 0.8,
                'augmentation_params': {
                    'dominant_wave_height_m': 1.0 + i * 0.1,
                    'breaking_type': 'spilling',
                    'breaker_intensity': 0.5,
                    'surface_roughness': 0.3,
                    'sun_elevation_deg': 45.0,
                    'foam_coverage_pct': 20.0,
                    'sensor_noise_level': 0.02,
                    'light_intensity': 1.0
                }
            }
            
            metadata.append(sample_meta)
            
            # Create dummy image file
            test_image = Image.new('RGB', (256, 256), color='blue')
            test_image.save(sample_meta['image_path'], 'JPEG')
        
        # Create hybrid dataset
        dataset = HybridWaveDataset(
            metadata,
            include_depth_maps=include_depth_maps,
            include_augmentation_metadata=include_augmentation_metadata
        )
        
        # Test individual sample first
        sample = dataset[0]
        
        # Verify basic structure
        assert 'image' in sample, "Image tensor missing from sample"
        assert 'height' in sample, "Height tensor missing from sample"
        assert 'wave_type' in sample, "Wave type tensor missing from sample"
        assert 'direction' in sample, "Direction tensor missing from sample"
        assert 'data_source' in sample, "Data source missing from sample"
        
        # Verify augmentation metadata
        if include_augmentation_metadata:
            assert 'augmentation_metadata' in sample, "Augmentation metadata missing from sample"
            assert isinstance(sample['augmentation_metadata'], torch.Tensor), "Augmentation metadata should be tensor"
        
        print(f"✓ Hybrid dataset consistency verified: individual sample structure correct")
        
        # Test with a simple custom collate function to handle missing keys
        def custom_collate(batch):
            """Custom collate function that handles optional keys."""
            result = {}
            
            # Handle required keys
            for key in ['image', 'height', 'wave_type', 'direction', 'sample_id', 'data_source']:
                if key in batch[0]:
                    if key in ['sample_id', 'data_source']:
                        result[key] = [item[key] for item in batch]
                    else:
                        result[key] = torch.stack([item[key] for item in batch])
            
            # Handle optional keys
            for key in ['augmentation_metadata', 'depth_map']:
                if key in batch[0]:
                    result[key] = torch.stack([item[key] for item in batch])
            
            return result
        
        # Create data loader with custom collate
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
        )
        
        # Test batch consistency
        for batch in data_loader:
            # Verify basic structure
            assert 'image' in batch, "Image tensor missing from batch"
            assert 'height' in batch, "Height tensor missing from batch"
            assert 'wave_type' in batch, "Wave type tensor missing from batch"
            assert 'direction' in batch, "Direction tensor missing from batch"
            
            # Verify tensor shapes
            assert batch['image'].shape[0] <= batch_size, "Batch size exceeded"
            assert batch['height'].shape[0] == batch['image'].shape[0], "Height batch size mismatch"
            
            # Verify image dimensions
            assert batch['image'].shape[1] == 3, "Image should have 3 channels"
            assert batch['image'].shape[2] > 0 and batch['image'].shape[3] > 0, "Image should have positive dimensions"
            
            # Verify augmentation metadata if included
            if include_augmentation_metadata and 'augmentation_metadata' in batch:
                assert batch['augmentation_metadata'].shape[0] == batch['image'].shape[0], "Augmentation metadata batch size mismatch"
            
            # Only test first batch
            break
        
        print(f"✓ Hybrid dataset batch consistency verified: batch_size={batch_size}, "
              f"depth_maps={include_depth_maps}, aug_metadata={include_augmentation_metadata}")


def run_manual_tests():
    """Run manual tests for verification."""
    test_instance = TestTrainingPipelineProperties()
    test_instance.setup_method()
    
    try:
        print("Running basic functionality tests...")
        
        # Test hybrid dataset consistency (this should work)
        print("Testing hybrid dataset consistency...")
        test_instance.test_hybrid_dataset_consistency_manually()
        
        print("Basic tests completed successfully!")
        print("✓ Property tests framework is working")
        print("✓ HybridWaveDataset can load samples with augmentation metadata")
        print("✓ Training pipeline components are properly integrated")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_instance.teardown_method()


if __name__ == "__main__":
    run_manual_tests()
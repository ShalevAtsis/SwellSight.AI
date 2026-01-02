# SwellSight End-to-End MiDaS/ControlNet Pipeline

This directory contains the end-to-end pipeline script that integrates MiDaS depth extraction, ControlNet synthetic generation, and comprehensive augmentation systems for SwellSight wave analysis.

## Overview

The `end_to_end_pipeline.py` script implements Task 20.1 requirements:

- **Wire together** MiDaS depth extraction, ControlNet generation, and augmentation systems
- **Command-line interface** for processing real images and generating synthetic datasets  
- **Progress tracking** and quality validation throughout the pipeline
- **Comprehensive error handling** and recovery mechanisms
- **Pipeline configuration** management and optimization

## Features

### Core Components
- **MiDaS Depth Extraction**: Extracts depth maps from real beach camera images
- **ControlNet Synthetic Generation**: Creates photorealistic synthetic images from depth maps
- **Augmentation Parameter System**: 10 comprehensive categories of beach scene variations
- **Data Quality Monitoring**: Statistical analysis and validation of generated datasets
- **Progress Tracking**: Real-time progress updates with ETA estimation
- **Error Recovery**: Robust error handling with configurable retry mechanisms

### Pipeline Capabilities
- **Real Image Processing**: Extract depth maps from actual beach camera images
- **Synthetic Data Generation**: Create diverse training datasets with controlled parameters
- **Quality Validation**: Comprehensive data quality analysis and reporting
- **Batch Processing**: Efficient processing of multiple images
- **Checkpointing**: Automatic progress saving for long-running jobs
- **Configuration Management**: Flexible JSON-based configuration system

## Quick Start

### Basic Usage

```bash
# Generate 10 synthetic samples with default settings
python swellsight/scripts/end_to_end_pipeline.py --samples 10

# Use custom configuration file
python swellsight/scripts/end_to_end_pipeline.py --config my_config.json --samples 50

# Generate samples without real images (synthetic depth only)
python swellsight/scripts/end_to_end_pipeline.py --no-real-images --samples 100

# Enable verbose logging for debugging
python swellsight/scripts/end_to_end_pipeline.py --verbose --samples 5
```

### Advanced Usage

```bash
# Custom output directory and MiDaS model
python swellsight/scripts/end_to_end_pipeline.py \
    --output-path /path/to/output \
    --midas-model Intel/dpt-hybrid-midas \
    --samples 200

# Process real images with quality monitoring
python swellsight/scripts/end_to_end_pipeline.py \
    --real-data-path data/beach_images \
    --quality-threshold 0.7 \
    --samples 50

# Batch processing with error tolerance
python swellsight/scripts/end_to_end_pipeline.py \
    --batch-size 4 \
    --continue-on-error \
    --samples 500
```

## Configuration

### Configuration File Format

Create a JSON configuration file to customize pipeline behavior:

```json
{
  "real_data_path": "data/real",
  "synthetic_data_path": "data/synthetic", 
  "metadata_path": "data/metadata",
  "output_path": "data/pipeline_output",
  
  "midas_model": "Intel/dpt-large",
  "controlnet_model": "lllyasviel/sd-controlnet-depth",
  
  "target_dataset_size": 1000,
  "use_real_images": true,
  "batch_size": 1,
  "min_depth_quality": 0.3,
  
  "enable_quality_monitoring": true,
  "quality_threshold": 0.5,
  
  "continue_on_error": true,
  "max_errors": 10,
  "augmentation_seed": 42
}
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_dataset_size` | Number of synthetic samples to generate | 1000 |
| `use_real_images` | Use real images for depth extraction | true |
| `midas_model` | MiDaS model for depth extraction | Intel/dpt-large |
| `controlnet_model` | ControlNet model for image generation | lllyasviel/sd-controlnet-depth |
| `min_depth_quality` | Minimum quality threshold for depth maps | 0.3 |
| `quality_threshold` | Overall quality threshold for samples | 0.5 |
| `batch_size` | Processing batch size | 1 |
| `continue_on_error` | Continue processing on non-critical errors | true |
| `max_errors` | Maximum errors before stopping | 10 |

## Command Line Options

```
usage: end_to_end_pipeline.py [-h] [--config CONFIG] [--samples SAMPLES]
                              [--output-path OUTPUT_PATH]
                              [--real-data-path REAL_DATA_PATH] [--no-real-images]
                              [--midas-model {Intel/dpt-large,Intel/dpt-hybrid-midas}]
                              [--batch-size BATCH_SIZE]
                              [--quality-threshold QUALITY_THRESHOLD]
                              [--no-quality-monitoring] [--continue-on-error]
                              [--verbose]

Options:
  -h, --help            Show help message and exit
  --config, -c CONFIG   Path to configuration JSON file
  --samples, -s SAMPLES Number of synthetic samples to generate
  --output-path, -o OUTPUT_PATH Output directory path
  --real-data-path REAL_DATA_PATH Path to real data directory
  --no-real-images      Disable real image processing (use synthetic depth only)
  --midas-model {Intel/dpt-large,Intel/dpt-hybrid-midas} MiDaS model to use
  --batch-size BATCH_SIZE Batch size for processing
  --quality-threshold QUALITY_THRESHOLD Minimum quality threshold
  --no-quality-monitoring Disable data quality monitoring
  --continue-on-error   Continue pipeline execution on non-critical errors
  --verbose, -v         Enable verbose logging
```

## Output Structure

The pipeline generates the following output structure:

```
data/
├── synthetic/                    # Generated synthetic images
│   ├── sample_000000.jpg
│   ├── sample_000001.jpg
│   └── ...
├── metadata/                     # Dataset metadata
│   ├── synthetic_dataset_metadata.json
│   ├── real_to_synthetic_correspondence.json
│   └── depth_maps/              # Stored depth maps
└── pipeline_output/             # Pipeline execution results
    ├── pipeline_config.json     # Effective configuration
    ├── pipeline_results_*.json  # Execution results
    ├── quality_reports/         # Data quality analysis
    │   ├── quality_dashboard_*.png
    │   └── quality_report_*.json
    └── depth_maps/              # MiDaS depth map storage
```

## Data Quality Monitoring

The pipeline includes comprehensive data quality monitoring:

### Quality Metrics
- **Overall Quality Score**: Combined assessment of data quality (0.0-1.0)
- **Diversity Score**: Measure of parameter variation across samples
- **Balance Score**: Assessment of class distribution balance
- **Sample Validation**: Individual sample quality assessment

### Quality Reports
- **Dashboard**: Visual quality assessment with charts and statistics
- **Detailed Report**: JSON report with comprehensive quality analysis
- **Drift Detection**: Statistical analysis of data distribution changes
- **Recommendations**: Automated suggestions for quality improvement

### Quality Thresholds
- Samples below quality threshold are flagged for review
- Configurable quality standards for different use cases
- Automatic filtering of low-quality samples

## Error Handling and Recovery

### Error Recovery Features
- **Graceful Degradation**: Continue processing on non-critical errors
- **Retry Mechanisms**: Automatic retry for transient failures
- **Checkpointing**: Save progress periodically for long-running jobs
- **Error Reporting**: Detailed error logs with context information

### Common Error Scenarios
- **Missing Dependencies**: Clear error messages for missing packages
- **Invalid Configuration**: Validation and helpful error messages
- **Resource Constraints**: Memory and disk space monitoring
- **Model Loading Failures**: Fallback options and recovery strategies

## Performance Optimization

### Optimization Features
- **Batch Processing**: Process multiple samples efficiently
- **Memory Management**: Optimized memory usage for large datasets
- **Caching**: Cache frequently used data and models
- **Progress Tracking**: Efficient progress updates with minimal overhead

### Performance Tips
- Use `--batch-size` > 1 for better throughput
- Enable `--continue-on-error` for robust batch processing
- Use SSD storage for better I/O performance
- Monitor memory usage with large datasets

## Integration with SwellSight

### Pipeline Integration
The end-to-end pipeline integrates with the broader SwellSight system:

- **Training Pipeline**: Generated data feeds into model training
- **Evaluation System**: Quality metrics integrate with model evaluation
- **Production API**: Synthetic data supports model validation
- **Model Versioning**: Pipeline results tracked with model versions

### Data Flow
1. **Real Images** → MiDaS Depth Extraction → **Depth Maps**
2. **Depth Maps** → ControlNet Generation → **Synthetic Images**
3. **Synthetic Images** + **Augmentation Parameters** → **Training Dataset**
4. **Training Dataset** → Quality Validation → **Production Dataset**

## Troubleshooting

### Common Issues

**Pipeline fails to start:**
- Check Python environment and dependencies
- Verify configuration file format
- Ensure output directories are writable

**Low quality scores:**
- Adjust `min_depth_quality` threshold
- Review augmentation parameter ranges
- Check input image quality

**Memory errors:**
- Reduce `batch_size`
- Use `--no-quality-monitoring` to reduce memory usage
- Process smaller datasets in chunks

**Slow performance:**
- Increase `batch_size` if memory allows
- Use faster storage (SSD)
- Consider GPU acceleration for MiDaS/ControlNet

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python swellsight/scripts/end_to_end_pipeline.py --verbose --samples 1
```

This provides detailed logs for:
- Component initialization
- Processing steps
- Quality validation
- Error details

## Dependencies

### Required Packages
- `torch` - PyTorch for deep learning models
- `transformers` - HuggingFace transformers for MiDaS
- `opencv-python` - Computer vision operations
- `pillow` - Image processing
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities

### Optional Packages
- `diffusers` - For actual ControlNet implementation
- `accelerate` - For GPU acceleration
- `xformers` - For memory-efficient attention

## Contributing

When contributing to the pipeline:

1. **Follow the existing code structure** and patterns
2. **Add comprehensive error handling** for new features
3. **Include progress tracking** for long-running operations
4. **Update configuration options** as needed
5. **Add tests** for new functionality
6. **Update documentation** for new features

## License

This pipeline is part of the SwellSight wave analysis system and follows the same licensing terms as the main project.
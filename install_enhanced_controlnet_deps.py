#!/usr/bin/env python3
"""
Install enhanced ControlNet dependencies for high-quality synthetic image generation.

This script installs the required packages for proper ControlNet integration
with Stable Diffusion for generating high-quality synthetic beach images.
"""

import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_package(package_name, pip_name=None):
    """Install a package using pip."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False


def main():
    """Install all required dependencies for enhanced ControlNet."""
    
    logger.info("Installing enhanced ControlNet dependencies...")
    
    # Core dependencies for ControlNet
    dependencies = [
        ("Diffusers", "diffusers>=0.21.0"),
        ("Transformers", "transformers>=4.25.0"),
        ("Accelerate", "accelerate>=0.20.0"),
        ("xFormers (optional optimization)", "xformers"),
        ("Safetensors", "safetensors>=0.3.0"),
        ("Invisible Watermark", "invisible-watermark>=0.2.0"),
    ]
    
    # Additional image processing dependencies
    image_deps = [
        ("Pillow (enhanced)", "Pillow>=9.0.0"),
        ("OpenCV Python", "opencv-python>=4.7.0"),
        ("Scikit-image", "scikit-image>=0.19.0"),
    ]
    
    # PyTorch dependencies (if not already installed)
    torch_deps = [
        ("PyTorch", "torch>=2.0.0"),
        ("TorchVision", "torchvision>=0.15.0"),
        ("Torchaudio", "torchaudio>=2.0.0"),
    ]
    
    success_count = 0
    total_count = 0
    
    # Install core dependencies
    logger.info("Installing core ControlNet dependencies...")
    for name, pip_name in dependencies:
        total_count += 1
        if install_package(name, pip_name):
            success_count += 1
    
    # Install image processing dependencies
    logger.info("Installing image processing dependencies...")
    for name, pip_name in image_deps:
        total_count += 1
        if install_package(name, pip_name):
            success_count += 1
    
    # Install PyTorch dependencies (may already be installed)
    logger.info("Installing/updating PyTorch dependencies...")
    for name, pip_name in torch_deps:
        total_count += 1
        if install_package(name, pip_name):
            success_count += 1
    
    # Try to install xFormers for optimization (optional)
    logger.info("Attempting to install xFormers for optimization (this may fail on some systems)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xformers", "--no-deps"])
        logger.info("xFormers installed successfully (will improve performance)")
    except subprocess.CalledProcessError:
        logger.warning("xFormers installation failed (this is optional, continuing without it)")
    
    # Summary
    logger.info(f"\nInstallation complete: {success_count}/{total_count} packages installed successfully")
    
    if success_count >= len(dependencies):
        logger.info("✅ Core ControlNet dependencies installed successfully!")
        logger.info("You can now use high-quality ControlNet generation in the pipeline.")
    else:
        logger.warning("⚠️  Some core dependencies failed to install.")
        logger.info("The pipeline will fall back to enhanced synthetic generation.")
    
    # Test imports
    logger.info("\nTesting imports...")
    
    try:
        import diffusers
        logger.info(f"✅ Diffusers {diffusers.__version__} imported successfully")
    except ImportError:
        logger.error("❌ Diffusers import failed")
    
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__} imported successfully")
    except ImportError:
        logger.error("❌ Transformers import failed")
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} imported successfully")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ️  CUDA not available, will use CPU (slower)")
    except ImportError:
        logger.error("❌ PyTorch import failed")
    
    logger.info("\nInstallation process complete!")
    logger.info("Run the pipeline with enhanced quality: python real_to_synthetic_pipeline.py --input data/real/images --output data/synthetic")


if __name__ == '__main__':
    main()
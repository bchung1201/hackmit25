#!/usr/bin/env python3
"""
Safe dependency installer for Mentra SLAM pipeline
Handles problematic packages gracefully
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_pip_install(packages, description="packages"):
    """Install packages with error handling"""
    if isinstance(packages, str):
        packages = [packages]
    
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Failed to install {package}: {e}")
            logger.info(f"   You can install {package} manually later if needed")

def install_core_dependencies():
    """Install core dependencies that should work on most systems"""
    logger.info("🔧 Installing core dependencies...")
    
    core_packages = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3", 
        "Pillow==10.0.1",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.64.0",
        "imageio>=2.25.0",
        "anthropic>=0.7.8",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "PyYAML>=6.0.0"
    ]
    
    for package in core_packages:
        run_pip_install(package)

def install_pytorch():
    """Install PyTorch with proper handling"""
    logger.info("🔥 Installing PyTorch...")
    
    try:
        # Try to install PyTorch
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch>=2.1.0", "torchvision>=0.16.0"
        ], stdout=subprocess.DEVNULL)
        logger.info("✅ PyTorch installed successfully")
    except subprocess.CalledProcessError:
        logger.warning("⚠️  PyTorch installation failed")
        logger.info("   Please install PyTorch manually from https://pytorch.org/")

def install_optional_packages():
    """Install optional packages that might fail"""
    logger.info("📦 Installing optional packages...")
    
    optional_packages = [
        "scikit-image>=0.20.0",
        "ffmpeg-python>=0.2.0",
        "open3d>=0.17.0",
        "trimesh>=3.23.5",
        "plotly>=5.11.0",
        "gdown>=4.6.0"
    ]
    
    for package in optional_packages:
        run_pip_install(package)

def check_installation():
    """Check if key packages can be imported"""
    logger.info("🧪 Testing installation...")
    
    test_imports = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("anthropic", "anthropic"),
        ("yaml", "PyYAML"),
        ("matplotlib", "matplotlib"),
    ]
    
    for module, package_name in test_imports:
        try:
            __import__(module)
            logger.info(f"✅ {package_name} working")
        except ImportError:
            logger.warning(f"⚠️  {package_name} not available")

def main():
    """Main installation process"""
    print("🚀 Mentra Pipeline Dependency Installer")
    print("=" * 50)
    
    try:
        # Step 1: Core dependencies
        install_core_dependencies()
        
        # Step 2: PyTorch
        install_pytorch()
        
        # Step 3: Optional packages
        install_optional_packages()
        
        # Step 4: Test installation
        check_installation()
        
        print("\n✅ Installation completed!")
        print("🧪 Test the installation with: python test_slam_integration.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Installation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

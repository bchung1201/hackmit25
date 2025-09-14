#!/usr/bin/env python3
"""
Install Emotion Detection Dependencies
Installs all required packages for the video emotion summary system
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name, import_name=None):
    """Install a package and test import"""
    if import_name is None:
        import_name = package_name
    
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        
        # Test import
        __import__(import_name)
        logger.info(f"✅ {package_name} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package_name}: {e}")
        return False
    except ImportError as e:
        logger.error(f"❌ Failed to import {import_name}: {e}")
        return False

def main():
    """Install all emotion detection dependencies"""
    print("🚀 Installing Emotion Detection Dependencies")
    print("=" * 50)
    
    # Core dependencies
    packages = [
        ("opencv-python==4.8.1.78", "cv2"),
        ("numpy==1.24.3", "numpy"),
        ("Pillow==10.0.1", "PIL"),
        ("torch>=2.1.0", "torch"),
        ("torchvision>=0.16.0", "torchvision"),
        ("scipy>=1.10.0", "scipy"),
        ("scikit-image>=0.20.0", "skimage"),
        ("matplotlib>=3.6.0", "matplotlib"),
        ("tqdm>=4.64.0", "tqdm"),
        ("imageio>=2.25.0", "imageio"),
        ("anthropic>=0.7.8", "anthropic"),
        ("ffmpeg-python>=0.2.0", "ffmpeg"),
        ("open3d>=0.17.0", "open3d"),
        ("trimesh>=3.23.5", "trimesh"),
        ("python-dotenv>=1.0.0", "dotenv"),
        ("pydantic>=2.5.0", "pydantic"),
        ("PyYAML>=6.0.0", "yaml"),
    ]
    
    # Emotion detection specific
    emotion_packages = [
        ("face-alignment>=1.1.1", "face_alignment"),
        ("mediapipe>=0.10.0", "mediapipe"),
        ("fer>=22.4.0", "fer"),
    ]
    
    all_packages = packages + emotion_packages
    
    success_count = 0
    total_count = len(all_packages)
    
    for package, import_name in all_packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_count}")
    print(f"❌ Failed installations: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 All dependencies installed successfully!")
        print("\nYou can now run:")
        print("  python demo_video_emotion_summary.py")
        print("  python video_emotion_summary.py your_video.mp4")
    else:
        print(f"\n⚠️  Some dependencies failed to install.")
        print("You may need to install them manually or check for conflicts.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

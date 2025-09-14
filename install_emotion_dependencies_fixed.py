#!/usr/bin/env python3
"""
Fixed Emotion Detection Dependencies Installer
Handles the 2 failed dependencies: numpy and fer
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
    """Install the 2 failed dependencies"""
    print("🔧 Fixing Failed Dependencies")
    print("=" * 40)
    
    # The 2 failed dependencies
    failed_packages = [
        ("numpy>=1.24.3", "numpy"),  # Use >= instead of == for Python 3.12 compatibility
        ("moviepy==1.0.3", "moviepy.editor"),  # Use older version that works with fer
    ]
    
    success_count = 0
    total_count = len(failed_packages)
    
    for package, import_name in failed_packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"\n📊 Fix Summary:")
    print(f"✅ Successfully fixed: {success_count}/{total_count}")
    print(f"❌ Still failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 All failed dependencies are now fixed!")
        print("\nYou can now run:")
        print("  python demo_video_emotion_summary.py")
        print("  python video_emotion_summary.py your_video.mp4")
        
        # Test the complete system
        print("\n🧪 Testing complete system...")
        try:
            import fer
            import numpy
            import moviepy.editor
            print("✅ All emotion detection dependencies are working!")
        except ImportError as e:
            print(f"❌ Some dependencies still have issues: {e}")
            return False
    else:
        print(f"\n⚠️  Some dependencies still failed.")
        print("You may need to install them manually or check for conflicts.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

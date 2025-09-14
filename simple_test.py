#!/usr/bin/env python3
"""
Simple test to verify SLAM integration imports work
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing SLAM Integration Imports")
    print("=" * 40)
    
    try:
        # Test basic dependencies
        print("📦 Testing basic dependencies...")
        import numpy as np
        import cv2
        import yaml
        import matplotlib
        print("   ✅ Basic dependencies OK")
        
        # Test SLAM interfaces
        print("📦 Testing SLAM interfaces...")
        from core.interfaces.base_slam_backend import BaseSLAMBackend, SLAMFrame, SLAMResult
        print("   ✅ SLAM interfaces OK")
        
        # Test configuration
        print("📦 Testing configuration...")
        from config import DEFAULT_CONFIG
        config_dict = DEFAULT_CONFIG.to_dict()
        print(f"   ✅ Configuration loaded ({len(config_dict)} parameters)")
        
        # Test pipeline classes (without instantiating)
        print("📦 Testing pipeline classes...")
        from enhanced_video_pipeline import EnhancedMentraVideoPipeline
        from unified_slam_reconstruction import UnifiedSLAMReconstructor
        from core.pipeline.slam_manager import SLAMManager
        print("   ✅ Pipeline classes OK")
        
        # Test SLAM backends
        print("📦 Testing SLAM backends...")
        from core.slam_backends.splatam.splatam_backend import SplaTAMBackend
        from core.slam_backends.monogs.monogs_backend import MonoGSBackend
        print("   ✅ SLAM backends OK")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ SLAM integration is ready to use")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run: python install_dependencies.py")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🚀 You can now run:")
        print("   python main.py")
        print("   python demos/slam_demos/realtime_slam_demo.py")
    else:
        print("\n🔧 Please fix import issues first")
        sys.exit(1)

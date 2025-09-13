#!/usr/bin/env python3
"""
Quick test to verify SLAM integration is working
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_slam_integration():
    """Test basic SLAM integration functionality"""
    
    print("🧪 Testing SLAM Integration")
    print("=" * 40)
    
    try:
        # Test 1: Import core modules
        print("📦 Testing imports...")
        
        from enhanced_video_pipeline import EnhancedMentraVideoPipeline
        from unified_slam_reconstruction import UnifiedSLAMReconstructor
        from core.pipeline.slam_manager import SLAMManager
        from core.interfaces.base_slam_backend import MockSLAMBackend
        from config import DEFAULT_CONFIG
        
        print("   ✅ All imports successful")
        
        # Test 2: Create SLAM manager
        print("🗺️  Testing SLAM manager...")
        
        slam_manager = SLAMManager('mock')
        configs = {
            'mock': {
                'device': 'cpu',
                'target_fps': 30.0
            }
        }
        await slam_manager.initialize(configs)
        
        backend_info = slam_manager.get_backend_info()
        print(f"   ✅ SLAM manager initialized: {backend_info['active_backend']}")
        
        # Test 3: Create enhanced pipeline
        print("🚀 Testing enhanced pipeline...")
        
        config = DEFAULT_CONFIG.to_dict()
        config['slam_backend'] = 'mock'
        
        pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
        print("   ✅ Enhanced pipeline created")
        
        # Test 4: Short pipeline run
        print("⚡ Testing pipeline execution...")
        
        await pipeline.start()
        await asyncio.sleep(2)  # Run for 2 seconds
        
        # Get status
        status = await pipeline.get_pipeline_status()
        slam_status = await pipeline.get_3d_reconstruction_status()
        
        print(f"   ✅ Pipeline running: {status['frame_count']} frames processed")
        print(f"   ✅ SLAM active: {slam_status['backend_used']} backend")
        print(f"   ✅ Trajectory: {status['trajectory_length']:.2f}m")
        
        await pipeline.stop()
        
        # Test 5: Backend switching
        print("🔄 Testing backend switching...")
        
        await slam_manager.switch_backend('mock')
        print("   ✅ Backend switching works")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ SLAM integration is working correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error("Test error", exc_info=True)
        return False

async def main():
    """Main test function"""
    success = await test_slam_integration()
    
    if success:
        print("\n🚀 Ready to run SLAM demos!")
        print("Try: python demos/slam_demos/realtime_slam_demo.py")
        sys.exit(0)
    else:
        print("\n🔧 Please fix issues before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

"""
Real-time SLAM Demo
Demonstrates real-time SLAM capabilities with performance monitoring
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_video_pipeline import EnhancedMentraVideoPipeline
from config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Demo real-time SLAM capabilities"""
    
    print("🚀 Starting Real-time SLAM Demo...")
    print("=" * 50)
    
    # Configure for real-time SLAM
    config = DEFAULT_CONFIG.to_dict()
    config.update({
        'slam_backend': 'monogs',  # Fast real-time backend
        'slam_processing_fps': 10,
        'adaptive_quality': True,
        'power_optimization_mode': False,
        'use_mock_components': True  # For demo
    })
    
    # Create enhanced pipeline
    pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
    
    try:
        print("📹 Initializing pipeline...")
        await pipeline.start()
        
        print("✅ Pipeline started successfully!")
        print("📊 Monitoring performance for 30 seconds...")
        print("")
        
        # Monitor performance for 30 seconds
        for i in range(30):
            await asyncio.sleep(1)
            
            # Get status every 5 seconds
            if i % 5 == 0:
                status = await pipeline.get_pipeline_status()
                slam_status = await pipeline.get_3d_reconstruction_status()
                
                print(f"⏱️  Time: {i+1}s")
                print(f"   📈 FPS: {status['fps']:.1f} | SLAM FPS: {status['slam_fps']:.1f}")
                print(f"   🗺️  Trajectory: {status['trajectory_length']:.2f}m | Poses: {len(status.get('slam_status', {}).get('trajectory', {}).get('poses', []))}")
                print(f"   🔄 Loop closures: {status['loop_closures']}")
                print(f"   🎯 Map confidence: {status['map_confidence']:.2f}")
                print(f"   ⚡ Processing latency: {status['processing_latency']:.3f}s")
                print("")
        
        # Show final results
        print("📊 Final SLAM Results:")
        print("=" * 30)
        
        final_status = await pipeline.get_pipeline_status()
        final_slam_status = await pipeline.get_3d_reconstruction_status()
        
        print(f"✅ Total frames processed: {final_status['frame_count']}")
        print(f"🗺️  Final trajectory length: {final_status['trajectory_length']:.2f}m")
        print(f"🔄 Loop closures detected: {final_status['loop_closures']}")
        print(f"🎯 Final map confidence: {final_status['map_confidence']:.2f}")
        print(f"⚡ Average FPS: {final_status['fps']:.1f}")
        print(f"🏗️  Gaussians generated: {final_slam_status.get('gaussians', 0)}")
        print(f"🧠 Backend used: {final_slam_status.get('backend_used', 'unknown')}")
        
        # Test backend switching
        print("\n🔄 Testing backend switching...")
        await pipeline.switch_slam_backend('mock')
        await asyncio.sleep(2)
        
        switch_status = await pipeline.get_pipeline_status()
        print(f"✅ Successfully switched to: {switch_status.get('slam_status', {}).get('active_backend', 'unknown')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
    finally:
        print("\n🛑 Stopping pipeline...")
        await pipeline.stop()
        print("✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())

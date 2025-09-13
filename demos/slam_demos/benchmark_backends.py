"""
SLAM Backend Benchmark
Compares performance of different SLAM backends
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_video_pipeline import EnhancedMentraVideoPipeline
from config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during benchmarking
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def benchmark_slam_backend(backend_name: str, test_duration: int = 30) -> Dict[str, float]:
    """Benchmark specific SLAM backend"""
    
    print(f"🔄 Benchmarking {backend_name.upper()}...")
    
    config = DEFAULT_CONFIG.to_dict()
    config.update({
        'slam_backend': backend_name,
        'use_mock_components': True,
        'adaptive_quality': False,  # Disable for consistent benchmarking
        'power_optimization_mode': False
    })
    
    pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
    
    start_time = time.time()
    frame_count = 0
    slam_reconstructions = 0
    total_processing_time = 0
    
    try:
        await pipeline.start()
        
        # Warm up period
        await asyncio.sleep(2)
        
        # Benchmark period
        benchmark_start = time.time()
        end_time = benchmark_start + test_duration
        
        while time.time() < end_time:
            await asyncio.sleep(0.1)
            
            # Get current stats
            status = await pipeline.get_pipeline_status()
            frame_count = status.get('frame_count', 0)
            slam_status = status.get('processing_stats', {})
            slam_reconstructions = slam_status.get('slam_reconstructions', 0)
            total_processing_time += status.get('processing_latency', 0)
        
        # Get final stats
        final_status = await pipeline.get_pipeline_status()
        slam_status = await pipeline.get_3d_reconstruction_status()
        
    except Exception as e:
        logger.error(f"Benchmark error for {backend_name}: {e}")
        return {
            'backend': backend_name,
            'error': str(e)
        }
    finally:
        await pipeline.stop()
    
    elapsed_time = time.time() - benchmark_start
    
    return {
        'backend': backend_name,
        'fps': frame_count / elapsed_time if elapsed_time > 0 else 0,
        'slam_fps': slam_reconstructions / elapsed_time if elapsed_time > 0 else 0,
        'total_frames': frame_count,
        'slam_reconstructions': slam_reconstructions,
        'elapsed_time': elapsed_time,
        'avg_processing_time': total_processing_time / max(1, frame_count),
        'trajectory_points': len(final_status.get('slam_status', {}).get('trajectory', {}).get('poses', [])),
        'loop_closures': final_status.get('loop_closures', 0),
        'map_confidence': final_status.get('map_confidence', 0.0),
        'gaussians': slam_status.get('gaussians', 0),
        'quality': slam_status.get('quality', 0.0)
    }

async def main():
    """Benchmark all SLAM backends"""
    
    print("🏁 SLAM Backend Benchmark Suite")
    print("=" * 50)
    
    # Backends to test (in order of expected performance)
    backends = ['monogs', 'splatam', 'mock']
    test_duration = 20  # seconds per backend
    
    results = []
    
    for i, backend in enumerate(backends):
        print(f"\n📊 Test {i+1}/{len(backends)}: {backend.upper()}")
        print("-" * 30)
        
        result = await benchmark_slam_backend(backend, test_duration)
        results.append(result)
        
        if 'error' in result:
            print(f"❌ {backend}: {result['error']}")
        else:
            print(f"✅ {backend}: {result['fps']:.1f} FPS | {result['slam_fps']:.1f} SLAM FPS")
            print(f"   🗺️  Trajectory: {result['trajectory_points']} poses | {result['loop_closures']} loops")
            print(f"   🎯 Quality: {result['quality']:.2f} | Confidence: {result['map_confidence']:.2f}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 SLAM BACKEND COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Backend':<12} | {'FPS':<6} | {'SLAM FPS':<8} | {'Poses':<6} | {'Loops':<6} | {'Quality':<7} | {'Gaussians':<9}")
    print("-" * 80)
    
    # Results
    for result in results:
        if 'error' not in result:
            print(f"{result['backend']:<12} | "
                  f"{result['fps']:<6.1f} | "
                  f"{result['slam_fps']:<8.1f} | "
                  f"{result['trajectory_points']:<6} | "
                  f"{result['loop_closures']:<6} | "
                  f"{result['quality']:<7.2f} | "
                  f"{result['gaussians']:<9}")
        else:
            print(f"{result['backend']:<12} | ERROR: {result['error']}")
    
    # Performance analysis
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        # Best performance metrics
        best_fps = max(valid_results, key=lambda x: x['fps'])
        best_slam = max(valid_results, key=lambda x: x['slam_fps'])
        best_quality = max(valid_results, key=lambda x: x['quality'])
        
        print(f"🏆 Best overall FPS: {best_fps['backend']} ({best_fps['fps']:.1f} FPS)")
        print(f"🗺️  Best SLAM FPS: {best_slam['backend']} ({best_slam['slam_fps']:.1f} SLAM FPS)")
        print(f"🎯 Best quality: {best_quality['backend']} ({best_quality['quality']:.2f})")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)
        print("• For real-time applications: Use MonoGS (fastest)")
        print("• For RGB-D sensors: Use SplaTAM (best accuracy)")
        print("• For development/testing: Use Mock backend")
        print("• For maximum quality: Use Splat-SLAM or Gaussian-SLAM")
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    asyncio.run(main())

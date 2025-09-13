"""
SLAM Trajectory Visualization Demo
Visualizes SLAM trajectory and map in real-time
"""

import asyncio
import logging
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

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

class TrajectoryVisualizer:
    """Real-time trajectory visualizer"""
    
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.loop_closures = []
        self.pipeline = None
        
        # Setup plots
        self.ax1.set_title("SLAM Trajectory (Top View)")
        self.ax1.set_xlabel("X (meters)")
        self.ax1.set_ylabel("Y (meters)")
        self.ax1.grid(True)
        self.ax1.axis('equal')
        
        self.ax2.set_title("Performance Metrics")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("FPS")
        self.ax2.grid(True)
        
        # Performance tracking
        self.time_points = []
        self.fps_points = []
        self.slam_fps_points = []
        self.start_time = None
        
    async def update_data(self):
        """Update trajectory data from pipeline"""
        if not self.pipeline:
            return
            
        try:
            status = await self.pipeline.get_pipeline_status()
            trajectory = await self.pipeline.get_slam_trajectory()
            
            if trajectory:
                # Extract positions
                positions = [pose[:3, 3] for pose in trajectory]
                if positions:
                    self.trajectory_x = [pos[0] for pos in positions]
                    self.trajectory_y = [pos[1] for pos in positions]
                    self.trajectory_z = [pos[2] for pos in positions]
            
            # Update performance metrics
            if self.start_time is None:
                self.start_time = asyncio.get_event_loop().time()
            
            current_time = asyncio.get_event_loop().time() - self.start_time
            self.time_points.append(current_time)
            self.fps_points.append(status.get('fps', 0))
            self.slam_fps_points.append(status.get('slam_fps', 0))
            
            # Keep only recent data
            if len(self.time_points) > 100:
                self.time_points.pop(0)
                self.fps_points.pop(0)
                self.slam_fps_points.pop(0)
            
            # Update loop closures
            self.loop_closures = status.get('loop_closures', 0)
            
        except Exception as e:
            logger.error(f"Data update error: {e}")
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot trajectory
        if self.trajectory_x and self.trajectory_y:
            self.ax1.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=2, label='Trajectory')
            self.ax1.plot(self.trajectory_x[0], self.trajectory_y[0], 'go', markersize=8, label='Start')
            if len(self.trajectory_x) > 1:
                self.ax1.plot(self.trajectory_x[-1], self.trajectory_y[-1], 'ro', markersize=8, label='Current')
            
            # Add loop closure indicators
            if self.loop_closures > 0:
                self.ax1.text(0.02, 0.98, f'Loop Closures: {self.loop_closures}', 
                             transform=self.ax1.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        self.ax1.set_title("SLAM Trajectory (Top View)")
        self.ax1.set_xlabel("X (meters)")
        self.ax1.set_ylabel("Y (meters)")
        self.ax1.grid(True)
        self.ax1.legend()
        self.ax1.axis('equal')
        
        # Plot performance
        if self.time_points and self.fps_points:
            self.ax2.plot(self.time_points, self.fps_points, 'g-', label='Video FPS')
            self.ax2.plot(self.time_points, self.slam_fps_points, 'r-', label='SLAM FPS')
        
        self.ax2.set_title("Performance Metrics")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("FPS")
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Add statistics
        if self.trajectory_x:
            total_distance = 0
            for i in range(1, len(self.trajectory_x)):
                dx = self.trajectory_x[i] - self.trajectory_x[i-1]
                dy = self.trajectory_y[i] - self.trajectory_y[i-1]
                total_distance += np.sqrt(dx*dx + dy*dy)
            
            stats_text = f'Distance: {total_distance:.2f}m\nPoses: {len(self.trajectory_x)}'
            self.ax1.text(0.02, 0.02, stats_text, transform=self.ax1.transAxes,
                         verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

async def run_slam_pipeline(visualizer):
    """Run SLAM pipeline in background"""
    config = DEFAULT_CONFIG.to_dict()
    config.update({
        'slam_backend': 'monogs',  # Good balance of speed and accuracy
        'slam_processing_fps': 10,
        'use_mock_components': True
    })
    
    visualizer.pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
    
    try:
        await visualizer.pipeline.start()
        
        # Update data periodically
        while True:
            await visualizer.update_data()
            await asyncio.sleep(0.1)  # 10 Hz update rate
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        if visualizer.pipeline:
            await visualizer.pipeline.stop()

def run_async_pipeline(visualizer):
    """Run async pipeline in thread"""
    asyncio.run(run_slam_pipeline(visualizer))

async def main():
    """Main demo function"""
    print("🗺️  SLAM Trajectory Visualization Demo")
    print("=" * 50)
    print("📊 Starting real-time visualization...")
    print("🔄 Close the plot window to stop the demo")
    
    # Create visualizer
    visualizer = TrajectoryVisualizer()
    
    # Start SLAM pipeline in background thread
    pipeline_thread = threading.Thread(target=run_async_pipeline, args=(visualizer,))
    pipeline_thread.daemon = True
    pipeline_thread.start()
    
    # Wait for pipeline to start
    await asyncio.sleep(2)
    
    # Start animation
    try:
        ani = FuncAnimation(visualizer.fig, visualizer.animate, interval=100, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
    except Exception as e:
        print(f"\n❌ Visualization error: {e}")
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1)

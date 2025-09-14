"""
Real-time Visual Dashboard for Mentra Pipeline
Shows live visualization of what the pipeline is doing to images
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
import os
import time
import threading
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from sam_segmentation import MockSAMSegmenter, SegmentationResult
from claude_vlm import MockClaudeVLMProcessor, SceneDescription
from modal_3d_reconstruction import Mock3DReconstructor
from core.interfaces.base_slam_backend import MockSLAMBackend, SLAMFrame, SLAMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualDashboard:
    """Real-time visual dashboard for pipeline processing"""
    
    def __init__(self):
        self.current_frame = None
        self.segmentation_result = None
        self.scene_description = None
        self.slam_trajectory = []
        self.point_cloud = []
        self.processing_stats = {
            'fps': 0.0,
            'frame_count': 0,
            'objects_detected': 0,
            'processing_time': 0.0,
            'slam_poses': 0
        }
        
        # Initialize components
        self.segmenter = MockSAMSegmenter()
        self.vlm_processor = MockClaudeVLMProcessor()
        self.reconstructor_3d = Mock3DReconstructor()
        self.slam_backend = MockSLAMBackend()
        
        # Setup matplotlib for real-time plotting
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Setup matplotlib for real-time visualization"""
        plt.ion()  # Turn on interactive mode
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Mentra Pipeline Visual Dashboard - Real-time Processing', fontsize=16)
        
        # Main image display
        self.ax_main = plt.subplot(2, 3, 1)
        self.ax_main.set_title('Original Image')
        self.ax_main.axis('off')
        self.img_main = None
        
        # Segmentation overlay
        self.ax_seg = plt.subplot(2, 3, 2)
        self.ax_seg.set_title('Object Segmentation')
        self.ax_seg.axis('off')
        self.img_seg = None
        
        # 3D Point Cloud
        self.ax_3d = plt.subplot(2, 3, 3, projection='3d')
        self.ax_3d.set_title('3D Point Cloud')
        self.scatter_3d = None
        
        # SLAM Trajectory
        self.ax_traj = plt.subplot(2, 3, 4)
        self.ax_traj.set_title('SLAM Trajectory')
        self.traj_line = None
        
        # Processing Stats
        self.ax_stats = plt.subplot(2, 3, 5)
        self.ax_stats.set_title('Processing Statistics')
        
        # Scene Analysis
        self.ax_scene = plt.subplot(2, 3, 6)
        self.ax_scene.set_title('Scene Analysis')
        
        plt.tight_layout()
        plt.show(block=False)
        
    async def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("🔧 Initializing pipeline components...")
        
        # Initialize SLAM backend
        slam_config = SLAMConfig(backend_name="mock")
        await self.slam_backend.initialize(slam_config)
        
        logger.info("✅ All components initialized")
    
    def draw_segmentation_overlay(self, frame: np.ndarray, segmentation_result: SegmentationResult) -> np.ndarray:
        """Draw segmentation masks on the frame"""
        overlay = frame.copy()
        
        # Define colors for different objects
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, mask in enumerate(segmentation_result.masks):
            color = colors[i % len(colors)]
            
            # Draw bounding box
            x, y, w, h = mask.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            cv2.putText(overlay, f"{mask.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw semi-transparent mask
            mask_overlay = overlay.copy()
            mask_bool = mask.mask
            mask_overlay[mask_bool] = color
            overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        return overlay
    
    def update_main_image(self, frame: np.ndarray):
        """Update the main image display"""
        self.ax_main.clear()
        self.ax_main.set_title(f'Original Image - Frame {self.processing_stats["frame_count"]}')
        self.ax_main.axis('off')
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax_main.imshow(frame_rgb)
    
    def update_segmentation(self, frame: np.ndarray, segmentation_result: SegmentationResult):
        """Update segmentation visualization"""
        self.ax_seg.clear()
        self.ax_seg.set_title(f'Object Segmentation - {len(segmentation_result.masks)} objects')
        self.ax_seg.axis('off')
        
        # Draw segmentation overlay
        overlay = self.draw_segmentation_overlay(frame, segmentation_result)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        self.ax_seg.imshow(overlay_rgb)
    
    def update_3d_point_cloud(self, point_cloud: np.ndarray):
        """Update 3D point cloud visualization"""
        self.ax_3d.clear()
        self.ax_3d.set_title(f'3D Point Cloud - {len(point_cloud)} points')
        
        if len(point_cloud) > 0:
            # Sample points for visualization (max 1000 for performance)
            if len(point_cloud) > 1000:
                indices = np.random.choice(len(point_cloud), 1000, replace=False)
                points = point_cloud[indices]
            else:
                points = point_cloud
            
            self.ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=points[:, 2], cmap='viridis', s=1)
            
            # Set equal aspect ratio
            max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                                 points[:, 1].max() - points[:, 1].min(),
                                 points[:, 2].max() - points[:, 2].min()]).max() / 2.0
            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def update_slam_trajectory(self, trajectory: List[np.ndarray]):
        """Update SLAM trajectory visualization"""
        self.ax_traj.clear()
        self.ax_traj.set_title(f'SLAM Trajectory - {len(trajectory)} poses')
        
        if len(trajectory) > 1:
            # Extract positions
            positions = np.array([pose[:3, 3] for pose in trajectory])
            
            # Plot trajectory
            self.ax_traj.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
            self.ax_traj.scatter(positions[:, 0], positions[:, 2], c=range(len(positions)), 
                                cmap='viridis', s=50)
            
            # Mark start and end
            self.ax_traj.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', label='Start')
            self.ax_traj.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='s', label='Current')
            
            self.ax_traj.legend()
            self.ax_traj.grid(True)
            self.ax_traj.set_xlabel('X (m)')
            self.ax_traj.set_ylabel('Z (m)')
    
    def update_processing_stats(self):
        """Update processing statistics display"""
        self.ax_stats.clear()
        self.ax_stats.set_title('Processing Statistics')
        
        # Create bar chart of stats
        stats = [
            self.processing_stats['fps'],
            self.processing_stats['objects_detected'],
            self.processing_stats['slam_poses'],
            self.processing_stats['processing_time'] * 1000  # Convert to ms
        ]
        
        labels = ['FPS', 'Objects', 'SLAM Poses', 'Time (ms)']
        colors = ['blue', 'green', 'orange', 'red']
        
        bars = self.ax_stats.bar(labels, stats, color=colors)
        self.ax_stats.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats):
            height = bar.get_height()
            self.ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{value:.1f}', ha='center', va='bottom')
    
    def update_scene_analysis(self, scene_description: SceneDescription):
        """Update scene analysis display"""
        self.ax_scene.clear()
        self.ax_scene.set_title('Scene Analysis')
        self.ax_scene.axis('off')
        
        # Display scene information as text
        text_info = f"""
Room Type: {scene_description.room_type}

Description:
{scene_description.overall_description}

Objects Detected: {len(scene_description.objects)}
- {scene_description.objects[0].category if scene_description.objects else "None"}
- {scene_description.objects[1].category if len(scene_description.objects) > 1 else ""}

Accessibility Features:
{chr(10).join(scene_description.accessibility_features)}
        """
        
        self.ax_scene.text(0.05, 0.95, text_info, transform=self.ax_scene.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def update_dashboard(self, frame: np.ndarray, segmentation_result: SegmentationResult, 
                        scene_description: SceneDescription, slam_result, point_cloud: np.ndarray):
        """Update the entire dashboard"""
        # Update main image
        self.update_main_image(frame)
        
        # Update segmentation
        self.update_segmentation(frame, segmentation_result)
        
        # Update 3D point cloud
        self.update_3d_point_cloud(point_cloud)
        
        # Update SLAM trajectory
        self.slam_trajectory = slam_result.camera_trajectory
        self.update_slam_trajectory(self.slam_trajectory)
        
        # Update processing stats
        self.update_processing_stats()
        
        # Update scene analysis
        self.update_scene_analysis(scene_description)
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.01)
    
    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the pipeline"""
        start_time = time.time()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        segmentation_result = await self.segmenter.segment_frame(
            frame=frame_rgb,
            frame_id=self.processing_stats['frame_count'],
            timestamp=time.time()
        )
        
        # Run VLM analysis
        scene_description = await self.vlm_processor.analyze_scene(
            frame=frame_rgb,
            segmentation_result=segmentation_result
        )
        
        # Run SLAM processing
        slam_frame = SLAMFrame(
            image=frame_rgb,
            timestamp=time.time(),
            frame_id=self.processing_stats['frame_count'],
            camera_intrinsics=np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)
        )
        
        slam_result = await self.slam_backend.process_frame(slam_frame)
        
        # Run 3D reconstruction (every 5th frame for performance)
        point_cloud = slam_result.point_cloud
        if self.processing_stats['frame_count'] % 5 == 0:
            reconstruction_result = await self.reconstructor_3d.reconstruct_scene(
                frames=[frame_rgb],
                camera_intrinsics=np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32),
                camera_poses=[slam_result.current_pose]
            )
            point_cloud = reconstruction_result.point_cloud
        
        # Update processing stats
        processing_time = time.time() - start_time
        self.processing_stats.update({
            'frame_count': self.processing_stats['frame_count'] + 1,
            'objects_detected': len(segmentation_result.masks),
            'slam_poses': len(slam_result.camera_trajectory),
            'processing_time': processing_time
        })
        
        return {
            'segmentation_result': segmentation_result,
            'scene_description': scene_description,
            'slam_result': slam_result,
            'point_cloud': point_cloud
        }

async def run_dashboard_with_webcam():
    """Run dashboard with webcam input"""
    logger.info("🎥 Starting Visual Dashboard with Webcam")
    
    # Create dashboard
    dashboard = VisualDashboard()
    await dashboard.initialize_components()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open webcam")
        return
    
    logger.info("📷 Webcam opened. Move around to see SLAM in action!")
    logger.info("Press 'q' in the OpenCV window to quit")
    
    frame_count = 0
    last_fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame through pipeline
            results = await dashboard.process_frame(frame)
            
            # Update dashboard
            dashboard.update_dashboard(
                frame,
                results['segmentation_result'],
                results['scene_description'],
                results['slam_result'],
                results['point_cloud']
            )
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                dashboard.processing_stats['fps'] = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("⏹️ Dashboard stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

async def run_dashboard_with_video(video_path: str):
    """Run dashboard with video file"""
    logger.info(f"🎬 Starting Visual Dashboard with Video: {video_path}")
    
    # Create dashboard
    dashboard = VisualDashboard()
    await dashboard.initialize_components()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return
    
    logger.info("📹 Video opened. Processing frames...")
    logger.info("Press 'q' in the OpenCV window to quit")
    
    frame_count = 0
    last_fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                results = await dashboard.process_frame(frame)
                dashboard.update_dashboard(
                    frame,
                    results['segmentation_result'],
                    results['scene_description'],
                    results['slam_result'],
                    results['point_cloud']
                )
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                dashboard.processing_stats['fps'] = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("⏹️ Dashboard stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

async def main():
    """Main entry point"""
    print("🎨 Mentra Pipeline Visual Dashboard")
    print("=" * 60)
    print("Real-time visualization of what the pipeline is doing!")
    print("Shows: Object detection, Segmentation, 3D reconstruction, SLAM trajectory")
    print("")
    
    print("Choose input source:")
    print("1. Webcam (live)")
    print("2. Video file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        await run_dashboard_with_webcam()
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        video_path = video_path.strip('"').strip("'")
        await run_dashboard_with_video(video_path)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

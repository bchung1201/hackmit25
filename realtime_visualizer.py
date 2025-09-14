"""
REAL-TIME Visual Dashboard with Live Image Processing
Shows actual images with real-time overlays, animations, and visual effects
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

class RealtimeVisualizer:
    """Real-time visualizer with live image processing and overlays"""
    
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
        
        # Animation variables
        self.animation_frame = 0
        self.pulse_alpha = 0.0
        self.scan_line_y = 0
        self.detection_boxes = []
        self.trajectory_points = []
        
        # Setup OpenCV windows for real-time display
        self.setup_opencv_windows()
        
    def setup_opencv_windows(self):
        """Setup OpenCV windows for real-time visualization"""
        cv2.namedWindow('🎥 LIVE PIPELINE VIEW', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('🔍 OBJECT DETECTION', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('🎨 SEGMENTATION MASKS', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('📊 LIVE STATS', cv2.WINDOW_AUTOSIZE)
        
        # Position windows
        cv2.moveWindow('🎥 LIVE PIPELINE VIEW', 50, 50)
        cv2.moveWindow('🔍 OBJECT DETECTION', 700, 50)
        cv2.moveWindow('🎨 SEGMENTATION MASKS', 50, 400)
        cv2.moveWindow('📊 LIVE STATS', 700, 400)
        
    async def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("🔧 Initializing pipeline components...")
        
        # Initialize SLAM backend
        slam_config = SLAMConfig(backend_name="mock")
        await self.slam_backend.initialize(slam_config)
        
        logger.info("✅ All components initialized")
    
    def create_animated_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create animated overlay effects"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Animated scanning line
        self.scan_line_y = (self.scan_line_y + 3) % height
        cv2.line(overlay, (0, self.scan_line_y), (width, self.scan_line_y), (0, 255, 255), 2)
        
        # Pulsing border
        self.pulse_alpha = (self.pulse_alpha + 0.1) % (2 * np.pi)
        pulse_intensity = int(50 + 50 * np.sin(self.pulse_alpha))
        cv2.rectangle(overlay, (5, 5), (width-5, height-5), (0, 255, 0), 3)
        cv2.rectangle(overlay, (8, 8), (width-8, height-8), (0, 255, 0), 1)
        
        # Processing indicator
        processing_text = "PROCESSING..." if self.processing_stats['processing_time'] > 0 else "READY"
        cv2.putText(overlay, processing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return overlay
    
    def draw_live_detection_boxes(self, frame: np.ndarray, segmentation_result: SegmentationResult) -> np.ndarray:
        """Draw animated detection boxes with real-time effects"""
        overlay = frame.copy()
        
        # Define colors for different objects
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
        for i, mask in enumerate(segmentation_result.masks):
            color = colors[i % len(colors)]
            x, y, w, h = mask.bbox
            
            # Animated bounding box
            thickness = 2 + int(2 * np.sin(self.animation_frame * 0.1 + i))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # Confidence score with animation
            confidence = mask.confidence
            cv2.putText(overlay, f"{confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Object ID
            cv2.putText(overlay, f"ID: {i}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Pulsing center point
            center_x, center_y = x + w//2, y + h//2
            pulse_radius = 5 + int(5 * np.sin(self.animation_frame * 0.2 + i))
            cv2.circle(overlay, (center_x, center_y), pulse_radius, color, -1)
        
        return overlay
    
    def create_segmentation_visualization(self, frame: np.ndarray, segmentation_result: SegmentationResult) -> np.ndarray:
        """Create animated segmentation mask visualization"""
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
            
            # Create mask overlay with animation
            mask_overlay = overlay.copy()
            mask_bool = mask.mask
            
            # Animated transparency
            alpha = 0.3 + 0.2 * np.sin(self.animation_frame * 0.15 + i)
            mask_overlay[mask_bool] = color
            
            # Blend with original
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_overlay, alpha, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay
    
    def create_live_stats_display(self) -> np.ndarray:
        """Create live statistics display"""
        # Create a black canvas
        stats_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(stats_img, "LIVE PIPELINE STATS", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # FPS with animation
        fps_color = (0, 255, 0) if self.processing_stats['fps'] > 10 else (0, 0, 255)
        cv2.putText(stats_img, f"FPS: {self.processing_stats['fps']:.1f}", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 2)
        
        # Object count
        cv2.putText(stats_img, f"Objects: {self.processing_stats['objects_detected']}", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # SLAM poses
        cv2.putText(stats_img, f"SLAM Poses: {self.processing_stats['slam_poses']}", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Processing time
        cv2.putText(stats_img, f"Process Time: {self.processing_stats['processing_time']*1000:.1f}ms", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Frame count
        cv2.putText(stats_img, f"Frame: {self.processing_stats['frame_count']}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Room type
        if self.scene_description:
            room_text = f"Room: {self.scene_description.room_type}"
            cv2.putText(stats_img, room_text, (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Animated progress bar
        progress = (self.animation_frame % 100) / 100.0
        bar_width = int(400 * progress)
        cv2.rectangle(stats_img, (50, 320), (50 + bar_width, 340), (0, 255, 0), -1)
        cv2.rectangle(stats_img, (50, 320), (450, 340), (255, 255, 255), 2)
        
        # Status indicator
        status_color = (0, 255, 0) if self.processing_stats['processing_time'] < 0.1 else (0, 0, 255)
        cv2.circle(stats_img, (550, 50), 20, status_color, -1)
        cv2.putText(stats_img, "STATUS", (520, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return stats_img
    
    def update_visualization(self, frame: np.ndarray, segmentation_result: SegmentationResult, 
                           scene_description: SceneDescription, slam_result, point_cloud: np.ndarray):
        """Update all visualizations in real-time"""
        self.animation_frame += 1
        
        # Main pipeline view with animated overlay
        main_view = self.create_animated_overlay(frame)
        cv2.imshow('🎥 LIVE PIPELINE VIEW', main_view)
        
        # Object detection with animated boxes
        detection_view = self.draw_live_detection_boxes(frame, segmentation_result)
        cv2.imshow('🔍 OBJECT DETECTION', detection_view)
        
        # Segmentation masks with animation
        segmentation_view = self.create_segmentation_visualization(frame, segmentation_result)
        cv2.imshow('🎨 SEGMENTATION MASKS', segmentation_view)
        
        # Live stats display
        stats_view = self.create_live_stats_display()
        cv2.imshow('📊 LIVE STATS', stats_view)
        
        # Update key
        key = cv2.waitKey(1) & 0xFF
        return key
    
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

async def run_realtime_webcam():
    """Run real-time visualization with webcam"""
    logger.info("🎥 Starting REAL-TIME Visual Dashboard with Webcam")
    
    # Create visualizer
    visualizer = RealtimeVisualizer()
    await visualizer.initialize_components()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open webcam")
        return
    
    logger.info("📷 Webcam opened. Move around to see REAL-TIME SLAM!")
    logger.info("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    last_fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame through pipeline
            results = await visualizer.process_frame(frame)
            
            # Update real-time visualization
            key = visualizer.update_visualization(
                frame,
                results['segmentation_result'],
                results['scene_description'],
                results['slam_result'],
                results['point_cloud']
            )
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                visualizer.processing_stats['fps'] = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                cv2.imwrite(f'screenshot_{int(time.time())}.jpg', frame)
                logger.info("📸 Screenshot saved!")
                
    except KeyboardInterrupt:
        logger.info("⏹️ Real-time dashboard stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def run_realtime_video(video_path: str):
    """Run real-time visualization with video file"""
    logger.info(f"🎬 Starting REAL-TIME Visual Dashboard with Video: {video_path}")
    
    # Create visualizer
    visualizer = RealtimeVisualizer()
    await visualizer.initialize_components()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return
    
    logger.info("📹 Video opened. Processing frames in REAL-TIME...")
    logger.info("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    last_fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_count += 1
            
            # Process every 2nd frame for performance
            if frame_count % 2 == 0:
                results = await visualizer.process_frame(frame)
                key = visualizer.update_visualization(
                    frame,
                    results['segmentation_result'],
                    results['scene_description'],
                    results['slam_result'],
                    results['point_cloud']
                )
                
                # Handle key presses
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    cv2.imwrite(f'screenshot_{int(time.time())}.jpg', frame)
                    logger.info("📸 Screenshot saved!")
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                visualizer.processing_stats['fps'] = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
                
    except KeyboardInterrupt:
        logger.info("⏹️ Real-time dashboard stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    """Main entry point"""
    print("🎨 REAL-TIME Mentra Pipeline Visual Dashboard")
    print("=" * 60)
    print("LIVE visualization with animated overlays and real-time effects!")
    print("Shows: Animated detection boxes, pulsing masks, live stats, scanning lines")
    print("")
    
    print("Choose input source:")
    print("1. Webcam (LIVE - best for SLAM)")
    print("2. Video file (LIVE processing)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        await run_realtime_webcam()
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        video_path = video_path.strip('"').strip("'")
        await run_realtime_video(video_path)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Real-time dashboard closed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

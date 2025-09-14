"""
Simple Video Upload with Real-time Visualization
Just upload a video and see what the AI understands in real-time
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from sam_segmentation import MockSAMSegmenter, SegmentationResult
from claude_vlm import MockClaudeVLMProcessor, SceneDescription
from modal_3d_reconstruction import Mock3DReconstructor
from core.interfaces.base_slam_backend import MockSLAMBackend, SLAMFrame, SLAMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoProcessor:
    """Simple video processor with real-time visualization"""
    
    def __init__(self):
        # Initialize components
        self.segmenter = MockSAMSegmenter()
        self.vlm_processor = MockClaudeVLMProcessor()
        self.reconstructor_3d = Mock3DReconstructor()
        self.slam_backend = MockSLAMBackend()
        
        # Animation variables
        self.animation_frame = 0
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🔧 Initializing pipeline...")
        slam_config = SLAMConfig(backend_name="mock")
        await self.slam_backend.initialize(slam_config)
        logger.info("✅ Ready to process videos!")
    
    def draw_detections(self, frame: np.ndarray, segmentation_result: SegmentationResult) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        overlay = frame.copy()
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, mask in enumerate(segmentation_result.masks):
            color = colors[i % len(colors)]
            x, y, w, h = mask.bbox
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            cv2.putText(overlay, f"{mask.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw object ID
            cv2.putText(overlay, f"Object {i}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay
    
    def draw_segmentation_masks(self, frame: np.ndarray, segmentation_result: SegmentationResult) -> np.ndarray:
        """Draw segmentation masks on frame"""
        overlay = frame.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, mask in enumerate(segmentation_result.masks):
            color = colors[i % len(colors)]
            
            # Create mask overlay
            mask_overlay = overlay.copy()
            mask_bool = mask.mask
            mask_overlay[mask_bool] = color
            
            # Blend with original
            overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        
        return overlay
    
    def create_info_display(self, frame: np.ndarray, scene_description: SceneDescription, 
                           objects_count: int, slam_poses: int, processing_time: float) -> np.ndarray:
        """Create info display overlay"""
        overlay = frame.copy()
        
        # Create semi-transparent background
        overlay_bg = np.zeros_like(overlay)
        overlay_bg = cv2.addWeighted(overlay, 0.7, overlay_bg, 0.3, 0)
        
        # Add text information
        y_pos = 30
        cv2.putText(overlay_bg, f"Room: {scene_description.room_type}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 30
        cv2.putText(overlay_bg, f"Objects: {objects_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        cv2.putText(overlay_bg, f"SLAM Poses: {slam_poses}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_pos += 30
        cv2.putText(overlay_bg, f"Process Time: {processing_time*1000:.1f}ms", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_pos += 30
        cv2.putText(overlay_bg, "Press 'q' to quit", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay_bg
    
    async def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame"""
        start_time = time.time()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        segmentation_result = await self.segmenter.segment_frame(
            frame=frame_rgb,
            frame_id=frame_id,
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
            frame_id=frame_id,
            camera_intrinsics=np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)
        )
        
        slam_result = await self.slam_backend.process_frame(slam_frame)
        
        processing_time = time.time() - start_time
        
        return {
            'segmentation_result': segmentation_result,
            'scene_description': scene_description,
            'slam_result': slam_result,
            'processing_time': processing_time
        }

async def process_video(video_path: str):
    """Process uploaded video with real-time visualization"""
    logger.info(f"🎬 Processing video: {video_path}")
    
    # Create processor
    processor = SimpleVideoProcessor()
    await processor.initialize()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return
    
    logger.info("📹 Video opened. Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                results = await processor.process_frame(frame, frame_count)
                
                # Create visualizations
                detection_view = processor.draw_detections(frame, results['segmentation_result'])
                mask_view = processor.draw_segmentation_masks(frame, results['segmentation_result'])
                info_view = processor.create_info_display(
                    frame, 
                    results['scene_description'],
                    len(results['segmentation_result'].masks),
                    len(results['slam_result'].camera_trajectory),
                    results['processing_time']
                )
                
                # Display windows
                cv2.imshow('🎥 Original Video', frame)
                cv2.imshow('🔍 Object Detection', detection_view)
                cv2.imshow('🎨 Segmentation Masks', mask_view)
                cv2.imshow('📊 AI Understanding', info_view)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        logger.info("⏹️ Processing stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    """Main entry point"""
    print("🎬 Simple Video Upload with Real-time AI Visualization")
    print("=" * 60)
    print("Upload a video and see what the AI understands in real-time!")
    print("")
    
    # Get video path from user
    video_path = input("Enter path to your video file: ").strip()
    video_path = video_path.strip('"').strip("'")
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        return
    
    print(f"✅ Found video: {video_path}")
    print("🚀 Starting real-time visualization...")
    
    await process_video(video_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

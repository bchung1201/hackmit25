"""
Focused video processing pipeline for Mentra glasses
Handles video streaming, 3D reconstruction, segmentation, and VLM analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from video_streaming import MentraVideoStreamer, MockMentraStreamer, VideoFrame
from modal_3d_reconstruction import Modal3DReconstructor, Mock3DReconstructor, ReconstructionResult
from sam_segmentation import SAMSegmenter, MockSAMSegmenter, SegmentationResult
from claude_vlm import ClaudeVLMProcessor, MockClaudeVLMProcessor, SceneDescription

logger = logging.getLogger(__name__)

@dataclass
class VideoPipelineState:
    """Current state of the video processing pipeline"""
    is_running: bool = False
    current_frame: Optional[VideoFrame] = None
    current_3d_model: Optional[ReconstructionResult] = None
    current_segmentation: Optional[SegmentationResult] = None
    current_scene_description: Optional[SceneDescription] = None
    frame_count: int = 0
    processing_latency: float = 0.0
    fps: float = 0.0

class MentraVideoPipeline:
    """
    Focused video processing pipeline for real-time environment understanding
    """
    
    def __init__(self, config: Dict[str, Any], use_mock_components: bool = True):
        self.config = config
        self.state = VideoPipelineState()
        
        # Initialize components
        if use_mock_components:
            self.video_streamer = MockMentraStreamer()
            self.reconstructor_3d = Mock3DReconstructor()
            self.segmenter = MockSAMSegmenter()
            self.vlm_processor = MockClaudeVLMProcessor()
        else:
            self.video_streamer = MentraVideoStreamer(
                stream_url=config.get("mentra_stream_url", "rtsp://mentra-glasses.local:8554/stream"),
                buffer_size=config.get("video_buffer_size", 10)
            )
            self.reconstructor_3d = Modal3DReconstructor(
                app_name=config.get("modal_app_name", "mentra-reality-pipeline")
            )
            self.segmenter = SAMSegmenter(
                model_type=config.get("sam_model_type", "vit_h"),
                checkpoint_path=config.get("sam_checkpoint_path", "sam_vit_h_4b8939.pth")
            )
            self.vlm_processor = ClaudeVLMProcessor(
                api_key=config.get("claude_api_key"),
                model=config.get("claude_model", "claude-3-5-sonnet-20241022")
            )
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Processing queues and timing
        self.frame_queue: List[VideoFrame] = []
        self.max_queue_size = config.get("max_queue_size", 5)
        self.target_fps = config.get("processing_fps", 30)
        self.frame_times = []
        
    def _setup_callbacks(self):
        """Setup callbacks between pipeline components"""
        self.video_streamer.add_frame_callback(self._on_new_frame)
    
    async def start(self):
        """Start the video processing pipeline"""
        logger.info("Starting Mentra Video Processing Pipeline...")
        
        try:
            self.state.is_running = True
            
            # Start all components concurrently
            await asyncio.gather(
                self.video_streamer.start_streaming(),
                self._main_processing_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Video pipeline startup error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping video processing pipeline...")
        
        self.state.is_running = False
        
        # Stop all components
        await asyncio.gather(
            self.video_streamer.stop_streaming(),
            return_exceptions=True
        )
        
        logger.info("Video pipeline stopped successfully")
    
    async def _main_processing_loop(self):
        """Main processing loop that coordinates all components"""
        logger.info("Starting main video processing loop...")
        
        while self.state.is_running:
            try:
                # Process frames from queue
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
                    await self._process_frame(frame)
                
                # Process 3D reconstruction queue
                await self.reconstructor_3d.process_queue()
                
                # Calculate FPS
                self._update_fps()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Main processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _on_new_frame(self, frame: VideoFrame):
        """Callback for new video frames"""
        logger.debug(f"New frame received: {frame.frame_id}")
        
        # Add to processing queue
        self.frame_queue.append(frame)
        
        # Maintain queue size
        if len(self.frame_queue) > self.max_queue_size:
            self.frame_queue.pop(0)
        
        self.state.current_frame = frame
        self.state.frame_count += 1
        
        # Track frame timing
        current_time = asyncio.get_event_loop().time()
        self.frame_times.append(current_time)
        
        # Keep only recent frame times (last 30 frames)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
    
    async def _process_frame(self, frame: VideoFrame):
        """Process a single frame through the pipeline"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.debug(f"Processing frame {frame.frame_id}")
            
            # Run segmentation
            segmentation_result = await self.segmenter.segment_frame(
                frame=frame.frame,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp
            )
            
            self.state.current_segmentation = segmentation_result
            
            # Run VLM analysis
            scene_description = await self.vlm_processor.analyze_scene(
                frame=frame.frame,
                segmentation_result=segmentation_result
            )
            
            self.state.current_scene_description = scene_description
            
            # Queue 3D reconstruction (every 10th frame to reduce load)
            if frame.frame_id % 10 == 0:
                await self._queue_3d_reconstruction(frame)
            
            # Calculate processing latency
            processing_time = asyncio.get_event_loop().time() - start_time
            self.state.processing_latency = processing_time
            
            logger.debug(f"Frame {frame.frame_id} processed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    async def _queue_3d_reconstruction(self, frame: VideoFrame):
        """Queue 3D reconstruction for a frame"""
        try:
            # Get recent frames for reconstruction
            recent_frames = self.frame_queue[-5:] if len(self.frame_queue) >= 5 else self.frame_queue
            
            if len(recent_frames) < 3:
                return  # Need at least 3 frames for reconstruction
            
            # Convert frames to numpy arrays
            frame_arrays = [f.frame for f in recent_frames]
            
            # Mock camera intrinsics and poses (in practice, these would come from camera calibration)
            camera_intrinsics = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ])
            
            # Mock camera poses (in practice, these would come from SLAM/visual odometry)
            camera_poses = [np.eye(4) for _ in recent_frames]
            
            # Queue reconstruction
            await self.reconstructor_3d.queue_reconstruction(
                frames=frame_arrays,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses
            )
            
        except Exception as e:
            logger.error(f"3D reconstruction queuing error: {e}")
    
    def _update_fps(self):
        """Update FPS calculation"""
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.state.fps = (len(self.frame_times) - 1) / time_diff
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "is_running": self.state.is_running,
            "frame_count": self.state.frame_count,
            "processing_latency": self.state.processing_latency,
            "fps": self.state.fps,
            "queue_size": len(self.frame_queue),
            "current_frame_id": self.state.current_frame.frame_id if self.state.current_frame else None,
            "segmentation_objects": len(self.state.current_segmentation.masks) if self.state.current_segmentation else 0,
            "scene_room_type": self.state.current_scene_description.room_type if self.state.current_scene_description else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_current_scene(self) -> Optional[SceneDescription]:
        """Analyze the current scene and return description"""
        if not self.state.current_frame:
            return None
        
        return await self.vlm_processor.analyze_scene(
            frame=self.state.current_frame.frame,
            segmentation_result=self.state.current_segmentation
        )
    
    async def get_segmentation_visualization(self) -> Optional[np.ndarray]:
        """Get visualization of current segmentation"""
        if not self.state.current_frame or not self.state.current_segmentation:
            return None
        
        return self.segmenter.visualize_segmentation(
            frame=self.state.current_frame.frame,
            result=self.state.current_segmentation
        )
    
    async def get_3d_reconstruction_status(self) -> Dict[str, Any]:
        """Get status of 3D reconstruction"""
        if not self.state.current_3d_model:
            return {"status": "no_reconstruction", "gaussians": 0, "quality": 0.0}
        
        return {
            "status": "completed",
            "gaussians": len(self.state.current_3d_model.gaussians),
            "quality": self.state.current_3d_model.reconstruction_quality,
            "processing_time": self.state.current_3d_model.processing_time
        }

# Example usage and testing
async def main():
    """Main entry point for testing the video pipeline"""
    
    # Configuration
    config = {
        "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",
        "video_buffer_size": 10,
        "modal_app_name": "mentra-reality-pipeline",
        "sam_model_type": "vit_h",
        "sam_checkpoint_path": "sam_vit_h_4b8939.pth",
        "claude_api_key": None,  # Set your API key here
        "claude_model": "claude-3-5-sonnet-20241022",
        "max_queue_size": 5,
        "processing_fps": 30
    }
    
    # Create pipeline (using mock components for development)
    pipeline = MentraVideoPipeline(config, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Run for a while to test
        await asyncio.sleep(10)
        
        # Get status
        status = await pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
        # Analyze current scene
        scene = await pipeline.analyze_current_scene()
        if scene:
            logger.info(f"Current scene: {scene.room_type}")
            logger.info(f"Description: {scene.overall_description}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())

"""
Pipeline integration module
Orchestrates all components for the complete Mentra Reality Pipeline
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
from claude_vlm import ClaudeVLMProcessor, MockClaudeVLMProcessor, SceneDescription, VoiceCommand
from wispr_voice import WisprVoiceProcessor, MockWisprVoiceProcessor, VoiceResponse

logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Current state of the pipeline"""
    is_running: bool = False
    current_frame: Optional[VideoFrame] = None
    current_3d_model: Optional[ReconstructionResult] = None
    current_segmentation: Optional[SegmentationResult] = None
    current_scene_description: Optional[SceneDescription] = None
    last_voice_command: Optional[VoiceCommand] = None
    last_voice_response: Optional[VoiceResponse] = None
    frame_count: int = 0
    processing_latency: float = 0.0

class MentraRealityPipeline:
    """
    Complete pipeline orchestrator for real-time environment understanding
    """
    
    def __init__(self, config: Dict[str, Any], use_mock_components: bool = True):
        self.config = config
        self.state = PipelineState()
        
        # Initialize components
        if use_mock_components:
            self.video_streamer = MockMentraStreamer()
            self.reconstructor_3d = Mock3DReconstructor()
            self.segmenter = MockSAMSegmenter()
            self.vlm_processor = MockClaudeVLMProcessor()
            self.voice_processor = MockWisprVoiceProcessor()
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
            self.voice_processor = WisprVoiceProcessor(
                model_size=config.get("whisper_model_size", "base"),
                language=config.get("language", "en")
            )
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Processing queues
        self.frame_queue: List[VideoFrame] = []
        self.max_queue_size = config.get("max_queue_size", 5)
        
    def _setup_callbacks(self):
        """Setup callbacks between pipeline components"""
        
        # Video streamer callbacks
        self.video_streamer.add_frame_callback(self._on_new_frame)
        
        # Voice processor callbacks
        self.voice_processor.add_command_callback(self._on_voice_command)
    
    async def start(self):
        """Start the complete pipeline"""
        logger.info("Starting Mentra Reality Pipeline...")
        
        try:
            self.state.is_running = True
            
            # Start all components concurrently
            await asyncio.gather(
                self.video_streamer.start_streaming(),
                self.voice_processor.start_listening(),
                self._main_processing_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Pipeline startup error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping Mentra Reality Pipeline...")
        
        self.state.is_running = False
        
        # Stop all components
        await asyncio.gather(
            self.video_streamer.stop_streaming(),
            self.voice_processor.stop_listening(),
            return_exceptions=True
        )
        
        logger.info("Pipeline stopped successfully")
    
    async def _main_processing_loop(self):
        """Main processing loop that coordinates all components"""
        logger.info("Starting main processing loop...")
        
        while self.state.is_running:
            try:
                # Process frames from queue
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
                    await self._process_frame(frame)
                
                # Process 3D reconstruction queue
                await self.reconstructor_3d.process_queue()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
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
    
    async def _on_voice_command(self, command: VoiceCommand):
        """Callback for voice commands"""
        logger.info(f"Voice command received: {command.text}")
        
        self.state.last_voice_command = command
        
        # Process command with current context
        context = self._build_context()
        response = await self.voice_processor.process_voice_command(command, context)
        
        self.state.last_voice_response = response
    
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
    
    def _build_context(self) -> Dict[str, Any]:
        """Build context for voice command processing"""
        context = {
            "timestamp": asyncio.get_event_loop().time(),
            "frame_count": self.state.frame_count,
            "processing_latency": self.state.processing_latency
        }
        
        if self.state.current_scene_description:
            context["scene_description"] = {
                "overall_description": self.state.current_scene_description.overall_description,
                "room_type": self.state.current_scene_description.room_type,
                "num_objects": len(self.state.current_scene_description.objects),
                "safety_concerns": self.state.current_scene_description.safety_concerns,
                "accessibility_features": self.state.current_scene_description.accessibility_features
            }
        
        if self.state.current_segmentation:
            context["segmentation"] = {
                "num_objects": len(self.state.current_segmentation.masks),
                "processing_time": self.state.current_segmentation.processing_time
            }
        
        if self.state.current_3d_model:
            context["3d_model"] = {
                "num_gaussians": len(self.state.current_3d_model.gaussians),
                "reconstruction_quality": self.state.current_3d_model.reconstruction_quality,
                "processing_time": self.state.current_3d_model.processing_time
            }
        
        return context
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "is_running": self.state.is_running,
            "frame_count": self.state.frame_count,
            "processing_latency": self.state.processing_latency,
            "queue_size": len(self.frame_queue),
            "current_frame_id": self.state.current_frame.frame_id if self.state.current_frame else None,
            "last_command": self.state.last_voice_command.text if self.state.last_voice_command else None,
            "last_response": self.state.last_voice_response.text if self.state.last_voice_response else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_voice_command(self, command_text: str) -> str:
        """Handle a voice command programmatically"""
        command = VoiceCommand(
            text=command_text,
            confidence=1.0,
            timestamp=asyncio.get_event_loop().time(),
            duration=0.0
        )
        
        context = self._build_context()
        response = await self.voice_processor.process_voice_command(command, context)
        
        return response.text

# Example usage and testing
async def main():
    """Main entry point for testing the pipeline"""
    
    # Configuration
    config = {
        "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",
        "video_buffer_size": 10,
        "modal_app_name": "mentra-reality-pipeline",
        "sam_model_type": "vit_h",
        "sam_checkpoint_path": "sam_vit_h_4b8939.pth",
        "claude_api_key": None,  # Set your API key here
        "claude_model": "claude-3-5-sonnet-20241022",
        "whisper_model_size": "base",
        "language": "en",
        "max_queue_size": 5
    }
    
    # Create pipeline (using mock components for development)
    pipeline = MentraRealityPipeline(config, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Run for a while to test
        await asyncio.sleep(30)
        
        # Test voice command
        response = await pipeline.handle_voice_command("Describe this room to me")
        logger.info(f"Voice command response: {response}")
        
        # Get status
        status = await pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())

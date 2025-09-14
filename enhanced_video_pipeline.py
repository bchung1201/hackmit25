"""
Enhanced Video Pipeline with Advanced SLAM Integration
Replaces video_pipeline.py with full SLAM capabilities
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import torch

from video_streaming import MentraVideoStreamer, MockMentraStreamer, VideoFrame
from unified_slam_reconstruction import UnifiedSLAMReconstructor, MockSLAMReconstructor, ReconstructionResult
from sam_segmentation import SAMSegmenter, MockSAMSegmenter, SegmentationResult
from claude_vlm import ClaudeVLMProcessor, MockClaudeVLMProcessor, SceneDescription
from core.interfaces.base_slam_backend import SLAMResult
from emotion_room_mapper import EmotionRoomMapper

logger = logging.getLogger(__name__)

@dataclass
class EnhancedVideoPipelineState:
    """Enhanced pipeline state with SLAM integration"""
    is_running: bool = False
    current_frame: Optional[VideoFrame] = None
    current_3d_model: Optional[ReconstructionResult] = None
    current_segmentation: Optional[SegmentationResult] = None
    current_scene_description: Optional[SceneDescription] = None
    current_slam_result: Optional[SLAMResult] = None
    
    # SLAM-specific state
    camera_trajectory: List[np.ndarray] = None
    trajectory_length: float = 0.0
    loop_closures: List[tuple] = None
    map_confidence: float = 0.0
    
    # Emotion detection state
    current_emotion: Optional[str] = None
    emotion_intensity: float = 0.0
    highlighted_rooms: List[str] = None
    emotion_confidence: float = 0.0
    
    # Performance metrics
    frame_count: int = 0
    processing_latency: float = 0.0
    fps: float = 0.0
    slam_fps: float = 0.0
    
    def __post_init__(self):
        if self.camera_trajectory is None:
            self.camera_trajectory = []
        if self.loop_closures is None:
            self.loop_closures = []
        if self.highlighted_rooms is None:
            self.highlighted_rooms = []

class EnhancedMentraVideoPipeline:
    """Enhanced video pipeline with advanced SLAM integration"""
    
    def __init__(self, config: Dict[str, Any], use_mock_components: bool = True):
        self.config = config
        self.state = EnhancedVideoPipelineState()
        
        # Initialize components with SLAM
        if use_mock_components:
            self.video_streamer = MockMentraStreamer()
            self.slam_reconstructor = MockSLAMReconstructor()
            self.segmenter = MockSAMSegmenter()
            self.vlm_processor = MockClaudeVLMProcessor()
        else:
            self.video_streamer = MentraVideoStreamer(
                stream_url=config.get("mentra_stream_url"),
                buffer_size=config.get("video_buffer_size", 10)
            )
            self.slam_reconstructor = UnifiedSLAMReconstructor(
                backend_type=config.get("slam_backend", "auto")
            )
            self.segmenter = SAMSegmenter(
                model_type=config.get("sam_model_type", "vit_h"),
                checkpoint_path=config.get("sam_checkpoint_path")
            )
            self.vlm_processor = ClaudeVLMProcessor(
                api_key=config.get("claude_api_key"),
                model=config.get("claude_model")
            )
        
        # SLAM-specific enhancements
        self.enable_loop_closure = config.get("enable_loop_closure", True)
        self.enable_global_optimization = config.get("enable_global_optimization", True)
        self.slam_processing_fps = config.get("slam_processing_fps", 10)
        self.slam_keyframe_every = config.get("slam_keyframe_every", 5)
        
        # Emotion detection and room highlighting
        self.emotion_room_mapper = EmotionRoomMapper()
        self.enable_emotion_detection = config.get("enable_emotion_detection", True)
        self.enable_room_highlighting = config.get("enable_room_highlighting", True)
        
        # Processing queues and timing
        self.frame_queue: List[VideoFrame] = []
        self.slam_frame_buffer: List[VideoFrame] = []
        self.max_queue_size = config.get("max_queue_size", 5)
        self.max_slam_buffer = config.get("max_slam_buffer", 10)
        self.target_fps = config.get("processing_fps", 30)
        
        # Performance tracking
        self.frame_times = []
        self.slam_times = []
        self.processing_stats = {
            'frames_processed': 0,
            'slam_reconstructions': 0,
            'segmentations': 0,
            'vlm_analyses': 0,
            'loop_closures_detected': 0
        }
        
        # Adaptive processing
        self.adaptive_quality = config.get("adaptive_quality", True)
        self.power_optimization = config.get("power_optimization_mode", False)
        
    async def start(self):
        """Start enhanced pipeline with SLAM"""
        logger.info("Starting Enhanced Mentra Video Pipeline with SLAM...")
        
        try:
            self.state.is_running = True
            
            # Initialize SLAM configs
            slam_configs = await self._create_slam_configs()
            
            # Initialize SLAM reconstructor
            await self.slam_reconstructor.initialize(slam_configs)
            
            # Setup callbacks
            self._setup_callbacks()
            
            # Start all components concurrently
            await asyncio.gather(
                self.video_streamer.start_streaming(),
                self._enhanced_processing_loop(),
                self._slam_processing_loop(),
                self._performance_monitoring_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Enhanced video pipeline startup error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the enhanced pipeline gracefully"""
        logger.info("Stopping enhanced video processing pipeline...")
        
        self.state.is_running = False
        
        # Stop all components
        await asyncio.gather(
            self.video_streamer.stop_streaming(),
            self.slam_reconstructor.reset() if hasattr(self.slam_reconstructor, 'reset') else asyncio.sleep(0),
            return_exceptions=True
        )
        
        # Log final statistics
        await self._log_final_statistics()
        
        logger.info("Enhanced video pipeline stopped successfully")
    
    async def _create_slam_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create SLAM backend configurations"""
        base_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_keyframes': self.config.get('max_keyframes', 50),
            'keyframe_every': self.slam_keyframe_every,
            'tracking_iterations': self.config.get('tracking_iterations', 10),
            'mapping_iterations': self.config.get('mapping_iterations', 60),
            'enable_loop_closure': self.enable_loop_closure,
            'enable_global_optimization': self.enable_global_optimization,
            'render_resolution': self.config.get('render_resolution', [640, 480]),
            'target_fps': self.slam_processing_fps
        }
        
        return {
            'splatam': {
                **base_config,
                'config_path': 'configs/slam_configs/splatam_config.yaml',
                'use_rgbd': self.config.get('use_rgbd', False)
            },
            'monogs': {
                **base_config,
                'config_path': 'configs/slam_configs/monogs_config.yaml',
                'use_gui': False,  # Disable GUI for headless operation
                'target_fps': max(10, self.slam_processing_fps)  # MonoGS optimized for 10+ FPS
            },
            'mock': {
                **base_config,
                'backend_name': 'mock'
            }
        }
    
    def _setup_callbacks(self):
        """Setup callbacks between pipeline components"""
        self.video_streamer.add_frame_callback(self._on_new_frame)
    
    async def _enhanced_processing_loop(self):
        """Enhanced processing loop with SLAM integration"""
        logger.info("Starting enhanced video processing loop...")
        
        while self.state.is_running:
            try:
                # Process frames from queue
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
                    await self._process_frame_enhanced(frame)
                
                # Calculate FPS
                self._update_fps()
                
                # Adaptive delay based on performance
                delay = self._calculate_adaptive_delay()
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Enhanced processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _slam_processing_loop(self):
        """Dedicated SLAM processing loop"""
        logger.info("Starting SLAM processing loop...")
        
        slam_frame_interval = max(1, int(self.target_fps / self.slam_processing_fps))
        
        while self.state.is_running:
            try:
                # Process SLAM frames at target FPS
                if len(self.slam_frame_buffer) >= 3:  # Need minimum frames for SLAM
                    frames_to_process = self.slam_frame_buffer[:5]  # Process in batches
                    self.slam_frame_buffer = self.slam_frame_buffer[3:]  # Keep overlap
                    
                    await self._process_slam_batch(frames_to_process)
                
                # SLAM processing interval
                await asyncio.sleep(1.0 / self.slam_processing_fps)
                
            except Exception as e:
                logger.error(f"SLAM processing loop error: {e}")
                await asyncio.sleep(2.0)
    
    async def _performance_monitoring_loop(self):
        """Monitor and adapt performance"""
        while self.state.is_running:
            try:
                # Check performance every 10 seconds
                await asyncio.sleep(10.0)
                
                if self.adaptive_quality:
                    await self._adapt_processing_quality()
                
                if self.power_optimization:
                    await self._optimize_power_consumption()
                
                # Log performance stats
                await self._log_performance_stats()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    async def _on_new_frame(self, frame: VideoFrame):
        """Callback for new video frames with SLAM integration"""
        logger.debug(f"New frame received: {frame.frame_id}")
        
        # Add to processing queue
        self.frame_queue.append(frame)
        
        # Maintain queue size
        if len(self.frame_queue) > self.max_queue_size:
            self.frame_queue.pop(0)
        
        # Add to SLAM buffer (less frequent)
        if frame.frame_id % max(1, int(self.target_fps / self.slam_processing_fps)) == 0:
            self.slam_frame_buffer.append(frame)
            
            # Maintain SLAM buffer size
            if len(self.slam_frame_buffer) > self.max_slam_buffer:
                self.slam_frame_buffer.pop(0)
        
        self.state.current_frame = frame
        self.state.frame_count += 1
        
        # Track frame timing
        current_time = asyncio.get_event_loop().time()
        self.frame_times.append(current_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > 60:  # Last 2 seconds at 30 FPS
            self.frame_times.pop(0)
    
    async def _process_frame_enhanced(self, frame: VideoFrame):
        """Process a single frame with enhanced SLAM context"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.debug(f"Processing frame {frame.frame_id} with SLAM context")
            
            # Run emotion detection and room highlighting (every frame)
            if self.enable_emotion_detection:
                emotion_mapper_result = await self.emotion_room_mapper.process_frame(
                    frame=frame.frame,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp
                )
                
                # Update state with emotion results
                self.state.current_emotion = emotion_mapper_result.get('current_emotion')
                self.state.emotion_intensity = emotion_mapper_result.get('intensity', 0.0)
                self.state.highlighted_rooms = emotion_mapper_result.get('rooms_highlighted', [])
                
                # Get emotion confidence from the emotion_result
                emotion_result = emotion_mapper_result.get('emotion_result')
                if emotion_result and hasattr(emotion_result, 'dominant_emotion') and emotion_result.dominant_emotion:
                    self.state.emotion_confidence = emotion_result.dominant_emotion.confidence
                else:
                    self.state.emotion_confidence = 0.0
                
                logger.debug(f"Detected emotion: {self.state.current_emotion} "
                           f"(intensity: {self.state.emotion_intensity:.2f}, "
                           f"rooms: {len(self.state.highlighted_rooms)})")
            
            # Run segmentation (every frame for responsiveness)
            segmentation_result = await self.segmenter.segment_frame(
                frame=frame.frame,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp
            )
            
            self.state.current_segmentation = segmentation_result
            self.processing_stats['segmentations'] += 1
            
            # Run VLM analysis with spatial context (less frequent for performance)
            if frame.frame_id % 5 == 0:  # Every 5th frame
                scene_description = await self._analyze_scene_with_slam_context(
                    frame, segmentation_result
                )
                self.state.current_scene_description = scene_description
                self.processing_stats['vlm_analyses'] += 1
            
            # Calculate processing latency
            processing_time = asyncio.get_event_loop().time() - start_time
            self.state.processing_latency = processing_time
            
            self.processing_stats['frames_processed'] += 1
            
            logger.debug(f"Frame {frame.frame_id} processed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Enhanced frame processing error: {e}")
    
    async def _process_slam_batch(self, frames: List[VideoFrame]):
        """Process batch of frames with SLAM"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.debug(f"Processing SLAM batch of {len(frames)} frames")
            
            # Get camera intrinsics
            camera_intrinsics = self._get_camera_intrinsics()
            
            # Extract frame arrays
            frame_arrays = [f.frame for f in frames]
            
            # Run SLAM reconstruction
            slam_result = await self.slam_reconstructor.reconstruct_scene(
                frames=frame_arrays,
                camera_intrinsics=camera_intrinsics
            )
            
            if slam_result:
                # Update pipeline state with SLAM results
                self.state.current_3d_model = slam_result
                self.state.camera_trajectory = slam_result.camera_poses
                self.state.trajectory_length = slam_result.trajectory_length
                self.state.loop_closures = slam_result.loop_closures
                self.state.map_confidence = slam_result.map_confidence
                
                # Track loop closures
                if len(slam_result.loop_closures) > self.processing_stats['loop_closures_detected']:
                    new_loops = len(slam_result.loop_closures) - self.processing_stats['loop_closures_detected']
                    logger.info(f"Detected {new_loops} new loop closures!")
                    self.processing_stats['loop_closures_detected'] = len(slam_result.loop_closures)
                
                self.processing_stats['slam_reconstructions'] += 1
            
            # Track SLAM processing time
            slam_processing_time = asyncio.get_event_loop().time() - start_time
            self.slam_times.append(slam_processing_time)
            
            # Keep recent SLAM times
            if len(self.slam_times) > 30:
                self.slam_times.pop(0)
            
            logger.debug(f"SLAM batch processed in {slam_processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"SLAM batch processing error: {e}")
    
    async def _analyze_scene_with_slam_context(
        self, 
        frame: VideoFrame, 
        segmentation_result: SegmentationResult
    ) -> SceneDescription:
        """Analyze scene with enhanced spatial context from SLAM"""
        try:
            # Build spatial context from SLAM data
            spatial_context = self._build_spatial_context()
            
            # Enhanced VLM analysis with spatial context
            if spatial_context:
                enhanced_query = f"""
                Analyze this scene with the following spatial context:
                - Current position in trajectory: {len(self.state.camera_trajectory)} poses recorded
                - Trajectory length: {self.state.trajectory_length:.2f} meters
                - Map confidence: {self.state.map_confidence:.2f}
                - Loop closures detected: {len(self.state.loop_closures)}
                
                Provide enhanced spatial understanding considering this movement history.
                """
                
                scene_description = await self.vlm_processor.analyze_scene(
                    frame=frame.frame,
                    segmentation_result=segmentation_result,
                    user_query=enhanced_query
                )
            else:
                # Fallback to standard analysis
                scene_description = await self.vlm_processor.analyze_scene(
                    frame=frame.frame,
                    segmentation_result=segmentation_result
                )
            
            return scene_description
            
        except Exception as e:
            logger.error(f"Enhanced scene analysis error: {e}")
            # Fallback to standard analysis
            return await self.vlm_processor.analyze_scene(
                frame=frame.frame,
                segmentation_result=segmentation_result
            )
    
    def _build_spatial_context(self) -> Optional[Dict[str, Any]]:
        """Build spatial context from SLAM data"""
        if not self.state.camera_trajectory:
            return None
        
        return {
            'trajectory_length': self.state.trajectory_length,
            'num_poses': len(self.state.camera_trajectory),
            'loop_closures': len(self.state.loop_closures),
            'map_confidence': self.state.map_confidence,
            'current_position': self.state.camera_trajectory[-1][:3, 3].tolist() if self.state.camera_trajectory else None
        }
    
    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics (from config or calibration)"""
        # TODO: Get actual camera intrinsics from Mentra glasses
        return np.array([
            [800, 0, 320],
            [0, 800, 240], 
            [0, 0, 1]
        ])
    
    def _update_fps(self):
        """Update FPS calculations"""
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.state.fps = (len(self.frame_times) - 1) / time_diff
        
        if len(self.slam_times) >= 2:
            slam_time_diff = sum(self.slam_times[-10:]) / min(10, len(self.slam_times))
            if slam_time_diff > 0:
                self.state.slam_fps = 1.0 / slam_time_diff
    
    def _calculate_adaptive_delay(self) -> float:
        """Calculate adaptive processing delay based on performance"""
        if not self.adaptive_quality:
            return 0.01  # Fixed delay
        
        # Adapt based on current performance
        target_frame_time = 1.0 / self.target_fps
        current_processing_time = self.state.processing_latency
        
        if current_processing_time > target_frame_time:
            # Processing is slow, increase delay
            return max(0.005, target_frame_time - current_processing_time)
        else:
            # Processing is fast, minimal delay
            return 0.005
    
    async def _adapt_processing_quality(self):
        """Adapt processing quality based on performance"""
        avg_fps = self.state.fps
        target_fps = self.target_fps
        
        if avg_fps < target_fps * 0.8:  # 20% below target
            logger.info(f"Performance below target ({avg_fps:.1f} < {target_fps}), adapting quality")
            
            # Reduce SLAM processing frequency
            if self.slam_processing_fps > 5:
                self.slam_processing_fps = max(5, self.slam_processing_fps - 1)
                logger.info(f"Reduced SLAM FPS to {self.slam_processing_fps}")
            
            # Reduce segmentation frequency
            # TODO: Implement segmentation frequency adaptation
            
        elif avg_fps > target_fps * 1.1:  # 10% above target
            # Increase quality if performance allows
            if self.slam_processing_fps < 15:
                self.slam_processing_fps = min(15, self.slam_processing_fps + 1)
                logger.info(f"Increased SLAM FPS to {self.slam_processing_fps}")
    
    async def _optimize_power_consumption(self):
        """Optimize for power consumption (wearable device)"""
        if not self.power_optimization:
            return
        
        # Reduce processing frequency if battery is low
        # TODO: Integrate with actual battery monitoring
        
        # Reduce SLAM quality for power saving
        if hasattr(self.slam_reconstructor, 'slam_manager'):
            current_backend = self.slam_reconstructor.slam_manager.backend_type
            if current_backend == 'splatam':
                # Switch to more efficient MonoGS
                await self.slam_reconstructor.switch_backend('monogs')
                logger.info("Switched to MonoGS for power optimization")
    
    async def _log_performance_stats(self):
        """Log current performance statistics"""
        logger.info("=== Performance Statistics ===")
        logger.info(f"FPS: {self.state.fps:.1f} (target: {self.target_fps})")
        logger.info(f"SLAM FPS: {self.state.slam_fps:.1f} (target: {self.slam_processing_fps})")
        logger.info(f"Processing latency: {self.state.processing_latency:.3f}s")
        logger.info(f"Frames processed: {self.processing_stats['frames_processed']}")
        logger.info(f"SLAM reconstructions: {self.processing_stats['slam_reconstructions']}")
        logger.info(f"Loop closures: {self.processing_stats['loop_closures_detected']}")
        logger.info(f"Trajectory length: {self.state.trajectory_length:.2f}m")
        logger.info(f"Map confidence: {self.state.map_confidence:.2f}")
        
        if hasattr(self.slam_reconstructor, 'slam_manager'):
            backend_info = self.slam_reconstructor.slam_manager.get_backend_info()
            logger.info(f"Active SLAM backend: {backend_info['active_backend']}")
    
    async def _log_final_statistics(self):
        """Log final statistics when stopping"""
        logger.info("=== Final Statistics ===")
        logger.info(f"Total frames processed: {self.processing_stats['frames_processed']}")
        logger.info(f"Total SLAM reconstructions: {self.processing_stats['slam_reconstructions']}")
        logger.info(f"Total segmentations: {self.processing_stats['segmentations']}")
        logger.info(f"Total VLM analyses: {self.processing_stats['vlm_analyses']}")
        logger.info(f"Loop closures detected: {self.processing_stats['loop_closures_detected']}")
        logger.info(f"Final trajectory length: {self.state.trajectory_length:.2f}m")
        logger.info(f"Final map confidence: {self.state.map_confidence:.2f}")
    
    # Enhanced API methods
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        slam_status = await self.slam_reconstructor.get_slam_status() if hasattr(self.slam_reconstructor, 'get_slam_status') else {}
        
        return {
            "is_running": self.state.is_running,
            "frame_count": self.state.frame_count,
            "processing_latency": self.state.processing_latency,
            "fps": self.state.fps,
            "slam_fps": self.state.slam_fps,
            "queue_size": len(self.frame_queue),
            "slam_buffer_size": len(self.slam_frame_buffer),
            "current_frame_id": self.state.current_frame.frame_id if self.state.current_frame else None,
            "segmentation_objects": len(self.state.current_segmentation.masks) if self.state.current_segmentation else 0,
            "scene_room_type": self.state.current_scene_description.room_type if self.state.current_scene_description else None,
            "trajectory_length": self.state.trajectory_length,
            "loop_closures": len(self.state.loop_closures),
            "map_confidence": self.state.map_confidence,
            "processing_stats": self.processing_stats,
            "slam_status": slam_status,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_slam_trajectory(self) -> List[np.ndarray]:
        """Get current SLAM trajectory"""
        return self.state.camera_trajectory.copy()
    
    async def get_3d_reconstruction_status(self) -> Dict[str, Any]:
        """Get enhanced 3D reconstruction status"""
        if not self.state.current_3d_model:
            return {"status": "no_reconstruction", "gaussians": 0, "quality": 0.0}
        
        return {
            "status": "completed",
            "gaussians": len(self.state.current_3d_model.gaussians),
            "quality": self.state.current_3d_model.reconstruction_quality,
            "processing_time": self.state.current_3d_model.processing_time,
            "trajectory_length": self.state.current_3d_model.trajectory_length,
            "loop_closures": len(self.state.current_3d_model.loop_closures),
            "map_confidence": self.state.current_3d_model.map_confidence,
            "backend_used": self.state.current_3d_model.backend_used
        }
    
    async def get_emotion_status(self) -> Dict[str, Any]:
        """Get emotion detection and room highlighting status"""
        return {
            "current_emotion": self.state.current_emotion,
            "emotion_intensity": self.state.emotion_intensity,
            "emotion_confidence": self.state.emotion_confidence,
            "highlighted_rooms": self.state.highlighted_rooms,
            "emotion_trends": self.emotion_room_mapper.get_emotion_trends(),
            "mapper_status": self.emotion_room_mapper.get_current_status()
        }
    
    async def get_room_highlighting_status(self) -> Dict[str, Any]:
        """Get room highlighting status"""
        return self.emotion_room_mapper.room_highlighter.get_highlighting_status()
    
    async def get_highlighted_map_image(self, width: int = 1000, height: int = 600) -> Optional[bytes]:
        """Get highlighted map image as bytes"""
        return self.emotion_room_mapper.get_highlighted_map_image(width, height)
    
    async def switch_slam_backend(self, backend_name: str) -> None:
        """Switch SLAM backend"""
        await self.slam_reconstructor.switch_backend(backend_name)
        logger.info(f"Switched SLAM backend to: {backend_name}")
    
    async def save_slam_session(self, filepath: str) -> None:
        """Save complete SLAM session"""
        await self.slam_reconstructor.save_map(filepath)
        logger.info(f"Saved SLAM session to: {filepath}")
    
    async def load_slam_session(self, filepath: str) -> None:
        """Load SLAM session"""
        await self.slam_reconstructor.load_map(filepath)
        logger.info(f"Loaded SLAM session from: {filepath}")

# Example usage and testing
async def main():
    """Main entry point for testing the enhanced video pipeline"""
    
    # Enhanced configuration with SLAM
    config = {
        "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",
        "video_buffer_size": 10,
        "slam_backend": "auto",  # auto, splatam, monogs, mock
        "sam_model_type": "vit_h",
        "sam_checkpoint_path": "sam_vit_h_4b8939.pth",
        "claude_api_key": None,  # Set your API key here
        "claude_model": "claude-3-5-sonnet-20241022",
        "max_queue_size": 5,
        "processing_fps": 30,
        "slam_processing_fps": 10,
        "slam_keyframe_every": 5,
        "enable_loop_closure": True,
        "enable_global_optimization": True,
        "adaptive_quality": True,
        "power_optimization_mode": False
    }
    
    # Create enhanced pipeline
    pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Run for a while to test
        await asyncio.sleep(30)
        
        # Get comprehensive status
        status = await pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
        # Get SLAM trajectory
        trajectory = await pipeline.get_slam_trajectory()
        logger.info(f"SLAM trajectory has {len(trajectory)} poses")
        
        # Get 3D reconstruction status
        reconstruction_status = await pipeline.get_3d_reconstruction_status()
        logger.info(f"3D reconstruction: {reconstruction_status}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())

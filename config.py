"""
Configuration management for Mentra Reality Pipeline
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Enhanced configuration class with SLAM support"""
    
    # API Keys
    claude_api_key: Optional[str] = None
    modal_token: Optional[str] = None
    
    # Mentra Configuration
    mentra_stream_url: str = "rtsp://mentra-glasses.local:8554/stream"
    
    # SAM Configuration
    sam_checkpoint_path: str = "sam_vit_h_4b8939.pth"
    
    # Pipeline Settings
    video_buffer_size: int = 10
    max_queue_size: int = 5
    processing_fps: int = 30
    
    # SLAM Configuration
    slam_backend: str = "auto"  # auto, splatam, monogs, splat_slam, gaussian_slam, mock
    slam_config_dir: str = "configs/slam_configs"
    enable_loop_closure: bool = True
    enable_global_optimization: bool = True
    
    # Real-time Performance
    slam_processing_fps: int = 10
    slam_keyframe_every: int = 5
    max_trajectory_length: int = 1000
    max_slam_buffer: int = 10
    
    # Wearable Optimizations
    power_optimization_mode: bool = False
    adaptive_quality: bool = True
    motion_prediction: bool = True
    
    # Advanced SLAM Features
    enable_mesh_reconstruction: bool = False
    enable_dense_mapping: bool = True
    enable_photo_realistic_rendering: bool = False
    
    # Emotion Detection Settings
    enable_emotion_detection: bool = True
    enable_room_highlighting: bool = True
    emotion_detection_fps: int = 10
    face_detection_method: str = "opencv"  # opencv, mtcnn
    emonet_model_path: str = "pretrained/emonet_8.pth"
    n_emotion_classes: int = 8
    
    # Development Settings
    use_mock_components: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Create configuration from environment variables"""
        return cls(
            claude_api_key=os.getenv('CLAUDE_API_KEY'),
            modal_token=os.getenv('MODAL_TOKEN'),
            mentra_stream_url=os.getenv('MENTRA_STREAM_URL', 'rtsp://mentra-glasses.local:8554/stream'),
            sam_checkpoint_path=os.getenv('SAM_CHECKPOINT_PATH', 'sam_vit_h_4b8939.pth'),
            video_buffer_size=int(os.getenv('VIDEO_BUFFER_SIZE', '10')),
            max_queue_size=int(os.getenv('MAX_QUEUE_SIZE', '5')),
            processing_fps=int(os.getenv('PROCESSING_FPS', '30')),
            enable_emotion_detection=os.getenv('ENABLE_EMOTION_DETECTION', 'true').lower() == 'true',
            enable_room_highlighting=os.getenv('ENABLE_ROOM_HIGHLIGHTING', 'true').lower() == 'true',
            emotion_detection_fps=int(os.getenv('EMOTION_DETECTION_FPS', '10')),
            face_detection_method=os.getenv('FACE_DETECTION_METHOD', 'opencv'),
            emonet_model_path=os.getenv('EMONET_MODEL_PATH', 'pretrained/emonet_8.pth'),
            n_emotion_classes=int(os.getenv('N_EMOTION_CLASSES', '8')),
            use_mock_components=os.getenv('USE_MOCK_COMPONENTS', 'true').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'claude_api_key': self.claude_api_key,
            'modal_token': self.modal_token,
            'mentra_stream_url': self.mentra_stream_url,
            'sam_checkpoint_path': self.sam_checkpoint_path,
            'video_buffer_size': self.video_buffer_size,
            'max_queue_size': self.max_queue_size,
            'processing_fps': self.processing_fps,
            'use_mock_components': self.use_mock_components,
            'log_level': self.log_level,
            'modal_app_name': 'mentra-reality-pipeline',
            'sam_model_type': 'vit_h',
            'claude_model': 'claude-3-5-sonnet-20241022',
            'whisper_model_size': 'base',
            'language': 'en',
            # SLAM parameters
            'slam_backend': self.slam_backend,
            'slam_config_dir': self.slam_config_dir,
            'enable_loop_closure': self.enable_loop_closure,
            'enable_global_optimization': self.enable_global_optimization,
            'slam_processing_fps': self.slam_processing_fps,
            'slam_keyframe_every': self.slam_keyframe_every,
            'max_trajectory_length': self.max_trajectory_length,
            'max_slam_buffer': self.max_slam_buffer,
            'power_optimization_mode': self.power_optimization_mode,
            'adaptive_quality': self.adaptive_quality,
            'motion_prediction': self.motion_prediction,
            'enable_mesh_reconstruction': self.enable_mesh_reconstruction,
            'enable_dense_mapping': self.enable_dense_mapping,
            'enable_photo_realistic_rendering': self.enable_photo_realistic_rendering
        }

# Default configuration
DEFAULT_CONFIG = PipelineConfig.from_env()

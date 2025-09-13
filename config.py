"""
Configuration management for Mentra Reality Pipeline
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Configuration class for the pipeline"""
    
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
            'language': 'en'
        }

# Default configuration
DEFAULT_CONFIG = PipelineConfig.from_env()

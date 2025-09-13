"""
Video streaming module for Mentra glasses integration
Handles real-time video capture and streaming from MentraOS
"""

import cv2
import asyncio
import logging
from typing import Optional, Callable, Any
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoFrame:
    """Represents a video frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    resolution: tuple[int, int]

class MentraVideoStreamer:
    """
    Handles video streaming from Mentra glasses using MentraOS features
    """
    
    def __init__(self, stream_url: str, buffer_size: int = 10):
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer: list[VideoFrame] = []
        self.frame_callbacks: list[Callable[[VideoFrame], None]] = []
        self.is_streaming = False
        self.frame_count = 0
        
    async def start_streaming(self):
        """Start streaming video from Mentra glasses"""
        logger.info(f"Starting video stream from: {self.stream_url}")
        
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video stream: {self.stream_url}")
            
            # Configure video capture
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Stream properties: {width}x{height} @ {fps} FPS")
            
            self.is_streaming = True
            
            # Start frame capture loop
            await self._capture_loop()
            
        except Exception as e:
            logger.error(f"Video streaming error: {e}")
            await self.stop_streaming()
            raise
    
    async def stop_streaming(self):
        """Stop video streaming"""
        logger.info("Stopping video stream...")
        self.is_streaming = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    async def _capture_loop(self):
        """Main frame capture loop"""
        while self.is_streaming:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    await asyncio.sleep(0.1)
                    continue
                
                # Create VideoFrame object
                video_frame = VideoFrame(
                    frame=frame.copy(),
                    timestamp=asyncio.get_event_loop().time(),
                    frame_id=self.frame_count,
                    resolution=(frame.shape[1], frame.shape[0])
                )
                
                # Add to buffer
                self.frame_buffer.append(video_frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Notify callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(video_frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                self.frame_count += 1
                
                # Control frame rate
                await asyncio.sleep(1/30)  # Target 30 FPS
                
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                await asyncio.sleep(0.1)
    
    def add_frame_callback(self, callback: Callable[[VideoFrame], None]):
        """Add a callback function to be called for each new frame"""
        self.frame_callbacks.append(callback)
    
    def get_latest_frame(self) -> Optional[VideoFrame]:
        """Get the most recent frame"""
        return self.frame_buffer[-1] if self.frame_buffer else None
    
    def get_frame_buffer(self) -> list[VideoFrame]:
        """Get the current frame buffer"""
        return self.frame_buffer.copy()

class MockMentraStreamer(MentraVideoStreamer):
    """
    Mock video streamer for development/testing without actual Mentra glasses
    """
    
    def __init__(self, mock_video_path: str = None):
        super().__init__("mock://stream")
        self.mock_video_path = mock_video_path or 0  # Default to webcam
    
    async def start_streaming(self):
        """Start mock streaming using webcam or video file"""
        logger.info("Starting mock video stream for development...")
        
        try:
            self.cap = cv2.VideoCapture(self.mock_video_path)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open mock video source: {self.mock_video_path}")
            
            self.is_streaming = True
            await self._capture_loop()
            
        except Exception as e:
            logger.error(f"Mock streaming error: {e}")
            await self.stop_streaming()
            raise

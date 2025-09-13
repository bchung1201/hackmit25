"""
Real Video Processing Demo
Demonstrates actual video processing with webcam or video file
"""

import cv2
import asyncio
import logging
import numpy as np
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealVideoProcessor:
    """
    Real video processor that works with actual video sources
    """
    
    def __init__(self, video_source: str = "webcam"):
        self.video_source = video_source
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    async def start_processing(self):
        """Start processing real video"""
        logger.info(f"🎥 Starting real video processing from: {self.video_source}")
        
        try:
            # Initialize video capture
            if self.video_source == "webcam":
                self.cap = cv2.VideoCapture(0)  # Default webcam
            else:
                self.cap = cv2.VideoCapture(self.video_source)  # Video file
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.video_source}")
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📐 Video properties: {width}x{height} @ {fps} FPS")
            
            self.is_running = True
            
            # Start processing loop
            await self._processing_loop()
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            await self.stop_processing()
            raise
    
    async def stop_processing(self):
        """Stop video processing"""
        logger.info("🛑 Stopping video processing...")
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
    
    async def _processing_loop(self):
        """Main video processing loop"""
        logger.info("🔄 Starting video processing loop...")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = await self._process_frame(frame)
                
                # Display frame
                cv2.imshow('Mentra Video Processing Pipeline', processed_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exit key pressed")
                    break
                
                # Update FPS counter
                self._update_fps()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        self.frame_count += 1
        
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Add processing information overlay
        self._add_info_overlay(processed_frame)
        
        # Simulate object detection (draw rectangles)
        self._simulate_object_detection(processed_frame)
        
        # Simulate segmentation (draw circles)
        self._simulate_segmentation(processed_frame)
        
        return processed_frame
    
    def _add_info_overlay(self, frame: np.ndarray):
        """Add information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Add text information
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {width}x{height}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def _simulate_object_detection(self, frame: np.ndarray):
        """Simulate object detection by drawing rectangles"""
        height, width = frame.shape[:2]
        
        # Simulate detected objects
        objects = [
            {"name": "Person", "bbox": (width//4, height//4, width//4, height//4), "color": (0, 255, 0)},
            {"name": "Chair", "bbox": (width//2, height//2, width//6, height//6), "color": (255, 0, 0)},
            {"name": "Table", "bbox": (width//3, height//3, width//3, height//6), "color": (0, 0, 255)}
        ]
        
        for obj in objects:
            x, y, w, h = obj["bbox"]
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], 2)
            # Draw label
            cv2.putText(frame, obj["name"], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj["color"], 2)
    
    def _simulate_segmentation(self, frame: np.ndarray):
        """Simulate segmentation by drawing circles"""
        height, width = frame.shape[:2]
        
        # Simulate segmented regions
        centers = [
            (width//4, height//4),
            (width//2, height//2),
            (3*width//4, 3*height//4)
        ]
        
        for i, center in enumerate(centers):
            color = (255, 255, 0) if i == 0 else (255, 0, 255) if i == 1 else (0, 255, 255)
            cv2.circle(frame, center, 30, color, -1)
            cv2.circle(frame, center, 30, (255, 255, 255), 2)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

async def main():
    """Main function"""
    print("🎥 Real Video Processing Demo")
    print("=" * 50)
    print("This demo processes REAL video from your webcam or video file")
    print("You'll see:")
    print("- Live video feed")
    print("- Simulated object detection (green rectangles)")
    print("- Simulated segmentation (colored circles)")
    print("- Real-time FPS counter")
    print("- Frame information overlay")
    print("")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Ask user for video source
    print("\nSelect video source:")
    print("1. Webcam (default)")
    print("2. Video file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    video_source = "webcam"
    if choice == "2":
        video_path = input("Enter path to video file: ").strip()
        if video_path:
            video_source = video_path
        else:
            print("No file path provided, using webcam")
    
    # Create and run processor
    processor = RealVideoProcessor(video_source)
    
    try:
        await processor.start_processing()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await processor.stop_processing()
        print("👋 Video processing stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

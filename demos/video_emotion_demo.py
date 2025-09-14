#!/usr/bin/env python3
"""
Video Emotion Detection Demo
Process a video file for emotion detection and room highlighting
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from emotion_room_mapper import EmotionRoomMapper
from room_highlighting.roborock_parser import RoborockMapParser
from config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoEmotionDemo:
    """Demo for processing video files with emotion detection"""
    
    def __init__(self, video_path: str, output_dir: str = "video_demo_outputs"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.mapper = EmotionRoomMapper()
        self.map_parser = RoborockMapParser()
        
        # Load default map
        self._load_default_map()
        
        # Video processing
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.total_frames = 0
        
        # Results tracking
        self.emotion_history = []
        self.highlighted_maps = []
        
    def _load_default_map(self):
        """Load default Roborock map for demo"""
        # Create a sample map based on the provided image
        sample_map = {
            "width": 1000,
            "height": 600,
            "rooms": [
                {
                    "id": "bar_room",
                    "name": "Bar Room",
                    "color": [255, 255, 0],
                    "coordinates": [(50, 50), (300, 50), (300, 400), (50, 400)]
                },
                {
                    "id": "laundry_room", 
                    "name": "Laundry Room",
                    "color": [255, 127, 80],
                    "coordinates": [(300, 50), (500, 50), (500, 200), (300, 200)]
                },
                {
                    "id": "hallway",
                    "name": "Hallway", 
                    "color": [64, 224, 208],
                    "coordinates": [(300, 200), (500, 200), (500, 250), (300, 250)]
                },
                {
                    "id": "game_room",
                    "name": "Game Room",
                    "color": [135, 206, 235],
                    "coordinates": [(500, 50), (800, 50), (800, 200), (500, 200)]
                },
                {
                    "id": "living_room",
                    "name": "Living Room",
                    "color": [147, 112, 219],
                    "coordinates": [(500, 250), (800, 250), (800, 500), (500, 500)]
                }
            ],
            "walls": [],
            "obstacles": [],
            "charging_station": [400, 100],
            "robot_position": [200, 200],
            "map_scale": 1.0
        }
        
        self.mapper.room_highlighter.load_map_from_json(sample_map)
        logger.info("Loaded default map for demo")
    
    async def process_video(self, save_frames: bool = True, save_maps: bool = True, 
                           frame_interval: int = 30):
        """
        Process video file for emotion detection
        
        Args:
            save_frames: Save processed frames with face detection
            save_maps: Save highlighted maps
            frame_interval: Process every Nth frame (1 = every frame)
        """
        logger.info(f"🎬 Starting video processing: {self.video_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 Video info: {width}x{height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Create output video writer if saving frames
        out_writer = None
        if save_frames:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = self.output_dir / "processed_video.mp4"
            out_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, self.fps, (width, height)
            )
        
        try:
            frame_idx = 0
            processed_frames = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % frame_interval == 0:
                    logger.info(f"🎯 Processing frame {frame_idx}/{self.total_frames}")
                    
                    # Process frame for emotion detection
                    result = await self.mapper.process_frame(
                        frame=frame,
                        frame_id=frame_idx,
                        timestamp=frame_idx / self.fps
                    )
                    
                    # Store results
                    self.emotion_history.append(result)
                    
                    # Display results
                    self._display_frame_results(result, frame_idx)
                    
                    # Draw face detection and emotion on frame
                    if save_frames:
                        annotated_frame = self._annotate_frame(frame, result)
                        out_writer.write(annotated_frame)
                    
                    # Save highlighted map
                    if save_maps and processed_frames % 10 == 0:  # Save every 10th processed frame
                        await self._save_highlighted_map(result, processed_frames)
                    
                    processed_frames += 1
                
                frame_idx += 1
                
                # Show progress
                if frame_idx % 100 == 0:
                    progress = (frame_idx / self.total_frames) * 100
                    logger.info(f"📈 Progress: {progress:.1f}% ({frame_idx}/{self.total_frames})")
            
            logger.info(f"✅ Video processing completed!")
            logger.info(f"📊 Processed {processed_frames} frames")
            
            # Generate summary
            await self._generate_summary()
            
        finally:
            if self.cap:
                self.cap.release()
            if out_writer:
                out_writer.release()
    
    def _display_frame_results(self, result: dict, frame_idx: int):
        """Display emotion detection results for a frame"""
        emotion = result.get('current_emotion', 'unknown')
        intensity = result.get('intensity', 0.0)
        highlighted_rooms = result.get('rooms_highlighted', [])
        confidence = 0.0
        
        # Get confidence from emotion result
        emotion_result = result.get('emotion_result')
        if emotion_result and hasattr(emotion_result, 'dominant_emotion') and emotion_result.dominant_emotion:
            confidence = emotion_result.dominant_emotion.confidence
        
        print(f"  Frame {frame_idx:4d}: {emotion:8s} (conf: {confidence:.2f}, intensity: {intensity:.2f}) - {len(highlighted_rooms)} rooms")
    
    def _annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Annotate frame with emotion detection results"""
        annotated = frame.copy()
        
        # Get emotion info
        emotion = result.get('current_emotion', 'unknown')
        intensity = result.get('intensity', 0.0)
        highlighted_rooms = result.get('rooms_highlighted', [])
        
        # Draw emotion text
        text = f"Emotion: {emotion} (Intensity: {intensity:.2f})"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw highlighted rooms
        if highlighted_rooms:
            rooms_text = f"Rooms: {', '.join(highlighted_rooms[:3])}"  # Show first 3 rooms
            if len(highlighted_rooms) > 3:
                rooms_text += "..."
            cv2.putText(annotated, rooms_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw face detection boxes if available
        emotion_result = result.get('emotion_result')
        if emotion_result and hasattr(emotion_result, 'emotions'):
            for emotion_data in emotion_result.emotions:
                if hasattr(emotion_data, 'face_box'):
                    x, y, w, h = emotion_data.face_box
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw emotion label
                    label = f"{emotion_data.emotion} ({emotion_data.confidence:.2f})"
                    cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
    
    async def _save_highlighted_map(self, result: dict, frame_idx: int):
        """Save highlighted map image"""
        try:
            # Generate highlighted map
            map_image_bytes = self.mapper.get_highlighted_map_image(1000, 600)
            
            if map_image_bytes:
                # Save image
                output_path = self.output_dir / f"map_frame_{frame_idx:04d}.png"
                
                with open(output_path, 'wb') as f:
                    f.write(map_image_bytes)
                
                self.highlighted_maps.append(output_path)
                logger.info(f"  💾 Saved map: {output_path.name}")
            else:
                logger.warning(f"  ⚠️  Could not generate map for frame {frame_idx}")
                
        except Exception as e:
            logger.error(f"Error saving map for frame {frame_idx}: {e}")
    
    async def _generate_summary(self):
        """Generate processing summary"""
        logger.info("\n📊 Processing Summary")
        logger.info("=" * 40)
        
        if not self.emotion_history:
            logger.info("No frames processed")
            return
        
        # Count emotions
        emotion_counts = {}
        total_intensity = 0.0
        total_confidence = 0.0
        rooms_highlighted_count = 0
        
        for result in self.emotion_history:
            emotion = result.get('current_emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            total_intensity += result.get('intensity', 0.0)
            rooms_highlighted_count += len(result.get('rooms_highlighted', []))
            
            # Get confidence
            emotion_result = result.get('emotion_result')
            if emotion_result and hasattr(emotion_result, 'dominant_emotion') and emotion_result.dominant_emotion:
                total_confidence += emotion_result.dominant_emotion.confidence
        
        # Display statistics
        logger.info(f"Total frames processed: {len(self.emotion_history)}")
        logger.info(f"Average intensity: {total_intensity / len(self.emotion_history):.3f}")
        logger.info(f"Average confidence: {total_confidence / len(self.emotion_history):.3f}")
        logger.info(f"Total room highlights: {rooms_highlighted_count}")
        logger.info(f"Highlighted maps saved: {len(self.highlighted_maps)}")
        
        logger.info("\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.emotion_history)) * 100
            logger.info(f"  {emotion:12s}: {count:4d} frames ({percentage:5.1f}%)")
        
        # Get emotion trends
        trends = self.mapper.get_emotion_trends()
        if trends and 'trend' in trends:
            logger.info(f"\nOverall emotion trend: {trends['trend']}")
        
        logger.info(f"\n📁 Output files saved to: {self.output_dir}")
        logger.info("  - processed_video.mp4 (annotated video)")
        logger.info("  - map_frame_*.png (highlighted maps)")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process video for emotion detection')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='video_demo_outputs', 
                       help='Output directory for results')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--no-frames', action='store_true',
                       help='Don\'t save processed frames')
    parser.add_argument('--no-maps', action='store_true',
                       help='Don\'t save highlighted maps')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    # Create demo
    demo = VideoEmotionDemo(args.video_path, args.output_dir)
    
    try:
        await demo.process_video(
            save_frames=not args.no_frames,
            save_maps=not args.no_maps,
            frame_interval=args.frame_interval
        )
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

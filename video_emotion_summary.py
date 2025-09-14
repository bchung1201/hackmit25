#!/usr/bin/env python3
"""
Video Emotion Summary Generator
Processes a video and generates a single map showing overall emotions per room
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from emotion_detection.room_aware_processor import RoomAwareEmotionProcessor
from room_highlighting.emotion_summary_generator import EmotionSummaryMapGenerator, EmotionMapConfig
from room_highlighting.roborock_parser import RoborockMapParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoEmotionSummary:
    """Main class for video emotion summary processing"""
    
    def __init__(self, video_path: str, output_dir: str = "emotion_summary_outputs"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.room_processor = RoomAwareEmotionProcessor()
        self.map_generator = EmotionSummaryMapGenerator()
        self.map_parser = RoborockMapParser()
        
        # Load default map
        self._load_default_map()
        
        # Processing state
        self.cap = None
        self.total_frames = 0
        self.processed_frames = 0
        self.start_time = None
        
    def _load_default_map(self):
        """Load default Roborock map"""
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
        
        self.map_generator.room_highlighter.load_map_from_json(sample_map)
        logger.info("Loaded default map for processing")
    
    async def process_video(self, frame_interval: int = 30, 
                           simulate_movement: bool = True) -> Dict[str, Any]:
        """
        Process video and generate emotion summary
        
        Args:
            frame_interval: Process every Nth frame
            simulate_movement: Simulate room movement for demo
            
        Returns:
            Processing results
        """
        logger.info(f"🎬 Starting video emotion summary processing")
        logger.info(f"📹 Video: {self.video_path}")
        logger.info(f"📁 Output: {self.output_dir}")
        
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 Video info: {width}x{height}, {fps} FPS, {self.total_frames} frames")
        
        self.start_time = time.time()
        
        try:
            frame_idx = 0
            self.processed_frames = 0
            
            # Simulate room positions for demo (in real use, this would come from SLAM)
            room_positions = self._generate_room_positions(self.total_frames)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % frame_interval == 0:
                    # Get simulated position (in real use, this would come from SLAM)
                    if simulate_movement:
                        position = room_positions[min(frame_idx // frame_interval, len(room_positions) - 1)]
                    else:
                        position = (400, 300)  # Default center position
                    
                    # Process frame
                    result = await self.room_processor.process_frame_with_room(
                        frame=frame,
                        position=position,
                        frame_id=frame_idx,
                        timestamp=frame_idx / fps
                    )
                    
                    self.processed_frames += 1
                    
                    # Log progress
                    if self.processed_frames % 10 == 0:
                        progress = (frame_idx / self.total_frames) * 100
                        logger.info(f"📈 Progress: {progress:.1f}% - "
                                  f"Room: {result['current_room']}, "
                                  f"Emotions: {len(result['emotions_detected'])}")
                
                frame_idx += 1
            
            # Generate summary
            logger.info("📊 Generating emotion summary...")
            emotion_summary = self.room_processor.get_room_emotion_summary()
            
            # Generate and save summary map
            await self._generate_summary_outputs(emotion_summary)
            
            # Calculate processing statistics
            processing_time = time.time() - self.start_time
            stats = self.room_processor.get_processing_statistics()
            
            results = {
                "processing_time": processing_time,
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "fps": self.processed_frames / processing_time if processing_time > 0 else 0,
                "emotion_summary": emotion_summary,
                "statistics": stats
            }
            
            logger.info("✅ Video processing completed!")
            return results
            
        finally:
            if self.cap:
                self.cap.release()
    
    def _generate_room_positions(self, total_frames: int) -> List[Tuple[float, float]]:
        """Generate simulated room positions for demo"""
        # Simulate movement through different rooms
        positions = []
        
        # Define room centers
        room_centers = {
            "Bar Room": (175, 225),
            "Laundry Room": (400, 125),
            "Hallway": (400, 225),
            "Game Room": (650, 125),
            "Living Room": (650, 375)
        }
        
        # Create a path through rooms
        room_sequence = ["Bar Room", "Hallway", "Living Room", "Game Room", "Laundry Room", "Hallway"]
        
        frames_per_room = total_frames // len(room_sequence)
        
        for i, room in enumerate(room_sequence):
            start_frame = i * frames_per_room
            end_frame = min((i + 1) * frames_per_room, total_frames)
            
            room_center = room_centers[room]
            
            # Add some movement within the room
            for frame in range(start_frame, end_frame):
                # Add small random movement
                x_offset = np.random.normal(0, 20)
                y_offset = np.random.normal(0, 20)
                
                position = (
                    max(0, min(1000, room_center[0] + x_offset)),
                    max(0, min(600, room_center[1] + y_offset))
                )
                positions.append(position)
        
        return positions
    
    async def _generate_summary_outputs(self, emotion_summary):
        """Generate summary outputs"""
        # Generate summary map
        map_config = EmotionMapConfig(
            map_width=1200,
            map_height=800,
            show_statistics=True,
            show_emotion_distribution=True,
            show_confidence_scores=True,
            show_trends=True
        )
        
        # Save summary map
        map_path = self.output_dir / "emotion_summary_map.png"
        self.map_generator.save_summary_map(emotion_summary, str(map_path), map_config)
        
        # Save emotion data
        data_path = self.output_dir / "emotion_data.json"
        self.map_generator.save_emotion_data(emotion_summary, str(data_path))
        
        # Generate detailed report
        await self._generate_detailed_report(emotion_summary)
        
        logger.info(f"📁 Summary outputs saved to: {self.output_dir}")
        logger.info(f"  - emotion_summary_map.png (main summary map)")
        logger.info(f"  - emotion_data.json (raw data)")
        logger.info(f"  - emotion_report.txt (detailed report)")
    
    async def _generate_detailed_report(self, emotion_summary):
        """Generate detailed text report"""
        report_path = self.output_dir / "emotion_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("EMOTION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Mood: {emotion_summary.overall_mood}\n")
            f.write(f"Most Emotional Room: {emotion_summary.most_emotional_room}\n")
            f.write(f"Least Emotional Room: {emotion_summary.least_emotional_room}\n")
            f.write(f"Total Processing Time: {emotion_summary.total_processing_time:.2f} seconds\n")
            f.write(f"Rooms with Data: {len(emotion_summary.rooms)}\n\n")
            
            # Room-by-room analysis
            f.write("ROOM-BY-ROOM ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            for room_name, room_data in emotion_summary.rooms.items():
                f.write(f"\n{room_name.upper()}\n")
                f.write("-" * len(room_name) + "\n")
                f.write(f"Dominant Emotion: {room_data.dominant_emotion}\n")
                f.write(f"Average Confidence: {room_data.average_confidence:.3f}\n")
                f.write(f"Average Intensity: {room_data.average_intensity:.3f}\n")
                f.write(f"Emotion Trend: {room_data.emotion_trend}\n")
                f.write(f"Total Time: {room_data.total_time:.2f} seconds\n")
                f.write(f"Emotion Count: {len(room_data.emotions)}\n")
                
                f.write("Emotion Distribution:\n")
                for emotion, percentage in sorted(room_data.emotion_distribution.items(), 
                                               key=lambda x: x[1], reverse=True):
                    f.write(f"  {emotion}: {percentage:.1%}\n")
            
            # Model statistics
            stats = self.room_processor.get_processing_statistics()
            f.write(f"\nPROCESSING STATISTICS\n")
            f.write("-" * 22 + "\n")
            f.write(f"Total Frames Processed: {stats['total_frames_processed']}\n")
            f.write(f"Total Emotions Detected: {stats['total_emotions_detected']}\n")
            f.write(f"Rooms with Emotions: {stats['rooms_with_emotions']}\n")
            f.write(f"Room History Length: {stats['room_history_length']}\n")
            
            if 'model_statistics' in stats:
                model_stats = stats['model_statistics']
                f.write(f"Available Models: {', '.join(model_stats['available_models'])}\n")
                f.write(f"Model Priorities: {', '.join(model_stats['model_priorities'])}\n")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate emotion summary from video')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='emotion_summary_outputs', 
                       help='Output directory for results')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--no-simulation', action='store_true',
                       help='Disable room movement simulation')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video_path).exists():
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    # Create processor
    processor = VideoEmotionSummary(args.video_path, args.output_dir)
    
    try:
        # Process video
        results = await processor.process_video(
            frame_interval=args.frame_interval,
            simulate_movement=not args.no_simulation
        )
        
        # Display results
        print("\n🎉 PROCESSING COMPLETED!")
        print("=" * 40)
        print(f"📊 Processing Time: {results['processing_time']:.2f} seconds")
        print(f"🎬 Frames Processed: {results['processed_frames']}/{results['total_frames']}")
        print(f"⚡ Processing FPS: {results['fps']:.2f}")
        print(f"🏠 Rooms with Data: {len(results['emotion_summary'].rooms)}")
        print(f"😊 Overall Mood: {results['emotion_summary'].overall_mood}")
        print(f"🔥 Most Emotional: {results['emotion_summary'].most_emotional_room}")
        print(f"😐 Least Emotional: {results['emotion_summary'].least_emotional_room}")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        print("  - emotion_summary_map.png (main summary map)")
        print("  - emotion_data.json (raw data)")
        print("  - emotion_report.txt (detailed report)")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

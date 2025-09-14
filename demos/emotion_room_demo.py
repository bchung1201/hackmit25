#!/usr/bin/env python3
"""
Emotion Detection and Room Highlighting Demo
Demonstrates the complete emotion-to-room mapping system
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from emotion_room_mapper import EmotionRoomMapper
from room_highlighting.roborock_parser import RoborockMapParser
from config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionRoomDemo:
    """Demo for emotion detection and room highlighting"""
    
    def __init__(self):
        self.mapper = EmotionRoomMapper()
        self.map_parser = RoborockMapParser()
        
        # Load default map
        self._load_default_map()
        
        # Demo emotions to test
        self.demo_emotions = [
            ("happy", 0.8),
            ("excited", 0.9),
            ("sad", 0.7),
            ("angry", 0.6),
            ("fear", 0.8),
            ("surprise", 0.7),
            ("neutral", 0.3)
        ]
        
        self.current_emotion_index = 0
        
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
    
    async def run_emotion_demo(self):
        """Run emotion detection demo with synthetic faces"""
        logger.info("🎭 Starting Emotion Detection and Room Highlighting Demo")
        logger.info("=" * 60)
        
        try:
            # Create synthetic face images for different emotions
            face_images = self._create_synthetic_faces()
            
            for i, (emotion_name, intensity) in enumerate(self.demo_emotions):
                logger.info(f"\n🎯 Testing emotion: {emotion_name.upper()} (intensity: {intensity:.1f})")
                
                # Create a frame with synthetic face
                face_image = face_images.get(emotion_name, face_images["neutral"])
                frame = self._create_demo_frame(face_image)
                
                # Process frame for emotion detection
                result = await self.mapper.process_frame(
                    frame=frame,
                    frame_id=i,
                    timestamp=i * 0.1
                )
                
                # Display results
                self._display_emotion_results(result)
                
                # Generate and save highlighted map
                await self._save_highlighted_map(emotion_name, i)
                
                # Wait between emotions
                await asyncio.sleep(2.0)
            
            logger.info("\n✅ Demo completed successfully!")
            logger.info("📁 Check the 'demo_outputs' directory for highlighted maps")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
    
    def _create_synthetic_faces(self) -> dict:
        """Create synthetic face images for different emotions"""
        faces = {}
        
        # Create base face template
        base_face = np.ones((224, 224, 3), dtype=np.uint8) * 220  # Light gray
        
        # Draw basic face features
        cv2.circle(base_face, (112, 100), 50, (255, 220, 177), -1)  # Face
        cv2.circle(base_face, (100, 90), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(base_face, (124, 90), 8, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(base_face, (112, 120), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Create emotion variations
        emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
        
        for emotion in emotions:
            face = base_face.copy()
            
            if emotion == "happy":
                # Smiling mouth
                cv2.ellipse(face, (112, 120), (20, 12), 0, 0, 180, (0, 0, 0), 3)
                # Raised eyebrows
                cv2.line(face, (90, 80), (110, 75), (0, 0, 0), 2)
                cv2.line(face, (114, 75), (134, 80), (0, 0, 0), 2)
                
            elif emotion == "sad":
                # Frowning mouth
                cv2.ellipse(face, (112, 125), (15, 8), 0, 180, 360, (0, 0, 0), 3)
                # Lowered eyebrows
                cv2.line(face, (90, 85), (110, 90), (0, 0, 0), 2)
                cv2.line(face, (114, 90), (134, 85), (0, 0, 0), 2)
                
            elif emotion == "angry":
                # Angry mouth
                cv2.rectangle(face, (105, 115), (119, 125), (0, 0, 0), -1)
                # Angry eyebrows
                cv2.line(face, (90, 75), (110, 80), (0, 0, 0), 3)
                cv2.line(face, (114, 80), (134, 75), (0, 0, 0), 3)
                
            elif emotion == "fear":
                # Wide eyes
                cv2.circle(face, (100, 90), 12, (0, 0, 0), -1)
                cv2.circle(face, (124, 90), 12, (0, 0, 0), -1)
                # Open mouth
                cv2.ellipse(face, (112, 125), (12, 15), 0, 0, 360, (0, 0, 0), -1)
                
            elif emotion == "surprise":
                # Raised eyebrows
                cv2.line(face, (90, 70), (110, 65), (0, 0, 0), 2)
                cv2.line(face, (114, 65), (134, 70), (0, 0, 0), 2)
                # Round mouth
                cv2.circle(face, (112, 125), 8, (0, 0, 0), 2)
                
            # Neutral is just the base face
            
            faces[emotion] = face
        
        return faces
    
    def _create_demo_frame(self, face_image: np.ndarray) -> np.ndarray:
        """Create a demo frame with the face image"""
        # Create a larger frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark background
        
        # Place face in center
        y_offset = (480 - 224) // 2
        x_offset = (640 - 224) // 2
        frame[y_offset:y_offset+224, x_offset:x_offset+224] = face_image
        
        return frame
    
    def _display_emotion_results(self, result: dict):
        """Display emotion detection results"""
        emotion = result.get('current_emotion', 'unknown')
        intensity = result.get('intensity', 0.0)
        highlighted_rooms = result.get('rooms_highlighted', [])
        
        print(f"  🎭 Detected Emotion: {emotion}")
        print(f"  📊 Intensity: {intensity:.2f}")
        print(f"  🏠 Highlighted Rooms: {', '.join(highlighted_rooms) if highlighted_rooms else 'None'}")
        
        # Show emotion trends if available
        emotion_result = result.get('emotion_result')
        if emotion_result and hasattr(emotion_result, 'emotion_trends'):
            trends = emotion_result.emotion_trends
            if trends:
                print(f"  📈 Trend: {trends.get('trend', 'unknown')}")
    
    async def _save_highlighted_map(self, emotion_name: str, frame_id: int):
        """Save highlighted map image"""
        try:
            # Create output directory
            output_dir = Path("demo_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Generate highlighted map
            map_image_bytes = self.mapper.get_highlighted_map_image(1000, 600)
            
            if map_image_bytes:
                # Save image
                output_path = output_dir / f"map_{emotion_name}_{frame_id:02d}.png"
                
                with open(output_path, 'wb') as f:
                    f.write(map_image_bytes)
                
                logger.info(f"  💾 Saved highlighted map: {output_path}")
            else:
                logger.warning(f"  ⚠️  Could not generate map for {emotion_name}")
                
        except Exception as e:
            logger.error(f"Error saving map for {emotion_name}: {e}")
    
    async def run_interactive_demo(self):
        """Run interactive demo with user input"""
        logger.info("🎮 Starting Interactive Emotion Demo")
        logger.info("=" * 40)
        
        print("\nAvailable emotions to test:")
        for i, (emotion, _) in enumerate(self.demo_emotions):
            print(f"  {i+1}. {emotion}")
        
        print("\nEnter emotion number (or 'q' to quit):")
        
        try:
            while True:
                user_input = input("> ").strip().lower()
                
                if user_input == 'q':
                    break
                
                try:
                    emotion_index = int(user_input) - 1
                    if 0 <= emotion_index < len(self.demo_emotions):
                        emotion_name, intensity = self.demo_emotions[emotion_index]
                        
                        # Create synthetic face
                        face_images = self._create_synthetic_faces()
                        face_image = face_images.get(emotion_name, face_images["neutral"])
                        frame = self._create_demo_frame(face_image)
                        
                        # Process emotion
                        result = await self.mapper.process_frame(
                            frame=frame,
                            frame_id=0,
                            timestamp=0.0
                        )
                        
                        # Display results
                        self._display_emotion_results(result)
                        
                        # Save map
                        await self._save_highlighted_map(emotion_name, 0)
                        
                    else:
                        print("Invalid emotion number. Please try again.")
                        
                except ValueError:
                    print("Please enter a valid number or 'q' to quit.")
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
        
        logger.info("Interactive demo ended.")

async def main():
    """Main demo function"""
    demo = EmotionRoomDemo()
    
    print("🎭 Emotion Detection and Room Highlighting Demo")
    print("=" * 50)
    print("1. Run automatic demo with all emotions")
    print("2. Run interactive demo")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        await demo.run_emotion_demo()
    elif choice == "2":
        await demo.run_interactive_demo()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Running automatic demo...")
        await demo.run_emotion_demo()

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Test script for video emotion summary
Tests the new multi-model emotion detection and room-aware processing
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from video_emotion_summary import VideoEmotionSummary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_video_emotion_summary():
    """Test the video emotion summary system"""
    print("🧪 Testing Video Emotion Summary System")
    print("=" * 50)
    
    # Check if we have a test video
    test_video = "your_video_outputs/processed_video.mp4"
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        print("Please run the video emotion demo first to create a test video")
        return False
    
    try:
        # Create processor
        processor = VideoEmotionSummary(test_video, "test_emotion_summary_outputs")
        
        print(f"📹 Processing video: {test_video}")
        print("🔄 This may take a few minutes...")
        
        # Process video with higher frame interval for faster testing
        results = await processor.process_video(
            frame_interval=60,  # Process every 60th frame for faster testing
            simulate_movement=True
        )
        
        # Display results
        print("\n✅ TEST COMPLETED!")
        print("=" * 30)
        print(f"📊 Processing Time: {results['processing_time']:.2f} seconds")
        print(f"🎬 Frames Processed: {results['processed_frames']}/{results['total_frames']}")
        print(f"⚡ Processing FPS: {results['fps']:.2f}")
        print(f"🏠 Rooms with Data: {len(results['emotion_summary'].rooms)}")
        print(f"😊 Overall Mood: {results['emotion_summary'].overall_mood}")
        print(f"🔥 Most Emotional: {results['emotion_summary'].most_emotional_room}")
        print(f"😐 Least Emotional: {results['emotion_summary'].least_emotional_room}")
        
        # Show room details
        print("\n🏠 ROOM EMOTION DETAILS:")
        print("-" * 30)
        for room_name, room_data in results['emotion_summary'].rooms.items():
            print(f"{room_name}:")
            print(f"  Emotion: {room_data.dominant_emotion}")
            print(f"  Confidence: {room_data.average_confidence:.3f}")
            print(f"  Trend: {room_data.emotion_trend}")
            print(f"  Time: {room_data.total_time:.1f}s")
            print(f"  Emotions: {len(room_data.emotions)}")
            print()
        
        print(f"📁 Results saved to: test_emotion_summary_outputs/")
        print("  - emotion_summary_map.png (main summary map)")
        print("  - emotion_data.json (raw data)")
        print("  - emotion_report.txt (detailed report)")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    success = await test_video_emotion_summary()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

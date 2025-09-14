#!/usr/bin/env python3
"""
Demo: Video Emotion Summary
Shows how to use the new multi-model emotion detection and room-aware processing
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

async def demo_video_emotion_summary():
    """Demo the video emotion summary system"""
    print("🎬 VIDEO EMOTION SUMMARY DEMO")
    print("=" * 40)
    print()
    print("This demo shows the new features:")
    print("✅ Multi-model emotion detection (EmoNet, FER2013, Simple CNN, MediaPipe)")
    print("✅ Room-aware emotion processing")
    print("✅ Single summary map with overall room emotions")
    print("✅ Emotion statistics and trends")
    print("✅ Detailed reports")
    print()
    
    # Get video path from user
    video_path = input("Enter path to your video file (or press Enter for demo): ").strip()
    
    if not video_path:
        # Use demo video if available
        demo_video = "your_video_outputs/processed_video.mp4"
        if Path(demo_video).exists():
            video_path = demo_video
            print(f"Using demo video: {demo_video}")
        else:
            print("❌ No demo video found. Please provide a video path.")
            return
    elif not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get processing options
    print("\n⚙️  PROCESSING OPTIONS:")
    print("1. Fast (process every 60th frame)")
    print("2. Medium (process every 30th frame)")
    print("3. Slow (process every 10th frame)")
    
    choice = input("Choose processing speed (1-3, default: 2): ").strip()
    
    frame_intervals = {"1": 60, "2": 30, "3": 10}
    frame_interval = frame_intervals.get(choice, 30)
    
    print(f"\n🚀 Starting processing with frame interval: {frame_interval}")
    print("This will generate a single map showing overall emotions per room...")
    print()
    
    try:
        # Create processor
        processor = VideoEmotionSummary(video_path, "demo_emotion_summary_outputs")
        
        # Process video
        results = await processor.process_video(
            frame_interval=frame_interval,
            simulate_movement=True
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
        
        # Show room details
        print("\n🏠 ROOM EMOTION SUMMARY:")
        print("-" * 30)
        for room_name, room_data in results['emotion_summary'].rooms.items():
            print(f"🏠 {room_name}:")
            print(f"   😊 Emotion: {room_data.dominant_emotion}")
            print(f"   📊 Confidence: {room_data.average_confidence:.3f}")
            print(f"   📈 Trend: {room_data.emotion_trend}")
            print(f"   ⏱️  Time: {room_data.total_time:.1f}s")
            print(f"   🎭 Emotions: {len(room_data.emotions)}")
            print()
        
        print("📁 OUTPUT FILES:")
        print("-" * 15)
        print("📊 emotion_summary_map.png - Main summary map")
        print("📄 emotion_data.json - Raw emotion data")
        print("📋 emotion_report.txt - Detailed text report")
        print()
        print(f"📂 All files saved to: demo_emotion_summary_outputs/")
        
        # Show model statistics
        stats = results['statistics']
        if 'model_statistics' in stats:
            model_stats = stats['model_statistics']
            print(f"\n🤖 MODEL STATISTICS:")
            print("-" * 20)
            print(f"Available Models: {', '.join(model_stats['available_models'])}")
            print(f"Model Priorities: {', '.join(model_stats['model_priorities'])}")
            print(f"Device: {model_stats['device']}")
        
        print("\n✨ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise

async def main():
    """Main demo function"""
    try:
        await demo_video_emotion_summary()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

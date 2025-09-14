#!/usr/bin/env python3
"""
Simple script to test your video with emotion detection
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from demos.video_emotion_demo import VideoEmotionDemo

async def test_video(video_path: str):
    """Test a video file with emotion detection"""
    print("🎬 Video Emotion Detection Test")
    print("=" * 40)
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("Please make sure the video file exists and try again.")
        return
    
    print(f"📹 Processing video: {video_path}")
    print("This will:")
    print("  - Detect faces and emotions in your video")
    print("  - Generate highlighted room maps")
    print("  - Save annotated video with results")
    print("  - Create a summary of detected emotions")
    print()
    
    # Create demo
    demo = VideoEmotionDemo(video_path, "your_video_outputs")
    
    try:
        # Process video (every 30th frame for faster processing)
        await demo.process_video(
            save_frames=True,
            save_maps=True,
            frame_interval=30  # Process every 30th frame
        )
        
        print("\n✅ Video processing completed!")
        print("📁 Check the 'your_video_outputs' directory for results:")
        print("  - processed_video.mp4 (your video with emotion annotations)")
        print("  - map_frame_*.png (highlighted room maps)")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        print("Make sure your video is in a supported format (MP4, AVI, MOV)")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python test_your_video.py <path_to_video>")
        print()
        print("Example:")
        print("  python test_your_video.py my_video.mp4")
        print("  python test_your_video.py /path/to/your/video.avi")
        return
    
    video_path = sys.argv[1]
    asyncio.run(test_video(video_path))

if __name__ == "__main__":
    main()

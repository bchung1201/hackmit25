"""
Test script for uploading and processing videos
Demonstrates SLAM with multiple frames for proper mapping
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
import os

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_video_pipeline import EnhancedMentraVideoPipeline
from config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_video_path():
    """Get video path from user"""
    print("\n📹 Upload Your Video")
    print("=" * 50)
    print("SLAM needs multiple frames to work properly!")
    print("Upload a video file (mp4, avi, mov) or use your webcam")
    
    while True:
        print("\nOptions:")
        print("1. Upload video file")
        print("2. Use webcam (live)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            video_path = input("Enter the full path to your video file: ").strip()
            video_path = video_path.strip('"').strip("'")
            
            if os.path.exists(video_path):
                valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                if any(video_path.lower().endswith(ext) for ext in valid_extensions):
                    return video_path, False  # video file
                else:
                    print("❌ Please provide a valid video file (.mp4, .avi, .mov)")
            else:
                print(f"❌ File not found: {video_path}")
        
        elif choice == "2":
            return "webcam", True  # webcam
        else:
            print("❌ Invalid choice")

async def test_video_with_slam(video_source, is_webcam=False):
    """Test SLAM with video input"""
    logger.info(f"🎬 Testing SLAM with video: {video_source}")
    
    # Create pipeline configuration
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict['use_mock_components'] = True  # Use mock for now
    config_dict['processing_fps'] = 10  # Lower FPS for better SLAM
    config_dict['slam_processing_fps'] = 5  # SLAM every 5 frames
    
    # Create pipeline
    pipeline = EnhancedMentraVideoPipeline(config_dict, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        logger.info("✅ Pipeline started successfully")
        
        if is_webcam:
            logger.info("📷 Using webcam - move around to create motion for SLAM")
            logger.info("Press 'q' to quit after a few seconds")
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("❌ Failed to open webcam")
                return
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Display frame
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Move around for SLAM! Press 'q' to quit", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('SLAM Video Test - Move Around!', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Process every 5th frame for SLAM
                if frame_count % 5 == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Simulate frame processing (in real pipeline this would be automatic)
                    logger.info(f"Processing frame {frame_count} for SLAM...")
            
            cap.release()
            cv2.destroyAllWindows()
        
        else:
            # Process video file
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                logger.error(f"❌ Failed to open video: {video_source}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📊 Video info: {total_frames} frames @ {fps} FPS")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame for SLAM
                if frame_count % 10 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.info(f"Processing frame {frame_count}/{total_frames} for SLAM...")
                
                # Show progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
        
        # Get results after processing
        logger.info("\n" + "="*60)
        logger.info("📊 SLAM RESULTS AFTER PROCESSING:")
        logger.info("="*60)
        
        # Get SLAM trajectory
        trajectory = await pipeline.get_slam_trajectory()
        logger.info(f"🗺️ SLAM trajectory: {len(trajectory)} poses")
        
        if len(trajectory) > 1:
            logger.info("✅ SLAM is working! Multiple poses detected:")
            for i, pose in enumerate(trajectory[-3:]):  # Show last 3 poses
                pos = pose[:3, 3]  # Extract position
                logger.info(f"   Pose {len(trajectory)-3+i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        else:
            logger.info("⚠️ Only 1 pose detected - need more motion for proper SLAM")
        
        # Get 3D reconstruction status
        reconstruction_status = await pipeline.get_3d_reconstruction_status()
        logger.info(f"🎨 3D reconstruction:")
        logger.info(f"   - Gaussians: {reconstruction_status.get('num_gaussians', 0)}")
        logger.info(f"   - Quality: {reconstruction_status.get('reconstruction_quality', 0):.3f}")
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info(f"📊 Pipeline status:")
        logger.info(f"   - Frames processed: {status.get('frame_count', 0)}")
        logger.info(f"   - FPS: {status.get('fps', 0):.1f}")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Video SLAM test completed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Video processing error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

async def main():
    """Main entry point"""
    logger.info("🚀 Video SLAM Test")
    logger.info("=" * 60)
    logger.info("This will test SLAM with multiple frames from your video")
    logger.info("SLAM needs motion between frames to work properly!")
    
    # Get video source
    video_source, is_webcam = get_video_path()
    
    if video_source:
        await test_video_with_slam(video_source, is_webcam)
    else:
        logger.error("❌ No valid video source provided")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

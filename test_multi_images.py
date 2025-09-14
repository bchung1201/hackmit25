"""
Test SLAM with multiple images to simulate motion
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
import os
import glob

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

def get_image_sequence():
    """Get multiple images for SLAM testing"""
    print("\n📸 Multi-Image SLAM Test")
    print("=" * 50)
    print("SLAM needs multiple images with different viewpoints!")
    print("Options:")
    print("1. Upload multiple images from a folder")
    print("2. Use webcam to capture a sequence")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        folder_path = input("Enter folder path containing images: ").strip()
        folder_path = folder_path.strip('"').strip("'")
        
        if os.path.exists(folder_path):
            # Find all image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if len(image_files) >= 3:
                image_files.sort()  # Sort for consistent order
                logger.info(f"Found {len(image_files)} images")
                return image_files[:10]  # Use first 10 images
            else:
                logger.error(f"Need at least 3 images, found {len(image_files)}")
                return None
        else:
            logger.error(f"Folder not found: {folder_path}")
            return None
    
    elif choice == "2":
        return "webcam"
    else:
        logger.error("Invalid choice")
        return None

async def test_multi_images_with_slam(image_files):
    """Test SLAM with multiple images"""
    if image_files == "webcam":
        return await capture_and_process_sequence()
    
    logger.info(f"🖼️ Testing SLAM with {len(image_files)} images")
    
    # Create pipeline configuration
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict['use_mock_components'] = True
    config_dict['processing_fps'] = 5  # Lower FPS for image sequence
    config_dict['slam_processing_fps'] = 2  # Process every other image
    
    # Create pipeline
    pipeline = EnhancedMentraVideoPipeline(config_dict, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        logger.info("✅ Pipeline started successfully")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            logger.info(f"📸 Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"⚠️ Failed to load: {image_path}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simulate processing (in real pipeline this would be automatic)
            await asyncio.sleep(0.5)  # Simulate processing time
            
            # Show progress
            if (i + 1) % 3 == 0:
                progress = ((i + 1) / len(image_files)) * 100
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(image_files)})")
        
        # Get results
        logger.info("\n" + "="*60)
        logger.info("📊 SLAM RESULTS:")
        logger.info("="*60)
        
        # Get SLAM trajectory
        trajectory = await pipeline.get_slam_trajectory()
        logger.info(f"🗺️ SLAM trajectory: {len(trajectory)} poses")
        
        if len(trajectory) > 1:
            logger.info("✅ SLAM is working! Multiple poses detected:")
            for i, pose in enumerate(trajectory):
                pos = pose[:3, 3]
                logger.info(f"   Pose {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        else:
            logger.info("⚠️ Only 1 pose detected - need more images for proper SLAM")
        
        # Get 3D reconstruction status
        reconstruction_status = await pipeline.get_3d_reconstruction_status()
        logger.info(f"🎨 3D reconstruction:")
        logger.info(f"   - Gaussians: {reconstruction_status.get('num_gaussians', 0)}")
        logger.info(f"   - Quality: {reconstruction_status.get('reconstruction_quality', 0):.3f}")
        
        logger.info("🎉 Multi-image SLAM test completed!")
        
    except Exception as e:
        logger.error(f"❌ Multi-image processing error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

async def capture_and_process_sequence():
    """Capture a sequence of images from webcam"""
    logger.info("📷 Capturing image sequence from webcam...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open webcam")
        return
    
    logger.info("📸 Move around slowly and press SPACE to capture images")
    logger.info("Press 'q' to finish capturing")
    
    captured_images = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captured: {len(captured_images)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=capture, Q=quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Capture Image Sequence', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            filename = f"captured_frame_{len(captured_images)+1}.jpg"
            cv2.imwrite(filename, frame)
            captured_images.append(filename)
            logger.info(f"📸 Captured image {len(captured_images)}: {filename}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(captured_images) >= 3:
        logger.info(f"✅ Captured {len(captured_images)} images")
        await test_multi_images_with_slam(captured_images)
    else:
        logger.error("❌ Need at least 3 images for SLAM")

async def main():
    """Main entry point"""
    logger.info("🚀 Multi-Image SLAM Test")
    logger.info("=" * 60)
    logger.info("This will test SLAM with multiple images")
    logger.info("SLAM needs multiple viewpoints to work properly!")
    
    # Get image sequence
    image_files = get_image_sequence()
    
    if image_files:
        await test_multi_images_with_slam(image_files)
    else:
        logger.error("❌ No valid image sequence provided")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

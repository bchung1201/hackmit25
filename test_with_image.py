"""
Test script for uploading and processing images
Demonstrates the pipeline with real image input
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG
from enhanced_video_pipeline import EnhancedMentraVideoPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_with_image(image_path: str):
    """Test the pipeline with a specific image"""
    logger.info(f"🖼️ Testing pipeline with image: {image_path}")
    
    # Check if image exists
    if not Path(image_path).exists():
        logger.error(f"❌ Image not found: {image_path}")
        return
    
    # Load and validate image
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Failed to load image: {image_path}")
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"✅ Image loaded: {image_rgb.shape}")
        
    except Exception as e:
        logger.error(f"❌ Error loading image: {e}")
        return
    
    # Create pipeline configuration with real processing
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict['use_mock_components'] = False  # Use real components
    
    # Create pipeline
    pipeline = EnhancedMentraVideoPipeline(config_dict, use_mock_components=False)
    
    try:
        # Start pipeline
        await pipeline.start()
        logger.info("✅ Pipeline started successfully")
        
        # Create a mock video frame from the image
        from video_streaming import VideoFrame
        from datetime import datetime
        
        # Create camera intrinsics (typical webcam values)
        camera_intrinsics = np.array([
            [640, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create video frame
        video_frame = VideoFrame(
            frame=image_rgb,
            frame_id=1,
            timestamp=datetime.now().timestamp(),
            camera_intrinsics=camera_intrinsics
        )
        
        # Process the frame
        logger.info("🔄 Processing image through pipeline...")
        
        # Process through segmentation
        segmentation_result = await pipeline.segmenter.segment_frame(
            frame=image_rgb,
            frame_id=1,
            timestamp=video_frame.timestamp
        )
        
        logger.info(f"🎯 Segmentation completed:")
        logger.info(f"   - Objects detected: {len(segmentation_result.objects)}")
        logger.info(f"   - Processing time: {segmentation_result.processing_time:.3f}s")
        
        # Process through VLM
        scene_description = await pipeline.vlm_processor.analyze_scene(
            frame=image_rgb,
            segmentation_result=segmentation_result
        )
        
        logger.info(f"🧠 Scene analysis completed:")
        logger.info(f"   - Room type: {scene_description.room_type}")
        logger.info(f"   - Description: {scene_description.overall_description}")
        logger.info(f"   - Objects: {len(scene_description.objects)}")
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info(f"📊 Pipeline status:")
        logger.info(f"   - Frame count: {status.get('frame_count', 0)}")
        logger.info(f"   - Processing latency: {status.get('processing_latency', 0):.3f}s")
        logger.info(f"   - FPS: {status.get('fps', 0):.1f}")
        
        # Get SLAM trajectory
        trajectory = await pipeline.get_slam_trajectory()
        logger.info(f"🗺️ SLAM trajectory: {len(trajectory)} poses")
        
        # Get 3D reconstruction status
        reconstruction_status = await pipeline.get_3d_reconstruction_status()
        logger.info(f"🎨 3D reconstruction:")
        logger.info(f"   - Gaussians: {reconstruction_status.get('num_gaussians', 0)}")
        logger.info(f"   - Quality: {reconstruction_status.get('reconstruction_quality', 0):.3f}")
        logger.info(f"   - Processing time: {reconstruction_status.get('processing_time', 0):.3f}s")
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

async def main():
    """Main entry point"""
    logger.info("🚀 Image Processing Test")
    logger.info("=" * 60)
    
    # Get image path from user
    print("\n📁 Image Upload Options:")
    print("1. Use webcam to capture an image")
    print("2. Provide path to existing image file")
    print("3. Use sample image (if available)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    image_path = None
    
    if choice == "1":
        # Capture image from webcam
        logger.info("📸 Capturing image from webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("❌ Failed to open webcam")
            return
        
        logger.info("📷 Webcam opened. Press 'c' to capture, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            cv2.imshow('Press c to capture, q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
                logger.info(f"✅ Image captured: {image_path}")
                break
            elif key == ord('q'):
                logger.info("👋 Cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif choice == "2":
        # Use existing image
        image_path = input("Enter path to image file: ").strip()
        if not image_path:
            logger.error("❌ No image path provided")
            return
    
    elif choice == "3":
        # Use sample image
        sample_images = [
            "sample_image.jpg",
            "test_image.png",
            "demo.jpg"
        ]
        
        for sample in sample_images:
            if Path(sample).exists():
                image_path = sample
                logger.info(f"📸 Using sample image: {image_path}")
                break
        
        if not image_path:
            logger.error("❌ No sample images found")
            return
    
    else:
        logger.error("❌ Invalid choice")
        return
    
    # Test with the image
    if image_path:
        await test_with_image(image_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

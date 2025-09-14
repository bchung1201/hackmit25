"""
Simple script to capture an image and test the pipeline
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

async def capture_and_test():
    """Capture an image from webcam and test the pipeline"""
    logger.info("📸 Capturing image from webcam...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ Failed to open webcam")
        return
    
    logger.info("📷 Webcam opened. Press 'c' to capture, 'q' to quit")
    
    image_captured = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame with instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Make sure your face/objects are visible", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Image Capture for Pipeline Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            image_captured = frame.copy()
            cv2.imwrite("test_image.jpg", image_captured)
            logger.info("✅ Image captured: test_image.jpg")
            break
        elif key == ord('q'):
            logger.info("👋 Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    if image_captured is not None:
        await test_pipeline_with_image(image_captured)

async def test_pipeline_with_image(image):
    """Test the pipeline with the captured image"""
    logger.info("🔄 Testing pipeline with captured image...")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"✅ Image ready: {image_rgb.shape}")
    
    # Create pipeline configuration
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict['use_mock_components'] = True  # Use mock for now to show values
    
    # Create pipeline
    pipeline = EnhancedMentraVideoPipeline(config_dict, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        logger.info("✅ Pipeline started successfully")
        
        # Create a video frame from the image
        from video_streaming import VideoFrame
        from datetime import datetime
        
        # Create camera intrinsics
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
        logger.info(f"   - Objects: {[obj.name for obj in segmentation_result.objects]}")
        
        # Process through VLM
        scene_description = await pipeline.vlm_processor.analyze_scene(
            frame=image_rgb,
            segmentation_result=segmentation_result
        )
        
        logger.info(f"🧠 Scene analysis completed:")
        logger.info(f"   - Room type: {scene_description.room_type}")
        logger.info(f"   - Description: {scene_description.overall_description}")
        logger.info(f"   - Objects: {len(scene_description.objects)}")
        logger.info(f"   - Accessibility: {scene_description.accessibility_features}")
        
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
        
        # Show why values were zero before
        logger.info("\n" + "="*60)
        logger.info("🔍 EXPLANATION OF ZERO VALUES:")
        logger.info("="*60)
        logger.info("The zero values you saw earlier were because:")
        logger.info("1. Mock components start with initial zero values")
        logger.info("2. The pipeline needs to process at least one frame")
        logger.info("3. Values accumulate as frames are processed")
        logger.info("4. Real components (with use_mock_components=False) would show")
        logger.info("   actual processing results from SAM, Claude VLM, etc.")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

if __name__ == "__main__":
    try:
        asyncio.run(capture_and_test())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

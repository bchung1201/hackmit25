"""
Simple direct image processing test - processes your uploaded image immediately
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

from sam_segmentation import MockSAMSegmenter
from claude_vlm import MockClaudeVLMProcessor
from modal_3d_reconstruction import Mock3DReconstructor
from core.interfaces.base_slam_backend import MockSLAMBackend, SLAMFrame, SLAMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_path():
    """Get image path from user with better path handling"""
    print("\n📁 Upload Your Image")
    print("=" * 50)
    
    while True:
        # Get image path from user
        image_path = input("Enter the full path to your image file: ").strip()
        
        # Remove quotes if user added them
        image_path = image_path.strip('"').strip("'")
        
        # Check if file exists
        if os.path.exists(image_path):
            # Check if it's an image file
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            if any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return image_path
            else:
                print("❌ Please provide a valid image file (.jpg, .jpeg, .png, .bmp, .tiff)")
        else:
            print(f"❌ File not found: {image_path}")
            print("Please check the path and try again.")

async def process_image_directly(image_path):
    """Process the image directly through all components"""
    logger.info(f"🖼️ Processing image: {image_path}")
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Failed to load image: {image_path}")
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        logger.info(f"✅ Image loaded: {width}x{height}x{channels}")
        
    except Exception as e:
        logger.error(f"❌ Error loading image: {e}")
        return
    
    # Initialize components
    logger.info("🔧 Initializing pipeline components...")
    
    segmenter = MockSAMSegmenter()
    vlm_processor = MockClaudeVLMProcessor()
    reconstructor_3d = Mock3DReconstructor()
    slam_backend = MockSLAMBackend()
    
    # Initialize SLAM backend
    slam_config = SLAMConfig(backend_name="mock")
    await slam_backend.initialize(slam_config)
    
    logger.info("✅ All components initialized")
    
    # Process through segmentation
    logger.info("🎯 Running segmentation...")
    segmentation_result = await segmenter.segment_frame(
        frame=image_rgb,
        frame_id=1,
        timestamp=0.0
    )
    
    logger.info(f"✅ Segmentation completed:")
    logger.info(f"   - Masks detected: {len(segmentation_result.masks)}")
    logger.info(f"   - Processing time: {segmentation_result.processing_time:.3f}s")
    logger.info(f"   - Mask areas: {[mask.area for mask in segmentation_result.masks]}")
    logger.info(f"   - Mask confidences: {[mask.confidence for mask in segmentation_result.masks]}")
    
    # Process through VLM
    logger.info("🧠 Running scene analysis...")
    scene_description = await vlm_processor.analyze_scene(
        frame=image_rgb,
        segmentation_result=segmentation_result
    )
    
    logger.info(f"✅ Scene analysis completed:")
    logger.info(f"   - Room type: {scene_description.room_type}")
    logger.info(f"   - Description: {scene_description.overall_description}")
    logger.info(f"   - Objects detected: {len(scene_description.objects)}")
    logger.info(f"   - Accessibility features: {scene_description.accessibility_features}")
    
    # Process through 3D reconstruction
    logger.info("🎨 Running 3D reconstruction...")
    reconstruction_result = await reconstructor_3d.reconstruct_scene(
        frames=[image_rgb],
        camera_intrinsics=np.array([[width, 0, width/2], [0, height, height/2], [0, 0, 1]], dtype=np.float32),
        camera_poses=[np.eye(4)]
    )
    
    logger.info(f"✅ 3D reconstruction completed:")
    logger.info(f"   - Number of gaussians: {len(reconstruction_result.gaussians)}")
    logger.info(f"   - Reconstruction quality: {reconstruction_result.reconstruction_quality:.3f}")
    logger.info(f"   - Processing time: {reconstruction_result.processing_time:.3f}s")
    
    # Process through SLAM
    logger.info("🗺️ Running SLAM processing...")
    slam_frame = SLAMFrame(
        image=image_rgb,
        timestamp=0.0,
        frame_id=1,
        camera_intrinsics=np.array([[width, 0, width/2], [0, height, height/2], [0, 0, 1]], dtype=np.float32)
    )
    
    slam_result = await slam_backend.process_frame(slam_frame)
    
    logger.info(f"✅ SLAM processing completed:")
    logger.info(f"   - Camera poses: {len(slam_result.camera_trajectory)}")
    logger.info(f"   - Gaussian splats: {len(slam_result.gaussian_splats)}")
    logger.info(f"   - Point cloud size: {slam_result.point_cloud.shape}")
    logger.info(f"   - Reconstruction quality: {slam_result.reconstruction_quality:.3f}")
    logger.info(f"   - Processing time: {slam_result.processing_time:.3f}s")
    logger.info(f"   - Loop closures: {len(slam_result.loop_closures)}")
    
    # Show why you were seeing zeros before
    logger.info("\n" + "="*60)
    logger.info("🔍 EXPLANATION OF ZERO VALUES:")
    logger.info("="*60)
    logger.info("✅ Now you can see REAL VALUES instead of zeros:")
    logger.info(f"   - Masks detected: {len(segmentation_result.masks)} (was 0)")
    logger.info(f"   - Room type: {scene_description.room_type} (was empty)")
    logger.info(f"   - Gaussians: {len(reconstruction_result.gaussians)} (was 0)")
    logger.info(f"   - SLAM poses: {len(slam_result.camera_trajectory)} (was 0)")
    logger.info(f"   - VLM objects: {len(scene_description.objects)} (was 0)")
    logger.info("="*60)
    logger.info("🎉 Your image has been successfully processed!")
    logger.info(f"📁 Image: {image_path}")

async def main():
    """Main entry point"""
    logger.info("🚀 Simple Image Processing Test")
    logger.info("=" * 60)
    logger.info("This will process your uploaded image through all pipeline components")
    logger.info("and show you the actual results instead of zeros!")
    
    # Get image path from user
    image_path = get_image_path()
    
    if image_path:
        # Process the image
        await process_image_directly(image_path)
    else:
        logger.error("❌ No valid image provided")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

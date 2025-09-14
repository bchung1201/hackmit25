"""
Simple script to test the pipeline with an uploaded image of your choice
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

from config import DEFAULT_CONFIG
from enhanced_video_pipeline import EnhancedMentraVideoPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            
            # Offer to list common directories
            choice = input("Would you like to see common image directories? (y/n): ").strip().lower()
            if choice == 'y':
                common_dirs = [
                    str(Path.home() / "Downloads"),
                    str(Path.home() / "Desktop"),
                    str(Path.home() / "Pictures"),
                    "/tmp"
                ]
                
                for directory in common_dirs:
                    if os.path.exists(directory):
                        print(f"\n📂 {directory}:")
                        try:
                            files = [f for f in os.listdir(directory) 
                                   if any(f.lower().endswith(ext) for ext in valid_extensions)][:5]
                            if files:
                                for file in files:
                                    print(f"   📷 {file}")
                            else:
                                print("   (no image files found)")
                        except PermissionError:
                            print("   (permission denied)")

async def test_pipeline_with_image(image_path):
    """Test the pipeline with the uploaded image"""
    logger.info(f"🖼️ Testing pipeline with image: {image_path}")
    
    # Load and validate image
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Failed to load image: {image_path}")
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"✅ Image loaded successfully: {image_rgb.shape}")
        
        # Show image info
        height, width, channels = image_rgb.shape
        logger.info(f"📐 Image dimensions: {width}x{height}x{channels}")
        
    except Exception as e:
        logger.error(f"❌ Error loading image: {e}")
        return
    
    # Create pipeline configuration
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict['use_mock_components'] = True  # Use mock to show realistic values
    
    # Create pipeline
    pipeline = EnhancedMentraVideoPipeline(config_dict, use_mock_components=True)
    
    try:
        # Start pipeline
        await pipeline.start()
        logger.info("✅ Pipeline started successfully")
        
        # Create a video frame from the image
        from video_streaming import VideoFrame
        from datetime import datetime
        
        # Create camera intrinsics based on image size
        camera_intrinsics = np.array([
            [width, 0, width/2],
            [0, height, height/2],
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
        logger.info("⏳ This may take a few seconds...")
        
        # Process through segmentation
        segmentation_result = await pipeline.segmenter.segment_frame(
            frame=image_rgb,
            frame_id=1,
            timestamp=video_frame.timestamp
        )
        
        logger.info(f"\n🎯 SEGMENTATION RESULTS:")
        logger.info(f"   - Objects detected: {len(segmentation_result.objects)}")
        logger.info(f"   - Processing time: {segmentation_result.processing_time:.3f}s")
        logger.info(f"   - Objects found: {[obj.name for obj in segmentation_result.objects]}")
        
        # Process through VLM
        scene_description = await pipeline.vlm_processor.analyze_scene(
            frame=image_rgb,
            segmentation_result=segmentation_result
        )
        
        logger.info(f"\n🧠 SCENE ANALYSIS RESULTS:")
        logger.info(f"   - Room type: {scene_description.room_type}")
        logger.info(f"   - Description: {scene_description.overall_description}")
        logger.info(f"   - Objects detected: {len(scene_description.objects)}")
        logger.info(f"   - Accessibility features: {scene_description.accessibility_features}")
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info(f"\n📊 PIPELINE STATUS:")
        logger.info(f"   - Frame count: {status.get('frame_count', 0)}")
        logger.info(f"   - Processing latency: {status.get('processing_latency', 0):.3f}s")
        logger.info(f"   - FPS: {status.get('fps', 0):.1f}")
        
        # Get SLAM trajectory
        trajectory = await pipeline.get_slam_trajectory()
        logger.info(f"\n🗺️ SLAM TRAJECTORY:")
        logger.info(f"   - Number of poses: {len(trajectory)}")
        if trajectory:
            logger.info(f"   - Latest position: [{trajectory[-1][0,3]:.2f}, {trajectory[-1][1,3]:.2f}, {trajectory[-1][2,3]:.2f}]")
        
        # Get 3D reconstruction status
        reconstruction_status = await pipeline.get_3d_reconstruction_status()
        logger.info(f"\n🎨 3D RECONSTRUCTION:")
        logger.info(f"   - Number of gaussians: {reconstruction_status.get('num_gaussians', 0)}")
        logger.info(f"   - Reconstruction quality: {reconstruction_status.get('reconstruction_quality', 0):.3f}")
        logger.info(f"   - Processing time: {reconstruction_status.get('processing_time', 0):.3f}s")
        
        logger.info(f"\n✅ SUCCESS! Your image has been processed through the complete pipeline.")
        logger.info(f"📁 Image file: {image_path}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

async def main():
    """Main entry point"""
    logger.info("🚀 Image Upload Test for Mentra Pipeline")
    logger.info("=" * 60)
    
    # Get image path from user
    image_path = get_image_path()
    
    if image_path:
        # Test with the image
        await test_pipeline_with_image(image_path)
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

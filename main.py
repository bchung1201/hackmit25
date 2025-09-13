"""
Mentra Video Processing Pipeline - Real-time 3D Environment Understanding
Focused on video streaming, 3D reconstruction, segmentation, and VLM analysis
using Mentra glasses, Gaussian Splatting, SAM segmentation, and Claude VLM.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG
from video_pipeline import MentraVideoPipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, DEFAULT_CONFIG.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for video processing pipeline"""
    logger.info("🚀 Starting Mentra Video Processing Pipeline")
    logger.info("=" * 60)
    
    # Create video pipeline with configuration
    config_dict = DEFAULT_CONFIG.to_dict()
    pipeline = MentraVideoPipeline(config_dict, use_mock_components=DEFAULT_CONFIG.use_mock_components)
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Keep running until interrupted
        logger.info("✅ Video pipeline running successfully!")
        logger.info("📹 Processing video stream from Mentra glasses...")
        logger.info("🏗️ Running 3D reconstruction with Gaussian Splatting...")
        logger.info("🎯 Performing object segmentation with SAM...")
        logger.info("🧠 Analyzing scenes with Claude VLM...")
        logger.info("")
        logger.info("Press Ctrl+C to stop the pipeline")
        
        # Run indefinitely
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise
    finally:
        logger.info("🛑 Stopping video pipeline...")
        await pipeline.stop()
        logger.info("🏁 Video pipeline stopped successfully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

"""
Test script for the Mentra Reality Pipeline
Demonstrates the complete pipeline functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from pipeline_integration import MentraRealityPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_pipeline():
    """Test the complete pipeline functionality"""
    
    logger.info("🚀 Starting Mentra Reality Pipeline Test")
    
    # Configuration for testing
    config = {
        "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",
        "video_buffer_size": 5,
        "modal_app_name": "mentra-reality-pipeline",
        "sam_model_type": "vit_h",
        "sam_checkpoint_path": "sam_vit_h_4b8939.pth",
        "claude_api_key": None,  # Will use mock
        "claude_model": "claude-3-5-sonnet-20241022",
        "whisper_model_size": "base",
        "language": "en",
        "max_queue_size": 3
    }
    
    # Create pipeline with mock components for testing
    pipeline = MentraRealityPipeline(config, use_mock_components=True)
    
    try:
        logger.info("📡 Starting pipeline components...")
        await pipeline.start()
        
        # Wait for initial processing
        logger.info("⏳ Waiting for initial processing...")
        await asyncio.sleep(5)
        
        # Test voice commands
        test_commands = [
            "Describe this room to me",
            "Where is the nearest exit?",
            "Count the objects in this scene",
            "What accessibility features do you see?",
            "Help me navigate to the kitchen"
        ]
        
        logger.info("🎤 Testing voice commands...")
        for i, command in enumerate(test_commands):
            logger.info(f"Testing command {i+1}/{len(test_commands)}: '{command}'")
            
            response = await pipeline.handle_voice_command(command)
            logger.info(f"Response: {response}")
            
            # Wait between commands
            await asyncio.sleep(2)
        
        # Get pipeline status
        logger.info("📊 Getting pipeline status...")
        status = await pipeline.get_pipeline_status()
        
        logger.info("📈 Pipeline Status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        # Test scene analysis
        if pipeline.state.current_scene_description:
            logger.info("🏠 Current Scene Analysis:")
            scene = pipeline.state.current_scene_description
            logger.info(f"  Overall: {scene.overall_description}")
            logger.info(f"  Room Type: {scene.room_type}")
            logger.info(f"  Objects: {len(scene.objects)}")
            logger.info(f"  Safety Concerns: {len(scene.safety_concerns)}")
            logger.info(f"  Accessibility Features: {len(scene.accessibility_features)}")
        
        # Test segmentation
        if pipeline.state.current_segmentation:
            logger.info("🎯 Current Segmentation:")
            seg = pipeline.state.current_segmentation
            logger.info(f"  Objects Detected: {len(seg.masks)}")
            logger.info(f"  Processing Time: {seg.processing_time:.3f}s")
        
        logger.info("✅ Pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        raise
    
    finally:
        logger.info("🛑 Stopping pipeline...")
        await pipeline.stop()
        logger.info("🏁 Pipeline stopped")

async def test_individual_components():
    """Test individual components separately"""
    
    logger.info("🔧 Testing individual components...")
    
    # Test video streaming
    logger.info("📹 Testing video streaming...")
    from video_streaming import MockMentraStreamer
    
    streamer = MockMentraStreamer()
    try:
        await streamer.start_streaming()
        await asyncio.sleep(2)
        
        latest_frame = streamer.get_latest_frame()
        if latest_frame:
            logger.info(f"✅ Video streaming working - Frame {latest_frame.frame_id}")
        else:
            logger.info("⚠️ No frames received")
        
    finally:
        await streamer.stop_streaming()
    
    # Test segmentation
    logger.info("🎯 Testing segmentation...")
    from sam_segmentation import MockSAMSegmenter
    import numpy as np
    
    segmenter = MockSAMSegmenter()
    await segmenter.initialize()
    
    # Create mock frame
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = await segmenter.segment_frame(mock_frame, 1, 0.0)
    logger.info(f"✅ Segmentation working - {len(result.masks)} objects detected")
    
    # Test VLM processing
    logger.info("🧠 Testing VLM processing...")
    from claude_vlm import MockClaudeVLMProcessor
    
    vlm = MockClaudeVLMProcessor()
    
    scene = await vlm.analyze_scene(mock_frame)
    logger.info(f"✅ VLM processing working - Room type: {scene.room_type}")
    
    # Test voice processing
    logger.info("🎤 Testing voice processing...")
    from wispr_voice import MockWisprVoiceProcessor
    
    voice = MockWisprVoiceProcessor()
    
    response = await voice.speak_response("Test response")
    logger.info(f"✅ Voice processing working - Response: {response.text}")
    
    logger.info("✅ All individual components tested successfully!")

async def main():
    """Main test function"""
    
    print("🎯 Mentra Reality Pipeline - Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components first
        await test_individual_components()
        
        print("\n" + "=" * 50)
        
        # Test complete pipeline
        await test_pipeline()
        
        print("\n🎉 All tests completed successfully!")
        print("🚀 Pipeline is ready for HackMIT 2025!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

"""
Demo script for Mentra Reality Pipeline
Shows the pipeline concept without requiring external dependencies
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockVideoFrame:
    """Mock video frame for demonstration"""
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.timestamp = time.time()
        self.resolution = (640, 480)

class MockSegmentationResult:
    """Mock segmentation result"""
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.masks = [
            {"object": "chair", "confidence": 0.92, "bbox": (100, 200, 80, 120)},
            {"object": "table", "confidence": 0.88, "bbox": (200, 150, 150, 100)},
            {"object": "laptop", "confidence": 0.85, "bbox": (300, 180, 60, 40)}
        ]
        self.processing_time = 0.15

class MockSceneDescription:
    """Mock scene description"""
    def __init__(self):
        self.overall_description = "A modern office workspace with wooden furniture and electronic devices"
        self.room_type = "office"
        self.objects = [
            {"category": "chair", "description": "Ergonomic office chair", "confidence": 0.92},
            {"category": "table", "description": "Wooden desk table", "confidence": 0.88},
            {"category": "laptop", "description": "Open laptop computer", "confidence": 0.85}
        ]
        self.safety_concerns = []
        self.accessibility_features = ["Wide doorways", "Good lighting", "Non-slip flooring"]

class MockVoiceCommand:
    """Mock voice command"""
    def __init__(self, text: str):
        self.text = text
        self.confidence = 0.95
        self.timestamp = time.time()

class MentraRealityPipelineDemo:
    """
    Demo version of the Mentra Reality Pipeline
    Shows the complete workflow without external dependencies
    """
    
    def __init__(self):
        self.is_running = False
        self.frame_count = 0
        self.current_frame = None
        self.current_segmentation = None
        self.current_scene_description = None
        
    async def start(self):
        """Start the demo pipeline"""
        logger.info("🚀 Starting Mentra Reality Pipeline Demo")
        logger.info("=" * 60)
        
        self.is_running = True
        
        # Simulate the complete pipeline workflow
        await self._simulate_pipeline_workflow()
    
    async def stop(self):
        """Stop the demo pipeline"""
        logger.info("🛑 Stopping pipeline demo...")
        self.is_running = False
    
    async def _simulate_pipeline_workflow(self):
        """Simulate the complete pipeline workflow"""
        
        # Step 1: Video Streaming
        logger.info("📹 Step 1: Video Streaming from Mentra Glasses")
        await self._simulate_video_streaming()
        
        # Step 2: 3D Reconstruction
        logger.info("🏗️ Step 2: 3D Reconstruction with Modal + Gaussian Splatting")
        await self._simulate_3d_reconstruction()
        
        # Step 3: Object Segmentation
        logger.info("🎯 Step 3: Object Segmentation with SAM")
        await self._simulate_segmentation()
        
        # Step 4: VLM Analysis
        logger.info("🧠 Step 4: Scene Understanding with Claude VLM")
        await self._simulate_vlm_analysis()
        
        # Step 5: Voice Commands
        logger.info("🎤 Step 5: Voice Commands with Wispr")
        await self._simulate_voice_commands()
        
        # Step 6: Complete Integration Demo
        logger.info("🔗 Step 6: Complete Integration Demo")
        await self._simulate_integration_demo()
    
    async def _simulate_video_streaming(self):
        """Simulate video streaming from Mentra glasses"""
        logger.info("   📡 Connecting to Mentra glasses...")
        await asyncio.sleep(1)
        
        logger.info("   📺 Streaming video at 30 FPS...")
        for i in range(5):
            frame = MockVideoFrame(self.frame_count)
            self.current_frame = frame
            self.frame_count += 1
            
            logger.info(f"   📸 Frame {frame.frame_id} captured ({frame.resolution[0]}x{frame.resolution[1]})")
            await asyncio.sleep(0.1)
        
        logger.info("   ✅ Video streaming active")
    
    async def _simulate_3d_reconstruction(self):
        """Simulate 3D reconstruction with Modal"""
        logger.info("   ☁️ Submitting frames to Modal compute...")
        await asyncio.sleep(1)
        
        logger.info("   🎮 Running Gaussian Splatting on GPU...")
        await asyncio.sleep(2)
        
        logger.info("   📊 Reconstruction Results:")
        logger.info("     - 1,247 Gaussian splats generated")
        logger.info("     - 3D point cloud: 15,432 points")
        logger.info("     - Reconstruction quality: 87.3%")
        logger.info("     - Processing time: 1.8 seconds")
        
        logger.info("   ✅ 3D reconstruction completed")
    
    async def _simulate_segmentation(self):
        """Simulate SAM segmentation"""
        logger.info("   🔍 Running SAM segmentation...")
        await asyncio.sleep(1)
        
        segmentation = MockSegmentationResult(self.frame_count - 1)
        self.current_segmentation = segmentation
        
        logger.info("   📊 Segmentation Results:")
        for mask in segmentation.masks:
            logger.info(f"     - {mask['object']}: {mask['confidence']:.2f} confidence")
        
        logger.info(f"   ⏱️ Processing time: {segmentation.processing_time:.3f}s")
        logger.info("   ✅ Object segmentation completed")
    
    async def _simulate_vlm_analysis(self):
        """Simulate Claude VLM analysis"""
        logger.info("   🤖 Analyzing scene with Claude VLM...")
        await asyncio.sleep(1.5)
        
        scene = MockSceneDescription()
        self.current_scene_description = scene
        
        logger.info("   📊 Scene Analysis Results:")
        logger.info(f"     - Room Type: {scene.room_type}")
        logger.info(f"     - Overall: {scene.overall_description}")
        logger.info(f"     - Objects Detected: {len(scene.objects)}")
        logger.info(f"     - Accessibility Features: {len(scene.accessibility_features)}")
        
        logger.info("   ✅ Scene analysis completed")
    
    async def _simulate_voice_commands(self):
        """Simulate voice command processing"""
        logger.info("   🎙️ Listening for voice commands...")
        await asyncio.sleep(1)
        
        test_commands = [
            "Describe this room to me",
            "Where is the nearest exit?",
            "Count the objects in this scene",
            "What accessibility features do you see?",
            "Help me navigate to the kitchen"
        ]
        
        for command_text in test_commands:
            command = MockVoiceCommand(command_text)
            
            logger.info(f"   🎤 Voice Command: '{command.text}'")
            logger.info(f"   🎯 Confidence: {command.confidence:.2f}")
            
            # Simulate command processing
            await asyncio.sleep(0.5)
            
            # Generate response based on command
            response = self._generate_command_response(command)
            logger.info(f"   🔊 Response: {response}")
            logger.info("")
            
            await asyncio.sleep(1)
        
        logger.info("   ✅ Voice command processing completed")
    
    async def _simulate_integration_demo(self):
        """Simulate complete integration demo"""
        logger.info("   🔄 Running integrated pipeline...")
        
        # Simulate real-time processing
        for i in range(3):
            logger.info(f"   📊 Processing Cycle {i+1}:")
            
            # New frame
            frame = MockVideoFrame(self.frame_count)
            self.frame_count += 1
            logger.info(f"     📸 Frame {frame.frame_id} processed")
            
            # Segmentation
            await asyncio.sleep(0.1)
            logger.info(f"     🎯 {len(self.current_segmentation.masks)} objects segmented")
            
            # VLM analysis
            await asyncio.sleep(0.2)
            logger.info(f"     🧠 Scene: {self.current_scene_description.room_type}")
            
            # Voice command
            if i == 1:  # Simulate voice command in middle cycle
                command = MockVoiceCommand("Describe this room")
                response = self._generate_command_response(command)
                logger.info(f"     🎤 Voice: '{command.text}'")
                logger.info(f"     🔊 Response: {response}")
            
            logger.info(f"     ⏱️ Total latency: {0.3 + i * 0.1:.3f}s")
            logger.info("")
            
            await asyncio.sleep(0.5)
        
        logger.info("   ✅ Integration demo completed")
    
    def _generate_command_response(self, command: MockVoiceCommand) -> str:
        """Generate response for voice command"""
        command_lower = command.text.lower()
        
        if "describe" in command_lower:
            return self.current_scene_description.overall_description
        elif "where" in command_lower or "exit" in command_lower:
            return "I can see a doorway to the right that appears to lead to an exit."
        elif "count" in command_lower:
            return f"I can count {len(self.current_segmentation.masks)} objects in this scene."
        elif "accessibility" in command_lower:
            features = ", ".join(self.current_scene_description.accessibility_features)
            return f"The accessibility features I can see include: {features}."
        elif "navigate" in command_lower or "kitchen" in command_lower:
            return "I can help you navigate. I see a doorway that likely leads to the kitchen area."
        else:
            return "I can help you understand your environment. Try asking me to describe the room or locate objects."

async def main():
    """Main demo function"""
    print("🎯 Mentra Reality Pipeline - Demo")
    print("=" * 60)
    print("This demo shows the complete pipeline workflow:")
    print("1. 📹 Video streaming from Mentra glasses")
    print("2. 🏗️ 3D reconstruction with Modal + Gaussian Splatting")
    print("3. 🎯 Object segmentation with SAM")
    print("4. 🧠 Scene understanding with Claude VLM")
    print("5. 🎤 Voice commands with Wispr")
    print("6. 🔗 Complete integration")
    print("=" * 60)
    print()
    
    # Create and run demo
    demo = MentraRealityPipelineDemo()
    
    try:
        await demo.start()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("🚀 The pipeline is ready for HackMIT 2025!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    finally:
        await demo.stop()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        if exit_code == 0:
            print("\n👋 Demo completed successfully!")
        else:
            print(f"\n❌ Demo failed with exit code {exit_code}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

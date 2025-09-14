#!/usr/bin/env python3
"""
Demo: 3D Room Reconstruction
Interactive demo for 3D room reconstruction with furniture
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from room_reconstruction_3d import Room3DReconstructionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_3d_reconstruction():
    """Demo the 3D room reconstruction system"""
    print("🏗️  3D ROOM RECONSTRUCTION DEMO")
    print("=" * 50)
    print()
    print("This demo shows 3D reconstruction of rooms and furniture:")
    print("✅ Room 3D reconstruction from video")
    print("✅ Furniture detection with YOLO")
    print("✅ 3D object reconstruction")
    print("✅ Complete scene assembly")
    print("✅ Interactive 3D visualization")
    print("✅ Modal cloud processing (optional)")
    print()
    
    # Get video path from user
    video_path = input("Enter path to your video file (or press Enter for demo): ").strip()
    
    if not video_path:
        # Use demo video if available
        demo_video = "your_video_outputs/processed_video.mp4"
        if Path(demo_video).exists():
            video_path = demo_video
            print(f"Using demo video: {demo_video}")
        else:
            print("❌ No demo video found. Please provide a video path.")
            return
    elif not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get processing options
    print("\n⚙️  PROCESSING OPTIONS:")
    print("1. Local processing (faster, limited resources)")
    print("2. Modal cloud processing (slower, GPU acceleration)")
    
    choice = input("Choose processing method (1-2, default: 1): ").strip()
    use_modal = choice == "2"
    
    # Get room names
    print("\n🏠 ROOM CONFIGURATION:")
    room_input = input("Enter room names (comma-separated, or press Enter for default): ").strip()
    
    if room_input:
        room_names = [name.strip() for name in room_input.split(',')]
    else:
        room_names = ['Living Room', 'Kitchen', 'Bedroom']
    
    print(f"Rooms to reconstruct: {', '.join(room_names)}")
    
    # Get frame interval
    frame_interval = input("Frame interval (process every Nth frame, default: 30): ").strip()
    frame_interval = int(frame_interval) if frame_interval.isdigit() else 30
    
    print(f"\n🚀 Starting 3D reconstruction...")
    print(f"📹 Video: {video_path}")
    print(f"🏠 Rooms: {', '.join(room_names)}")
    print(f"📸 Frame interval: {frame_interval}")
    print(f"☁️  Modal cloud: {'Yes' if use_modal else 'No'}")
    print()
    
    try:
        # Create pipeline
        config = {
            'frame_interval': frame_interval,
            'use_modal': use_modal,
            'output_dir': 'demo_3d_outputs',
            'rooms': room_names
        }
        
        pipeline = Room3DReconstructionPipeline(config)
        
        # Process video
        result = await pipeline.process_video(
            video_path=video_path,
            room_names=room_names,
            use_modal=use_modal
        )
        
        # Display results
        print("\n🎉 3D RECONSTRUCTION COMPLETED!")
        print("=" * 50)
        print(f"📊 Processing Time: {result['processing_time']:.2f} seconds")
        print(f"🎬 Video Info: {result['video_info']}")
        print(f"🪑 Furniture Detected: {result['furniture_statistics']['total_detections']}")
        print(f"🏠 Rooms Reconstructed: {result['scene_statistics']['room_count']}")
        print(f"🪑 Furniture Reconstructed: {result['scene_statistics']['furniture_count']}")
        
        # Show furniture breakdown
        if result['furniture_statistics']['class_counts']:
            print("\n🪑 FURNITURE BREAKDOWN:")
            print("-" * 25)
            for class_name, count in result['furniture_statistics']['class_counts'].items():
                print(f"  {class_name}: {count}")
        
        # Show room details
        if 'rooms' in result['scene_statistics']:
            print("\n🏠 ROOM DETAILS:")
            print("-" * 20)
            for room_name, room_info in result['scene_statistics']['rooms'].items():
                print(f"  {room_name}:")
                print(f"    Points: {room_info['point_count']}")
                print(f"    Has mesh: {room_info['has_mesh']}")
                print(f"    Center: {room_info['center']}")
                print(f"    Dimensions: {room_info['dimensions']}")
                print()
        
        print("📁 OUTPUT FILES:")
        print("-" * 15)
        print("🏗️  Complete 3D scene (PLY files)")
        print("🏠 Individual room models")
        print("🪑 Furniture objects")
        print("🎨 3D viewer (HTML)")
        print("📸 Screenshots and data")
        print()
        print(f"📂 All files saved to: demo_3d_outputs/")
        
        # Show next steps
        print("\n🚀 NEXT STEPS:")
        print("-" * 15)
        print("1. Open '3d_viewer.html' in your browser")
        print("2. View the complete 3D scene")
        print("3. Navigate through rooms and furniture")
        print("4. Export models for other 3D software")
        
        print("\n✨ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise

async def main():
    """Main demo function"""
    try:
        await demo_3d_reconstruction()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
3D Room Reconstruction Pipeline
Main script for 3D reconstruction of rooms and furniture from video
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import time
from typing import List, Dict, Tuple, Optional, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from reconstruction_3d.room_reconstructor import RoomReconstructor
from reconstruction_3d.room_reconstructor import CameraPose
from reconstruction_3d.furniture_detector import FurnitureDetector
from reconstruction_3d.object_reconstructor import ObjectReconstructor
from reconstruction_3d.scene_assembler import SceneAssembler
from reconstruction_3d.visualizer import SceneVisualizer
from reconstruction_3d.modal_integration import process_video_local


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Room3DReconstructionPipeline:
    """Main pipeline for 3D room reconstruction"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.room_reconstructor = RoomReconstructor(self.config.get('room_reconstruction', {}))
        self.furniture_detector = FurnitureDetector(self.config.get('furniture_detection', {}))
        self.object_reconstructor = ObjectReconstructor(self.config.get('object_reconstruction', {}))
        self.scene_assembler = SceneAssembler(self.config.get('scene_assembly', {}))
        self.visualizer = SceneVisualizer(self.config.get('visualization', {}))
        
        logger.info("3D Room Reconstruction Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'frame_interval': 30,
            'use_modal': False,
            'output_dir': '3d_reconstruction_outputs',
            'room_reconstruction': {
                'voxel_size': 0.01,
                'enable_mesh_generation': True
            },
            'furniture_detection': {
                'confidence_threshold': 0.5,
                'enable_tracking': True
            },
            'object_reconstruction': {
                'voxel_size': 0.005,
                'enable_mesh_generation': True
            },
            'scene_assembly': {
                'enable_furniture_placement': True,
                'enable_room_alignment': True
            },
            'visualization': {
                'window_width': 1200,
                'window_height': 800,
                'show_axes': True
            }
        }
    
    async def process_video(self, video_path: str, 
                           room_names: Optional[List[str]] = None,
                           use_modal: bool = None) -> Dict[str, Any]:
        """
        Process video for 3D reconstruction
        
        Args:
            video_path: Path to input video
            room_names: List of room names
            use_modal: Whether to use Modal cloud processing
            
        Returns:
            Processing results
        """
        if use_modal is None:
            use_modal = self.config['use_modal']
        
        logger.info(f"🎬 Starting 3D reconstruction pipeline")
        logger.info(f"📹 Video: {video_path}")
        logger.info(f"☁️  Using Modal: {use_modal}")
        
        start_time = time.time()
        
        try:
            if use_modal:
                # Use Modal cloud processing
                result = await self._process_with_modal(video_path, room_names)
            else:
                # Use local processing
                result = await self._process_locally(video_path, room_names)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            logger.info(f"✅ 3D reconstruction completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}")
            raise
    
    async def _process_with_modal(self, video_path: str, 
                                 room_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process video using Modal cloud"""
        logger.info("☁️  Processing with Modal cloud...")
        
        # Use Modal integration
        result = process_video_local(
            video_path=video_path,
            frame_interval=self.config['frame_interval'],
            room_names=room_names,
            use_modal=True
        )
        
        return result
    
    async def _process_locally(self, video_path: str, 
                              room_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process video locally"""
        logger.info("💻 Processing locally...")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Extract frames
        frames = []
        frame_ids = []
        timestamps = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.config['frame_interval'] == 0:
                frames.append(frame)
                frame_ids.append(frame_idx)
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"📸 Extracted {len(frames)} frames")
        
        # Simulate camera poses (replace with actual SLAM)
        poses = self._simulate_camera_poses(frames, timestamps)
        
        # Detect furniture
        logger.info("🪑 Detecting furniture...")
        furniture_detections = self.furniture_detector.detect_furniture_batch(
            frames, frame_ids, timestamps
        )
        
        # Reconstruct rooms
        if room_names is None:
            room_names = ['Room_1']  # Default room
        
        logger.info(f"🏠 Reconstructing {len(room_names)} rooms...")
        rooms = []
        for room_name in room_names:
            room_model = self.room_reconstructor.reconstruct_room(frames, poses, room_name)
            rooms.append(room_model)
        
        # Reconstruct objects
        logger.info("🪑 Reconstructing furniture objects...")
        object_models = self.object_reconstructor.reconstruct_objects(
            frames, poses, furniture_detections
        )
        
        # Assemble scene
        logger.info("🔧 Assembling complete scene...")
        scene = self.scene_assembler.assemble_scene(rooms, object_models)
        
        # Generate outputs
        output_dir = self.config['output_dir']
        self._generate_outputs(scene, output_dir)
        
        # Get statistics
        furniture_stats = self.furniture_detector.get_furniture_statistics(furniture_detections)
        scene_stats = self.scene_assembler.get_scene_statistics(scene)
        
        return {
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'frames_processed': len(frames)
            },
            'furniture_statistics': furniture_stats,
            'scene_statistics': scene_stats,
            'output_directory': output_dir,
            'success': True
        }
    
    def _simulate_camera_poses(self, frames: List[np.ndarray], 
                              timestamps: List[float]) -> List[CameraPose]:
        """Simulate camera poses (replace with actual SLAM)"""
        poses = []
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Simulate camera movement
            position = np.array([i * 0.1, 0, 0])  # Move forward
            rotation = np.eye(3)  # No rotation
            
            pose = CameraPose(
                position=position,
                rotation=rotation,
                timestamp=timestamp,
                frame_id=i
            )
            poses.append(pose)
        
        return poses
    
    def _generate_outputs(self, scene, output_dir: str):
        """Generate output files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scene
        logger.info("💾 Saving scene...")
        saved_files = self.scene_assembler.save_scene(scene, str(output_path))
        
        # Generate visualization
        logger.info("🎨 Generating visualization...")
        screenshot_path = output_path / "scene_screenshot.png"
        self.visualizer.capture_screenshot(scene, str(screenshot_path))
        
        # Generate HTML viewer
        html_path = output_path / "3d_viewer.html"
        self.visualizer.generate_html_viewer(scene, str(html_path))
        
        # Export scene data
        data_path = output_path / "scene_data"
        self.visualizer.export_scene_data(scene, str(data_path))
        
        logger.info(f"📁 Outputs saved to: {output_dir}")
        logger.info(f"  - Complete scene: {saved_files.get('point_cloud', 'N/A')}")
        logger.info(f"  - Scene mesh: {saved_files.get('mesh', 'N/A')}")
        logger.info(f"  - Screenshot: {screenshot_path}")
        logger.info(f"  - 3D Viewer: {html_path}")
        logger.info(f"  - Scene data: {data_path}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='3D Room Reconstruction from Video')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='3d_reconstruction_outputs', 
                       help='Output directory for results')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Process every Nth frame (default: 30)')
    parser.add_argument('--rooms', nargs='+', default=['Room_1'],
                       help='Room names to reconstruct')
    parser.add_argument('--use-modal', action='store_true',
                       help='Use Modal cloud processing')
    parser.add_argument('--local', action='store_true',
                       help='Use local processing only')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video_path).exists():
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    # Create pipeline
    config = {
        'frame_interval': args.frame_interval,
        'use_modal': args.use_modal and not args.local,
        'output_dir': args.output_dir,
        'rooms': args.rooms
    }
    
    pipeline = Room3DReconstructionPipeline(config)
    
    try:
        # Process video
        result = await pipeline.process_video(
            video_path=args.video_path,
            room_names=args.rooms,
            use_modal=args.use_modal and not args.local
        )
        
        # Display results
        print("\n🎉 3D RECONSTRUCTION COMPLETED!")
        print("=" * 50)
        print(f"📊 Processing Time: {result['processing_time']:.2f} seconds")
        print(f"🎬 Video Info: {result['video_info']}")
        print(f"🪑 Furniture Detected: {result['furniture_statistics']['total_detections']}")
        print(f"🏠 Rooms Reconstructed: {result['scene_statistics']['room_count']}")
        print(f"🪑 Furniture Reconstructed: {result['scene_statistics']['furniture_count']}")
        
        if 'output_directory' in result:
            print(f"\n📁 Results saved to: {result['output_directory']}")
            print("  - Complete 3D scene (PLY files)")
            print("  - Individual room models")
            print("  - Furniture objects")
            print("  - 3D viewer (HTML)")
            print("  - Screenshots and data")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())

"""
Modal Integration for 3D Room Reconstruction
Cloud processing with GPU acceleration
"""

import modal
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import tempfile
import os

from .room_reconstructor import RoomReconstructor, CameraPose, Room3DModel
from .furniture_detector import FurnitureDetector, FurnitureDetections
from .object_reconstructor import ObjectReconstructor, Object3DModel
from .scene_assembler import SceneAssembler, Scene3D
from .visualizer import SceneVisualizer

logger = logging.getLogger(__name__)

# Modal app
app = modal.App("3d-room-reconstruction")

# Define image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "opencv-python==4.8.1.78",
    "numpy>=1.24.3",
    "open3d>=0.17.0",
    "trimesh>=3.23.5",
    "ultralytics>=8.0.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "scikit-image>=0.20.0",
    "matplotlib>=3.6.0",
    "tqdm>=4.64.0",
    "imageio>=2.25.0",
    "Pillow>=10.0.1",
    "scipy>=1.10.0",
    "pyyaml>=6.0.0"
])

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600
)
def process_video_frames_modal(video_data: bytes, frame_interval: int = 30) -> Dict[str, Any]:
    """Process video frames on Modal with GPU acceleration"""
    try:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_data)
            video_path = f.name
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract frames
        frames = []
        frame_ids = []
        timestamps = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                frame_ids.append(frame_idx)
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        # Clean up
        os.unlink(video_path)
        
        return {
            'frames': frames,
            'frame_ids': frame_ids,
            'timestamps': timestamps,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height
        }
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800
)
def run_slam_modal(frames: List[np.ndarray], timestamps: List[float]) -> Dict[str, Any]:
    """Run SLAM on Modal with GPU acceleration"""
    try:
        # Simulate SLAM processing (replace with actual SLAM implementation)
        poses = []
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Simulate camera pose (replace with actual SLAM)
            position = np.array([i * 0.1, 0, 0])  # Move forward
            rotation = np.eye(3)  # No rotation
            
            pose = CameraPose(
                position=position,
                rotation=rotation,
                timestamp=timestamp,
                frame_id=i
            )
            poses.append(pose)
        
        # Simulate 3D map (replace with actual SLAM output)
        map_points = np.random.rand(1000, 3) * 10  # Random 3D points
        
        return {
            'poses': poses,
            'map_points': map_points,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"SLAM processing failed: {e}")
        raise

@app.function(
    image=image,
    gpu="T4",
    timeout=1800
)
def detect_furniture_modal(frames: List[np.ndarray], 
                          frame_ids: List[int], 
                          timestamps: List[float]) -> Dict[str, Any]:
    """Detect furniture using YOLO on Modal"""
    try:
        # Initialize furniture detector
        detector = FurnitureDetector()
        
        # Detect furniture in all frames
        detections = detector.detect_furniture_batch(frames, frame_ids, timestamps)
        
        # Get statistics
        stats = detector.get_furniture_statistics(detections)
        
        return {
            'detections': detections,
            'statistics': stats,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Furniture detection failed: {e}")
        raise

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800
)
def reconstruct_room_3d_modal(frames: List[np.ndarray], 
                             poses: List[CameraPose], 
                             room_name: str) -> Dict[str, Any]:
    """Reconstruct 3D room on Modal"""
    try:
        # Initialize room reconstructor
        reconstructor = RoomReconstructor()
        
        # Reconstruct room
        room_model = reconstructor.reconstruct_room(frames, poses, room_name)
        
        # Save room model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = reconstructor.save_room_model(room_model, temp_dir)
            
            # Read saved files
            room_data = {}
            for file_type, file_path in saved_files.items():
                if file_path.endswith('.ply'):
                    # Read PLY file
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(file_path)
                    room_data[file_type] = {
                        'points': np.asarray(pcd.points).tolist(),
                        'colors': np.asarray(pcd.colors).tolist() if pcd.colors else None
                    }
                elif file_path.endswith('.json'):
                    # Read JSON file
                    with open(file_path, 'r') as f:
                        room_data[file_type] = json.load(f)
        
        return {
            'room_model': room_data,
            'room_name': room_name,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Room reconstruction failed: {e}")
        raise

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800
)
def reconstruct_objects_3d_modal(frames: List[np.ndarray], 
                                poses: List[CameraPose], 
                                furniture_detections: List[FurnitureDetections]) -> Dict[str, Any]:
    """Reconstruct 3D objects on Modal"""
    try:
        # Initialize object reconstructor
        reconstructor = ObjectReconstructor()
        
        # Reconstruct objects
        object_models = reconstructor.reconstruct_objects(frames, poses, furniture_detections)
        
        # Save object models to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = reconstructor.save_object_models(object_models, temp_dir)
            
            # Read saved files
            objects_data = {}
            for obj_id, obj_dir in saved_files.items():
                obj_data = {}
                for file_name in os.listdir(obj_dir):
                    file_path = os.path.join(obj_dir, file_name)
                    if file_name.endswith('.ply'):
                        # Read PLY file
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(file_path)
                        obj_data[file_name.replace('.ply', '')] = {
                            'points': np.asarray(pcd.points).tolist(),
                            'colors': np.asarray(pcd.colors).tolist() if pcd.colors else None
                        }
                    elif file_name.endswith('.json'):
                        # Read JSON file
                        with open(file_path, 'r') as f:
                            obj_data[file_name.replace('.json', '')] = json.load(f)
                objects_data[obj_id] = obj_data
        
        return {
            'object_models': objects_data,
            'object_count': len(object_models),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Object reconstruction failed: {e}")
        raise

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800
)
def assemble_scene_modal(rooms_data: Dict[str, Any], 
                        objects_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble complete scene on Modal"""
    try:
        # Reconstruct room models from data
        rooms = {}
        for room_name, room_data in rooms_data.items():
            # Create point cloud
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(room_data['point_cloud']['points']))
            if room_data['point_cloud']['colors']:
                pcd.colors = o3d.utility.Vector3dVector(np.array(room_data['point_cloud']['colors']))
            
            # Create room model
            room_model = Room3DModel(
                room_name=room_name,
                point_cloud=pcd,
                room_center=np.array(room_data['metadata']['room_center']) if room_data['metadata']['room_center'] else None,
                room_dimensions=np.array(room_data['metadata']['room_dimensions']) if room_data['metadata']['room_dimensions'] else None
            )
            rooms[room_name] = room_model
        
        # Reconstruct object models from data
        furniture = []
        for obj_id, obj_data in objects_data.items():
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(obj_data['pointcloud']['points']))
            if obj_data['pointcloud']['colors']:
                pcd.colors = o3d.utility.Vector3dVector(np.array(obj_data['pointcloud']['colors']))
            
            # Create object model
            obj_model = Object3DModel(
                object_id=obj_id,
                class_name=obj_data['metadata']['class_name'],
                point_cloud=pcd,
                center=np.array(obj_data['metadata']['center']) if obj_data['metadata']['center'] else None,
                dimensions=np.array(obj_data['metadata']['dimensions']) if obj_data['metadata']['dimensions'] else None,
                confidence=obj_data['metadata']['confidence']
            )
            furniture.append(obj_model)
        
        # Assemble scene
        assembler = SceneAssembler()
        scene = assembler.assemble_scene(list(rooms.values()), furniture)
        
        # Get scene statistics
        stats = assembler.get_scene_statistics(scene)
        
        return {
            'scene_statistics': stats,
            'room_count': len(rooms),
            'furniture_count': len(furniture),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Scene assembly failed: {e}")
        raise

@app.function(
    image=image,
    timeout=600
)
def generate_visualization_modal(scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate 3D visualization on Modal"""
    try:
        # This would generate HTML viewer, screenshots, etc.
        # For now, return success
        return {
            'visualization_url': 'placeholder_url',
            'screenshot_url': 'placeholder_screenshot_url',
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise

@app.function(
    image=image,
    timeout=3600
)
def process_video_3d_reconstruction_modal(video_data: bytes, 
                                         frame_interval: int = 30,
                                         room_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Complete 3D reconstruction pipeline on Modal"""
    try:
        logger.info("Starting 3D reconstruction pipeline on Modal")
        
        # Step 1: Process video frames
        video_result = process_video_frames_modal.remote(video_data, frame_interval)
        frames = video_result['frames']
        frame_ids = video_result['frame_ids']
        timestamps = video_result['timestamps']
        
        # Step 2: Run SLAM
        slam_result = run_slam_modal.remote(frames, timestamps)
        poses = slam_result['poses']
        
        # Step 3: Detect furniture
        furniture_result = detect_furniture_modal.remote(frames, frame_ids, timestamps)
        furniture_detections = furniture_result['detections']
        
        # Step 4: Reconstruct rooms
        if room_names is None:
            room_names = ['Room_1']  # Default room
        
        rooms_data = {}
        for room_name in room_names:
            room_result = reconstruct_room_3d_modal.remote(frames, poses, room_name)
            rooms_data[room_name] = room_result['room_model']
        
        # Step 5: Reconstruct objects
        objects_result = reconstruct_objects_3d_modal.remote(frames, poses, furniture_detections)
        objects_data = objects_result['object_models']
        
        # Step 6: Assemble scene
        scene_result = assemble_scene_modal.remote(rooms_data, objects_data)
        
        # Step 7: Generate visualization
        viz_result = generate_visualization_modal.remote(scene_result)
        
        return {
            'video_info': {
                'fps': video_result['fps'],
                'total_frames': video_result['total_frames'],
                'width': video_result['width'],
                'height': video_result['height']
            },
            'furniture_statistics': furniture_result['statistics'],
            'scene_statistics': scene_result['scene_statistics'],
            'visualization': viz_result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"3D reconstruction pipeline failed: {e}")
        raise

# Local client functions
def process_video_local(video_path: str, 
                       frame_interval: int = 30,
                       room_names: Optional[List[str]] = None,
                       use_modal: bool = True) -> Dict[str, Any]:
    """
    Process video for 3D reconstruction
    
    Args:
        video_path: Path to input video
        frame_interval: Process every Nth frame
        room_names: List of room names
        use_modal: Whether to use Modal cloud processing
        
    Returns:
        Processing results
    """
    if use_modal:
        # Use Modal cloud processing
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        result = process_video_3d_reconstruction_modal.remote(
            video_data, frame_interval, room_names
        )
        return result
    else:
        # Use local processing (implement if needed)
        raise NotImplementedError("Local processing not implemented yet")

def main():
    """Main function for testing Modal integration"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python modal_integration.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        result = process_video_local(video_path)
        print("3D Reconstruction Results:")
        print(f"Success: {result['success']}")
        print(f"Video Info: {result['video_info']}")
        print(f"Furniture Statistics: {result['furniture_statistics']}")
        print(f"Scene Statistics: {result['scene_statistics']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Scene Assembly Module
Combines room models and furniture objects into complete 3D scenes
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from .room_reconstructor import Room3DModel
from .object_reconstructor import Object3DModel

logger = logging.getLogger(__name__)

@dataclass
class Scene3D:
    """Complete 3D scene data"""
    scene_name: str
    rooms: Dict[str, Room3DModel]
    furniture: List[Object3DModel]
    scene_bounds: Optional[o3d.geometry.AxisAlignedBoundingBox] = None
    scene_center: Optional[np.ndarray] = None
    scene_dimensions: Optional[np.ndarray] = None

class SceneAssembler:
    """Assembles complete 3D scenes from rooms and furniture"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        logger.info("Scene assembler initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'furniture_placement_threshold': 0.1,  # Distance threshold for furniture placement
            'enable_furniture_placement': True,
            'enable_room_alignment': True,
            'voxel_size': 0.01,
            'enable_scene_optimization': True
        }
    
    def assemble_scene(self, rooms: List[Room3DModel], 
                      furniture: List[Object3DModel], 
                      scene_name: str = "complete_scene") -> Scene3D:
        """
        Assemble complete 3D scene from rooms and furniture
        
        Args:
            rooms: List of room models
            furniture: List of furniture objects
            scene_name: Name of the scene
            
        Returns:
            Scene3D with complete assembled scene
        """
        logger.info(f"Assembling scene: {scene_name}")
        
        # Create room dictionary
        room_dict = {room.room_name: room for room in rooms}
        
        # Place furniture in appropriate rooms
        if self.config['enable_furniture_placement']:
            furniture = self._place_furniture_in_rooms(furniture, room_dict)
        
        # Align rooms if needed
        if self.config['enable_room_alignment']:
            room_dict = self._align_rooms(room_dict)
        
        # Create complete scene
        scene = Scene3D(
            scene_name=scene_name,
            rooms=room_dict,
            furniture=furniture
        )
        
        # Calculate scene properties
        scene = self._calculate_scene_properties(scene)
        
        # Optimize scene if enabled
        if self.config['enable_scene_optimization']:
            scene = self._optimize_scene(scene)
        
        logger.info(f"Scene assembled: {len(room_dict)} rooms, {len(furniture)} furniture pieces")
        return scene
    
    def _place_furniture_in_rooms(self, furniture: List[Object3DModel], 
                                 rooms: Dict[str, Room3DModel]) -> List[Object3DModel]:
        """Place furniture objects in appropriate rooms"""
        if not furniture or not rooms:
            return furniture
        
        # For each furniture piece, find the best room
        for obj in furniture:
            best_room = None
            best_distance = float('inf')
            
            for room_name, room in rooms.items():
                if room.room_center is not None and obj.center is not None:
                    # Calculate distance from furniture to room center
                    distance = np.linalg.norm(obj.center - room.room_center)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_room = room_name
            
            # Assign furniture to best room
            if best_room and best_distance < self.config['furniture_placement_threshold']:
                # Move furniture to room coordinate system
                if best_room in rooms and rooms[best_room].room_center is not None:
                    room_center = rooms[best_room].room_center
                    # Adjust furniture position relative to room
                    if obj.center is not None:
                        offset = room_center - obj.center
                        obj.point_cloud.translate(offset)
                        if obj.mesh is not None:
                            obj.mesh.translate(offset)
                        obj.center = room_center
        
        return furniture
    
    def _align_rooms(self, rooms: Dict[str, Room3DModel]) -> Dict[str, Room3DModel]:
        """Align rooms in a common coordinate system"""
        if len(rooms) <= 1:
            return rooms
        
        # Use first room as reference
        reference_room = list(rooms.values())[0]
        reference_center = reference_room.room_center
        
        if reference_center is None:
            return rooms
        
        # Align other rooms to reference
        aligned_rooms = {reference_room.room_name: reference_room}
        
        for room_name, room in rooms.items():
            if room_name == reference_room.room_name:
                continue
            
            if room.room_center is not None:
                # Calculate offset to align with reference
                offset = reference_center - room.room_center
                
                # Apply offset to room
                room.point_cloud.translate(offset)
                if room.mesh is not None:
                    room.mesh.translate(offset)
                
                # Update room properties
                room.room_center = reference_center
                if room.bounding_box is not None:
                    room.bounding_box = room.point_cloud.get_axis_aligned_bounding_box()
            
            aligned_rooms[room_name] = room
        
        return aligned_rooms
    
    def _calculate_scene_properties(self, scene: Scene3D) -> Scene3D:
        """Calculate scene-level properties"""
        all_points = []
        
        # Collect points from all rooms
        for room in scene.rooms.values():
            if len(room.point_cloud.points) > 0:
                points = np.asarray(room.point_cloud.points)
                all_points.append(points)
        
        # Collect points from all furniture
        for obj in scene.furniture:
            if len(obj.point_cloud.points) > 0:
                points = np.asarray(obj.point_cloud.points)
                all_points.append(points)
        
        if all_points:
            all_points = np.vstack(all_points)
            
            # Create scene point cloud
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(all_points)
            
            # Calculate scene bounds
            scene.scene_bounds = scene_pcd.get_axis_aligned_bounding_box()
            scene.scene_center = scene.scene_bounds.get_center()
            scene.scene_dimensions = scene.scene_bounds.get_extent()
        
        return scene
    
    def _optimize_scene(self, scene: Scene3D) -> Scene3D:
        """Optimize scene for better visualization"""
        # Downsample point clouds for better performance
        for room in scene.rooms.values():
            if len(room.point_cloud.points) > 0:
                room.point_cloud = room.point_cloud.voxel_down_sample(
                    voxel_size=self.config['voxel_size']
                )
        
        for obj in scene.furniture:
            if len(obj.point_cloud.points) > 0:
                obj.point_cloud = obj.point_cloud.voxel_down_sample(
                    voxel_size=self.config['voxel_size']
                )
        
        return scene
    
    def create_combined_point_cloud(self, scene: Scene3D) -> o3d.geometry.PointCloud:
        """Create a single point cloud combining all rooms and furniture"""
        all_point_clouds = []
        
        # Add room point clouds
        for room in scene.rooms.values():
            if len(room.point_cloud.points) > 0:
                all_point_clouds.append(room.point_cloud)
        
        # Add furniture point clouds
        for obj in scene.furniture:
            if len(obj.point_cloud.points) > 0:
                all_point_clouds.append(obj.point_cloud)
        
        if not all_point_clouds:
            return o3d.geometry.PointCloud()
        
        # Combine all point clouds
        combined_pcd = all_point_clouds[0]
        for pcd in all_point_clouds[1:]:
            combined_pcd += pcd
        
        return combined_pcd
    
    def create_combined_mesh(self, scene: Scene3D) -> o3d.geometry.TriangleMesh:
        """Create a single mesh combining all rooms and furniture"""
        all_meshes = []
        
        # Add room meshes
        for room in scene.rooms.values():
            if room.mesh is not None and len(room.mesh.vertices) > 0:
                all_meshes.append(room.mesh)
        
        # Add furniture meshes
        for obj in scene.furniture:
            if obj.mesh is not None and len(obj.mesh.vertices) > 0:
                all_meshes.append(obj.mesh)
        
        if not all_meshes:
            return o3d.geometry.TriangleMesh()
        
        # Combine all meshes
        combined_mesh = all_meshes[0]
        for mesh in all_meshes[1:]:
            combined_mesh += mesh
        
        return combined_mesh
    
    def get_scene_statistics(self, scene: Scene3D) -> Dict[str, Any]:
        """Get statistics about the assembled scene"""
        stats = {
            'scene_name': scene.scene_name,
            'room_count': len(scene.rooms),
            'furniture_count': len(scene.furniture),
            'rooms': {},
            'furniture_by_class': {},
            'scene_bounds': None,
            'scene_center': None,
            'scene_dimensions': None
        }
        
        # Room statistics
        for room_name, room in scene.rooms.items():
            stats['rooms'][room_name] = {
                'point_count': len(room.point_cloud.points),
                'has_mesh': room.mesh is not None,
                'center': room.room_center.tolist() if room.room_center is not None else None,
                'dimensions': room.room_dimensions.tolist() if room.room_dimensions is not None else None
            }
        
        # Furniture statistics
        for obj in scene.furniture:
            class_name = obj.class_name
            if class_name not in stats['furniture_by_class']:
                stats['furniture_by_class'][class_name] = 0
            stats['furniture_by_class'][class_name] += 1
        
        # Scene properties
        if scene.scene_center is not None:
            stats['scene_center'] = scene.scene_center.tolist()
        if scene.scene_dimensions is not None:
            stats['scene_dimensions'] = scene.scene_dimensions.tolist()
        
        return stats
    
    def save_scene(self, scene: Scene3D, output_dir: str) -> Dict[str, str]:
        """Save complete scene to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save combined point cloud
        combined_pcd = self.create_combined_point_cloud(scene)
        if len(combined_pcd.points) > 0:
            pcd_path = output_path / f"{scene.scene_name}_complete.ply"
            o3d.io.write_point_cloud(str(pcd_path), combined_pcd)
            saved_files['point_cloud'] = str(pcd_path)
        
        # Save combined mesh
        combined_mesh = self.create_combined_mesh(scene)
        if len(combined_mesh.vertices) > 0:
            mesh_path = output_path / f"{scene.scene_name}_complete_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), combined_mesh)
            saved_files['mesh'] = str(mesh_path)
        
        # Save individual rooms
        rooms_dir = output_path / "rooms"
        rooms_dir.mkdir(exist_ok=True)
        
        for room_name, room in scene.rooms.items():
            room_pcd_path = rooms_dir / f"{room_name}.ply"
            o3d.io.write_point_cloud(str(room_pcd_path), room.point_cloud)
            
            if room.mesh is not None:
                room_mesh_path = rooms_dir / f"{room_name}_mesh.ply"
                o3d.io.write_triangle_mesh(str(room_mesh_path), room.mesh)
        
        # Save individual furniture
        furniture_dir = output_path / "furniture"
        furniture_dir.mkdir(exist_ok=True)
        
        for i, obj in enumerate(scene.furniture):
            obj_pcd_path = furniture_dir / f"{obj.class_name}_{i}.ply"
            o3d.io.write_point_cloud(str(obj_pcd_path), obj.point_cloud)
            
            if obj.mesh is not None:
                obj_mesh_path = furniture_dir / f"{obj.class_name}_{i}_mesh.ply"
                o3d.io.write_triangle_mesh(str(obj_mesh_path), obj.mesh)
        
        # Save scene statistics
        stats = self.get_scene_statistics(scene)
        stats_path = output_path / f"{scene.scene_name}_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        saved_files['statistics'] = str(stats_path)
        
        # Save scene metadata
        metadata = {
            'scene_name': scene.scene_name,
            'room_count': len(scene.rooms),
            'furniture_count': len(scene.furniture),
            'scene_center': scene.scene_center.tolist() if scene.scene_center is not None else None,
            'scene_dimensions': scene.scene_dimensions.tolist() if scene.scene_dimensions is not None else None,
            'files': saved_files
        }
        
        metadata_path = output_path / f"{scene.scene_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        logger.info(f"Scene saved: {saved_files}")
        return saved_files

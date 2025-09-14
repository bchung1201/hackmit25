"""
3D Room Reconstruction Module
Reconstructs 3D room models from video frames and camera poses
"""

import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple, Dict, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CameraPose:
    """Camera pose data"""
    position: np.ndarray  # 3D position
    rotation: np.ndarray  # 3x3 rotation matrix
    timestamp: float
    frame_id: int

@dataclass
class Room3DModel:
    """3D room model data"""
    room_name: str
    point_cloud: o3d.geometry.PointCloud
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    bounding_box: Optional[o3d.geometry.AxisAlignedBoundingBox] = None
    room_center: Optional[np.ndarray] = None
    room_dimensions: Optional[np.ndarray] = None

class RoomReconstructor:
    """3D room reconstruction from video frames"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize stereo matcher
        self.stereo_matcher = cv2.StereoBM_create(
            numDisparities=self.config['stereo_num_disparities'],
            blockSize=self.config['stereo_block_size']
        )
        
        # Initialize feature detector
        self.feature_detector = cv2.ORB_create(
            nfeatures=self.config['max_features']
        )
        
        # Initialize matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        logger.info("Room reconstructor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'stereo_num_disparities': 64,
            'stereo_block_size': 15,
            'max_features': 1000,
            'min_matches': 50,
            'reprojection_threshold': 1.0,
            'voxel_size': 0.01,
            'mesh_resolution': 0.02,
            'enable_mesh_generation': True
        }
    
    def reconstruct_room(self, frames: List[np.ndarray], 
                        poses: List[CameraPose], 
                        room_name: str) -> Room3DModel:
        """
        Reconstruct 3D room from frames and poses
        
        Args:
            frames: List of video frames
            poses: List of camera poses
            room_name: Name of the room
            
        Returns:
            Room3DModel with 3D reconstruction
        """
        logger.info(f"Reconstructing room: {room_name}")
        
        # Extract features and matches
        features, descriptors = self._extract_features(frames)
        matches = self._match_features(descriptors)
        
        # Triangulate 3D points
        points_3d = self._triangulate_points(frames, poses, features, matches)
        
        # Create point cloud
        point_cloud = self._create_point_cloud(points_3d, frames, poses)
        
        # Clean and filter point cloud
        point_cloud = self._clean_point_cloud(point_cloud)
        
        # Generate mesh if enabled
        mesh = None
        if self.config['enable_mesh_generation']:
            mesh = self._generate_mesh(point_cloud)
        
        # Calculate room properties
        bounding_box = point_cloud.get_axis_aligned_bounding_box()
        room_center = bounding_box.get_center()
        room_dimensions = bounding_box.get_extent()
        
        return Room3DModel(
            room_name=room_name,
            point_cloud=point_cloud,
            mesh=mesh,
            bounding_box=bounding_box,
            room_center=room_center,
            room_dimensions=room_dimensions
        )
    
    def _extract_features(self, frames: List[np.ndarray]) -> Tuple[List, List]:
        """Extract features from frames"""
        features = []
        descriptors = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            keypoints, desc = self.feature_detector.detectAndCompute(gray, None)
            
            features.append(keypoints)
            descriptors.append(desc)
        
        return features, descriptors
    
    def _match_features(self, descriptors: List) -> List:
        """Match features between consecutive frames"""
        matches = []
        
        for i in range(len(descriptors) - 1):
            if descriptors[i] is not None and descriptors[i+1] is not None:
                # Match features
                frame_matches = self.matcher.match(descriptors[i], descriptors[i+1])
                
                # Filter matches by distance
                good_matches = [m for m in frame_matches if m.distance < 50]
                matches.append(good_matches)
            else:
                matches.append([])
        
        return matches
    
    def _triangulate_points(self, frames: List[np.ndarray], 
                           poses: List[CameraPose], 
                           features: List, 
                           matches: List) -> np.ndarray:
        """Triangulate 3D points from feature matches"""
        points_3d = []
        
        for i, match_list in enumerate(matches):
            if len(match_list) < self.config['min_matches']:
                continue
            
            # Get matched keypoints
            kp1 = features[i]
            kp2 = features[i + 1]
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in match_list]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in match_list]).reshape(-1, 1, 2)
            
            # Get camera matrices
            K = self._get_camera_matrix(frames[i].shape)
            R1, t1 = self._pose_to_rt(poses[i])
            R2, t2 = self._pose_to_rt(poses[i + 1])
            
            # Triangulate points
            P1 = K @ np.hstack([R1, t1])
            P2 = K @ np.hstack([R2, t2])
            
            points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            # Avoid division by zero
            w_coords = points_4d[3]
            valid_mask = np.abs(w_coords) > 1e-8  # Avoid division by very small numbers
            
            if np.any(valid_mask):
                points_3d_batch = points_4d[:3, valid_mask] / w_coords[valid_mask]
                points_3d.append(points_3d_batch.T)
        
        if points_3d:
            return np.vstack(points_3d)
        else:
            return np.array([]).reshape(0, 3)
    
    def _create_point_cloud(self, points_3d: np.ndarray, 
                           frames: List[np.ndarray], 
                           poses: List[CameraPose]) -> o3d.geometry.PointCloud:
        """Create Open3D point cloud from 3D points"""
        if len(points_3d) == 0:
            return o3d.geometry.PointCloud()
        
        # Create point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        
        # Add colors (simplified - use average color)
        colors = np.ones((len(points_3d), 3)) * 0.5  # Gray color
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return point_cloud
    
    def _clean_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Clean and filter point cloud"""
        if len(point_cloud.points) == 0:
            return point_cloud
        
        # Remove statistical outliers
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Remove radius outliers
        point_cloud, _ = point_cloud.remove_radius_outlier(
            nb_points=16, radius=0.05
        )
        
        # Voxel downsampling
        point_cloud = point_cloud.voxel_down_sample(
            voxel_size=self.config['voxel_size']
        )
        
        return point_cloud
    
    def _generate_mesh(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Generate mesh from point cloud"""
        if len(point_cloud.points) == 0:
            return o3d.geometry.TriangleMesh()
        
        try:
            # Estimate normals
            point_cloud.estimate_normals()
            
            # Orient normals
            point_cloud.orient_normals_consistent_tangent_plane(100)
            
            # Try Poisson reconstruction with error handling
            try:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    point_cloud, depth=8  # Reduced depth for stability
                )
                
                # Clean mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                return mesh
                
            except Exception as e:
                logger.warning(f"Poisson reconstruction failed: {e}. Trying alternative method...")
                
                # Fallback: Use alpha shapes or ball pivoting
                try:
                    # Try ball pivoting algorithm
                    distances = point_cloud.compute_nearest_neighbor_distance()
                    avg_dist = np.mean(distances)
                    radius = 3 * avg_dist
                    
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        point_cloud, o3d.utility.DoubleVector([radius, radius * 2])
                    )
                    
                    return mesh
                    
                except Exception as e2:
                    logger.warning(f"Ball pivoting failed: {e2}. Using convex hull...")
                    
                    # Final fallback: convex hull
                    mesh = point_cloud.compute_convex_hull()[0]
                    return mesh
                    
        except Exception as e:
            logger.error(f"Mesh generation completely failed: {e}")
            return o3d.geometry.TriangleMesh()
    
    def _get_camera_matrix(self, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Get camera intrinsic matrix"""
        h, w = image_shape[:2]
        fx = fy = max(w, h)  # Simplified focal length
        cx, cy = w / 2, h / 2
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def _pose_to_rt(self, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pose to rotation matrix and translation vector"""
        return pose.rotation, pose.position.reshape(3, 1)
    
    def save_room_model(self, room_model: Room3DModel, output_dir: str) -> Dict[str, str]:
        """Save room model to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save point cloud
        pcd_path = output_path / f"{room_model.room_name}_pointcloud.ply"
        o3d.io.write_point_cloud(str(pcd_path), room_model.point_cloud)
        saved_files['point_cloud'] = str(pcd_path)
        
        # Save mesh if available
        if room_model.mesh is not None:
            mesh_path = output_path / f"{room_model.room_name}_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), room_model.mesh)
            saved_files['mesh'] = str(mesh_path)
        
        # Save metadata
        metadata = {
            'room_name': room_model.room_name,
            'room_center': room_model.room_center.tolist() if room_model.room_center is not None else None,
            'room_dimensions': room_model.room_dimensions.tolist() if room_model.room_dimensions is not None else None,
            'point_count': len(room_model.point_cloud.points),
            'has_mesh': room_model.mesh is not None
        }
        
        import json
        metadata_path = output_path / f"{room_model.room_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        logger.info(f"Room model saved: {saved_files}")
        return saved_files

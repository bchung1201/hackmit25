"""
3D Object Reconstruction Module
Reconstructs 3D models of individual furniture pieces
"""

import numpy as np
import cv2
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

from .furniture_detector import FurnitureDetection, FurnitureDetections
from .room_reconstructor import CameraPose

logger = logging.getLogger(__name__)

@dataclass
class Object3DModel:
    """3D object model data"""
    object_id: str
    class_name: str
    point_cloud: o3d.geometry.PointCloud
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    bounding_box: Optional[o3d.geometry.AxisAlignedBoundingBox] = None
    center: Optional[np.ndarray] = None
    dimensions: Optional[np.ndarray] = None
    confidence: float = 0.0

class ObjectReconstructor:
    """3D reconstruction of individual furniture objects"""
    
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
        
        logger.info("Object reconstructor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'stereo_num_disparities': 32,
            'stereo_block_size': 11,
            'max_features': 500,
            'min_matches': 5,  # Reduced from 20
            'reprojection_threshold': 1.0,
            'voxel_size': 0.005,
            'mesh_resolution': 0.01,
            'enable_mesh_generation': True,
            'min_object_points': 50  # Reduced from 100
        }
    
    def reconstruct_objects(self, frames: List[np.ndarray], 
                          poses: List[CameraPose], 
                          furniture_detections: List[FurnitureDetections]) -> List[Object3DModel]:
        """
        Reconstruct 3D models of detected furniture objects
        
        Args:
            frames: List of video frames
            poses: List of camera poses
            furniture_detections: List of furniture detections per frame
            
        Returns:
            List of Object3DModel with 3D reconstructions
        """
        logger.info("Starting 3D object reconstruction")
        
        # Track objects across frames
        object_tracks = self._track_objects(furniture_detections)
        
        # Reconstruct each tracked object
        object_models = []
        for track_id, track_detections in object_tracks.items():
            if len(track_detections) < 2:  # Need at least 2 detections
                continue
            
            try:
                object_model = self._reconstruct_single_object(
                    frames, poses, track_detections, track_id
                )
                if object_model is not None:
                    object_models.append(object_model)
            except Exception as e:
                logger.warning(f"Failed to reconstruct object {track_id}: {e}")
                continue
        
        logger.info(f"Reconstructed {len(object_models)} 3D objects")
        return object_models
    
    def _track_objects(self, furniture_detections: List[FurnitureDetections]) -> Dict[int, List[FurnitureDetection]]:
        """Track individual objects across frames"""
        tracks = {}
        next_track_id = 0
        
        total_detections = sum(len(fd.detections) for fd in furniture_detections)
        logger.info(f"Tracking objects across {len(furniture_detections)} frames with {total_detections} total detections")
        
        for frame_idx, frame_detections in enumerate(furniture_detections):
            current_tracks = {}
            
            for detection in frame_detections.detections:
                best_track_id = None
                best_iou = 0.0
                
                # Find best matching track
                for track_id, track_detections in tracks.items():
                    if track_detections:
                        last_detection = track_detections[-1]
                        iou = self._calculate_iou(detection.bbox, last_detection.bbox)
                        
                        # Check class consistency
                        class_match = detection.class_name == last_detection.class_name
                        
                        # Lower IoU threshold for better tracking
                        if iou > 0.1 and iou > best_iou and class_match:
                            best_iou = iou
                            best_track_id = track_id
                
                # Assign to track or create new one
                if best_track_id is not None:
                    current_tracks[best_track_id] = detection
                else:
                    current_tracks[next_track_id] = detection
                    next_track_id += 1
            
            # Update tracks
            for track_id, detection in current_tracks.items():
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(detection)
        
        # Filter tracks with sufficient detections
        valid_tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
        logger.info(f"Found {len(tracks)} total tracks, {len(valid_tracks)} with >= 2 detections")
        
        # Log track details
        for track_id, detections in valid_tracks.items():
            class_name = detections[0].class_name if detections else "unknown"
            logger.info(f"Track {track_id}: {class_name} with {len(detections)} detections")
        
        return valid_tracks
    
    def _reconstruct_single_object(self, frames: List[np.ndarray], 
                                 poses: List[CameraPose], 
                                 detections: List[FurnitureDetection], 
                                 track_id: int) -> Optional[Object3DModel]:
        """Reconstruct a single 3D object from multiple detections"""
        if len(detections) < 2:
            logger.warning(f"Track {track_id}: Not enough detections ({len(detections)})")
            return None
        
        # Get object class name
        class_name = detections[0].class_name
        logger.info(f"Reconstructing track {track_id}: {class_name} with {len(detections)} detections")
        
        # Extract object regions from frames
        object_regions = []
        object_poses = []
        
        for detection in detections:
            frame_idx = detection.frame_id
            if frame_idx < len(frames) and frame_idx < len(poses):
                # Extract object region
                x1, y1, x2, y2 = detection.bbox
                object_region = frames[frame_idx][y1:y2, x1:x2]
                
                if object_region.size > 0:
                    object_regions.append(object_region)
                    object_poses.append(poses[frame_idx])
        
        logger.info(f"Track {track_id}: Extracted {len(object_regions)} valid regions")
        
        if len(object_regions) < 2:
            logger.warning(f"Track {track_id}: Not enough valid regions ({len(object_regions)})")
            return None
        
        # Reconstruct 3D points for this object
        points_3d = self._triangulate_object_points(object_regions, object_poses, detections)
        logger.info(f"Track {track_id}: Triangulated {len(points_3d)} 3D points")
        
        if len(points_3d) < self.config['min_object_points']:
            logger.warning(f"Track {track_id}: Not enough 3D points ({len(points_3d)} < {self.config['min_object_points']})")
            return None
        
        # Create point cloud
        point_cloud = self._create_object_point_cloud(points_3d, object_regions)
        
        # Clean point cloud
        point_cloud = self._clean_object_point_cloud(point_cloud)
        
        if len(point_cloud.points) < self.config['min_object_points']:
            return None
        
        # Generate mesh if enabled
        mesh = None
        if self.config['enable_mesh_generation']:
            mesh = self._generate_object_mesh(point_cloud)
        
        # Calculate object properties
        bounding_box = point_cloud.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        dimensions = bounding_box.get_extent()
        
        # Calculate average confidence
        avg_confidence = np.mean([d.confidence for d in detections])
        
        return Object3DModel(
            object_id=f"{class_name}_{track_id}",
            class_name=class_name,
            point_cloud=point_cloud,
            mesh=mesh,
            bounding_box=bounding_box,
            center=center,
            dimensions=dimensions,
            confidence=avg_confidence
        )
    
    def _triangulate_object_points(self, object_regions: List[np.ndarray], 
                                 poses: List[CameraPose], 
                                 detections: List[FurnitureDetection]) -> np.ndarray:
        """Triangulate 3D points for an object"""
        points_3d = []
        
        for i in range(len(object_regions) - 1):
            region1 = object_regions[i]
            region2 = object_regions[i + 1]
            
            # Extract features from object regions
            gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
            
            kp1, desc1 = self.feature_detector.detectAndCompute(gray1, None)
            kp2, desc2 = self.feature_detector.detectAndCompute(gray2, None)
            
            if desc1 is None or desc2 is None:
                continue
            
            # Match features
            matches = self.matcher.match(desc1, desc2)
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) < self.config['min_matches']:
                continue
            
            # Get matched keypoints
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Adjust coordinates to full frame
            detection1 = detections[i]
            detection2 = detections[i + 1]
            
            x1_1, y1_1, _, _ = detection1.bbox
            x1_2, y1_2, _, _ = detection2.bbox
            
            pts1[:, 0, 0] += x1_1
            pts1[:, 0, 1] += y1_1
            pts2[:, 0, 0] += x1_2
            pts2[:, 0, 1] += y1_2
            
            # Get camera matrices
            K = self._get_camera_matrix(object_regions[i].shape)
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
    
    def _create_object_point_cloud(self, points_3d: np.ndarray, 
                                 object_regions: List[np.ndarray]) -> o3d.geometry.PointCloud:
        """Create point cloud for an object"""
        if len(points_3d) == 0:
            return o3d.geometry.PointCloud()
        
        # Create point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        
        # Add colors (simplified)
        colors = np.ones((len(points_3d), 3)) * 0.7  # Light gray
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return point_cloud
    
    def _clean_object_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Clean object point cloud"""
        if len(point_cloud.points) == 0:
            return point_cloud
        
        # Remove statistical outliers
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=10, std_ratio=1.5
        )
        
        # Remove radius outliers
        point_cloud, _ = point_cloud.remove_radius_outlier(
            nb_points=5, radius=0.02
        )
        
        # Voxel downsampling
        point_cloud = point_cloud.voxel_down_sample(
            voxel_size=self.config['voxel_size']
        )
        
        return point_cloud
    
    def _generate_object_mesh(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Generate mesh for an object"""
        if len(point_cloud.points) == 0:
            return o3d.geometry.TriangleMesh()
        
        try:
            # Estimate normals
            point_cloud.estimate_normals()
            
            # Orient normals
            point_cloud.orient_normals_consistent_tangent_plane(50)
            
            # Try Poisson reconstruction with error handling
            try:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    point_cloud, depth=7  # Reduced depth for objects
                )
                
                # Clean mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                return mesh
                
            except Exception as e:
                logger.warning(f"Object Poisson reconstruction failed: {e}. Trying alternative method...")
                
                # Fallback: Use ball pivoting algorithm
                try:
                    distances = point_cloud.compute_nearest_neighbor_distance()
                    avg_dist = np.mean(distances)
                    radius = 2 * avg_dist
                    
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        point_cloud, o3d.utility.DoubleVector([radius, radius * 1.5])
                    )
                    
                    return mesh
                    
                except Exception as e2:
                    logger.warning(f"Object ball pivoting failed: {e2}. Using convex hull...")
                    
                    # Final fallback: convex hull
                    mesh = point_cloud.compute_convex_hull()[0]
                    return mesh
                    
        except Exception as e:
            logger.error(f"Object mesh generation completely failed: {e}")
            return o3d.geometry.TriangleMesh()
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_camera_matrix(self, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Get camera intrinsic matrix"""
        h, w = image_shape[:2]
        fx = fy = max(w, h)
        cx, cy = w / 2, h / 2
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def _pose_to_rt(self, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pose to rotation matrix and translation vector"""
        return pose.rotation, pose.position.reshape(3, 1)
    
    def save_object_models(self, object_models: List[Object3DModel], 
                          output_dir: str) -> Dict[str, str]:
        """Save object models to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for i, obj_model in enumerate(object_models):
            obj_dir = output_path / f"object_{i}_{obj_model.class_name}"
            obj_dir.mkdir(exist_ok=True)
            
            # Save point cloud
            pcd_path = obj_dir / "pointcloud.ply"
            o3d.io.write_point_cloud(str(pcd_path), obj_model.point_cloud)
            
            # Save mesh if available
            if obj_model.mesh is not None:
                mesh_path = obj_dir / "mesh.ply"
                o3d.io.write_triangle_mesh(str(mesh_path), obj_model.mesh)
            
            # Save metadata
            metadata = {
                'object_id': obj_model.object_id,
                'class_name': obj_model.class_name,
                'center': obj_model.center.tolist() if obj_model.center is not None else None,
                'dimensions': obj_model.dimensions.tolist() if obj_model.dimensions is not None else None,
                'point_count': len(obj_model.point_cloud.points),
                'has_mesh': obj_model.mesh is not None,
                'confidence': obj_model.confidence
            }
            
            import json
            metadata_path = obj_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files[obj_model.object_id] = str(obj_dir)
        
        logger.info(f"Object models saved: {len(saved_files)} objects")
        return saved_files

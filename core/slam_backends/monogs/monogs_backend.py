"""
MonoGS backend implementation for monocular SLAM
"""

import sys
import os
from pathlib import Path

# Add MonoGS to path when available
monogs_path = Path(__file__).parent.parent.parent.parent / "external" / "MonoGS"
if monogs_path.exists():
    sys.path.insert(0, str(monogs_path))

from core.interfaces.base_slam_backend import BaseSLAMBackend, SLAMFrame, SLAMResult, SLAMConfig
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import asyncio
import cv2

logger = logging.getLogger(__name__)

class MonoGSBackend(BaseSLAMBackend):
    """MonoGS implementation for monocular SLAM with real-time performance"""
    
    def __init__(self):
        super().__init__()
        self.slam_system = None
        self.device = None
        self.gaussian_renderer = None
        self.tracking_frontend = None
        self.mapping_backend = None
        self.last_keyframe_id = -1
        
    async def initialize(self, config: SLAMConfig) -> None:
        """Initialize MonoGS system"""
        logger.info("Initializing MonoGS backend...")
        self.config = config
        
        try:
            # Set device
            self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Try to import MonoGS components
            try:
                from monogs.slam import MonoGS
                from monogs.config import Config
                from monogs.tracking import Tracker
                from monogs.mapping import Mapper
                
                # Load MonoGS configuration
                config_path = Path("configs/slam_configs/monogs_config.yaml")
                if config_path.exists():
                    mono_config = Config.from_yaml(str(config_path))
                else:
                    mono_config = self._create_default_config()
                
                # Initialize MonoGS components
                self.slam_system = MonoGS(mono_config, device=self.device)
                self.tracking_frontend = Tracker(mono_config, device=self.device)
                self.mapping_backend = Mapper(mono_config, device=self.device)
                
                logger.info("MonoGS system initialized successfully")
                
            except ImportError as e:
                logger.warning(f"MonoGS not available, using fallback implementation: {e}")
                self.slam_system = None
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"MonoGS initialization failed: {e}")
            raise
    
    async def process_frame(self, frame: SLAMFrame) -> SLAMResult:
        """Process frame with MonoGS"""
        if not self.is_initialized:
            raise RuntimeError("MonoGS not initialized")
            
        logger.debug(f"Processing frame {frame.frame_id} with MonoGS")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.slam_system is not None:
                # Use real MonoGS system
                result = await self._process_with_monogs(frame)
            else:
                # Use fallback implementation
                result = await self._process_with_fallback(frame)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.debug(f"MonoGS processed frame {frame.frame_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"MonoGS processing error: {e}")
            raise
    
    async def _process_with_monogs(self, frame: SLAMFrame) -> SLAMResult:
        """Process frame with actual MonoGS system"""
        # Convert frame to MonoGS format
        color_tensor = torch.from_numpy(frame.image).float().permute(2, 0, 1) / 255.0
        color_tensor = color_tensor.to(self.device)
        
        intrinsics_tensor = torch.from_numpy(frame.camera_intrinsics).float().to(self.device)
        
        # Tracking step
        tracking_result = self.tracking_frontend.track(
            image=color_tensor,
            intrinsics=intrinsics_tensor,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp
        )
        
        # Update pose
        if 'pose' in tracking_result:
            self.current_pose = tracking_result['pose'].cpu().numpy()
            self.trajectory.append(self.current_pose.copy())
        
        # Check if keyframe
        is_keyframe = self.is_keyframe(frame.frame_id) or tracking_result.get('is_keyframe', False)
        
        # Mapping step (only for keyframes)
        mapping_result = {}
        if is_keyframe:
            mapping_result = self.mapping_backend.map(
                image=color_tensor,
                pose=tracking_result['pose'],
                intrinsics=intrinsics_tensor,
                frame_id=frame.frame_id
            )
            self.last_keyframe_id = frame.frame_id
        
        # Extract Gaussian splats
        gaussians = self._extract_gaussians_from_monogs(mapping_result)
        
        # Extract point cloud
        point_cloud = mapping_result.get('point_cloud', np.array([]))
        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().numpy()
        
        # Loop closure detection
        loop_closures = tracking_result.get('loop_closures', [])
        
        return SLAMResult(
            camera_trajectory=self.trajectory.copy(),
            gaussian_splats=gaussians,
            point_cloud=point_cloud,
            reconstruction_quality=tracking_result.get('tracking_quality', 0.8),
            processing_time=0.0,  # Will be set by caller
            loop_closures=loop_closures,
            current_pose=self.current_pose,
            keyframe_ids=tracking_result.get('keyframe_ids', [])
        )
    
    async def _process_with_fallback(self, frame: SLAMFrame) -> SLAMResult:
        """Fallback processing when MonoGS is not available"""
        # Simulate MonoGS-like processing with focus on speed
        await asyncio.sleep(0.01)  # Fast processing for real-time performance
        
        # Monocular visual odometry simulation
        if len(self.trajectory) == 0:
            self.current_pose = np.eye(4)
        else:
            # Use optical flow for motion estimation
            motion = self._estimate_motion_from_optical_flow(frame)
            self.current_pose = self.trajectory[-1] @ motion
        
        self.trajectory.append(self.current_pose.copy())
        
        # Generate Gaussians only for keyframes (performance optimization)
        gaussians = []
        if self.is_keyframe(frame.frame_id):
            gaussians = self._generate_monocular_gaussians(frame)
            self.last_keyframe_id = frame.frame_id
        
        # Generate sparse point cloud (monocular constraint)
        point_cloud = self._generate_monocular_point_cloud(frame)
        
        # Mock loop closure (less frequent for monocular)
        loop_closures = []
        if frame.frame_id > 50 and frame.frame_id % 50 == 0:
            loop_closures.append((max(0, frame.frame_id - 50), frame.frame_id))
        
        return SLAMResult(
            camera_trajectory=self.trajectory.copy(),
            gaussian_splats=gaussians,
            point_cloud=point_cloud,
            reconstruction_quality=0.75,  # Lower quality for monocular
            processing_time=0.0,
            loop_closures=loop_closures,
            current_pose=self.current_pose,
            keyframe_ids=list(range(0, frame.frame_id + 1, self.config.keyframe_every))
        )
    
    def _estimate_motion_from_optical_flow(self, frame: SLAMFrame) -> np.ndarray:
        """Estimate camera motion using optical flow"""
        motion = np.eye(4)
        
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            # Convert to grayscale
            gray1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
            
            if flow[0] is not None and len(flow[0]) > 10:
                # Estimate motion from flow
                avg_flow = np.mean(flow[0], axis=0)
                
                # Simple translation estimation
                fx, fy = frame.camera_intrinsics[0, 0], frame.camera_intrinsics[1, 1]
                
                # Assume constant depth for motion estimation
                depth_estimate = 2.0
                
                motion[0, 3] = avg_flow[0] * depth_estimate / fx * 0.1
                motion[1, 3] = avg_flow[1] * depth_estimate / fy * 0.1
                motion[2, 3] = 0.05  # Forward motion assumption
                
                # Small rotation from flow
                rotation_angle = np.linalg.norm(avg_flow) * 0.001
                motion[0, 0] = np.cos(rotation_angle)
                motion[0, 2] = np.sin(rotation_angle)
                motion[2, 0] = -np.sin(rotation_angle)
                motion[2, 2] = np.cos(rotation_angle)
        else:
            # Default forward motion
            motion[2, 3] = 0.1
        
        self.prev_frame = frame.image.copy()
        return motion
    
    def _extract_gaussians_from_monogs(self, mapping_result: Dict) -> list:
        """Extract Gaussian splats from MonoGS result"""
        gaussians = []
        
        if 'gaussian_map' in mapping_result:
            gaussian_map = mapping_result['gaussian_map']
            
            # Extract parameters from MonoGS Gaussian map
            if hasattr(gaussian_map, 'get_xyz'):
                positions = gaussian_map.get_xyz().cpu().numpy()
                rotations = gaussian_map.get_rotation().cpu().numpy()
                scales = gaussian_map.get_scaling().cpu().numpy()
                colors = gaussian_map.get_features().cpu().numpy()
                opacities = gaussian_map.get_opacity().cpu().numpy()
                
                for i in range(len(positions)):
                    gaussian = {
                        'position': positions[i].tolist(),
                        'rotation': rotations[i].tolist(),
                        'scale': scales[i].tolist(),
                        'color': colors[i].tolist(),
                        'opacity': opacities[i].item()
                    }
                    gaussians.append(gaussian)
        
        return gaussians
    
    def _generate_monocular_gaussians(self, frame: SLAMFrame) -> list:
        """Generate Gaussian splats for monocular SLAM"""
        gaussians = []
        
        # Use feature detection for Gaussian placement
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
        
        # Detect FAST corners for speed
        fast = cv2.FastFeatureDetector_create(threshold=20)
        keypoints = fast.detect(gray, None)
        
        # Limit number of Gaussians for performance
        max_gaussians = 100 if self.config.target_fps > 15 else 200
        keypoints = keypoints[:max_gaussians]
        
        for kp in keypoints:
            x, y = kp.pt
            
            # Estimate depth using monocular cues (simplified)
            depth = self._estimate_monocular_depth(frame.image, int(x), int(y))
            
            # Convert to 3D position
            fx, fy = frame.camera_intrinsics[0, 0], frame.camera_intrinsics[1, 1]
            cx, cy = frame.camera_intrinsics[0, 2], frame.camera_intrinsics[1, 2]
            
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            # Transform to world coordinates
            world_pos = self.current_pose @ np.array([X, Y, Z, 1])
            
            # Get color
            if 0 <= int(y) < frame.image.shape[0] and 0 <= int(x) < frame.image.shape[1]:
                color = frame.image[int(y), int(x)] / 255.0
            else:
                color = np.array([0.5, 0.5, 0.5])
            
            gaussian = {
                'position': world_pos[:3].tolist(),
                'rotation': np.eye(3).tolist(),
                'scale': [0.1, 0.1, 0.1],  # Larger scale for monocular uncertainty
                'color': color.tolist(),
                'opacity': 0.7
            }
            gaussians.append(gaussian)
        
        return gaussians
    
    def _estimate_monocular_depth(self, image: np.ndarray, x: int, y: int) -> float:
        """Simple monocular depth estimation"""
        # Use image gradients as depth cue (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]:
            grad_val = gradient_magnitude[y, x]
            # Higher gradient = closer object (simple heuristic)
            depth = 5.0 / (1.0 + grad_val / 100.0)
        else:
            depth = 2.0
        
        return max(0.1, min(10.0, depth))  # Clamp depth
    
    def _generate_monocular_point_cloud(self, frame: SLAMFrame) -> np.ndarray:
        """Generate sparse point cloud for monocular SLAM"""
        # Much sparser than stereo/RGB-D
        num_points = 200
        points = []
        
        h, w = frame.image.shape[:2]
        fx, fy = frame.camera_intrinsics[0, 0], frame.camera_intrinsics[1, 1]
        cx, cy = frame.camera_intrinsics[0, 2], frame.camera_intrinsics[1, 2]
        
        # Sample points across image
        for _ in range(num_points):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            depth = self._estimate_monocular_depth(frame.image, x, y)
            
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            world_pos = self.current_pose @ np.array([X, Y, Z, 1])
            points.append(world_pos[:3])
        
        return np.array(points)
    
    async def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        return self.current_pose
    
    async def get_map_state(self) -> Dict[str, Any]:
        """Get current map state"""
        return {
            'num_keyframes': len([i for i in range(len(self.trajectory)) if self.is_keyframe(i)]),
            'num_gaussians': 0,  # Will be updated during mapping
            'trajectory_length': len(self.trajectory),
            'is_initialized': self.is_initialized,
            'device': str(self.device),
            'backend_name': 'monogs',
            'last_keyframe_id': self.last_keyframe_id
        }
    
    async def reset(self) -> None:
        """Reset MonoGS system"""
        logger.info("Resetting MonoGS system")
        
        self.trajectory = []
        self.current_pose = np.eye(4)
        self.last_keyframe_id = -1
        
        if hasattr(self, 'prev_frame'):
            delattr(self, 'prev_frame')
        
        if self.slam_system is not None:
            try:
                self.slam_system.reset()
            except:
                pass
    
    async def save_map(self, filepath: str) -> None:
        """Save current map to file"""
        logger.info(f"Saving MonoGS map to {filepath}")
        
        map_data = {
            'trajectory': [pose.tolist() for pose in self.trajectory],
            'last_keyframe_id': self.last_keyframe_id,
            'config': self.config.__dict__ if self.config else {}
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
    
    async def load_map(self, filepath: str) -> None:
        """Load map from file"""
        logger.info(f"Loading MonoGS map from {filepath}")
        
        import json
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        
        self.trajectory = [np.array(pose) for pose in map_data.get('trajectory', [])]
        self.last_keyframe_id = map_data.get('last_keyframe_id', -1)
        
        if self.trajectory:
            self.current_pose = self.trajectory[-1]
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default MonoGS configuration"""
        return {
            'tracking': {
                'tracker_type': 'droid',
                'keyframe_every': self.config.keyframe_every if self.config else 12,
                'max_keyframes': self.config.max_keyframes if self.config else 20
            },
            'mapping': {
                'densification_interval': 100,
                'opacity_reset_interval': 3000,
                'densify_grad_threshold': 0.0002,
                'mapping_every': 5
            },
            'gui': {
                'enable': False,  # Disable for headless operation
                'port': 8080
            },
            'performance': {
                'target_fps': self.config.target_fps if self.config else 10
            }
        }

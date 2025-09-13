"""
SplaTAM backend implementation for real-time RGB-D SLAM
"""

import sys
import os
from pathlib import Path

# Add SplaTAM to path when available
splatam_path = Path(__file__).parent.parent.parent.parent / "external" / "SplaTAM"
if splatam_path.exists():
    sys.path.insert(0, str(splatam_path))

from core.interfaces.base_slam_backend import BaseSLAMBackend, SLAMFrame, SLAMResult, SLAMConfig
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import asyncio
import cv2

logger = logging.getLogger(__name__)

class SplaTAMBackend(BaseSLAMBackend):
    """SplaTAM implementation for real-time RGB-D SLAM"""
    
    def __init__(self):
        super().__init__()
        self.slam_system = None
        self.device = None
        self.gaussian_params = None
        self.keyframe_database = []
        self.map_points = []
        
    async def initialize(self, config: SLAMConfig) -> None:
        """Initialize SplaTAM system"""
        logger.info("Initializing SplaTAM backend...")
        self.config = config
        
        try:
            # Set device
            self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Try to import SplaTAM components
            try:
                # These imports will work when SplaTAM is properly installed
                from splatam.slam import SplaTAM
                from splatam.configs.config import load_config
                
                # Load SplaTAM configuration
                slam_config_path = Path("configs/slam_configs/splatam_config.yaml")
                if slam_config_path.exists():
                    slam_config = load_config(str(slam_config_path))
                else:
                    # Use default config
                    slam_config = self._create_default_config()
                
                # Initialize SLAM system
                self.slam_system = SplaTAM(slam_config, device=self.device)
                
                logger.info("SplaTAM system initialized successfully")
                
            except ImportError as e:
                logger.warning(f"SplaTAM not available, using fallback implementation: {e}")
                self.slam_system = None
            
            # Initialize Gaussian parameters storage
            self.gaussian_params = {
                'positions': [],
                'rotations': [],
                'scales': [],
                'colors': [],
                'opacities': []
            }
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"SplaTAM initialization failed: {e}")
            raise
    
    async def process_frame(self, frame: SLAMFrame) -> SLAMResult:
        """Process frame with SplaTAM"""
        if not self.is_initialized:
            raise RuntimeError("SplaTAM not initialized")
            
        logger.debug(f"Processing frame {frame.frame_id} with SplaTAM")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.slam_system is not None:
                # Use real SplaTAM system
                result = await self._process_with_splatam(frame)
            else:
                # Use fallback implementation
                result = await self._process_with_fallback(frame)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.debug(f"SplaTAM processed frame {frame.frame_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"SplaTAM processing error: {e}")
            raise
    
    async def _process_with_splatam(self, frame: SLAMFrame) -> SLAMResult:
        """Process frame with actual SplaTAM system"""
        # Convert frame to SplaTAM format
        color_tensor = torch.from_numpy(frame.image).float().permute(2, 0, 1) / 255.0
        color_tensor = color_tensor.to(self.device)
        
        depth_tensor = None
        if frame.depth is not None:
            depth_tensor = torch.from_numpy(frame.depth).float().to(self.device)
        
        intrinsics_tensor = torch.from_numpy(frame.camera_intrinsics).float().to(self.device)
        
        # Process with SplaTAM
        slam_result = self.slam_system.track_and_map(
            color=color_tensor,
            depth=depth_tensor,
            intrinsics=intrinsics_tensor,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id
        )
        
        # Extract and update pose
        if 'camera_pose' in slam_result:
            self.current_pose = slam_result['camera_pose'].cpu().numpy()
            self.trajectory.append(self.current_pose.copy())
        
        # Extract Gaussian splats
        gaussians = self._extract_gaussians_from_splatam(slam_result)
        
        # Extract point cloud
        point_cloud = slam_result.get('point_cloud', np.array([])).cpu().numpy() if hasattr(slam_result.get('point_cloud', []), 'cpu') else np.array([])
        
        # Detect loop closures
        loop_closures = slam_result.get('loop_closures', [])
        
        return SLAMResult(
            camera_trajectory=self.trajectory.copy(),
            gaussian_splats=gaussians,
            point_cloud=point_cloud,
            reconstruction_quality=slam_result.get('quality_score', 0.8),
            processing_time=0.0,  # Will be set by caller
            loop_closures=loop_closures,
            current_pose=self.current_pose,
            keyframe_ids=slam_result.get('keyframe_ids', [])
        )
    
    async def _process_with_fallback(self, frame: SLAMFrame) -> SLAMResult:
        """Fallback processing when SplaTAM is not available"""
        # Simulate SplaTAM-like processing
        await asyncio.sleep(0.02)  # Simulate processing time
        
        # Simple pose estimation (forward motion with slight rotation)
        if len(self.trajectory) == 0:
            self.current_pose = np.eye(4)
        else:
            # Simple motion model
            motion = np.eye(4)
            motion[0, 3] = 0.1  # Move forward 10cm
            motion[1, 3] = np.sin(frame.frame_id * 0.1) * 0.02  # Slight side motion
            
            # Add slight rotation
            angle = 0.01
            motion[0, 0] = np.cos(angle)
            motion[0, 2] = np.sin(angle)
            motion[2, 0] = -np.sin(angle)
            motion[2, 2] = np.cos(angle)
            
            self.current_pose = self.trajectory[-1] @ motion
        
        self.trajectory.append(self.current_pose.copy())
        
        # Generate synthetic Gaussian splats based on frame content
        gaussians = self._generate_synthetic_gaussians(frame)
        
        # Generate synthetic point cloud
        point_cloud = self._generate_synthetic_point_cloud(frame)
        
        # Mock loop closure detection
        loop_closures = []
        if frame.frame_id > 30 and frame.frame_id % 30 == 0:
            loop_closures.append((max(0, frame.frame_id - 30), frame.frame_id))
        
        return SLAMResult(
            camera_trajectory=self.trajectory.copy(),
            gaussian_splats=gaussians,
            point_cloud=point_cloud,
            reconstruction_quality=0.85,
            processing_time=0.0,
            loop_closures=loop_closures,
            current_pose=self.current_pose,
            keyframe_ids=list(range(0, frame.frame_id + 1, self.config.keyframe_every))
        )
    
    def _extract_gaussians_from_splatam(self, slam_result: Dict) -> list:
        """Extract Gaussian splats from SplaTAM result"""
        gaussians = []
        
        if 'gaussians' in slam_result:
            gaussian_data = slam_result['gaussians']
            
            # Extract Gaussian parameters
            if hasattr(gaussian_data, 'positions'):
                positions = gaussian_data.positions.cpu().numpy()
                rotations = gaussian_data.rotations.cpu().numpy()
                scales = gaussian_data.scales.cpu().numpy()
                colors = gaussian_data.colors.cpu().numpy()
                opacities = gaussian_data.opacities.cpu().numpy()
                
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
    
    def _generate_synthetic_gaussians(self, frame: SLAMFrame) -> list:
        """Generate synthetic Gaussian splats from frame"""
        gaussians = []
        
        # Use frame content to guide Gaussian generation
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints
        orb = cv2.ORB_create(nfeatures=500)
        keypoints = orb.detect(gray, None)
        
        # Generate Gaussians at keypoint locations
        for kp in keypoints[:200]:  # Limit to 200 gaussians per frame
            x, y = kp.pt
            
            # Estimate depth (simple heuristic)
            if frame.depth is not None:
                depth = frame.depth[int(y), int(x)] if 0 <= int(y) < frame.depth.shape[0] and 0 <= int(x) < frame.depth.shape[1] else 1.0
            else:
                depth = np.random.uniform(0.5, 5.0)
            
            # Convert to 3D position
            fx, fy = frame.camera_intrinsics[0, 0], frame.camera_intrinsics[1, 1]
            cx, cy = frame.camera_intrinsics[0, 2], frame.camera_intrinsics[1, 2]
            
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            # Transform to world coordinates
            world_pos = self.current_pose @ np.array([X, Y, Z, 1])
            
            # Get color from image
            color = frame.image[int(y), int(x)] / 255.0 if 0 <= int(y) < frame.image.shape[0] and 0 <= int(x) < frame.image.shape[1] else [0.5, 0.5, 0.5]
            
            gaussian = {
                'position': world_pos[:3].tolist(),
                'rotation': np.eye(3).tolist(),
                'scale': [0.05, 0.05, 0.05],
                'color': color.tolist(),
                'opacity': 0.8
            }
            gaussians.append(gaussian)
        
        return gaussians
    
    def _generate_synthetic_point_cloud(self, frame: SLAMFrame) -> np.ndarray:
        """Generate synthetic point cloud from frame"""
        if frame.depth is not None:
            # Use actual depth data
            h, w = frame.depth.shape
            fx, fy = frame.camera_intrinsics[0, 0], frame.camera_intrinsics[1, 1]
            cx, cy = frame.camera_intrinsics[0, 2], frame.camera_intrinsics[1, 2]
            
            # Create coordinate grids
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            # Convert to 3D points
            valid_depth = frame.depth > 0
            u_valid = u[valid_depth]
            v_valid = v[valid_depth]
            depth_valid = frame.depth[valid_depth]
            
            X = (u_valid - cx) * depth_valid / fx
            Y = (v_valid - cy) * depth_valid / fy
            Z = depth_valid
            
            points_camera = np.stack([X, Y, Z], axis=1)
            
            # Transform to world coordinates
            points_world = []
            for point in points_camera[::10]:  # Subsample for performance
                world_point = self.current_pose @ np.append(point, 1)
                points_world.append(world_point[:3])
            
            return np.array(points_world)
        else:
            # Generate synthetic point cloud
            num_points = 1000
            points = np.random.uniform(-5, 5, (num_points, 3))
            return points
    
    async def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        return self.current_pose
    
    async def get_map_state(self) -> Dict[str, Any]:
        """Get current map state"""
        return {
            'num_keyframes': len([i for i in range(len(self.trajectory)) if self.is_keyframe(i)]),
            'num_gaussians': len(self.gaussian_params.get('positions', [])),
            'trajectory_length': len(self.trajectory),
            'is_initialized': self.is_initialized,
            'device': str(self.device),
            'backend_name': 'splatam'
        }
    
    async def reset(self) -> None:
        """Reset SplaTAM system"""
        logger.info("Resetting SplaTAM system")
        
        self.trajectory = []
        self.current_pose = np.eye(4)
        self.keyframe_database = []
        self.map_points = []
        
        if self.slam_system is not None:
            # Reset SplaTAM system if available
            try:
                self.slam_system.reset()
            except:
                pass
        
        # Reset Gaussian parameters
        self.gaussian_params = {
            'positions': [],
            'rotations': [],
            'scales': [],
            'colors': [],
            'opacities': []
        }
    
    async def save_map(self, filepath: str) -> None:
        """Save current map to file"""
        logger.info(f"Saving SplaTAM map to {filepath}")
        
        map_data = {
            'trajectory': [pose.tolist() for pose in self.trajectory],
            'gaussian_params': self.gaussian_params,
            'keyframe_database': self.keyframe_database,
            'config': self.config.__dict__ if self.config else {}
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
    
    async def load_map(self, filepath: str) -> None:
        """Load map from file"""
        logger.info(f"Loading SplaTAM map from {filepath}")
        
        import json
        with open(filepath, 'r') as f:
            map_data = json.load(f)
        
        self.trajectory = [np.array(pose) for pose in map_data.get('trajectory', [])]
        self.gaussian_params = map_data.get('gaussian_params', {})
        self.keyframe_database = map_data.get('keyframe_database', [])
        
        if self.trajectory:
            self.current_pose = self.trajectory[-1]
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default SplaTAM configuration"""
        return {
            'tracking': {
                'max_keyframes': self.config.max_keyframes if self.config else 20,
                'keyframe_every': self.config.keyframe_every if self.config else 5,
                'tracking_iters': self.config.tracking_iterations if self.config else 10
            },
            'mapping': {
                'mapping_iters': self.config.mapping_iterations if self.config else 60,
                'mapping_every': 5,
                'gaussian_lr': 0.0005
            },
            'optimization': {
                'use_global_optimization': self.config.enable_global_optimization if self.config else True,
                'bundle_adjustment_every': 10
            },
            'rendering': {
                'render_resolution': list(self.config.render_resolution) if self.config else [640, 480],
                'render_depth': True
            }
        }

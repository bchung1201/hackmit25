"""
Base SLAM backend interface for unified SLAM integration
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio

@dataclass
class SLAMFrame:
    """Unified frame representation for SLAM"""
    image: np.ndarray
    timestamp: float
    frame_id: int
    camera_intrinsics: np.ndarray
    depth: Optional[np.ndarray] = None
    pose_estimate: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate frame data"""
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Image must be RGB with shape (H, W, 3)")
        if self.camera_intrinsics.shape != (3, 3):
            raise ValueError("Camera intrinsics must be 3x3 matrix")
        if self.depth is not None and self.depth.shape[:2] != self.image.shape[:2]:
            raise ValueError("Depth map must match image dimensions")

@dataclass 
class SLAMResult:
    """Unified SLAM result representation"""
    camera_trajectory: List[np.ndarray]
    gaussian_splats: List[Dict[str, Any]]
    point_cloud: np.ndarray
    reconstruction_quality: float
    processing_time: float
    loop_closures: List[Tuple[int, int]]
    current_pose: np.ndarray
    keyframe_ids: List[int]
    map_points: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None

@dataclass
class SLAMConfig:
    """SLAM configuration parameters"""
    backend_name: str
    device: str = "cuda"
    max_keyframes: int = 50
    keyframe_every: int = 5
    tracking_iterations: int = 10
    mapping_iterations: int = 60
    enable_loop_closure: bool = True
    enable_global_optimization: bool = True
    render_resolution: Tuple[int, int] = (640, 480)
    target_fps: float = 30.0
    
class BaseSLAMBackend(ABC):
    """Abstract base class for all SLAM backends"""
    
    def __init__(self):
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.config: Optional[SLAMConfig] = None
    
    @abstractmethod
    async def initialize(self, config: SLAMConfig) -> None:
        """Initialize the SLAM backend"""
        pass
    
    @abstractmethod  
    async def process_frame(self, frame: SLAMFrame) -> SLAMResult:
        """Process a single frame"""
        pass
    
    @abstractmethod
    async def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        pass
    
    @abstractmethod
    async def get_map_state(self) -> Dict[str, Any]:
        """Get current map state"""
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        """Reset SLAM system"""
        pass
    
    @abstractmethod
    async def save_map(self, filepath: str) -> None:
        """Save current map to file"""
        pass
    
    @abstractmethod
    async def load_map(self, filepath: str) -> None:
        """Load map from file"""
        pass
    
    # Common utility methods
    def is_keyframe(self, frame_id: int) -> bool:
        """Determine if frame should be a keyframe"""
        if not self.config:
            return frame_id % 5 == 0  # Default
        return frame_id % self.config.keyframe_every == 0
    
    def calculate_pose_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """Calculate distance between two poses"""
        translation_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        rotation_dist = np.linalg.norm(pose1[:3, :3] - pose2[:3, :3], 'fro')
        return translation_dist + 0.1 * rotation_dist
    
    def extract_camera_parameters(self, frame: SLAMFrame) -> Dict[str, Any]:
        """Extract camera parameters from frame"""
        fx = frame.camera_intrinsics[0, 0]
        fy = frame.camera_intrinsics[1, 1]
        cx = frame.camera_intrinsics[0, 2]
        cy = frame.camera_intrinsics[1, 2]
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': frame.image.shape[1],
            'height': frame.image.shape[0]
        }

class MockSLAMBackend(BaseSLAMBackend):
    """Mock SLAM backend for development and testing"""
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.mock_trajectory = []
        
    async def initialize(self, config: SLAMConfig) -> None:
        """Initialize mock SLAM backend"""
        self.config = config
        self.is_initialized = True
        print(f"Mock SLAM backend '{config.backend_name}' initialized")
    
    async def process_frame(self, frame: SLAMFrame) -> SLAMResult:
        """Generate mock SLAM result"""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Generate mock camera pose (simple forward motion with slight rotation)
        t = frame.frame_id * 0.1
        mock_pose = np.eye(4)
        mock_pose[0, 3] = t * 0.5  # Move forward
        mock_pose[1, 3] = np.sin(t) * 0.1  # Slight side motion
        mock_pose[2, 3] = 0.0  # Constant height
        
        # Add slight rotation
        angle = t * 0.05
        mock_pose[0, 0] = np.cos(angle)
        mock_pose[0, 2] = np.sin(angle)
        mock_pose[2, 0] = -np.sin(angle)
        mock_pose[2, 2] = np.cos(angle)
        
        self.current_pose = mock_pose
        self.mock_trajectory.append(mock_pose.copy())
        
        # Generate mock Gaussian splats
        num_gaussians = min(500, frame.frame_id * 10)
        mock_gaussians = []
        
        for i in range(num_gaussians):
            gaussian = {
                'position': [
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    np.random.uniform(-2, 2)
                ],
                'rotation': np.eye(3).tolist(),
                'scale': [0.1, 0.1, 0.1],
                'color': [
                    np.random.uniform(0, 1),
                    np.random.uniform(0, 1),
                    np.random.uniform(0, 1)
                ],
                'opacity': np.random.uniform(0.5, 1.0)
            }
            mock_gaussians.append(gaussian)
        
        # Generate mock point cloud
        point_cloud = np.random.uniform(-5, 5, (num_gaussians * 2, 3))
        
        # Mock loop closures (every 50 frames)
        loop_closures = []
        if frame.frame_id > 50 and frame.frame_id % 50 == 0:
            loop_closures.append((frame.frame_id - 50, frame.frame_id))
        
        result = SLAMResult(
            camera_trajectory=self.mock_trajectory.copy(),
            gaussian_splats=mock_gaussians,
            point_cloud=point_cloud,
            reconstruction_quality=0.85 + np.random.uniform(-0.1, 0.1),
            processing_time=0.01,
            loop_closures=loop_closures,
            current_pose=self.current_pose,
            keyframe_ids=list(range(0, frame.frame_id + 1, 5))
        )
        
        self.frame_count += 1
        return result
    
    async def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        return self.current_pose
    
    async def get_map_state(self) -> Dict[str, Any]:
        """Get current map state"""
        return {
            'num_keyframes': len(self.mock_trajectory) // 5,
            'num_map_points': len(self.mock_trajectory) * 10,
            'trajectory_length': len(self.mock_trajectory),
            'is_initialized': self.is_initialized
        }
    
    async def reset(self) -> None:
        """Reset mock SLAM system"""
        self.frame_count = 0
        self.mock_trajectory = []
        self.current_pose = np.eye(4)
    
    async def save_map(self, filepath: str) -> None:
        """Mock save map"""
        print(f"Mock: Saving map to {filepath}")
    
    async def load_map(self, filepath: str) -> None:
        """Mock load map"""
        print(f"Mock: Loading map from {filepath}")

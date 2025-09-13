"""
SLAM Manager for handling multiple SLAM backends and adaptive switching
"""

from typing import Dict, Any, Optional, Type, List
import logging
import asyncio
import numpy as np
import torch

from core.interfaces.base_slam_backend import BaseSLAMBackend, SLAMFrame, SLAMResult, SLAMConfig, MockSLAMBackend
from core.slam_backends.splatam.splatam_backend import SplaTAMBackend
from core.slam_backends.monogs.monogs_backend import MonoGSBackend

logger = logging.getLogger(__name__)

class SLAMManager:
    """Manages multiple SLAM backends and handles adaptive switching"""
    
    BACKENDS = {
        'splatam': SplaTAMBackend,
        'monogs': MonoGSBackend,
        'mock': MockSLAMBackend
    }
    
    def __init__(self, backend_type: str = 'auto'):
        self.backend_type = backend_type
        self.active_backend: Optional[BaseSLAMBackend] = None
        self.backend_configs: Dict[str, SLAMConfig] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.frame_count = 0
        self.last_switch_time = 0
        
    async def initialize(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Initialize SLAM manager with backend configs"""
        logger.info("Initializing SLAM Manager...")
        
        # Convert dict configs to SLAMConfig objects
        for backend_name, config_dict in configs.items():
            self.backend_configs[backend_name] = SLAMConfig(
                backend_name=backend_name,
                device=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                max_keyframes=config_dict.get('max_keyframes', 50),
                keyframe_every=config_dict.get('keyframe_every', 5),
                tracking_iterations=config_dict.get('tracking_iterations', 10),
                mapping_iterations=config_dict.get('mapping_iterations', 60),
                enable_loop_closure=config_dict.get('enable_loop_closure', True),
                enable_global_optimization=config_dict.get('enable_global_optimization', True),
                render_resolution=tuple(config_dict.get('render_resolution', [640, 480])),
                target_fps=config_dict.get('target_fps', 30.0)
            )
        
        # Auto-select backend if needed
        if self.backend_type == 'auto':
            self.backend_type = await self._auto_select_backend()
        
        # Initialize selected backend
        await self._initialize_backend(self.backend_type)
        
        logger.info(f"SLAM Manager initialized with backend: {self.backend_type}")
    
    async def _auto_select_backend(self) -> str:
        """Automatically select best backend based on hardware and requirements"""
        logger.info("Auto-selecting SLAM backend...")
        
        # Check hardware capabilities
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"CUDA available with {gpu_memory:.1f}GB GPU memory")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Check for RGB-D camera capability
        has_rgbd = self._has_rgbd_camera()
        
        # Selection logic
        if has_rgbd and has_cuda and gpu_memory > 4:
            # Use SplaTAM for RGB-D with good GPU
            selected = 'splatam'
            reason = "RGB-D camera detected with sufficient GPU memory"
        elif has_cuda and gpu_memory > 2:
            # Use MonoGS for monocular with decent GPU
            selected = 'monogs'
            reason = "Monocular camera with GPU acceleration"
        elif 'mock' in self.backend_configs:
            # Fallback to mock for development
            selected = 'mock'
            reason = "Fallback to mock backend for development"
        else:
            # Default to MonoGS (most compatible)
            selected = 'monogs'
            reason = "Default monocular backend"
        
        logger.info(f"Auto-selected '{selected}': {reason}")
        return selected
    
    def _has_rgbd_camera(self) -> bool:
        """Check if RGB-D camera is available"""
        # TODO: Implement actual RGB-D camera detection
        # For now, assume no RGB-D camera (monocular only)
        return False
    
    async def _initialize_backend(self, backend_name: str) -> None:
        """Initialize specific backend"""
        if backend_name not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend_name}. Available: {list(self.BACKENDS.keys())}")
        
        if backend_name not in self.backend_configs:
            # Create default config
            self.backend_configs[backend_name] = SLAMConfig(backend_name=backend_name)
        
        backend_class = self.BACKENDS[backend_name]
        self.active_backend = backend_class()
        
        config = self.backend_configs[backend_name]
        await self.active_backend.initialize(config)
        
        # Initialize performance tracking
        self.performance_metrics[backend_name] = []
        
        logger.info(f"Initialized SLAM backend: {backend_name}")
    
    async def process_frame(self, frame: SLAMFrame) -> SLAMResult:
        """Process frame with active backend and track performance"""
        if not self.active_backend:
            raise RuntimeError("No active SLAM backend")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process frame
            result = await self.active_backend.process_frame(frame)
            
            # Track performance
            processing_time = asyncio.get_event_loop().time() - start_time
            self.performance_metrics[self.backend_type].append(processing_time)
            
            # Keep only recent metrics (last 100 frames)
            if len(self.performance_metrics[self.backend_type]) > 100:
                self.performance_metrics[self.backend_type].pop(0)
            
            self.frame_count += 1
            
            # Check if backend switching is needed
            if self.frame_count % 30 == 0:  # Check every 30 frames
                await self._evaluate_backend_performance()
            
            return result
            
        except Exception as e:
            logger.error(f"SLAM processing error with {self.backend_type}: {e}")
            
            # Try to recover by switching backends
            if self.backend_type != 'mock':
                logger.info("Attempting recovery by switching to mock backend")
                await self.switch_backend('mock')
                return await self.active_backend.process_frame(frame)
            else:
                raise
    
    async def _evaluate_backend_performance(self) -> None:
        """Evaluate current backend performance and consider switching"""
        if not self.performance_metrics[self.backend_type]:
            return
        
        current_metrics = self.performance_metrics[self.backend_type]
        avg_processing_time = np.mean(current_metrics[-30:])  # Last 30 frames
        target_fps = self.backend_configs[self.backend_type].target_fps
        target_time = 1.0 / target_fps
        
        logger.debug(f"Backend {self.backend_type}: avg processing time {avg_processing_time:.3f}s (target: {target_time:.3f}s)")
        
        # Check if performance is degrading
        if avg_processing_time > target_time * 1.5:  # 50% slower than target
            logger.warning(f"Backend {self.backend_type} performance degraded: {avg_processing_time:.3f}s > {target_time:.3f}s")
            
            # Consider switching to faster backend
            if self.backend_type == 'splatam':
                await self._consider_backend_switch('monogs', 'Performance optimization')
            elif self.backend_type == 'monogs' and avg_processing_time > target_time * 2:
                await self._consider_backend_switch('mock', 'Severe performance issues')
    
    async def _consider_backend_switch(self, new_backend: str, reason: str) -> None:
        """Consider switching to a different backend"""
        current_time = asyncio.get_event_loop().time()
        
        # Avoid frequent switching (minimum 30 seconds between switches)
        if current_time - self.last_switch_time < 30:
            return
        
        if new_backend in self.backend_configs:
            logger.info(f"Switching backend from {self.backend_type} to {new_backend}: {reason}")
            await self.switch_backend(new_backend)
            self.last_switch_time = current_time
    
    async def switch_backend(self, new_backend: str) -> None:
        """Switch to different SLAM backend"""
        if new_backend == self.backend_type:
            logger.info(f"Already using backend: {new_backend}")
            return
        
        logger.info(f"Switching SLAM backend from {self.backend_type} to {new_backend}")
        
        old_backend = self.active_backend
        old_backend_name = self.backend_type
        
        try:
            # Save current state if possible
            current_pose = None
            trajectory = []
            if old_backend:
                try:
                    current_pose = await old_backend.get_current_pose()
                    map_state = await old_backend.get_map_state()
                    trajectory = getattr(old_backend, 'trajectory', [])
                except:
                    pass
            
            # Initialize new backend
            await self._initialize_backend(new_backend)
            self.backend_type = new_backend
            
            # Transfer state if possible
            if current_pose is not None and hasattr(self.active_backend, 'current_pose'):
                self.active_backend.current_pose = current_pose
                self.active_backend.trajectory = trajectory
            
            logger.info(f"Successfully switched to backend: {new_backend}")
            
        except Exception as e:
            logger.error(f"Backend switch failed: {e}")
            # Revert to old backend
            self.active_backend = old_backend
            self.backend_type = old_backend_name
            raise
    
    async def get_current_pose(self) -> np.ndarray:
        """Get current camera pose from active backend"""
        if not self.active_backend:
            return np.eye(4)
        return await self.active_backend.get_current_pose()
    
    async def get_map_state(self) -> Dict[str, Any]:
        """Get current map state from active backend"""
        if not self.active_backend:
            return {}
        
        state = await self.active_backend.get_map_state()
        
        # Add manager-specific information
        state.update({
            'active_backend': self.backend_type,
            'frame_count': self.frame_count,
            'available_backends': list(self.BACKENDS.keys()),
            'performance_metrics': {
                backend: {
                    'avg_processing_time': np.mean(metrics[-30:]) if metrics else 0.0,
                    'fps': 1.0 / np.mean(metrics[-30:]) if metrics else 0.0,
                    'sample_count': len(metrics)
                }
                for backend, metrics in self.performance_metrics.items()
            }
        })
        
        return state
    
    async def reset(self) -> None:
        """Reset active SLAM backend"""
        if self.active_backend:
            await self.active_backend.reset()
        
        self.frame_count = 0
        self.performance_metrics = {}
        self.last_switch_time = 0
    
    async def save_map(self, filepath: str) -> None:
        """Save current map from active backend"""
        if self.active_backend:
            await self.active_backend.save_map(filepath)
    
    async def load_map(self, filepath: str) -> None:
        """Load map into active backend"""
        if self.active_backend:
            await self.active_backend.load_map(filepath)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends"""
        return {
            'active_backend': self.backend_type,
            'available_backends': list(self.BACKENDS.keys()),
            'backend_configs': {
                name: {
                    'device': config.device,
                    'target_fps': config.target_fps,
                    'max_keyframes': config.max_keyframes
                }
                for name, config in self.backend_configs.items()
            },
            'performance_summary': {
                backend: {
                    'avg_fps': 1.0 / np.mean(metrics[-30:]) if metrics else 0.0,
                    'frames_processed': len(metrics)
                }
                for backend, metrics in self.performance_metrics.items()
            }
        }

class TrajectoryTracker:
    """Enhanced trajectory tracking with loop closure detection"""
    
    def __init__(self, max_trajectory_length: int = 1000):
        self.max_trajectory_length = max_trajectory_length
        self.trajectory: List[np.ndarray] = []
        self.keyframe_poses: List[np.ndarray] = []
        self.keyframe_ids: List[int] = []
        
    def add_pose(self, pose: np.ndarray, frame_id: int, is_keyframe: bool = False) -> None:
        """Add pose to trajectory"""
        self.trajectory.append(pose.copy())
        
        if is_keyframe:
            self.keyframe_poses.append(pose.copy())
            self.keyframe_ids.append(frame_id)
        
        # Maintain trajectory length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get full trajectory"""
        return self.trajectory.copy()
    
    def get_keyframe_trajectory(self) -> List[np.ndarray]:
        """Get keyframe trajectory"""
        return self.keyframe_poses.copy()
    
    def calculate_trajectory_length(self) -> float:
        """Calculate total trajectory length"""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            pos1 = self.trajectory[i-1][:3, 3]
            pos2 = self.trajectory[i][:3, 3]
            total_length += np.linalg.norm(pos2 - pos1)
        
        return total_length

class LoopClosureDetector:
    """Simple loop closure detection"""
    
    def __init__(self, distance_threshold: float = 1.0, angle_threshold: float = 0.5):
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.detected_loops: List[Tuple[int, int]] = []
        
    def detect_loop_closure(
        self, 
        current_pose: np.ndarray, 
        current_frame_id: int,
        trajectory: List[np.ndarray],
        min_loop_gap: int = 30
    ) -> List[Tuple[int, int]]:
        """Detect loop closures based on pose similarity"""
        new_loops = []
        
        # Check against poses that are far enough in time
        for i, past_pose in enumerate(trajectory[:-min_loop_gap]):
            if self._poses_similar(current_pose, past_pose):
                loop_pair = (i, current_frame_id)
                if loop_pair not in self.detected_loops:
                    new_loops.append(loop_pair)
                    self.detected_loops.append(loop_pair)
                    logger.info(f"Loop closure detected: frame {i} <-> frame {current_frame_id}")
        
        return new_loops
    
    def _poses_similar(self, pose1: np.ndarray, pose2: np.ndarray) -> bool:
        """Check if two poses are similar enough for loop closure"""
        # Translation distance
        trans_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        
        # Rotation distance (Frobenius norm of difference)
        rot_dist = np.linalg.norm(pose1[:3, :3] - pose2[:3, :3], 'fro')
        
        return trans_dist < self.distance_threshold and rot_dist < self.angle_threshold

class GlobalOptimizer:
    """Global pose graph optimization"""
    
    def __init__(self):
        self.optimization_interval = 50  # Optimize every 50 frames
        self.last_optimization = 0
        
    async def should_optimize(self, frame_count: int) -> bool:
        """Check if global optimization should be performed"""
        return frame_count - self.last_optimization >= self.optimization_interval
    
    async def optimize_trajectory(
        self, 
        trajectory: List[np.ndarray],
        loop_closures: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """Perform global trajectory optimization"""
        if not loop_closures:
            return trajectory
        
        logger.info(f"Performing global optimization with {len(loop_closures)} loop closures")
        
        # Simplified optimization (in practice, would use pose graph optimization)
        optimized_trajectory = trajectory.copy()
        
        # Apply simple correction based on loop closures
        for start_idx, end_idx in loop_closures:
            if start_idx < len(optimized_trajectory) and end_idx < len(optimized_trajectory):
                # Calculate drift
                start_pose = optimized_trajectory[start_idx]
                end_pose = optimized_trajectory[end_idx]
                
                # Simple drift correction (distribute error)
                drift = end_pose[:3, 3] - start_pose[:3, 3]
                num_poses = end_idx - start_idx
                
                for i in range(start_idx + 1, end_idx):
                    if i < len(optimized_trajectory):
                        correction_factor = (i - start_idx) / num_poses
                        correction = drift * correction_factor
                        optimized_trajectory[i][:3, 3] -= correction
        
        self.last_optimization = len(trajectory)
        return optimized_trajectory

"""
Unified SLAM Reconstructor - Replaces modal_3d_reconstruction.py
Integrates advanced Gaussian Splatting SLAM systems for real-time 3D reconstruction
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from core.pipeline.slam_manager import SLAMManager, TrajectoryTracker, LoopClosureDetector, GlobalOptimizer
from core.interfaces.base_slam_backend import SLAMFrame, SLAMResult, SLAMConfig

logger = logging.getLogger(__name__)

@dataclass
class Point3D:
    """Represents a 3D point with color and opacity (kept for compatibility)"""
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    opacity: float

@dataclass
class GaussianSplat:
    """Represents a Gaussian splat in 3D space (kept for compatibility)"""
    position: Point3D
    rotation: np.ndarray  # 3x3 rotation matrix
    scale: np.ndarray     # 3D scaling factors
    color: np.ndarray     # RGB color
    opacity: float

@dataclass
class ReconstructionResult:
    """Enhanced reconstruction result with SLAM integration"""
    gaussians: List[GaussianSplat]
    camera_poses: List[np.ndarray]
    point_cloud: np.ndarray
    reconstruction_quality: float
    processing_time: float
    # Enhanced SLAM features
    trajectory_length: float
    loop_closures: List[Tuple[int, int]]
    keyframe_poses: List[np.ndarray]
    map_confidence: float
    backend_used: str

class UnifiedSLAMReconstructor:
    """Unified SLAM reconstructor replacing Modal-based reconstruction"""
    
    def __init__(self, backend_type: str = "auto"):
        self.slam_manager = SLAMManager(backend_type)
        self.trajectory_tracker = TrajectoryTracker()
        self.loop_closure_detector = LoopClosureDetector()
        self.global_optimizer = GlobalOptimizer()
        
        self.is_initialized = False
        self.reconstruction_queue: List[Dict[str, Any]] = []
        self.is_processing = False
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = []
        self.quality_scores = []
        
    async def initialize(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Initialize SLAM reconstructor"""
        logger.info("Initializing Unified SLAM Reconstructor...")
        
        try:
            await self.slam_manager.initialize(configs)
            self.is_initialized = True
            logger.info("Unified SLAM Reconstructor initialized successfully")
            
        except Exception as e:
            logger.error(f"SLAM reconstructor initialization failed: {e}")
            raise
    
    async def reconstruct_scene(
        self,
        frames: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        camera_poses: Optional[List[np.ndarray]] = None
    ) -> ReconstructionResult:
        """Reconstruct scene using SLAM backend"""
        if not self.is_initialized:
            raise RuntimeError("SLAM reconstructor not initialized")
        
        logger.info(f"Processing {len(frames)} frames with SLAM")
        
        start_time = asyncio.get_event_loop().time()
        all_results = []
        
        try:
            for i, frame in enumerate(frames):
                # Create SLAM frame
                slam_frame = SLAMFrame(
                    image=frame,
                    timestamp=i * (1/30),  # Assume 30 FPS
                    frame_id=self.frame_count + i,
                    camera_intrinsics=camera_intrinsics,
                    pose_estimate=camera_poses[i] if camera_poses else None
                )
                
                # Process with SLAM
                result = await self.slam_manager.process_frame(slam_frame)
                all_results.append(result)
                
                # Update trajectory tracker
                self.trajectory_tracker.add_pose(
                    result.current_pose, 
                    slam_frame.frame_id,
                    is_keyframe=slam_frame.frame_id in result.keyframe_ids
                )
                
                # Detect loop closures
                new_loops = self.loop_closure_detector.detect_loop_closure(
                    result.current_pose,
                    slam_frame.frame_id,
                    self.trajectory_tracker.get_trajectory()
                )
                
                # Global optimization if needed
                if await self.global_optimizer.should_optimize(self.frame_count + i):
                    optimized_trajectory = await self.global_optimizer.optimize_trajectory(
                        self.trajectory_tracker.get_trajectory(),
                        self.loop_closure_detector.detected_loops
                    )
                    # Update trajectory tracker with optimized poses
                    self.trajectory_tracker.trajectory = optimized_trajectory
            
            self.frame_count += len(frames)
            
            # Combine results from all frames
            if all_results:
                final_result = await self._combine_slam_results(all_results)
                
                # Convert to legacy format for compatibility
                reconstruction_result = await self._convert_to_legacy_format(
                    final_result, 
                    asyncio.get_event_loop().time() - start_time
                )
                
                # Track performance
                self.processing_times.append(reconstruction_result.processing_time)
                self.quality_scores.append(reconstruction_result.reconstruction_quality)
                
                return reconstruction_result
            else:
                raise RuntimeError("No SLAM results generated")
                
        except Exception as e:
            logger.error(f"SLAM reconstruction failed: {e}")
            raise
    
    async def _combine_slam_results(self, results: List[SLAMResult]) -> SLAMResult:
        """Combine multiple SLAM results into a single result"""
        if not results:
            raise ValueError("No results to combine")
        
        # Use the last result as base
        final_result = results[-1]
        
        # Combine Gaussian splats from all results
        all_gaussians = []
        for result in results:
            all_gaussians.extend(result.gaussian_splats)
        
        # Remove duplicate Gaussians (simple distance-based deduplication)
        unique_gaussians = self._deduplicate_gaussians(all_gaussians)
        
        # Combine point clouds
        all_point_clouds = [result.point_cloud for result in results if len(result.point_cloud) > 0]
        combined_point_cloud = np.vstack(all_point_clouds) if all_point_clouds else np.array([])
        
        # Combine loop closures
        all_loop_closures = []
        for result in results:
            all_loop_closures.extend(result.loop_closures)
        
        # Update final result
        final_result.gaussian_splats = unique_gaussians
        final_result.point_cloud = combined_point_cloud
        final_result.loop_closures = list(set(all_loop_closures))  # Remove duplicates
        final_result.reconstruction_quality = np.mean([r.reconstruction_quality for r in results])
        
        return final_result
    
    def _deduplicate_gaussians(self, gaussians: List[Dict[str, Any]], distance_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Remove duplicate Gaussian splats based on position"""
        if not gaussians:
            return []
        
        unique_gaussians = []
        
        for gaussian in gaussians:
            pos = np.array(gaussian['position'])
            
            # Check if similar Gaussian already exists
            is_duplicate = False
            for existing in unique_gaussians:
                existing_pos = np.array(existing['position'])
                if np.linalg.norm(pos - existing_pos) < distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_gaussians.append(gaussian)
        
        logger.debug(f"Deduplicated {len(gaussians)} -> {len(unique_gaussians)} Gaussians")
        return unique_gaussians
    
    async def _convert_to_legacy_format(self, slam_result: SLAMResult, processing_time: float) -> ReconstructionResult:
        """Convert SLAM result to legacy ReconstructionResult format"""
        # Convert Gaussian splats to legacy format
        legacy_gaussians = []
        for gaussian_dict in slam_result.gaussian_splats:
            position = Point3D(
                x=gaussian_dict['position'][0],
                y=gaussian_dict['position'][1],
                z=gaussian_dict['position'][2],
                r=gaussian_dict['color'][0],
                g=gaussian_dict['color'][1],
                b=gaussian_dict['color'][2],
                opacity=gaussian_dict['opacity']
            )
            
            legacy_gaussian = GaussianSplat(
                position=position,
                rotation=np.array(gaussian_dict['rotation']),
                scale=np.array(gaussian_dict['scale']),
                color=np.array(gaussian_dict['color']),
                opacity=gaussian_dict['opacity']
            )
            legacy_gaussians.append(legacy_gaussian)
        
        # Calculate additional metrics
        trajectory_length = self.trajectory_tracker.calculate_trajectory_length()
        keyframe_poses = self.trajectory_tracker.get_keyframe_trajectory()
        
        # Calculate map confidence based on various factors
        map_confidence = self._calculate_map_confidence(slam_result)
        
        # Get backend information
        map_state = await self.slam_manager.get_map_state()
        backend_used = map_state.get('active_backend', 'unknown')
        
        return ReconstructionResult(
            gaussians=legacy_gaussians,
            camera_poses=slam_result.camera_trajectory,
            point_cloud=slam_result.point_cloud,
            reconstruction_quality=slam_result.reconstruction_quality,
            processing_time=processing_time,
            trajectory_length=trajectory_length,
            loop_closures=slam_result.loop_closures,
            keyframe_poses=keyframe_poses,
            map_confidence=map_confidence,
            backend_used=backend_used
        )
    
    def _calculate_map_confidence(self, slam_result: SLAMResult) -> float:
        """Calculate overall map confidence score"""
        factors = []
        
        # Factor 1: Reconstruction quality
        factors.append(slam_result.reconstruction_quality)
        
        # Factor 2: Number of loop closures (more is better, up to a point)
        loop_closure_score = min(1.0, len(slam_result.loop_closures) / 5.0)
        factors.append(loop_closure_score)
        
        # Factor 3: Trajectory consistency
        if len(slam_result.camera_trajectory) > 1:
            pose_distances = []
            for i in range(1, len(slam_result.camera_trajectory)):
                pos1 = slam_result.camera_trajectory[i-1][:3, 3]
                pos2 = slam_result.camera_trajectory[i][:3, 3]
                pose_distances.append(np.linalg.norm(pos2 - pos1))
            
            # Penalize large jumps in pose
            avg_distance = np.mean(pose_distances)
            consistency_score = max(0.0, 1.0 - avg_distance / 1.0)  # Normalize by 1 meter
            factors.append(consistency_score)
        
        # Factor 4: Number of Gaussian splats (more detail is better)
        gaussian_score = min(1.0, len(slam_result.gaussian_splats) / 1000.0)
        factors.append(gaussian_score)
        
        return np.mean(factors)
    
    async def queue_reconstruction(
        self,
        frames: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        camera_poses: Optional[List[np.ndarray]] = None
    ) -> None:
        """Queue a reconstruction job for batch processing"""
        job = {
            "frames": frames,
            "camera_intrinsics": camera_intrinsics,
            "camera_poses": camera_poses,
            "timestamp": asyncio.get_event_loop().time()
        }
        self.reconstruction_queue.append(job)
        logger.info(f"Queued SLAM reconstruction job (queue size: {len(self.reconstruction_queue)})")
    
    async def process_queue(self) -> None:
        """Process queued reconstruction jobs"""
        if self.is_processing or not self.reconstruction_queue:
            return
        
        self.is_processing = True
        
        try:
            while self.reconstruction_queue:
                job = self.reconstruction_queue.pop(0)
                
                result = await self.reconstruct_scene(
                    frames=job["frames"],
                    camera_intrinsics=job["camera_intrinsics"],
                    camera_poses=job["camera_poses"]
                )
                
                # Process result (e.g., update 3D model, trigger segmentation)
                await self._process_reconstruction_result(result)
                
        finally:
            self.is_processing = False
    
    async def _process_reconstruction_result(self, result: ReconstructionResult) -> None:
        """Process a completed reconstruction result"""
        logger.info(f"Processing SLAM reconstruction result:")
        logger.info(f"  - Backend: {result.backend_used}")
        logger.info(f"  - Gaussians: {len(result.gaussians)}")
        logger.info(f"  - Trajectory length: {result.trajectory_length:.2f}m")
        logger.info(f"  - Loop closures: {len(result.loop_closures)}")
        logger.info(f"  - Map confidence: {result.map_confidence:.2f}")
        logger.info(f"  - Processing time: {result.processing_time:.3f}s")
        
        # TODO: Integrate with segmentation and VLM pipelines
        # Could trigger:
        # - 3D object segmentation on the Gaussian splats
        # - Scene understanding with spatial context
        # - Navigation path planning with trajectory
    
    async def get_slam_status(self) -> Dict[str, Any]:
        """Get comprehensive SLAM status"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        map_state = await self.slam_manager.get_map_state()
        
        # Add reconstructor-specific stats
        avg_processing_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0.0
        avg_quality = np.mean(self.quality_scores[-30:]) if self.quality_scores else 0.0
        
        status = {
            "status": "running" if self.is_initialized else "stopped",
            "backend_info": self.slam_manager.get_backend_info(),
            "trajectory": {
                "length": self.trajectory_tracker.calculate_trajectory_length(),
                "poses": len(self.trajectory_tracker.trajectory),
                "keyframes": len(self.trajectory_tracker.keyframe_poses)
            },
            "loop_closures": len(self.loop_closure_detector.detected_loops),
            "performance": {
                "avg_processing_time": avg_processing_time,
                "avg_fps": 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0,
                "avg_quality": avg_quality
            },
            "queue_status": {
                "queue_size": len(self.reconstruction_queue),
                "is_processing": self.is_processing
            }
        }
        
        status.update(map_state)
        return status
    
    async def switch_backend(self, backend_name: str) -> None:
        """Switch SLAM backend"""
        await self.slam_manager.switch_backend(backend_name)
    
    async def reset(self) -> None:
        """Reset SLAM system"""
        logger.info("Resetting Unified SLAM Reconstructor")
        
        await self.slam_manager.reset()
        self.trajectory_tracker = TrajectoryTracker()
        self.loop_closure_detector = LoopClosureDetector()
        self.global_optimizer = GlobalOptimizer()
        
        self.reconstruction_queue = []
        self.is_processing = False
        self.frame_count = 0
        self.processing_times = []
        self.quality_scores = []
    
    async def save_map(self, filepath: str) -> None:
        """Save SLAM map and trajectory"""
        logger.info(f"Saving SLAM map to {filepath}")
        
        # Save SLAM backend map
        await self.slam_manager.save_map(filepath)
        
        # Save additional trajectory data
        trajectory_file = str(Path(filepath).with_suffix('.trajectory.json'))
        
        import json
        trajectory_data = {
            'trajectory': [pose.tolist() for pose in self.trajectory_tracker.trajectory],
            'keyframe_poses': [pose.tolist() for pose in self.trajectory_tracker.keyframe_poses],
            'keyframe_ids': self.trajectory_tracker.keyframe_ids,
            'loop_closures': self.loop_closure_detector.detected_loops,
            'trajectory_length': self.trajectory_tracker.calculate_trajectory_length(),
            'frame_count': self.frame_count
        }
        
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        logger.info(f"Saved trajectory data to {trajectory_file}")
    
    async def load_map(self, filepath: str) -> None:
        """Load SLAM map and trajectory"""
        logger.info(f"Loading SLAM map from {filepath}")
        
        # Load SLAM backend map
        await self.slam_manager.load_map(filepath)
        
        # Load trajectory data if available
        trajectory_file = str(Path(filepath).with_suffix('.trajectory.json'))
        
        if Path(trajectory_file).exists():
            import json
            with open(trajectory_file, 'r') as f:
                trajectory_data = json.load(f)
            
            # Restore trajectory tracker
            self.trajectory_tracker.trajectory = [np.array(pose) for pose in trajectory_data.get('trajectory', [])]
            self.trajectory_tracker.keyframe_poses = [np.array(pose) for pose in trajectory_data.get('keyframe_poses', [])]
            self.trajectory_tracker.keyframe_ids = trajectory_data.get('keyframe_ids', [])
            
            # Restore loop closure detector
            self.loop_closure_detector.detected_loops = trajectory_data.get('loop_closures', [])
            
            self.frame_count = trajectory_data.get('frame_count', 0)
            
            logger.info(f"Loaded trajectory data from {trajectory_file}")

# Mock implementation for development without full SLAM systems
class MockSLAMReconstructor(UnifiedSLAMReconstructor):
    """Mock SLAM reconstructor for development/testing"""
    
    def __init__(self, backend_type: str = "mock"):
        super().__init__(backend_type)
    
    async def initialize(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize mock SLAM reconstructor"""
        if configs is None:
            configs = {
                'mock': {
                    'device': 'cpu',
                    'target_fps': 30.0
                }
            }
        
        await super().initialize(configs)
        logger.info("Mock SLAM Reconstructor initialized")

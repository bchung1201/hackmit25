"""
SAM (Segment Anything Model) integration for object segmentation
Handles real-time object segmentation on video frames
"""

import cv2
import numpy as np
import torch
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class SegmentationMask:
    """Represents a segmentation mask with metadata"""
    mask: np.ndarray
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area: int
    category: Optional[str] = None

@dataclass
class SegmentationResult:
    """Result of segmentation on a frame"""
    masks: List[SegmentationMask]
    frame_id: int
    timestamp: float
    processing_time: float

class SAMSegmenter:
    """
    SAM-based object segmentation for video frames
    """
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = "sam_vit_h_4b8939.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.predictor = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the SAM model"""
        logger.info(f"Initializing SAM model ({self.model_type}) on {self.device}")
        
        try:
            # Import SAM components
            from segment_anything import sam_model_registry, SamPredictor
            
            # Load SAM model
            self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.model.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(self.model)
            
            self.is_initialized = True
            logger.info("SAM model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM model: {e}")
            raise
    
    async def segment_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        auto_segment: bool = True,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None
    ) -> SegmentationResult:
        """
        Segment objects in a video frame
        
        Args:
            frame: Input video frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            auto_segment: Whether to use automatic segmentation
            points: Manual points for segmentation (if not auto_segment)
            labels: Labels for manual points (1 for foreground, 0 for background)
            
        Returns:
            SegmentationResult object
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"Segmenting frame {frame_id}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Set image for predictor
            self.predictor.set_image(frame)
            
            if auto_segment:
                # Use automatic segmentation
                masks, scores, logits = self.predictor.predict()
            else:
                # Use manual points
                if points is None or labels is None:
                    raise ValueError("Points and labels required for manual segmentation")
                
                points_array = np.array(points)
                labels_array = np.array(labels)
                
                masks, scores, logits = self.predictor.predict(
                    point_coords=points_array,
                    point_labels=labels_array
                )
            
            # Process masks
            segmentation_masks = []
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score > 0.5:  # Filter low-confidence masks
                    # Calculate bounding box
                    bbox = self._calculate_bbox(mask)
                    
                    # Calculate area
                    area = np.sum(mask)
                    
                    segmentation_mask = SegmentationMask(
                        mask=mask,
                        confidence=float(score),
                        bbox=bbox,
                        area=int(area)
                    )
                    segmentation_masks.append(segmentation_mask)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = SegmentationResult(
                masks=segmentation_masks,
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time=processing_time
            )
            
            logger.info(f"Segmented {len(segmentation_masks)} objects in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            raise
    
    def _calculate_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box from mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    async def segment_frame_sequence(
        self,
        frames: List[np.ndarray],
        frame_ids: List[int],
        timestamps: List[float]
    ) -> List[SegmentationResult]:
        """Segment a sequence of frames"""
        logger.info(f"Segmenting sequence of {len(frames)} frames")
        
        results = []
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            result = await self.segment_frame(frame, frame_id, timestamp)
            results.append(result)
        
        return results
    
    def visualize_segmentation(
        self,
        frame: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize segmentation results on frame"""
        vis_frame = frame.copy()
        
        # Generate colors for each mask
        colors = self._generate_colors(len(result.masks))
        
        for i, mask_data in enumerate(result.masks):
            mask = mask_data.mask
            color = colors[i]
            
            # Create colored mask
            colored_mask = np.zeros_like(vis_frame)
            colored_mask[mask] = color
            
            # Blend with original frame
            vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, colored_mask, alpha, 0)
            
            # Draw bounding box
            x, y, w, h = mask_data.bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            cv2.putText(
                vis_frame,
                f"{mask_data.confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return vis_frame
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    async def get_object_trajectories(
        self,
        segmentation_results: List[SegmentationResult],
        min_track_length: int = 5
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Track objects across frames to create trajectories
        
        Args:
            segmentation_results: List of segmentation results
            min_track_length: Minimum number of frames for a valid trajectory
            
        Returns:
            Dictionary mapping trajectory IDs to list of (x, y) positions
        """
        logger.info("Computing object trajectories...")
        
        trajectories = {}
        next_trajectory_id = 0
        
        for result in segmentation_results:
            for mask_data in result.masks:
                # Calculate centroid
                centroid = self._calculate_centroid(mask_data.mask)
                
                # Find closest existing trajectory
                closest_traj_id = None
                min_distance = float('inf')
                
                for traj_id, traj_points in trajectories.items():
                    if traj_points:
                        last_point = traj_points[-1]
                        distance = np.sqrt(
                            (centroid[0] - last_point[0])**2 + 
                            (centroid[1] - last_point[1])**2
                        )
                        
                        if distance < min_distance and distance < 50:  # Max distance threshold
                            min_distance = distance
                            closest_traj_id = traj_id
                
                # Add to trajectory
                if closest_traj_id is not None:
                    trajectories[closest_traj_id].append(centroid)
                else:
                    trajectories[next_trajectory_id] = [centroid]
                    next_trajectory_id += 1
        
        # Filter trajectories by minimum length
        filtered_trajectories = {
            traj_id: points for traj_id, points in trajectories.items()
            if len(points) >= min_track_length
        }
        
        logger.info(f"Found {len(filtered_trajectories)} object trajectories")
        return filtered_trajectories
    
    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Calculate centroid of a mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return (0, 0)

# Mock implementation for development without SAM
class MockSAMSegmenter:
    """Mock SAM segmenter for development/testing"""
    
    def __init__(self, *args, **kwargs):
        self.is_initialized = True
    
    async def initialize(self):
        """Mock initialization"""
        logger.info("Mock SAM segmenter initialized")
    
    async def segment_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        auto_segment: bool = True,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None
    ) -> SegmentationResult:
        """Generate mock segmentation result"""
        logger.info(f"Generating mock segmentation for frame {frame_id}")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock masks
        height, width = frame.shape[:2]
        num_objects = np.random.randint(2, 8)
        
        masks = []
        for i in range(num_objects):
            # Generate random mask
            mask = np.zeros((height, width), dtype=bool)
            
            # Random rectangle
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(50, width // 2)
            h = np.random.randint(50, height // 2)
            
            mask[y:y+h, x:x+w] = True
            
            # Calculate bbox and area
            bbox = (x, y, w, h)
            area = np.sum(mask)
            confidence = np.random.uniform(0.6, 0.95)
            
            segmentation_mask = SegmentationMask(
                mask=mask,
                confidence=confidence,
                bbox=bbox,
                area=area
            )
            masks.append(segmentation_mask)
        
        result = SegmentationResult(
            masks=masks,
            frame_id=frame_id,
            timestamp=timestamp,
            processing_time=0.1
        )
        
        logger.info(f"Mock segmentation completed with {len(masks)} objects")
        return result

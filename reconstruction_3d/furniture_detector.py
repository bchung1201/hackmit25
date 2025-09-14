"""
Furniture Detection Module
Detects furniture in video frames using YOLO
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

@dataclass
class FurnitureDetection:
    """Furniture detection result"""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    frame_id: int = 0  # Add frame_id attribute

@dataclass
class FurnitureDetections:
    """Collection of furniture detections for a frame"""
    frame_id: int
    timestamp: float
    detections: List[FurnitureDetection]
    frame_shape: Tuple[int, int, int]

class FurnitureDetector:
    """Furniture detection using YOLO"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLO model
        self._initialize_yolo()
        
        # Furniture class mapping
        self.furniture_classes = self._get_furniture_classes()
        
        logger.info(f"Furniture detector initialized on {self.device}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model_path': 'yolov8n.pt',  # YOLOv8 nano for speed
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'max_detections': 100,
            'input_size': 640,
            'enable_tracking': True
        }
    
    def _initialize_yolo(self):
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            self.model = YOLO(self.config['model_path'])
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"YOLO model loaded: {self.config['model_path']}")
            
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
    
    def _get_furniture_classes(self) -> Dict[int, str]:
        """Get furniture class mapping"""
        # COCO classes that are furniture
        furniture_classes = {
            0: 'person',  # Keep person for context
            15: 'cat',    # Keep pets for context
            16: 'dog',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        
        # Filter to only furniture-related classes
        furniture_only = {
            k: v for k, v in furniture_classes.items() 
            if v in ['chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator']
        }
        
        return furniture_only
    
    def detect_furniture(self, frame: np.ndarray, 
                        frame_id: int = 0, 
                        timestamp: float = 0.0) -> FurnitureDetections:
        """
        Detect furniture in a single frame
        
        Args:
            frame: Input video frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            FurnitureDetections with detected furniture
        """
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.config['confidence_threshold'],
                iou=self.config['nms_threshold'],
                max_det=self.config['max_detections'],
                verbose=False
            )
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Get first (and only) result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Check if it's a furniture class
                        if class_id in self.furniture_classes:
                            x1, y1, x2, y2 = box.astype(int)
                            
                            detection = FurnitureDetection(
                                class_name=self.furniture_classes[class_id],
                                class_id=class_id,
                                confidence=float(conf),
                                bbox=(x1, y1, x2, y2),
                                center=((x1 + x2) // 2, (y1 + y2) // 2),
                                area=(x2 - x1) * (y2 - y1),
                                frame_id=frame_id
                            )
                            detections.append(detection)
            
            return FurnitureDetections(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=detections,
                frame_shape=frame.shape
            )
            
        except Exception as e:
            logger.error(f"Furniture detection failed for frame {frame_id}: {e}")
            return FurnitureDetections(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=[],
                frame_shape=frame.shape
            )
    
    def detect_furniture_batch(self, frames: List[np.ndarray], 
                              frame_ids: Optional[List[int]] = None,
                              timestamps: Optional[List[float]] = None) -> List[FurnitureDetections]:
        """
        Detect furniture in multiple frames
        
        Args:
            frames: List of video frames
            frame_ids: List of frame identifiers
            timestamps: List of timestamps
            
        Returns:
            List of FurnitureDetections
        """
        if frame_ids is None:
            frame_ids = list(range(len(frames)))
        if timestamps is None:
            timestamps = [i * 0.1 for i in range(len(frames))]  # Assume 10 FPS
        
        detections = []
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            detection = self.detect_furniture(frame, frame_id, timestamp)
            detections.append(detection)
        
        return detections
    
    def track_furniture(self, detections: List[FurnitureDetections]) -> Dict[int, List[FurnitureDetection]]:
        """
        Track furniture across frames using simple IoU tracking
        
        Args:
            detections: List of furniture detections per frame
            
        Returns:
            Dictionary mapping track_id to list of detections
        """
        if not self.config['enable_tracking']:
            return {}
        
        tracks = {}
        next_track_id = 0
        
        for frame_detections in detections:
            current_tracks = {}
            
            for detection in frame_detections.detections:
                best_track_id = None
                best_iou = 0.0
                
                # Find best matching track
                for track_id, track_detections in tracks.items():
                    if track_detections:
                        last_detection = track_detections[-1]
                        iou = self._calculate_iou(detection.bbox, last_detection.bbox)
                        
                        if iou > 0.3 and iou > best_iou:  # IoU threshold
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
        
        return tracks
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
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
    
    def get_furniture_statistics(self, detections: List[FurnitureDetections]) -> Dict[str, Any]:
        """Get statistics about detected furniture"""
        total_detections = sum(len(d.detections) for d in detections)
        
        # Count by class
        class_counts = {}
        for frame_detections in detections:
            for detection in frame_detections.detections:
                class_name = detection.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Average confidence
        all_confidences = []
        for frame_detections in detections:
            for detection in frame_detections.detections:
                all_confidences.append(detection.confidence)
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'total_detections': total_detections,
            'class_counts': class_counts,
            'average_confidence': avg_confidence,
            'frames_processed': len(detections),
            'furniture_classes_detected': len(class_counts)
        }
    
    def save_detections(self, detections: List[FurnitureDetections], 
                       output_path: str) -> str:
        """Save furniture detections to JSON file"""
        import json
        
        output_data = []
        for frame_detections in detections:
            frame_data = {
                'frame_id': frame_detections.frame_id,
                'timestamp': frame_detections.timestamp,
                'frame_shape': frame_detections.frame_shape,
                'detections': []
            }
            
            for detection in frame_detections.detections:
                detection_data = {
                    'class_name': detection.class_name,
                    'class_id': detection.class_id,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center': detection.center,
                    'area': detection.area
                }
                frame_data['detections'].append(detection_data)
            
            output_data.append(frame_data)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Furniture detections saved to: {output_file}")
        return str(output_file)

"""
Room-Aware Emotion Processing
Tracks emotions per room and generates room-level summaries
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import asyncio

from .multi_model_detector import MultiModelEmotionDetector, EmotionPrediction
from .face_detector import FaceDetector

logger = logging.getLogger(__name__)

@dataclass
class RoomEmotionData:
    """Emotion data for a specific room"""
    room_name: str
    emotions: List[EmotionPrediction]
    timestamps: List[float]
    total_time: float
    dominant_emotion: str
    emotion_distribution: Dict[str, float]
    average_confidence: float
    average_intensity: float
    emotion_trend: str  # improving, declining, stable

@dataclass
class RoomEmotionSummary:
    """Summary of emotions across all rooms"""
    rooms: Dict[str, RoomEmotionData]
    overall_mood: str
    most_emotional_room: str
    least_emotional_room: str
    total_processing_time: float

class RoomAwareEmotionProcessor:
    """Processes emotions with room awareness"""
    
    def __init__(self, room_positions: Optional[Dict[str, Tuple[float, float]]] = None):
        self.face_detector = FaceDetector()
        self.emotion_detector = MultiModelEmotionDetector()
        
        # Room tracking
        self.room_positions = room_positions or self._get_default_room_positions()
        self.current_room = None
        self.room_history = []  # List of (timestamp, room_name, position)
        
        # Emotion tracking per room
        self.room_emotions = defaultdict(list)  # room_name -> List[EmotionPrediction]
        self.room_timestamps = defaultdict(list)  # room_name -> List[timestamp]
        
        # Processing statistics
        self.total_frames_processed = 0
        self.total_emotions_detected = 0
        self.processing_start_time = None
        
        logger.info("Room-aware emotion processor initialized")
    
    def _get_default_room_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get default room positions for testing"""
        return {
            "Bar Room": (175, 225),
            "Laundry Room": (400, 125),
            "Hallway": (400, 225),
            "Game Room": (650, 125),
            "Living Room": (650, 375),
            "Bedroom": (200, 300),
            "Kitchen": (300, 350),
            "Bathroom": (100, 200)
        }
    
    def set_room_positions(self, room_positions: Dict[str, Tuple[float, float]]):
        """Set room positions for tracking"""
        self.room_positions = room_positions
        logger.info(f"Updated room positions: {list(room_positions.keys())}")
    
    def determine_current_room(self, position: Tuple[float, float]) -> str:
        """
        Determine current room based on position
        
        Args:
            position: (x, y) position coordinates
            
        Returns:
            Room name or 'Unknown'
        """
        if not self.room_positions:
            return "Unknown"
        
        x, y = position
        min_distance = float('inf')
        closest_room = "Unknown"
        
        for room_name, (room_x, room_y) in self.room_positions.items():
            distance = np.sqrt((x - room_x)**2 + (y - room_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_room = room_name
        
        # Only consider it a valid room if within reasonable distance
        if min_distance < 200:  # Adjust threshold as needed
            return closest_room
        else:
            return "Unknown"
    
    async def process_frame_with_room(self, frame: np.ndarray, position: Tuple[float, float], 
                                    frame_id: int = 0, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Process frame with room awareness
        
        Args:
            frame: Input video frame
            position: Current position (x, y)
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            Processing result with room and emotion information
        """
        if self.processing_start_time is None:
            self.processing_start_time = timestamp
        
        try:
            # Determine current room
            current_room = self.determine_current_room(position)
            
            # Update room history
            self.room_history.append((timestamp, current_room, position))
            self.current_room = current_room
            
            # Detect faces
            face_boxes = self.face_detector.detect_faces(frame)
            
            emotions_detected = []
            
            # Process each detected face
            for i, face_box in enumerate(face_boxes):
                try:
                    # Extract face region
                    face_image = self.face_detector.extract_face_region(frame, face_box)
                    
                    # Detect emotions with multiple models
                    predictions = self.emotion_detector.detect_emotions(face_image)
                    
                    if predictions:
                        # Get best prediction
                        best_prediction = self.emotion_detector.get_best_prediction(predictions)
                        
                        if best_prediction:
                            emotions_detected.append(best_prediction)
                            
                            # Store emotion for current room
                            self.room_emotions[current_room].append(best_prediction)
                            self.room_timestamps[current_room].append(timestamp)
                            
                            self.total_emotions_detected += 1
                            
                            logger.debug(f"Detected {best_prediction.emotion} "
                                       f"(conf: {best_prediction.confidence:.2f}) "
                                       f"in {current_room} using {best_prediction.model_name}")
                
                except Exception as e:
                    logger.error(f"Error processing face {i}: {e}")
                    continue
            
            # Update statistics
            self.total_frames_processed += 1
            
            # Create result
            result = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "current_room": current_room,
                "position": position,
                "faces_detected": len(face_boxes),
                "emotions_detected": emotions_detected,
                "best_emotion": emotions_detected[0] if emotions_detected else None,
                "room_emotion_count": len(self.room_emotions[current_room])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "current_room": "Unknown",
                "position": position,
                "faces_detected": 0,
                "emotions_detected": [],
                "best_emotion": None,
                "room_emotion_count": 0,
                "error": str(e)
            }
    
    def get_room_emotion_summary(self) -> RoomEmotionSummary:
        """
        Get summary of emotions across all rooms
        
        Returns:
            RoomEmotionSummary with aggregated data
        """
        room_data = {}
        
        for room_name, emotions in self.room_emotions.items():
            if not emotions:
                continue
            
            timestamps = self.room_timestamps[room_name]
            
            # Calculate statistics
            total_time = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            confidences = [e.confidence for e in emotions]
            average_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate emotion distribution
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion.emotion] = emotion_counts.get(emotion.emotion, 0) + 1
            
            total_emotions = len(emotions)
            emotion_distribution = {
                emotion: count / total_emotions 
                for emotion, count in emotion_counts.items()
            }
            
            # Find dominant emotion
            dominant_emotion = max(emotion_distribution.items(), key=lambda x: x[1])[0]
            
            # Calculate average intensity (simplified)
            average_intensity = average_confidence  # Use confidence as intensity proxy
            
            # Calculate emotion trend
            emotion_trend = self._calculate_emotion_trend(emotions, timestamps)
            
            room_data[room_name] = RoomEmotionData(
                room_name=room_name,
                emotions=emotions,
                timestamps=timestamps,
                total_time=total_time,
                dominant_emotion=dominant_emotion,
                emotion_distribution=emotion_distribution,
                average_confidence=average_confidence,
                average_intensity=average_intensity,
                emotion_trend=emotion_trend
            )
        
        # Calculate overall statistics
        overall_mood = self._calculate_overall_mood(room_data)
        most_emotional_room = self._find_most_emotional_room(room_data)
        least_emotional_room = self._find_least_emotional_room(room_data)
        
        total_processing_time = 0.0
        if self.processing_start_time is not None:
            total_processing_time = max([max(ts) for ts in self.room_timestamps.values()] + [0]) - self.processing_start_time
        
        return RoomEmotionSummary(
            rooms=room_data,
            overall_mood=overall_mood,
            most_emotional_room=most_emotional_room,
            least_emotional_room=least_emotional_room,
            total_processing_time=total_processing_time
        )
    
    def _calculate_emotion_trend(self, emotions: List[EmotionPrediction], 
                                timestamps: List[float]) -> str:
        """Calculate emotion trend over time"""
        if len(emotions) < 3:
            return "insufficient_data"
        
        # Split into early and late periods
        mid_point = len(emotions) // 2
        early_emotions = emotions[:mid_point]
        late_emotions = emotions[mid_point:]
        
        # Calculate average confidence for each period
        early_avg = np.mean([e.confidence for e in early_emotions])
        late_avg = np.mean([e.confidence for e in late_emotions])
        
        if late_avg > early_avg * 1.1:
            return "improving"
        elif late_avg < early_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _calculate_overall_mood(self, room_data: Dict[str, RoomEmotionData]) -> str:
        """Calculate overall mood across all rooms"""
        if not room_data:
            return "unknown"
        
        # Weight by room size and time spent
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for room_data_item in room_data.values():
            weight = room_data_item.total_time * room_data_item.average_confidence
            total_weighted_confidence += weight
            total_weight += weight
        
        if total_weight == 0:
            return "unknown"
        
        overall_confidence = total_weighted_confidence / total_weight
        
        if overall_confidence > 0.7:
            return "positive"
        elif overall_confidence < 0.3:
            return "negative"
        else:
            return "neutral"
    
    def _find_most_emotional_room(self, room_data: Dict[str, RoomEmotionData]) -> str:
        """Find room with highest emotional intensity"""
        if not room_data:
            return "none"
        
        return max(room_data.items(), key=lambda x: x[1].average_intensity)[0]
    
    def _find_least_emotional_room(self, room_data: Dict[str, RoomEmotionData]) -> str:
        """Find room with lowest emotional intensity"""
        if not room_data:
            return "none"
        
        return min(room_data.items(), key=lambda x: x[1].average_intensity)[0]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_emotions_detected": self.total_emotions_detected,
            "rooms_with_emotions": len([r for r in self.room_emotions.values() if r]),
            "current_room": self.current_room,
            "room_history_length": len(self.room_history),
            "model_statistics": self.emotion_detector.get_model_statistics()
        }
    
    def reset_statistics(self):
        """Reset all statistics and data"""
        self.room_emotions.clear()
        self.room_timestamps.clear()
        self.room_history.clear()
        self.total_frames_processed = 0
        self.total_emotions_detected = 0
        self.processing_start_time = None
        self.current_room = None
        logger.info("Room-aware processor statistics reset")

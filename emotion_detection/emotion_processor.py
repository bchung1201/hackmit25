"""
Main emotion processing pipeline that combines face detection and emotion recognition
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from dataclasses import dataclass

from .face_detector import FaceDetector
from .emonet_detector import EmoNetDetector

logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Result of emotion detection for a single face"""
    face_id: int
    face_box: Tuple[int, int, int, int]
    emotion: str
    emotion_id: int
    confidence: float
    valence: float
    arousal: float
    intensity: float
    category: str
    emotion_probs: Dict[str, float]

@dataclass
class FrameEmotionResult:
    """Result of emotion detection for an entire frame"""
    frame_id: int
    timestamp: float
    faces_detected: int
    emotions: List[EmotionResult]
    dominant_emotion: Optional[EmotionResult]
    overall_mood: str

class EmotionProcessor:
    """Main emotion processing pipeline"""
    
    def __init__(self, 
                 face_detection_method: str = "opencv",
                 emonet_model_path: str = "pretrained/emonet_8.pth",
                 n_emotion_classes: int = 8):
        
        self.face_detector = FaceDetector(method=face_detection_method)
        self.emonet_detector = EmoNetDetector(
            model_path=emonet_model_path,
            n_classes=n_emotion_classes
        )
        
        # Processing statistics
        self.total_frames_processed = 0
        self.total_faces_detected = 0
        self.emotion_history = []
        
        logger.info("Emotion processor initialized")
    
    async def process_frame(self, frame: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> FrameEmotionResult:
        """
        Process a single frame for emotion detection
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            FrameEmotionResult with all detected emotions
        """
        try:
            # Detect faces in frame
            face_boxes = self.face_detector.detect_faces(frame)
            
            emotions = []
            dominant_emotion = None
            max_confidence = 0.0
            
            # Process each detected face
            for i, face_box in enumerate(face_boxes):
                try:
                    # Extract face region
                    face_image = self.face_detector.extract_face_region(frame, face_box)
                    
                    # Detect emotions
                    emotion_result = self.emonet_detector.detect_emotions(face_image)
                    
                    # Calculate additional metrics
                    intensity = self.emonet_detector.get_emotion_intensity(emotion_result)
                    category = self.emonet_detector.get_emotion_category(emotion_result)
                    
                    # Create emotion result
                    emotion = EmotionResult(
                        face_id=i,
                        face_box=face_box,
                        emotion=emotion_result['emotion'],
                        emotion_id=emotion_result['emotion_id'],
                        confidence=emotion_result['confidence'],
                        valence=emotion_result['valence'],
                        arousal=emotion_result['arousal'],
                        intensity=intensity,
                        category=category,
                        emotion_probs=emotion_result['emotion_probs']
                    )
                    
                    emotions.append(emotion)
                    
                    # Track dominant emotion (highest confidence)
                    if emotion.confidence > max_confidence:
                        max_confidence = emotion.confidence
                        dominant_emotion = emotion
                    
                except Exception as e:
                    logger.error(f"Error processing face {i}: {e}")
                    continue
            
            # Determine overall mood
            overall_mood = self._determine_overall_mood(emotions)
            
            # Create frame result
            frame_result = FrameEmotionResult(
                frame_id=frame_id,
                timestamp=timestamp,
                faces_detected=len(face_boxes),
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                overall_mood=overall_mood
            )
            
            # Update statistics
            self.total_frames_processed += 1
            self.total_faces_detected += len(face_boxes)
            self.emotion_history.append(frame_result)
            
            # Keep only recent history (last 100 frames)
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-100:]
            
            return frame_result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return FrameEmotionResult(
                frame_id=frame_id,
                timestamp=timestamp,
                faces_detected=0,
                emotions=[],
                dominant_emotion=None,
                overall_mood="unknown"
            )
    
    def _determine_overall_mood(self, emotions: List[EmotionResult]) -> str:
        """
        Determine overall mood from multiple emotions
        
        Args:
            emotions: List of detected emotions
            
        Returns:
            Overall mood category
        """
        if not emotions:
            return "neutral"
        
        # Count emotion categories
        category_counts = {}
        total_confidence = 0.0
        weighted_categories = {}
        
        for emotion in emotions:
            category = emotion.category
            confidence = emotion.confidence
            
            if category not in category_counts:
                category_counts[category] = 0
                weighted_categories[category] = 0.0
            
            category_counts[category] += 1
            weighted_categories[category] += confidence
            total_confidence += confidence
        
        if total_confidence == 0:
            return "neutral"
        
        # Find most confident category
        best_category = max(weighted_categories.items(), key=lambda x: x[1])[0]
        
        # Special cases for mixed emotions
        if len(category_counts) > 1:
            if "positive" in category_counts and "negative" in category_counts:
                return "mixed"
            elif "high_arousal" in category_counts and category_counts["high_arousal"] > 1:
                return "excited"
        
        return best_category
    
    def get_emotion_trends(self, window_size: int = 10) -> Dict[str, float]:
        """
        Analyze emotion trends over recent frames
        
        Args:
            window_size: Number of recent frames to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.emotion_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_frames = self.emotion_history[-window_size:]
        
        # Calculate average valence and arousal
        total_valence = 0.0
        total_arousal = 0.0
        total_intensity = 0.0
        frame_count = 0
        
        for frame_result in recent_frames:
            if frame_result.emotions:
                for emotion in frame_result.emotions:
                    total_valence += emotion.valence
                    total_arousal += emotion.arousal
                    total_intensity += emotion.intensity
                    frame_count += 1
        
        if frame_count == 0:
            return {"trend": "no_faces_detected"}
        
        avg_valence = total_valence / frame_count
        avg_arousal = total_arousal / frame_count
        avg_intensity = total_intensity / frame_count
        
        # Determine trend direction
        if len(recent_frames) >= 2:
            early_valence = 0.0
            late_valence = 0.0
            early_count = 0
            late_count = 0
            
            mid_point = len(recent_frames) // 2
            
            for i, frame_result in enumerate(recent_frames):
                if frame_result.emotions:
                    for emotion in frame_result.emotions:
                        if i < mid_point:
                            early_valence += emotion.valence
                            early_count += 1
                        else:
                            late_valence += emotion.valence
                            late_count += 1
            
            if early_count > 0 and late_count > 0:
                early_avg = early_valence / early_count
                late_avg = late_valence / late_count
                valence_trend = "improving" if late_avg > early_avg else "declining"
            else:
                valence_trend = "stable"
        else:
            valence_trend = "stable"
        
        return {
            "trend": valence_trend,
            "avg_valence": avg_valence,
            "avg_arousal": avg_arousal,
            "avg_intensity": avg_intensity,
            "frames_analyzed": len(recent_frames),
            "faces_analyzed": frame_count
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get processing statistics"""
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_faces_detected": self.total_faces_detected,
            "avg_faces_per_frame": self.total_faces_detected / max(1, self.total_frames_processed),
            "recent_emotion_history": len(self.emotion_history)
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.total_frames_processed = 0
        self.total_faces_detected = 0
        self.emotion_history = []
        logger.info("Emotion processor statistics reset")

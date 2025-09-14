"""
Emotion to room mapping system
Maps detected emotions to room highlighting states
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio

from emotion_detection.emotion_processor import EmotionProcessor, FrameEmotionResult
from room_highlighting.room_highlighter import RoomHighlighter, HighlightingRule
from room_highlighting.roborock_parser import RoborockMapParser

logger = logging.getLogger(__name__)

@dataclass
class EmotionRoomMapping:
    """Mapping between emotion and room highlighting"""
    emotion: str
    room_names: List[str]
    color: Tuple[int, int, int]
    intensity: float
    animation_type: str
    priority: int
    conditions: Optional[Dict[str, Any]] = None

class EmotionRoomMapper:
    """Main emotion to room mapping system"""
    
    def __init__(self, map_data: Optional[Dict[str, Any]] = None):
        self.emotion_processor = EmotionProcessor()
        self.room_highlighter = RoomHighlighter()
        self.map_parser = RoborockMapParser()
        
        # Load map if provided
        if map_data:
            self.room_highlighter.load_map_from_json(map_data)
        
        # Emotion to room mappings
        self.emotion_mappings: List[EmotionRoomMapping] = []
        
        # Current state
        self.current_emotion: Optional[str] = None
        self.current_intensity: float = 0.0
        self.current_room: Optional[str] = None
        self.emotion_history: List[FrameEmotionResult] = []
        
        # Setup default mappings
        self._setup_default_mappings()
        
        logger.info("Emotion-Room mapper initialized")
    
    def _setup_default_mappings(self):
        """Setup default emotion to room mappings"""
        self.emotion_mappings = [
            # Happy emotions - brighten social and entertainment areas
            EmotionRoomMapping(
                emotion="happy",
                room_names=["Game Room", "Bar Room", "Living Room", "Dining Room"],
                color=(0, 255, 0),  # Bright green
                intensity=0.8,
                animation_type="glow",
                priority=3,
                conditions={"min_confidence": 0.6}
            ),
            
            # Excited emotions - highlight active areas
            EmotionRoomMapping(
                emotion="excited",
                room_names=["Game Room", "Bar Room"],
                color=(255, 255, 0),  # Bright yellow
                intensity=1.0,
                animation_type="pulse",
                priority=4,
                conditions={"min_arousal": 0.7}
            ),
            
            # Sad emotions - highlight calming and private areas
            EmotionRoomMapping(
                emotion="sad",
                room_names=["Living Room", "Bedroom", "Bathroom"],
                color=(0, 100, 200),  # Deep blue
                intensity=0.6,
                animation_type="fade",
                priority=2,
                conditions={"max_valence": -0.3}
            ),
            
            # Angry emotions - highlight safe spaces, dim social areas
            EmotionRoomMapping(
                emotion="angry",
                room_names=["Living Room", "Bedroom"],
                color=(255, 0, 0),  # Red
                intensity=0.7,
                animation_type="static",
                priority=3,
                conditions={"min_arousal": 0.5}
            ),
            
            # Fear emotions - highlight current location and safe spaces
            EmotionRoomMapping(
                emotion="fear",
                room_names=["Living Room", "Bedroom", "Bathroom"],
                color=(128, 0, 128),  # Purple
                intensity=0.8,
                animation_type="pulse",
                priority=3,
                conditions={"min_arousal": 0.6}
            ),
            
            # Surprise emotions - highlight current location
            EmotionRoomMapping(
                emotion="surprise",
                room_names=["Game Room", "Living Room", "Hallway"],
                color=(255, 165, 0),  # Orange
                intensity=0.9,
                animation_type="glow",
                priority=2,
                conditions={"min_arousal": 0.5}
            ),
            
            # Disgust emotions - highlight clean areas
            EmotionRoomMapping(
                emotion="disgust",
                room_names=["Bathroom", "Kitchen", "Laundry Room"],
                color=(139, 69, 19),  # Brown
                intensity=0.5,
                animation_type="fade",
                priority=1,
                conditions={"max_valence": -0.2}
            ),
            
            # Contempt emotions - subtle highlighting
            EmotionRoomMapping(
                emotion="contempt",
                room_names=["Living Room", "Office"],
                color=(105, 105, 105),  # Dim gray
                intensity=0.4,
                animation_type="static",
                priority=1,
                conditions={"max_valence": -0.1}
            ),
            
            # High arousal - highlight active areas
            EmotionRoomMapping(
                emotion="high_arousal",
                room_names=["Game Room", "Bar Room", "Living Room", "Dining Room"],
                color=(255, 69, 0),  # Red orange
                intensity=0.8,
                animation_type="pulse",
                priority=3,
                conditions={"min_arousal": 0.7}
            ),
            
            # Positive mood - brighten social areas
            EmotionRoomMapping(
                emotion="positive",
                room_names=["Game Room", "Bar Room", "Living Room", "Dining Room", "Kitchen"],
                color=(0, 255, 127),  # Spring green
                intensity=0.7,
                animation_type="glow",
                priority=2,
                conditions={"min_valence": 0.3}
            ),
            
            # Negative mood - highlight private spaces
            EmotionRoomMapping(
                emotion="negative",
                room_names=["Bedroom", "Living Room", "Bathroom"],
                color=(220, 20, 60),  # Crimson
                intensity=0.6,
                animation_type="fade",
                priority=2,
                conditions={"max_valence": -0.2}
            ),
            
            # Mixed emotions - subtle highlighting
            EmotionRoomMapping(
                emotion="mixed",
                room_names=["Living Room", "Hallway"],
                color=(255, 192, 203),  # Pink
                intensity=0.4,
                animation_type="fade",
                priority=1,
                conditions={"min_confidence": 0.4}
            ),
            
            # Neutral - minimal highlighting
            EmotionRoomMapping(
                emotion="neutral",
                room_names=["Living Room", "Hallway"],
                color=(200, 200, 200),  # Light gray
                intensity=0.2,
                animation_type="static",
                priority=0,
                conditions={"max_intensity": 0.3}
            )
        ]
    
    async def process_frame(self, frame, frame_id: int = 0, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Process frame for emotion detection and room highlighting
        
        Args:
            frame: Input video frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            Processing result with emotion and highlighting information
        """
        try:
            # Detect emotions in frame
            emotion_result = await self.emotion_processor.process_frame(
                frame, frame_id, timestamp
            )
            
            # Update emotion history
            self.emotion_history.append(emotion_result)
            if len(self.emotion_history) > 50:  # Keep last 50 frames
                self.emotion_history = self.emotion_history[-50:]
            
            # Determine current emotion and intensity
            current_emotion, intensity = self._determine_current_emotion(emotion_result)
            
            # Update room highlighting
            highlighting_result = self._update_room_highlighting(
                current_emotion, intensity, emotion_result
            )
            
            # Update current state
            self.current_emotion = current_emotion
            self.current_intensity = intensity
            
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "emotion_result": emotion_result,
                "current_emotion": current_emotion,
                "intensity": intensity,
                "highlighting_result": highlighting_result,
                "rooms_highlighted": list(highlighting_result.get("highlighted_rooms", [])),
                "processing_stats": self.emotion_processor.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "error": str(e),
                "current_emotion": "unknown",
                "intensity": 0.0,
                "highlighting_result": {},
                "rooms_highlighted": []
            }
    
    def _determine_current_emotion(self, emotion_result: FrameEmotionResult) -> Tuple[str, float]:
        """
        Determine current emotion and intensity from frame result
        
        Args:
            emotion_result: Result from emotion detection
            
        Returns:
            Tuple of (emotion_name, intensity)
        """
        if not emotion_result.emotions:
            return "neutral", 0.0
        
        # Use dominant emotion if available
        if emotion_result.dominant_emotion:
            return (
                emotion_result.dominant_emotion.emotion,
                emotion_result.dominant_emotion.intensity
            )
        
        # Fallback to overall mood
        return emotion_result.overall_mood, 0.5
    
    def _update_room_highlighting(self, emotion: str, intensity: float, 
                                 emotion_result: FrameEmotionResult) -> Dict[str, Any]:
        """
        Update room highlighting based on emotion
        
        Args:
            emotion: Current emotion
            intensity: Emotion intensity
            emotion_result: Full emotion detection result
            
        Returns:
            Highlighting result information
        """
        try:
            # Find applicable mappings
            applicable_mappings = self._find_applicable_mappings(emotion, intensity, emotion_result)
            
            if not applicable_mappings:
                logger.info(f"No applicable mappings for emotion: {emotion}")
                return {"highlighted_rooms": [], "mappings_applied": 0}
            
            # Clear previous highlights
            self.room_highlighter.clear_highlights()
            
            # Apply mappings
            mappings_applied = 0
            for mapping in applicable_mappings:
                if self._apply_mapping(mapping, intensity, emotion_result):
                    mappings_applied += 1
            
            # Get highlighting status
            status = self.room_highlighter.get_highlighting_status()
            
            return {
                "highlighted_rooms": status["highlighted_rooms"],
                "mappings_applied": mappings_applied,
                "total_rooms": status["total_rooms"],
                "animation_time": status["animation_time"]
            }
            
        except Exception as e:
            logger.error(f"Error updating room highlighting: {e}")
            return {"highlighted_rooms": [], "mappings_applied": 0, "error": str(e)}
    
    def _find_applicable_mappings(self, emotion: str, intensity: float, 
                                 emotion_result: FrameEmotionResult) -> List[EmotionRoomMapping]:
        """Find mappings applicable to current emotion and context"""
        applicable = []
        
        for mapping in self.emotion_mappings:
            if self._is_mapping_applicable(mapping, emotion, intensity, emotion_result):
                applicable.append(mapping)
        
        # Sort by priority (highest first)
        applicable.sort(key=lambda x: x.priority, reverse=True)
        
        return applicable
    
    def _is_mapping_applicable(self, mapping: EmotionRoomMapping, emotion: str, 
                              intensity: float, emotion_result: FrameEmotionResult) -> bool:
        """Check if a mapping is applicable to current context"""
        # Check emotion match
        if mapping.emotion.lower() != emotion.lower():
            return False
        
        # Check conditions if specified
        if mapping.conditions:
            if not self._check_conditions(mapping.conditions, emotion_result):
                return False
        
        return True
    
    def _check_conditions(self, conditions: Dict[str, Any], emotion_result: FrameEmotionResult) -> bool:
        """Check if emotion result meets mapping conditions"""
        if not emotion_result.dominant_emotion:
            return False
        
        emotion = emotion_result.dominant_emotion
        
        # Check confidence threshold
        if "min_confidence" in conditions:
            if emotion.confidence < conditions["min_confidence"]:
                return False
        
        # Check valence threshold
        if "min_valence" in conditions:
            if emotion.valence < conditions["min_valence"]:
                return False
        if "max_valence" in conditions:
            if emotion.valence > conditions["max_valence"]:
                return False
        
        # Check arousal threshold
        if "min_arousal" in conditions:
            if emotion.arousal < conditions["min_arousal"]:
                return False
        if "max_arousal" in conditions:
            if emotion.arousal > conditions["max_arousal"]:
                return False
        
        # Check intensity threshold
        if "min_intensity" in conditions:
            if emotion.intensity < conditions["min_intensity"]:
                return False
        if "max_intensity" in conditions:
            if emotion.intensity > conditions["max_intensity"]:
                return False
        
        return True
    
    def _apply_mapping(self, mapping: EmotionRoomMapping, intensity: float, 
                      emotion_result: FrameEmotionResult) -> bool:
        """Apply a specific mapping to room highlighting"""
        try:
            # Create highlighting rule
            rule = HighlightingRule(
                emotion=mapping.emotion,
                room_names=mapping.room_names,
                color=mapping.color,
                intensity=mapping.intensity * intensity,
                animation_type=mapping.animation_type,
                priority=mapping.priority
            )
            
            # Apply to room highlighter
            self.room_highlighter.add_highlighting_rule(rule)
            self.room_highlighter.update_emotion(
                mapping.emotion, 
                mapping.intensity * intensity
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying mapping {mapping.emotion}: {e}")
            return False
    
    def get_highlighted_map_image(self, width: int = 1000, height: int = 600) -> Optional[bytes]:
        """
        Get highlighted map image as bytes
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Image bytes or None if no map available
        """
        try:
            import cv2
            
            # Generate highlighted map image
            map_image = self.room_highlighter.generate_highlighted_map_image(width, height)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', map_image)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error generating map image: {e}")
            return None
    
    def add_custom_mapping(self, mapping: EmotionRoomMapping):
        """Add a custom emotion to room mapping"""
        self.emotion_mappings.append(mapping)
        logger.info(f"Added custom mapping for emotion: {mapping.emotion}")
    
    def get_emotion_trends(self) -> Dict[str, Any]:
        """Get emotion trends over recent frames"""
        return self.emotion_processor.get_emotion_trends()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "current_emotion": self.current_emotion,
            "current_intensity": self.current_intensity,
            "current_room": self.current_room,
            "emotion_history_length": len(self.emotion_history),
            "mappings_count": len(self.emotion_mappings),
            "highlighting_status": self.room_highlighter.get_highlighting_status(),
            "emotion_stats": self.emotion_processor.get_statistics()
        }
    
    def reset_state(self):
        """Reset system state"""
        self.current_emotion = None
        self.current_intensity = 0.0
        self.current_room = None
        self.emotion_history = []
        self.room_highlighter.clear_highlights()
        self.emotion_processor.reset_statistics()
        logger.info("System state reset")

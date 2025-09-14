"""
Emotion Summary Map Generator
Creates a single map showing overall emotions per room
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
from pathlib import Path

from .roborock_parser import RoborockMap, Room
from .room_highlighter import RoomHighlighter
from emotion_detection.room_aware_processor import RoomEmotionSummary, RoomEmotionData

logger = logging.getLogger(__name__)

@dataclass
class EmotionMapConfig:
    """Configuration for emotion summary map"""
    map_width: int = 1000
    map_height: int = 600
    show_statistics: bool = True
    show_emotion_distribution: bool = True
    show_confidence_scores: bool = True
    show_trends: bool = True
    font_scale: float = 0.6
    line_thickness: int = 2

class EmotionSummaryMapGenerator:
    """Generates single summary map with room emotions"""
    
    def __init__(self, map_data: Optional[RoborockMap] = None):
        self.room_highlighter = RoomHighlighter(map_data)
        self.config = EmotionMapConfig()
        
        # Emotion color mapping
        self.emotion_colors = {
            "Happy": (0, 255, 0),      # Green
            "Sad": (0, 100, 200),      # Blue
            "Angry": (255, 0, 0),      # Red
            "Fear": (128, 0, 128),     # Purple
            "Surprise": (255, 165, 0), # Orange
            "Disgust": (139, 69, 19),  # Brown
            "Contempt": (105, 105, 105), # Dim Gray
            "Neutral": (200, 200, 200), # Light Gray
            "Excited": (255, 255, 0),  # Yellow
            "positive": (0, 255, 127), # Spring Green
            "negative": (220, 20, 60), # Crimson
            "unknown": (128, 128, 128) # Gray
        }
        
        logger.info("Emotion summary map generator initialized")
    
    def generate_summary_map(self, emotion_summary: RoomEmotionSummary, 
                           config: Optional[EmotionMapConfig] = None) -> np.ndarray:
        """
        Generate single summary map with room emotions
        
        Args:
            emotion_summary: Room emotion summary data
            config: Optional configuration
            
        Returns:
            Summary map image (BGR format)
        """
        if config:
            self.config = config
        
        # Create base map
        map_image = np.ones((self.config.map_height, self.config.map_width, 3), dtype=np.uint8) * 240
        
        # Draw rooms with emotion-based colors
        self._draw_emotion_rooms(map_image, emotion_summary)
        
        # Add statistics overlay
        if self.config.show_statistics:
            self._add_statistics_overlay(map_image, emotion_summary)
        
        # Add emotion distribution
        if self.config.show_emotion_distribution:
            self._add_emotion_distribution(map_image, emotion_summary)
        
        # Add confidence scores
        if self.config.show_confidence_scores:
            self._add_confidence_scores(map_image, emotion_summary)
        
        # Add trends
        if self.config.show_trends:
            self._add_trend_indicators(map_image, emotion_summary)
        
        return map_image
    
    def _draw_emotion_rooms(self, map_image: np.ndarray, emotion_summary: RoomEmotionSummary):
        """Draw rooms colored by dominant emotion"""
        if not self.room_highlighter.map_data:
            logger.warning("No map data available")
            return
        
        scale_x = self.config.map_width / self.room_highlighter.map_data.width
        scale_y = self.config.map_height / self.room_highlighter.map_data.height
        
        for room in self.room_highlighter.map_data.rooms:
            # Get emotion data for this room
            room_data = emotion_summary.rooms.get(room.name)
            
            if room_data:
                # Get emotion color
                emotion_color = self.emotion_colors.get(
                    room_data.dominant_emotion, 
                    self.emotion_colors["unknown"]
                )
                
                # Adjust intensity based on confidence
                intensity = room_data.average_confidence
                final_color = self._adjust_color_intensity(emotion_color, intensity)
                
                # Draw room
                self._draw_room_with_emotion(map_image, room, final_color, scale_x, scale_y, room_data)
            else:
                # No emotion data - draw in neutral color
                self._draw_room_with_emotion(map_image, room, self.emotion_colors["unknown"], scale_x, scale_y)
    
    def _draw_room_with_emotion(self, map_image: np.ndarray, room: Room, color: Tuple[int, int, int], 
                               scale_x: float, scale_y: float, room_data: Optional[RoomEmotionData] = None):
        """Draw a single room with emotion color and information"""
        if not room.coordinates:
            return
        
        # Scale coordinates
        scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in room.coordinates]
        pts = np.array(scaled_coords, np.int32)
        
        # Draw room polygon
        cv2.fillPoly(map_image, [pts], color)
        
        # Draw room border
        cv2.polylines(map_image, [pts], True, (0, 0, 0), self.config.line_thickness)
        
        # Draw room name
        center_x = int(room.center[0] * scale_x)
        center_y = int(room.center[1] * scale_y)
        
        # Add background for text
        text_size = cv2.getTextSize(room.name, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 2)[0]
        cv2.rectangle(map_image, 
                     (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                     (center_x + text_size[0]//2 + 5, center_y + 5),
                     (255, 255, 255), -1)
        
        cv2.putText(map_image, room.name, (center_x - text_size[0]//2, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (0, 0, 0), 2)
        
        # Add emotion information if available
        if room_data:
            emotion_text = f"{room_data.dominant_emotion} ({room_data.average_confidence:.2f})"
            emotion_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(map_image, emotion_text, 
                       (center_x - emotion_size[0]//2, center_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def _adjust_color_intensity(self, color: Tuple[int, int, int], intensity: float) -> Tuple[int, int, int]:
        """Adjust color intensity based on confidence"""
        if intensity <= 0:
            return (128, 128, 128)  # Gray for no confidence
        
        # Blend with white for lower intensity
        white = (255, 255, 255)
        adjusted_color = tuple(
            int(color[i] * intensity + white[i] * (1 - intensity))
            for i in range(3)
        )
        return adjusted_color
    
    def _add_statistics_overlay(self, map_image: np.ndarray, emotion_summary: RoomEmotionSummary):
        """Add statistics overlay to the map"""
        # Create statistics panel
        panel_height = 120
        panel_width = 300
        panel_x = self.config.map_width - panel_width - 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(map_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), -1)
        cv2.rectangle(map_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), 2)
        
        # Add statistics text
        y_offset = panel_y + 20
        line_height = 20
        
        stats_text = [
            f"Overall Mood: {emotion_summary.overall_mood}",
            f"Most Emotional: {emotion_summary.most_emotional_room}",
            f"Least Emotional: {emotion_summary.least_emotional_room}",
            f"Processing Time: {emotion_summary.total_processing_time:.1f}s",
            f"Rooms with Data: {len(emotion_summary.rooms)}"
        ]
        
        for text in stats_text:
            cv2.putText(map_image, text, (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += line_height
    
    def _add_emotion_distribution(self, map_image: np.ndarray, emotion_summary: RoomEmotionSummary):
        """Add emotion distribution chart"""
        # Count emotions across all rooms
        emotion_counts = {}
        for room_data in emotion_summary.rooms.values():
            for emotion in room_data.emotions:
                emotion_counts[emotion.emotion] = emotion_counts.get(emotion.emotion, 0) + 1
        
        if not emotion_counts:
            return
        
        # Create distribution panel
        panel_height = 150
        panel_width = 200
        panel_x = 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(map_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), -1)
        cv2.rectangle(map_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), 2)
        
        # Add title
        cv2.putText(map_image, "Emotion Distribution", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add emotion counts
        y_offset = panel_y + 45
        total_emotions = sum(emotion_counts.values())
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_emotions) * 100
            text = f"{emotion}: {count} ({percentage:.1f}%)"
            cv2.putText(map_image, text, (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 18
    
    def _add_confidence_scores(self, map_image: np.ndarray, emotion_summary: RoomEmotionSummary):
        """Add confidence scores for each room"""
        scale_x = self.config.map_width / self.room_highlighter.map_data.width
        scale_y = self.config.map_height / self.room_highlighter.map_data.height
        
        for room in self.room_highlighter.map_data.rooms:
            room_data = emotion_summary.rooms.get(room.name)
            if room_data:
                # Draw confidence bar
                center_x = int(room.center[0] * scale_x)
                center_y = int(room.center[1] * scale_y)
                
                # Confidence bar
                bar_width = 60
                bar_height = 8
                bar_x = center_x - bar_width // 2
                bar_y = center_y + 35
                
                # Background bar
                cv2.rectangle(map_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (200, 200, 200), -1)
                
                # Confidence fill
                fill_width = int(bar_width * room_data.average_confidence)
                cv2.rectangle(map_image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                             (0, 255, 0), -1)
                
                # Confidence text
                conf_text = f"{room_data.average_confidence:.2f}"
                cv2.putText(map_image, conf_text, (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    def _add_trend_indicators(self, map_image: np.ndarray, emotion_summary: RoomEmotionSummary):
        """Add trend indicators for each room"""
        scale_x = self.config.map_width / self.room_highlighter.map_data.width
        scale_y = self.config.map_height / self.room_highlighter.map_data.height
        
        for room in self.room_highlighter.map_data.rooms:
            room_data = emotion_summary.rooms.get(room.name)
            if room_data:
                center_x = int(room.center[0] * scale_x)
                center_y = int(room.center[1] * scale_y)
                
                # Draw trend arrow
                if room_data.emotion_trend == "improving":
                    # Green up arrow
                    cv2.arrowedLine(map_image, (center_x, center_y + 50), (center_x, center_y + 40), 
                                   (0, 255, 0), 2)
                elif room_data.emotion_trend == "declining":
                    # Red down arrow
                    cv2.arrowedLine(map_image, (center_x, center_y + 40), (center_x, center_y + 50), 
                                   (0, 0, 255), 2)
                else:
                    # Gray horizontal line for stable
                    cv2.line(map_image, (center_x - 10, center_y + 45), (center_x + 10, center_y + 45), 
                            (128, 128, 128), 2)
    
    def save_summary_map(self, emotion_summary: RoomEmotionSummary, output_path: str, 
                        config: Optional[EmotionMapConfig] = None) -> bool:
        """
        Save summary map to file
        
        Args:
            emotion_summary: Room emotion summary data
            output_path: Path to save the image
            config: Optional configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate map
            map_image = self.generate_summary_map(emotion_summary, config)
            
            # Save image
            cv2.imwrite(output_path, map_image)
            logger.info(f"Summary map saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving summary map: {e}")
            return False
    
    def save_emotion_data(self, emotion_summary: RoomEmotionSummary, output_path: str) -> bool:
        """
        Save emotion data as JSON
        
        Args:
            emotion_summary: Room emotion summary data
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to serializable format
            data = {
                "overall_mood": emotion_summary.overall_mood,
                "most_emotional_room": emotion_summary.most_emotional_room,
                "least_emotional_room": emotion_summary.least_emotional_room,
                "total_processing_time": emotion_summary.total_processing_time,
                "rooms": {}
            }
            
            for room_name, room_data in emotion_summary.rooms.items():
                data["rooms"][room_name] = {
                    "dominant_emotion": room_data.dominant_emotion,
                    "emotion_distribution": room_data.emotion_distribution,
                    "average_confidence": room_data.average_confidence,
                    "average_intensity": room_data.average_intensity,
                    "emotion_trend": room_data.emotion_trend,
                    "total_time": room_data.total_time,
                    "emotion_count": len(room_data.emotions)
                }
            
            # Save JSON
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Emotion data saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving emotion data: {e}")
            return False

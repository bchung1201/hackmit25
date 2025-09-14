"""
Room highlighting engine for Roborock maps
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json

from .roborock_parser import RoborockMap, Room, RoborockMapParser

logger = logging.getLogger(__name__)

@dataclass
class HighlightingRule:
    """Rule for room highlighting based on emotions"""
    emotion: str
    room_names: List[str]
    color: Tuple[int, int, int]
    intensity: float
    animation_type: str = "static"  # static, pulse, fade, glow
    priority: int = 1  # Higher number = higher priority

class RoomHighlighter:
    """Main room highlighting engine"""
    
    def __init__(self, map_data: Optional[RoborockMap] = None):
        self.map_data = map_data
        self.parser = RoborockMapParser()
        
        # Current highlighting state
        self.current_highlights: Dict[str, Dict[str, Any]] = {}
        self.highlighting_rules: List[HighlightingRule] = []
        
        # Animation state
        self.animation_time = 0.0
        self.animation_speed = 1.0
        
        # Initialize default rules
        self._setup_default_rules()
        
        logger.info("Room highlighter initialized")
    
    def load_map(self, map_data: RoborockMap):
        """Load Roborock map data"""
        self.map_data = map_data
        logger.info(f"Loaded map with {len(map_data.rooms)} rooms")
    
    def load_map_from_json(self, json_data: Dict[str, Any]):
        """Load map from JSON data"""
        self.map_data = self.parser.parse_json_map(json_data)
        logger.info(f"Loaded map from JSON with {len(self.map_data.rooms)} rooms")
    
    def load_map_from_xml(self, xml_data: str):
        """Load map from XML data"""
        self.map_data = self.parser.parse_xml_map(xml_data)
        logger.info(f"Loaded map from XML with {len(self.map_data.rooms)} rooms")
    
    def _setup_default_rules(self):
        """Setup default emotion-to-room highlighting rules"""
        self.highlighting_rules = [
            # Happy emotions - brighten social areas
            HighlightingRule(
                emotion="happy",
                room_names=["Game Room", "Bar Room", "Living Room"],
                color=(0, 255, 0),
                intensity=0.8,
                animation_type="glow",
                priority=3
            ),
            HighlightingRule(
                emotion="excited",
                room_names=["Game Room", "Bar Room"],
                color=(255, 255, 0),
                intensity=1.0,
                animation_type="pulse",
                priority=4
            ),
            
            # Sad emotions - highlight calming areas
            HighlightingRule(
                emotion="sad",
                room_names=["Living Room", "Bedroom"],
                color=(0, 100, 200),
                intensity=0.6,
                animation_type="fade",
                priority=2
            ),
            
            # Angry emotions - dim certain areas, highlight safe spaces
            HighlightingRule(
                emotion="angry",
                room_names=["Living Room", "Bedroom"],
                color=(255, 0, 0),
                intensity=0.7,
                animation_type="static",
                priority=3
            ),
            
            # Fear - highlight current location and safe spaces
            HighlightingRule(
                emotion="fear",
                room_names=["Living Room", "Bedroom"],
                color=(128, 0, 128),
                intensity=0.8,
                animation_type="pulse",
                priority=3
            ),
            
            # Surprise - highlight current location
            HighlightingRule(
                emotion="surprise",
                room_names=["Game Room", "Living Room"],
                color=(255, 165, 0),
                intensity=0.9,
                animation_type="glow",
                priority=2
            ),
            
            # High arousal - highlight active areas
            HighlightingRule(
                emotion="high_arousal",
                room_names=["Game Room", "Bar Room", "Living Room"],
                color=(255, 69, 0),
                intensity=0.8,
                animation_type="pulse",
                priority=3
            ),
            
            # Positive mood - brighten all social areas
            HighlightingRule(
                emotion="positive",
                room_names=["Game Room", "Bar Room", "Living Room", "Dining Room"],
                color=(0, 255, 127),
                intensity=0.7,
                animation_type="glow",
                priority=2
            ),
            
            # Negative mood - dim social areas, highlight private spaces
            HighlightingRule(
                emotion="negative",
                room_names=["Bedroom", "Living Room"],
                color=(220, 20, 60),
                intensity=0.6,
                animation_type="fade",
                priority=2
            ),
            
            # Mixed emotions - subtle highlighting
            HighlightingRule(
                emotion="mixed",
                room_names=["Living Room", "Hallway"],
                color=(255, 192, 203),
                intensity=0.4,
                animation_type="fade",
                priority=1
            )
        ]
    
    def add_highlighting_rule(self, rule: HighlightingRule):
        """Add a custom highlighting rule"""
        self.highlighting_rules.append(rule)
        logger.info(f"Added highlighting rule for emotion: {rule.emotion}")
    
    def update_emotion(self, emotion: str, intensity: float = 1.0, 
                      current_room: Optional[str] = None, 
                      additional_context: Optional[Dict[str, Any]] = None):
        """
        Update room highlighting based on detected emotion
        
        Args:
            emotion: Detected emotion
            intensity: Emotion intensity (0.0 to 1.0)
            current_room: Current room name (if known)
            additional_context: Additional context information
        """
        if not self.map_data:
            logger.warning("No map data loaded")
            return
        
        # Clear previous highlights
        self.current_highlights.clear()
        
        # Find applicable rules
        applicable_rules = [rule for rule in self.highlighting_rules 
                          if rule.emotion.lower() == emotion.lower()]
        
        if not applicable_rules:
            logger.info(f"No highlighting rules found for emotion: {emotion}")
            return
        
        # Apply rules (highest priority first)
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)
        
        for rule in applicable_rules:
            self._apply_highlighting_rule(rule, intensity, current_room, additional_context)
        
        logger.info(f"Updated highlighting for emotion: {emotion} (intensity: {intensity:.2f})")
    
    def _apply_highlighting_rule(self, rule: HighlightingRule, intensity: float,
                                current_room: Optional[str], additional_context: Optional[Dict[str, Any]]):
        """Apply a specific highlighting rule"""
        for room_name in rule.room_names:
            # Find room in map
            room = next((r for r in self.map_data.rooms if r.name == room_name), None)
            if not room:
                continue
            
            # Calculate final intensity
            final_intensity = rule.intensity * intensity
            
            # Apply highlighting
            room.is_highlighted = True
            room.highlight_color = rule.color
            room.highlight_intensity = final_intensity
            room.emotion_context = rule.emotion
            
            # Store in current highlights
            self.current_highlights[room_name] = {
                'color': rule.color,
                'intensity': final_intensity,
                'animation_type': rule.animation_type,
                'emotion': rule.emotion,
                'priority': rule.priority
            }
    
    def generate_highlighted_map_image(self, width: int = 1000, height: int = 600) -> np.ndarray:
        """
        Generate highlighted map image
        
        Args:
            width: Output image width
            height: Output image height
            
        Returns:
            Highlighted map image (BGR format)
        """
        if not self.map_data:
            logger.warning("No map data available")
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create base map
        map_image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Scale factor
        scale_x = width / self.map_data.width
        scale_y = height / self.map_data.height
        
        # Draw rooms
        for room in self.map_data.rooms:
            self._draw_room(map_image, room, scale_x, scale_y)
        
        # Draw walls
        self._draw_walls(map_image, scale_x, scale_y)
        
        # Draw obstacles
        self._draw_obstacles(map_image, scale_x, scale_y)
        
        # Draw special points
        self._draw_special_points(map_image, scale_x, scale_y)
        
        # Apply highlighting effects
        self._apply_highlighting_effects(map_image, scale_x, scale_y)
        
        return map_image
    
    def _draw_room(self, image: np.ndarray, room: Room, scale_x: float, scale_y: float):
        """Draw a room on the map image"""
        if not room.coordinates:
            return
        
        # Scale coordinates
        scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in room.coordinates]
        
        # Choose color
        if room.is_highlighted and room.highlight_color:
            color = room.highlight_color
        else:
            color = room.color
        
        # Draw room polygon
        pts = np.array(scaled_coords, np.int32)
        cv2.fillPoly(image, [pts], color)
        
        # Draw room border
        cv2.polylines(image, [pts], True, (0, 0, 0), 2)
        
        # Draw room name
        center_x = int(room.center[0] * scale_x)
        center_y = int(room.center[1] * scale_y)
        
        # Add background for text
        text_size = cv2.getTextSize(room.name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, 
                     (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                     (center_x + text_size[0]//2 + 5, center_y + 5),
                     (255, 255, 255), -1)
        
        cv2.putText(image, room.name, (center_x - text_size[0]//2, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _draw_walls(self, image: np.ndarray, scale_x: float, scale_y: float):
        """Draw walls on the map image"""
        for wall in self.map_data.walls:
            x1, y1, x2, y2 = wall
            start_point = (int(x1 * scale_x), int(y1 * scale_y))
            end_point = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.line(image, start_point, end_point, (0, 0, 0), 3)
    
    def _draw_obstacles(self, image: np.ndarray, scale_x: float, scale_y: float):
        """Draw obstacles on the map image"""
        for obs in self.map_data.obstacles:
            x1, y1, x2, y2 = obs
            top_left = (int(x1 * scale_x), int(y1 * scale_y))
            bottom_right = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.rectangle(image, top_left, bottom_right, (100, 100, 100), -1)
    
    def _draw_special_points(self, image: np.ndarray, scale_x: float, scale_y: float):
        """Draw charging station and robot position"""
        # Charging station
        if self.map_data.charging_station:
            x, y = self.map_data.charging_station
            center = (int(x * scale_x), int(y * scale_y))
            cv2.circle(image, center, 10, (0, 255, 0), -1)
            cv2.putText(image, "CS", (center[0] - 10, center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Robot position
        if self.map_data.robot_position:
            x, y = self.map_data.robot_position
            center = (int(x * scale_x), int(y * scale_y))
            cv2.circle(image, center, 8, (255, 0, 0), -1)
            cv2.putText(image, "R", (center[0] - 5, center[1] + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _apply_highlighting_effects(self, image: np.ndarray, scale_x: float, scale_y: float):
        """Apply highlighting effects to the map image"""
        for room_name, highlight_data in self.current_highlights.items():
            room = next((r for r in self.map_data.rooms if r.name == room_name), None)
            if not room or not room.coordinates:
                continue
            
            # Scale coordinates
            scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in room.coordinates]
            pts = np.array(scaled_coords, np.int32)
            
            # Apply animation effects
            animation_type = highlight_data.get('animation_type', 'static')
            intensity = highlight_data.get('intensity', 1.0)
            
            if animation_type == "pulse":
                self._apply_pulse_effect(image, pts, highlight_data['color'], intensity)
            elif animation_type == "glow":
                self._apply_glow_effect(image, pts, highlight_data['color'], intensity)
            elif animation_type == "fade":
                self._apply_fade_effect(image, pts, highlight_data['color'], intensity)
            else:  # static
                self._apply_static_highlight(image, pts, highlight_data['color'], intensity)
    
    def _apply_pulse_effect(self, image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], intensity: float):
        """Apply pulsing effect to room"""
        # Calculate pulse based on animation time
        pulse_factor = 0.5 + 0.5 * np.sin(self.animation_time * 4)
        final_intensity = intensity * pulse_factor
        
        # Create overlay
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend with original
        cv2.addWeighted(image, 1 - final_intensity * 0.3, overlay, final_intensity * 0.3, 0, image)
    
    def _apply_glow_effect(self, image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], intensity: float):
        """Apply glowing effect to room"""
        # Create multiple layers for glow effect
        for i in range(3):
            glow_intensity = intensity * (0.3 - i * 0.1)
            if glow_intensity <= 0:
                continue
            
            # Create slightly larger polygon
            kernel = np.ones((5 + i * 3, 5 + i * 3), np.uint8)
            dilated_pts = cv2.dilate(pts.astype(np.uint8), kernel, iterations=1)
            
            # Create overlay
            overlay = image.copy()
            cv2.fillPoly(overlay, [dilated_pts], color)
            
            # Blend with original
            cv2.addWeighted(image, 1 - glow_intensity * 0.2, overlay, glow_intensity * 0.2, 0, image)
    
    def _apply_fade_effect(self, image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], intensity: float):
        """Apply fading effect to room"""
        # Calculate fade based on animation time
        fade_factor = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(self.animation_time * 2))
        final_intensity = intensity * fade_factor
        
        # Create overlay
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend with original
        cv2.addWeighted(image, 1 - final_intensity * 0.4, overlay, final_intensity * 0.4, 0, image)
    
    def _apply_static_highlight(self, image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], intensity: float):
        """Apply static highlighting to room"""
        # Create overlay
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend with original
        cv2.addWeighted(image, 1 - intensity * 0.5, overlay, intensity * 0.5, 0, image)
    
    def update_animation(self, delta_time: float):
        """Update animation state"""
        self.animation_time += delta_time * self.animation_speed
    
    def get_highlighting_status(self) -> Dict[str, Any]:
        """Get current highlighting status"""
        return {
            "highlighted_rooms": list(self.current_highlights.keys()),
            "total_rooms": len(self.map_data.rooms) if self.map_data else 0,
            "animation_time": self.animation_time,
            "rules_count": len(self.highlighting_rules)
        }
    
    def clear_highlights(self):
        """Clear all room highlights"""
        self.current_highlights.clear()
        if self.map_data:
            for room in self.map_data.rooms:
                room.is_highlighted = False
                room.highlight_color = None
                room.highlight_intensity = 0.0
                room.emotion_context = None
        logger.info("Cleared all room highlights")

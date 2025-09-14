"""
Roborock map parser and data structures
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Room:
    """Represents a room in the Roborock map"""
    id: str
    name: str
    color: Tuple[int, int, int]  # RGB color
    coordinates: List[Tuple[int, int]]  # Room boundary coordinates
    center: Tuple[int, int]  # Room center point
    area: float  # Room area in pixels
    is_highlighted: bool = False
    highlight_color: Optional[Tuple[int, int, int]] = None
    highlight_intensity: float = 0.0
    emotion_context: Optional[str] = None

@dataclass
class RoborockMap:
    """Complete Roborock map data structure"""
    width: int
    height: int
    rooms: List[Room]
    walls: List[Tuple[int, int, int, int]]  # Wall segments (x1, y1, x2, y2)
    obstacles: List[Tuple[int, int, int, int]]  # Obstacle rectangles
    charging_station: Optional[Tuple[int, int]] = None
    robot_position: Optional[Tuple[int, int]] = None
    map_scale: float = 1.0  # Pixels per meter

class RoborockMapParser:
    """Parser for Roborock map data"""
    
    def __init__(self):
        self.default_room_colors = {
            "Bar Room": (255, 255, 0),      # Yellow
            "Laundry Room": (255, 127, 80), # Coral
            "Hallway": (64, 224, 208),      # Turquoise
            "Game Room": (135, 206, 235),   # Sky Blue
            "Bedroom": (147, 112, 219),     # Purple
            "Living Room": (144, 238, 144), # Light Green
            "Kitchen": (255, 182, 193),     # Light Pink
            "Bathroom": (176, 224, 230),    # Powder Blue
            "Office": (221, 160, 221),      # Plum
            "Dining Room": (255, 218, 185)  # Peach
        }
        
        # Emotion-based highlighting colors
        self.emotion_colors = {
            "happy": (0, 255, 0),           # Bright Green
            "excited": (255, 255, 0),       # Bright Yellow
            "sad": (0, 100, 200),           # Deep Blue
            "angry": (255, 0, 0),           # Red
            "fear": (128, 0, 128),          # Purple
            "surprise": (255, 165, 0),      # Orange
            "disgust": (139, 69, 19),       # Brown
            "contempt": (105, 105, 105),    # Dim Gray
            "neutral": (200, 200, 200),     # Light Gray
            "mixed": (255, 192, 203),       # Pink
            "positive": (0, 255, 127),      # Spring Green
            "negative": (220, 20, 60),      # Crimson
            "high_arousal": (255, 69, 0),   # Red Orange
            "excited": (255, 215, 0)        # Gold
        }
    
    def parse_json_map(self, json_data: Dict[str, Any]) -> RoborockMap:
        """
        Parse Roborock map from JSON format
        
        Args:
            json_data: JSON data containing map information
            
        Returns:
            Parsed RoborockMap object
        """
        try:
            # Extract basic map properties
            width = json_data.get('width', 1000)
            height = json_data.get('height', 1000)
            
            # Parse rooms
            rooms = []
            rooms_data = json_data.get('rooms', [])
            
            for room_data in rooms_data:
                room = self._parse_room_data(room_data)
                if room:
                    rooms.append(room)
            
            # Parse walls
            walls = self._parse_walls(json_data.get('walls', []))
            
            # Parse obstacles
            obstacles = self._parse_obstacles(json_data.get('obstacles', []))
            
            # Parse special points
            charging_station = json_data.get('charging_station')
            robot_position = json_data.get('robot_position')
            map_scale = json_data.get('map_scale', 1.0)
            
            return RoborockMap(
                width=width,
                height=height,
                rooms=rooms,
                walls=walls,
                obstacles=obstacles,
                charging_station=charging_station,
                robot_position=robot_position,
                map_scale=map_scale
            )
            
        except Exception as e:
            logger.error(f"Error parsing JSON map: {e}")
            return self._create_default_map()
    
    def parse_xml_map(self, xml_data: str) -> RoborockMap:
        """
        Parse Roborock map from XML format
        
        Args:
            xml_data: XML string containing map information
            
        Returns:
            Parsed RoborockMap object
        """
        try:
            root = ET.fromstring(xml_data)
            
            # Extract basic properties
            width = int(root.get('width', 1000))
            height = int(root.get('height', 1000))
            
            # Parse rooms
            rooms = []
            for room_elem in root.findall('room'):
                room = self._parse_room_xml(room_elem)
                if room:
                    rooms.append(room)
            
            # Parse walls
            walls = []
            for wall_elem in root.findall('wall'):
                x1 = int(wall_elem.get('x1', 0))
                y1 = int(wall_elem.get('y1', 0))
                x2 = int(wall_elem.get('x2', 0))
                y2 = int(wall_elem.get('y2', 0))
                walls.append((x1, y1, x2, y2))
            
            # Parse obstacles
            obstacles = []
            for obs_elem in root.findall('obstacle'):
                x = int(obs_elem.get('x', 0))
                y = int(obs_elem.get('y', 0))
                w = int(obs_elem.get('width', 0))
                h = int(obs_elem.get('height', 0))
                obstacles.append((x, y, x + w, y + h))
            
            # Parse special points
            charging_elem = root.find('charging_station')
            charging_station = None
            if charging_elem is not None:
                charging_station = (
                    int(charging_elem.get('x', 0)),
                    int(charging_elem.get('y', 0))
                )
            
            robot_elem = root.find('robot')
            robot_position = None
            if robot_elem is not None:
                robot_position = (
                    int(robot_elem.get('x', 0)),
                    int(robot_elem.get('y', 0))
                )
            
            map_scale = float(root.get('scale', 1.0))
            
            return RoborockMap(
                width=width,
                height=height,
                rooms=rooms,
                walls=walls,
                obstacles=obstacles,
                charging_station=charging_station,
                robot_position=robot_position,
                map_scale=map_scale
            )
            
        except Exception as e:
            logger.error(f"Error parsing XML map: {e}")
            return self._create_default_map()
    
    def _parse_room_data(self, room_data: Dict[str, Any]) -> Optional[Room]:
        """Parse individual room data from JSON"""
        try:
            room_id = room_data.get('id', '')
            name = room_data.get('name', 'Unknown Room')
            
            # Get coordinates
            coordinates = room_data.get('coordinates', [])
            if not coordinates:
                return None
            
            # Calculate center and area
            center = self._calculate_center(coordinates)
            area = self._calculate_area(coordinates)
            
            # Get color
            color = self.default_room_colors.get(name, (128, 128, 128))
            if 'color' in room_data:
                color = tuple(room_data['color'])
            
            return Room(
                id=room_id,
                name=name,
                color=color,
                coordinates=coordinates,
                center=center,
                area=area
            )
            
        except Exception as e:
            logger.error(f"Error parsing room data: {e}")
            return None
    
    def _parse_room_xml(self, room_elem) -> Optional[Room]:
        """Parse individual room data from XML"""
        try:
            room_id = room_elem.get('id', '')
            name = room_elem.get('name', 'Unknown Room')
            
            # Get coordinates
            coordinates = []
            for coord_elem in room_elem.findall('coordinate'):
                x = int(coord_elem.get('x', 0))
                y = int(coord_elem.get('y', 0))
                coordinates.append((x, y))
            
            if not coordinates:
                return None
            
            # Calculate center and area
            center = self._calculate_center(coordinates)
            area = self._calculate_area(coordinates)
            
            # Get color
            color = self.default_room_colors.get(name, (128, 128, 128))
            if room_elem.get('color'):
                color_str = room_elem.get('color')
                color = tuple(map(int, color_str.split(',')))
            
            return Room(
                id=room_id,
                name=name,
                color=color,
                coordinates=coordinates,
                center=center,
                area=area
            )
            
        except Exception as e:
            logger.error(f"Error parsing room XML: {e}")
            return None
    
    def _parse_walls(self, walls_data: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """Parse wall data"""
        walls = []
        for wall in walls_data:
            try:
                x1 = int(wall.get('x1', 0))
                y1 = int(wall.get('y1', 0))
                x2 = int(wall.get('x2', 0))
                y2 = int(wall.get('y2', 0))
                walls.append((x1, y1, x2, y2))
            except (ValueError, TypeError):
                continue
        return walls
    
    def _parse_obstacles(self, obstacles_data: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """Parse obstacle data"""
        obstacles = []
        for obs in obstacles_data:
            try:
                x = int(obs.get('x', 0))
                y = int(obs.get('y', 0))
                w = int(obs.get('width', 0))
                h = int(obs.get('height', 0))
                obstacles.append((x, y, x + w, y + h))
            except (ValueError, TypeError):
                continue
        return obstacles
    
    def _calculate_center(self, coordinates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate center point of room"""
        if not coordinates:
            return (0, 0)
        
        x_sum = sum(coord[0] for coord in coordinates)
        y_sum = sum(coord[1] for coord in coordinates)
        
        return (x_sum // len(coordinates), y_sum // len(coordinates))
    
    def _calculate_area(self, coordinates: List[Tuple[int, int]]) -> float:
        """Calculate area of room using shoelace formula"""
        if len(coordinates) < 3:
            return 0.0
        
        area = 0.0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        
        return abs(area) / 2.0
    
    def _create_default_map(self) -> RoborockMap:
        """Create a default map for testing"""
        # Create a simple test map based on the provided image
        rooms = [
            Room(
                id="bar_room",
                name="Bar Room",
                color=(255, 255, 0),
                coordinates=[(50, 50), (300, 50), (300, 400), (50, 400)],
                center=(175, 225),
                area=87500.0
            ),
            Room(
                id="laundry_room",
                name="Laundry Room",
                color=(255, 127, 80),
                coordinates=[(300, 50), (500, 50), (500, 200), (300, 200)],
                center=(400, 125),
                area=40000.0
            ),
            Room(
                id="hallway",
                name="Hallway",
                color=(64, 224, 208),
                coordinates=[(300, 200), (500, 200), (500, 250), (300, 250)],
                center=(400, 225),
                area=10000.0
            ),
            Room(
                id="game_room",
                name="Game Room",
                color=(135, 206, 235),
                coordinates=[(500, 50), (800, 50), (800, 200), (500, 200)],
                center=(650, 125),
                area=45000.0
            ),
            Room(
                id="purple_room",
                name="Living Room",
                color=(147, 112, 219),
                coordinates=[(500, 250), (800, 250), (800, 500), (500, 500)],
                center=(650, 375),
                area=62500.0
            )
        ]
        
        return RoborockMap(
            width=1000,
            height=600,
            rooms=rooms,
            walls=[],
            obstacles=[],
            charging_station=(400, 100),
            robot_position=(200, 200),
            map_scale=1.0
        )
    
    def get_emotion_color(self, emotion: str, intensity: float = 1.0) -> Tuple[int, int, int]:
        """
        Get highlighting color for emotion
        
        Args:
            emotion: Emotion name
            intensity: Intensity value (0.0 to 1.0)
            
        Returns:
            RGB color tuple
        """
        base_color = self.emotion_colors.get(emotion.lower(), (200, 200, 200))
        
        # Adjust intensity
        if intensity < 1.0:
            # Blend with white for lower intensity
            white = (255, 255, 255)
            base_color = tuple(
                int(base_color[i] * intensity + white[i] * (1 - intensity))
                for i in range(3)
            )
        
        return base_color

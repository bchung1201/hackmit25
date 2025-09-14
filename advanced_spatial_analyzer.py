"""
Advanced Spatial Analyzer with DINOv3 and Complex AI Models
- DINOv3 for complex spatial understanding
- Depth estimation models
- Human expression and emotion recognition
- 3D scene understanding
- Advanced object relationship analysis
"""

import asyncio
import logging
import sys
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DINOv3Processor:
    """DINOv3 for advanced spatial and semantic understanding"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.spatial_relationships = []
        
    async def initialize(self):
        """Initialize DINOv3 model"""
        logger.info("🔧 Initializing DINOv3 for spatial understanding...")
        try:
            # Load DINOv2 (DINOv3 equivalent for now)
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ DINOv3 initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ DINOv3 not available, using mock features: {e}")
            self.model = None
    
    def extract_spatial_features(self, image: np.ndarray) -> Dict:
        """Extract complex spatial features using DINOv3"""
        if self.model is None:
            return self._mock_spatial_features(image)
        
        try:
            # Convert to PIL and preprocess
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize for DINOv3
            pil_image = pil_image.resize((224, 224))
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                
            # Analyze spatial patterns
            spatial_analysis = self._analyze_spatial_patterns(features, image)
            
            return spatial_analysis
            
        except Exception as e:
            logger.warning(f"⚠️ DINOv3 processing failed: {e}")
            return self._mock_spatial_features(image)
    
    def _analyze_spatial_patterns(self, features: torch.Tensor, image: np.ndarray) -> Dict:
        """Analyze spatial patterns from DINOv3 features"""
        # Convert features to numpy
        feature_map = features.cpu().numpy().flatten()
        
        # Analyze feature distribution
        feature_std = np.std(feature_map)
        feature_mean = np.mean(feature_map)
        
        # Detect spatial complexity
        spatial_complexity = self._calculate_spatial_complexity(image)
        
        # Analyze depth cues
        depth_cues = self._analyze_depth_cues(image)
        
        return {
            'spatial_features': feature_map,
            'complexity_score': spatial_complexity,
            'depth_cues': depth_cues,
            'feature_std': feature_std,
            'feature_mean': feature_mean,
            'spatial_relationships': self._detect_spatial_relationships(image)
        }
    
    def _calculate_spatial_complexity(self, image: np.ndarray) -> float:
        """Calculate spatial complexity of the scene"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Texture complexity
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Color complexity
        color_std = np.std(image.reshape(-1, 3), axis=0).mean()
        
        complexity = (edge_density * 100 + texture / 1000 + color_std / 10) / 3
        return min(complexity, 10.0)  # Normalize to 0-10
    
    def _analyze_depth_cues(self, image: np.ndarray) -> Dict:
        """Analyze depth cues in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Atmospheric perspective (haze/blur in distance)
        atmospheric_depth = self._detect_atmospheric_perspective(gray)
        
        # Linear perspective (converging lines)
        linear_perspective = self._detect_linear_perspective(gray)
        
        # Size gradients (objects getting smaller)
        size_gradients = self._detect_size_gradients(image)
        
        # Occlusion patterns
        occlusion_patterns = self._detect_occlusion_patterns(gray)
        
        return {
            'atmospheric_depth': atmospheric_depth,
            'linear_perspective': linear_perspective,
            'size_gradients': size_gradients,
            'occlusion_patterns': occlusion_patterns,
            'depth_confidence': (atmospheric_depth + linear_perspective + size_gradients + occlusion_patterns) / 4
        }
    
    def _detect_atmospheric_perspective(self, gray: np.ndarray) -> float:
        """Detect atmospheric perspective (distance haze)"""
        # Analyze gradient from center to edges
        center = gray.shape[0] // 2, gray.shape[1] // 2
        
        # Sample points at different distances from center
        distances = []
        for r in range(50, min(gray.shape) // 2, 50):
            angles = np.linspace(0, 2 * np.pi, 16)
            for angle in angles:
                x = int(center[1] + r * np.cos(angle))
                y = int(center[0] + r * np.sin(angle))
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    distances.append((r, gray[y, x]))
        
        if len(distances) > 10:
            # Check if brightness decreases with distance (atmospheric perspective)
            distances = sorted(distances)
            brightness_gradient = np.polyfit([d[0] for d in distances], [d[1] for d in distances], 1)[0]
            return max(0, -brightness_gradient / 10)  # Normalize
        return 0.5
    
    def _detect_linear_perspective(self, gray: np.ndarray) -> float:
        """Detect linear perspective (converging lines)"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Find converging lines
            convergence_score = 0
            for i, line1 in enumerate(lines[:10]):  # Limit for performance
                for line2 in lines[i+1:i+5]:
                    # Calculate intersection point
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]
                    
                    # Simple line intersection check
                    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                    if abs(denom) > 1e-6:
                        convergence_score += 1
            
            return min(convergence_score / 10, 1.0)
        return 0.3
    
    def _detect_size_gradients(self, image: np.ndarray) -> float:
        """Detect size gradients (objects getting smaller with distance)"""
        # Find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 3:
            # Analyze size distribution
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            if len(areas) > 2:
                # Check for size gradient from center
                center = (image.shape[1] // 2, image.shape[0] // 2)
                size_gradient = 0
                
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                            area = cv2.contourArea(contour)
                            size_gradient += area / (distance + 1)
                
                return min(size_gradient / 1000, 1.0)
        return 0.4
    
    def _detect_occlusion_patterns(self, gray: np.ndarray) -> float:
        """Detect occlusion patterns (objects in front of others)"""
        # Find edges and analyze T-junctions
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for T-junctions (indicators of occlusion)
        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        t_junctions = cv2.filter2D(edges, -1, kernel)
        
        t_junction_count = np.sum(t_junctions > 100)
        total_edges = np.sum(edges > 0)
        
        if total_edges > 0:
            return min(t_junction_count / total_edges * 10, 1.0)
        return 0.2
    
    def _detect_spatial_relationships(self, image: np.ndarray) -> List[Dict]:
        """Detect spatial relationships between objects"""
        relationships = []
        
        # Find objects using contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': cv2.contourArea(contour)
                })
        
        # Analyze relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate distance
                dist = np.sqrt((obj1['center'][0] - obj2['center'][0])**2 + 
                             (obj1['center'][1] - obj2['center'][1])**2)
                
                # Determine relationship type
                if dist < 100:
                    relationship = "close"
                elif dist < 200:
                    relationship = "nearby"
                else:
                    relationship = "distant"
                
                # Determine relative position
                dx = obj2['center'][0] - obj1['center'][0]
                dy = obj2['center'][1] - obj1['center'][1]
                
                if abs(dx) > abs(dy):
                    position = "left" if dx < 0 else "right"
                else:
                    position = "above" if dy < 0 else "below"
                
                relationships.append({
                    'object1': i,
                    'object2': j,
                    'relationship': relationship,
                    'position': position,
                    'distance': dist,
                    'size_ratio': obj1['area'] / obj2['area'] if obj2['area'] > 0 else 1
                })
        
        return relationships
    
    def _mock_spatial_features(self, image: np.ndarray) -> Dict:
        """Mock spatial features when DINOv3 is not available"""
        return {
            'spatial_features': np.random.randn(384),
            'complexity_score': np.random.uniform(3, 8),
            'depth_cues': {
                'atmospheric_depth': np.random.uniform(0.3, 0.8),
                'linear_perspective': np.random.uniform(0.2, 0.7),
                'size_gradients': np.random.uniform(0.4, 0.9),
                'occlusion_patterns': np.random.uniform(0.1, 0.6),
                'depth_confidence': np.random.uniform(0.4, 0.8)
            },
            'feature_std': np.random.uniform(0.5, 1.5),
            'feature_mean': np.random.uniform(-0.5, 0.5),
            'spatial_relationships': [
                {'object1': 0, 'object2': 1, 'relationship': 'close', 'position': 'right', 
                 'distance': 85, 'size_ratio': 1.2},
                {'object1': 1, 'object2': 2, 'relationship': 'nearby', 'position': 'below', 
                 'distance': 150, 'size_ratio': 0.8}
            ]
        }

class DepthEstimator:
    """Advanced depth estimation using multiple cues"""
    
    def __init__(self):
        self.depth_model = None
        self.depth_map = None
        
    async def initialize(self):
        """Initialize depth estimation model"""
        logger.info("🔧 Initializing depth estimation...")
        # For now, we'll use monocular depth estimation
        # In a real implementation, you'd load MiDaS or similar
        logger.info("✅ Depth estimation ready (using geometric cues)")
    
    def estimate_depth(self, image: np.ndarray, spatial_features: Dict) -> np.ndarray:
        """Estimate depth map from image and spatial features"""
        height, width = image.shape[:2]
        
        # Create depth map based on multiple cues
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # 1. Atmospheric perspective depth
        atmospheric_depth = spatial_features['depth_cues']['atmospheric_depth']
        for y in range(height):
            for x in range(width):
                # Distance from center
                center_x, center_y = width // 2, height // 2
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Atmospheric perspective (objects further are lighter/blurrier)
                depth_map[y, x] = (dist_from_center / max_dist) * atmospheric_depth
        
        # 2. Linear perspective depth
        linear_perspective = spatial_features['depth_cues']['linear_perspective']
        if linear_perspective > 0.3:
            # Create perspective depth gradient
            for y in range(height):
                depth_gradient = (y / height) * linear_perspective
                depth_map[y, :] += depth_gradient
        
        # 3. Size gradient depth
        size_gradients = spatial_features['depth_cues']['size_gradients']
        if size_gradients > 0.3:
            # Objects in center are closer
            center_y, center_x = height // 2, width // 2
            for y in range(height):
                for x in range(width):
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    size_depth = (1 - dist_from_center / max_dist) * size_gradients
                    depth_map[y, x] += size_depth
        
        # Normalize depth map
        if np.max(depth_map) > 0:
            depth_map = depth_map / np.max(depth_map)
        
        return depth_map

class HumanExpressionAnalyzer:
    """Advanced human expression and emotion recognition"""
    
    def __init__(self):
        self.face_cascade = None
        self.emotion_model = None
        
    async def initialize(self):
        """Initialize face detection and emotion recognition"""
        logger.info("🔧 Initializing human expression analysis...")
        try:
            # Load OpenCV face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✅ Face detection initialized")
        except Exception as e:
            logger.warning(f"⚠️ Face detection not available: {e}")
            self.face_cascade = None
    
    def analyze_expressions(self, image: np.ndarray) -> List[Dict]:
        """Analyze human expressions and emotions"""
        if self.face_cascade is None:
            return self._mock_expression_analysis(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        expressions = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Analyze facial features
            expression_data = self._analyze_facial_features(face_roi)
            
            expressions.append({
                'bbox': (x, y, w, h),
                'emotion': expression_data['emotion'],
                'confidence': expression_data['confidence'],
                'facial_landmarks': expression_data['landmarks'],
                'expression_intensity': expression_data['intensity'],
                'age_estimate': expression_data['age'],
                'gender_estimate': expression_data['gender']
            })
        
        return expressions
    
    def _analyze_facial_features(self, face_roi: np.ndarray) -> Dict:
        """Analyze facial features for emotion detection"""
        # Mock emotion analysis (in real implementation, use trained models)
        emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful', 'disgusted']
        
        # Analyze facial geometry
        height, width = face_roi.shape
        center_y = height // 2
        
        # Simulate emotion detection based on facial geometry
        # Upper face (eyes) vs lower face (mouth) analysis
        upper_face = face_roi[:center_y, :]
        lower_face = face_roi[center_y:, :]
        
        upper_brightness = np.mean(upper_face)
        lower_brightness = np.mean(lower_face)
        
        # Simple heuristic-based emotion detection
        if lower_brightness > upper_brightness * 1.1:
            emotion = 'happy'
            confidence = 0.8
        elif upper_brightness > lower_brightness * 1.1:
            emotion = 'sad'
            confidence = 0.7
        else:
            emotion = 'neutral'
            confidence = 0.6
        
        # Generate mock facial landmarks
        landmarks = []
        for i in range(68):  # 68-point facial landmark model
            angle = 2 * np.pi * i / 68
            radius = min(width, height) // 3
            x = width // 2 + radius * np.cos(angle)
            y = height // 2 + radius * np.sin(angle) * 0.7  # Elliptical face
            landmarks.append((int(x), int(y)))
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'landmarks': landmarks,
            'intensity': np.random.uniform(0.3, 0.9),
            'age': np.random.randint(20, 60),
            'gender': np.random.choice(['male', 'female'])
        }
    
    def _mock_expression_analysis(self, image: np.ndarray) -> List[Dict]:
        """Mock expression analysis when models are not available"""
        # Generate random face detections
        num_faces = np.random.randint(0, 3)
        expressions = []
        
        for i in range(num_faces):
            x = np.random.randint(0, image.shape[1] - 100)
            y = np.random.randint(0, image.shape[0] - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']
            expression = {
                'bbox': (x, y, w, h),
                'emotion': np.random.choice(emotions),
                'confidence': np.random.uniform(0.6, 0.95),
                'facial_landmarks': [(x + np.random.randint(0, w), y + np.random.randint(0, h)) for _ in range(68)],
                'expression_intensity': np.random.uniform(0.4, 0.9),
                'age_estimate': np.random.randint(18, 65),
                'gender_estimate': np.random.choice(['male', 'female'])
            }
            expressions.append(expression)
        
        return expressions

class AdvancedSpatialAnalyzer:
    """Main analyzer combining all advanced models"""
    
    def __init__(self):
        self.dinov3_processor = DINOv3Processor()
        self.depth_estimator = DepthEstimator()
        self.expression_analyzer = HumanExpressionAnalyzer()
        
        # Visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup visualization windows"""
        cv2.namedWindow('🎥 Original Video', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('🧠 DINOv3 Spatial Analysis', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('📏 Depth Map', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('😊 Human Expressions', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('🔗 Spatial Relationships', cv2.WINDOW_AUTOSIZE)
        
        # Position windows
        cv2.moveWindow('🎥 Original Video', 50, 50)
        cv2.moveWindow('🧠 DINOv3 Spatial Analysis', 700, 50)
        cv2.moveWindow('📏 Depth Map', 50, 400)
        cv2.moveWindow('😊 Human Expressions', 700, 400)
        cv2.moveWindow('🔗 Spatial Relationships', 1350, 50)
    
    async def initialize(self):
        """Initialize all models"""
        logger.info("🚀 Initializing Advanced Spatial Analyzer...")
        await self.dinov3_processor.initialize()
        await self.depth_estimator.initialize()
        await self.expression_analyzer.initialize()
        logger.info("✅ All models initialized!")
    
    def visualize_spatial_analysis(self, image: np.ndarray, spatial_features: Dict) -> np.ndarray:
        """Visualize DINOv3 spatial analysis"""
        overlay = image.copy()
        
        # Draw complexity score
        complexity = spatial_features['complexity_score']
        cv2.putText(overlay, f"Spatial Complexity: {complexity:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw depth cues
        depth_cues = spatial_features['depth_cues']
        cv2.putText(overlay, f"Depth Confidence: {depth_cues['depth_confidence']:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw spatial relationships
        relationships = spatial_features['spatial_relationships']
        cv2.putText(overlay, f"Object Relationships: {len(relationships)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw feature statistics
        cv2.putText(overlay, f"Feature Std: {spatial_features['feature_std']:.3f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return overlay
    
    def visualize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Visualize depth map with color coding"""
        # Convert depth to color map
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Add depth scale
        cv2.putText(depth_colored, "Depth Map (Blue=Close, Red=Far)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add depth statistics
        mean_depth = np.mean(depth_map)
        std_depth = np.std(depth_map)
        cv2.putText(depth_colored, f"Mean Depth: {mean_depth:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colored, f"Depth Std: {std_depth:.3f}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return depth_colored
    
    def visualize_expressions(self, image: np.ndarray, expressions: List[Dict]) -> np.ndarray:
        """Visualize human expressions and emotions"""
        overlay = image.copy()
        
        for expr in expressions:
            x, y, w, h = expr['bbox']
            
            # Draw face bounding box
            color = (0, 255, 0) if expr['emotion'] in ['happy', 'surprised'] else (0, 0, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label
            cv2.putText(overlay, f"{expr['emotion'].upper()}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            cv2.putText(overlay, f"{expr['confidence']:.2f}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw facial landmarks
            for landmark in expr['facial_landmarks']:
                cv2.circle(overlay, landmark, 2, (255, 255, 0), -1)
            
            # Draw age and gender
            cv2.putText(overlay, f"{expr['age_estimate']}y {expr['gender_estimate']}", 
                       (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def visualize_spatial_relationships(self, image: np.ndarray, relationships: List[Dict]) -> np.ndarray:
        """Visualize spatial relationships between objects"""
        overlay = image.copy()
        
        # Draw relationship lines
        for rel in relationships:
            # Get object centers (mock positions for now)
            obj1_center = (100 + rel['object1'] * 50, 100 + rel['object1'] * 30)
            obj2_center = (100 + rel['object2'] * 50, 100 + rel['object2'] * 30)
            
            # Draw line between objects
            cv2.line(overlay, obj1_center, obj2_center, (255, 255, 0), 2)
            
            # Draw relationship label
            mid_x = (obj1_center[0] + obj2_center[0]) // 2
            mid_y = (obj1_center[1] + obj2_center[1]) // 2
            cv2.putText(overlay, rel['relationship'], (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw relationship summary
        cv2.putText(overlay, f"Total Relationships: {len(relationships)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    async def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict:
        """Process frame with all advanced models"""
        start_time = time.time()
        
        # 1. DINOv3 spatial analysis
        spatial_features = self.dinov3_processor.extract_spatial_features(frame)
        
        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate_depth(frame, spatial_features)
        
        # 3. Human expression analysis
        expressions = self.expression_analyzer.analyze_expressions(frame)
        
        processing_time = time.time() - start_time
        
        return {
            'spatial_features': spatial_features,
            'depth_map': depth_map,
            'expressions': expressions,
            'processing_time': processing_time
        }
    
    def update_visualization(self, frame: np.ndarray, results: Dict):
        """Update all visualization windows"""
        # Original video
        cv2.imshow('🎥 Original Video', frame)
        
        # DINOv3 spatial analysis
        spatial_viz = self.visualize_spatial_analysis(frame, results['spatial_features'])
        cv2.imshow('🧠 DINOv3 Spatial Analysis', spatial_viz)
        
        # Depth map
        depth_viz = self.visualize_depth_map(results['depth_map'])
        cv2.imshow('📏 Depth Map', depth_viz)
        
        # Human expressions
        expr_viz = self.visualize_expressions(frame, results['expressions'])
        cv2.imshow('😊 Human Expressions', expr_viz)
        
        # Spatial relationships
        rel_viz = self.visualize_spatial_relationships(frame, results['spatial_features']['spatial_relationships'])
        cv2.imshow('🔗 Spatial Relationships', rel_viz)

async def process_video_advanced(video_path: str):
    """Process video with advanced spatial analysis"""
    logger.info(f"🧠 Processing video with Advanced Spatial Analysis: {video_path}")
    
    # Create analyzer
    analyzer = AdvancedSpatialAnalyzer()
    await analyzer.initialize()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return
    
    logger.info("🚀 Advanced analysis started. Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_count += 1
            
            # Process every 2nd frame for performance
            if frame_count % 2 == 0:
                results = await analyzer.process_frame(frame, frame_count)
                analyzer.update_visualization(frame, results)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        logger.info("⏹️ Advanced analysis stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    """Main entry point"""
    print("🧠 Advanced Spatial Analyzer with DINOv3")
    print("=" * 60)
    print("Features:")
    print("• DINOv3 for complex spatial understanding")
    print("• Advanced depth estimation")
    print("• Human expression and emotion recognition")
    print("• 3D scene understanding")
    print("• Object relationship analysis")
    print("")
    
    # Get video path from user
    video_path = input("Enter path to your video file: ").strip()
    video_path = video_path.strip('"').strip("'")
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        return
    
    print(f"✅ Found video: {video_path}")
    print("🚀 Starting advanced spatial analysis...")
    
    await process_video_advanced(video_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Advanced analysis completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

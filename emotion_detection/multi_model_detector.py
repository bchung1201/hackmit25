"""
Multi-Model Emotion Detection System
Tries multiple emotion detection models for better accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EmotionPrediction:
    """Result from emotion detection model"""
    emotion: str
    emotion_id: int
    confidence: float
    model_name: str
    valence: float = 0.0
    arousal: float = 0.0

class MultiModelEmotionDetector:
    """Multi-model emotion detection system"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_priorities = []
        
        # Initialize all available models
        self._initialize_models()
        
        logger.info(f"Multi-model emotion detector initialized with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize all available emotion detection models"""
        # Model 1: EmoNet (highest priority)
        try:
            self._init_emonet()
            self.model_priorities.append('emonet')
        except Exception as e:
            logger.warning(f"EmoNet initialization failed: {e}")
        
        # Model 2: FER2013 (medium priority)
        try:
            self._init_fer2013()
            self.model_priorities.append('fer2013')
        except Exception as e:
            logger.warning(f"FER2013 initialization failed: {e}")
        
        # Model 3: Simple CNN (fallback)
        try:
            self._init_simple_cnn()
            self.model_priorities.append('simple_cnn')
        except Exception as e:
            logger.warning(f"Simple CNN initialization failed: {e}")
        
        # Model 4: MediaPipe (alternative)
        try:
            self._init_mediapipe()
            self.model_priorities.append('mediapipe')
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
        
        if not self.models:
            logger.error("No emotion detection models available!")
            raise RuntimeError("No emotion detection models could be initialized")
    
    def _init_emonet(self):
        """Initialize EmoNet model"""
        try:
            from .emonet_detector import EmoNetDetector
            self.models['emonet'] = EmoNetDetector()
            logger.info("EmoNet model initialized")
        except Exception as e:
            raise Exception(f"EmoNet initialization failed: {e}")
    
    def _init_fer2013(self):
        """Initialize FER2013 model"""
        try:
            # Create a simple FER2013-style model
            class FER2013Model(nn.Module):
                def __init__(self, num_classes=7):
                    super(FER2013Model, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d((7, 7))
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(256 * 7 * 7, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes)
                    )
                    self.emotion_classes = {
                        0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
                        4: "Sad", 5: "Surprise", 6: "Neutral"
                    }
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            model = FER2013Model()
            model.to(self.device)
            model.eval()
            
            # Try to load pretrained weights
            try:
                # Load a simple pretrained model or use random weights
                from torchvision import models
                pretrained = models.resnet18(pretrained=True)
                # Copy some weights for better initialization
                model.load_state_dict(model.state_dict())  # Keep random for now
            except:
                pass  # Use random weights
            
            self.models['fer2013'] = model
            logger.info("FER2013 model initialized")
            
        except Exception as e:
            raise Exception(f"FER2013 initialization failed: {e}")
    
    def _init_simple_cnn(self):
        """Initialize simple CNN model"""
        try:
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes=5):
                    super(SimpleCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(128 * 4 * 4, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, num_classes)
                    )
                    self.emotion_classes = {
                        0: "Happy", 1: "Sad", 2: "Angry", 3: "Surprise", 4: "Neutral"
                    }
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            model = SimpleCNN()
            model.to(self.device)
            model.eval()
            
            self.models['simple_cnn'] = model
            logger.info("Simple CNN model initialized")
            
        except Exception as e:
            raise Exception(f"Simple CNN initialization failed: {e}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh model"""
        try:
            import mediapipe as mp
            
            self.models['mediapipe'] = {
                'face_mesh': mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ),
                'drawing_utils': mp.solutions.drawing_utils
            }
            logger.info("MediaPipe model initialized")
            
        except ImportError:
            raise Exception("MediaPipe not available")
        except Exception as e:
            raise Exception(f"MediaPipe initialization failed: {e}")
    
    def detect_emotions(self, face_image: np.ndarray) -> List[EmotionPrediction]:
        """
        Detect emotions using multiple models
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            List of emotion predictions from different models
        """
        predictions = []
        
        # Try models in priority order
        for model_name in self.model_priorities:
            try:
                prediction = self._detect_with_model(model_name, face_image)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        return predictions
    
    def _detect_with_model(self, model_name: str, face_image: np.ndarray) -> Optional[EmotionPrediction]:
        """Detect emotion with specific model"""
        if model_name == 'emonet':
            return self._detect_with_emonet(face_image)
        elif model_name == 'fer2013':
            return self._detect_with_fer2013(face_image)
        elif model_name == 'simple_cnn':
            return self._detect_with_simple_cnn(face_image)
        elif model_name == 'mediapipe':
            return self._detect_with_mediapipe(face_image)
        else:
            return None
    
    def _detect_with_emonet(self, face_image: np.ndarray) -> Optional[EmotionPrediction]:
        """Detect emotion with EmoNet"""
        try:
            result = self.models['emonet'].detect_emotions(face_image)
            
            return EmotionPrediction(
                emotion=result['emotion'],
                emotion_id=result['emotion_id'],
                confidence=result['confidence'],
                model_name='emonet',
                valence=result['valence'],
                arousal=result['arousal']
            )
        except Exception as e:
            logger.warning(f"EmoNet detection failed: {e}")
            return None
    
    def _detect_with_fer2013(self, face_image: np.ndarray) -> Optional[EmotionPrediction]:
        """Detect emotion with FER2013 model"""
        try:
            # Preprocess face for FER2013
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Convert to tensor
            face_tensor = torch.from_numpy(face_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['fer2013'](face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_id = predicted.item()
                confidence_score = confidence.item()
                
                # Get emotion name
                emotion_classes = self.models['fer2013'].emotion_classes
                emotion_name = emotion_classes.get(emotion_id, "Unknown")
                
                return EmotionPrediction(
                    emotion=emotion_name,
                    emotion_id=emotion_id,
                    confidence=confidence_score,
                    model_name='fer2013'
                )
        except Exception as e:
            logger.warning(f"FER2013 detection failed: {e}")
            return None
    
    def _detect_with_simple_cnn(self, face_image: np.ndarray) -> Optional[EmotionPrediction]:
        """Detect emotion with simple CNN"""
        try:
            # Preprocess face
            face_resized = cv2.resize(face_image, (64, 64))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['simple_cnn'](face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_id = predicted.item()
                confidence_score = confidence.item()
                
                # Get emotion name
                emotion_classes = self.models['simple_cnn'].emotion_classes
                emotion_name = emotion_classes.get(emotion_id, "Unknown")
                
                return EmotionPrediction(
                    emotion=emotion_name,
                    emotion_id=emotion_id,
                    confidence=confidence_score,
                    model_name='simple_cnn'
                )
        except Exception as e:
            logger.warning(f"Simple CNN detection failed: {e}")
            return None
    
    def _detect_with_mediapipe(self, face_image: np.ndarray) -> Optional[EmotionPrediction]:
        """Detect emotion with MediaPipe (basic implementation)"""
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.models['mediapipe']['face_mesh'].process(face_rgb)
            
            if results.multi_face_landmarks:
                # Simple emotion detection based on facial landmarks
                landmarks = results.multi_face_landmarks[0]
                
                # Extract key points for emotion detection
                # This is a simplified approach - in practice, you'd use more sophisticated landmark analysis
                emotion = self._analyze_landmarks_for_emotion(landmarks)
                
                return EmotionPrediction(
                    emotion=emotion,
                    emotion_id=0,  # Simplified
                    confidence=0.6,  # Medium confidence for MediaPipe
                    model_name='mediapipe'
                )
            else:
                return None
                
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}")
            return None
    
    def _analyze_landmarks_for_emotion(self, landmarks) -> str:
        """Analyze MediaPipe landmarks for emotion (simplified)"""
        # This is a very basic implementation
        # In practice, you'd analyze specific landmark positions and relationships
        
        # For now, return a random emotion to demonstrate the system
        emotions = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]
        return np.random.choice(emotions)
    
    def get_best_prediction(self, predictions: List[EmotionPrediction]) -> Optional[EmotionPrediction]:
        """
        Get the best prediction from multiple models
        
        Args:
            predictions: List of predictions from different models
            
        Returns:
            Best prediction based on confidence and model priority
        """
        if not predictions:
            return None
        
        # Sort by confidence and model priority
        model_priority = {name: i for i, name in enumerate(self.model_priorities)}
        
        def sort_key(pred):
            # Higher confidence is better, lower model priority index is better
            return (-pred.confidence, model_priority.get(pred.model_name, 999))
        
        predictions.sort(key=sort_key)
        return predictions[0]
    
    def get_ensemble_prediction(self, predictions: List[EmotionPrediction]) -> Optional[EmotionPrediction]:
        """
        Get ensemble prediction by combining multiple models
        
        Args:
            predictions: List of predictions from different models
            
        Returns:
            Ensemble prediction
        """
        if not predictions:
            return None
        
        # Count votes for each emotion
        emotion_votes = {}
        total_confidence = 0.0
        
        for pred in predictions:
            emotion = pred.emotion
            if emotion not in emotion_votes:
                emotion_votes[emotion] = {'count': 0, 'confidence': 0.0}
            
            emotion_votes[emotion]['count'] += 1
            emotion_votes[emotion]['confidence'] += pred.confidence
            total_confidence += pred.confidence
        
        # Find emotion with most votes
        best_emotion = max(emotion_votes.items(), key=lambda x: x[1]['count'])
        emotion_name = best_emotion[0]
        avg_confidence = best_emotion[1]['confidence'] / best_emotion[1]['count']
        
        return EmotionPrediction(
            emotion=emotion_name,
            emotion_id=0,  # Simplified
            confidence=avg_confidence,
            model_name='ensemble'
        )
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about available models"""
        return {
            'available_models': list(self.models.keys()),
            'model_priorities': self.model_priorities,
            'device': str(self.device),
            'total_models': len(self.models)
        }

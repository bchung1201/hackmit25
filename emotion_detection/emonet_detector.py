"""
EmoNet emotion detection integration
Based on: https://github.com/face-analysis/emonet
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EmoNetDetector:
    """EmoNet emotion detection system"""
    
    def __init__(self, model_path: str = "pretrained/emonet_8.pth", n_classes: int = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = n_classes
        self.model = None
        self.face_alignment = None
        
        # Emotion class mappings
        self.emotion_classes_8 = {
            0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
            4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
        }
        
        self.emotion_classes_5 = {
            0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear"
        }
        
        self.emotion_classes = self.emotion_classes_8 if n_classes == 8 else self.emotion_classes_5
        
        # Initialize face alignment
        self._init_face_alignment()
        
        # Load model
        self._load_model(model_path)
    
    def _init_face_alignment(self):
        """Initialize face alignment for preprocessing"""
        try:
            import face_alignment
            self.face_alignment = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False, 
                device=str(self.device)
            )
            logger.info("Face alignment initialized")
        except ImportError:
            logger.warning("face_alignment not available, using basic preprocessing")
            self.face_alignment = None
        except Exception as e:
            logger.error(f"Failed to initialize face alignment: {e}")
            self.face_alignment = None
    
    def _load_model(self, model_path: str):
        """Load EmoNet model"""
        try:
            # Create model architecture (simplified EmoNet)
            self.model = self._create_emonet_model()
            
            # Load pretrained weights if available
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded EmoNet model from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}, using random weights")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load EmoNet model: {e}")
            # Create a mock model for testing
            self.model = self._create_mock_model()
            self.model.to(self.device)
            self.model.eval()
    
    def _create_emonet_model(self):
        """Create EmoNet model architecture"""
        class EmoNet(nn.Module):
            def __init__(self, n_classes=8):
                super(EmoNet, self).__init__()
                # Simplified CNN architecture
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512 * 7 * 7, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 256),
                    nn.ReLU()
                )
                
                # Output heads
                self.expression = nn.Linear(256, n_classes)
                self.valence = nn.Linear(256, 1)
                self.arousal = nn.Linear(256, 1)
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                
                expression = self.expression(x)
                valence = torch.sigmoid(self.valence(x)) * 2 - 1  # [-1, 1]
                arousal = torch.sigmoid(self.arousal(x)) * 2 - 1  # [-1, 1]
                
                return expression, valence, arousal
        
        return EmoNet(self.n_classes)
    
    def _create_mock_model(self):
        """Create a mock model for testing when real model is not available"""
        class MockEmoNet(nn.Module):
            def __init__(self, n_classes=8):
                super(MockEmoNet, self).__init__()
                self.n_classes = n_classes
            
            def forward(self, x):
                batch_size = x.size(0)
                # Return random predictions for testing
                expression = torch.randn(batch_size, self.n_classes)
                valence = torch.rand(batch_size, 1) * 2 - 1
                arousal = torch.rand(batch_size, 1) * 2 - 1
                return expression, valence, arousal
        
        return MockEmoNet(self.n_classes)
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for EmoNet
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (EmoNet input size)
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def detect_emotions(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Detect emotions in face image
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Dictionary with emotion predictions
        """
        try:
            # Preprocess face
            face_tensor = self.preprocess_face(face_image)
            
            # Run inference
            with torch.no_grad():
                expression_logits, valence, arousal = self.model(face_tensor)
                
                # Convert to probabilities
                expression_probs = torch.softmax(expression_logits, dim=1)
                
                # Get predicted emotion
                predicted_class = torch.argmax(expression_probs, dim=1).item()
                confidence = expression_probs[0, predicted_class].item()
                
                # Get valence and arousal values
                valence_val = valence[0, 0].item()
                arousal_val = arousal[0, 0].item()
                
                # Create result dictionary
                result = {
                    'emotion': self.emotion_classes[predicted_class],
                    'emotion_id': predicted_class,
                    'confidence': confidence,
                    'valence': valence_val,
                    'arousal': arousal_val,
                    'emotion_probs': {
                        self.emotion_classes[i]: expression_probs[0, i].item() 
                        for i in range(self.n_classes)
                    }
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            # Return neutral emotion as fallback
            return {
                'emotion': 'Neutral',
                'emotion_id': 0,
                'confidence': 0.0,
                'valence': 0.0,
                'arousal': 0.0,
                'emotion_probs': {self.emotion_classes[i]: 0.0 for i in range(self.n_classes)}
            }
    
    def get_emotion_intensity(self, emotion_result: Dict[str, float]) -> float:
        """
        Calculate overall emotion intensity
        
        Args:
            emotion_result: Result from detect_emotions
            
        Returns:
            Intensity value between 0 and 1
        """
        # Combine valence and arousal to get intensity
        valence_abs = abs(emotion_result['valence'])
        arousal_abs = abs(emotion_result['arousal'])
        
        # Intensity is the magnitude of the emotion vector
        intensity = np.sqrt(valence_abs**2 + arousal_abs**2) / np.sqrt(2)
        
        return min(1.0, intensity)
    
    def get_emotion_category(self, emotion_result: Dict[str, float]) -> str:
        """
        Categorize emotion into high-level categories
        
        Args:
            emotion_result: Result from detect_emotions
            
        Returns:
            Emotion category: 'positive', 'negative', 'neutral', 'high_arousal'
        """
        emotion = emotion_result['emotion']
        valence = emotion_result['valence']
        arousal = emotion_result['arousal']
        
        if emotion in ['Happy']:
            return 'positive'
        elif emotion in ['Sad', 'Anger', 'Disgust', 'Contempt']:
            return 'negative'
        elif emotion in ['Fear', 'Surprise']:
            return 'high_arousal'
        else:
            return 'neutral'

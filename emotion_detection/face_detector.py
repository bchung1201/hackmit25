"""
Face detection module using OpenCV and MTCNN
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """Face detection using OpenCV Haar Cascades and MTCNN"""
    
    def __init__(self, method: str = "opencv"):
        self.method = method
        self.face_cascade = None
        self.mtcnn = None
        
        if method == "opencv":
            self._init_opencv()
        elif method == "mtcnn":
            self._init_mtcnn()
        else:
            raise ValueError(f"Unknown face detection method: {method}")
    
    def _init_opencv(self):
        """Initialize OpenCV face detector"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("OpenCV face detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV face detector: {e}")
            raise
    
    def _init_mtcnn(self):
        """Initialize MTCNN face detector"""
        try:
            from mtcnn import MTCNN
            self.mtcnn = MTCNN()
            logger.info("MTCNN face detector initialized")
        except ImportError:
            logger.warning("MTCNN not available, falling back to OpenCV")
            self._init_opencv()
            self.method = "opencv"
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            self._init_opencv()
            self.method = "opencv"
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self.method == "opencv":
            return self._detect_faces_opencv(image)
        elif self.method == "mtcnn":
            return self._detect_faces_mtcnn(image)
        else:
            return []
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN"""
        if self.mtcnn is None:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mtcnn.detect_faces(rgb_image)
        
        faces = []
        for result in results:
            if result['confidence'] > 0.9:  # High confidence threshold
                x, y, w, h = result['box']
                faces.append((int(x), int(y), int(w), int(h)))
        
        return faces
    
    def extract_face_region(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from image
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_box
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
    
    def get_face_landmarks(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks for a detected face
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Facial landmarks array or None
        """
        if self.method == "mtcnn" and self.mtcnn is not None:
            # MTCNN provides landmarks
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mtcnn.detect_faces(rgb_image)
            
            for result in results:
                if result['confidence'] > 0.9:
                    landmarks = result['keypoints']
                    return np.array([
                        [landmarks['left_eye'][0], landmarks['left_eye'][1]],
                        [landmarks['right_eye'][0], landmarks['right_eye'][1]],
                        [landmarks['nose'][0], landmarks['nose'][1]],
                        [landmarks['mouth_left'][0], landmarks['mouth_left'][1]],
                        [landmarks['mouth_right'][0], landmarks['mouth_right'][1]]
                    ])
        
        return None

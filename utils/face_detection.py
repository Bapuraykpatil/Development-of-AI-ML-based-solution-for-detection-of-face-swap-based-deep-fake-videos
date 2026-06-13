"""
Face detection utilities
Detect and extract faces from images
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces in images using multiple detection methods"""

    def __init__(self, detector_type: str = "cascade"):
        """
        Initialize face detector

        Args:
            detector_type (str): Detection method ('cascade', 'dnn', 'mtcnn')
        """
        self.detector_type = detector_type
        self.cascade_classifier = None
        self._load_detector()

    def _load_detector(self):
        """Load the face detector"""
        if self.detector_type == "cascade":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
            logger.info("Cascade classifier loaded")
        elif self.detector_type == "dnn":
            logger.info("DNN detector selected")
        elif self.detector_type == "mtcnn":
            logger.info("MTCNN detector selected (requires additional installation)")
        else:
            logger.warning(f"Unknown detector type: {self.detector_type}, using cascade")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade_classifier = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1,
                    min_neighbors: int = 5) -> List[Tuple]:
        """
        Detect faces in an image

        Args:
            image (np.ndarray): Input image
            scale_factor (float): Scale factor for cascade classifier
            min_neighbors (int): Minimum neighbors for cascade classifier

        Returns:
            List[Tuple]: List of face bounding boxes (x, y, w, h)
        """
        try:
            if self.detector_type == "cascade":
                return self._detect_cascade(image, scale_factor, min_neighbors)
            elif self.detector_type == "dnn":
                return self._detect_dnn(image)
            elif self.detector_type == "mtcnn":
                return self._detect_mtcnn(image)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def _detect_cascade(self, image: np.ndarray, scale_factor: float,
                       min_neighbors: int) -> List[Tuple]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade_classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        # Convert to bounding box format
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append((x, y, x + w, y + h))

        return bboxes

    def _detect_dnn(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces using DNN (requires model files)"""
        logger.warning("DNN detection not fully implemented")
        return []

    def _detect_mtcnn(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces using MTCNN (requires mtcnn package)"""
        try:
            from mtcnn import MTCNN
            detector = MTCNN()
            detections = detector.detect_faces(image)
            
            bboxes = []
            for detection in detections:
                bbox = detection['box']
                x1, y1, w, h = bbox
                bboxes.append((x1, y1, x1 + w, y1 + h))
            
            return bboxes
        except ImportError:
            logger.error("MTCNN not installed. Install with: pip install mtcnn")
            return []

    def extract_faces(self, image: np.ndarray, bboxes: List[Tuple],
                     size: Optional[Tuple] = None) -> List[np.ndarray]:
        """
        Extract face regions from image

        Args:
            image (np.ndarray): Input image
            bboxes (List[Tuple]): List of bounding boxes
            size (Tuple): Resize faces to this size

        Returns:
            List[np.ndarray]: List of extracted face images
        """
        faces = []

        for (x1, y1, x2, y2) in bboxes:
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # Extract face
            face = image[y1:y2, x1:x2]

            # Resize if specified
            if size:
                face = cv2.resize(face, size)

            faces.append(face)

        return faces

    def draw_faces(self, image: np.ndarray, bboxes: List[Tuple],
                  color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image (np.ndarray): Input image
            bboxes (List[Tuple]): List of bounding boxes
            color (Tuple): Box color in BGR
            thickness (int): Line thickness

        Returns:
            np.ndarray: Image with drawn boxes
        """
        result = image.copy()

        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        return result
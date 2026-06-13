"""
Main Deepfake Detection Module
Provides the core DeepfakeDetector class for detecting face-swap based deepfakes
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")

import config
from utils.video_processing import VideoProcessor
from utils.face_detection import FaceDetector
from utils.visualization import create_visualization_report

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Main class for detecting deepfake videos using deep learning
    Supports video analysis, frame-level detection, and batch processing
    """

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the DeepfakeDetector

        Args:
            model_path (str): Path to pre-trained model. If None, uses default.
            use_gpu (bool): Whether to use GPU for inference
        """
        self.model_path = model_path or str(config.DETECTOR_MODEL_PATH)
        self.use_gpu = use_gpu
        self.model = None
        self.video_processor = VideoProcessor()
        self.face_detector = FaceDetector(detector_type=config.FACE_DETECTOR_TYPE)

        # GPU Configuration
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self._setup_gpu()
        self._load_model()

        logger.info("DeepfakeDetector initialized successfully")

    def _setup_gpu(self):
        """Configure GPU settings"""
        if self.use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU support enabled: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    logger.warning(f"Could not set GPU memory growth: {e}")
            else:
                logger.warning("GPU requested but no GPUs found. Using CPU.")
        else:
            logger.info("Using CPU for inference")

    def _load_model(self):
        """Load the pre-trained deepfake detection model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}")
                logger.info("Using a dummy model for testing")
                self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Create a dummy model for testing (replace with real model in production)"""
        logger.info("Creating dummy model for testing")
        # This is a placeholder. Replace with actual model architecture
        inputs = keras.Input(shape=(config.INPUT_SIZE[0], config.INPUT_SIZE[1], 3))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def detect_video(self, video_path: str, return_confidence: bool = True) -> Dict:
        """
        Detect if a video is a deepfake

        Args:
            video_path (str): Path to video file
            return_confidence (bool): Return confidence scores

        Returns:
            Dict: Detection results with prediction and confidence
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        try:
            # Extract frames from video
            frames = self.video_processor.extract_frames(
                video_path,
                sample_rate=config.FRAME_SAMPLE_RATE
            )

            if len(frames) == 0:
                logger.error("No frames extracted from video")
                return {
                    'filename': os.path.basename(video_path),
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'frames_analyzed': 0,
                    'error': 'No frames extracted'
                }

            # Detect faces and extract face regions
            face_frames = []
            for frame in frames:
                faces = self.face_detector.detect_faces(frame)
                if faces:
                    for face in faces:
                        face_region = self._extract_face_region(frame, face)
                        face_frames.append(face_region)

            if len(face_frames) == 0:
                logger.warning("No faces detected in video")
                return {
                    'filename': os.path.basename(video_path),
                    'prediction': 'NO_FACES_DETECTED',
                    'confidence': 0.0,
                    'frames_analyzed': len(frames),
                    'faces_detected': 0
                }

            # Predict on extracted faces
            predictions = self._predict_batch(face_frames)

            # Calculate statistics
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            max_prediction = np.max(predictions)
            min_prediction = np.min(predictions)

            # Determine if video is fake based on threshold
            is_fake = mean_prediction > config.PREDICTION_THRESHOLD

            result = {
                'filename': os.path.basename(video_path),
                'file_path': video_path,
                'prediction': 'FAKE' if is_fake else 'REAL',
                'confidence': float(mean_prediction),
                'confidence_range': {
                    'min': float(min_prediction),
                    'max': float(max_prediction),
                    'std': float(std_prediction)
                },
                'frames_analyzed': len(frames),
                'faces_detected': len(face_frames),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Detection complete: {result['prediction']} (confidence: {result['confidence']:.2%})")
            return result

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'filename': os.path.basename(video_path),
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }

    def analyze_frames(self, video_path: str, sample_rate: int = 5) -> Dict:
        """
        Analyze individual frames of a video

        Args:
            video_path (str): Path to video file
            sample_rate (int): Process every Nth frame

        Returns:
            Dict: Frame-level analysis results
        """
        logger.info(f"Analyzing frames from: {video_path}")

        frames = self.video_processor.extract_frames(video_path, sample_rate=sample_rate)
        results = {}

        for frame_idx, frame in enumerate(tqdm(frames, desc="Analyzing frames")):
            faces = self.face_detector.detect_faces(frame)

            if faces:
                for face_idx, face in enumerate(faces):
                    face_region = self._extract_face_region(frame, face)
                    prediction = self._predict_single(face_region)

                    frame_key = f"frame_{frame_idx:05d}_face_{face_idx}"
                    results[frame_key] = {
                        'frame_number': frame_idx,
                        'face_number': face_idx,
                        'prediction': 'FAKE' if prediction > config.PREDICTION_THRESHOLD else 'REAL',
                        'confidence': float(prediction),
                        'bbox': face.tolist() if isinstance(face, np.ndarray) else face
                    }

        return results

    def batch_process(self, video_directory: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Process multiple videos from a directory

        Args:
            video_directory (str): Directory containing video files
            output_file (str): Optional file to save results

        Returns:
            List[Dict]: Results for all processed videos
        """
        logger.info(f"Batch processing videos from: {video_directory}")

        video_files = []
        for ext in config.VIDEO_FORMATS:
            video_files.extend(Path(video_directory).glob(f"*{ext}"))

        logger.info(f"Found {len(video_files)} video files")

        results = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            result = self.detect_video(str(video_path))
            results.append(result)

        # Save results if specified
        if output_file:
            self._save_results(results, output_file)

        return results

    def _extract_face_region(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract face region from frame and resize to model input size

        Args:
            frame (np.ndarray): Input frame
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)

        Returns:
            np.ndarray: Extracted and resized face region
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_region = frame[y1:y2, x1:x2]
        face_region = cv2.resize(face_region, config.INPUT_SIZE)
        return face_region

    def _predict_single(self, image: np.ndarray) -> float:
        """
        Predict deepfake probability for a single image

        Args:
            image (np.ndarray): Input image

        Returns:
            float: Prediction probability (0-1)
        """
        # Normalize image
        image = image.astype('float32') / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = self.model.predict(image, verbose=0)[0][0]
        return float(prediction)

    def _predict_batch(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Predict deepfake probability for multiple images

        Args:
            images (List[np.ndarray]): List of input images
            batch_size (int): Batch size for prediction

        Returns:
            np.ndarray: Array of predictions
        """
        images = np.array(images).astype('float32') / 255.0
        predictions = self.model.predict(images, batch_size=batch_size, verbose=0)
        return predictions.flatten()

    def _save_results(self, results: List[Dict], output_file: str):
        """
        Save results to file

        Args:
            results (List[Dict]): Results to save
            output_file (str): Output file path
        """
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_file.endswith('.csv'):
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


class BatchProcessor:
    """Handle batch processing of multiple videos"""

    def __init__(self, detector: Optional[DeepfakeDetector] = None):
        """
        Initialize batch processor

        Args:
            detector (DeepfakeDetector): Detector instance. Creates new if None.
        """
        self.detector = detector or DeepfakeDetector()

    def process_directory(self, directory: str) -> List[Dict]:
        """Process all videos in a directory"""
        return self.detector.batch_process(directory)

    def generate_report(self, results: List[Dict], output_file: str):
        """Generate a report from results"""
        self.detector._save_results(results, output_file)


if __name__ == "__main__":
    # Example usage
    print("Deepfake Detection Module")
    print("=" * 50)

    # Initialize detector
    detector = DeepfakeDetector()

    # Example: Process a single video (uncomment to use)
    # result = detector.detect_video('path/to/video.mp4')
    # print(result)

    # Example: Batch processing
    # results = detector.batch_process('path/to/videos/')
    # print(f"Processed {len(results)} videos")

    print("\nModule loaded successfully!")
"""
Configuration file for Deepfake Detection Project
Centralized settings for model, data, and processing parameters
"""

import os
from pathlib import Path

# ===========================
# Project Paths
# ===========================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
UTILS_DIR = PROJECT_ROOT / "utils"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================
# Model Configuration
# ===========================
# Model Architecture
MODEL_NAME = "EfficientNetB4"  # Options: 'ResNet50', 'EfficientNetB4', 'InceptionV3'
MODEL_WEIGHTS = "imagenet"
INPUT_SIZE = (224, 224)
NUM_CLASSES = 2  # Binary classification: Real (0) or Fake (1)

# Pre-trained Model Paths
DETECTOR_MODEL_PATH = MODELS_DIR / "detector_model.h5"
FACE_DETECTOR_PATH = MODELS_DIR / "face_detector"
PRETRAINED_WEIGHTS = MODELS_DIR / "pretrained_weights"

# ===========================
# Face Detection Configuration
# ===========================
FACE_DETECTOR_TYPE = "mtcnn"  # Options: 'mtcnn', 'retinaface', 'mediapipe'
FACE_CONFIDENCE_THRESHOLD = 0.9
MIN_FACE_SIZE = (20, 20)
MAX_FACE_SIZE = (500, 500)

# ===========================
# Video Processing
# ===========================
FRAME_SAMPLE_RATE = 5  # Process every Nth frame (e.g., 5 = every 5th frame)
FPS = 30
VIDEO_DURATION_MAX = 300  # Maximum video duration in seconds
VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# ===========================
# Training Configuration
# ===========================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Optimizer
OPTIMIZER = "adam"  # Options: 'adam', 'sgd', 'rmsprop'
LOSS_FUNCTION = "binary_crossentropy"

# ===========================
# Data Augmentation
# ===========================
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.2,
    "brightness_range": 0.2,
    "contrast_range": 0.2,
}

# ===========================
# Prediction Configuration
# ===========================
PREDICTION_THRESHOLD = 0.5  # Confidence threshold for classification
ENSEMBLE_MODELS = False  # Use multiple models for prediction
CONFIDENCE_CALIBRATION = True

# ===========================
# Output & Reporting
# ===========================
SAVE_PREDICTIONS = True
SAVE_VISUALIZATIONS = True
GENERATE_REPORT = True
REPORT_FORMAT = "csv"  # Options: 'csv', 'json', 'xlsx'

# Detection Results
DETECTION_LABELS = {
    0: "REAL",
    1: "FAKE"
}

# ===========================
# Logging Configuration
# ===========================
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FILE = PROJECT_ROOT / "logs" / "deepfake_detection.log"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ===========================
# Dataset Configuration
# ===========================
# Dataset paths
TRAIN_DATA_PATH = DATA_DIR / "train"
TEST_DATA_PATH = DATA_DIR / "test"
VALIDATION_DATA_PATH = DATA_DIR / "validation"
SAMPLE_VIDEOS_PATH = DATA_DIR / "sample_videos"

# Dataset statistics
TRAIN_SAMPLES = None  # Set after data preparation
TEST_SAMPLES = None
VALIDATION_SAMPLES = None

# ===========================
# GPU Configuration
# ===========================
USE_GPU = True
GPU_ID = 0  # GPU device ID to use
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use

# ===========================
# Advanced Options
# ===========================
# Ensemble Settings
ENSEMBLE_METHOD = "voting"  # Options: 'voting', 'stacking', 'averaging'
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.3]  # Weights for ensemble models

# Post-processing
USE_TEMPORAL_CONSISTENCY = True
TEMPORAL_SMOOTHING_WINDOW = 5  # Frames for smoothing

# Feature Extraction
EXTRACT_FEATURES = True
FEATURE_EXTRACTION_LAYER = "conv5_block3_out"  # Layer for feature extraction

# ===========================
# Database Configuration (if using)
# ===========================
USE_DATABASE = False
DATABASE_URL = "sqlite:///deepfake_detection.db"

# ===========================
# API Configuration (if deploying)
# ===========================
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
MAX_FILE_SIZE_MB = 500

# ===========================
# Thresholds & Parameters
# ===========================
# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.9
MEDIUM_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.5

# Detection parameters
MIN_DETECTED_FACES = 1
MIN_FRAMES_FOR_DETECTION = 10

# ===========================
# Utility Functions
# ===========================
def get_config_dict():
    """Return all configuration as a dictionary"""
    return {
        k: v for k, v in globals().items()
        if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))
    }

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("DEEPFAKE DETECTION PROJECT CONFIGURATION")
    print("=" * 60)
    config = get_config_dict()
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
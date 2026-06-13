# AI/ML-Based Solution for Detection of Face-Swap Based Deepfake Videos

A comprehensive machine learning solution for detecting face-swap based deepfake videos using advanced computer vision and deep learning techniques.

## 📋 Project Overview

This project aims to detect manipulated videos where faces have been swapped using deep learning models. It combines multiple detection techniques including:
- Face detection and extraction
- Facial feature analysis
- CNN-based deepfake classification
- Video frame analysis
- Temporal consistency checking

## 🎯 Features

- **Video Analysis**: Process video files frame by frame
- **Face Detection**: Automatic face detection using MTCNN or RetinaFace
- **Deepfake Classification**: Binary classification (real/fake) using trained CNN models
- **Frame-Level Detection**: Analyze individual frames for manipulation artifacts
- **Batch Processing**: Process multiple videos efficiently
- **Result Visualization**: Generate detailed reports and visualizations
- **Model Training**: Scripts to train custom detection models

## 🛠️ Technology Stack

- **Python 3.8+**
- **TensorFlow / PyTorch** - Deep learning framework
- **OpenCV** - Video processing and face detection
- **NumPy & Pandas** - Data processing
- **scikit-learn** - ML utilities
- **Matplotlib & Seaborn** - Visualization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- GPU support recommended (CUDA, cuDNN)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Bapuraykpatil/Development-of-AI-ML-based-solution-for-detection-of-face-swap-based-deep-fake-videos.git
cd Development-of-AI-ML-based-solution-for-detection-of-face-swap-based-deep-fake-videos
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Detect Deepfakes in a Single Video

```python
from deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(model_path='models/detector_model.h5')

# Detect deepfakes
result = detector.detect_video('path/to/video.mp4')
print(result)
```

### 2. Analyze a Video with Detailed Report

```python
detector = DeepfakeDetector()
report = detector.analyze_video('video.mp4', save_frames=True, visualize=True)

print(f"Video: {report['filename']}")
print(f"Prediction: {report['prediction']}")  # real/fake
print(f"Confidence: {report['confidence']:.2%}")
print(f"Frames Analyzed: {report['frames_analyzed']}")
```

### 3. Batch Processing

```python
from deepfake_detector import BatchProcessor

processor = BatchProcessor()
results = processor.process_directory('path/to/videos/')
processor.generate_report(results, 'report.csv')
```

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py
├── deepfake_detector.py
├── train_model.py
├── detect.py
├── data/
│   ├── train/
│   ├── test/
│   └── sample_videos/
├── models/
│   ├── detector_model.h5
│   └── face_detector/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
├── results/
│   ├── detections/
│   └── reports/
└── utils/
    ├── video_processing.py
    ├── face_detection.py
    └── visualization.py
```

## 🏋️ Model Training

### Train a Custom Model

```bash
python train_model.py --data_path data/train --epochs 50 --batch_size 32
```

### Training Parameters
- **Epochs**: Number of training iterations (default: 50)
- **Batch Size**: Samples per batch (default: 32)
- **Learning Rate**: Optimizer learning rate (default: 0.001)
- **Validation Split**: Train/validation split ratio (default: 0.2)

## 📊 Model Architecture

The detector uses a CNN architecture optimized for deepfake detection:
- **Input**: Video frames (224x224 RGB)
- **Backbone**: EfficientNet or ResNet50 with pretrained ImageNet weights
- **Head**: Dense layers for binary classification
- **Output**: Probability of deepfake (0-1 range)

## 📈 Results & Evaluation

Model performance metrics:
- **Accuracy**: Classification accuracy on test set
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

See `results/` directory for detailed evaluation reports and visualizations.

## 🔍 Testing

Run unit tests:
```bash
pytest tests/ -v
```

## 📝 Usage Examples

### Example 1: Simple Detection
```python
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()
is_fake, confidence = detector.detect_video('sample_video.mp4')
print(f"Deepfake: {is_fake}, Confidence: {confidence:.2%}")
```

### Example 2: Frame-Level Analysis
```python
frames_results = detector.analyze_frames('video.mp4', sample_rate=5)
for frame_num, prediction in frames_results.items():
    print(f"Frame {frame_num}: {prediction['label']} ({prediction['confidence']:.2%})")
```

## 🎓 Dataset Information

The model is trained on:
- **FaceForensics++**: Large-scale deepfake detection dataset
- **DFDC Dataset**: DeepFake Detection Challenge dataset
- **Celeb-DF**: Celebrity deepfake dataset
- Custom annotated videos

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚠️ Ethical Considerations

This tool is designed for:
- ✅ Research and security purposes
- ✅ Educational demonstrations
- ✅ Content verification
- ✅ Combating disinformation

**Not intended for**:
- ❌ Creating or distributing deepfakes
- ❌ Unauthorized surveillance
- ❌ Malicious purposes

## 🐛 Issues & Support

For bugs, feature requests, or questions:
1. Check existing [Issues](../../issues)
2. Create a new [Issue](../../issues/new) with details
3. Contact: [Your Email/Contact]

## 📚 References

- FaceForensics++: Learning to Detect Manipulated Facial Images
- The Eyes Tell All: Detecting Political Orientation from Eye Movement Data
- MediaPipe: Framework for Building Multimodal ML Pipelines
- OpenCV: Computer Vision Library

## 👨‍💻 Author

**Bapuraykpatil**  
GitHub: [@Bapuraykpatil](https://github.com/Bapuraykpatil)

---

**Last Updated**: June 2026  
**Status**: Active Development 🚀

"""
Visualization utilities
Create reports and visualizations for detection results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def create_visualization_report(results: Dict, output_path: str = None):
    """
    Create a visualization report from detection results

    Args:
        results (Dict): Detection results
        output_path (str): Path to save visualization
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Deepfake Detection Report', fontsize=16)

        # Plot 1: Confidence distribution
        if 'confidence_data' in results:
            axes[0, 0].hist(results['confidence_data'], bins=20)
            axes[0, 0].set_title('Confidence Distribution')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].set_ylabel('Frequency')

        # Plot 2: Results summary
        axes[0, 1].axis('off')
        summary_text = f"""
        Total Videos: {results.get('total_videos', 0)}
        Fake Videos: {results.get('fake_count', 0)}
        Real Videos: {results.get('real_count', 0)}
        Accuracy: {results.get('accuracy', 0):.2%}
        """
        axes[0, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')

        # Plot 3: Prediction pie chart
        if 'prediction_counts' in results:
            labels = ['Real', 'Fake']
            sizes = results['prediction_counts']
            axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[1, 0].set_title('Prediction Distribution')

        # Plot 4: Confidence over time
        if 'confidence_timeline' in results:
            axes[1, 1].plot(results['confidence_timeline'])
            axes[1, 1].set_title('Confidence Over Time')
            axes[1, 1].set_xlabel('Sample')
            axes[1, 1].set_ylabel('Confidence')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


def draw_detection_results(image: np.ndarray, detections: List[Dict],
                          confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Draw detection results on image

    Args:
        image (np.ndarray): Input image
        detections (List[Dict]): List of detections with bboxes and labels
        confidence_threshold (float): Threshold for visualization

    Returns:
        np.ndarray: Image with drawn results
    """
    result = image.copy()

    for detection in detections:
        if detection.get('confidence', 0) > confidence_threshold:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']

            # Draw bounding box
            color = (0, 0, 255) if label == 'FAKE' else (0, 255, 0)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label
            text = f"{label}: {confidence:.2%}"
            cv2.putText(result, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result


def create_comparison_image(original: np.ndarray, marked: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison image

    Args:
        original (np.ndarray): Original image
        marked (np.ndarray): Marked/processed image

    Returns:
        np.ndarray: Combined comparison image
    """
    # Ensure same size
    h1, w1 = original.shape[:2]
    h2, w2 = marked.shape[:2]

    if h1 != h2 or w1 != w2:
        marked = cv2.resize(marked, (w1, h1))

    # Concatenate horizontally
    comparison = np.hstack([original, marked])

    return comparison


def generate_heatmap(predictions: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """
    Generate a heatmap from prediction scores

    Args:
        predictions (np.ndarray): Prediction scores
        frame_shape (tuple): Shape of the frame

    Returns:
        np.ndarray: Heatmap visualization
    """
    # Normalize predictions to 0-255
    heatmap = ((predictions - predictions.min()) /
              (predictions.max() - predictions.min() + 1e-6) * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize to frame shape
    heatmap = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]))

    return heatmap
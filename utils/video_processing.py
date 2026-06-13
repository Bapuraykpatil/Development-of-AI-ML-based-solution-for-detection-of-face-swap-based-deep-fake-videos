"""
Video processing utilities
Handle video reading, frame extraction, and preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video file processing and frame extraction"""

    def __init__(self, max_duration: int = 300):
        """
        Initialize video processor

        Args:
            max_duration (int): Maximum video duration in seconds
        """
        self.max_duration = max_duration

    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 5,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file

        Args:
            video_path (str): Path to video file
            sample_rate (int): Extract every Nth frame
            max_frames (int): Maximum number of frames to extract

        Returns:
            List[np.ndarray]: List of extracted frames
        """
        frames = []

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return frames

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")

            # Extract frames
            frame_count = 0
            extracted_count = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample frames based on sample_rate
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                    extracted_count += 1

                    # Check max_frames limit
                    if max_frames and extracted_count >= max_frames:
                        break

                frame_count += 1

            cap.release()
            logger.info(f"Extracted {extracted_count} frames from {video_path}")

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")

        return frames

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video file information

        Args:
            video_path (str): Path to video file

        Returns:
            dict: Video information
        """
        info = {}

        try:
            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
                cap.release()

        except Exception as e:
            logger.error(f"Error getting video info: {e}")

        return info

    def resize_frame(self, frame: np.ndarray, size: tuple) -> np.ndarray:
        """
        Resize a frame to specified size

        Args:
            frame (np.ndarray): Input frame
            size (tuple): Target size (width, height)

        Returns:
            np.ndarray: Resized frame
        """
        return cv2.resize(frame, size)

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to 0-1 range

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Normalized frame
        """
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str):
        """
        Save a frame to file

        Args:
            frame (np.ndarray): Frame to save
            output_path (str): Output file path
        """
        cv2.imwrite(output_path, frame)
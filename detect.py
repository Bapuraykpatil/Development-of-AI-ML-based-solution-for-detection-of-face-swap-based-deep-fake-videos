"""
Main script for running deepfake detection on videos
Provides command-line interface for video analysis
"""

import argparse
import sys
import logging
from pathlib import Path
from deepfake_detector import DeepfakeDetector, BatchProcessor
import config
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_single_video(video_path: str, verbose: bool = False) -> dict:
    """
    Detect deepfakes in a single video

    Args:
        video_path (str): Path to video file
        verbose (bool): Print detailed output

    Returns:
        dict: Detection results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_path}")
        print(f"{'=' * 60}")

    # Initialize detector
    detector = DeepfakeDetector()

    # Run detection
    result = detector.detect_video(video_path)

    if verbose:
        print(f"\nResults:")
        print(f"  Prediction: {result.get('prediction', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2%}")
        print(f"  Frames Analyzed: {result.get('frames_analyzed', 0)}")
        print(f"  Faces Detected: {result.get('faces_detected', 0)}")

        if 'confidence_range' in result:
            conf_range = result['confidence_range']
            print(f"  Confidence Range:")
            print(f"    Min: {conf_range['min']:.2%}")
            print(f"    Max: {conf_range['max']:.2%}")
            print(f"    Std: {conf_range['std']:.4f}")

    return result


def detect_batch_videos(video_directory: str, output_file: str = None, verbose: bool = False) -> list:
    """
    Detect deepfakes in multiple videos

    Args:
        video_directory (str): Directory containing videos
        output_file (str): Optional output file for results
        verbose (bool): Print detailed output

    Returns:
        list: Detection results for all videos
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Batch Processing: {video_directory}")
        print(f"{'=' * 60}")

    # Initialize processor
    processor = BatchProcessor()

    # Process videos
    results = processor.process_directory(video_directory)

    if verbose:
        print(f"\nProcessed {len(results)} videos")
        for result in results:
            print(f"\n  File: {result['filename']}")
            print(f"    Prediction: {result['prediction']}")
            print(f"    Confidence: {result['confidence']:.2%}")

    # Save results if specified
    if output_file:
        processor.generate_report(results, output_file)
        if verbose:
            print(f"\nResults saved to: {output_file}")

    return results


def analyze_video_frames(video_path: str, sample_rate: int = 5, output_file: str = None) -> dict:
    """
    Analyze individual frames of a video

    Args:
        video_path (str): Path to video file
        sample_rate (int): Process every Nth frame
        output_file (str): Optional output file for results

    Returns:
        dict: Frame-level analysis results
    """
    print(f"\n{'=' * 60}")
    print(f"Frame Analysis: {video_path}")
    print(f"{'=' * 60}")

    # Initialize detector
    detector = DeepfakeDetector()

    # Analyze frames
    results = detector.analyze_frames(video_path, sample_rate=sample_rate)

    print(f"\nAnalyzed {len(results)} face regions")

    # Calculate statistics
    predictions = [r['confidence'] for r in results.values()]
    if predictions:
        fake_count = sum(1 for p in predictions if p > config.PREDICTION_THRESHOLD)
        real_count = len(predictions) - fake_count

        print(f"  Fake faces: {fake_count} ({fake_count/len(predictions)*100:.1f}%)")
        print(f"  Real faces: {real_count} ({real_count/len(predictions)*100:.1f}%)")

    # Save results if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Tool - Detect face-swap based deepfake videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect a single video
  python detect.py --video sample.mp4

  # Batch process videos in a directory
  python detect.py --batch /path/to/videos/ --output results.json

  # Analyze individual frames
  python detect.py --video sample.mp4 --frames --frame-rate 5

  # Verbose output
  python detect.py --video sample.mp4 --verbose
        """
    )

    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--batch', type=str, help='Directory containing multiple videos')
    parser.add_argument('--output', type=str, help='Output file for results (JSON or CSV)')
    parser.add_argument('--frames', action='store_true', help='Analyze individual frames')
    parser.add_argument('--frame-rate', type=int, default=5, help='Frame sampling rate (default: 5)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--config', action='store_true', help='Show configuration')

    args = parser.parse_args()

    # Show configuration
    if args.config:
        config.print_config()
        return

    # Validate arguments
    if not args.video and not args.batch:
        parser.print_help()
        sys.exit(1)

    # Generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"detection_results_{timestamp}.json"

    try:
        # Single video detection
        if args.video:
            if not Path(args.video).exists():
                print(f"Error: Video file not found: {args.video}")
                sys.exit(1)

            if args.frames:
                # Frame analysis
                analyze_video_frames(
                    args.video,
                    sample_rate=args.frame_rate,
                    output_file=args.output if args.output else None
                )
            else:
                # Single video detection
                result = detect_single_video(args.video, verbose=args.verbose)

                # Save result
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                    if args.verbose:
                        print(f"\nResults saved to: {args.output}")

        # Batch processing
        elif args.batch:
            if not Path(args.batch).is_dir():
                print(f"Error: Directory not found: {args.batch}")
                sys.exit(1)

            detect_batch_videos(
                args.batch,
                output_file=args.output,
                verbose=args.verbose
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
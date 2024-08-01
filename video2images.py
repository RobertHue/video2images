# Python Module Index
import argparse
import logging
import shutil
import skimage # scikit-image for loaded image analysis

from pathlib import Path

# 3rd-Party
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline_tools import estimate_geometric_overlap


# Constants
GOOD_MATCH_DISTANCE_THRESHOLD = 200  # Threshold for determining a good match


# Basic configuration for the root logger
logging.basicConfig(
    level=logging.INFO,  # Set default level for root logger
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logging.log', mode='w'),  # File handler
        logging.StreamHandler()  # Console handler
    ]
)


def extract_frames(video_path):
    """
    Extract frames from a video file and perform quality checks on them.

    :param video_path: Path to the video file.
    """
    path_obj = Path(video_path)
    directory_path = path_obj.parent
    video_file_name = path_obj.stem
    output_directory = directory_path / Path(f"{video_file_name}_frames")
    clear_output_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return

    # Get the total number of frames and fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"FPS of video: {fps}")
    logging.info(f"Frame count of video: {frame_count}")

    count = 0
    prev_frame = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            if not is_image_sharp(frame):
                logging.warning(f"Frame {count} is not sharp; hence too blurry. Skipping...")
                continue

            if prev_frame is not None and estimate_geometric_overlap(prev_frame, frame) >= 0.95:
                logging.warning(f"Frame {count} does not overlap with previous frame enough. Skipping...")
                continue

            # Save frame as image
            output_file = output_directory / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            logging.info(f"Frame {count} has been extracted and saved as {output_file.name}")

            prev_frame = frame

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Total frames extracted: {count}")

################################################################################


def main():
    """
    Main function to parse arguments and call the frame extraction function.
    """
    parser = argparse.ArgumentParser(description="Split video into images")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    extract_frames(args.video_path)


if __name__ == "__main__":
    main()

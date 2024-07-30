# Python module index
import argparse
import logging
import os
import shutil

from pathlib import Path

# 3rd party
import cv2
import numpy as np

# Constants
GOOD_MATCH_DISTANCE_THRESHOLD = 30  # Threshold for determining a good match

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Checks if the image is in focus
def is_image_sharp(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold


# Checks for proper exposure
def is_image_well_exposed(image, low_threshold=0.1, high_threshold=0.9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)
    return (cdf[0] > low_threshold) and (cdf[-1] < high_threshold)

def calculate_overlap(image1, image2, overlap_fraction=0.6):
    """
    Calculate the overlap percentage between two images using ORB feature matching.
    
    :param image1: First input image in BGR format.
    :param image2: Second input image in BGR format.
    :param overlap_fraction: Fraction of overlap required between images.
    :return: True if the overlap is greater than or equal to the specified fraction.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < GOOD_MATCH_DISTANCE_THRESHOLD]
    if len(kp1) == 0:
        return False
    overlap_percentage = len(good_matches) / len(kp1)
    return overlap_percentage >= overlap_fraction

def clear_output_directory(directory_path):
    if directory_path.exists():
        logging.info(f"Clearing output directory: {directory_path}")
        # Remove all files and subdirectories in the directory
        for item in directory_path.iterdir():
            if item.is_file():
                item.unlink()  # Remove file
            elif item.is_dir():
                shutil.rmtree(item)  # Remove directory
    else:
        logging.info(f"Output directory does not exist. Creating new one: {directory_path}")

def extract_frames(video_path, overlap_fraction):
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

    # Initialize frame count
    count = 0
    prev_frame = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Quality checks
            if not is_image_sharp(frame, threshold=100.0):
                logging.debug(f"Frame {count} is not sharp; hence too blurry. Skipping...")
                continue

            if not is_image_well_exposed(frame):
                logging.debug(f"Frame {count} is not well exposed; hence too dark or too bright. Skipping...")
                continue

            if prev_frame is not None:
                if calculate_overlap(prev_frame, frame, overlap_fraction):
                    logging.info(f"Frame {count} overlaps with previous frame by at least {overlap_fraction*100}%. Saving...")
                else:
                    logging.warning(f"Frame {count} does not overlap with previous frame by at least {overlap_fraction*100}%. Skipping...")
                    continue

            # Save frame as image
            output_file = output_directory / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            logging.info(f"Frame {count} has been extracted and saved as {output_file.name}")
            
            # Update
            prev_frame = frame 
            count += 1

    finally:
        # Release video capture object
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Total frames extracted: {count}")


# Main function to parse arguments and call the split function
def main():
    parser = argparse.ArgumentParser(description="Split video into images")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--overlap_fraction", type=float, default=0.6, help="Minimum overlap fraction between frames")
    args = parser.parse_args()

    extract_frames(args.video_path, args.overlap_fraction)

if __name__ == "__main__":
    main()
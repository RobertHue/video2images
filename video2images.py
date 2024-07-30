# Python module index
import argparse
import logging
import os
import shutil

from pathlib import Path

# 3rd party
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim


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
    
    
# Checks for geometric consistency using feature matching
def has_good_feature_match(image1, image2, min_matches=10):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # logging.info(f"len(matches): {len(matches)}")
    return len(matches) > min_matches


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


# Extract frames from a video until reaching the desired frame count
def extract_frames(video_path):

    # Create an output folder with a name corresponding to the video
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
            if prev_frame is not None and has_good_feature_match(prev_frame, frame, min_matches=200):
                logging.warning(f"Frame {count} does have good feature matches and therefore is too similar. Skipping...")
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
    args = parser.parse_args()

    extract_frames(args.video_path)

if __name__ == "__main__":
    main()
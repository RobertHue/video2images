# Python module index
import argparse
import logging
import shutil

from pathlib import Path

# 3rd party
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Constants
GOOD_MATCH_DISTANCE_THRESHOLD = 200  # Threshold for determining a good match


# Basic configuration for the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set default level for root logger
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',         # Use 'w' to overwrite the log file each time
    handlers=[
        logging.FileHandler('logging.log', mode='w'),  # File handler
        logging.StreamHandler()  # Console handler
    ]
)


def is_image_sharp(image, threshold=50.0):
    """
    Check if the image is in focus based on the variance of the Laplacian.

    :param image: Input image in BGR format.
    :param threshold: Threshold for the variance of the Laplacian.
    :return: True if the image is sharp, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold


def is_image_well_exposed(image, low_threshold=0.1, high_threshold=0.9):
    """
    Check if the image is well-exposed based on its histogram.

    :param image: Input image in BGR format.
    :param low_threshold: Lower bound for cumulative distribution function.
    :param high_threshold: Upper bound for cumulative distribution function.
    :return: True if the image is well-exposed, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)
    return (cdf[0] > low_threshold) and (cdf[-1] < high_threshold)


def debug_matches(gray1, gray2, kp1, kp2, matches, good_matches):
    logging.info(f"{len(good_matches)=}")
    logging.info(f"{len(kp1)=}")
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img_with_keypoints = cv2.drawKeypoints(gray1, kp1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # Convert BGR image to RGB (matplotlib uses RGB)
    img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)

    img4 = cv2.drawMatches(gray1,kp1,gray2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.title("Are frames overlapping by at least 60%")

    # Display the images
    ax1.imshow(img_with_keypoints_rgb)
    ax1.set_title('Image 3')

    ax2.imshow(img4)
    ax2.set_title('Image 4')

    plt.figtext(0.5, 0.01, f"{len(good_matches)=} {len(matches)=} {len(kp1)=}", ha='center', fontsize=12)

    # Show the plot
    plt.show()


def is_overlapping(image1, image2, overlap_fraction=0.4):
    """
    Check if the overlap percentage between two images using ORB feature matching.

    :param image1: First input image in BGR format.
    :param image2: Second input image in BGR format.
    :param overlap_fraction: Fraction of overlap required between images.
    :return: True if the overlap is greater than or equal to the specified fraction.
             False if one of the images lack details or distinct features...
    """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is None or des2 is None:
        logging.error("One of the images lack details or distinct features!")
        return False

    matches = bf.match(des1, des2)

    good_matches = [m for m in matches if m.distance < GOOD_MATCH_DISTANCE_THRESHOLD]
    if len(kp1) == 0: # avoids division by 0
        return False
    overlap_percentage = len(good_matches) / len(kp1)

    # if  overlap_percentage < overlap_fraction:
    #     debug_matches(gray1, gray2, kp1, kp2, matches, good_matches)

    return overlap_percentage >= overlap_fraction


def clear_output_directory(directory_path):
    """
    Clear the output directory of all files and subdirectories.

    :param directory_path: Path object representing the directory to clear.
    """
    if directory_path.exists():
        logging.info(f"Clearing output directory: {directory_path}")
        for item in directory_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        logging.info(
            f"Output directory does not exist. Creating new one: {directory_path}"
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

            # Quality checks
            if not is_image_sharp(frame):
                logging.warning(f"Frame {count} is not sharp; hence too blurry. Skipping...")
                continue

            if prev_frame is not None and not is_overlapping(prev_frame, frame):
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

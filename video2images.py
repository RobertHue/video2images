"""Main Application video2images.py"""

# Python Module Index
import argparse
import logging
from pathlib import Path

# 3rd-Party
import cv2

# own libraries
from pipeline import clear_directory
from pipeline import get_feature_match_ratio
from pipeline import get_laplacian_variance


# Basic configuration for the root logger
logging.basicConfig(
    level=logging.INFO,  # Set default level for root logger
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logging.log", mode="w"),  # File handler
        logging.StreamHandler(),  # Console handler
    ],
)


# Define default values
DEFAULT_FORMAT = "png"
DEFAULT_BLUR_MIN_THRESHOLD = 20.0
DEFAULT_FEATURE_MAX_THRESHOLD = 0.10


def extract_frames(
    video_path,
    image_format=DEFAULT_FORMAT,
    blur_min_threshold=DEFAULT_BLUR_MIN_THRESHOLD,
    feature_max_threshold=DEFAULT_FEATURE_MAX_THRESHOLD,
):
    """
    Extracts and saves frames from a video file after performing quality checks.

    Args:
        video_path (str): Path to the video file.
        image_format (str, optional): Format to save the extracted frames (default is
            {DEFAULT_FORMAT}).
        blur_min_threshold (float, optional): Minimum variance of the Laplacian to
            consider a frame sharp (default is {DEFAULT_BLUR_MIN_THRESHOLD}).
        feature_max_threshold (float, optional): Maximum feature threshold until
            frames are filtered out (default is {DEFAULT_FEATURE_MAX_THRESHOLD}).

    Returns:
        None
    """
    # 1. Create an output directory or clear if already existent
    path_obj = Path(video_path)
    directory_path = path_obj.parent
    output_directory = directory_path / Path("extracted_frames")
    clear_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 2. Open the video file and retrieve needed properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"FPS of video: {fps}")
    logging.info(f"Frame count of video: {frame_count}")

    # 3. Processes each frame with the following quality checks:
    count = 0
    blur_count = 0
    common_feature_count = 0
    extracted_count = 0
    prev_frame = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            # # i. Blurriness Check: Filters out frames that are too blurry
            blur = get_laplacian_variance(frame)
            if blur < blur_min_threshold:
                logging.warning(
                    f"Frame {count} is not sharp; hence too blurry. "
                    f"Should be at least {blur_min_threshold} but is {blur}."
                    f"Skipping..."
                )
                blur_count += 1
                continue

            # ii. Feature Check: Filters out frames that do have too many
            #                    features in common with its previous frame
            if (
                prev_frame is not None
            ):  # start frame does not have anything to compare against
                feature_ratio = get_feature_match_ratio(prev_frame, frame)
                if feature_ratio > feature_max_threshold:
                    logging.warning(
                        f"Frame {count} does have too many features in common "
                        f"with its previous frame."
                        f"Should be at most {feature_max_threshold*100}% "
                        f"of feature match ratio but is {feature_ratio*100}%. "
                        f"Skipping..."
                    )
                    common_feature_count += 1
                    continue

            # 4. Save Valid Frames
            output_file = output_directory / f"frame_{count:04d}.{image_format}"
            cv2.imwrite(str(output_file), frame)
            logging.info(
                f"Frame {count} has been extracted and saved as {output_file.name}"
            )
            extracted_count += 1

            # Advance
            prev_frame = frame
    finally:
        # 5. Cleanup & Print Stats
        cap.release()
        cv2.destroyAllWindows()
        logging.info("#" * 32)
        logging.info(f"Total frames extracted: {count}")
        logging.info(f"Extracted frames: {extracted_count}")
        logging.info(
            f"Filtered frames with below {blur_min_threshold} "
            f"blur: {blur_count}"
        )
        logging.info(
            f"Filtered frames with {feature_max_threshold*100}% "
            f"common features: {common_feature_count}"
        )


################################################################################


def main():
    """
    Main function to parse arguments and call the frame extraction function.
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file with quality checks."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument(
        "--image_format",
        type=str,
        default=DEFAULT_FORMAT,
        help=f"Format to save the extracted frames (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--blur_min_threshold",
        type=float,
        default=DEFAULT_BLUR_MIN_THRESHOLD,
        help=f"Minimum blur threshold below which frames are discarded "
        f"(default: {DEFAULT_BLUR_MIN_THRESHOLD})",
    )
    parser.add_argument(
        "--feature_max_threshold",
        type=float,
        default=DEFAULT_FEATURE_MAX_THRESHOLD,
        help=f"Maximum feature threshold above which frames are discarded "
        f"(default: {DEFAULT_FEATURE_MAX_THRESHOLD})",
    )
    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        image_format=args.image_format,
        blur_min_threshold=args.blur_min_threshold,
        feature_max_threshold=args.feature_max_threshold,
    )


if __name__ == "__main__":
    main()

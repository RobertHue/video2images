# Python Module Index
import argparse
import logging
from pathlib import Path

# 3rd-Party
import cv2

# own libraries
from pipeline import clear_directory
from pipeline import compute_overlap
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
DEFAULT_BLUR_THRESHOLD = 50.0
DEFAULT_OVERLAP_THRESHOLD = 0.95


def extract_frames(
    video_path,
    format=DEFAULT_FORMAT,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
):
    """
    Extracts and saves frames from a video file after performing quality checks.

    Args:
        video_path (str): Path to the video file.
        format (str, optional): Format to save the extracted frames (default is
            {DEFAULT_FORMAT}).
        blur_threshold (float, optional): Minimum variance of the Laplacian to
            consider a frame sharp (default is {DEFAULT_BLUR_THRESHOLD}).
        overlap_threshold (float, optional): Minimum geometric overlap required
            between consecutive frames (default is {DEFAULT_OVERLAP_THRESHOLD}).

    Returns:
        None
    """
    # 1. Create an Output Directory: Sets up a directory to store the extracted frames.
    path_obj = Path(video_path)
    directory_path = path_obj.parent
    output_directory = directory_path / Path("extracted_frames")
    clear_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 2. Open the Video File: Loads the video and retrieves its properties,
    #                         for further processing.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"FPS of video: {fps}")
    logging.info(f"Frame count of video: {frame_count}")

    # 3. Iterate through frames
    count = 0
    prev_frame = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            # i. Blurriness Check: Filters out frames that are too blurry
            blur = get_laplacian_variance(frame)
            if blur >= blur_threshold:
                logging.warning(
                    f"Frame {count} is not sharp; hence too blurry. "
                    f"Should be at least {blur_threshold} but is {blur}."
                    f"Skipping..."
                )
                continue

            # ii. Overlap Check: Filters out consecutive frames that do not meet
            #                    the required geometric overlap
            if (
                prev_frame is not None
            ):  # start frame does not have anything to compare against
                overlap = compute_overlap(prev_frame, frame)
                if overlap >= overlap_threshold:
                    logging.warning(
                        f"Frame {count} is not overlapping enough with "
                        f"its previous frame."
                        f"Should be at least {overlap_threshold*100}% "
                        f"of geometric overlap but is {overlap*100}%. "
                        f"Skipping..."
                    )
                    continue

            # 4. Save valid frame as image
            output_file = output_directory / f"frame_{count:04d}.{format}"
            cv2.imwrite(str(output_file), frame)
            logging.info(
                f"Frame {count} has been extracted and saved as {output_file.name}"
            )

            # advance
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
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file with quality checks."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument(
        "--format",
        type=str,
        default=DEFAULT_FORMAT,
        help=f"Format to save the extracted frames (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=DEFAULT_BLUR_THRESHOLD,
        help=f"Minimum variance of the Laplacian to consider a frame sharp "
        f"(default: {DEFAULT_BLUR_THRESHOLD})",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help=f"Minimum geometric overlap required between consecutive frames "
        f"(default: {DEFAULT_OVERLAP_THRESHOLD})",
    )
    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        format=args.format,
        blur_threshold=args.blur_threshold,
        overlap_threshold=args.overlap_threshold,
    )


if __name__ == "__main__":
    main()

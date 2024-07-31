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

from dask import bag as db
from dask import compute, delayed
from dask.distributed import Client, LocalCluster

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

def save_frame(frame, filename):
    if frame is not None:
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the image
        cv2.imwrite(filename, frame_rgb)
    else:
        print("Failed to capture frame")




class Frame:

    __slots__ = ['index', 'blur', 'raw_frame', 'keypoints', 'descriptors', 'inlier_matches_num']

    def __init__(self, index=None, raw_frame=None, blur=None, keypoints=None, descriptors=None):
        self.index = index
        self.blur = blur
        self.raw_frame = raw_frame
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.inlier_matches_num = {}



# due to bugs in scikit-video with opening and reading files
# resorted to using OpenCV for reading frames

class VideoFile:

    def __init__(self, file):
        self.file = file
        # look at opencv documentation: Flags for video I/O
        # the cv2 properties did not function properly,
        # passing the integer value of the flag did
        self.capture = cv2.VideoCapture(self.file)
        self.number_of_frames = int(self.capture.get(7))
        self.capture = None
        self.current_index = None

    def __len__(self):
        return self.number_of_frames

    def __iter__(self):
        self.current_index = 0
        self.capture = cv2.VideoCapture(self.file)
        self.number_of_frames = int(self.capture.get(7))
        return self

    def __next__(self):
        self.current_index += 1
        ret, frame = self.capture.read() # ret is false at EOF
        if ret is False:
            self.current_index = None
            self.capture = None
            raise StopIteration
        elif ret is True:
            # cv2 opens in bgr mode and needs to be converted to RGB
            return {'index': self.current_index, 'raw_frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)}

from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from numba import jit
class Analysis:

    @staticmethod
    def compute_keypoints_descriptors_blur(frame, n_keypoints = 500):
        # using opencv to reduce dependencies needed
        # estimate blur by taking the image's laplacian vairance (blurry=low)
        blur = cv2.Laplacian(frame['raw_frame'], cv2.CV_64F).var()

        # skimage has poor detection speed, 10x slower as of writing this
        # keeping it here if in the future its better

        orb = skimage.feature.ORB(n_keypoints = n_keypoints, downscale=2)
        # skimage ORB needs grayscale image
        orb.detect_and_extract(skimage.color.rgb2gray(frame['raw_frame']))
        keypoints = orb.keypoints
        descriptors = orb.descriptors

        return {'index': frame['index'], 'blur': blur,
                'keypoints': keypoints, 'descriptors': descriptors}
        """
        # boilerplate from opencv python reference
        orb = cv2.ORB_create(nfeatures = n_keypoints) # Initiate ORB detector
        keypoints = orb.detect(frame['raw_frame'], None)
        keypoints, descriptors = orb.compute(frame['raw_frame'], keypoints)
        """
        # cannot pickle openCV keypoint objects unfortunately
        return {'index': frame['index'], 'blur': blur, 'descriptors': descriptors}

    @staticmethod
    def match_descriptors(descriptors1, descriptors2,
                          minsamples=8, maxtrials=500):
        # skimage has nicer matching then opencv
        # modified boilerplate example code from doc of skimage
        matches = match_descriptors(descriptors1, descriptors2,
                                    cross_check = True)
        # filtering out outliers, note first return is 'model', we dont care
        _, inliers = ransac((frame1.keypoints[matches[:, 0]],
                            frame2.keypoints[matches[:, 1]]),
                            FundamentalMatrixTransform,
                            min_samples = minsamples,
                            residual_threshold = 1, max_trials = maxtrials)

        # only the number of inliers matter to us
        inliers_sum = inliers.sum()
        return inliers_sum

################################################################################

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

################################################################################


def main():
    """
    Main function to parse arguments and call the frame extraction function.
    """
    parser = argparse.ArgumentParser(description="Split video into images")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    f = VideoFile(args.video_path)
    gen = iter(f) # get generator object basically
    # plt.imshow(next(gen)['raw_frame'])

    # test if analysis works
    a = Analysis.compute_keypoints_descriptors_blur
    m = Analysis.match_descriptors
    frame = next(f)
    res = a(frame)
    print(res)

    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)
    client

    #reset workers/clear RAM
    cluster.scale_down(cluster.workers)
    cluster.scale_up(2)

    from queue import Queue

    # pipeline setup
    # Note: if input data is big, and the input_q.put() is blocking code execution
    # in main thread, have the input_q be as little as possible IF the input
    # data is generated constantly and faster then it is consumed
    # Observation: input_q is sent to remote_q only when remote_q has been consumed
    input_q = Queue(maxsize=20) # input queue stored on local machine
    remote_q = client.scatter(input_q, maxsize=1) # queue on cluster for each worker
    calc_q = client.map(Analysis.compute_keypoints_descriptors_blur,
                        remote_q, maxsize = 1)
    output_q = client.gather(calc_q, maxsize = 5)

    # extract_frames(args.video_path)


if __name__ == "__main__":
    main()

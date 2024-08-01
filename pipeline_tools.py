# Python Module Index
import argparse
import itertools
import logging
import os
import pprint
import random
import time

from pathlib import Path

# 3rd-Party
import cv2 # opencv-python for frame reading
import dask # parallelized python EZ mode
import matplotlib.pyplot as plt # pretty charts no?
import numpy as np # yep
import skimage # scikit-image for loaded image analysis

from dask.distributed import Client
from itertools import repeat, islice
from numba import jit
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def compute_keypoints_descriptors_blur(frame, n_keypoints = 500,
                                       opencv=True, sift=False):
    """
    Compute keypoints and descriptors for a given frame, and estimate the blur level.

    Parameters:
    - frame (dict): A dictionary containing the 'raw_frame' (image data) and 'index' (frame index).
    - n_keypoints (int, optional): Number of keypoints to detect. Default is 500.
    - opencv (bool, optional): If True, use OpenCV for keypoint detection and descriptor computation. If False, use scikit-image. Default is True.
    - sift (bool, optional): If True and opencv is False, use SIFT (Scale-Invariant Feature Transform). If False, use ORB (Oriented FAST and Rotated BRIEF). Default is False.

    Returns:
    - dict: A dictionary with the following keys:
        - 'raw_frame': The original image data.
        - 'index': The index of the frame.
        - 'blur': The estimated blur level of the image, measured by the variance of the Laplacian.
        - 'keypoints': Array of detected keypoints (coordinates).
        - 'descriptors': Array of descriptors for the keypoints.

    Notes:
    - When opencv is False, the function uses scikit-image for keypoint detection, which might be slower.
    - For scikit-image ORB, the image is converted to grayscale.
    - OpenCV's ORB does not support direct pickle of keypoint objects, so keypoints are returned as coordinates.
    - Logging is used to report blur values and errors during processing.

    Example:
    >>> frame = {'raw_frame': some_image_data, 'index': 1}
    >>> result = compute_keypoints_descriptors_blur(frame, n_keypoints=300, opencv=False, sift=True)
    """
    # using opencv to reduce dependencies needed
    # estimate blur by taking the image's laplacian vairance (blurry=low)
    blur = cv2.Laplacian(frame['raw_frame'], cv2.CV_64F).var()

    if not opencv:
        # skimage has poor detection speed, 10x slower as of writing this
        # keeping it here if in the future its better

        if sift:
            orb = cv2.SIFT(nfeatures=n_keypoints)
        else:
            orb = skimage.feature.ORB(n_keypoints = n_keypoints, downscale=2)
        # skimage ORB needs grayscale image
        orb.detect_and_extract(skimage.color.rgb2gray(frame['raw_frame']))
        keypoints = orb.keypoints
        descriptors = orb.descriptors

        return {'raw_frame': frame['raw_frame'], 'index': frame['index'], 'blur': blur,
                'keypoints': keypoints, 'descriptors': descriptors}

    else:
        # boilerplate from opencv python reference
        orb = cv2.ORB_create(nfeatures = n_keypoints) # Initiate ORB detector
        keypoints_o = orb.detect(frame['raw_frame'], None)
        keypoints_o, descriptors = orb.compute(frame['raw_frame'], keypoints_o)
        # make keypoints compatible with scikit-image
        # array of [[x, y],] coords ndarray
        keypoints = np.ndarray(shape=(n_keypoints, 2), dtype=np.int64)

        try:
            for i, k in enumerate(keypoints_o, start=0):
                keypoints[i] = k.pt
        except Exception as e:
            logging.error(f"Error during enumerate: {e}")
            keypoints = None # if something goes catastrophically wrong

    logging.info(f"frame {frame['index']} has blur val of: {blur}")

    #cannot pickle openCV keypoint objects unfortunately, need to convert to coords (x,y aray)
    return {'raw_frame': frame['raw_frame'], 'index': frame['index'], 'blur': blur,
            'keypoints': keypoints, 'descriptors': descriptors}


def match_frames(frame1, frame2, minsamples=8, maxtrials=100, opencv=False):
    """
    Match keypoints between two image frames and count inliers.

    Parameters:
    - frame1 (dict): The first frame containing 'keypoints' and 'descriptors'.
    - frame2 (dict): The second frame containing 'keypoints' and 'descriptors'.
    - minsamples (int): Minimum samples for RANSAC. Default is 8.
    - maxtrials (int): Maximum trials for RANSAC. Default is 100.
    - opencv (bool): Flag to use OpenCV instead of skimage. Default is False.

    Returns:
    - int: Number of inliers after RANSAC filtering.
    """
    if opencv is False:
        # skimage has nicer matching then opencv
        # modified boilerplate example code from doc of skimage
        # ORB
        matches = match_descriptors(frame1['descriptors'],
                                    frame2['descriptors'],
                                    cross_check = True)
        try:
            # filtering out outliers, note first return is 'model', we dont care
            _, inliers = ransac((frame1['keypoints'][matches[:, 0]],
                                frame2['keypoints'][matches[:, 1]]),
                                FundamentalMatrixTransform,
                                min_samples = minsamples,
                                residual_threshold = 1, max_trials = maxtrials)

            # only the number of inliers matter to us
            inliers_sum = inliers.sum()

        except Exception as e:
            # just show raw matches if RANSAC errors out
            inliers_sum = len(matches)

        finally:
            return inliers_sum

    else:
        pass # I doubt anyone wants to use opencv here


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


# Define a constant for good match distance threshold (update as needed)
GOOD_MATCH_DISTANCE_THRESHOLD = 30.0  # Example threshold, adjust based on needs

def check_feature_match_ratio(image1, image2, overlap_fraction=0.4):
    """
    Determine if the ratio of good keypoint matches between two images meets or exceeds a specified threshold using ORB feature matching.

    Parameters:
    - image1: The first input image in BGR format (numpy.ndarray).
    - image2: The second input image in BGR format (numpy.ndarray).
    - overlap_fraction: The minimum fraction of keypoints in image1 that need to be matched in image2 (float).

    Returns:
    - True if the ratio of good matches is greater than or equal to the specified overlap_fraction.
    - False if one or both images lack distinct features or if the match ratio is insufficient.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Initialize brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Handle cases where descriptors are None
    if des1 is None or des2 is None:
        logging.error("One of the images lacks distinct features or details.")
        return False

    # Match descriptors
    matches = bf.match(des1, des2)

    # Filter good matches based on distance
    good_matches = [m for m in matches if m.distance < GOOD_MATCH_DISTANCE_THRESHOLD]

    # Avoid division by zero
    if len(kp1) == 0:
        return False

    # Calculate match ratio
    match_ratio = len(good_matches) / len(kp1)

    # Return whether the match ratio meets the threshold
    return match_ratio >= overlap_fraction


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

################################################################################

# Define a constant for good match distance threshold (update as needed)
GOOD_MATCH_DISTANCE_THRESHOLD = 30.0  # Example threshold, adjust based on needs


def detect_and_compute_features(image):
    """
    Detect keypoints and compute descriptors using ORB.

    :param image: Input image in BGR format (numpy.ndarray).
    :return: Tuple of (keypoints, descriptors).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(des1, des2):
    """
    Match descriptors between two sets using a brute-force matcher.

    :param des1: Descriptors of the first image (numpy.ndarray).
    :param des2: Descriptors of the second image (numpy.ndarray).
    :return: List of matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches


def estimate_homography(src_pts, dst_pts):
    """
    Estimate homography matrix using RANSAC.

    :param src_pts: Source points (numpy.ndarray).
    :param dst_pts: Destination points (numpy.ndarray).
    :return: Homography matrix (numpy.ndarray) or None if estimation fails.
    """
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H


def warp_image(image, H, shape):
    """
    Warp image using a homography matrix.

    :param image: Input image in BGR format (numpy.ndarray).
    :param H: Homography matrix (numpy.ndarray).
    :param shape: Shape of the target image (tuple).
    :return: Warped image (numpy.ndarray).
    """
    warped_image = cv2.warpPerspective(image, H, (shape[1], shape[0]))
    return warped_image


def compute_overlap(image1, image2, warped_image1):
    """
    Compute overlap fraction between the two images.

    :param image1: First image in BGR format (numpy.ndarray).
    :param image2: Second image in BGR format (numpy.ndarray).
    :param warped_image1: Warped version of the first image (numpy.ndarray).
    :return: Overlap fraction (float).
    """
    gray_warped = cv2.cvtColor(warped_image1, cv2.COLOR_BGR2GRAY)
    _, binary_warped = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(gray_image2, 1, 255, cv2.THRESH_BINARY)

    intersection = np.logical_and(binary_warped, binary_image2).sum()
    union = np.logical_or(binary_warped, binary_image2).sum()

    overlap_fraction = intersection / union if union != 0 else 0.0
    return overlap_fraction


def estimate_geometric_overlap(image1, image2):
    """
    Estimate the geometric overlap between two images.

    :param image1: The first input image in BGR format (numpy.ndarray).
    :param image2: The second input image in BGR format (numpy.ndarray).
    :return: Overlap fraction (float).
    """
    # Detect and compute features
    kp1, des1 = detect_and_compute_features(image1)
    kp2, des2 = detect_and_compute_features(image2)

    if des1 is None or des2 is None:
        raise ValueError("One of the images lacks distinct features or details.")

    # Match descriptors
    matches = match_descriptors(des1, des2)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography
    H = estimate_homography(src_pts, dst_pts)

    if H is None:
        raise ValueError("Homography estimation failed.")

    # Warp the first image
    warped_image1 = warp_image(image1, H, image2.shape)

    # Compute overlap
    overlap_fraction = compute_overlap(image1, image2, warped_image1)

    return overlap_fraction

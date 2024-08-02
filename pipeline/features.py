"""Pipeline Helper Tools used by the video to images process."""

# Python Module Index
import logging

# 3rd-Party
import cv2
import numpy as np


def get_laplacian_variance(image):
    """
    Calculate the Laplacian variance of an image.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        float: The variance of the Laplacian, representing the sharpness of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def is_well_exposed(image, low_threshold=0.1, high_threshold=0.9):
    """
    Check if the image is well-exposed based on its histogram.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        low_threshold (float, optional): Lower bound for cumulative
            distribution function (default: 0.1).
        high_threshold (float, optional): Upper bound for cumulative
            distribution function (default: 0.9).

    Returns:
        bool: True if the image is well-exposed, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    cdf = np.cumsum(hist)
    return (cdf[0] > low_threshold) and (cdf[-1] < high_threshold)


def get_feature_match_ratio(image1, image2, good_match_distance=30.0):
    """
    Determine the ratio of good keypoint matches between two images using ORB feature matching.

    Args:
        image1 (numpy.ndarray): The first input image in BGR format.
        image2 (numpy.ndarray): The second input image in BGR format.
        good_match_distance (float, optional): Maximum match distance to be considered a good match (default: 30.0).

    Returns:
        float: The match ratio of the two images.
    """
    DO_NOT_MATCH = 0.0  # used for when the frames do not match or in error cases

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
        return DO_NOT_MATCH

    # Match descriptors
    matches = bf.match(des1, des2)

    # Filter good matches based on distance
    good_matches = [m for m in matches if m.distance < good_match_distance]

    # Avoid division by zero
    if len(kp1) == 0:
        return DO_NOT_MATCH

    # Calculate match ratio
    match_ratio = len(good_matches) / len(kp1)

    # Return whether the match ratio meets the threshold
    return match_ratio

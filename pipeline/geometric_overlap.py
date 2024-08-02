# Python Module Index

# 3rd-Party
import cv2
import numpy as np


def detect_and_compute_features(image):
    """
    Detect keypoints and compute descriptors using ORB.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        Tuple: A tuple containing:
            - keypoints (tuple of cv2.KeyPoint): Detected keypoints.
            - descriptors (numpy.ndarray): Computed descriptors.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(des1, des2):
    """
    Match descriptors between two sets using a brute-force matcher.

    Args:
        des1 (numpy.ndarray): Descriptors of the first image.
        des2 (numpy.ndarray): Descriptors of the second image.

    Returns:
        tuple: tuple of matches between the descriptors.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches


def estimate_homography(src_pts, dst_pts):
    """
    Estimate the homography matrix using RANSAC.

    Args:
        src_pts (numpy.ndarray): Source points.
        dst_pts (numpy.ndarray): Destination points.

    Returns:
        numpy.ndarray or None: Homography matrix if estimation is successful; otherwise, None.
    """
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H


def warp_image(image, H, shape):
    """
    Warp the image using a homography matrix.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        H (numpy.ndarray): Homography matrix.
        shape (tuple): Shape of the target image.

    Returns:
        numpy.ndarray: Warped image.
    """
    warped_image = cv2.warpPerspective(image, H, (shape[1], shape[0]))
    return warped_image


def compute_overlap(image1, image2):
    """
    Compute the overlap fraction between two images.

    Args:
        image1 (numpy.ndarray): The first input image in BGR format.
        image2 (numpy.ndarray): The second input image in BGR format.

    Returns:
        float: Overlap fraction between the two images.
    """
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    _, binary_image1 = cv2.threshold(gray_image1, 1, 255, cv2.THRESH_BINARY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(gray_image2, 1, 255, cv2.THRESH_BINARY)

    intersection = np.logical_and(binary_image1, binary_image2).sum()
    union = np.logical_or(binary_image1, binary_image2).sum()

    overlap_fraction = intersection / union if union != 0 else 0.0
    return overlap_fraction


def estimate_geometric_overlap(image1, image2):
    """
    Estimate the geometric overlap between two images.

    Args:
        image1 (numpy.ndarray): The first input image in BGR format.
        image2 (numpy.ndarray): The second input image in BGR format.

    Returns:
        float: Overlap fraction between the two images.

    Raises:
        ValueError: If one of the images lacks distinct features or details, or if homography estimation fails.
    """
    kp1, des1 = detect_and_compute_features(image1)
    kp2, des2 = detect_and_compute_features(image2)

    if des1 is None or des2 is None:
        raise ValueError(
            "One of the images lacks distinct features or details."
        )

    matches = match_descriptors(des1, des2)

    # Extract keypoints from matches for src and dst points
    src_pts_list = [kp1[m.queryIdx].pt for m in matches]
    dst_pts_list = [kp2[m.trainIdx].pt for m in matches]

    # Convert keypoints to numpy arrays and reshape
    if not src_pts_list or not dst_pts_list:
        raise ValueError("No matches found between images.")
    src_pts = np.array(src_pts_list, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts_list, dtype=np.float32).reshape(-1, 1, 2)

    H = estimate_homography(src_pts, dst_pts)

    if H is None:
        raise ValueError("Homography estimation failed.")

    warped_image1 = warp_image(image1, H, image2.shape)
    overlap_fraction = compute_overlap(image2, warped_image1)

    return overlap_fraction

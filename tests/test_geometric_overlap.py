import cv2
import numpy as np
import pytest

from pipeline.geometric_overlap import compute_overlap
from pipeline.geometric_overlap import detect_and_compute_features
from pipeline.geometric_overlap import estimate_geometric_overlap
from pipeline.geometric_overlap import estimate_homography
from pipeline.geometric_overlap import match_descriptors
from pipeline.geometric_overlap import warp_image


def test_detect_and_compute_features():
    """
    Test that detect_and_compute_features returns a list of keypoints and
    descriptors as either None or a numpy array.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints, descriptors = detect_and_compute_features(image)
    assert isinstance(keypoints, tuple)  # keypoints should be a tuple
    assert descriptors is None or isinstance(
        descriptors, np.ndarray
    )  # descriptors can be None or a numpy array


def test_match_descriptors():
    """
    Test that match_descriptors returns a list of cv2.DMatch objects.
    """
    des1 = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
    des2 = np.random.randint(0, 256, (10, 32), dtype=np.uint8)
    matches = match_descriptors(des1, des2)
    assert isinstance(
        matches, tuple
    )  # matches should be a tuple of cv2.DMatch objects
    assert all(
        isinstance(m, cv2.DMatch) for m in matches
    )  # all elements in matches should be cv2.DMatch objects


def test_estimate_homography():
    """
    Test that estimate_homography returns a numpy array or None.
    """
    src_pts = np.random.rand(4, 1, 2).astype(np.float32)
    dst_pts = np.random.rand(4, 1, 2).astype(np.float32)
    H = estimate_homography(src_pts, dst_pts)
    assert H is None or isinstance(H, np.ndarray)


def test_warp_image():
    """
    Test that warp_image returns a numpy array with the expected shape.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    H = np.eye(3)
    shape = (100, 100)
    warped_image = warp_image(image, H, shape)
    assert isinstance(warped_image, np.ndarray)
    assert warped_image.shape == (100, 100, 3)


def test_compute_overlap():
    """
    Test that compute_overlap returns a float.
    """
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    overlap = compute_overlap(image1, image2)
    assert isinstance(overlap, float)


def test_estimate_geometric_overlap():
    """
    Test that estimate_geometric_overlap raises a ValueError for invalid input images.
    """
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_geometric_overlap(image1, image2)

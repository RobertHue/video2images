# Python Module Index
import logging

# 3rd-Party
import cv2
import matplotlib.pyplot as plt


def debug_matches(gray1, gray2, kp1, kp2, matches, good_matches):
    """
    Debug and visualize keypoint matches between two images.

    Args:
        gray1 (numpy.ndarray): The first image in grayscale.
        gray2 (numpy.ndarray): The second image in grayscale.
        kp1 (list of cv2.KeyPoint): Keypoints from the first image.
        kp2 (list of cv2.KeyPoint): Keypoints from the second image.
        matches (list of cv2.DMatch): All matches between keypoints.
        good_matches (list of cv2.DMatch): Good matches that passed quality checks.

    Returns:
        None
    """
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

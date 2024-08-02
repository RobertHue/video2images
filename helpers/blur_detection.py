"""Tutorial from https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

Title:  Blur detection with OpenCV
Author: Adrian Rosebrock
Date:   September 7, 2015
"""

# Python Module Index
import argparse

# 3rd-Party
import cv2
from imutils import paths

# own libraries
from pipeline import get_laplacian_variance


def main():
    """
    Parse command-line arguments, process images to detect blur, and display results.

    Parses the path to an input directory of images and a blur threshold. For each image in the directory,
    calculates the focus measure using the Variance of Laplacian method. Displays each image with a label indicating
    whether it is "Blurry" or "Not Blurry" based on the focus measure and the specified threshold.

    Args:
        None

    Returns:
        None
    """
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--images",
        required=True,
        help="path to input directory of images",
    )
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=100.0,
        help="focus measures that fall below this value will be considered 'blurry'",
    )
    args = vars(ap.parse_args())

    # loop over the input images
    for imagePath in paths.list_images(args["images"]):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        image = cv2.imread(imagePath)
        fm = get_laplacian_variance(image)
        text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < args["threshold"]:
            text = "Blurry"
        # show the image
        cv2.putText(
            image,
            "{}: {:.2f}".format(text, fm),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )
        cv2.imshow("Image", image)
        _ = cv2.waitKey(0)


if __name__ == "__main__":
    main()

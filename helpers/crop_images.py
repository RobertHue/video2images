"""crops all the images in a given directory (input_dir) into an output_dir
placed above the input_dir folder:

Usage:
    python crop_images.py /path/to/your/images --crop_box 50 50 200 200

"""

import argparse
import os

from PIL import Image


class ImageCropper:
    def __init__(self, input_dir, x, y, width, height):
        self.input_dir = input_dir
        self.output_dir = os.path.abspath(os.path.join(input_dir, os.pardir))
        self.output_dir = os.path.join(self.output_dir, "cropped")

        # Convert x, y, width, height to (left, upper, right, lower)
        self.crop_box = (x, y, x + width, y + height)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def crop_images(self):
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.input_dir, filename)
                with Image.open(image_path) as img:
                    # Perform the crop
                    cropped_image = img.crop(self.crop_box)

                    # Save the cropped image
                    cropped_image_path = os.path.join(self.output_dir, filename)
                    cropped_image.save(cropped_image_path)

                print(f"Cropped and saved: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Crop images in a directory and save them in a 'cropped' subdirectory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("--crop_box", type=int, nargs=4, default=[100, 100, 400, 400],
                        help="Properties for the crop box (x, y, width, height).")

    args = parser.parse_args()

    cropper = ImageCropper(args.input_dir, *args.crop_box)
    cropper.crop_images()
    print(f"{cropper.input_dir=}")
    print(f"{cropper.output_dir=}")
    print(f"{cropper.crop_box=}")

if __name__ == "__main__":
    main()

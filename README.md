# convert-video2image-and-image2video using Python
Simple Python code for converting video to images and images to video

Medium article: https://medium.com/@wayandadangunsri/converting-video-to-images-using-python-and-opencv-72b2ea66a692

**A. Converting Video to Images**

This code is a Python script used to convert a video into a series of images with a specific time interval. Here are some key points about this code:

- **Using OpenCV Library**: This code utilizes the OpenCV library to read the video and process its frames.
- **Customizable Frame Rate**: You can set the desired frame rate (in this example, 1 frame every 0.5 seconds) to determine the frame-cutting interval.
- **Image Storage**: Each frame image is saved in an output folder named after the video's name without an extension, such as "video_name_frames," in JPG format.
- **How to Run the Code**: Ensure you have Python installed on your computer and install the required dependencies by running the `pip install -r requirements.txt` command, where `requirements.txt` contains a list of dependencies, including OpenCV. Add the video you want to process to the same directory as this code. Replace the video file's name on the line `video_file = "your_video.mp4"` with your video file's name, and run the Python script with the `python script_name.py` command, where `script_name.py` is the name of your Python script. The resulting cut images will be saved in the `video_name_frames` folder.

**Example Usage**:

Suppose you have a video named "video.mp4." You run the script with the `python cut_video_to_images.py` command. The resulting cut images will be saved in the "video_frames" folder.

**B. Converting Images to Video**

This code is a Python script used to convert a series of images in a specific order into a video. Here are some important considerations about this code:

- **Using OpenCV**: This code leverages the OpenCV library to read the images based on their file names.
- **Customizable Sequence**: Images are sorted based on their file names.
- **Uniform Video Resolution**: The resolution of the video is determined based on the resolution of the first image, ensuring uniform video quality.
- **Video Format and Frame Rate**: The resulting video is saved in MP4 format with a frame rate of 2 frames per second.
- **How to Run the Code**: Make sure you have Python installed on your computer and install the required dependencies by running the `pip install -r requirements.txt` command, where `requirements.txt` contains a list of dependencies, including OpenCV. Ensure that the images you want to combine into a video are placed in the same directory as this code. Replace `"your_image_directory"` with the directory where your images are stored, and specify the output video file name on the line `output_video = "output_video.mp4"`. Then, run the Python script with the `python script_name.py` command, where `script_name.py` is the name of your Python script. The resulting video will be saved as "output_video.mp4."

**Example Usage**:

Suppose you have a series of images named "frame_1.jpg," "frame_2.jpg," and so on, that you want to combine into a video. You run the script with the `python combine_images_to_video.py` command. The resulting video will be saved as "output_video.mp4."

**How to Run Both Codes**

1. **Converting Video to Images**:

   - Ensure you have Python installed on your computer.
   - Install the required dependencies by running the `pip install -r requirements.txt` command, where `requirements.txt` contains a list of dependencies, including OpenCV.
   - Add the video you want to process to the same directory as this code.
   - Replace the video file's name on the line `video_file = "your_video.mp4"` with your video file's name.
   - Run the Python script with the `python script_name.py` command, where `script_name.py` is the name of your Python script.
   - The resulting cut images will be saved in the `video_name_frames` folder.

2. **Converting Images to Video**:

   - Ensure you have Python installed on your computer.
   - Install the required dependencies by running the `pip install -r requirements.txt` command, where `requirements.txt` contains a list of dependencies, including OpenCV.
   - Ensure that the images you want to combine into a video are placed in the same directory as this code.
   - Replace `"your_image_directory"` with the directory where your images are stored.
   - Specify the output video file name on the line `output_video = "output_video.mp4"`.
   - Run the Python script with the `python script_name.py` command, where `script_name.py` is the name of your Python script.
   - The resulting video will be saved as "output_video.mp4."

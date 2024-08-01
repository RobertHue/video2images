# video2images

Converts a video into images filtered for [Photogrammetry](https://en.wikipedia.org/wiki/Photogrammetry), such as in [Meshroom](https://github.com/alicevision/Meshroom).

This project aligns with the goals and motivation outlined in [Photogrammetry Datasets from Video - A Slightly Less Naive Approach](https://gist.github.com/AzureDVBB/49f5240faedc421e7c3939567eaddb59).

## Goal of This Project

Provide an easy-to-use collection of scripts to:

- Ingest video files
- Analyze them for optimal frames
- Save those frames as a sequence of image files suitable for mainstream photogrammetry software

The project aims to:

- **Reduce on-site time** needed for data capture
- **Minimize the number of input images** by selecting the best frames while maintaining a consistent amount of overlap between them
- **Ensure the least blurry frames** are used


## Table of Contents

- [video2images](#video2images)
  - [Goal of This Project](#goal-of-this-project)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Installation Prerequisites](#installation-prerequisites)
    - [Usage](#usage)
  - [How It Works](#how-it-works)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Getting Started

### Installation Prerequisites

- [Git](https://git-scm.com/downloads)
- [VSCode](https://code.visualstudio.com/)
- [Python](https://www.python.org/)

Then, install the required packages by running:

  ```console
  pip install -r requirements.txt
  ```

### Usage

For command line arguments, use:

  ```console
  python video2images.py --help
  ```

To convert a video into images, run:

  ```console
  python video2images.py PATH_TO_VIDEO
  ```

This command will create a folder named `extracted_frames` at the same location as `PATH_TO_VIDEO`.

## How It Works

The process is based on a simple, pipeline-oriented approach. No multiprocessing, threads, or clusters are used.

The pipeline operates as follows:

1. **Create an Output Directory**: Sets up a directory to store the extracted frames.
2. **Open the Video File**: Loads the video and retrieves its properties, such as FPS (frames per second) and the total number of frames.
3. **Iterate Through Frames**: Processes each frame with the following quality checks:
   1. **Blurriness Check**: Filters out frames that are too blurry based on the blur_threshold.
   2. **Overlap Check**: Filters out consecutive frames that do not meet the required geometric overlap based on the overlap_threshold.
4. **Save Valid Frames**: Saves the frames that pass the quality checks to the output directory.

## Acknowledgments

We would like to express our gratitude to the following resources:

- [Adrian Rosebrock's Tutorial](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) for providing valuable insights on detecting and debugging blur in frames.
- [AzureDVBB](https://gist.github.com/AzureDVBB/) for demonstrating an alternative approach to filtering frames from a video. For additional context, see the related article, [Photogrammetry Datasets from Video - A Slightly Less Naive Approach](https://gist.github.com/AzureDVBB/49f5240faedc421e7c3939567eaddb59).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

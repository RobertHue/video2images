# video2images

![Python](https://img.shields.io/badge/python-3.12+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/RobertHue/video2images/actions/workflows/ci-test.yml/badge.svg?branch=master)](https://github.com/RobertHue/video2images/actions/workflows/ci-test.yml)
[![PyLint](https://github.com/RobertHue/video2images/actions/workflows/pylint.yml/badge.svg?branch=master)](https://github.com/RobertHue/video2images/actions/workflows/pylint.yml)

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

## Getting Started

### Installation Prerequisites

- [Git](https://git-scm.com/downloads)
- [VSCode](https://code.visualstudio.com/)
- [Python](https://www.python.org/)
- [Poetry](https://python-poetry.org/)

To install Poetry, follow these steps:

   ```console
   python -m pip install --upgrade pip
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

To install the project dependencies, execute the following command:

   ```console
   poetry install
   ```

To activate the virtual environment, run:

   ```console
   poetry shell
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
   1. **Blurriness Check**: Filters out frames that are too blurry based on the `blur_threshold`.
   2. **Feature Check**: Feature Check: Filters out frame that does not have too many features in common with its previous frame based on `feature_threshold`.
4. **Save Valid Frames**: Saves the frames that pass the quality checks to the output directory.

## Acknowledgments

Special thanks to the following resources for their contributions:

- [Adrian Rosebrock's Tutorial](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) for providing valuable insights on detecting and debugging blur in frames.
- [AzureDVBB](https://gist.github.com/AzureDVBB/) for demonstrating an alternative approach to filtering frames from a video. For additional context, see the related article, [Photogrammetry Datasets from Video - A Slightly Less Naive Approach](https://gist.github.com/AzureDVBB/49f5240faedc421e7c3939567eaddb59).

If you have other possible approaches or suggestions, please contact me at [robert.huem@gmail.com](mailto:robert.huem@gmail.com).

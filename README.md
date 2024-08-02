# video2images

![Python](https://img.shields.io/badge/python-3.12+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ci-tests](https://github.com/RobertHue/video2images/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/RobertHue/video2images/actions/workflows/ci-tests.yml)
[![ci-linting](https://github.com/RobertHue/video2images/actions/workflows/ci-linting.yml/badge.svg)](https://github.com/RobertHue/video2images/actions/workflows/ci-linting.yml)

Converts a video into images filtered for [Photogrammetry](https://en.wikipedia.org/wiki/Photogrammetry).

Its main goal is to reduce the on-site time needed for data capture and also the time for photogrammetry software, such as [Meshroom](https://github.com/alicevision/Meshroom), to compute by only using a small selection of frames from a video data capture. This tool filters out frames that are too blurry to be used and consecutive frames that have too many features in common.

## Table of Contents

- [video2images](#video2images)
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

1. **Create an output directory or clear if already existant**
2. **Open the video file and retrieve needed properties**
3. Processes each frame with the following quality checks:
   1. **Blurriness Check**: Filters out frames that are too blurry based on the `blur_min_threshold`.
   2. **Feature Check**: Filters out frame that does have too many features in common with its previous frame based on `feature_max_threshold`.
4. **Save Valid Frames**: Saves the frames that pass the quality checks to the output directory.
5. **Cleanup & Print Stats**

## Acknowledgments

Special thanks to the following resources for their contributions:

- [Adrian Rosebrock's Tutorial](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) for providing valuable insights on detecting and debugging blur in frames.
- [AzureDVBB](https://gist.github.com/AzureDVBB/) for demonstrating an alternative approach to filtering frames from a video. For additional context, see the related article, [Photogrammetry Datasets from Video - A Slightly Less Naive Approach](https://gist.github.com/AzureDVBB/49f5240faedc421e7c3939567eaddb59).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

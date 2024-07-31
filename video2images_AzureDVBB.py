# look into 'streamz' package, neat pipelining with dask integration

# Python Module Index
import argparse
import itertools
import logging
import os
import pprint
import random
import time

from pathlib import Path

# 3rd-Party
import cv2 # opencv-python for frame reading
import dask # parallelized python EZ mode
import matplotlib.pyplot as plt # pretty charts no?
import numpy as np # yep
import skimage # scikit-image for loaded image analysis

from dask.distributed import Client
from numba import jit
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


# Basic configuration for the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set default level for root logger
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logging.log', mode='w'),  # File handler
        logging.StreamHandler()  # Console handler
    ]
)


# due to bugs in scikit-video with opening and reading files
# resorted to using OpenCV for reading frames

class VideoFile_p:

    def __init__(self, file):
        self.file = file
        # look at opencv documentation: Flags for video I/O
        # the cv2 properties did not function properly,
        # passing the integer value of the flag did
        self.capture = cv2.VideoCapture(self.file)
        self.number_of_frames = int(self.capture.get(7))
        self.current_index = 0

    def __len__(self):
        return self.number_of_frames

    def __iter__(self):
        self.current_index = 0
        self.capture = cv2.VideoCapture(self.file)
        self.number_of_frames = int(self.capture.get(7))
        return self

    def __next__(self):
        self.current_index += 1
        ret, frame = self.capture.read() # ret is false at EOF
        if ret is False:
            self.current_index = None
            self.capture = None
            raise StopIteration
        elif ret is True:
            # cv2 opens in bgr mode and needs to be converted to RGB
            return {'index': self.current_index,
                    'raw_frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)}

    def save_frames(self, list_of_frames, save_folder):
        # checks and makes the directory path if not existing
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        reader = iter(self)
        for img in reader:
            # premature termination on last frame write
            if img['index'] > max(list_of_frames):
                break

            elif img['index'] in list_of_frames:
                # padding out number for up to 6 digits
                filename = os.path.join(save_folder, str(img['index']).zfill(6)) + '.jpg'
                frame = cv2.cvtColor(img['raw_frame'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, frame)


def save_frame(frame, filename):
    if frame is not None:
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the image
        cv2.imwrite(filename, frame_rgb)
    else:
        print("Failed to capture frame")


class Analysis_p:

    @staticmethod
    def compute_keypoints_descriptors_blur(frame, n_keypoints = 500,
                                           opencv=True, sift=False):
        # using opencv to reduce dependencies needed
        # estimate blur by taking the image's laplacian vairance (blurry=low)
        blur = cv2.Laplacian(frame['raw_frame'], cv2.CV_64F).var()

        if not opencv:
            # skimage has poor detection speed, 10x slower as of writing this
            # keeping it here if in the future its better

            if sift:
                orb = cv2.SIFT(nfeatures=n_keypoints)
            else:
                orb = skimage.feature.ORB(n_keypoints = n_keypoints, downscale=2)
            # skimage ORB needs grayscale image
            orb.detect_and_extract(skimage.color.rgb2gray(frame['raw_frame']))
            keypoints = orb.keypoints
            descriptors = orb.descriptors

            return {'raw_frame': frame['raw_frame'], 'index': frame['index'], 'blur': blur,
                    'keypoints': keypoints, 'descriptors': descriptors}

        else:
            # boilerplate from opencv python reference
            orb = cv2.ORB_create(nfeatures = n_keypoints) # Initiate ORB detector
            keypoints_o = orb.detect(frame['raw_frame'], None)
            keypoints_o, descriptors = orb.compute(frame['raw_frame'], keypoints_o)
            # make keypoints compatible with scikit-image
            # array of [[x, y],] coords ndarray
            keypoints = np.ndarray(shape=(n_keypoints, 2), dtype=np.int64)

            try:
                for i, k in enumerate(keypoints_o, start=0):
                    keypoints[i] = k.pt
            except Exception as e:
                logging.error(f"Error during enumerate: {e}")
                keypoints = None # if something goes catastrophically wrong

        #cannot pickle openCV keypoint objects unfortunately, need to convert to coords (x,y aray)
        return {'raw_frame': frame['raw_frame'], 'index': frame['index'], 'blur': blur,
                'keypoints': keypoints, 'descriptors': descriptors}


    @staticmethod
    def match_frames(frame1, frame2, minsamples=8, maxtrials=100, opencv=False):
        """
        Match keypoints between two image frames and count inliers.

        Parameters:
        - frame1 (dict): The first frame containing 'keypoints' and 'descriptors'.
        - frame2 (dict): The second frame containing 'keypoints' and 'descriptors'.
        - minsamples (int): Minimum samples for RANSAC. Default is 8.
        - maxtrials (int): Maximum trials for RANSAC. Default is 100.
        - opencv (bool): Flag to use OpenCV instead of skimage. Default is False.

        Returns:
        - int: Number of inliers after RANSAC filtering.
        """
        if opencv is False:
            # skimage has nicer matching then opencv
            # modified boilerplate example code from doc of skimage
            # ORB
            matches = match_descriptors(frame1['descriptors'],
                                        frame2['descriptors'],
                                        cross_check = True)
            try:
                # filtering out outliers, note first return is 'model', we dont care
                _, inliers = ransac((frame1['keypoints'][matches[:, 0]],
                                    frame2['keypoints'][matches[:, 1]]),
                                    FundamentalMatrixTransform,
                                    min_samples = minsamples,
                                    residual_threshold = 1, max_trials = maxtrials)

                # only the number of inliers matter to us
                inliers_sum = inliers.sum()

            except Exception as e:
                # just show raw matches if RANSAC errors out
                inliers_sum = len(matches)

            finally:
                return inliers_sum

        else:
            pass # I doubt anyone wants to use opencv here


class FrameSelection_p:

    def __init__(self):
        pass

    def variance_picker(matches_to_base_frame, min_variance=0.1):
        new = None
        old = None

        for i, _ in enumerate(matches_to_base_frame, start=0):
            if old is None:
                old = matches_to_base_frame[i]
                new = old
            else:
                old = new
                #new = sum(matches_to_base_frame[:i])/(i+1)
                new = matches_to_base_frame[i]
                variance = abs(new - old) / old

                if variance <= min_variance:
                    return i

        return None # too much variance in dataset

    def compute_best_frames(frame_stream, last_frame_index, client,
                            batch_size=10, min_variance=0.05):

        from itertools import repeat, islice

        last_frame_index = last_frame_index-1 # removes infinite loop bug
        frame_generator = itertools.islice(frame_stream, last_frame_index)
        base_descriptor = None
        batch_num = 1
        descriptor_collection = []
        found_at_collection_index = None
        matches_to_base_frame = []
        good_frame_indexes = [1]   # include first frame
        last_batch = False


        while True:

            if base_descriptor is None:
                base_descriptor = client.submit(
                        Analysis_p.compute_keypoints_descriptors_blur,
                        next(frame_generator))

            # check if the next batch is the last one
            if good_frame_indexes[-1] + batch_num*batch_size >= last_frame_index:
                if last_batch is True:
                    break  # end the loop if it has been
                else:
                    last_batch = True
                    # put the appropriate amount onto the collection
                    for frame in islice(frame_generator,
                                        last_frame_index - good_frame_indexes[-1] - batch_size * (batch_num - 1)):
                        future = client.submit(Analysis_p.compute_keypoints_descriptors_blur, frame)
                        descriptor_collection.append(future)

            else:
                for frame in islice(frame_generator,
                                    batch_size * batch_num - len(descriptor_collection)):
                    future = client.submit(Analysis_p.compute_keypoints_descriptors_blur, frame)
                    descriptor_collection.append(future)

            # match all elements in the collection against base
            for frame_future in islice(descriptor_collection,
                                       len(matches_to_base_frame),
                                       batch_size * batch_num):
                match_future = client.submit(Analysis_p.match_frames, base_descriptor, frame_future)
                matches_to_base_frame.append(match_future)

            matches_to_base_frame = client.gather(matches_to_base_frame)

            # selection pass
            found_at_collection_index = FrameSelection_p.variance_picker(
                                                        matches_to_base_frame,
                                                        min_variance=min_variance)

            if found_at_collection_index is not None:
                # save the frame's index as good
                frame_index = descriptor_collection[found_at_collection_index].result()['index']
                base_descriptor = descriptor_collection[found_at_collection_index]
                good_frame_indexes.append(frame_index)
                # make the good frame the base
                base_descriptor = descriptor_collection[found_at_collection_index]

                # delete frame dictionaries at new base and before
                # and reset variables
                del descriptor_collection[:found_at_collection_index+1]
                found_at_collection_index = None
                matches_to_base_frame.clear()
                batch_num = 1

            else:
                # if not found then repeat
                batch_num += 1

        # repeat untill input frames are exhausted

        return good_frame_indexes # finished


def main():
    """
    Main function to parse arguments and call the frame extraction function.
    """
    from dask.distributed import LocalCluster
    cluster = LocalCluster()
    client = cluster.get_client()
    # from dask.distributed import Client
    # client = Client('tcp://127.0.0.1:8786') #change address for cluster's one

    parser = argparse.ArgumentParser(description="Split video into images")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    path_obj = Path(args.video_path)
    directory_path = path_obj.parent

    vid_stream = VideoFile_p(args.video_path)
    #slc = itertools.islice(vid_stream, 2000)
    good = FrameSelection_p.compute_best_frames(vid_stream, vid_stream.number_of_frames, client,
                                                min_variance=0.08, batch_size=20)
    vid_stream.save_frames(good, directory_path / "selected_images")


if __name__ == "__main__":
    main()

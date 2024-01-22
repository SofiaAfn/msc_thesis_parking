
import multiprocessing as mp
import pickle
import line_intersect
import matplotlib.pylab as pl
import time
import h5py
import jaxlie
import numpy as np
import pandas as pd
import cv2
import bisect
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math
from camera import Camera
from functools import partial
from typing import List


class System:
    """
    Args:
        sequence_f (str): Path to the sequence hdf5 file. (The vehicle detection hdf5 file will be inferred from the same directory location)
    """
    image_timestamps: List[int] = None

    sequence_f: h5py.File
    vehicle_detection_f: h5py.File
    detections: np.ndarray
    image_width: int
    image_height: int

    first_timestamp: int = math.inf
    last_timestamp: int = -math.inf
    _pose = None
    camera: Camera

    camera_to_system: jaxlie.SE3 = jaxlie.SE3.from_matrix(
        np.array(
            [[0, -1,  0,  0],
             [0,  0, -1,  0],
             [1,  0,  0,  0],
             [0,  0,  0,  1]]))

    pose_data_size: int

    def __init__(self, sequence_f_path: str):

        self._sequence_f_path = sequence_f_path

        if not os.path.exists(sequence_f_path):
            print("File not found")

            return
        self.sequence_f = h5py.File(sequence_f_path, "r")

        try:

            self.vehicle_detection_f = h5py.File(
                sequence_f_path.replace(".hdf5", '_vehicle_detections.hdf5'), 'r')
        except:
            print("Error opening file, place the vehicle detection file in the same directory as the sequence file")
            exit(1)
        self.pose_data_size = len(self.sequence_f["FusionTimestampedPose"])

        self.image_height = self.sequence_f["YUV420 Images"].attrs["image_height"]
        self.image_width = self.sequence_f["YUV420 Images"].attrs["image_width"]

        # Camera intrinsics
        self.fx = self.sequence_f["CameraModel"].attrs['fx']
        self.fy = self.sequence_f["CameraModel"].attrs['fy']
        self.cx = self.sequence_f["CameraModel"].attrs['cx']
        self.cy = self.sequence_f["CameraModel"].attrs['cy']

        # Distortion coefficients
        self.k1 = self.sequence_f["CameraModel"].attrs['k1']
        self.k2 = self.sequence_f["CameraModel"].attrs['k2']
        self.p1 = self.sequence_f["CameraModel"].attrs['p1']
        self.p2 = self.sequence_f["CameraModel"].attrs['p2']

        self.n_detections = self.vehicle_detection_f['Detection'].shape[0]

        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.camera = Camera(
            K=self.K, camera_coordinate_system=self.camera_to_system)

    def edit_sequence_f(self):
        self.sequence_f.close()
        self.sequence_f = h5py.File(self._sequence_f_path, "r+")
        return self.sequence_f

    def lock_sequence_f(self):
        self.sequence_f.close()
        self.sequence_f = h5py.File(self._sequence_f_path, "r")
        return self.sequence_f

    def get_trajectory(self, start: int = None, end: int = None) -> np.ndarray:
        """
        Returns:
            np.ndarray[3]: X, Y, Z The trajectory of the vehicle in the world coordinate system.

        """
        trajectory = np.array([jaxlie.SE3.exp(pose).translation()
                               for pose in self.sequence_f["FusionTimestampedPose"]["pose"]])
        if start is not None:
            trajectory = trajectory[trajectory["timestamp"] >= start]

        if end is not None:
            trajectory = trajectory[trajectory["timestamp"] <= end]

        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        return x, y, z

    def load_image_timestamps(self):
        # for image in tqdm(self.sequence_f['YUV420 Images']):
        #     self.first_timestamp = min(
        #         self.first_timestamp, image["timestamp"])
        #     self.last_timestamp = max(self.last_timestamp, image["timestamp"])
        #     self.image_timestamps.append(image["timestamp"])
        self.image_timestamps = self.sequence_f['YUV420 Images']['timestamp'][:]
        self.first_timestamp = self.image_timestamps[0]
        self.last_timestamp = self.image_timestamps[-1]

    def detection_histogram_similarity(self, benchmark, comparison):
        """

        Calculates the similarity between two detections based on the cropped bounding box images.
        Args:
            benchmark (np.ndarray): The benchmark detection
            comparison (np.ndarray): The comparison detection

        Returns:
            float: The similarity score between the two detections [0,1]

        """

        benchmark_image = self.get_image_at_timestamp(benchmark["timestamp"])[
            benchmark["top"]:benchmark["bottom"], benchmark["left"]:benchmark["right"]]

        comparison_image = self.get_image_at_timestamp(comparison["timestamp"])[
            comparison["top"]:comparison["bottom"], comparison["left"]: comparison["right"]]
        comparison_image = cv2.resize(
            comparison_image, (benchmark_image.shape[1], benchmark_image.shape[0]))

        H_benchmark = cv2.calcHist([benchmark_image], [0, 1, 2], None, [8, 8, 8], [
            0, 256, 0, 256, 0, 256])
        H_comparison = cv2.calcHist([comparison_image], [0, 1, 2], None, [8, 8, 8], [
            0, 256, 0, 256, 0, 256])

        similarity = cv2.compareHist(
            H_benchmark, H_comparison, cv2.HISTCMP_CORREL)

        return similarity

    def load_detections(self):
        self.detections = np.array(
            self.vehicle_detection_f['Detection'][self.vehicle_detection_f['Detection']["type"] == 29])

    def get_image_at_timestamp(self, query_timestamp):
        if self.image_timestamps is None:
            self.load_image_timestamps()
        image_idx = bisect.bisect_left(self.image_timestamps, query_timestamp)
        assert(self.image_timestamps[image_idx] == query_timestamp)
        image_byte_array_yuv = self.sequence_f['YUV420 Images'][image_idx]["bytes"]

        '''
        YUV420 with interleaved U and V , with uv_row_stride = y_row_stride = image_width.
        For more details look at N12 or N21: https://www.fourcc.org/pixel-format/yuv-nv12/
        '''
        e = self.image_width * self.image_height
        Y = image_byte_array_yuv[0:e]
        Y = np.reshape(Y, (self.image_height, self.image_width))
        V = image_byte_array_yuv[e::2]
        V = np.repeat(V, 2, 0)
        V = np.reshape(V, (int(self.image_height/2), self.image_width))
        V = np.repeat(V, 2, 0)
        U = image_byte_array_yuv[e+1::2]
        U = np.repeat(U, 2, 0)
        U = np.reshape(U, (int(self.image_height/2), self.image_width))
        U = np.repeat(U, 2, 0)

        RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
        RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
        return RGBMatrix

    def get_pose_at_timestamp(self, query_timestamp, lower_bound: int = 0) -> np.ndarray:
        """
        Returns the pose at the given timestamp. If the timestamp is not exact, it will interpolate between the two closest poses.

        :argument query_timestamp: The timestamp to query
        :argument lower_bound: The lower bound for the binary search (default: 0)
        :argument upper_bound: The upper bound for the binary search (default: -1)

        :returns:
        {
            "timestamp": The timestamp of the pose,
            "pose": The pose as a jaxlie se3 object,
            "pose_idx":  The index of the pose in the dataset, (useful for debugging and for lower_bound when using bisect left)

        }
        """

        if self._pose:
            return self._pose[query_timestamp]

        pose_idx = bisect.bisect_left(
            self.sequence_f["FusionTimestampedPose"]["timestamp"], query_timestamp, lo=lower_bound)
        try:
            pose = self.sequence_f["FusionTimestampedPose"][pose_idx]
        except IndexError:
            return None

        # # Return None if the query timestamp is not close enough to the first or last timestamp
        # if abs(pose[0] - query_timestamp) > 100000:  # Works really bad for some reason, what time frame is this? Not ms?
        #     return None

        # Interpolate between poses if timestamp is not exact
        if pose["timestamp"] != query_timestamp and pose_idx > 0 and pose_idx < self.pose_data_size:

            pose_idx -= 1
            pose = self.sequence_f["FusionTimestampedPose"][pose_idx]
            next_pose = self.sequence_f["FusionTimestampedPose"][pose_idx + 1]
            alpha = (query_timestamp - pose["timestamp"]) / \
                (next_pose["timestamp"] - pose["timestamp"])
            pose = {
                "timestamp": query_timestamp,
                "se3": jaxlie.SE3.exp(
                    (1 - alpha) * pose[1] + alpha * next_pose[1]),
                "pose_idx": pose_idx
            }

            return pose

        pose = {
            "timestamp": pose[0],
            "se3": jaxlie.SE3.exp(pose[1]),

            "pose_idx": pose_idx

        }
        return pose

    def precompute_poses(self, saveFile: str = None):
        """
        Precomputes all poses and saves them to a pickle file. Poses does not line up with detection timestmaps, meaning that live lookups are slow"""

        if os.path.exists(saveFile):
            with open(saveFile, "rb") as f:
                self._pose = pickle.load(f)
                return

        detection_timestamps = np.unique(self.detections["timestamp"])
        n_procs = mp.cpu_count()

        batches = np.array_split(detection_timestamps, n_procs)

        with mp.Pool(n_procs) as pool:
            dicts = pool.starmap(_prefetch_batch_worker, zip(
                batches, [self.sequence_f.filename]*n_procs))
            self._pose = {}
            for d in dicts:
                self._pose.update(d)

        self._pose = {timestamp: self.get_pose_at_timestamp(
            timestamp) for timestamp in tqdm(detection_timestamps)}
        if saveFile is not None:
            with open(saveFile, "wb") as f:
                pickle.dump(self._pose, f)

    def get_vehicle_detections_at_timestamp(self, query_timestamp):

        return self.vehicle_detection_f[self.vehicle_detection_f["timestamp"] == query_timestamp]


def _prefetch_batch_worker(batch, sequence_f_path):
    """
    Helper function for precomputing poses"""
    _system = System(sequence_f_path)
    _system.load_detections()
    return {timestamp: _system.get_pose_at_timestamp(
        timestamp) for timestamp in batch}

import numpy as np
from system import System

import sys
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from triangulation import lineIntersect3D
from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial
import time
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Model:
    location: np.ndarray
    # orientation: np.ndarray
    cost: float
    mean_cost: float
    inliers: dict

    def __init__(self, location, cost, mean_cost, inliers) -> None:
        self.location = location
        self.cost = cost
        self.mean_cost = mean_cost
        self.inliers = inliers
        if len(inliers) == 0:
            self.time_window = None
        else:
            self.time_window = [min(inliers.values(), key=lambda x: x.get("data")["timestamp"])["data"]["timestamp"],
                                max(inliers.values(), key=lambda x: x.get("data")["timestamp"])["data"]["timestamp"]]
        pass

    def __str__(self) -> str:
        return f"Model at {self.location} with cost {self.cost} and {len(self.inliers)} inliers"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Model) and np.allclose(self.location, __value.location, atol=0.1)

    def plot_inliers(self):
        """ Plots the inliers of the model in the world view and the camera views"""
        fig = plt.figure("Model world view")
        world_ax = fig.add_subplot(111, projection='3d')
        n_rows = int(np.sqrt(len(self.inliers)))
        n_cols = len(self.inliers) // n_rows + \
            (len(self.inliers) % n_rows > 0)
        n_rows = n_rows if n_rows * \
            n_cols >= len(self.inliers) else n_rows + 1
        fig3, camera_axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, num="Voters", figsize=(15, 15))

        camera_axs = camera_axs.flatten()
        poses = [system.get_pose_at_timestamp(
            inlier["data"]["timestamp"])["se3"] for _, inlier in self.inliers.items()]
        detection_centers = [np.array([d["data"]["left"] + (d["data"]["right"] - d["data"]["left"]) / 2,
                                       d["data"]["top"] + (d["data"]["bottom"] - d["data"]["top"]) / 2]) for _, d in self.inliers.items()]
        rays = system.camera.ray_cast(
            poses, np.array(detection_centers), ray_length=30)
        for ax in camera_axs:
            ax.axis('off')
        for i, (ray, (inlier_uuid, inlier_information)) in enumerate(zip(rays, self.inliers.items())):
            camera_axs[i].imshow(system.get_image_at_timestamp(
                inlier_information["data"]["timestamp"]))
            camera_axs[i].plot(*detection_centers[i], "o",
                               color="red", markersize=2)
            camera_axs[i].plot(*system.camera.project_onto_camera(
                poses[i], self.location), "o", color="green", markersize=2)

            world_ax.plot3D(*ray.reshape(2, 3).T, color='red')

        world_ax.plot3D(*self.location.reshape(1, 3).T,
                        "o", color="blue", markersize=2)
        x, y, z = system.get_trajectory()
        world_ax.plot3D(x, y, z, color='green')
        world_ax.set_xlabel('X Label')
        world_ax.set_ylabel('Y Label')
        world_ax.set_zlabel('Z Label')

        x_size = 30
        y_size = 30
        z_size = 30

        intersection_x = self.location[0]
        intersection_y = self.location[1]
        intersection_z = self.location[2]

        world_ax.set_xlim([intersection_x - x_size, intersection_x + x_size])
        world_ax.set_ylim([intersection_y - y_size, intersection_y + y_size])
        world_ax.set_zlim([intersection_z - z_size, intersection_z + z_size])

        plt.show()


def projection_and_histogram_error(system: System, detections, point,  cap=None, benchmark_detection=None):
    """ Selects inliers and calculates the projection error, in the future also the histogram error"""

    error = 0
    timestamps = np.unique([detection["timestamp"]
                            for detection in detections])
    timestamped_poses = {timestamp: system.get_pose_at_timestamp(
        timestamp)["se3"] for timestamp in timestamps}

    # Remove all detections that are more than 20m away from the point
    detections = np.array([d for d in detections if np.linalg.norm(
        timestamped_poses[d["timestamp"]].translation() - point) < 20])

    # Remove detections where the pose of the camera makes it impossible to see the point
    camera_heading = [t.rotation() @ np.array([0, 0, 1])
                      for t in timestamped_poses.values()]
    camera_to_point = [point - t.translation()
                       for t in timestamped_poses.values()]
    dot_products = [np.dot(c, p)
                    for c, p in zip(camera_heading, camera_to_point)]
    detections = np.array(
        [d for d, dp in zip(detections, dot_products) if dp > 0])

    if len(detections) == 0:
        return np.inf, np.inf, {}

    # Projection error
    detection_centers = np.array([np.array([d["left"] + (d["right"] - d["left"]) / 2,
                                            d["top"] + (d["bottom"] - d["top"]) / 2]) for d in detections])
    poses = np.array([timestamped_poses[detection["timestamp"]]
                      for detection in detections])

    projections = system.camera.project_onto_camera(
        poses, point)

    # 30% error allowed (bounding box center diff with projection center compared to bounding box size)
    allowed_percentage_error = 0.3
    must_hit_area = [{"left": d_center[0] - allowed_percentage_error * (d["right"] - d["left"]),
                      "right": d_center[0] + allowed_percentage_error * (d["right"] - d["left"]),
                      "top": d_center[1] - allowed_percentage_error * (d["bottom"] - d["top"]),
                      "bottom": d_center[1] + allowed_percentage_error * (d["bottom"] - d["top"])
                      } for d, d_center in zip(detections, detection_centers)]

    inliers = np.array([i for box, p, i in zip(must_hit_area, projections, range(
        len(detections))) if box["left"] <= p[0] <= box["right"] and box["top"] <= p[1] <= box["bottom"]])  # Inliers are counted as detections that are within the allowed percentage error of the projection
    if len(inliers) == 0:
        return np.inf, np.inf, {}
    box_hits_centers = detection_centers[inliers]
    box_hits_projections = projections[inliers]
    box_hits_detections = detections[inliers]

    box_hits_areas = np.array([int(d["right"]) - int(d["left"]) * int(d["bottom"]) - int(d["top"])
                               for d in box_hits_detections])

    individual_errors_squared = np.linalg.norm(
        box_hits_centers - box_hits_projections, axis=1)**2 * (1 / box_hits_areas)  # Scale errors by area of detection, so that large detections are not penalized as much

    # Remove duplicate inliers (same timestamp) with higher error
    if len(inliers) != len(np.unique(detections[inliers]["timestamp"])):

        for timestamp in np.unique(detections[inliers]["timestamp"]):
            indexes = np.where(
                detections[inliers]["timestamp"] == timestamp)[0]
            if len(indexes) == 1:
                continue

            # Remove all but the one with the lowest error
            min_error_index = np.argmin(individual_errors_squared[indexes])
            indexes = np.delete(indexes, min_error_index)
            individual_errors_squared = np.delete(
                individual_errors_squared, indexes)
            box_hits_detections = np.delete(box_hits_detections, indexes)

    # Take similarity into account (Histogram error) # Not working

    #  Sort by time
    # _detections = sorted(
    #     detections[inliers], key=lambda x: x["timestamp"])
    # benchmark = _detections[0]
    # inliers = [0]

    # for i in range(1, len(_detections) - 1):
    #     current = _detections[i]
    #     if current["timestamp"] == benchmark["timestamp"]:
    #         continue
    #     else:
    #         similarity = system.detection_histogram_similarity(
    #             benchmark=benchmark, comparison=current)
    #         individual_errors_squared[i] /= similarity * similarity
    #         if similarity > .8:
    #             benchmark = current
    #             inliers.append(i)

    # individual_errors = individual_errors[inliers]

    error = np.sum(individual_errors_squared)

    voters = {f"{detection['uuid'][-1]}": {"error": error, "data": detection} for detection,
              error in zip(box_hits_detections, individual_errors_squared)}
    return error, error/(len(voters)**1.2), voters


def get_valid_samples(system: System, detections, n_sampels: int, show_plots=False):
    """
    Returns a list of sample detections that are valid for RANSAC as well as other detections that are within 10 seconds window for both lower and upper bound to the sample detections.
    """

    samples = []
    first_sample = detections[np.random.randint(
        detections.shape[0])]
    timestamp = first_sample["timestamp"]

    detections_within_ten_seconds = system.detections[
        (system.detections["timestamp"] > timestamp - 10e9) & (system.detections["timestamp"] < timestamp + 10e9)]

    samples.append(first_sample)

    tries = 0
    while True:
        tries += 1
        if len(samples) == n_sampels:
            break

        if tries > 400:
            try:
                return get_valid_samples(system, detections, n_sampels, show_plots)
            except RecursionError:
                return None, None
        sample = detections_within_ten_seconds[np.random.randint(
            detections_within_ten_seconds.shape[0])]

        # Make sure the sample is not already in the list
        if sample["uuid"][-1] in [s["uuid"][-1] for s in samples]:
            continue

        # Calculate intersection point and verify it is in front of the camera for each sample
        candidates = np.copy(samples)
        candidates = np.append(candidates, sample)
        detection_centers = np.array([[d["left"] + (d["right"] - d["left"]) / 2,
                                       d["top"] + (d["bottom"] - d["top"]) / 2] for d in candidates])

        poses = np.array([system.get_pose_at_timestamp(
            d["timestamp"]) for d in candidates])
        if np.any(poses == None):
            for bad_pose_index in np.where(poses == None)[0]:
                detection_centers = np.delete(
                    detection_centers, bad_pose_index, axis=0)

        se3 = [pose["se3"] for pose in poses if pose is not None]
        rays = system.camera.ray_cast(se3, detection_centers)
        intersection = lineIntersect3D(rays[:, :3], rays[:, 3:])

        # Make sure the intersection is in front of the camera, for each sample
        vehicle_heading = [s.rotation() @ np.array([1, 0, 0]) for s in se3]
        camera_to_point = [intersection - s.translation() for s in se3]
        products = [np.dot(vh, ctp)
                    for vh, ctp in zip(vehicle_heading, camera_to_point)]
        if np.any(np.array(products) < 0):
            continue

        # Make sure the intersection is not too far away from the camera

        distance = [np.linalg.norm(
            s.translation() - intersection) for s in se3]
        if np.any(np.array(distance) > 20):
            continue

        # Make sure the intersection is not too close to the camera
        if np.any(np.array(distance) < 3):
            continue

        # Make sure the objects have similar enough histograms
        similarity = system.detection_histogram_similarity(
            benchmark=samples[0], comparison=sample)

        if similarity < .8:
            continue

        # # Compare the area of the bounding boxes
        # for s in samples:
        #     area = (int(s["right"]) - int(s["left"])) * \
        #         (int(s["bottom"]) - int(s["top"]))
        #     sample_area = (int(sample["right"]) - int(sample["left"])) * \
        #         (int(sample["bottom"]) - int(sample["top"]))
        #     threshold = .5
        #     if area / sample_area > 1 + threshold or area / sample_area < 1 - threshold:
        #         break
        # else:

        # Check that the projection of the point is near the center of the detection for each sample
        detection_centers = np.array([[d["left"] + (d["right"] - d["left"]) / 2,
                                       d["top"] + (d["bottom"] - d["top"]) / 2] for d in samples])
        se3 = np.array([system.get_pose_at_timestamp(
            d["timestamp"])["se3"] for d in samples])
        projections = system.camera.project_onto_camera(se3, intersection)
        distances = np.linalg.norm(detection_centers - projections, axis=1)
        if np.any(distances > 70):
            continue

        # Calculate cosine similarity between the samples,
        vectors = rays[:, :3] - rays[:, 3:]
        similaity = cosine_similarity(vectors)
        if not np.any(similaity < .99):
            return get_valid_samples(system, detections, n_sampels, show_plots)

        # Sample seems to fit all criteria
        samples.append(sample)

        if show_plots and len(samples) == n_sampels:

            world = plt.figure("World")
            world_ax = world.add_subplot(111, projection='3d')
            for r in rays:
                world_ax.plot3D(*r.reshape(-1, 3).T, color='red')
            x_size = 30
            y_size = 30
            z_size = 30
            x, y, z = system.get_trajectory()
            world_ax.plot3D(x, y, z, color='blue')

            intersection_x = intersection[0]
            intersection_y = intersection[1]
            intersection_z = intersection[2]

            world_ax.plot3D(intersection_x, intersection_y,
                            intersection_z, 'o', color='green')
            world_ax.set_xlim(
                [intersection_x - x_size, intersection_x + x_size])
            world_ax.set_ylim(
                [intersection_y - y_size, intersection_y + y_size])
            world_ax.set_zlim(
                [intersection_z - z_size, intersection_z + z_size])

            fig, axs = plt.subplots(1, len(samples), num="Samples")
            samples = sorted(samples, key=lambda x: x["timestamp"])
            projections = system.camera.project_onto_camera([system.get_pose_at_timestamp(
                d["timestamp"])["se3"] for d in samples], intersection)

            axs = axs.flatten()
            for i, s in enumerate(samples):
                image = system.get_image_at_timestamp(s["timestamp"])
                detection_center = np.array([s["left"] + (s["right"] - s["left"]) / 2,
                                             s["top"] + (s["bottom"] - s["top"]) / 2])
                axs[i].imshow(image)
                axs[i].plot(detection_center[0],
                            detection_center[1], 'o', color='red', markersize=2)
                axs[i].plot(projections[i][0],
                            projections[i][1], 'o', color='green', markersize=2)

                axs[i].axis('off')

            # image = system.get_image_at_timestamp(sample["timestamp"])
            # detection_center = np.array([sample["left"] + (sample["right"] - sample["left"]) / 2,
            #                              sample["top"] + (sample["bottom"] - sample["top"]) / 2])
            # axs[-1].imshow(image)
            # axs[-1].plot(detection_center[0],
            #              detection_center[1], 'o', color='blue')
            # axs[-1].axis('off')
            plt.show()

    # Retrieve reasonable other candidates ( +- 10 seconds)
    time_period = 10e9
    lower_bound = min([s["timestamp"] for s in samples]) - time_period
    upper_bound = max([s["timestamp"] for s in samples]) + time_period
    detections = system.detections[
        (system.detections["timestamp"] > lower_bound) & (system.detections["timestamp"] < upper_bound)]
    return samples, detections


def RANSAC(sequence_f_path: str, detections, n_iterations, n_samples, show_plots=False):
    """
    RANSAC algorithm for estimating the vehicle's position
    """

    system = System(sequence_f_path)
    system.load_detections()
    system.precompute_poses(saveFile="poses.pickle")

    models: List[Model] = []

    for i in tqdm(range(n_iterations)):

        # samples = system.detections[np.random.choice(
        #     system.detections.shape[0], n_samples, replace=False)]
        samples, detections = get_valid_samples(
            system, detections, n_samples, show_plots)
        if samples is None:
            continue

        poses = [system.get_pose_at_timestamp(
            d["timestamp"])["se3"] for d in samples]

        detection_centers = [np.array([d["left"] + (d["right"] - d["left"]) / 2,
                                       d["top"] + (d["bottom"] - d["top"]) / 2]) for d in samples]
        rays = system.camera.ray_cast(
            poses, np.array(detection_centers), ray_length=30)

        # Calculate the model (3D intersection point)
        intersection = lineIntersect3D(rays[:, :3], rays[:, 3:])

        # Calculate the error
        error, mean_error, voters = projection_and_histogram_error(
            system, detections,  intersection)

        if len(voters) > 8:
            model = Model(intersection, error, mean_error, voters)

            models.append(model)

    return models


def filter_models(models: List[Model], plot_result_each_iteration=False):
    """
    Compares models and returns the best model for each triangulated point,
    as well as makes sure that one detections is only used for one triangulated point.

    It also makes sure that the inliers are not to similar to each other
    """

    best_models: List[Model] = []

    for model in models:
        if len(best_models) == 0:
            best_models.append(model)
            continue

        for best_model in reversed(best_models):
            # Check if the models share inliers
            if len(model.inliers.keys() & best_model.inliers.keys()) > 0:
                if model.mean_cost < best_model.mean_cost:

                    best_models.remove(best_model)
                    best_models.append(model)

                break

        else:

            best_models.append(model)

    if show_plots or plot_result_each_iteration:
        for model in best_models:
            model.plot_inliers()

    return best_models


if __name__ == "__main__":

    # debug
    show_plots = False
    plot_result_each_iteration = True

    # Setup data
    hdf_file_path = sys.argv[1]
    if not os.path.exists(hdf_file_path):
        print("File does not exist")
        sys.exit(1)

    output_file_path = sys.argv[2]
    if os.path.exists(output_file_path):
        print("File alread exists, overwrite? (y/n)")
        if input() != "y":
            sys.exit(1)
    # Load the data
    system = System(hdf_file_path)
    system.load_detections()

    # TODO: Check how we handle detection file, if detections are stored separately or in the same file

    # Precompute poses for faster access,
    system.precompute_poses(saveFile="poses.pickle")

    max_iterations = 50
    n_samples = 2
    n_procs = mp.cpu_count()
    use_mp = True
    if use_mp and max_iterations < n_procs:
        max_iterations = n_procs

    models: List[Model] = []
    detections = system.detections

    # Remove detections that are too close to each other
    detection_timestamps = np.unique(detections["timestamp"])
    poses = [system.get_pose_at_timestamp(
        t) for t in detection_timestamps]
    ok_timestamps = [poses[0]["timestamp"]]
    last = poses[0]
    distance_threshold = 1
    for pose in poses[1:]:

        if pose is None:
            continue
        if np.linalg.norm(pose["se3"].translation() - last["se3"].translation()) > distance_threshold:
            ok_timestamps.append(pose["timestamp"])
            last = pose

    detections = detections[np.isin(detections["timestamp"], ok_timestamps)]

    while True:

        # Ransac

        best_model = None
        best_inliers = None
        best_error = np.inf
        start = time.time()
        print(f"{len(detections)} detections left ")

        if use_mp:
            iterations_per_process = [len(it) for it in np.array_split(
                list(range(max_iterations)), n_procs)]
            with mp.Pool(n_procs) as pool:
                results = pool.starmap(
                    RANSAC, [(hdf_file_path, detections, n_iter,  n_samples) for n_iter in iterations_per_process])
            results = [item for sublist in results for item in sublist]
            models.extend(results)
            models = filter_models(models, plot_result_each_iteration)

            # Remove inliers from detections
            for model in models:
                inliers = [d["data"] for _, d in model.inliers.items()]
                detections = detections[~np.isin(detections, inliers)]

        else:

            result = RANSAC(hdf_file_path, detections,
                            max_iterations, n_samples)
            print(f"Ransac took {(time.time() - start):.2f} seconds")
            models.extend(result)
            models = filter_models(models, plot_result_each_iteration)

            # Remove inliers from detections
            for model in models:
                inliers = [d["data"] for _, d in model.inliers.items()]
                detections = detections[~np.isin(detections, inliers)]

        if True:
            fig = plt.figure("Result")
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = system.get_trajectory()
            ax.plot3D(x, y, z, color='blue')
            for model in models:
                ax.plot3D(*model.location.reshape(1, 3).T, "o")

            ax.set_xlabel('X ')
            ax.set_ylabel('Y ')
            ax.set_zlabel('Z ')
            ax.set_zlim([-50, 50])

            plt.show()

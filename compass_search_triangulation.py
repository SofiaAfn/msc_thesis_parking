from line_intersect import lineIntersect3D
from vehicle import System
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pylab as pl
import pickle
import jaxlie
import multiprocessing as mp
import time
from typing import List



def compass_search(detections, starting_point, cutoff_step_size=.25, step_size=1, error=None):

    # Get the initial error
    if error is None:
        error = projection_error(detections, starting_point)

    search_directions = np.eye(3) * step_size
    search_directions = np.concatenate([search_directions, -search_directions])
    search_directions = np.concatenate(
        [search_directions, np.array([[1, 1, 1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[-1, -1, -1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[1, -1, 1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[-1, 1, -1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[1, -1, -1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[-1, 1, 1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[-1, -1, 1]])])
    search_directions = np.concatenate(
        [search_directions, np.array([[1, 1, -1]])])
    search_directions = search_directions * step_size

    # Start the search
    while True:
        # Move the point in all directions
        for direction in search_directions:
            new_point = starting_point + direction
            new_error = projection_error(detections, new_point, cap=error)

            # If the error is lower, move in that direction
            if new_error < error * .98:  # Has to be more than 2% better for performance, as an end condition
                print("Found a better point")
                return compass_search(detections, new_point,
                                      cutoff_step_size, step_size, new_error)
        else:
            if step_size / 2 < cutoff_step_size:
                return starting_point, error
            print("Halving step size")
            return compass_search(detections, starting_point,
                                  cutoff_step_size, step_size / 2, error)


def compare_detection_centers_with_projections(detections, point, sys: System):

    n_rows = int(np.sqrt(len(detections)))
    n_cols = len(detections) // n_rows + (len(detections) % n_rows > 0)
    n_rows = n_rows if n_rows * \
        n_cols >= len(detections) else n_rows + 1
    fig3, axs = plt.subplots(nrows=n_rows, ncols=n_cols, num="Voters", subplot_kw={
                             'xticks': [], 'yticks': []})

    axs = axs.flatten()
    fig3.set_size_inches(10, 10)

    for i, detection in enumerate(detections):

        se3 = sys.get_pose_at_timestamp(detection['timestamp'])["se3"]
        detection_center = np.array([detection["left"] + (detection["right"] - detection["left"]) / 2,
                                    detection["top"] + (detection["bottom"] - detection["top"]) / 2])
        projection_center = sys.camera.project_onto_camera(
            se3, intersection)
        axs[i].imshow(sys.get_image_at_timestamp(
            detection['timestamp']))
        axs[i].plot(detection_center[0],
                    detection_center[1], 'ro', markersize=5)
        axs[i].axis('off')

        if detection["timestamp"] in sample["timestamp"]:
            axs[i].set_title(
                f"{'Green' if detection['timestamp'] == sample[0]['timestamp'] else 'Orange'} sample")

        if projection_center[0] > sys.image_width or projection_center[0] < 0 or projection_center[1] > sys.image_height or projection_center[1] < 0:
            axs[i].set_title("Projection outside image")
            continue

        vehicle_heading = se3.rotation() @ np.array([1, 0, 0])

        if np.dot(vehicle_heading, intersection - se3.translation()) < 0:

            axs[i].set_title("Behind vehicle")
            continue

        axs[i].plot(projection_center[0], projection_center[1], "bo")


if __name__ == "__main__":

    # load matched ids
    matched_uuids = pickle.load(
        open('/Users/emanuelwreeby/Plugg/Terminer/Exjobb/src/matched_uuids.pkl', 'rb'))[10:]

    # load system
    sys = System(
        "/Users/emanuelwreeby/Plugg/Terminer/Exjobb/src/data/univrses/record_2022-04-22_07-17-55.hdf5")

    n_procs = mp.cpu_count()

    # Ransac parameters
    n_iterations = 30
    n_samples = 2
    threshold = 0.1

    scaling_factor = 1.0
    n_matched_objects = len(matched_uuids)

    # load detections
    sys.load_detections()

    # Filter out detections with matching uuids
    matched_detections = sys.detections[np.isin(
        sys.detections['uuid'][:, -1], [id for uids in matched_uuids for id in uids])]

    x, y, z = sys.get_trajectory()

    for vehicle_nr, uuids in enumerate(tqdm(matched_uuids)):
        start = time.time()
        fig = plt.figure("World")
        ax = fig.add_subplot(111, projection='3d')
        if len(uuids) <= 10:
            continue

        detections = matched_detections[np.isin(
            matched_detections['uuid'][:, -1], uuids)]

        # # Plot all detections in the same color
        # for d in detections:
        #     pose = sys.get_pose_at_timestamp(d['timestamp'])
        #     center = np.array([d["left"] + (d["right"] - d["left"]) /
        #                       2, d["top"] + (d["bottom"] - d["top"]) / 2])
        #     A = sys.camera.ray_cast(
        #         pose["se3"], center, ray_length=30)

        attempts = 0
        max_attempts_to_find_inliers = 200

        # Sample valid points
        while True:
            if attempts > max_attempts_to_find_inliers:
                break
            attempts += 1

            sample = detections[np.random.choice(
                len(detections), n_samples, replace=False)]

            pose_A = sys.get_pose_at_timestamp(sample['timestamp'][0])
            pose_B = sys.get_pose_at_timestamp(sample['timestamp'][1])

            # Make sure the poses of the detections are not too close to eachother
            if np.linalg.norm(pose_A["se3"].translation() - pose_B["se3"].translation()) < 2:
                # print("Samples too close to eachother")
                continue
            else:
                center_A = np.array([sample[0]["left"] + (sample[0]["right"] - sample[0]["left"]) /
                                     2, sample[0]["top"] + (sample[0]["bottom"] - sample[0]["top"]) / 2])
                center_B = np.array([sample[1]["left"] + (sample[1]["right"] - sample[1]["left"]) /
                                    2, sample[1]["top"] + (sample[1]["bottom"] - sample[1]["top"]) / 2])

                # Ray cast
                A = sys.camera.ray_cast(
                    pose_A["se3"], center_A, ray_length=30)
                B = sys.camera.ray_cast(
                    pose_B["se3"], center_B, ray_length=30)

                lines = np.array([A, B]).reshape(-1, 6)
                # print(lines)

                intersection = lineIntersect3D(lines[:, :3], lines[:, 3:])

                # Make sure that the intersection is ahead of both cameras
                A_camera_to_intersection = intersection - \
                    pose_A["se3"].translation()
                B_camera_to_intersection = intersection - \
                    pose_B["se3"].translation()

                A_vehicle_heading = pose_A["se3"].rotation(
                ) @ np.array([1, 0, 0])
                B_vehicle_heading = pose_B["se3"].rotation(
                ) @ np.array([1, 0, 0])

                if np.dot(A_vehicle_heading, A_camera_to_intersection) < 0 or np.dot(B_vehicle_heading, B_camera_to_intersection) < 0:
                    continue

                if np.linalg.norm(pose_A["se3"].translation() - intersection) < 1 or np.linalg.norm(pose_B["se3"].translation() - intersection) < 1:
                    # print("Intersection too close to camera")
                    continue
                else:
                    break

        # print(sample["uuid"])
        if attempts > max_attempts_to_find_inliers:
            print("No valid samples found")
            continue

        # Plot the two sampled rays and the intersection point

        # Perform compass search starting with this intersection point
        resulting_point, error = compass_search(
            detections, intersection, cutoff_step_size=.1, step_size=1)

        print(f"Error: {error}")
        compare_detection_centers_with_projections(
            detections, resulting_point, sys)
        print(f"Compass search took {(time.time() - start):2f} seconds")

        ax.plot3D(*A.reshape(-1, 3).T, 'green')
        ax.plot3D(*B.reshape(-1, 3).T, 'orange')
        ax.plot3D(*intersection, 'o', c="orange", label="First intersection")
        ax.plot3D(x, y, z, 'b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim([-50, 50])
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])

        # Plot the resulting point
        ax.plot3D(*resulting_point.T, 'bo', label="resulting point")
        # Compute error
        ax.legend()
        plt.show()
        plt.close("all")

    plt.show()

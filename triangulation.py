from line_intersect import lineIntersect3D
from system import System
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pylab as pl
import pickle
import jaxlie
import multiprocessing as mp
import time


if __name__ == "__main__":

    # load matched ids
    matched_uuids = pickle.load(
        open('/Users/emanuelwreeby/Plugg/Terminer/Exjobb/src/matched_uuids.pkl', 'rb'))[1:10]

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Filter out detections with matching uuids
    matched_detections = sys.detections[np.isin(
        sys.detections['uuid'][:, -1], [id for uids in matched_uuids for id in uids])]

    x, y, z = sys.get_trajectory()
    plt.ion()

    for vehicle_nr, uuids in enumerate(tqdm(matched_uuids)):
        start = time.time()
        if len(uuids) <= 10:
            continue

        detections = matched_detections[np.isin(
            matched_detections['uuid'][:, -1], uuids)]

        # Plot all detections in the same color
        for d in detections:
            pose = sys.get_pose_at_timestamp(d['timestamp'])
            center = np.array([d["left"] + (d["right"] - d["left"]) /
                              2, d["top"] + (d["bottom"] - d["top"]) / 2])
            A = sys.camera.ray_cast(
                pose["se3"], center, ray_length=30)

        # # Plot detections
        # n_rows = int(np.sqrt(len(detections)))
        # n_cols = len(detections) // n_rows + (len(detections) % n_rows > 0)
        # n_rows = n_rows if n_rows * \
        #     n_cols >= len(detections) else n_rows + 1
        # fig3, axs = plt.subplots(nrows=n_rows, ncols=n_cols,)
        # axs = axs.flatten()
        # fig3.set_size_inches(10, 10)

        # for i, d in enumerate(detections):
        #     image = sys.get_image_at_timestamp(d['timestamp'])
        #     axs[i].imshow(image)
        #     detection_center = np.array([d["top"] + (d["bottom"] - d["top"]) / 2,
        #                                  d["left"] + (d["right"] - d["left"]) / 2])
        #     axs[i].plot(detection_center[1],
        #                 detection_center[0], 'o', color='red')

        #     axs[i].axis('off')

        # for a in axs[-n_rows * n_cols - len(detections):]:
        #     a.axis('off')

        # plt.show()
        # continue

        # Ransac
        best_model = None
        best_inliers = None
        best_error = np.inf

        max_attempts_to_find_inliers = 200

        for i in range(n_iterations):
            attempts = 0
            plt.close("Voters")

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
                    if np.linalg.norm(pose_A["se3"].translation() - intersection) < 1 or np.linalg.norm(pose_B["se3"].translation() - intersection) < 1:
                        # print("Intersection too close to camera")
                        continue
                    else:
                        break

            # print(sample["uuid"])
            if attempts > max_attempts_to_find_inliers:
                continue

            # Plot rays and intersection
            fig2 = plt.figure("Debug")
            fig2.clear()
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.plot3D(intersection[0], intersection[1], intersection[2], 'o')
            ax2.plot3D(*A.reshape(2, 3).T, 'g')
            ax2.plot3D(*B.reshape(2, 3).T, 'orange')
            ax2.plot3D(x, y, z, 'b')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_zlim([-50, 50])
            ax2.set_xlim([-50, 50])
            ax2.set_ylim([-50, 50])

            n_rows = int(np.sqrt(len(detections)))
            n_cols = len(detections) // n_rows + (len(detections) % n_rows > 0)
            n_rows = n_rows if n_rows * \
                n_cols >= len(detections) else n_rows + 1
            fig3, axs = plt.subplots(nrows=n_rows, ncols=n_cols, num="Voters")

            axs = axs.flatten()
            fig3.set_size_inches(10, 10)

            for a in axs:
                a.clear()
                a.axis('off')

            error = 0
            # Project intersection onto the image plane of each camera location and compute overall error
            for j, detection in enumerate(detections):

                if error > best_error:  # No point in continuing if we already have a better model
                    break

                se3 = sys.get_pose_at_timestamp(detection['timestamp'])["se3"]
                detection_center = np.array([detection["left"] + (detection["right"] - detection["left"]) / 2,
                                            detection["top"] + (detection["bottom"] - detection["top"]) / 2])
                projection_center = sys.camera.project_onto_camera(
                    se3, intersection)
                axs[j].imshow(sys.get_image_at_timestamp(
                    detection['timestamp']))
                axs[j].plot(detection_center[0], detection_center[1], 'ro')
                if detection["timestamp"] in sample["timestamp"]:
                    axs[j].set_title(
                        f"{'Green' if detection['timestamp'] == sample[0]['timestamp'] else 'Orange'} sample")

                if projection_center[0] > sys.image_width or projection_center[0] < 0 or projection_center[1] > sys.image_height or projection_center[1] < 0:
                    error += 4e3
                    # print("Projection outside image")
                    continue

                vehicle_heading = se3.rotation() @ np.array([1, 0, 0])

                if np.dot(vehicle_heading, intersection - se3.translation()) < 0:
                    error += 4e3
                    # print("Behind vehicle")
                    continue

                # => The intersection is in front of the vehicle and within the cameras field of view, draw the projected point
                axs[j].plot(projection_center[0], projection_center[1], 'bo')
                error += np.linalg.norm(detection_center - projection_center)

            if error < best_error:
                print(f"New best error: {error}")
                plt.show()
                plt.pause(4)

                best_error = error
                best_model = intersection
                best_inliers = detections

        print(f"Pure ransac took {(time.time() - start):2f} seconds")

        if best_model is not None:
            ax.plot3D(best_model[0], best_model[1], best_model[2],
                      'o', label=f"Detection {vehicle_nr}")
            plt.figure(f"Vehicle {vehicle_nr}")
            plt.imshow(sys.get_image_at_timestamp(
                best_inliers[0]['timestamp']))

    plt.ioff()
    plt.close("Voters")
    ax.plot3D(x, y, z, 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([-50, 50])
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    plt.legend()

    plt.show()

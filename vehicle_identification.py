import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from system import System
import cv2
from tqdm import tqdm

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'top', 'bottom', 'left', 'right'}
    bb2 : dict
        Keys: {'top', 'bottom', 'left', 'right'}


    Returns
    -------
    float
        in [0, 1]
    """

    assert bb1['left'] < bb1['right']
    assert bb1['top'] < bb1['bottom']
    assert bb2['left'] < bb2['right']
    assert bb2['top'] < bb2['bottom']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['left'], bb2['left'])
    y_top = max(bb1['top'], bb2['top'])
    x_right = min(bb1['right'], bb2['right'])
    y_bottom = min(bb1['bottom'], bb2['bottom'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = float(x_right - x_left) * float(y_bottom - y_top)

    b1_left = float(bb1['left'])
    b1_right = float(bb1['right'])
    b1_top = float(bb1['top'])
    b1_bottom = float(bb1['bottom'])

    b2_left = float(bb2['left'])
    b2_right = float(bb2['right'])
    b2_top = float(bb2['top'])
    b2_bottom = float(bb2['bottom'])

    # compute the area of both AABBs
    bb1_area = (b1_right - b1_left) * (b1_bottom - b1_top)
    bb2_area = (b2_right - b2_left) * (b2_bottom - b2_top)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


system = System(
    "/Users/emanuelwreeby/Plugg/Terminer/Exjobb/src/data/univrses/record_2022-04-22_07-17-55.hdf5")
system.load_detections()
n_detected_vehicles = 0

tracked_vehicles = []
matched_uuids = []
start = 692563522169
detections = system.detections[system.detections["timestamp"] > start][:3000]


# # create OpenCV video writer
# video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(
#     *"FMP4"), float(24), (1000, 1000))


def plot_vehicles_at_timestamp(timestamp: str):
    image = system.get_image_at_timestamp(timestamp)
    plt.close()
    fig = plt.figure(figsize=(10, 10))
    plt.title(timestamp)
    plt.imshow(image)
    for vehicle in tracked_vehicles:
        # If it is more than one second since we last saw this vehicle
        if timestamp - vehicle["latest_detection"] > 1e9 or vehicle["missing"] > 1:

            continue
        color = "red"
        plt.gca().add_patch(plt.Rectangle((vehicle["last_position"]["left"], vehicle["last_position"]["top"]), vehicle["last_position"]["right"] -
                                          vehicle["last_position"]["left"], vehicle["last_position"]["bottom"] - vehicle["last_position"]["top"], fill=False, color=color))
        # Add vehicle number above bounding box
        plt.text(vehicle["last_position"]["left"], vehicle["last_position"]["top"] - 10,
                 f"Vehicle {vehicle['vehicle_nr']}", color=color)

    # canvas = FigureCanvas(fig)
    # canvas.draw()
    # mat = np.array(canvas.renderer._renderer)
    # mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # video.write(mat)


def compute_image_color_histogram(img, title: str = "Histogram", show: bool = False):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    red, green, blue = red.flatten(), green.flatten(), blue.flatten()

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title(title)

        ax[1].hist(red, bins=256, color="red", alpha=0.5)
        ax[1].hist(green, bins=256, color="green", alpha=0.4)
        ax[1].hist(blue, bins=256, color="blue", alpha=0.3)

        plt.show()

    return red, green, blue


last_timestamp = None

indexes_to_remove = []

for detection in tqdm(detections):
    if detection["timestamp"] != last_timestamp:
        # plot_vehicles_at_timestamp(detection["timestamp"])
        last_timestamp = detection["timestamp"]
        for i, _ in enumerate(tracked_vehicles):
            tracked_vehicles[i]["missing"] += 1

    placed = False
    for index in sorted(indexes_to_remove, reverse=True):
        uuids = tracked_vehicles[index]["detection_uuids"]
        if len(uuids) > 6:
            matched_uuids.append(uuids)

        del tracked_vehicles[index]
    indexes_to_remove = []
    for i in range(len(tracked_vehicles)):
        vehicle = tracked_vehicles[i]
        time_since_last_seen = detection["timestamp"] - \
            vehicle["latest_detection"]

        # If it is more than 1.5 seconds since we last saw this vehicle, remove it from tracked vehicles
        if time_since_last_seen > 1.5e9 or vehicle["missing"] > 20:
            indexes_to_remove.append(i)
            continue

        iou = get_iou(detection[["top", "bottom", "left", "right"]],
                      vehicle["last_position"][["top", "bottom", "left", "right"]])

        detection_center = np.array(
            [(detection["left"] + detection["right"]) / 2, (detection["top"] + detection["bottom"]) / 2], dtype=np.float32)
        vehicle_center = np.array([(vehicle["last_position"]["left"] + vehicle["last_position"]["right"]) / 2,
                                  (vehicle["last_position"]["top"] + vehicle["last_position"]["bottom"]) / 2], dtype=np.float32)

        # Has to have been seen in the last second
        if iou >= 1 and detection["timestamp"] - vehicle["latest_detection"] < 1e9:
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # plt.gca().add_patch(plt.Rectangle((detection["left"], detection["top"]), detection["right"] -
            #                                   detection["left"], detection["bottom"] - detection["top"], fill=False, color="red"))
            # plt.gca().add_patch(plt.Rectangle((vehicle["last_position"]["left"], vehicle["last_position"]["top"]), vehicle["last_position"]["right"] -
            #                                   vehicle["last_position"]["left"], vehicle["last_position"]["bottom"] - vehicle["last_position"]["top"], fill=False, color="blue"))
            # # Add vehicle number above bounding box
            # plt.text(detection["left"], detection["top"] - 10,
            #          f"Vehicle {vehicle['vehicle_nr']}", color="red")

            # plt.show()
            # plt.close()

            # Add to vehicle

            vehicle["detection_uuids"].append(detection["uuid"][-1])
            vehicle["last_position"] = detection[[
                "top", "bottom", "left", "right"]]
            vehicle["latest_detection"] = detection["timestamp"]
            placed = True
            vehicle["missing"] = 0
            break
        elif iou > .4 and detection["timestamp"] - vehicle["latest_detection"] != 0:
            candidate_image = system.get_image_at_timestamp(
                detection["timestamp"])
            candidate = candidate_image[int(detection["top"]):int(
                detection["bottom"]), int(detection["left"]):int(detection["right"])]
            benchmark_image = system.get_image_at_timestamp(
                vehicle["latest_detection"])
            benchmark = benchmark_image[int(vehicle["last_position"]["top"]):int(
                vehicle["last_position"]["bottom"]), int(vehicle["last_position"]["left"]):int(vehicle["last_position"]["right"])]
            candidate = cv2.resize(
                candidate, (benchmark.shape[1], benchmark.shape[0]))
            # cv2 image color histogram
            H1 = cv2.calcHist([candidate], [0, 1, 2], None, [8, 8, 8], [
                0, 256, 0, 256, 0, 256])

            H2 = cv2.calcHist([benchmark], [0, 1, 2], None, [8, 8, 8], [
                0, 256, 0, 256, 0, 256])

            # Compare histograms
            similarity = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)
            if similarity > 0.72:

                vehicle["detection_uuids"].append(detection["uuid"][-1])
                vehicle["last_position"] = detection[[
                    "top", "bottom", "left", "right"]]
                vehicle["latest_detection"] = detection["timestamp"]
                placed = True
                vehicle["missing"] = 0
                break

    if not placed:
        # Create new vehicle
        n_detected_vehicles += 1
        tracked_vehicles.append({
            "vehicle_nr": n_detected_vehicles,
            "detection_uuids": [detection["uuid"][-1]],
            "last_position": detection[["top", "bottom", "left", "right"]],
            "latest_detection": detection["timestamp"],
            "missing": 0
        })


with open("matched_uuids.pkl", "wb") as f:
    pickle.dump(matched_uuids, f)


# video.release()
# cv2.destroyAllWindows()

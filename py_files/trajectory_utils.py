# image to trajectory utils

import bisect
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import jaxlie

# import plotly.graph_objects as go


def get_first_last_timestamps(sequence_f):
    first_timestamp = math.inf
    last_timestamp = -math.inf
    image_timestamps = []

    for image in tqdm.tqdm(sequence_f["YUV420 Images"]):
        first_timestamp = min(first_timestamp, image["timestamp"])
        last_timestamp = max(last_timestamp, image["timestamp"])
        image_timestamps.append(image["timestamp"])

    print(first_timestamp, last_timestamp)

    return image_timestamps


def get_image_at_timestamp(sequence_f, query_ts_idx, image_timestamps):
    """_summary_

    Args:
        sequence_f (_type_): _description_
        query_timestamp (_int_): _description_
        image_timestamps (_list_): _description_

    Returns:
        _type_: _description_
    """
    query_timestamp = image_timestamps[query_ts_idx]
    image_idx = bisect.bisect_left(image_timestamps, query_timestamp)
    assert image_timestamps[image_idx] == query_timestamp
    image_byte_array_yuv = sequence_f["YUV420 Images"][image_idx]["bytes"]
    image_width = 1280
    image_height = 720
    """
    YUV420 with interleaved U and V , with
    uv_row_stride = y_row_stride = image_width.
    For more details look at N12 or N21:
    https://www.fourcc.org/pixel-format/yuv-nv12/
     """
    e = image_width * image_height

    Y = image_byte_array_yuv[0:e]
    Y = np.reshape(Y, (image_height, image_width))

    V = image_byte_array_yuv[e::2]
    V = np.repeat(V, 2, 0)

    V = np.reshape(V, (int(image_height / 2), image_width))
    V = np.repeat(V, 2, 0)

    U = image_byte_array_yuv[e + 1 :: 2]
    U = np.repeat(U, 2, 0)

    U = np.reshape(U, (int(image_height / 2), image_width))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)

    return RGBMatrix


def get_matrix_at_timestamp(sequence_f, query_timestamp, image_idx, image_timestamps):
    image_idx = bisect.bisect_left(image_timestamps, query_timestamp)
    assert image_timestamps[image_idx] == query_timestamp
    image_byte_array_yuv = sequence_f["YUV420 Images"][image_idx]["bytes"]
    image_width = 1280
    image_height = 720

    """
    YUV420 with interleaved U and V ,
    with uv_row_stride = y_row_stride = image_width.
    For more details look at N12 or N21:
    https://www.fourcc.org/pixel-format/yuv-nv12/
     """
    e = image_width * image_height

    Y = image_byte_array_yuv[0:e]
    Y = np.reshape(Y, (image_height, image_width))

    V = image_byte_array_yuv[e::2]
    V = np.repeat(V, 2, 0)

    V = np.reshape(V, (int(image_height / 2), image_width))
    V = np.repeat(V, 2, 0)

    U = image_byte_array_yuv[e + 1 :: 2]
    U = np.repeat(U, 2, 0)

    U = np.reshape(U, (int(image_height / 2), image_width))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)

    return RGBMatrix


def get_fused_positions(sequence_f):
    """_summary_

    Args:
        sequence_f (_type_): _description_
    """
    fused_positions = []

    for d in tqdm.tqdm(list(sequence_f["FusionTimestampedPose"])):
        # print(d)
        fused_positions.append(jaxlie.SE3.exp(d[1]).translation())

    fused_positions = np.array(fused_positions)

    return fused_positions


def plot_2D_trajectory(fused_positions):
    # Extract the x and y positions from the poses
    x = fused_positions[:, 0]
    y = fused_positions[:, 1]

    # Plot the 2D trajectory
    plt.plot(x, y)
    plt.xlabel("Lateral position (m)")
    plt.ylabel("Longitudinal position (m)")
    plt.axis("equal")
    plt.show()


def plot_3D_trajectory(fused_positions):
    # Extract the x, y, and z positions from the poses
    x = fused_positions[:, 0]
    y = fused_positions[:, 1]
    z = fused_positions[:, 2]

    # Create a 3D plot of the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Longitudinal position (m)")
    ax.set_zlabel("Vertical position (m)")
    plt.show()


def get_positions_geolocation(sequence_f):
    positions = []
    for d in tqdm.tqdm(list(sequence_f["GeoLocation"])):
        positions.append(
            {
                "timestamp": int(d[0]),
                "latitude": d[1],
                "longitude": d[2],
                "altitude": d[3],
                "speed": d[4],
                "horizontal_accuracy": d[5],
                "vertical_accuracy": d[6],
                "speed_accuracy": d[7],
            }
        )
    return positions

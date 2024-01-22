# utils for car triangulations and permitted parking in Stockholm municipality


import numpy as np
# import pandas as pd
import pyproj
# import plotly.graph_objects as go
from tqdm import tqdm


def convert_crs(triangulations_df):
    """
    convert a point with epsg:4326 to epsg:3011

    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3011")

    x_3011 = []
    y_3011 = []

    for idx, row in triangulations_df.iterrows():
        y, x = transformer.transform(row["latitude"], row["longitude"])
        x_3011.append(x)
        y_3011.append(y)

    triangulations_df["x_3011"] = x_3011
    triangulations_df["y_3011"] = y_3011
    triangulations_df["xy_3011"] = triangulations_df.apply(
        lambda row: [row["x_3011"], row["y_3011"]], axis=1
    )

    return triangulations_df


def parking_proximity(tri_point, parking_json, radius):
    """
    filter permitted parking within the distance of a particular point
    """
    nearby_prkng_ids = []
    for prkng_idx, p in enumerate(parking_json["features"]):
        for coords in p["geometry"][
            "coordinates"
        ]:  # this is the node on the permitted  parking line?
            distance = np.linalg.norm(np.array(tri_point) - np.array(coords))
            if distance < radius:
                nearby_prkng_ids.append(prkng_idx)
                continue  # doing it for all the coords on pp line

    return nearby_prkng_ids


def parking_line_at_index(parking_json, idx):
    parking_line = parking_json["features"][idx]["geometry"]["coordinates"]

    return parking_line


def project_point_on_line_segment(tri_point, parking_line):
    """Calculate the projection of a point on a line segment using numpy."""
    # Define the two points of the line segment
    p1, p2 = np.array(parking_line)

    # Calculate the vector between the two points
    v = p2 - p1

    # Normalize the vector
    v_norm = v / np.linalg.norm(v)

    # Calculate the vector between the first point of the line segment
    # and the point we want to project
    u = tri_point - p1

    # Calculate the dot product of the normalized vector and the
    # vector we just calculated
    dot_product = np.dot(u, v_norm)

    # Multiply the dot product by the normalized vector
    projection = p1 + dot_product * v_norm

    # Check if the projection is within the line segment
    if np.dot(projection - p1, v) < 0:
        projection = p1
    elif np.dot(projection - p2, v) > 0:
        projection = p2

    return projection


def filter_nearby_parking_idx(car_tri_df):
    mask = car_tri_df["nearby_parking_ids"].apply(lambda x: len(x) != 0)
    car_tri_filtered_df = car_tri_df.loc[mask]
    return car_tri_filtered_df

def locate_closest_parking_idx(car_tri_df, parking_json):

    for idx, row in tqdm(car_tri_df.iterrows(),
                         total=len(car_tri_df)):
        tri_point = row["xy_3011"]
        parking_lines_idx_list = row["nearby_parking_ids"]
        # print(parking_lines_idx_list)
        if len(row["nearby_parking_ids"]) == 1:
            # print(idx,row['nearby_parking_ids'])
            car_tri_df.loc[idx, "closest_parking_id"] = row[
                                                        "nearby_parking_ids"
                                                        ][0]
            # break
            # print(idx,row['xy_3011'],row['closest_parking_line'])
        else:
            proj_dict = {}
            for one_parking_line_id in row["nearby_parking_ids"]:
                parking_line_coords = parking_line_at_index(
                    parking_json, one_parking_line_id
                )
                parking_line_p = [parking_line_coords[0],
                                  parking_line_coords[-1]]

                proj = project_point_on_line_segment(
                    tri_point, parking_line_p
                )
                # print(idx,parking_line_coords[0],parking_line_coords[-1])
                proj_dist = [np.linalg.norm(tri_point - proj)]  # list

                proj_dict[one_parking_line_id] = proj_dist
                # print('got proj',proj_dist)
            # print(proj_dict)
            car_tri_df.loc[idx, "closest_parking_id"] = int(round(min(proj_dict,
                                                            key=proj_dict.get)))
            car_tri_df.loc[idx, "all_close_parking_id"] = [proj_dict]
            # print(tri_point,proj_list)

            # print(one_parking_line)
            
    return car_tri_df

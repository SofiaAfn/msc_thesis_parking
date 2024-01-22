import numpy as np
import cv2
import jaxlie
from typing import List


class Camera:
    # Camera intrinsics
    fx: float
    fy: float
    cx: float
    cy: float

    K: np.ndarray

    # Distortion coefficients
    k1: float
    k2: float
    p1: float
    p2: float

    camera_coordinate_system: jaxlie.SE3

    def __init__(self, fx=None, fy=None, cx=None, cy=None, K=None, camera_coordinate_system: jaxlie.SE3 = jaxlie.SE3.from_matrix(np.eye(4))) -> None:
        assert (
            fx is not None and fy is not None and cx is not None and cy is not None) or K is not None
        if K is not None:
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.K = K

        else:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.K_inv = np.linalg.inv(self.K)

        self.camera_coordinate_system = jaxlie.SO3.from_y_radians(
            0) @ camera_coordinate_system.rotation()

    def project_onto_camera(self, se3: jaxlie.SE3 or List[jaxlie.SE3], X: np.ndarray) -> np.ndarray:
        """Project a 3D point onto the camera plane.

        arguments:
            pose: The pose of the camera in SE3
            X: The 3D point in the world


        returns:
            np.ndarray[2]: The 2D point on the camera plane as (x, y) with (0, 0) in the top left corner of the image, x is right, y is down
        """
        # if isinstance(se3, list) or isinstance(se3, np.ndarray):
        #     assert len(se3) == X.shape[0]
        #     if X.shape[1] != 3:
        #         X = np.repeat(X, len(se3), axis=0)

        #     X = X.T
        # else:
        #     assert X.shape[0] == 3

        # camera_coordinates = se3.inverse() @ X
        # system_coordinates = self.camera_coordinate_system @ camera_coordinates
        # pixel_coordinates = self.K @ system_coordinates
        # return (pixel_coordinates / pixel_coordinates[-1])[:-1]

        if isinstance(se3, list) or isinstance(se3, np.ndarray):
            if len(X.shape) > 1:
                assert X[0].shape[0] == 3
                assert len(se3) == X.shape[0]
                camera_coordinates = [s.inverse() @ x for s, x in zip(se3, X)]
            else:

                camera_coordinates = np.array([s.inverse() @ X for s in se3])

            system_coordinates = self.camera_coordinate_system.as_matrix(
            ) @ camera_coordinates.T
            pixel_coordinates = self.K @ system_coordinates
            pixel_coordinates = np.array(
                [(p / p[-1])[:-1] for p in pixel_coordinates.T])
            return pixel_coordinates

        else:
            assert X.shape[0] == 3
            camera_coordinates = se3.inverse() @ X
            system_coordinates = self.camera_coordinate_system.as_matrix(
            ) @ camera_coordinates
            pixel_coordinates = self.K @ system_coordinates
            return (pixel_coordinates / pixel_coordinates[-1])[:-1]

    def ray_cast(self, se3: jaxlie.SE3 or np.ndarray[jaxlie.SE3], x: np.ndarray, ray_length=150, frame="global"):
        """Cast a ray from the camera location into the world

        arguments:
            pose: The pose of the camera in SE3
            x: The 2D point on the camera plane as (x, y) with (0, 0) in the top left corner of the image, x is right, y is down
            ray_length: The distance from the camera to a point in the world that the ray intersects
            frame: The frame of reference, either "global" or "local"

        returns:
            np.ndarray[6]: Two 3D points in the world that the ray intersects, :3 is the start point and 3: is the end point, ray_length away from the start point
        """

        if isinstance(se3, list) or isinstance(se3, np.ndarray):
            assert len(se3) == x.shape[0]
            assert x.shape[1] == 2
            pixel_coordinates = np.c_[x, np.ones(x.shape[0])].T

        else:
            assert x.shape == (2,)
            pixel_coordinates = np.append(x, 1)

        # Pixel to camera coordinates
        camera_coordinates = self.K_inv @ pixel_coordinates

        if isinstance(se3, list) or isinstance(se3, np.ndarray):

            # Pick a point on the ray
            camera_coordinates = np.array(
                [c * ray_length / np.linalg.norm(c) for c in camera_coordinates.T]).T
        else:
            camera_coordinates = camera_coordinates * \
                ray_length / np.linalg.norm(camera_coordinates)

        # The camera has a different coordinate system to how we want to represent the image,
        # x forward, y left, z up translated into x right, y down, z forward (correct?)
        camera_coordinates = self.camera_coordinate_system.inverse(
        ).as_matrix() @ camera_coordinates
        if frame == "local":

            return np.array([np.zeros((3, 1)), camera_coordinates]).reshape(-1, 6)

        if isinstance(se3, list) or isinstance(se3, np.ndarray):
            world_coordinates = np.array(
                [p @ coords for p, coords in zip(se3, camera_coordinates.T)])
            return np.concatenate([[p.translation() for p in se3], world_coordinates], axis=1).reshape(-1, 6)

        world_coordinates = se3 @ camera_coordinates
        return np.concatenate([se3.translation(), world_coordinates]).reshape(-1, 6)

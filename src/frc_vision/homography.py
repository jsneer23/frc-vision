import numpy as np
from numpy import float64
from numpy.typing import NDArray

Point = tuple[float, float]

DEFAULT_K = np.array([[914.17, 0, 667.64], [0, 914.60, 396.03], [0, 0, 1]])
DEFAULT_K.flags.writeable = False


def make_H(  # noqa: N802
    yaw_deg: float, roll_deg: float, distance: float, K: NDArray[float64] = DEFAULT_K
) -> NDArray[float64]:

    # 3d rotation matrix
    yaw = np.deg2rad(yaw_deg)
    roll = np.deg2rad(roll_deg)

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

    R = Rz @ Ry

    # translation
    t = np.array([0, 0, distance])

    # [r1 r2 t] extrinsics matrix
    rrt = np.column_stack((R[:, :2], t))

    # homography
    return K @ rrt


def project(H: NDArray[float64], x: float, y: float) -> Point:
    pixels = [x, y, 1] @ H.T
    uv = pixels[:2] / pixels[2]
    return float(uv[0]), float(uv[1])


def project_all(H: NDArray[float64], points: NDArray[float64]) -> NDArray[float64]:
    # np.array([project(H, x, y) for x, y in points]) is slower approach
    q = np.column_stack((points, np.ones(len(points)))) @ H.T
    # divides all the points by the 3rd column to return the pixel
    return q[:, :2] / q[:, 2:3]


def dlt(tag_pts: NDArray[float64], img_pts: NDArray[float64]) -> NDArray[float64]:
    rows = []
    for (x, y), (u, v) in zip(tag_pts, img_pts, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(rows)

    H_est = np.linalg.svd(A).Vh[-1].reshape(3, 3)

    return H_est

"""Camera geometry: from pixels in your video to centimetres on the floor.

Three facts make this work:
- The camera calibration (calibrate.py) gives the lens: focal length, centre, distortion.
- The A4 sheet at the start of each clip gives where the camera is: its four corners,
  whose real positions we know (29.7 x 21 cm), pin down the camera's position and tilt.
- Each dot sits at a known height (box dots on the box's top, the nail dot about 2 cm up).
  A pixel only tells us the direction of a ray from the camera; where that ray crosses
  the dot's own height is where the dot really is.

The surface frame: origin at the centre of the A4 sheet, x along its long side, y along
its short side, z up, in metres.
"""
import json

import cv2
import numpy as np

A4 = (0.297, 0.210)  # metres


def load_camera(path):
    with open(path) as f:
        cam = json.load(f)
    return np.array(cam["K"]), np.array(cam["dist"])


def a4_corners_3d(long_side_first):
    """The sheet's corners in the surface frame, in the order they are walked round."""
    w, h = (A4[0] / 2, A4[1] / 2) if long_side_first else (A4[1] / 2, A4[0] / 2)
    pts = np.array([[-w, h, 0], [w, h, 0], [w, -h, 0], [-w, -h, 0]], np.float64)
    if not long_side_first:  # keep x along the long side
        pts = pts[:, [1, 0, 2]] * [1, -1, 1]
    return pts


def camera_pose_from_a4(corners_px, K, dist):
    """Camera rotation R and translation t (surface -> camera) from the sheet's 4 corners.

    corners_px: the corners in the image, walked round in order (either direction).
    Which image side is the long one isn't known in advance, so both are tried and
    the one that fits the corners best is kept. Returns R, t and the fit error in pixels.
    """
    corners_px = np.asarray(corners_px, np.float64).reshape(4, 2)
    best = None
    for long_first in (True, False):
        for order in (corners_px, corners_px[::-1]):  # either walking direction
            obj = a4_corners_3d(long_first)
            ok, rvec, tvec = cv2.solvePnP(obj, order, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            R = cv2.Rodrigues(rvec)[0]
            if (-R.T @ tvec)[2, 0] <= 0:  # the camera must be above the floor
                continue
            reproj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(4, 2)
            err = np.abs(reproj - order).max()
            if best is None or err < best[2]:
                best = (R, tvec.reshape(3), err)
    if best is None:
        raise ValueError("no camera pose fits these A4 corners")
    return best


def back_project(pixels, K, dist, R, t, height):
    """Where the rays through `pixels` (N x 2) cross the plane z = height (metres).

    Returns N x 2 surface coordinates (x, y).
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)    # remove the lens distortion
    rays = np.column_stack([norm, np.ones(len(norm))]) @ R          # ray directions, surface frame
    centre = -R.T @ t                                              # camera position, surface frame
    s = (height - centre[2]) / rays[:, 2]
    return (centre + s[:, None] * rays)[:, :2]


def project(points, K, dist, R, t):
    """Pixels of 3D surface-frame points (N x 3), for drawing overlays."""
    rvec = cv2.Rodrigues(R)[0]
    return cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 1, 3),
                             rvec, t.astype(np.float64), K, dist)[0].reshape(-1, 2)


def box_corners_3d(x, y, yaw, size, z):
    """The 4 corners of a size x size square centred at (x, y), turned by yaw, at height z."""
    h = size / 2
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[h, h], [-h, h], [-h, -h], [h, -h]])
    xy = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.column_stack([xy, np.full(4, z)])

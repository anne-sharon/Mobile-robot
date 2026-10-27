"""Marker detection from camera pixels only."""
import json, math, os
import cv2
import numpy as np

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "camera_calibration.json")


def load_camera_calibration(default_K):
    """Camera matrix + distortion from our own checkerboard calibration (eval/calibrate_camera.py).
    Falls back to the datasheet matrix if the calibration file has not been made yet."""
    try:
        with open(CALIBRATION_FILE) as f:
            c = json.load(f)
        return np.array(c["camera_matrix"]), np.array(c["dist_coeffs"]), "checkerboard calibration"
    except (OSError, KeyError, ValueError):
        return np.asarray(default_K, float), np.zeros(5), "datasheet"


def detect_marker(img_rgb, K, marker, dist=None):
    """Find the dock marker in an RGB image.

    K:      3x3 camera matrix
    marker: sensors.calibration["marker"] = dict(dictionary=..., id=..., size=...)  (size = black edge, metres)
    dist:   lens distortion coefficients (None = no distortion)
    Returns None, or (corners 4x2 pixels [TL,TR,BR,BL], rvec, tvec); tvec = marker centre (right, down, forward) [m]
    in the camera frame.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(marker["dictionary"]),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or marker["id"] not in ids.flatten():
        return None
    c = corners[list(ids.flatten()).index(marker["id"])].reshape(4, 2).astype(np.float32)
    s = marker["size"] / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (c, rvec, tvec.ravel()) if ok else None


def marker_in_robot_frame(rvec, tvec, cam_x):
    """Camera frame (x right, y down, z forward) -> robot frame (x forward, y left).
    Returns z = [ahead, left, angle]: marker centre ahead of / left of the robot origin [m], and the direction
    the marker faces (its outward normal) as an angle in the robot frame [rad].  None if the solution is unusable."""
    tvec, rvec = np.asarray(tvec, float).ravel(), np.asarray(rvec, float).ravel()
    if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))) or tvec[2] <= 0:
        return None
    right, _, fwd = tvec
    R = cv2.Rodrigues(rvec)[0]
    nx, ny = R[2, 2], -R[0, 2]                  # marker z-axis (points out of the marker, at the camera)
    if math.hypot(nx, ny) < 1e-6:
        return None
    return np.array([cam_x + fwd, -right, math.atan2(ny, nx)])

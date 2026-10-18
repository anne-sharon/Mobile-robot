"""Marker detection from camera pixels only."""
import cv2
import numpy as np

def detect_marker(img_rgb, K, marker):
    """Find the dock marker in an RGB image.

    K:      3x3 camera matrix (sensors.calibration["camera_matrix"])
    marker: sensors.calibration["marker"] = dict(dictionary=..., id=..., size=...)  (size = black edge, metres)
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
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (c, rvec, tvec.ravel()) if ok else None

"""Where am I, and where is the dock?  Uses only observations (encoders, IMU, camera)."""
import math
import numpy as np
from .contracts import Dock, wrap
from .marker import detect_marker


class Estimator:
    def __init__(self, calibration):
        self.cal = calibration
        self.K = calibration["camera_matrix"]
        self.cam_x = calibration["camera_pos"][0]          # camera is this far ahead of the robot origin
        self.pose = np.zeros(3)                            # [x, y, theta] in the estimator frame
        self.dock = None                                   # Dock, once the marker has been seen
        self.last_sighting = None                          # (t, distance [m]) of the latest marker detection
        self._prev_enc = None
        self._last_img = None
        self._weight = 0.0                                 # how much we trust the current dock estimate

    # ------------------------------------------------------------------ odometry
    def _odometry(self, enc, yaw_rate, t):
        """Distance from the wheel encoders, turning from the gyro.
        (The wheels slip a little when spinning on the spot, so encoder-only heading drifts ~0.1 rad per spin;
        the gyro measures the body's rotation directly.)"""
        if self._prev_enc is None:
            self._prev_enc, self._prev_t, self._prev_rate = enc, t, yaw_rate
            return
        r = self.cal["wheel_radius"]
        dl, dr = (enc[0] - self._prev_enc[0]) * r, (enc[1] - self._prev_enc[1]) * r
        dt = t - self._prev_t
        ds = (dl + dr) / 2
        dth = 0.5 * (yaw_rate + self._prev_rate) * dt       # trapezoidal integration of the gyro
        self._prev_enc, self._prev_t, self._prev_rate = enc, t, yaw_rate
        th_mid = self.pose[2] + dth / 2                    # midpoint integration
        self.pose[0] += ds * math.cos(th_mid)
        self.pose[1] += ds * math.sin(th_mid)
        self.pose[2] = wrap(self.pose[2] + dth)

    # ------------------------------------------------------------------ marker -> dock estimate
    def _marker_update(self, img, t):
        res = detect_marker(img, self.K, self.cal["marker"])
        if res is None:
            return
        _, rvec, tvec = res
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))) or tvec[2] <= 0:
            return                                         # unusable pose solution: ignore this frame
        right, _, fwd = tvec
        # marker centre and its outward normal in the ROBOT frame (x forward, y left)
        mx, my = self.cam_x + fwd, -right
        R = cv2_rodrigues(rvec)
        nx, ny = R[2, 2], -R[0, 2]                         # marker z-axis (points at the camera) -> robot frame
        n = math.hypot(nx, ny)
        if n < 1e-6:
            return
        nx, ny = nx / n, ny / n
        # ... then into the ESTIMATOR frame using the current pose
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        gx, gy = x + c * mx - s * my, y + s * mx + c * my
        gnx, gny = c * nx - s * ny, s * nx + c * ny

        dist = math.hypot(mx, my)
        self.last_sighting = (t, dist)
        w = 1.0 / max(dist, 0.3) ** 2                      # close sightings are much more accurate
        if self.dock is None:
            self.dock, self._weight = Dock(gx, gy, gnx, gny, 1), w
            return
        # older estimates fade, so recent (and closer) sightings win: this also absorbs odometry drift
        self._weight *= 0.7
        a = w / (self._weight + w)
        d = self.dock
        d.x += a * (gx - d.x)
        d.y += a * (gy - d.y)
        vx, vy = d.nx + a * (gnx - d.nx), d.ny + a * (gny - d.ny)
        n = math.hypot(vx, vy)
        if n > 0.1:                                        # skip if the two directions nearly cancel out
            d.nx, d.ny = vx / n, vy / n
        d.sightings += 1
        self._weight += w

    # ------------------------------------------------------------------ main entry
    def update(self, obs):
        self._odometry(obs["encoders"], obs["imu"][0], obs["t"])
        img = obs.get("image")
        if img is not None and img is not self._last_img:   # only process a NEW camera frame once
            self._last_img = img
            self._marker_update(img, obs["t"])
        return self.pose.copy()


def cv2_rodrigues(rvec):
    import cv2
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]

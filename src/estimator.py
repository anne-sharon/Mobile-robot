"""Where am I?  Extended Kalman Filter on the robot pose s = [x, y, theta]. Uses only observations.

Prediction (every 0.05 s): odometry model from src/model.py -- distance from the wheel encoders, rotation from the
    gyro (the wheels slip when turning on the spot, the gyro does not).
Update (every new camera frame that shows the marker): the camera measures the marker's pose in the robot frame,
    z = [ahead, left, angle]; the marker's pose in the dock frame is known (0, 0, 0), so this corrects the robot
    pose directly (measurement model h(s) in src/model.py).

Frames
  * Before the marker is first seen the filter runs in the START frame (origin = where the robot started); only
    prediction is possible, so the covariance P grows (the pose is not observable from odometry alone).
  * On the first sighting the filter re-initialises in the DOCK frame (origin at the marker centre, +x out of the
    marker into the room) from that one measurement, with its covariance mapped from R. From then on every sighting
    is an EKF update. Docking only needs the pose relative to the dock, so this is the frame the planner uses.

Noise parameters (from our own measurements, see eval/characterise_*.py and models/*.json)
  Q: per step  var_ds = (k_ds ds)^2 + floor_ds^2,   var_dth = (k_dth dth)^2 + floor_dth^2,  scaled by Q_SCALE
  R: diag(sigma_ahead(d), sigma_left(d), sigma_angle(d))^2 with sigma = a + b d^2 (d = distance to the marker)
  Camera bias: bias(d) = c d^2 subtracted from every measurement before the update.
  Outliers (planar-marker pose flips): rejected when the Mahalanobis distance exceeds the 99% chi-square gate.
"""
import json, math, os
import numpy as np
from .contracts import Dock, wrap
from .marker import detect_marker, marker_in_robot_frame, load_camera_calibration
from . import model

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")

def _load(name, default):
    try:
        with open(os.path.join(MODELS, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default

ODO = _load("odometry_noise.json", dict(k_ds=0.0, floor_ds=2.2e-4, k_dth=0.067, floor_dth=1.15e-3))
CAM = _load("camera_noise.json", dict(bias_c=dict(ahead=0.0075, left=0.0, angle=0.0),
                                      ahead=dict(a=1e-4, b=0.0042), left=dict(a=1e-4, b=0.0019),
                                      angle=dict(a=0.002, b=0.0074)))

Q_SCALE   = 4.0              # Q inflation: per-step odometry errors are correlated (slip lasts several steps)
R_SCALE   = 16.0             # R inflation: camera errors are correlated frame to frame (pixel quantisation changes
                             # slowly as the robot creeps forward), so each frame carries less new information
                             # than an independent measurement would. Both scales chosen by eval/tune_ekf.py (NEES).
R_FLOOR   = (0.003, 0.003, math.radians(0.5))   # minimum sigma: unmodelled effects (timing, corner jitter)
GATE_CHI2 = 11.34            # 99% point of chi-square with 3 degrees of freedom
REINIT_AFTER = 6             # this many rejected sightings in a row -> the dock-frame fix was wrong, start again


class Estimator:
    def __init__(self, calibration):
        self.cal = calibration
        self.K, self.dist, self.camera_source = load_camera_calibration(calibration["camera_matrix"])
        self.cam_x = calibration["camera_pos"][0]
        self.r, self.b = calibration["wheel_radius"], calibration["wheel_separation"]
        self.q_scale, self.r_scale = Q_SCALE, R_SCALE
        self.pose = np.zeros(3)                 # EKF mean
        self.P = np.zeros((3, 3))               # EKF covariance (start pose is exact by definition)
        self.frame = "start"                    # "start" until the marker is seen, then "dock"
        self.frame_changes = 0
        self.dock = None                        # Dock in the estimator frame (planner uses it)
        self.last_sighting = None               # (t, distance) of the last accepted marker measurement
        self.stats = dict(updates=0, rejected=0, reinits=0)
        self.last_nis = None                    # normalised innovation squared of the last measurement
        self._prev_enc = None
        self._last_img = None
        self._rejected_in_row = 0

    # ------------------------------------------------------------------ noise models
    def _Q(self, ds, dth):
        var_ds = (ODO["k_ds"] * ds) ** 2 + ODO["floor_ds"] ** 2
        var_dth = (ODO["k_dth"] * dth) ** 2 + ODO["floor_dth"] ** 2
        return self.q_scale * np.diag([var_ds, var_dth])

    def _R(self, d):
        s = [max(CAM[k]["a"] + CAM[k]["b"] * d * d, R_FLOOR[i]) for i, k in enumerate(("ahead", "left", "angle"))]
        return self.r_scale * np.diag(np.square(s))

    def _correct_bias(self, z):
        d2 = z[0] ** 2 + z[1] ** 2
        c = CAM["bias_c"]
        return z - np.array([c["ahead"], c["left"], c["angle"]]) * d2

    # ------------------------------------------------------------------ EKF steps
    def predict(self, enc, yaw_rate, t):
        if self._prev_enc is None:
            self._prev_enc, self._prev_t, self._prev_rate = enc, t, yaw_rate
            return
        ds = self.r * ((enc[0] - self._prev_enc[0]) + (enc[1] - self._prev_enc[1])) / 2
        dth = 0.5 * (yaw_rate + self._prev_rate) * (t - self._prev_t)        # trapezoidal gyro integration
        self._prev_enc, self._prev_t, self._prev_rate = enc, t, yaw_rate
        F, G = model.odometry_jacobians(self.pose, ds, dth)
        self.pose = model.odometry_step(self.pose, ds, dth)
        self.pose[2] = wrap(self.pose[2])
        self.P = F @ self.P @ F.T + G @ self._Q(ds, dth) @ G.T

    def _init_dock_frame(self, z, R, t):
        """First sighting (or a re-fix): robot pose in the dock frame straight from the measurement."""
        self.pose = model.pose_from_marker(z)
        J = np.empty((3, 3)); e = 1e-6                                    # Jacobian of the inverse measurement
        for i in range(3):
            dz = np.zeros(3); dz[i] = e
            J[:, i] = (model.pose_from_marker(z + dz) - self.pose) / e
        self.P = J @ R @ J.T
        if self.frame == "dock":
            self.stats["reinits"] += 1
        self.frame, self.frame_changes = "dock", self.frame_changes + 1
        self.dock = Dock(0.0, 0.0, 1.0, 0.0, 1)
        self.last_sighting = (t, float(math.hypot(z[0], z[1])))
        self._rejected_in_row = 0

    def correct(self, z, t):
        """EKF measurement update with a (bias-corrected) marker measurement z = [ahead, left, angle]."""
        d = float(math.hypot(z[0], z[1]))
        R = self._R(d)
        if self.frame != "dock":
            self._init_dock_frame(z, R, t)
            return True
        H = model.marker_jacobian(self.pose)
        nu = z - model.marker_measurement(self.pose)
        nu[2] = wrap(nu[2])
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        self.last_nis = nis
        if nis > GATE_CHI2:                                               # outlier: ignore it
            self.stats["rejected"] += 1
            self._rejected_in_row += 1
            if self._rejected_in_row >= REINIT_AFTER:
                self._init_dock_frame(z, R, t)
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.pose = self.pose + K @ nu
        self.pose[2] = wrap(self.pose[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T                     # Joseph form: keeps P symmetric, PSD
        self.stats["updates"] += 1
        self.dock.sightings += 1
        self.last_sighting = (t, d)
        self._rejected_in_row = 0
        return True

    # ------------------------------------------------------------------ main entry
    def update(self, obs):
        """One control step: predict with odometry, correct with the camera if a new frame shows the marker."""
        self.predict(obs["encoders"], obs["imu"][0], obs["t"])
        img = obs.get("image")
        if img is not None and img is not self._last_img:                 # each camera frame is used once
            self._last_img = img
            res = detect_marker(img, self.K, self.cal["marker"], self.dist)
            if res is not None:
                z = marker_in_robot_frame(res[1], res[2], self.cam_x)
                if z is not None:
                    self.correct(self._correct_bias(z), obs["t"])
        return self.pose.copy()

import numpy as np

class Estimator:
    """Where am I?  Uses only observations. Empty for now: holds the start pose."""
    def __init__(self, calibration):
        self.cal = calibration
        self.pose = np.zeros(3)              # [x, y, theta] in the estimator frame

    def update(self, obs):
        # TODO: odometry from obs["encoders"] / obs["imu"]; correct with marker sightings from obs["image"]
        return self.pose.copy()

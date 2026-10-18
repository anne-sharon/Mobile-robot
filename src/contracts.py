"""Shared conventions for the robot-side files: estimator.py, planner.py, control.py, marker.py.

Estimator frame: origin = where the robot started, +x = start heading, +y = left, z up.
Pose = np.array([x, y, theta]); metres, metres, radians; theta CCW from +x, wrapped to [-pi, pi).
Robot frame:  x forward, y left (origin at the wheel-axle midpoint).
Camera frame (OpenCV): x right, y down, z forward.   Camera sits 0.04 m ahead of the robot origin.
Marker in robot frame from a camera measurement tvec=(right, down, forward):  x = 0.04 + forward, y = -right.

Observation dict handed to the robot each tick:
  t (s) | encoders (left, right) cumulative wheel angles [rad] | imu (yaw_rate, accel[3])
  lidar: 360 ranges [m], index i = i deg CCW from the front, inf = no return, or None between reads
  image: 480x640x3 RGB uint8, or None between reads
"""
from dataclasses import dataclass
import math

CONTROL_DT = 0.05          # 20 Hz control loop
SENSE_EVERY = 4            # LiDAR + camera every 4th tick = 5 Hz

def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi

@dataclass
class Plan:
    """What the planner hands to the controller."""
    mode: str = "idle"                 # idle | search | approach | dock | done
    target_xy: tuple | None = None     # in the estimator frame, or None
    done: bool = False

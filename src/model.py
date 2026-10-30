"""State-space model of the TurtleBot3 Burger (differential drive), used by the EKF and the LQR controller.

1. Nonlinear kinematic model (unicycle), state s = [x, y, theta], input u = [v, w]
       x'     = v cos(theta)
       y'     = v sin(theta)
       theta' = w
   Assumptions: wheels roll without slipping, the robot moves on a flat floor, and the wheel speed
   controllers track the commanded speeds (MuJoCo velocity actuators + the 1 m/s^2, 6 rad/s^2 ramps in
   world.drive). The wheel speeds map to (v, w) through
       v = r (w_R + w_L) / 2,      w = r (w_R - w_L) / b       (r = 0.033 m, b = 0.160 m)

2. Discrete odometry model (what the EKF predicts with), from one control step dt:
       ds  = r (dphi_R + dphi_L) / 2        (wheel encoders)
       dth = gyro_rate * dt                 (IMU yaw rate; the wheels slip when turning on the spot)
       x_k+1     = x_k + ds cos(theta_k + dth/2)
       y_k+1     = y_k + ds sin(theta_k + dth/2)
       theta_k+1 = theta_k + dth

3. Linearised path-tracking error model (what the LQR is designed on).
   The robot follows a straight reference line at forward speed v_r.  e_y = signed distance from the line
   (left +), e_th = heading error relative to the line.  For small e_th:
       e_y'  = v_r sin(e_th) ~= v_r e_th
       e_th' = w
   =>  e' = A e + B w,   A = [[0, v_r], [0, 0]],   B = [[0], [1]]

4. Measurement model (camera, ArUco marker of known size).  In the dock frame (origin at the marker centre,
   +x pointing out of the marker into the room) the marker pose is known: (0, 0, 0).  The camera measures the
   marker pose in the robot frame:
       z = h(s) = [ -x cos(th) - y sin(th),     (marker ahead of the robot, m)
                     x sin(th) - y cos(th),     (marker to the left of the robot, m)
                    -th ]                       (marker normal direction in the robot frame, rad)
"""
import numpy as np
from scipy.linalg import solve_continuous_are

WHEEL_R = 0.033
WHEEL_SEP = 0.160


# ---------------------------------------------------------------- nonlinear model + Jacobians (EKF)
def odometry_step(s, ds, dth):
    """Discrete odometry model f(s, u). s = [x, y, th]."""
    x, y, th = s
    m = th + dth / 2
    return np.array([x + ds * np.cos(m), y + ds * np.sin(m), th + dth])

def odometry_jacobians(s, ds, dth):
    """F = df/ds (3x3), G = df/du (3x2) with u = [ds, dth]."""
    m = s[2] + dth / 2
    c, sn = np.cos(m), np.sin(m)
    F = np.array([[1, 0, -ds * sn], [0, 1, ds * c], [0, 0, 1]])
    G = np.array([[c, -0.5 * ds * sn], [sn, 0.5 * ds * c], [0, 1]])
    return F, G

def marker_measurement(s):
    """h(s): expected marker pose in the robot frame, for a robot pose s in the dock frame."""
    x, y, th = s
    c, sn = np.cos(th), np.sin(th)
    return np.array([-x * c - y * sn, x * sn - y * c, -th])

def marker_jacobian(s):
    """H = dh/ds (3x3)."""
    x, y, th = s
    c, sn = np.cos(th), np.sin(th)
    return np.array([[-c, -sn, x * sn - y * c],
                     [sn, -c, x * c + y * sn],
                     [0, 0, -1]])

def pose_from_marker(z):
    """Invert h: robot pose in the dock frame from one marker measurement z = [ahead, left, angle]."""
    mx, my, a = z
    th = -a
    c, sn = np.cos(th), np.sin(th)
    return np.array([-(c * mx - sn * my), -(sn * mx + c * my), th])


# ---------------------------------------------------------------- linear tracking-error model (LQR)
def tracking_error_model(v_r):
    A = np.array([[0.0, v_r], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    return A, B

def lqr(A, B, Q, R):
    """Continuous-time LQR: returns K (u = -K e) and the closed-loop poles."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, np.linalg.eigvals(A - B @ K)


# ---------------------------------------------------------------- structural analysis
def controllability_matrix(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])

def observability_matrix(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])

def analysis(v_r=0.15, s_example=(0.8, 0.1, 0.05)):
    """Controllability and observability results quoted on the project page."""
    out = {}
    # (a) unicycle linearised about a moving reference (v_r, w_r = 0): full 3-state pose error
    for v in (v_r, 0.0):
        A = np.array([[0, 0, 0], [0, 0, v], [0, 0, 0]], float)      # e = [e_x, e_y, e_th]
        B = np.array([[1, 0], [0, 0], [0, 1]], float)               # u = [dv, dw]
        out[f"pose_error_controllability_rank(v_r={v})"] = int(np.linalg.matrix_rank(controllability_matrix(A, B)))
    # (b) the 2-state lateral tracking model used for the LQR
    A, B = tracking_error_model(v_r)
    out["lateral_controllability_rank"] = int(np.linalg.matrix_rank(controllability_matrix(A, B)))
    # (c) observability of the robot pose from the camera, linearised at an example pose near the dock
    #     (pose is a constant between frames when the robot stops: A = 0, so O = H)
    H = marker_jacobian(np.array(s_example))
    out["pose_observability_rank_camera"] = int(np.linalg.matrix_rank(H))
    # (d) odometry only (no camera): nothing measures the pose -> C = 0, rank 0 (dead reckoning drifts)
    out["pose_observability_rank_odometry_only"] = 0
    return out


if __name__ == "__main__":
    for k, v in analysis().items():
        print(f"{k:45s} {v}")

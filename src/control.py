"""Turn the plan into wheel commands (v [m/s], w [rad/s]) with LQR state feedback on the EKF's estimated pose.

Line tracking (drive to a point): the reference is the straight line from the segment start S to the target T.
    error state  e = [e_y, e_th]   e_y  = sideways distance from the line (left +)
                                   e_th = heading error relative to the line
    model        e' = A e + B w,   A = [[0, v], [0, 0]],  B = [[0], [1]]     (src/model.py, linearised, small e_th)
    control      w = -K(v) e       K from the continuous-time algebraic Riccati equation, re-computed for the
                                   current forward speed v (gain scheduling: the model depends on v)
    speed        v = min(v_max, K_V * distance left along the line) * cos(e_th)
Weights (Bryson's rule: 1 / largest acceptable value^2):
    Q = diag(1/E_Y_MAX^2, 1/E_TH_MAX^2),  R = 1/W_MAX^2
Heading regulation (turn on the spot): th' = w, same weights -> w = -K_th (th - th_ref), K_th = W_MAX / E_TH_MAX.
"""
import math
import numpy as np
from .contracts import wrap
from . import model

E_Y_MAX  = 0.05                  # [m]    largest sideways error we're happy with
E_TH_MAX = math.radians(10)      # [rad]  largest heading error we're happy with
W_MAX    = 1.0                   # [rad/s] turn rate we're willing to spend on that
W_LIMIT  = 1.5                   # [rad/s] hard saturation
K_V      = 0.8                   # [1/s]  forward speed per metre left to go
TURN_ONLY = 0.8                  # [rad]  target further off than this -> turn on the spot first
NEW_SEGMENT = 0.15               # [m]    target moved more than this -> start a new line from where we are

Q_LQR = np.diag([1 / E_Y_MAX ** 2, 1 / E_TH_MAX ** 2])
R_LQR = np.array([[1 / W_MAX ** 2]])
K_HEADING = W_MAX / E_TH_MAX     # scalar LQR: sqrt(Q/R)


def lqr_gain(v):
    """LQR gain K = [k_y, k_th] for forward speed v (gain-scheduled)."""
    v = max(abs(v), 0.02)                             # model is uncontrollable at v = 0 (rank 1)
    K, _ = model.lqr(*model.tracking_error_model(v), Q_LQR, R_LQR)
    return K.ravel()

# gain table, so the Riccati equation isn't solved 20 times a second
_V_TABLE = np.linspace(0.02, 0.22, 21)
_K_TABLE = np.array([lqr_gain(v) for v in _V_TABLE])

def gain(v):
    return np.array([np.interp(abs(v), _V_TABLE, _K_TABLE[:, i]) for i in range(2)])


class Controller:
    def __init__(self, calibration):
        self.cal = calibration
        self._seg_start = None
        self._seg_target = None
        self.last_error = None              # (e_y, e_th) for plots

    def _heading(self, th, th_ref):
        e = wrap(th - th_ref)
        return float(np.clip(-K_HEADING * e, -W_LIMIT, W_LIMIT))

    def update(self, pose, plan):
        if plan.done or plan.mode == "idle":
            return 0.0, 0.0
        x, y, th = pose
        if plan.spin:                                          # searching: turn on the spot
            self._seg_target = None
            return 0.0, plan.spin
        if plan.heading is not None:                           # turn on the spot to a heading
            self._seg_target = None
            return 0.0, self._heading(th, plan.heading)
        if plan.target_xy is None:
            return 0.0, 0.0

        T = np.asarray(plan.target_xy, float)
        p = np.array([x, y])
        # reference line: given by the planner, or from where we were when this target was set
        if getattr(plan, "line_from", None) is not None:
            S = np.asarray(plan.line_from, float)
        else:
            if self._seg_target is None or np.linalg.norm(T - self._seg_target) > NEW_SEGMENT:
                self._seg_start = p.copy()
            S = self._seg_start
        self._seg_target = T.copy()

        if np.linalg.norm(T - S) < 0.01:                     # degenerate line: aim straight at the target
            S = p
        if np.linalg.norm(T - S) < 1e-6:
            return 0.0, 0.0
        u = (T - S) / np.linalg.norm(T - S)
        line_th = math.atan2(u[1], u[0])
        e_y = float(u[0] * (p[1] - S[1]) - u[1] * (p[0] - S[0]))
        e_th = wrap(th - line_th)
        self.last_error = (e_y, e_th)

        bearing = wrap(math.atan2(T[1] - y, T[0] - x) - th)    # where the target is, seen from the robot
        if abs(bearing) > TURN_ONLY and np.linalg.norm(T - p) > 0.05:
            return 0.0, self._heading(th, th + bearing)       # facing the wrong way: turn first

        remaining = float(np.dot(T - p, u))                   # distance left along the line
        v = min(plan.max_speed, K_V * max(remaining, 0.0)) * max(math.cos(e_th), 0.0)
        k_y, k_th = gain(v)
        w = -(k_y * e_y + k_th * e_th)
        return float(v), float(np.clip(w, -W_LIMIT, W_LIMIT))

"""Turn the plan into wheel commands (v [m/s], w [rad/s])."""
import math
from .contracts import wrap

K_TURN    = 2.5      # turn-rate gain [1/s]
K_DRIVE   = 0.8      # forward-speed gain [1/s]
W_LIMIT   = 1.5      # turn-rate limit [rad/s]
TURN_ONLY = 0.8      # if the target is more than this far off to the side [rad], turn on the spot first


class Controller:
    def __init__(self, calibration):
        self.cal = calibration

    def update(self, pose, plan):
        if plan.done or plan.mode == "idle":
            return 0.0, 0.0
        x, y, th = pose

        if plan.spin:                                    # searching: turn on the spot
            return 0.0, plan.spin

        if plan.heading is not None:                     # turn on the spot to a heading
            err = wrap(plan.heading - th)
            return 0.0, max(-W_LIMIT, min(W_LIMIT, 2.0 * err))

        if plan.target_xy is not None:                   # drive to a point
            dx, dy = plan.target_xy[0] - x, plan.target_xy[1] - y
            rho = math.hypot(dx, dy)
            a = wrap(math.atan2(dy, dx) - th)            # how far off to the side the target is
            w = max(-W_LIMIT, min(W_LIMIT, K_TURN * a))
            if abs(a) > TURN_ONLY:
                return 0.0, w
            v = min(plan.max_speed, K_DRIVE * rho) * math.cos(a) ** 2
            return v, w

        return 0.0, 0.0

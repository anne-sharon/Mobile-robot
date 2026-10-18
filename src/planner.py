"""What should I do next?

  search   -> spin on the spot looking for the marker
  explore  -> marker not seen from here: drive somewhere new (avoiding boxes with the LiDAR), then search again
  approach -> marker seen: drive to a point straight out in front of the dock, avoiding boxes
  align    -> turn on the spot to face the dock square-on
  dock     -> creep in along the dock's centre line until the robot is PARK_DIST from the marker
  done     -> parked
"""
import math
import numpy as np
from .contracts import Plan, wrap

PARK_DIST    = 0.35      # robot origin -> marker plane when parked [m]
PRE_DOCK     = 0.80      # approach point: this far out in front of the marker [m]
ROBOT_HALF_W = 0.20      # half-width of the corridor that must be free of LiDAR hits (robot radius + margin) [m]
LOOKAHEAD    = 0.70      # how far ahead the corridor is checked [m]
SPIN_RATE    = 0.7       # search spin [rad/s]
VISIT_BONUS  = 1.0       # explore: how much we prefer going somewhere we haven't been


class Planner:
    def __init__(self, calibration):
        self.cal = calibration
        self.lidar_x = calibration["lidar_pos"][0]
        self.mode = "search"
        self._spun = 0.0                 # how far we've turned in the current search spin [rad]
        self._last_th = None
        self._visited = []               # places we've already searched from
        self._goal = None                # current explore goal
        self._goal_t = 0.0
        self._side = 1                   # which way we last dodged an obstacle (+1 left, -1 right)

    # ------------------------------------------------------------------ helpers
    def _lidar_points(self, lidar):
        """LiDAR hits as (x, y) in the robot frame."""
        if lidar is None:
            return np.zeros((0, 2))
        a = np.deg2rad(np.arange(len(lidar)))
        ok = np.isfinite(lidar)
        r = lidar[ok]
        return np.stack([self.lidar_x + r * np.cos(a[ok]), r * np.sin(a[ok])], axis=1)

    def _clear(self, pts, heading, length):
        """Is the corridor straight ahead along `heading` (robot frame) free of obstacles?"""
        if len(pts) == 0:
            return True
        u = np.array([math.cos(heading), math.sin(heading)])
        along = pts @ u
        perp = np.abs(pts[:, 0] * u[1] - pts[:, 1] * u[0])
        return not np.any((along > 0) & (along < length) & (perp < ROBOT_HALF_W))

    def _safe_goal(self, pose, pts, target):
        """Head for `target`; if the way is blocked, pick the nearest free direction and a sub-goal along it.
        Returns a point in the estimator frame, or None if every direction is blocked."""
        x, y, th = pose
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        bearing = wrap(math.atan2(dy, dx) - th)                 # target direction in the robot frame
        length = min(dist, LOOKAHEAD)
        for off_deg in range(0, 181, 5):
            for side in ((self._side, -self._side) if off_deg else (1,)):
                h = bearing + side * math.radians(off_deg)
                if self._clear(pts, h, length):
                    if off_deg == 0:
                        return tuple(target)
                    self._side = side
                    step = min(dist, 0.5)
                    return (x + step * math.cos(th + h), y + step * math.sin(th + h))
        return None

    def _explore_goal(self, pose, lidar):
        """Pick a new place to search from: lots of free space, far from where we've already looked."""
        x, y, th = pose
        best, best_score = None, -1e9
        r = np.where(np.isfinite(lidar), lidar, self.cal["lidar_range"][1])
        for deg in range(0, 360, 10):
            window = [(deg + k) % 360 for k in range(-15, 16)]
            reach = min(float(np.min(r[window])) - 0.45, 1.5)
            if reach < 0.3:
                continue
            gx, gy = x + reach * math.cos(th + math.radians(deg)), y + reach * math.sin(th + math.radians(deg))
            novelty = min((math.hypot(gx - vx, gy - vy) for vx, vy in self._visited), default=0.0)
            score = reach + VISIT_BONUS * novelty
            if score > best_score:
                best, best_score = (gx, gy), score
        return best

    # ------------------------------------------------------------------ main entry
    def update(self, pose, obs, dock=None):
        x, y, th = pose
        pts = self._lidar_points(obs.get("lidar"))
        dth = 0.0 if self._last_th is None else abs(wrap(th - self._last_th))
        self._last_th = th

        # ---------------- marker never seen: search, then explore ----------------
        if dock is None:
            if self.mode == "search":
                self._spun += dth
                if self._spun < 2 * math.pi:
                    return Plan(mode="search", spin=SPIN_RATE)
                self._visited.append((x, y))
                self._goal, self._goal_t, self.mode = self._explore_goal(pose, obs["lidar"]), obs["t"], "explore"
            if self.mode == "explore":
                reached = self._goal is None or math.hypot(self._goal[0] - x, self._goal[1] - y) < 0.15
                if reached or obs["t"] - self._goal_t > 15.0:
                    self.mode, self._spun = "search", 0.0
                    return Plan(mode="search", spin=SPIN_RATE)
                sub = self._safe_goal(pose, pts, self._goal)
                return Plan(mode="explore", target_xy=sub) if sub else Plan(mode="explore", spin=SPIN_RATE)

        # ---------------- marker seen: approach, align, dock ----------------
        mx, my, nx, ny = dock.x, dock.y, dock.nx, dock.ny
        face = math.atan2(-ny, -nx)                              # heading that looks straight at the marker
        along = (x - mx) * nx + (y - my) * ny                    # distance out from the marker plane
        lateral = (x - mx) * (-ny) + (y - my) * nx               # sideways offset from the centre line

        if self.mode in ("search", "explore"):
            self.mode = "approach"

        if self.mode == "approach":
            pre = (mx + PRE_DOCK * nx, my + PRE_DOCK * ny)
            on_line = abs(lateral) < 0.04 and PARK_DIST < along < PRE_DOCK + 0.1
            if on_line or math.hypot(pre[0] - x, pre[1] - y) < 0.08:
                self.mode = "align"
            else:
                sub = self._safe_goal(pose, pts, pre)
                return Plan(mode="approach", target_xy=sub) if sub else Plan(mode="approach", spin=SPIN_RATE)

        if self.mode == "align":
            if abs(wrap(face - th)) > math.radians(2):
                return Plan(mode="align", heading=face)
            self.mode = "dock"

        if self.mode == "dock":
            if abs(lateral) > 0.12:                              # drifted too far off the line: go round again
                self.mode = "approach"
                return Plan(mode="approach")
            remaining = along - PARK_DIST
            if remaining <= 0.003:
                self.mode = "done"
                return Plan(mode="done", done=True)
            # follow the centre line: always aim at a point 0.25 m further along it (even past the parking
            # spot), so the robot keeps pointing at the dock instead of twisting towards a point under its nose
            ahead = along - 0.25
            target = (mx + ahead * nx, my + ahead * ny)
            speed = float(np.clip(0.6 * remaining, 0.02, 0.12))
            return Plan(mode="dock", target_xy=target, max_speed=speed)

        return Plan(mode="done", done=True)

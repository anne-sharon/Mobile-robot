"""Ground-truth logger.  Evaluation only: src/robot and src/perception must never import this file."""
import csv, math
import numpy as np
from src.sim.judge import Judge

class TruthLogger:
    def __init__(self, sim):
        self.judge = Judge(sim)
        self.x0, self.y0, self.th0 = self.judge.true_pose()   # start pose = origin of the estimator frame
        self.rows = []

    def _to_start_frame(self, x, y, th):
        c, s = math.cos(-self.th0), math.sin(-self.th0)
        dx, dy = x - self.x0, y - self.y0
        return dx * c - dy * s, dx * s + dy * c, (th - self.th0 + math.pi) % (2 * math.pi) - math.pi

    def record(self, t, est_pose, cmd):
        wx, wy, wth = self.judge.true_pose()
        tx, ty, tth = self._to_start_frame(wx, wy, wth)
        ex, ey, eth = (float(v) for v in est_pose)
        self.rows.append(dict(
            t=round(t, 3), est_x=ex, est_y=ey, est_th=eth, true_x=tx, true_y=ty, true_th=tth,
            err_pos=math.hypot(ex - tx, ey - ty), err_th=abs((eth - tth + math.pi) % (2 * math.pi) - math.pi),
            world_x=wx, world_y=wy, world_th=wth, v=cmd[0], w=cmd[1], collided=int(self.judge.collided())))

    def summary(self):
        s = dict(self.judge.park_error())
        s["collisions"] = sum(r["collided"] for r in self.rows)
        s["mean_pose_err_m"] = float(np.mean([r["err_pos"] for r in self.rows])) if self.rows else float("nan")
        return s

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader(); w.writerows(self.rows)

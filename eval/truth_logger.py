"""Ground-truth logger.  Evaluation only: estimator.py, planner.py, control.py and marker.py must never import this file."""
import csv, math
import numpy as np
from src.judge import Judge
from eval.frames import world_to_frame, world_to_dock, wrap


class TruthLogger:
    def __init__(self, sim):
        self.judge = Judge(sim)
        self.x0, self.y0, self.th0 = self.judge.true_pose()   # start pose = origin of the estimator's start frame
        self.rows = []

    def true_in(self, frame):
        """The true pose, expressed in the same frame as the estimate."""
        wx, wy, wth = self.judge.true_pose()
        if frame == "dock":
            return world_to_dock(wx, wy, wth)
        return world_to_frame(wx, wy, wth, self.x0, self.y0, self.th0)

    def record(self, t, est_pose, cmd, frame="start", P=None):
        wx, wy, wth = self.judge.true_pose()
        tx, ty, tth = self.true_in(frame)
        ex, ey, eth = (float(v) for v in est_pose)
        err = np.array([ex - tx, ey - ty, wrap(eth - tth)])
        row = dict(t=round(t, 3), frame=frame, est_x=ex, est_y=ey, est_th=eth, true_x=tx, true_y=ty, true_th=tth,
                   err_pos=math.hypot(err[0], err[1]), err_th=abs(err[2]),
                   world_x=wx, world_y=wy, world_th=wth, v=cmd[0], w=cmd[1], collided=int(self.judge.collided()))
        if P is not None:
            P = np.asarray(P)
            row.update(sd_x=math.sqrt(max(P[0, 0], 0)), sd_y=math.sqrt(max(P[1, 1], 0)),
                       sd_th=math.sqrt(max(P[2, 2], 0)))
            # NEES: normalised estimation error squared (should average 3 for a consistent 3-state filter)
            try:
                row["nees"] = float(err @ np.linalg.solve(P, err)) if np.linalg.det(P) > 1e-18 else float("nan")
            except np.linalg.LinAlgError:
                row["nees"] = float("nan")
        self.rows.append(row)

    def summary(self):
        s = dict(self.judge.park_error())
        s["collisions"] = sum(r["collided"] for r in self.rows)
        s["mean_pose_err_m"] = float(np.mean([r["err_pos"] for r in self.rows])) if self.rows else float("nan")
        dock = [r for r in self.rows if r["frame"] == "dock"]
        if dock:
            s["mean_pose_err_dock_frame_m"] = float(np.mean([r["err_pos"] for r in dock]))
            near = [r["err_pos"] for r in dock if r["true_x"] < 1.5]          # within 1.5 m of the marker
            if near:
                s["mean_pose_err_near_dock_m"] = float(np.mean(near))
            s["final_pose_err_m"] = dock[-1]["err_pos"]
            nees = [r["nees"] for r in dock if np.isfinite(r.get("nees", np.nan))]
            if nees:
                s["mean_nees"] = float(np.mean(nees))
        return s

    def save(self, path):
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(self.rows)

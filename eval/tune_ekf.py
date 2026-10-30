"""Tune the EKF noise scales with a consistency test.
Run from the repo root:  python -m eval.tune_ekf
For each (Q_SCALE, R_SCALE) pair, runs docking episodes and reports the mean NEES in the dock frame (a consistent
3-state filter averages 3), the mean and final pose error, and whether the robot parked. Evaluation only.
Writes eval/results/ekf_tuning.csv."""
import os, csv, itertools
import numpy as np
from src.world import Sim
from src.contracts import CONTROL_DT
from src.estimator import Estimator
from src.planner import Planner
from src.control import Controller
from eval.truth_logger import TruthLogger
from eval.run_episode import sense

STARTS = [(-2.3, -1.2, 2.4), (0.0, -1.6, 1.57), (1.0, 1.5, 3.0), (-0.5, -0.2, 3.14), (2.0, -1.5, 2.0)]


def episode(start, seed, q_scale, r_scale, max_time=150.0, obs_hook=None, step_hook=None):
    sim = Sim(start=start, seed=seed)
    cal = sim.sensors.calibration
    est, pl, ct = Estimator(cal), Planner(cal), Controller(cal)
    est.q_scale, est.r_scale = q_scale, r_scale
    truth = TruthLogger(sim)
    last, tick = dict(lidar=None, image=None), 0
    while sim.time < max_time:
        obs, last = sense(sim, tick, last)
        if obs_hook:
            obs = obs_hook(obs, sim)
        pose = est.update(obs)
        plan = pl.update(pose, obs, est.dock)
        v, w = ct.update(pose, plan)
        truth.record(sim.time, pose, (v, w), est.frame, est.P)
        if plan.done:
            break
        if step_hook:
            step_hook(sim)
        sim.drive(v, w, CONTROL_DT)
        tick += 1
    s = truth.summary(); s["time"] = sim.time; s.update(est.stats)
    truth.judge.close(); sim.close()
    return s, truth


def episode_summary(start, seed, q_scale, r_scale):
    return episode(start, seed, q_scale, r_scale)[0]


def tune(q_scales=(1, 4, 16), r_scales=(1, 4, 16, 64), out="eval/results/ekf_tuning.csv"):
    from eval.parallel import run_many
    grid = list(itertools.product(q_scales, r_scales))
    jobs = [(st, i, q, r) for q, r in grid for i, st in enumerate(STARTS)]
    flat = run_many(episode_summary, jobs)
    rows = []
    for g, (q, r) in enumerate(grid):
        res = flat[g * len(STARTS):(g + 1) * len(STARTS)]
        row = dict(q_scale=q, r_scale=r,
                   mean_nees=float(np.mean([x.get("mean_nees", np.nan) for x in res])),
                   mean_pose_err_cm=100 * float(np.mean([x.get("mean_pose_err_dock_frame_m", np.nan) for x in res])),
                   near_dock_err_cm=100 * float(np.mean([x.get("mean_pose_err_near_dock_m", np.nan) for x in res])),
                   final_pose_err_mm=1000 * float(np.mean([x.get("final_pose_err_m", np.nan) for x in res])),
                   park_err_cm=100 * float(np.mean([x["pos_m"] for x in res])),
                   parked=sum(x["parked"] for x in res), runs=len(res),
                   rejected=int(sum(x["rejected"] for x in res)))
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    tune()

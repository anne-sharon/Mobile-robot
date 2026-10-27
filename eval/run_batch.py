"""Repeated trials under varied conditions (robustness study). Evaluation only.
Run from the repo root:  python -m eval.run_batch            (all conditions, 20 start poses each, ~10 min)
                         python -m eval.run_batch nominal     (one condition)
Writes eval/results/batch_<condition>.csv (one row per run) and eval/results/batch_summary.csv.

Conditions (the disturbance is injected by this file, between the simulator and the robot; the robot code is
unchanged and never told which condition it is in):
  nominal     : as simulated
  gyro_noise  : gyro yaw rate + white noise (0.05 rad/s) + constant bias (0.01 rad/s, ~0.6 deg/s)
  camera_noise: Gaussian pixel noise (sigma = 12 grey levels) on every camera image
  push        : when the robot first comes within 1.5 m of the dock it is shoved: 0.5 m/s sideways and 2 rad/s
                yaw for one instant (the wheels slip, so the encoders do not see the sideways part)
  occlusion   : when the robot first comes within 2.0 m of the dock the camera goes blank for 6 s
                (marker blocked / out of view during the final approach)
The triggers use the simulator's true position (evaluation code only) so that every run is disturbed at the
same point of the task; each run records whether and when its disturbance fired.
  combined    : gyro_noise + camera_noise + push together
"""
import os, sys, csv, math
import numpy as np

CONDITIONS = ["nominal", "gyro_noise", "camera_noise", "push", "occlusion", "combined"]


def start_poses(n=20, seed=1):
    """Fixed hand-picked starts plus random ones that are clear of every obstacle."""
    from src.world import BOXES, SOFA, TABLE
    fixed = [(-2.3, -1.2, 2.4), (-2.5, 1.5, 0.0), (0.0, -1.6, 1.57), (1.0, 1.5, 3.0),
             (-0.5, -0.2, 3.14), (2.0, -1.5, 2.0), (-2.6, 0.0, -3.0)]
    def clear(x, y):
        if any(math.hypot(x - bx, y - by) < 0.55 for _, (bx, by), *_ in BOXES):
            return False
        sx, sy = SOFA["pos"]; hx, hy, _ = SOFA["half"]
        if abs(x - sx) < hx + 0.3 and abs(y - sy) < hy + 0.3:
            return False
        if math.hypot(x - TABLE["pos"][0], y - TABLE["pos"][1]) < TABLE["radius"] + 0.3:
            return False
        return math.hypot(x - 2.6, y - 0.3) > 0.6
    starts = [s for s in fixed if clear(*s[:2])][:n]
    rng = np.random.default_rng(seed)
    while len(starts) < n:
        x, y = rng.uniform(-2.6, 2.2), rng.uniform(-1.6, 1.6)
        if clear(x, y):
            starts.append((round(x, 2), round(y, 2), round(rng.uniform(-math.pi, math.pi), 2)))
    return starts


def make_hooks(condition, seed):
    rng = np.random.default_rng(1000 + seed)
    gyro = condition in ("gyro_noise", "combined")
    cam = condition in ("camera_noise", "combined")
    push = condition in ("push", "combined")
    occl = condition == "occlusion"
    state = dict(pushed=False, push_t=None, occl_start=None, last_img_in=None, last_img_out=None)

    def dist_to_dock(sim):
        from eval.frames import MARKER_X, MARKER_Y
        return math.hypot(sim.data.qpos[0] - MARKER_X, sim.data.qpos[1] - MARKER_Y)

    def occluded(sim):
        if state["occl_start"] is None and dist_to_dock(sim) < 2.0:
            state["occl_start"] = sim.time
        return state["occl_start"] is not None and sim.time < state["occl_start"] + 6.0

    def obs_hook(obs, sim):
        obs = dict(obs)
        if gyro:
            rate, acc = obs["imu"]
            obs["imu"] = (rate + 0.01 + rng.normal(0, 0.05), acc)
        img = obs.get("image")
        if img is not None and (cam or occl):
            if img is state["last_img_in"]:                    # same held frame -> hand back the same object
                obs["image"] = state["last_img_out"]
            else:
                out = img
                if occl and occluded(sim):
                    out = None
                elif cam:
                    out = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
                state["last_img_in"], state["last_img_out"] = img, out
                obs["image"] = out
        return obs

    def step_hook(sim):
        if push and not state["pushed"] and dist_to_dock(sim) < 1.5:
            q = sim.data.qpos[3:7]
            th = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            sim.data.qvel[0] += -0.5 * math.sin(th)            # sideways (to the robot's left)
            sim.data.qvel[1] += 0.5 * math.cos(th)
            sim.data.qvel[5] += 2.0
            state["pushed"], state["push_t"] = True, sim.time
    obs_hook.state = state
    return obs_hook, step_hook


def run_one(start, seed, condition):
    from eval.tune_ekf import episode
    from src.estimator import Q_SCALE, R_SCALE
    obs_hook, step_hook = make_hooks(condition, seed)
    s, _ = episode(start, seed, Q_SCALE, R_SCALE, obs_hook=obs_hook, step_hook=step_hook)
    st = obs_hook.state
    s.update(condition=condition, start=str(start), push_t=st["push_t"], occlusion_start_t=st["occl_start"],
             disturbed=bool(st["push_t"] is not None or st["occl_start"] is not None or
                            condition in ("gyro_noise", "camera_noise")))
    return s


def summarise(rows):
    f = lambda k: [r[k] for r in rows if k in r and np.isfinite(r[k])]
    return dict(condition=rows[0]["condition"], runs=len(rows),
                runs_disturbed=sum(bool(r.get("disturbed")) for r in rows) if rows[0]["condition"] != "nominal" else 0,
                parked=sum(r["parked"] for r in rows),
                runs_with_collision=sum(r["collisions"] > 0 for r in rows),
                park_err_cm_mean=100 * np.mean(f("pos_m")), park_err_cm_worst=100 * np.max(f("pos_m")),
                heading_err_deg_mean=np.mean(f("heading_deg")), heading_err_deg_worst=np.max(f("heading_deg")),
                time_s_mean=np.mean(f("time")),
                est_err_near_dock_cm=100 * np.mean(f("mean_pose_err_near_dock_m")),
                est_err_final_mm=1000 * np.mean(f("final_pose_err_m")),
                nees_mean=np.mean(f("mean_nees")) if f("mean_nees") else float("nan"),
                rejected_per_run=np.mean(f("rejected")), reinits_per_run=np.mean(f("reinits")))


def main(conditions=CONDITIONS, n=20, out_dir="eval/results"):
    from eval.parallel import run_many
    starts = start_poses(n)
    jobs = [(st, i, c) for c in conditions for i, st in enumerate(starts)]
    flat = run_many(run_one, jobs)
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for c in conditions:
        rows = [r for r in flat if r["condition"] == c]
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, f"batch_{c}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        summary.append(summarise(rows))
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary[-1].items()}, flush=True)
    with open(os.path.join(out_dir, "batch_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)

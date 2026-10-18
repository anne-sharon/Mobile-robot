"""Run one episode and draw the robot's path on the overhead view (evaluation only).
Run from the repo root:  python -m eval.plot_episode
Saves eval/results/docking_path.png"""
import os, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.world import Sim, PARK_POSE
from src.contracts import CONTROL_DT, SENSE_EVERY
from src.estimator import Estimator
from src.planner import Planner
from src.control import Controller
from src.judge import Judge
from eval.run_episode import sense

COLOURS = dict(search="#b0b0b0", explore="#e0a030", approach="#3070d0", align="#a040c0", dock="#30a050")

def run_and_plot(max_time=120.0, out="eval/results/docking_path.png"):
    sim = Sim()
    cal = sim.sensors.calibration
    est, planner, ctrl, judge = Estimator(cal), Planner(cal), Controller(cal), Judge(sim)
    last, tick, path = dict(lidar=None, image=None), 0, []
    while sim.time < max_time:
        obs, last = sense(sim, tick, last)
        pose = est.update(obs)
        plan = planner.update(pose, obs, est.dock)
        v, w = ctrl.update(pose, plan)
        x, y, _ = judge.true_pose()
        path.append((x, y, plan.mode))
        if plan.done:
            break
        sim.drive(v, w, CONTROL_DT)
        tick += 1

    top = judge.render_topdown()
    err = judge.park_error()
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax[0].imshow(top)
    for mode, colour in COLOURS.items():
        pts = [judge.world_to_px(x, y) for x, y, m in path if m == mode]
        if pts:
            px, py = zip(*pts)
            ax[0].scatter(px, py, s=4, c=colour, label=mode)
    sx, sy = judge.world_to_px(*path[0][:2])
    ax[0].plot(sx, sy, "ko", ms=8); ax[0].annotate("start", (sx, sy), xytext=(8, -8), textcoords="offset points")
    gx, gy = judge.world_to_px(*PARK_POSE[:2])
    ax[0].plot(gx, gy, "r+", ms=14, mew=2, label="parking spot")
    ax[0].legend(loc="lower right", markerscale=3)
    ax[0].set_title(f"Robot path ({sim.time:.0f} s)   parked: {err['parked']}   "
                    f"error {err['pos_m']*100:.1f} cm, {err['heading_deg']:.1f}°")
    ax[0].axis("off")
    ax[1].imshow(sim.sensors.camera()); ax[1].set_title("Robot camera when parked"); ax[1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return err

if __name__ == "__main__":
    print(run_and_plot())

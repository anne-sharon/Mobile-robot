"""Main loop: sense -> estimate -> plan -> control -> step the simulator.
Run from the repo root:  python -m eval.run_episode [seconds]
This is the only file that touches both the simulator and the robot code."""
import os, sys
from src.world import Sim
from src.contracts import CONTROL_DT, SENSE_EVERY
from src.estimator import Estimator
from src.planner import Planner
from src.control import Controller
from eval.truth_logger import TruthLogger

def sense(sim, tick, last):
    """Collect what the robot is allowed to know. LiDAR/camera are slower (5 Hz) and held between reads."""
    s = sim.sensors
    if tick % SENSE_EVERY == 0:
        last = dict(lidar=s.lidar(), image=s.camera())
    return dict(t=sim.time, encoders=s.encoders(), imu=s.imu(), **last), last

def run_episode(max_time=120.0, seed=0, out="eval/results/episode.csv"):
    sim = Sim(seed=seed)
    cal = sim.sensors.calibration            # datasheet-style constants; no pose in here
    estimator, planner, controller = Estimator(cal), Planner(cal), Controller(cal)
    truth = TruthLogger(sim)                 # records ground truth; nothing below reads it back
    last, tick = dict(lidar=None, image=None), 0

    while sim.time < max_time:
        obs, last = sense(sim, tick, last)               # 1. sense
        pose = estimator.update(obs)                     # 2. estimate
        plan = planner.update(pose, obs, estimator.dock)  # 3. plan
        v, w = controller.update(pose, plan)             # 4. control
        truth.record(sim.time, pose, (v, w), estimator.frame, estimator.P)  # (evaluation only)
        if plan.done:
            break
        sim.drive(v, w, CONTROL_DT)                      # 5. step the simulator
        tick += 1

    os.makedirs(os.path.dirname(out), exist_ok=True)
    truth.save(out)
    return truth.summary()

if __name__ == "__main__":
    print(run_episode(max_time=float(sys.argv[1]) if len(sys.argv) > 1 else 120.0))

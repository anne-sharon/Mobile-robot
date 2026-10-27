"""How much does odometry drift?  (Sets the EKF process noise Q.)
Run from the repo root:  python -m eval.characterise_odometry
Runs docking episodes from several start poses and, at every 0.05 s control step, compares the motion the EKF
predicts from the encoders + gyro with the robot's true motion (evaluation only). Fits
    var(error) = k^2 * motion^2 + floor^2
separately for distance (ds) and heading (dth). Writes models/odometry_noise.json."""
import os, json, math
import numpy as np
from src.world import Sim, WHEEL_R
from src.contracts import CONTROL_DT
from src.judge import Judge
from eval.frames import wrap

STARTS = [(-2.3, -1.2, 2.4), (0.0, -1.6, 1.57), (1.0, 1.5, 3.0), (-0.5, -0.2, 3.14), (2.0, -1.5, 2.0)]


def _fit(err, mag):
    """Least-squares fit of err^2 = k^2 mag^2 + floor^2."""
    X = np.column_stack([mag ** 2, np.ones_like(mag)])
    k2, f2 = np.linalg.lstsq(X, err ** 2, rcond=None)[0]
    return float(math.sqrt(max(k2, 0))), float(math.sqrt(max(f2, 0)))


def characterise(out_json="models/odometry_noise.json"):
    from src.estimator import Estimator
    from src.planner import Planner
    from src.control import Controller
    from eval.run_episode import sense
    e_ds, e_dth, m_ds, m_dth = [], [], [], []
    for i, start in enumerate(STARTS):
        sim = Sim(start=start, seed=i)
        cal, judge = sim.sensors.calibration, Judge(sim)
        est, pl, ct = Estimator(cal), Planner(cal), Controller(cal)
        last, tick = dict(lidar=None, image=None), 0
        enc0, p0, r0 = sim.sensors.encoders(), judge.true_pose(), sim.sensors.imu()[0]
        while sim.time < 120:
            obs, last = sense(sim, tick, last)
            pose = est.update(obs)
            plan = pl.update(pose, obs, est.dock)
            if plan.done:
                break
            sim.drive(*ct.update(pose, plan), CONTROL_DT); tick += 1
            enc1, p1, r1 = sim.sensors.encoders(), judge.true_pose(), sim.sensors.imu()[0]
            ds = WHEEL_R * ((enc1[0] - enc0[0]) + (enc1[1] - enc0[1])) / 2
            dth = 0.5 * (r0 + r1) * CONTROL_DT
            mid = p0[2] + dth / 2
            ds_true = (p1[0] - p0[0]) * math.cos(mid) + (p1[1] - p0[1]) * math.sin(mid)
            e_ds.append(ds - ds_true); e_dth.append(wrap(dth - wrap(p1[2] - p0[2])))
            m_ds.append(abs(ds)); m_dth.append(abs(dth))
            enc0, p0, r0 = enc1, p1, r1
    e_ds, e_dth, m_ds, m_dth = map(np.array, (e_ds, e_dth, m_ds, m_dth))
    k_ds, f_ds = _fit(e_ds, m_ds)
    k_dth, f_dth = _fit(e_dth, m_dth)
    out = dict(model="per 0.05 s step: var_ds = (k_ds*ds)^2 + floor_ds^2,  var_dth = (k_dth*dth)^2 + floor_dth^2",
               k_ds=k_ds, floor_ds=f_ds, k_dth=k_dth, floor_dth=f_dth,
               rms_err_ds_mm_per_step=float(1000 * np.sqrt(np.mean(e_ds ** 2))),
               rms_err_dth_deg_per_step=float(math.degrees(np.sqrt(np.mean(e_dth ** 2)))),
               steps=int(len(e_ds)), episodes=len(STARTS))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(characterise(), indent=2))

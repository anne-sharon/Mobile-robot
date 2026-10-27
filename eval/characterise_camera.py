"""How accurate is the camera's marker measurement?  (Sets the EKF measurement noise R.)
Run from the repo root:  python -m eval.characterise_camera
Places the robot at many random poses facing the dock (evaluation only: uses the simulator's true pose to
compute the true measurement), measures the marker with the calibrated camera, and fits error vs distance.
Writes models/camera_noise.json and eval/results/camera_noise.png."""
import os, json, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.world import Sim
from src.marker import detect_marker, marker_in_robot_frame, load_camera_calibration
from src.model import marker_measurement
from eval.frames import world_to_dock, wrap, MARKER_X, MARKER_Y


def characterise(n=400, seed=0, out_json="models/camera_noise.json", out_png="eval/results/camera_noise.png"):
    sim = Sim(seed=seed)
    cal = sim.sensors.calibration
    K, dist, _ = load_camera_calibration(cal["camera_matrix"])
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        d = rng.uniform(0.38, 3.6)                       # distance from the marker [m]
        bearing = rng.uniform(-50, 50)                   # where the robot stands, seen from the marker [deg]
        x = MARKER_X - d * math.cos(math.radians(bearing))
        y = MARKER_Y + d * math.sin(math.radians(bearing))
        if not (-2.85 < x < 2.85 and -1.85 < y < 1.85):
            continue
        look = math.atan2(MARKER_Y - y, MARKER_X - x) + math.radians(rng.uniform(-22, 22))
        sim.reset(x, y, look)
        res = detect_marker(sim.sensors.camera(), K, cal["marker"], dist)
        z = None if res is None else marker_in_robot_frame(res[1], res[2], cal["camera_pos"][0])
        if z is None:
            continue
        z_true = marker_measurement(np.array(world_to_dock(x, y, look)))
        e = z - z_true
        e[2] = wrap(e[2])
        rows.append((d, *e))
    a = np.array(rows)
    d, e = a[:, 0], a[:, 1:]

    # 1. systematic bias: median error per distance bin, fitted as bias(d) = c * d^2
    #    (a corner-detection offset of a fraction of a pixel makes the marker look smaller -> further away,
    #     and the effect grows with d^2 because the marker's image shrinks as 1/d)
    bins = np.linspace(0.38, 3.6, 8)
    centres, med = [], []
    for lo, hi in zip(bins, bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= 10:
            centres.append((lo + hi) / 2); med.append(np.median(e[m], 0))
    centres, med = np.array(centres), np.array(med)
    bias_c = [float(np.sum(centres ** 2 * med[:, k]) / np.sum(centres ** 4)) for k in range(3)]
    e_corr = e - np.outer(d ** 2, bias_c)
    # 2. spread after removing the bias: robust sigma (median absolute deviation) per bin, fit sigma = a + b d^2
    sig = []
    for lo, hi in zip(bins, bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= 10:
            sig.append(1.4826 * np.median(np.abs(e_corr[m] - np.median(e_corr[m], 0)), 0))
    sig = np.array(sig)
    X = np.column_stack([np.ones_like(centres), centres ** 2])
    coef = [np.linalg.lstsq(X, sig[:, k], rcond=None)[0].clip(min=1e-4) for k in range(3)]
    angle_outliers = float(np.mean(np.abs(e_corr[:, 2]) > math.radians(15)))

    model = dict(
        bias_model="bias(d) = c * d^2, subtracted from every measurement (d = distance to marker, m)",
        bias_c=dict(ahead=bias_c[0], left=bias_c[1], angle=bias_c[2]),
        sigma_model="sigma(d) = a + b * d^2  (after bias correction)",
        ahead=dict(a=float(coef[0][0]), b=float(coef[0][1])),
        left=dict(a=float(coef[1][0]), b=float(coef[1][1])),
        angle=dict(a=float(coef[2][0]), b=float(coef[2][1])),
        angle_outlier_rate=angle_outliers,
        samples=len(rows),
        note="angle errors > 15 deg are the planar-marker pose ambiguity (IPPE flip); the EKF rejects them with a "
             "chi-square gate instead of inflating R for every measurement")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(model, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = ["ahead error [cm]", "left error [cm]", "angle error [deg]"]
    scale = [100, 100, math.degrees(1)]
    dd = np.linspace(0.38, 3.6, 50)
    for k in range(3):
        ax[k].scatter(d, e[:, k] * scale[k], s=6, alpha=.35, c="#999", label="raw")
        ax[k].scatter(d, e_corr[:, k] * scale[k], s=6, alpha=.5, label="bias-corrected")
        ax[k].plot(dd, bias_c[k] * dd ** 2 * scale[k], "k--", lw=1, label="fitted bias")
        s = (coef[k][0] + coef[k][1] * dd ** 2) * scale[k]
        ax[k].plot(dd, 2 * s, "r", label="±2σ (R model)"); ax[k].plot(dd, -2 * s, "r")
        ax[k].set_xlabel("distance to marker [m]"); ax[k].set_title(names[k]); ax[k].grid(alpha=.3)
        lim = np.percentile(np.abs(e[:, k] * scale[k]), 97) * 1.3
        ax[k].set_ylim(-lim, lim)
    ax[0].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return model


if __name__ == "__main__":
    print(json.dumps(characterise(), indent=2))

"""Record a video of one docking run: true pose vs EKF estimate with its uncertainty. Evaluation only.
Run from the repo root:  python -m eval.make_video [condition]      (nominal | push | occlusion | combined ...)
Saves eval/results/docking_<condition>.mp4

Layout
  top-left : overhead view. Coloured line = TRUE path (colour = planner stage). White line = EKF ESTIMATE,
             white ellipse = 2-sigma position uncertainty from the EKF covariance P.
  top-right: the robot's camera; the marker is outlined green when detected.
  bottom   : estimation error (estimate - truth) in x, y and heading, with the EKF's +-2 sigma band.
"""
import os, sys, math
import cv2
import numpy as np
import imageio.v2 as imageio
from src.world import Sim, PARK_POSE
from src.contracts import CONTROL_DT
from src.estimator import Estimator
from src.planner import Planner
from src.control import Controller
from src.judge import Judge
from src.marker import detect_marker
from eval.run_episode import sense
from eval.truth_logger import TruthLogger
from eval.frames import MARKER_X, MARKER_Y, MARKER_YAW

TOP_W, TOP_H, CAM_W, PLOT_H = 720, 480, 640, 216
COLOURS = dict(search=(150, 150, 150), explore=(224, 160, 48), approach=(48, 112, 208), align=(160, 64, 192),
               dock=(48, 160, 80), square_up=(48, 160, 80), done=(48, 160, 80))
BG, FG = (46, 8, 18), (246, 238, 233)


def frame_origin(est, x0, y0, th0):
    return (MARKER_X, MARKER_Y, MARKER_YAW) if est.frame == "dock" else (x0, y0, th0)


def to_world(p, origin):
    ox, oy, oth = origin
    c, s = math.cos(oth), math.sin(oth)
    return ox + c * p[0] - s * p[1], oy + s * p[0] + c * p[1]


def ellipse_px(P, origin, judge, centre_w, n_sigma=2.0):
    """2-sigma covariance ellipse of the xy block of P, as a pixel polygon on the overhead view."""
    oth = origin[2]
    Rm = np.array([[math.cos(oth), -math.sin(oth)], [math.sin(oth), math.cos(oth)]])
    Pw = Rm @ P[:2, :2] @ Rm.T
    vals, vecs = np.linalg.eigh(Pw)
    vals = np.sqrt(np.maximum(vals, 0)) * n_sigma
    t = np.linspace(0, 2 * math.pi, 40)
    pts = (vecs @ np.diag(vals) @ np.vstack([np.cos(t), np.sin(t)])).T + np.array(centre_w)
    return np.array([judge.world_to_px(px, py, TOP_W, TOP_H) for px, py in pts], np.int32)


def error_panel(hist, t_now, width, events=()):
    """Three small charts: error and +-2 sigma for x, y, heading."""
    panel = np.full((PLOT_H, width, 3), 250, np.uint8)
    cw = width // 3
    specs = [("x error [cm]", 0, 100, 30.0), ("y error [cm]", 1, 100, 30.0), ("heading error [deg]", 2, 180 / math.pi, 20.0)]
    t_max = max(t_now, 20.0)
    for k, (name, i, scale, lim) in enumerate(specs):
        x0, x1, y0, y1 = k * cw + 46, (k + 1) * cw - 14, 34, PLOT_H - 26
        cv2.rectangle(panel, (x0, y0), (x1, y1), (205, 205, 205), 1)
        cv2.putText(panel, name, (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        ymid = (y0 + y1) // 2
        cv2.line(panel, (x0, ymid), (x1, ymid), (205, 205, 205), 1)
        for lab, yy in ((f"+{lim:g}", y0), (f"-{lim:g}", y1)):
            cv2.putText(panel, lab, (k * cw + 4, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
        for te, lab in events:
            xe = int(x0 + (x1 - x0) * te / max(t_now, 20.0))
            cv2.line(panel, (xe, y0), (xe, y1), (150, 150, 150), 1)
            cv2.putText(panel, lab, (xe + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
        if len(hist) < 2:
            continue
        tx = lambda t: int(x0 + (x1 - x0) * t / t_max)
        ty = lambda v: int(np.clip(ymid - (y1 - y0) / 2 * v / lim, y0, y1))
        band_top = [(tx(h[0]), ty(2 * h[4 + i] * scale)) for h in hist]
        band_bot = [(tx(h[0]), ty(-2 * h[4 + i] * scale)) for h in hist][::-1]
        cv2.fillPoly(panel, [np.array(band_top + band_bot, np.int32)], (214, 226, 246))
        cv2.polylines(panel, [np.array([(tx(h[0]), ty(h[1 + i] * scale)) for h in hist], np.int32)], False,
                      (40, 80, 170), 2, cv2.LINE_AA)
    cv2.putText(panel, "blue line: estimate - truth     shaded: EKF +-2 sigma     (before the marker is seen the "
                "filter runs on odometry only, so the band grows)", (12, PLOT_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return panel


def make_video(condition="nominal", start=None, out=None, max_time=150.0, every=2, fps=30):
    from eval.run_batch import make_hooks
    out = out or f"eval/results/docking_{condition}.mp4"
    obs_hook, step_hook = make_hooks(condition, 0)
    sim = Sim() if start is None else Sim(start=start)
    cal = sim.sensors.calibration
    est, planner, ctrl, judge = Estimator(cal), Planner(cal), Controller(cal), Judge(sim)
    truth = TruthLogger(sim)
    x0, y0, th0 = judge.true_pose()
    last, tick, path, est_path, hist, frames = dict(lidar=None, image=None), 0, [], [[]], [], []
    events, seen_frame_changes = [], 0
    gx, gy = judge.world_to_px(*PARK_POSE[:2], TOP_W, TOP_H)

    while sim.time < max_time:
        obs, last = sense(sim, tick, last)
        obs = obs_hook(obs, sim)
        pose = est.update(obs)
        plan = planner.update(pose, obs, est.dock)
        v, w = ctrl.update(pose, plan)
        truth.record(sim.time, pose, (v, w), est.frame, est.P)
        r = truth.rows[-1]
        tx_, ty_, _ = judge.true_pose()
        origin = frame_origin(est, x0, y0, th0)
        ew = to_world(pose, origin)
        path.append((*judge.world_to_px(tx_, ty_, TOP_W, TOP_H), plan.mode))
        if est.frame_changes != seen_frame_changes:           # new frame -> new estimate line (no false jump)
            seen_frame_changes = est.frame_changes
            est_path.append([])
            events.append((sim.time, "marker fix" if seen_frame_changes == 1 else "re-fix"))
        hs = obs_hook.state
        if hs["push_t"] is not None and not any(l == "push" for _, l in events):
            events.append((hs["push_t"], "push"))
        if hs["occl_start"] is not None and not any(l.startswith("blocked") for _, l in events):
            events.append((hs["occl_start"], "blocked 6 s"))
        est_path[-1].append(judge.world_to_px(*ew, TOP_W, TOP_H))
        hist.append((sim.time, r["est_x"] - r["true_x"], r["est_y"] - r["true_y"],
                     (r["est_th"] - r["true_th"] + math.pi) % (2 * math.pi) - math.pi,
                     r["sd_x"], r["sd_y"], r["sd_th"]))

        if tick % every == 0 or plan.done:
            top = judge.render_topdown(TOP_W, TOP_H)
            for (a0, b0, _), (a1, b1, m) in zip(path, path[1:]):
                cv2.line(top, (int(a0), int(b0)), (int(a1), int(b1)), COLOURS.get(m, (0, 0, 0)), 3)
            for seg in est_path:
                if len(seg) > 1:
                    cv2.polylines(top, [np.array(seg, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(top, [ellipse_px(est.P, origin, judge, ew)], True, (255, 255, 255), 2, cv2.LINE_AA)
            ex, ey = est_path[-1][-1]
            cv2.circle(top, (int(ex), int(ey)), 4, (255, 255, 255), -1)
            cv2.drawMarker(top, (int(gx), int(gy)), (220, 30, 30), cv2.MARKER_CROSS, 18, 2)
            cam = obs["image"].copy() if obs.get("image") is not None else np.full((480, CAM_W, 3), 40, np.uint8)
            if obs.get("image") is None:
                cv2.putText(cam, "camera blocked", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2)
            else:
                res = detect_marker(cam, est.K, cal["marker"], est.dist)
                if res is not None:
                    cv2.polylines(cam, [res[0].astype(np.int32)], True, (0, 220, 0), 3)
            frame = np.vstack([np.hstack([top, cam]), error_panel(hist, sim.time, TOP_W + CAM_W, events)])
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), BG, -1)
            sd = math.hypot(r["sd_x"], r["sd_y"])
            label = (f"{condition}   t = {sim.time:4.1f} s   stage: {plan.mode:9s}  frame: {est.frame:5s}  "
                     f"pos error {r['err_pos']*100:5.1f} cm  (EKF 1-sigma {sd*100:5.1f} cm)")
            cv2.putText(frame, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 2, cv2.LINE_AA)
            frames.append(frame)

        if plan.done:
            break
        step_hook(sim)
        sim.drive(v, w, CONTROL_DT)
        tick += 1

    err = judge.park_error()
    end = frames[-1].copy()
    msg = (f"PARKED  -  parking error {err['pos_m']*100:.1f} cm, {err['heading_deg']:.1f} deg    "
           f"final estimate error {truth.rows[-1]['err_pos']*1000:.0f} mm") if err["parked"] else "NOT PARKED"
    cv2.rectangle(end, (0, TOP_H - 50), (end.shape[1], TOP_H), BG, -1)
    cv2.putText(end, msg, (12, TOP_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85, FG, 2, cv2.LINE_AA)
    frames += [end] * (2 * fps)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    judge.close(); sim.close()
    return out, err


if __name__ == "__main__":
    print(make_video(sys.argv[1] if len(sys.argv) > 1 else "nominal"))

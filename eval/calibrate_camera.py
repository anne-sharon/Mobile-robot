"""Calibrate the robot's camera with a checkerboard (Zhang's method), like you would a real Pi camera.
Run from the repo root:  python -m eval.calibrate_camera
Writes models/camera_calibration.json, which estimator.py loads instead of the datasheet camera matrix.

A separate MuJoCo scene holds a camera with the same resolution and field of view as the robot's front camera
and a printed checkerboard that is moved to many random poses in front of it. OpenCV finds the inner corners in
each image and cv2.calibrateCamera solves for the focal lengths, principal point and lens distortion.
"""
import os, json, math
import numpy as np, cv2, mujoco
from src.world import CAM_W, CAM_H, CAM_FOVY, camera_intrinsics

ROWS, COLS, SQUARE = 6, 9, 0.025        # inner corners, square size [m]


def board_png(px=60):
    img = np.full(((ROWS + 3) * px, (COLS + 3) * px), 255, np.uint8)          # 1-square white margin
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                img[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    return cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()


def build_scene():
    w, h = (COLS + 3) * SQUARE / 2, (ROWS + 3) * SQUARE / 2
    xml = f"""
<mujoco><compiler angle="degree"/>
  <visual><global offwidth="{CAM_W}" offheight="{CAM_H}"/><headlight ambient=".6 .6 .6" diffuse=".4 .4 .4"/></visual>
  <asset><texture name="b" type="2d" file="board.png"/><material name="b" texture="b"/></asset>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
    <body name="board" mocap="true" pos="0.5 0 0">
      <geom type="plane" size="{w} {h} 0.01" material="b" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml, {"board.png": board_png()})
    return m, mujoco.MjData(m)


def euler_quat(roll, pitch, yaw):
    q = np.zeros(4); mujoco.mju_euler2Quat(q, [roll, pitch, yaw], "xyz"); return q


def calibrate(n_views=40, seed=0, out="models/camera_calibration.json"):
    m, d = build_scene()
    r = mujoco.Renderer(m, CAM_H, CAM_W)
    rng = np.random.default_rng(seed)
    obj = np.zeros((ROWS * COLS, 3), np.float32)
    obj[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    obj_pts, img_pts, tries = [], [], 0
    while len(img_pts) < n_views and tries < 400:
        tries += 1
        dist = rng.uniform(0.35, 0.9)
        d.mocap_pos[0] = (dist, rng.uniform(-0.25, 0.25) * dist, rng.uniform(-0.18, 0.18) * dist)
        d.mocap_quat[0] = euler_quat(*np.deg2rad(rng.uniform(-35, 35, 3)))
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="cam")
        gray = cv2.cvtColor(r.render(), cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        obj_pts.append(obj); img_pts.append(corners)

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (CAM_W, CAM_H), None, None)
    K_true = camera_intrinsics()
    result = dict(
        camera_matrix=K.tolist(), dist_coeffs=dist.ravel().tolist(),
        rms_reprojection_px=float(rms), views_used=len(img_pts), views_tried=tries,
        image_size=[CAM_W, CAM_H], board=dict(inner_corners=[COLS, ROWS], square_m=SQUARE),
        datasheet_camera_matrix=K_true.tolist(),
        focal_error_pct=float(100 * abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0]),
        principal_point_error_px=float(math.hypot(K[0, 2] - K_true[0, 2], K[1, 2] - K_true[1, 2])))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    res = calibrate()
    print(json.dumps({k: res[k] for k in ("rms_reprojection_px", "views_used", "focal_error_pct",
                                          "principal_point_error_px", "dist_coeffs")}, indent=2))
    print("K =", np.round(res["camera_matrix"], 2).tolist())

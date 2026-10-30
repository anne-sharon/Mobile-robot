"""The simulated world: room, boxes, dock + ArUco marker, TurtleBot3 Burger, and its sensors.
Ground truth is NOT here -- see judge.py (evaluation only)."""
import os, shutil, math, tempfile, urllib.request
os.environ.setdefault("MUJOCO_GL", "egl" if shutil.which("nvidia-smi") else "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import numpy as np, mujoco, cv2
assert hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"), "need OpenCV >= 4.7 (pip install -U opencv-python)"

# =====================================================================
#  CONSTANTS  (metres / radians, world frame: x east, y north, z up)
# =====================================================================
# --- room: 6 m x 4 m, centred on the origin ---
ROOM_X, ROOM_Y, WALL_H = 3.0, 2.0, 1.0          # half-sizes
BOXES = [  # name, (x, y), half-sizes (x, y, z), yaw [deg], rgba   (0.4 m tall -> LiDAR sees them)
    ("box1", (-1.0, -0.85), (0.20, 0.20, 0.20),  20, (0.85, 0.25, 0.20, 1)),
    ("box2", ( 0.1,  0.55), (0.25, 0.15, 0.20), -15, (0.20, 0.35, 0.85, 1)),
    ("box3", ( 1.2, -0.15), (0.15, 0.15, 0.20),  35, (0.25, 0.70, 0.30, 1)),
]
# --- living-room furniture (solid down to the floor, so the LiDAR sees them) ---
SOFA  = dict(pos=(-1.3, 1.55), half=(0.90, 0.35, 0.40), rgba=(0.45, 0.30, 0.22, 1))   # 1.8 m x 0.7 m, 0.8 m tall
TABLE = dict(pos=(-1.3, 0.75), radius=0.30, half_h=0.225, rgba=(0.62, 0.45, 0.30, 1))  # 0.6 m wide, 0.45 m tall

# --- charging station: back plate on the east wall, facing west (-x), marker on its face ---
DOCK_Y       = 0.30
DOCK_FACE_X  = ROOM_X - 0.02                     # x of the back plate's front face
ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_ID    = 7
MODULE       = 0.03                              # one marker cell; black marker = 6 cells
MARKER_SIZE  = 6 * MODULE                        # 0.18 m, edge of the black border
SHEET_SIZE   = 8 * MODULE                        # printed sheet incl. 1-cell white margin
MARKER_Z     = 0.12                              # height of the marker centre

# --- where the robot should end up (used only for scoring) ---
PARK_DIST    = 0.35                              # base_link -> marker plane
PARK_POSE    = (DOCK_FACE_X - PARK_DIST, DOCK_Y, 0.0)
PARK_TOL     = (0.05, math.radians(5))           # position [m], heading [rad]

# --- TurtleBot3 Burger (numbers from turtlebot3_burger.urdf) ---
WHEEL_R, WHEEL_SEP, BASE_Z = 0.033, 0.160, 0.010
V_MAX, W_MAX = 0.22, 2.84                        # Burger speed limits
MOTOR_KV = 1.0                                   # wheel speed-controller gain [N*m*s/rad]
A_LIN, A_ANG = 1.0, 6.0                          # acceleration limits [m/s^2], [rad/s^2] (keeps wheels from slipping)
START_POSE   = (-2.3, -1.2, 2.4)                 # x, y, yaw: facing away from the dock on purpose

# --- sensors ---
LIDAR_N, LIDAR_MIN, LIDAR_MAX = 360, 0.12, 3.5   # LDS-01: 1 deg steps, 0.12-3.5 m
CAM_W, CAM_H, CAM_HFOV = 640, 480, 62.2          # Pi-camera-like
CAM_POS   = (0.040, 0.0, 0.100)                  # camera in the base_footprint frame
LIDAR_POS = (-0.032, 0.0, 0.182)                 # LiDAR in the base_footprint frame
TOP_H, TOP_FOVY = 7.0, 40.0                      # overhead camera (for scoring / plots only)

# =====================================================================
#  ASSETS: TurtleBot3 meshes (downloaded) and the ArUco texture (generated)
# =====================================================================
_MESH_URL = "https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3/humble/turtlebot3_description/meshes/"
_MESHES = {"burger_base.stl": "bases/burger_base.stl", "lds.stl": "sensors/lds.stl",
           "left_tire.stl": "wheels/left_tire.stl", "right_tire.stl": "wheels/right_tire.stl"}

def fetch_meshes(cache=os.path.join(tempfile.gettempdir(), "tb3_meshes")):
    os.makedirs(cache, exist_ok=True)
    out = {}
    for name, rel in _MESHES.items():
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(_MESH_URL + rel, path)
        with open(path, "rb") as f:
            out[name] = f.read()
    return out

def make_marker_png(px_per_cell=64):
    """The 'printed sheet': ArUco marker + 1 cell white margin, as PNG bytes."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(d, MARKER_ID, 6 * px_per_cell)
    img = cv2.copyMakeBorder(img, px_per_cell, px_per_cell, px_per_cell, px_per_cell,
                             cv2.BORDER_CONSTANT, value=255)
    return cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()

# =====================================================================
#  MJCF
# =====================================================================
def camera_intrinsics():
    fy = (CAM_H / 2) / math.tan(math.radians(CAM_FOVY) / 2)
    return np.array([[fy, 0, CAM_W / 2], [0, fy, CAM_H / 2], [0, 0, 1.0]])

CAM_FOVY = math.degrees(2 * math.atan(math.tan(math.radians(CAM_HFOV / 2)) * CAM_H / CAM_W))

def build_xml(use_meshes):
    walls = {  # name: (x, y, half_x, half_y, rgba)
        "east":  ( ROOM_X + .05, 0, .05, ROOM_Y + .1, ".93 .90 .84 1"),
        "west":  (-ROOM_X - .05, 0, .05, ROOM_Y + .1, ".84 .90 .93 1"),
        "north": (0,  ROOM_Y + .05, ROOM_X + .1, .05, ".90 .93 .84 1"),
        "south": (0, -ROOM_Y - .05, ROOM_X + .1, .05, ".93 .86 .88 1")}
    wall_xml = "\n".join(
        f'<geom name="wall_{n}" type="box" pos="{x} {y} {WALL_H/2}" size="{sx} {sy} {WALL_H/2}" rgba="{c}"/>'
        for n, (x, y, sx, sy, c) in walls.items())
    box_xml = "\n".join(
        f'<geom name="{n}" type="box" pos="{x} {y} {h[2]}" size="{h[0]} {h[1]} {h[2]}" '
        f'euler="0 0 {yaw}" rgba="{" ".join(map(str, c))}"/>'
        for n, (x, y), h, yaw, c in BOXES)
    rgba = lambda c: " ".join(map(str, c))
    sx, sy = SOFA["pos"]; shx, shy, shz = SOFA["half"]
    tx, ty = TABLE["pos"]
    furniture_xml = (
        f'<geom name="sofa" type="box" pos="{sx} {sy} {shz}" size="{shx} {shy} {shz}" rgba="{rgba(SOFA["rgba"])}"/>'
        f'<geom name="coffee_table" type="cylinder" pos="{tx} {ty} {TABLE["half_h"]}" '
        f'size="{TABLE["radius"]} {TABLE["half_h"]}" rgba="{rgba(TABLE["rgba"])}"/>')

    if use_meshes:
        asset_mesh = "".join(f'<mesh name="{n[:-4]}" file="{n}" scale="0.001 0.001 0.001"/>' for n in _MESHES)
        vis = 'contype="0" conaffinity="0" group="1"'
        base_vis = f'<geom type="mesh" mesh="burger_base" pos="-0.032 0 0" rgba=".45 .45 .47 1" {vis}/>' \
                   f'<geom type="mesh" mesh="lds" pos="-0.032 0 0.172" rgba=".2 .2 .22 1" {vis}/>'
        wl_vis = f'<geom type="mesh" mesh="left_tire" rgba=".15 .15 .15 1" {vis}/>'
        wr_vis = f'<geom type="mesh" mesh="right_tire" rgba=".15 .15 .15 1" {vis}/>'
    else:  # fallback if the download fails: simple shapes with the same footprint
        asset_mesh, vis = "", 'contype="0" conaffinity="0" group="1"'
        base_vis = f'<geom type="box" pos="-0.032 0 0.07" size="0.07 0.07 0.07" rgba=".45 .45 .47 1" {vis}/>' \
                   f'<geom type="cylinder" pos="-0.032 0 0.172" size="0.05 0.02" rgba=".2 .2 .22 1" {vis}/>'
        wl_vis = wr_vis = ""

    return f"""
<mujoco model="dock_room">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient=".45 .45 .45" diffuse=".55 .55 .55" specular="0 0 0"/>
    <map znear="0.002"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".80 .82 .86" rgb2=".55 .57 .62" width="64" height="64"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1=".80 .80 .78" rgb2=".68 .68 .66" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="3 3" specular="0" reflectance="0"/>
    <texture name="aruco_tex" type="2d" file="aruco.png"/>
    <material name="aruco_mat" texture="aruco_tex" specular="0" shininess="0" reflectance="0"/>
    {asset_mesh}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse=".35 .35 .35" specular="0 0 0" castshadow="false"/>
    <camera name="topdown" pos="0 0 {TOP_H}" xyaxes="1 0 0 0 1 0" fovy="{TOP_FOVY}"/>
    <geom name="floor" type="plane" size="{ROOM_X} {ROOM_Y} 0.1" material="floor_mat"/>
    {wall_xml}
    {box_xml}
    {furniture_xml}

    <!-- charging station: back plate + printed marker sheet + floor decal -->
    <geom name="dock_plate" type="box" pos="{DOCK_FACE_X + .01} {DOCK_Y} .15" size=".01 .15 .15" rgba=".25 .27 .30 1"/>
    <geom name="dock_marker" type="plane" pos="{DOCK_FACE_X - .001} {DOCK_Y} {MARKER_Z}"
          xyaxes="0 -1 0 0 0 1" size="{SHEET_SIZE/2} {SHEET_SIZE/2} 0.01" material="aruco_mat"
          contype="0" conaffinity="0"/>
    <geom name="dock_pad" type="plane" pos="{DOCK_FACE_X - .25} {DOCK_Y} .0015" size=".25 .18 .01"
          rgba=".25 .75 .35 1" contype="0" conaffinity="0"/>

    <!-- TurtleBot3 Burger (differential drive, rear caster) -->
    <body name="base_link" pos="0 0 {BASE_Z + 0.0005}">
      <freejoint name="root"/>
      <inertial pos="-0.02 0 0" mass="0.826" diaginertia="2.21e-3 2.12e-3 2.01e-3"/>
      <geom name="chassis" type="box" pos="-0.032 0 0.070" size="0.070 0.070 0.0715" group="3"/>
      <geom name="caster"  type="sphere" pos="-0.081 0 0" size="0.01" friction="0.05 0.005 0.0001" group="3"/>
      <geom name="skid"    type="sphere" pos="0.028 0 0.0015" size="0.01" friction="0.05 0.005 0.0001" group="3"/>
      {base_vis}
      <site name="lidar_site" pos="-0.032 0 0.172" size="0.005" group="4"/>
      <site name="imu_site"   pos="-0.032 0 0.068" size="0.005" group="4"/>
      <camera name="front_cam" pos="{CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2] - BASE_Z}"
              xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
      <body name="wheel_left" pos="0 0.08 0.023">
        <joint name="wheel_left_joint" axis="0 1 0" damping="0.001" armature="5e-4"/>
        <inertial pos="0 0 0" mass="0.0285" diaginertia="1.1175e-5 2.0713e-5 1.1192e-5"/>
        <geom type="cylinder" euler="90 0 0" size="{WHEEL_R} 0.003" friction="1.2 0.005 0.0001" group="3"/>
        {wl_vis}
      </body>
      <body name="wheel_right" pos="0 -0.08 0.023">
        <joint name="wheel_right_joint" axis="0 1 0" damping="0.001" armature="5e-4"/>
        <inertial pos="0 0 0" mass="0.0285" diaginertia="1.1175e-5 2.0713e-5 1.1192e-5"/>
        <geom type="cylinder" euler="90 0 0" size="{WHEEL_R} 0.003" friction="1.2 0.005 0.0001" group="3"/>
        {wr_vis}
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="motor_left"  joint="wheel_left_joint"  kv="{MOTOR_KV}" ctrlrange="-15 15" forcerange="-0.3 0.3"/>
    <velocity name="motor_right" joint="wheel_right_joint" kv="{MOTOR_KV}" ctrlrange="-15 15" forcerange="-0.3 0.3"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu_site"/>
    <accelerometer name="accel" site="imu_site"/>
  </sensor>
</mujoco>"""

# =====================================================================
#  SENSORS  -- the only thing a controller is allowed to look at
# =====================================================================
class Sensors:
    """LiDAR, camera, wheel encoders, IMU.  Contains NO ground-truth pose."""
    def __init__(self, sim, lidar_noise=0.01, seed=0):
        self._sim, self._rng, self.lidar_noise = sim, np.random.default_rng(seed), lidar_noise
        m = sim.model
        self._lidar_site = m.site("lidar_site").id
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # rays only hit world geoms (group 0)
        self.lidar_angles = np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False)  # CCW from robot +x
        self._cam_renderer = mujoco.Renderer(m, CAM_H, CAM_W)
        self._cam_opt = mujoco.MjvOption()
        self._cam_opt.geomgroup[1] = 0                                    # camera can't see the robot's own body
        self._qadr = [m.jnt_qposadr[m.joint(n).id] for n in ("wheel_left_joint", "wheel_right_joint")]
        # calibration: what a real robot would know from its datasheet / a calibration routine
        self.calibration = dict(
            wheel_radius=WHEEL_R, wheel_separation=WHEEL_SEP,
            camera_matrix=camera_intrinsics(), camera_pos=CAM_POS,      # camera optical axis = robot +x
            lidar_pos=LIDAR_POS, lidar_range=(LIDAR_MIN, LIDAR_MAX),
            marker=dict(dictionary=ARUCO_DICT, id=MARKER_ID, size=MARKER_SIZE))

    def lidar(self):
        """360 ranges [m], index i = i degrees CCW from the front. np.inf = no return."""
        sim = self._sim
        mujoco.mj_kinematics(sim.model, sim.data)
        p = sim.data.site_xpos[self._lidar_site].copy()
        R = sim.data.site_xmat[self._lidar_site].reshape(3, 3)
        dirs = R @ np.stack([np.cos(self.lidar_angles), np.sin(self.lidar_angles), np.zeros(LIDAR_N)])
        out, geomid = np.full(LIDAR_N, np.inf), np.zeros(1, dtype=np.int32)
        for i in range(LIDAR_N):
            d = mujoco.mj_ray(sim.model, sim.data, p, dirs[:, i], self._ray_groups, 1, -1, geomid)
            if LIDAR_MIN <= d <= LIDAR_MAX:
                out[i] = d + self._rng.normal(0, self.lidar_noise)
        return out

    def camera(self):
        """RGB uint8 image, shape (480, 640, 3)."""
        sim = self._sim
        mujoco.mj_kinematics(sim.model, sim.data); mujoco.mj_camlight(sim.model, sim.data)
        self._cam_renderer.update_scene(sim.data, camera="front_cam", scene_option=self._cam_opt)
        return self._cam_renderer.render().copy()

    def encoders(self):
        """Cumulative wheel angles [rad] (left, right) -> dead-reckoning odometry."""
        q = self._sim.data.qpos
        return float(q[self._qadr[0]]), float(q[self._qadr[1]])

    def imu(self):
        """(yaw rate [rad/s], accel xyz [m/s^2]) in the robot frame."""
        s = self._sim.data.sensordata
        return float(s[2]), s[3:6].copy()

# =====================================================================
#  SIM
# =====================================================================
class Sim:
    def __init__(self, start=START_POSE, lidar_noise=0.01, seed=0):
        try:
            meshes = fetch_meshes()
        except Exception as e:
            print("TurtleBot3 mesh download failed (", e, ") -> using simple shapes for the body")
            meshes = None
        assets = {"aruco.png": make_marker_png(), **(meshes or {})}
        self.model = mujoco.MjModel.from_xml_string(build_xml(meshes is not None), assets)
        self.data = mujoco.MjData(self.model)
        self._ctrl = [self.model.actuator(n).id for n in ("motor_left", "motor_right")]
        self.sensors = Sensors(self, lidar_noise, seed)
        self.reset(*start)

    def close(self):
        """Free the offscreen renderers (lets many episodes run in one Python session)."""
        self.sensors._cam_renderer.close()

    @property
    def time(self):
        return self.data.time

    def reset(self, x=START_POSE[0], y=START_POSE[1], yaw=START_POSE[2]):
        """Place the robot (initial condition only, not a measurement)."""
        mujoco.mj_resetData(self.model, self.data)
        self._v = self._w = 0.0
        self.data.qpos[0:3] = (x, y, BASE_Z + 0.0005)
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):                      # let it settle on its wheels
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    def drive(self, v, w, duration=0.1):
        """Command forward speed v [m/s] and turn rate w [rad/s] (CCW+) for `duration` seconds.
        The commanded speeds are ramped with A_LIN / A_ANG, like a real motor controller would."""
        v, w = float(np.clip(v, -V_MAX, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))
        dt = self.model.opt.timestep
        for _ in range(max(1, round(duration / dt))):
            self._v += np.clip(v - self._v, -A_LIN * dt, A_LIN * dt)
            self._w += np.clip(w - self._w, -A_ANG * dt, A_ANG * dt)
            self.data.ctrl[self._ctrl[0]] = (self._v - self._w * WHEEL_SEP / 2) / WHEEL_R
            self.data.ctrl[self._ctrl[1]] = (self._v + self._w * WHEEL_SEP / 2) / WHEEL_R
            mujoco.mj_step(self.model, self.data)

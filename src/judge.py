"""Ground truth. For SCORING and PLOTS only -- estimator.py, planner.py, control.py, marker.py and contracts.py must never import this file."""
import math
import mujoco
from .world import PARK_POSE, PARK_TOL, TOP_H, TOP_FOVY

class Judge:
    """True pose lives here. Never feed any of this back into the controller."""
    def __init__(self, sim):
        self._sim = sim
        self._renderer = None

    def true_pose(self):
        q = self._sim.data.qpos
        w, x, y, z = q[3:7]
        return float(q[0]), float(q[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def park_error(self):
        x, y, yaw = self.true_pose()
        pos = math.hypot(x - PARK_POSE[0], y - PARK_POSE[1])
        head = abs((yaw - PARK_POSE[2] + math.pi) % (2 * math.pi) - math.pi)
        return dict(pos_m=pos, heading_deg=math.degrees(head),
                    parked=bool(pos <= PARK_TOL[0] and head <= PARK_TOL[1]))

    def collided(self):
        """True if any part of the robot is currently touching a box, wall or the dock (not the floor)."""
        m, d = self._sim.model, self._sim.data
        root, floor = m.body("base_link").id, m.geom("floor").id
        def mine(g): return m.body_rootid[m.geom_bodyid[g]] == root
        return any(mine(c.geom1) != mine(c.geom2) and floor not in (c.geom1, c.geom2)
                   for c in d.contact[:d.ncon])

    def render_topdown(self, width=1200, height=800):
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self._sim.model, height, width)
        mujoco.mj_forward(self._sim.model, self._sim.data)
        self._renderer.update_scene(self._sim.data, camera="topdown")
        return self._renderer.render().copy()

    def world_to_px(self, x, y, width=1200, height=800):
        """Floor-level world point -> pixel in render_topdown()."""
        f = (height / 2) / math.tan(math.radians(TOP_FOVY) / 2)
        return width / 2 + f * x / TOP_H, height / 2 - f * y / TOP_H

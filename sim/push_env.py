"""
Box-pushing scene: a Panda pushes a box on a table with a fingertip-sized ball on a stick.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

BOX_SIZE = (0.128, 0.128, 0.057)  # metres: 12.8 cm square , 5.7 cm tal
BOX_MASS = 0.2                 # kg. Slow pushes barely depend on it.
FRICTION = 0.4                 # box-table sliding friction, roughly cardboard on wood
TABLE_Z = 0.8                  # table top height in robosuite's Lift scene

TIP_RADIUS = 0.008             # the ball at the end of the stick: 1.6 cm across, about a fingertip
TIP_HEIGHT = BOX_SIZE[2] / 2   # the ball's centre pushes at the box's mid-height, as you do
STICK_RADIUS = 0.004           # thinner than the ball, so only the ball ever touches the box
STICK_LENGTH = BOX_SIZE[2] + 0.02  # gripper to ball centre; keeps the fingers ~3.5 cm above the box
PUSH_Z = TABLE_Z + TIP_HEIGHT + STICK_LENGTH  # gripper height that puts the ball's centre at TIP_HEIGHT
SAFE_Z = PUSH_Z + BOX_SIZE[2] + 0.03     # travel height: the ball clears the box by 3 cm
OSC_STEP = 0.05                # metres per unit action in robosuite's OSC_POSITION controller
GRIPPER_CLOSED = 1.0

# Where the box's centre may wander while pushing. Outside it, the arm is near the edge of
# its reach and follows the path less precisely (measured: within 2 mm inside, up to 2 cm outside).
# (Measured with a 7 cm box; shrunk by the 2.9 cm the bigger box adds on each side.)
WORKSPACE = dict(x=(-0.05, 0.07), y=(-0.10, 0.10))


class PushBox(Lift):
    """robosuite's Lift scene with its small cube swapped for a box shaped like the real one,
    and a stick with a ball on its end added below the gripper."""

    def _load_model(self):
        super(Lift, self)._load_model()  # loads the robot; we build the rest ourselves
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](self.table_full_size[0]))
        eef = robot.worldbody.find(".//body[@name='gripper0_eef']")
        # The gripper's z axis points down, so "0 0 L" is L metres below the gripper.
        # Orange, and in group 1 so cameras draw them (they still collide).
        ET.SubElement(eef, "geom", name="gripper0_stick", type="capsule",
                      fromto=f"0 0 0 0 0 {STICK_LENGTH}", size=str(STICK_RADIUS),
                      rgba="1 0.5 0.1 1", group="1")
        ET.SubElement(eef, "geom", name="gripper0_tip", type="sphere",
                      pos=f"0 0 {STICK_LENGTH}", size=str(TIP_RADIUS),
                      friction="0.8 0.005 0.0001", rgba="1 0.5 0.1 1", group="1")

        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # A close-up camera beside the table, looking down at about 40 degrees.
        # The arm reaches in from the left of its picture.
        ET.SubElement(arena.worldbody, "camera", mode="fixed", name="pushview",
                      pos="0.02 -0.48 1.22", xyaxes="1 0 0 0 0.656 0.755", fovy="45")

        # Named "cube" so Lift's references and observations keep working.
        self.cube = BoxObject(name="cube",
                              size=[s / 2 for s in BOX_SIZE],
                              rgba=[0.15, 0.35, 0.85, 1],
                              density=BOX_MASS / np.prod(BOX_SIZE),
                              friction=(FRICTION, 5e-3, 1e-4))
        self.placement_initializer = UniformRandomSampler(
            name="BoxSampler", mujoco_objects=self.cube,
            x_range=[-0.03, 0.05], y_range=[-0.07, 0.07], rotation=None,  # None = random yaw
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.002)

        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[robot],
                                      mujoco_objects=self.cube)

    def _check_success(self):
        return False  # success is judged by our own code, not by Lift's "cube lifted" test

    def box_pose(self):
        """(x, y, yaw) of the box: metres and radians, in the table's frame."""
        pos = self.sim.data.body_xpos[self.cube_body_id]
        w, x, y, z = self.sim.data.body_xquat[self.cube_body_id]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([pos[0], pos[1], yaw])

    def tip(self):
        """Gripper reference point (x, y, z). The stick hangs straight below it, so the ball has the same x, y."""
        return self.sim.data.site_xpos[self.robots[0].eef_site_id].copy()

    def set_box_pose(self, x, y, yaw):
        """Teleport the box (to reset between pushes without reloading the scene)."""
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # MuJoCo order: w, x, y, z
        z = TABLE_Z + BOX_SIZE[2] / 2 + 0.001
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([[x, y, z], quat]))
        self.sim.data.set_joint_qvel(self.cube.joints[0], np.zeros(6))
        self.sim.forward()


def make_env(render=False, camera="pushview", size=(320, 240), env_class=PushBox):
    cfg = load_controller_config(default_controller="OSC_POSITION")
    env = env_class(robots="Panda", controller_configs=cfg,
                  table_friction=(FRICTION, 5e-3, 1e-4),
                  has_renderer=False, has_offscreen_renderer=render, use_camera_obs=False,
                  control_freq=20, horizon=10**7, ignore_done=True, hard_reset=False)
    env.render_camera, env.render_size = camera, size
    env.reset()
    for _ in range(10):  # close the gripper before doing anything
        env.step(np.array([0, 0, 0, GRIPPER_CLOSED]))
    return env


def move_to(env, target, speed, on_step=None, settle=6):
    """Drive the gripper in a straight line to `target` (x, y, z) at `speed` m/s.

    A waypoint slides along the line at the requested speed, and each step asks the
    controller to close the gap to it. `on_step` is called after every step.
    """
    start = env.tip()
    n = max(1, int(np.ceil(np.linalg.norm(target - start) / (speed / env.control_freq))))
    for i in range(1, n + settle + 1):
        waypoint = start + (target - start) * min(i / n, 1.0)
        delta = np.clip((waypoint - env.tip()) / OSC_STEP, -1, 1)
        env.step(np.append(delta, GRIPPER_CLOSED))
        if on_step:
            on_step(env)


def push_path(box_pose, side, offset, angle, length, clearance=0.02):
    """Start and end (x, y in the table frame) of one straight push of the ball's centre.

    The push is defined in the box's frame:
    side    0..3, which face of the box to push (counter-clockwise from the box's +x face)
    offset  -1..1, where the ball first touches that face (0 = middle; +-1 = 80% of the
            way to a corner, so the ball doesn't start on the corner itself)
    angle   radians, push direction relative to straight-in (positive = counter-clockwise)
    length  metres the ball travels past the moment it touches the face
    """
    half = BOX_SIZE[0] / 2
    k = side * np.pi / 2
    normal = np.array([np.cos(k), np.sin(k)])            # outward normal of that face
    along = np.array([-np.sin(k), np.cos(k)])            # direction along that face
    contact = normal * half + along * offset * half * 0.8
    c, s = np.cos(angle), np.sin(angle)
    direction = np.array([[c, -s], [s, c]]) @ -normal    # straight-in, turned by `angle`
    touch = contact + normal * TIP_RADIUS                # ball centre when its edge reaches `contact`

    x, y, yaw = box_pose
    cy, sy = np.cos(yaw), np.sin(yaw)
    to_table = np.array([[cy, -sy], [sy, cy]])
    start = np.array([x, y]) + to_table @ (touch - direction * clearance)
    end = np.array([x, y]) + to_table @ (touch + direction * length)
    return start, end


def push(env, side, offset, angle, length, speed=0.04, on_step=None):
    """Execute one push: move above a point just outside the box, lower, push in a straight line, lift.
    on_step(env, phase) is called after every control step, with phase "travel" or "push" (used to
    record GIF frames)."""
    start, end = push_path(env.box_pose(), side, offset, angle, length)
    travel = (lambda e: on_step(e, "travel")) if on_step else None
    pushing = (lambda e: on_step(e, "push")) if on_step else None

    move_to(env, np.array([*env.tip()[:2], SAFE_Z]), speed=0.25, on_step=travel)
    move_to(env, np.array([*start, SAFE_Z]), speed=0.25, on_step=travel)
    move_to(env, np.array([*start, PUSH_Z]), speed=0.15, on_step=travel)
    move_to(env, np.array([*end, PUSH_Z]), speed=speed, on_step=pushing)
    move_to(env, np.array([*end, SAFE_Z]), speed=0.15, on_step=travel)


def render(env):
    w, h = env.render_size
    return env.sim.render(camera_name=env.render_camera, width=w, height=h)[::-1]

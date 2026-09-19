"""The lamp itself, in MuJoCo: its real shape, where its head camera looks, which poses are allowed, and
what it looks like.

LampTwin implements contract.Kinematics on top of the vendor's own robot description:

  head(units)        -> Pose            the head camera: optical centre and axes, base frame
  check(units)       -> Check           joint limits and true clearances between the vendor's meshes
  look_at(point)     -> (units, report) a whole-arm pose facing a point; never one check() refuses
  project(units, p)  -> (x, y) | None   where a base-frame point lands in the head camera's picture
  render(units, ...) -> HxWx3 uint8     a picture of the lamp (orbit camera) or from its head camera

The robot description (URDF, STL meshes, joint map) and the servo calibration belong to the lamp's vendor
and to the lamp. They are read at run time from FTM_ROBOT_DIR and FTM_CALIBRATION (or the paths passed in)
and nothing of them is copied into this repository: no geometry, no calibration value, no number derived
from them is written down here. What this file found in them is described in words.

Frames and units are contract.py's. The base frame is the URDF's root link (the base plate): x right,
y forward, z up. Joint values are SDK units, -100..100.

What we found in the vendor description (tests/twin/test_model.py checks each):
  * The lowest vertex of the base plate mesh lies at z = 0 in the base frame, and nothing of the base goes
    below it, so z = 0 is the table. The vendor's collision cylinder reaching below that is an envelope.
  * The camera module sits on top of the shade, under the cap. Its square board lies across the head
    link's x axis with no roll and its lens is the front of the mesh, so the optical axis is the head
    link's +x (the axis the shade opens along) and picture-down its +y, as lamp/spatial.py assumed. The
    optical centre is the lens, a few centimetres above and in front of the wrist-pitch axis.
  * With the lamp's servo calibration, a 61 x 44 deg camera reproduces both picture shifts measured on the
    lamp within 5%.
  * Across a sweep of the calibrated range, links two or more apart never touch; the closest is the head
    to the upper arm with the head tilted fully down (a few millimetres). The real risks are the table
    (leaning forward with the elbow folded) and the base (folding back), and the vendor's only check, head
    shade against a cylinder on the base plate, lets many table collisions through (sdk_head_base_clear).

Not thread-safe: one MjData is shared by head(), check() and look_at(); render() has its own.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import mujoco
import numpy as np

from twin.contract import JOINTS, Check, Person, Pose, Units

# ------------------------------------------------------------------ constants and where they come from
# Default locations, as tests/twin/conftest.py. Overridden by the arguments or FTM_ROBOT_DIR/FTM_CALIBRATION.
DEFAULT_ROBOT_DIR = Path.home() / "lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1"
DEFAULT_CALIBRATION = Path.home() / "lelamp-hackathon-2026/lelamp.json"

# Where the gravity sag's levels are stated (GravitySag): the lamp facing a seated head straight ahead,
# 0.95 m out and 0.46 m above the table. ASSUMPTION: the tracker's seated search row (twin/tracking.py
# search_rows) and world.py's seated listener, the pose the lamp holds most while locked.
SAG_REFERENCE_POINT = (0.0, 0.95, 0.46)

# Head camera field of view: measured on the lamp 2026-09-19. At neutral, base_yaw +7.4 units moved the
# picture -0.083 widths and wrist_pitch +8.1 units moved it -0.085 heights; through the calibrated joint
# scales below that is about 61 x 44 deg (the lamp-sdk branch's lamp/spatial.py records both measurements).
HFOV_DEG = 61.0
VFOV_DEG = 44.0

# SDK units span -100..100 over each joint's calibrated range (vendor source static/robots/lelamp_v1/
# pi5_feetech_r1/safety.yaml:2-8, mode calibrated_normalized), and the servo reports 4096 ticks per turn
# (vendor source modules/robot_base/robots/registry.py:498-503, ticks_per_degree default 4096/360). So one
# unit is (range_max - range_min) / 4096 turns / 200. This per-joint scale, with the vendor's neutral and joint
# offsets, reproduced both lamp measurements above (lamp/spatial.py on branch karanclaude/lamp-sdk).
TICKS_PER_TURN = 4096.0
UNITS_SPAN = 200.0

# The vendor's own mapping from units to URDF radians is delta * degrees_to_radians * motion_scale *
# sim_deg_scale + joint_offset (vendor source modules/robot_base/robots/kinematics/configured_model.py:34-42,
# sim_deg_scale defaults to 1.0 at registry.py:478-483). It uses one approximate scale for every joint; it
# is the fallback when no servo calibration is readable, and scale_source says so.

# Look-at solver weights. ASSUMPTION: tuned in the twin (tests/twin/test_model.py) so the aim error of a
# reachable target stays well under 1 deg while the arm keeps close to its neutral posture.
W_AIM = 10.0          # per unit of |forward - to/|to||, which is 2 sin(error/2): about 0.17 per degree
W_DISTANCE = 0.6      # per metre off the preferred viewing distance (saturating, see _residuals)
DISTANCE_SOFTNESS_M = 0.25
W_HEIGHT = 6.0        # per metre outside the head-height band
W_POSTURE = 0.35      # per 100 units away from neutral (base_pitch, elbow_pitch, wrist_pitch)
W_SEED = 0.10         # per 100 units away from the seed: keeps nearby targets on nearby poses
MAX_STEP_UNITS = 15.0
MAX_ITERATIONS = 80

# Rendering. ASSUMPTION (a typical room, only for pictures): a 0.75 m high table, a floor and four walls.
TABLE_HEIGHT_M = 0.75
MAX_PEOPLE = 6
ORBIT = (-150.0, -15.0, 1.4)            # azimuth, elevation (deg), distance (m): front-right of the lamp
ORBIT_LOOKAT = (0.0, 0.10, 0.20)
MAX_RENDER = (1920, 1440)
PANEL_OFF = (0.85, 0.85, 0.82, 1.0)      # an unlit diffuser: plain white plastic
SKIN = [(0.87, 0.70, 0.58), (0.62, 0.44, 0.33), (0.95, 0.80, 0.69), (0.45, 0.31, 0.23), (0.78, 0.60, 0.47),
        (0.69, 0.52, 0.40)]
SHIRT = [(0.20, 0.35, 0.60), (0.60, 0.25, 0.25), (0.25, 0.50, 0.30), (0.45, 0.40, 0.55), (0.70, 0.60, 0.30),
         (0.30, 0.30, 0.30)]

# Friendly names of the six links of the chain, root first. Our names, not the vendor's.
LINK_NAMES = ("base plate", "base", "lower arm", "upper arm", "wrist", "head")
# The meshes that glow (the shade's diffusers and the LED board), by the vendor's mesh file names. Every
# other part keeps the colour the vendor's CAD export gave it in the URDF.
GLOW_MESHES = ("head_diffuser", "fade_diffuser", "led_array")


class LampTwin:
    """contract.Kinematics for the LeLamp, built from the vendor's URDF and meshes with MuJoCo.

    limit_margin: units kept clear of each end of the -100..100 range (ASSUMPTION, as lamp/spatial.py).
    table_margin, base_margin, self_margin: the smallest allowed distance in metres between the moving
        links and the table, the lamp's own base, and each other (non-adjacent links). ASSUMPTION: the
        workflow brief's values; the vendor checks none of the table, and only the head shade against a
        cylinder for the base.
    """

    def __init__(self, robot_dir: str | Path | None = None, calibration: str | Path | None = None, *,
                 limit_margin: float = 6.0, table_margin: float = 0.03, base_margin: float = 0.02,
                 self_margin: float = 0.005, hfov_deg: float = HFOV_DEG, vfov_deg: float = VFOV_DEG):
        self.robot_dir = Path(robot_dir or os.environ.get("FTM_ROBOT_DIR") or DEFAULT_ROBOT_DIR)
        calibration = Path(calibration or os.environ.get("FTM_CALIBRATION") or DEFAULT_CALIBRATION)
        if not (self.robot_dir / "robot.urdf").exists():
            raise FileNotFoundError(f"no robot.urdf in {self.robot_dir} (set FTM_ROBOT_DIR)")
        mapping = json.loads((self.robot_dir / "joint_mapping.yaml").read_text())   # the file holds JSON
        self.neutral: Units = {j: float(mapping["neutral"][j]) for j in JOINTS}
        self._servo = {j: mapping["motor_to_joint"][j] for j in JOINTS}
        self._offset = {j: float(mapping["joint_offsets"][self._servo[j]]) for j in JOINTS}
        self.scale, self.scale_source = self._read_scale(mapping, calibration)
        self.limits = {j: (-100.0 + limit_margin, 100.0 - limit_margin) for j in JOINTS}
        self.table_z = 0.0
        self.table_margin, self.base_margin, self.self_margin = table_margin, base_margin, self_margin
        self.hfov_deg, self.vfov_deg = hfov_deg, vfov_deg
        self.fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)   # focal length in picture widths
        self.fy = 0.5 / math.tan(math.radians(vfov_deg) / 2)   # focal length in picture heights

        self.model = self._build()
        self.data = mujoco.MjData(self.model)
        self._render_data = mujoco.MjData(self.model)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._last: tuple | None = None
        self._sdk_rule_cache: dict | None | bool = False           # False = not read yet
        self._grav_data: mujoco.MjData | None = None                # gravity_torque()'s own MjData
        self._index_geometry()
        why = self.check(self.neutral).reasons
        if why:
            raise ValueError(f"the neutral pose fails check() with these margins: {why}")

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _read_scale(mapping: dict, calibration: Path) -> tuple[dict, str]:
        """Radians per SDK unit, per joint: from the lamp's servo calibration, else the vendor's one scale."""
        try:
            ticks = json.loads(Path(calibration).read_text())
            scale = {j: (float(ticks[j]["range_max"]) - float(ticks[j]["range_min"])) / TICKS_PER_TURN
                     * 2 * math.pi / UNITS_SPAN for j in JOINTS}
            return scale, f"servo calibration ({Path(calibration).name})"
        except (OSError, ValueError, KeyError, TypeError):
            approx = float(mapping["motion_scale"]) * float(mapping["degrees_to_radians"])
            return ({j: approx for j in JOINTS},
                    "vendor approximate joint map (motion_scale x degrees_to_radians): no servo calibration found")

    def _build(self) -> mujoco.MjModel:
        """The vendor URDF in MuJoCo, plus a table, a room, lights, cameras and people to render."""
        text = (self.robot_dir / "robot.urdf").read_text()
        # MuJoCo reads a <mujoco><compiler/></mujoco> block inside a URDF. Replace the vendor's with ours:
        # mesh paths relative to the robot directory, no fused static bodies (the base plate stays a body),
        # visual meshes kept, and inertias balanced (we never simulate dynamics, but the compiler checks them).
        block = (f'<mujoco><compiler meshdir="{self.robot_dir}" strippath="false" balanceinertia="true" '
                 f'discardvisual="false" fusestatic="false"/></mujoco>')
        text = re.sub(r"<mujoco>.*?</mujoco>", "", text, flags=re.S)
        text = re.sub(r"(<robot[^>]*>)", lambda found: found.group(1) + block, text, count=1)
        spec = mujoco.MjSpec.from_string(text)

        # The URDF lists every mesh twice, once as <visual> (group 1) and once as <collision> (group 0), at
        # the same pose. Keep the visual copies for both rendering and distances. MuJoCo builds a mesh's
        # convex hull only when a geom using it can collide, so they get collision flags; nothing here ever
        # runs collision detection (only mj_kinematics and mj_geomDistance).
        for geom in list(spec.geoms):
            if geom.group == 0:
                spec.delete(geom)
        for geom in spec.geoms:
            geom.contype, geom.conaffinity = 1, 1

        self._add_materials(spec)
        chain = self._chain_bodies(spec)
        for position, body in enumerate(chain):
            for geom in body.geoms:
                geom.group = 3 if position == len(chain) - 1 else 1     # head geoms: group 3 (hidden in head view)
                if geom.meshname in GLOW_MESHES:
                    geom.material = "lamp_glow"
                    geom.rgba = [0.5, 0.5, 0.5, 1]      # MuJoCo's default rgba lets the material's colour show

        # The table: an invisible plane at z = 0 for distances, and a visible table top whose top face is z = 0.
        spec.worldbody.add_geom(name="table_plane", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1],
                                group=5, contype=0, conaffinity=0, rgba=[0, 0, 0, 0])
        spec.worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.35, -0.02],
                                size=[0.8, 0.65, 0.02], material="table", group=2, contype=0, conaffinity=0)
        for sx in (-1, 1):
            for sy in (-1, 1):
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, TABLE_HEIGHT_M / 2 - 0.02],
                                        pos=[0.74 * sx, 0.35 + 0.6 * sy, -TABLE_HEIGHT_M / 2 - 0.02],
                                        material="table", group=2, contype=0, conaffinity=0)
        # The room: floor, four walls (each a plane facing inward) and a sky beyond, only for pictures.
        floor_z = -TABLE_HEIGHT_M
        room_x, room_y = (-3.5, 3.5), (-1.5, 4.0)
        centre_y, half_y = (room_y[0] + room_y[1]) / 2, (room_y[1] - room_y[0]) / 2
        wall_z = floor_z + 1.5
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, centre_y, floor_z],
                                size=[room_x[1], half_y, 0.1], material="floor", group=2, contype=0, conaffinity=0)
        walls = (([0, room_y[0], wall_z], [room_x[1], 1.5], ([1, 0, 0], [0, 0, -1], [0, 1, 0])),
                 ([0, room_y[1], wall_z], [room_x[1], 1.5], ([-1, 0, 0], [0, 0, -1], [0, -1, 0])),
                 ([room_x[0], centre_y, wall_z], [half_y, 1.5], ([0, -1, 0], [0, 0, -1], [1, 0, 0])),
                 ([room_x[1], centre_y, wall_z], [half_y, 1.5], ([0, 1, 0], [0, 0, -1], [-1, 0, 0])))
        for pos, size, axes in walls:
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, pos=pos, size=[*size, 0.1],
                                    quat=_quat_from_axes(*axes), material="wall", group=2, contype=0, conaffinity=0)

        spec.worldbody.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0.5, 1.5, 2.5],
                                 dir=[-0.2, -0.5, -1.0], diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1],
                                 castshadow=True)
        spec.worldbody.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[-1, -1, 2],
                                 dir=[0.4, 0.6, -0.7], diffuse=[0.25, 0.25, 0.27], castshadow=False)
        # The panel's light on the room: a point light just in front of the shade (placed and coloured at
        # render time). A spot light would suit better, but in MuJoCo 3.13's renderer a spot light stopped
        # the diffuser's emission from showing (found while building this; a point light does not).
        spec.body(chain[-1].name).add_light(name="panel", type=mujoco.mjtLightType.mjLIGHT_POINT,
                                            diffuse=[0, 0, 0], specular=[0, 0, 0], attenuation=[1, 0.5, 2.0],
                                            castshadow=False, active=False)
        spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
        spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
        spec.visual.global_.offwidth, spec.visual.global_.offheight = MAX_RENDER
        spec.visual.map.znear = 0.005
        spec.stat.extent = 1.5
        spec.stat.center = list(ORBIT_LOOKAT)

        # The head camera. MuJoCo cameras look along their -z with +y up, so camera x = -head z, y = -head y,
        # z = -head x. Its position is set after compiling, from the camera mesh (_index_geometry). The
        # lens is given as a sensor size over a focal length, so the picture spans exactly hfov x vfov
        # whatever its pixel size.
        spec.body(chain[-1].name).add_camera(
            name="head_camera", quat=_quat_from_axes([0, 0, -1], [0, -1, 0], [-1, 0, 0]),
            sensor_size=[2 * math.tan(math.radians(self.hfov_deg) / 2), 2 * math.tan(math.radians(self.vfov_deg) / 2)],
            focal_length=[1.0, 1.0], resolution=[640, 480])

        # People: a head body (sphere + face disc, full orientation) and a torso body (upright) per slot.
        for i in range(MAX_PEOPLE):
            head = spec.worldbody.add_body(name=f"person{i}_head", mocap=True, pos=[0, 0, -50])
            head.add_geom(name=f"person{i}_skull", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.09, 0, 0],
                          rgba=[*SKIN[i % len(SKIN)], 1], group=0, contype=0, conaffinity=0)
            head.add_geom(name=f"person{i}_face", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.055, 0.004, 0],
                          pos=[0.08, 0, 0], quat=_quat_from_axes([0, 0, -1], [0, 1, 0], [1, 0, 0]),
                          rgba=[*(0.6 * np.array(SKIN[i % len(SKIN)])), 1], group=0, contype=0, conaffinity=0)
            torso = spec.worldbody.add_body(name=f"person{i}_torso", mocap=True, pos=[0, 0, -50])
            torso.add_geom(name=f"person{i}_body", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.14, 0.22, 0],
                           rgba=[*SHIRT[i % len(SHIRT)], 1], group=0, contype=0, conaffinity=0)
        return spec.compile()

    @staticmethod
    def _add_materials(spec: mujoco.MjSpec) -> None:
        spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.42, 0.40, 0.38],
                         rgb2=[0.36, 0.34, 0.32], width=256, height=256)
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.85, 0.86, 0.9],
                         rgb2=[0.45, 0.46, 0.5], width=64, height=384)
        spec.add_material(name="floor", textures=["", "floor_tex"], texrepeat=[8, 8])
        spec.add_material(name="wall", rgba=[0.78, 0.78, 0.76, 1])
        spec.add_material(name="table", rgba=[0.55, 0.40, 0.28, 1])
        spec.add_material(name="lamp_glow", rgba=list(PANEL_OFF), emission=0.0)

    def _chain_bodies(self, spec: mujoco.MjSpec) -> list:
        """The six links root first: base plate, then the child link of each of the five joints in order."""
        chain = []
        for j in JOINTS:
            joint = spec.joint(self._servo[j])
            if joint is None:
                raise ValueError(f"joint {self._servo[j]} ({j}) is not in the URDF")
            body = joint.parent
            if not chain:
                chain.append(body.parent)
            chain.append(body)
        return chain

    def _index_geometry(self) -> None:
        m = self.model
        self._joint_ids = {j: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self._servo[j])) for j in JOINTS}
        self._qadr = {j: int(m.jnt_qposadr[self._joint_ids[j]]) for j in JOINTS}
        # The chain of bodies, root first (see _chain_bodies).
        self._chain = [int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self._servo[j])])
                       for j in JOINTS]
        self._chain.insert(0, int(m.body_parentid[self._chain[0]]))
        self._head_body = self._chain[-1]
        position = {b: i for i, b in enumerate(self._chain)}
        lamp_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in position and m.geom_dataid[g] >= 0]
        self._geom_link = {g: position[int(m.geom_bodyid[g])] for g in lamp_geoms}
        mesh_name = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) for g in lamp_geoms}
        self._mesh_name = mesh_name

        # Table: every geom of the four arm links (the base only turns about the vertical, so its height
        # above the table never changes). Distance to a plane is exact on the convex hull's vertices.
        verts, owners = [], []
        for g in lamp_geoms:
            if self._geom_link[g] >= 2:
                hull = self._hull_vertices(g)
                verts.append(hull)
                owners.append(np.full(len(hull), g))
        self._table_verts = np.concatenate(verts)
        self._table_owner = np.concatenate(owners)

        # Pairs for mesh-to-mesh distances. Links next to each other in the chain share a servo and touch
        # by design, so only pairs two or more links apart are checked. A pair with the base plate or the
        # turning base is a "base" pair; a pair of two arm links is a "self" pair.
        base_pairs, self_pairs = [], []
        for g1 in lamp_geoms:
            for g2 in lamp_geoms:
                a, b = self._geom_link[g1], self._geom_link[g2]
                if a < b and b - a >= 2:
                    (base_pairs if a <= 1 else self_pairs).append((g1, g2))
        self._pairs = {"base": np.array(base_pairs, dtype=int), "self": np.array(self_pairs, dtype=int)}
        self._fromto = np.zeros(6)

        # The head camera: the front face of the camera mesh in the head link. What we found in the vendor
        # description: the camera module sits on top of the shade under the cap, its square board lying
        # across the head link's x axis with no roll, and its lens (a small disc) is the part of the mesh
        # furthest along +x. So the optical axis is the head link's +x (the axis the shade opens along, as
        # lamp/spatial.py assumed) and picture-down is the head link's +y. The lens is a few centimetres
        # above and in front of the wrist-pitch axis; tests/twin/test_model.py checks this against the mesh.
        mujoco.mj_kinematics(m, self.data)
        head_pos = self.data.xpos[self._head_body].copy()
        head_rot = self.data.xmat[self._head_body].reshape(3, 3).copy()
        camera = [g for g in lamp_geoms if mesh_name[g] == "camera" and self._geom_link[g] == 5]
        if camera:
            local = (self._world_vertices(camera[0], self.data) - head_pos) @ head_rot
            front = local[local[:, 0] > local[:, 0].max() - 0.003]     # the lens: within 3 mm of the front
            self.camera_local = np.array([local[:, 0].max(), *(front[:, 1:].min(0) + front[:, 1:].max(0)) / 2])
            self.camera_source = "camera mesh (lens centre)"
        else:
            self.camera_local = np.zeros(3)
            self.camera_source = "head link origin (no camera mesh found)"
        m.cam_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")] = self.camera_local
        # The panel's light sits a few centimetres in front of the front diffuser's centre.
        glow = [g for g in lamp_geoms if mesh_name[g] == "head_diffuser" and self._geom_link[g] == 5]
        glow_local = (((self._world_vertices(glow[0], self.data) - head_pos) @ head_rot).mean(0)
                      if glow else np.array([0.1, 0.0, 0.0]))
        self._glow_material = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "lamp_glow")
        self._panel_light = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "panel")
        m.light_pos[self._panel_light] = glow_local + (0.04, 0.0, 0.0)

        def body(name: str) -> int:
            return int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)])

        def geom(name: str) -> int:
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._people_ids = [(body(f"person{i}_head"), body(f"person{i}_torso"), geom(f"person{i}_skull"),
                             geom(f"person{i}_face"), geom(f"person{i}_body")) for i in range(MAX_PEOPLE)]

    def _hull_vertices(self, g: int) -> np.ndarray:
        """Vertices of a mesh geom's convex hull, in the geom's own frame."""
        m = self.model
        mesh = int(m.geom_dataid[g])
        start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        verts = m.mesh_vert[start:start + count]
        adr = int(m.mesh_graphadr[mesh])
        if adr < 0:
            return verts.copy()
        numvert = int(m.mesh_graph[adr])
        ids = m.mesh_graph[adr + 2 + numvert: adr + 2 + 2 * numvert]
        return verts[ids].copy()

    def _world_vertices(self, g: int, data: mujoco.MjData) -> np.ndarray:
        m = self.model
        mesh = int(m.geom_dataid[g])
        start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        return data.geom_xpos[g] + m.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3).T

    # ------------------------------------------------------------------ units and radians
    def radians(self, units: Units) -> dict:
        """URDF joint angles (radians) for SDK units, keyed by our joint names."""
        return {j: (float(units[j]) - self.neutral[j]) * self.scale[j] + self._offset[j] for j in JOINTS}

    def units(self, radians: dict) -> Units:
        """The inverse of radians()."""
        return {j: (float(radians[j]) - self._offset[j]) / self.scale[j] + self.neutral[j] for j in JOINTS}

    def _qpos(self, units: Units) -> np.ndarray:
        q = np.zeros(self.model.nq)
        for j, angle in self.radians(units).items():
            q[self._qadr[j]] = angle
        return q

    def _set(self, units: Units, data: mujoco.MjData | None = None) -> None:
        key = tuple(float(units[j]) for j in JOINTS)
        if data is None:
            if key == self._last:
                return
            self._last, data = key, self.data
        data.qpos[:] = self._qpos(units)
        mujoco.mj_kinematics(self.model, data)

    # ------------------------------------------------------------------ forward kinematics
    def head(self, units: Units) -> Pose:
        """The head camera: lens centre and axes in the base frame."""
        self._set(units)
        position = self.data.xpos[self._head_body]
        rot = self.data.xmat[self._head_body].reshape(3, 3)
        forward, down = rot[:, 0].copy(), rot[:, 1].copy()
        return Pose(position=position + rot @ self.camera_local, forward=forward, down=down,
                    right=np.cross(down, forward))

    def head_link(self, units: Units) -> np.ndarray:
        """The wrist-pitch axis point (the head link's origin), base frame: what head_height constrains."""
        self._set(units)
        return self.data.xpos[self._head_body].copy()

    def project(self, units: Units, point) -> tuple[float, float] | None:
        """Where a base-frame point appears in the head camera picture (0..1, 0..1), None if behind it."""
        pose = self.head(units)
        to = np.asarray(point, dtype=float) - pose.position
        depth = float(to @ pose.forward)
        if depth <= 1e-6:
            return None
        return (0.5 + self.fx * float(to @ pose.right) / depth, 0.5 + self.fy * float(to @ pose.down) / depth)

    def ray(self, units: Units, where) -> np.ndarray:
        """Unit direction (base frame) of picture position `where` = (x, y) in 0..1."""
        pose = self.head(units)
        d = pose.forward + pose.right * ((where[0] - 0.5) / self.fx) + pose.down * ((where[1] - 0.5) / self.fy)
        return d / np.linalg.norm(d)

    def aim_error_deg(self, units: Units, point) -> float:
        pose = self.head(units)
        to = np.asarray(point, dtype=float) - pose.position
        norm = float(np.linalg.norm(to))
        if norm < 1e-9:
            return 0.0
        return math.degrees(math.acos(float(np.clip(pose.forward @ to / norm, -1.0, 1.0))))

    # ------------------------------------------------------------------ where the lamp may be
    def check(self, units: Units) -> Check:
        """Joint limits, and signed distances between the vendor's real meshes (their convex hulls).

        min_table_m: lowest point of the four arm links above the table plane z = 0.
        min_base_m:  arm links to the base plate and the turning base (links two or more apart in the chain).
        min_self_m:  arm links to each other, two or more apart in the chain.
        """
        reasons = []
        for j in JOINTS:
            lo, hi = self.limits[j]
            if not lo <= float(units[j]) <= hi:
                reasons.append(f"{j} {float(units[j]):+.1f} is outside {lo:+.0f}..{hi:+.0f}")
        self._set(units)
        d = self.data

        z = d.geom_xpos[self._table_owner, 2] + np.einsum(
            "ij,ij->i", d.geom_xmat[self._table_owner, 6:9], self._table_verts)
        lowest = int(np.argmin(z))
        table = float(z[lowest]) - self.table_z
        if table < self.table_margin:
            link = LINK_NAMES[self._geom_link[int(self._table_owner[lowest])]]
            reasons.append(f"{link} is {100 * table:.1f} cm above the table (needs {100 * self.table_margin:.1f} cm)")

        base, base_pair = self._closest("base")
        if base < self.base_margin:
            a, b = (LINK_NAMES[self._geom_link[g]] for g in base_pair)
            reasons.append(f"{b} is {100 * base:.1f} cm from the lamp's {a} (needs {100 * self.base_margin:.1f} cm)")
        near, self_pair = self._closest("self")
        if near < self.self_margin:
            a, b = (LINK_NAMES[self._geom_link[g]] for g in self_pair)
            reasons.append(f"{a} and {b} are {100 * near:.1f} cm apart (needs {100 * self.self_margin:.1f} cm)")
        return Check(ok=not reasons, reasons=reasons, min_table_m=table, min_base_m=base, min_self_m=near)

    def _closest(self, kind: str) -> tuple[float, tuple[int, int]]:
        """Smallest signed distance over one kind of geom pair, exact, with few GJK calls: pairs are taken
        in order of a bounding-sphere lower bound, and the search stops once that bound passes the best."""
        pairs = self._pairs[kind]
        d, m = self.data, self.model
        centre_gap = np.linalg.norm(d.geom_xpos[pairs[:, 0]] - d.geom_xpos[pairs[:, 1]], axis=1)
        lower = centre_gap - m.geom_rbound[pairs[:, 0]] - m.geom_rbound[pairs[:, 1]]
        best, best_pair = math.inf, (int(pairs[0, 0]), int(pairs[0, 1]))
        for k in np.argsort(lower):
            if lower[k] >= best:
                break
            g1, g2 = int(pairs[k, 0]), int(pairs[k, 1])
            dist = mujoco.mj_geomDistance(m, d, g1, g2, min(best, 1.0), self._fromto)
            if dist < best:
                best, best_pair = float(dist), (g1, g2)
        return best, best_pair

    # ------------------------------------------------------------------ inverse kinematics
    def look_at(self, point, seed: Units | None = None, *, prefer_distance: float = 0.45,
                head_height: tuple[float, float] = (0.20, 0.36), max_yaw_from_neutral: float = 70.0,
                max_pitch_elbow_from_neutral: float = 45.0) -> tuple[Units, dict]:
        """A whole-arm pose whose head camera faces `point`, and a report.

        Damped least squares over base_yaw, base_pitch, elbow_pitch and wrist_pitch (wrist_roll stays at
        neutral, which keeps the picture level) on: the aim residual forward - to/|to| (zero when facing
        the point, growing all the way to 180 deg), a preferred viewing distance, a band for the height of
        the wrist-pitch axis (head_height, metres above the table), staying near the neutral posture, and
        staying near the seed. Yaw is kept within max_yaw_from_neutral deg of neutral, base_pitch and
        elbow_pitch within max_pitch_elbow_from_neutral deg.

        Never returns a pose that fails check(). If the solution does, it falls back to a yaw+tilt-only
        pose from neutral, then to the seed, then to neutral; report["fallback"] says which was used.
        """
        point = np.asarray(point, dtype=float)
        free = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch")
        bounds = {}
        for j in free:
            lo, hi = self.limits[j]
            if j == "base_yaw":
                window = max_yaw_from_neutral
            elif j in ("base_pitch", "elbow_pitch"):
                window = max_pitch_elbow_from_neutral
            else:
                window = None
            if window is not None:
                span = math.radians(window) / self.scale[j]
                lo, hi = max(lo, self.neutral[j] - span), min(hi, self.neutral[j] + span)
            bounds[j] = (lo, hi)
        lo = np.array([bounds[j][0] for j in free])
        hi = np.array([bounds[j][1] for j in free])
        seed_pose = None if seed is None else {j: float(seed[j]) for j in JOINTS}
        start = dict(seed_pose or self.neutral)
        start["wrist_roll"] = self.neutral["wrist_roll"]
        template = dict(start)

        def pose_of(x: np.ndarray) -> Units:
            pose = dict(template)
            pose.update({j: float(v) for j, v in zip(free, x, strict=True)})
            return pose

        neutral_x = np.array([self.neutral[j] for j in free])
        seed_x = None if seed_pose is None else np.clip([seed_pose[j] for j in free], lo, hi)
        weights = dict(prefer=prefer_distance, band=head_height, seed=seed_x, neutral=neutral_x)

        x0 = np.clip([start[j] for j in free], lo, hi)
        x0[0] = self._yaw_guess(pose_of(x0), point, lo[0], hi[0])
        x, iterations = self._solve(lambda v: self._residuals(pose_of(v), point, weights), x0, lo, hi)
        pose = pose_of(x)
        result = self.check(pose)
        rejected = [] if result.ok else list(result.reasons)
        fallback = "none"
        if not result.ok and seed_pose is not None:
            # A second try from neutral before giving up on the whole arm.
            x1 = neutral_x.copy()
            x1[0] = self._yaw_guess(pose_of(x1), point, lo[0], hi[0])
            x, more = self._solve(lambda v: self._residuals(pose_of(v), point, weights), x1, lo, hi)
            iterations += more
            if self.check(pose_of(x)).ok:
                pose, fallback = pose_of(x), "neutral_start"
        if fallback == "none" and rejected:
            pose, fallback = self._yaw_tilt(point, lo, hi), "yaw_tilt"
            if pose is None:
                if seed_pose is not None and self.check(seed_pose).ok:
                    pose, fallback = dict(seed_pose), "seed"
                else:
                    pose, fallback = dict(self.neutral), "neutral"

        camera = self.head(pose).position
        report = {"aim_error_deg": self.aim_error_deg(pose, point),
                  "head_height_m": float(self.head_link(pose)[2]), "camera_height_m": float(camera[2]),
                  "distance_m": float(np.linalg.norm(point - camera)), "fallback": fallback,
                  "rejected": rejected, "iterations": iterations}
        return pose, report

    def _residuals(self, pose: Units, point: np.ndarray, w: dict) -> np.ndarray:
        cam = self.head(pose)
        to = point - cam.position
        dist = max(float(np.linalg.norm(to)), 1e-6)
        aim = cam.forward - to / dist
        # The distance pull saturates, so a far listener only makes the arm lean a little toward them.
        distance = DISTANCE_SOFTNESS_M * math.tanh((dist - w["prefer"]) / DISTANCE_SOFTNESS_M)
        z = float(self.data.xpos[self._head_body][2])
        band_lo, band_hi = w["band"]
        height = max(0.0, band_lo - z) + max(0.0, z - band_hi)
        x = np.array([pose[j] for j in ("base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch")])
        parts = [aim * W_AIM, [distance * W_DISTANCE, height * W_HEIGHT], (x[1:] - w["neutral"][1:]) / 100 * W_POSTURE]
        if w["seed"] is not None:
            parts.append((x - w["seed"]) / 100 * W_SEED)
        return np.concatenate(parts)

    @staticmethod
    def _solve(residuals, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, int]:
        """Levenberg-Marquardt with a forward-difference Jacobian, projected onto the joint bounds."""
        x = np.array(x, dtype=float)
        r = residuals(x)
        cost = float(r @ r)
        damping, step_units = 1e-2, 0.25
        for iteration in range(1, MAX_ITERATIONS + 1):
            jac = np.empty((len(r), len(x)))
            for k in range(len(x)):
                probe = x.copy()
                probe[k] += step_units if probe[k] + step_units <= hi[k] else -step_units
                jac[:, k] = (residuals(probe) - r) / (probe[k] - x[k])
            dx = np.linalg.solve(jac.T @ jac + damping * np.eye(len(x)), -jac.T @ r)
            dx = np.clip(dx, -MAX_STEP_UNITS, MAX_STEP_UNITS)
            candidate = np.clip(x + dx, lo, hi)
            r_new = residuals(candidate)
            new_cost = float(r_new @ r_new)
            if new_cost < cost:
                moved = float(np.max(np.abs(candidate - x)))
                x, r, cost, damping = candidate, r_new, new_cost, max(damping * 0.3, 1e-5)
                if moved < 0.01:
                    return x, iteration
            else:
                damping *= 5.0
                if damping > 1e6:
                    return x, iteration
        return x, MAX_ITERATIONS

    def _yaw_guess(self, pose: Units, point: np.ndarray, lo: float, hi: float) -> float:
        """base_yaw that turns the camera's heading toward the point in plan view (two refinements)."""
        yaw = float(pose["base_yaw"])
        for _ in range(2):
            trial = dict(pose, base_yaw=yaw)
            cam = self.head(trial)
            heading = math.atan2(cam.forward[0], cam.forward[1])
            wanted = math.atan2(point[0] - cam.position[0], point[1] - cam.position[1])
            nudged = self.head(dict(trial, base_yaw=yaw + 1.0)).forward
            per_unit = _wrap(math.atan2(nudged[0], nudged[1]) - heading)
            if abs(per_unit) < 1e-9:
                break
            yaw = float(np.clip(yaw + _wrap(wanted - heading) / per_unit, lo, hi))
        return yaw

    def _yaw_tilt(self, point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Units | None:
        """Neutral arm, only turned (base_yaw) and tilted (wrist_pitch) toward the point. If the tilt is
        refused, it is halved toward neutral until the pose passes. None if nothing passes."""
        base = dict(self.neutral)
        yaw_lo, yaw_hi = lo[0], hi[0]
        tilt_lo, tilt_hi = lo[3], hi[3]

        def residuals(v):
            pose = dict(base, base_yaw=float(v[0]), wrist_pitch=float(v[1]))
            cam = self.head(pose)
            to = point - cam.position
            return (cam.forward - to / max(float(np.linalg.norm(to)), 1e-6)) * W_AIM

        x0 = np.array([self._yaw_guess(base, point, yaw_lo, yaw_hi), base["wrist_pitch"]])
        x, _ = self._solve(residuals, x0, np.array([yaw_lo, tilt_lo]), np.array([yaw_hi, tilt_hi]))
        for fraction in (1.0, 0.5, 0.25, 0.0):
            pose = dict(base, base_yaw=float(x[0]),
                        wrist_pitch=float(base["wrist_pitch"] + fraction * (x[1] - base["wrist_pitch"])))
            if self.check(pose).ok:
                return pose
        return None

    # ------------------------------------------------------------------ load: gravity torque at the joints
    def gravity_torque(self, units: Units) -> dict:
        """The torque (N m) each joint's servo must hold against the weight of the links it carries, from the
        vendor URDF's link masses and centres of mass (MuJoCo's bias force at rest). Its sign is that of the
        URDF joint angle, which grows with SDK units (every calibrated scale is positive), so gravity pulls a
        joint toward -sign(torque) units. Only the lamp's own links count; the people drawn for pictures are
        mocap bodies outside the arm's chain."""
        if self._grav_data is None:
            self._grav_data = mujoco.MjData(self.model)
        d = self._grav_data
        d.qpos[:] = self._qpos(units)
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, d, 0, bias)
        return {j: float(bias[self.model.jnt_dofadr[self._joint_ids[j]]]) for j in JOINTS}

    def sag_reference_pose(self) -> Units:
        """The pose GravitySag's levels are stated at: looking at SAG_REFERENCE_POINT."""
        pose, _ = self.look_at(np.asarray(SAG_REFERENCE_POINT, float), seed=dict(self.neutral))
        return pose

    # ------------------------------------------------------------------ the vendor's own check, for comparison
    def sdk_head_base_clear(self, units: Units) -> bool:
        """The vendor planner's only self-collision rule: a capsule on the head link against a capped
        cylinder on the base plate, refused when their distance is at most the capsule radius plus a
        clearance (vendor source modules/robot_base/robots/kinematics/self_collision.py:31-60; the numbers
        are read at run time from the vendor's safety.yaml). True when the SDK would let this pose through.

        It knows nothing of the table, the rest of the arm or the real meshes, and its cylinder reaches from
        the top of the base plate down to below the table: it guards the plate, not the base body above it.
        check() is stricter on purpose. The segment is sampled at 64 points (about 2 mm apart), where the
        vendor searches it; the difference is far below the 5 mm clearance."""
        rule = self._sdk_rule()
        if rule is None:
            return True
        self._set(units)
        head_pos = self.data.xpos[self._head_body]
        head_rot = self.data.xmat[self._head_body].reshape(3, 3)
        plate_pos = self.data.xpos[self._chain[0]]
        plate_rot = self.data.xmat[self._chain[0]].reshape(3, 3)
        a = (head_pos + head_rot @ rule["start"] - plate_pos) @ plate_rot
        b = (head_pos + head_rot @ rule["end"] - plate_pos) @ plate_rot
        points = a + np.outer(np.linspace(0.0, 1.0, 64), b - a)
        radial = np.maximum(np.hypot(points[:, 0] - rule["centre"][0], points[:, 1] - rule["centre"][1])
                            - rule["cylinder_radius"], 0.0)
        vertical = np.maximum.reduce([rule["z_min"] - points[:, 2], np.zeros(len(points)),
                                      points[:, 2] - rule["z_max"]])
        return float(np.min(np.hypot(radial, vertical))) > rule["capsule_radius"] + rule["clearance"]

    def _sdk_rule(self) -> dict | None:
        if self._sdk_rule_cache is not False:
            return self._sdk_rule_cache
        rule = None
        try:
            text = (self.robot_dir / "safety.yaml").read_text()

            def numbers(key: str) -> list[float]:
                found = re.search(rf"{key}:\s*\[?([-+\d.eE,\s]+)\]?", text)
                return [float(v) for v in found.group(1).replace(",", " ").split()] if found else []

            rule = {"start": np.array(numbers("capsule_start_m")), "end": np.array(numbers("capsule_end_m")),
                    "capsule_radius": numbers("capsule_radius_m")[0], "centre": numbers("cylinder_center_xy_m"),
                    "cylinder_radius": numbers("cylinder_radius_m")[0], "z_min": numbers("cylinder_z_min_m")[0],
                    "z_max": numbers("cylinder_z_max_m")[0], "clearance": numbers("clearance_m")[0]}
            if len(rule["start"]) != 3 or len(rule["end"]) != 3 or len(rule["centre"]) != 2:
                rule = None
        except (OSError, IndexError, ValueError):
            rule = None
        self._sdk_rule_cache = rule
        return rule

    # ------------------------------------------------------------------ pictures
    def render(self, units: Units, *, people=(), light_rgb=None, camera: str = "orbit", width: int = 640,
               height: int = 480, orbit: tuple[float, float, float] = ORBIT,
               lookat: tuple[float, float, float] = ORBIT_LOOKAT) -> np.ndarray:
        """A picture of the lamp in `units`, as an H x W x 3 uint8 array.

        people: contract.Person heads to draw (at most MAX_PEOPLE): a skin-tone sphere, a darker face disc
            on the side the face points to, and an upright torso below.
        light_rgb: what the panel shows, LINEAR 0..1 (contract.LightFrame): one (r, g, b) or one row per
            pixel (the mean is used). The shade's diffusers glow in that colour and a light in front of the
            shade tints the room. None = panel off (plain plastic). For looking at, not for measuring light:
            the panel's own model (brightness cap, crossfades) belongs to whoever computes light_rgb.
        camera: "orbit" (a free camera at orbit = (azimuth deg, elevation deg, distance m) around `lookat`;
            azimuth -90 looks at the lamp's face) or "head" (the head camera: 61 x 44 deg whatever the
            picture size, so picture fractions mean what project() says; the head's own shell is hidden so
            the picture is what the lens sees).
        """
        if width > MAX_RENDER[0] or height > MAX_RENDER[1]:
            raise ValueError(f"render size is at most {MAX_RENDER[0]} x {MAX_RENDER[1]}")
        if camera not in ("orbit", "head"):
            raise ValueError(f"camera must be 'orbit' or 'head', not {camera!r}")
        m, d = self.model, self._render_data
        d.qpos[:] = self._qpos(units)
        self._place_people(list(people), d)
        self._apply_light(light_rgb)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = self._renderers[(width, height)] = mujoco.Renderer(m, height=height, width=width)
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 1, 0, 0]
        if camera == "head":
            option.geomgroup[3] = 0
            renderer.update_scene(d, camera="head_camera", scene_option=option)
        else:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat
            cam.azimuth, cam.elevation, cam.distance = orbit
            renderer.update_scene(d, camera=cam, scene_option=option)
        return renderer.render().copy()

    def _place_people(self, people: list, d: mujoco.MjData) -> None:
        m = self.model
        if len(people) > MAX_PEOPLE:
            raise ValueError(f"at most {MAX_PEOPLE} people can be drawn")
        for i, (head_mocap, torso_mocap, skull, face, body) in enumerate(self._people_ids):
            if i >= len(people):
                d.mocap_pos[head_mocap] = d.mocap_pos[torso_mocap] = (0.0, 0.0, -50.0)   # out of sight
                continue
            person: Person = people[i]
            r = float(person.head_radius)
            centre = np.asarray(person.head, dtype=float)
            facing = np.asarray(person.facing, dtype=float)
            if np.linalg.norm(facing) < 1e-9:
                facing = np.array([0.0, -1.0, 0.0])                # no direction given: face the lamp's side
            facing = facing / np.linalg.norm(facing)
            m.geom_size[skull, 0] = r
            m.geom_size[face, :2] = (0.62 * r, 0.004)
            m.geom_pos[face] = (0.86 * r, 0.0, 0.0)
            d.mocap_pos[head_mocap] = centre
            d.mocap_quat[head_mocap] = _quat_facing(facing)
            # An upright torso under the head, turned the way the face is (in plan view).
            torso_half = 0.22
            d.mocap_pos[torso_mocap] = centre - (0.0, 0.0, r + 0.06 + 0.14 + torso_half)
            flat = np.array([facing[0], facing[1], 0.0])
            d.mocap_quat[torso_mocap] = _quat_facing(flat if np.linalg.norm(flat) > 1e-6 else np.array([1.0, 0, 0]))
            m.geom_size[body, :2] = (0.14, torso_half)

    def _apply_light(self, light_rgb) -> None:
        m = self.model
        glow, lamp = self._glow_material, self._panel_light
        if light_rgb is None:
            m.mat_rgba[glow] = PANEL_OFF
            m.mat_emission[glow] = 0.0
            m.light_active[lamp] = 0
            return
        rgb = np.clip(np.asarray(light_rgb, dtype=float).reshape(-1, 3).mean(axis=0), 0.0, 1.0)
        srgb = rgb ** (1 / 2.2)                          # linear panel light -> display colour
        level = float(srgb.max())
        # Emissive: MuJoCo adds emission x colour to the lit colour, so the colour is scaled down to keep
        # a bright panel from washing out to white. A dark panel fades back to white plastic.
        plastic = (1.0 - level) * np.array(PANEL_OFF[:3])
        m.mat_rgba[glow] = (*np.clip(plastic + 0.6 * srgb, 0.0, 1.0), 1.0)
        m.mat_emission[glow] = 1.2 * level
        m.light_active[lamp] = 1 if level > 0.02 else 0
        m.light_diffuse[lamp] = 0.9 * srgb

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


class GravitySag:
    """Pose-dependent gravity sag for twin/motion.py SDKMotionModel(sag=...): goal (JOINTS order, units) ->
    the signed deflection (units) at which each loaded servo holds the arm.

    A position servo under load holds off its goal by load torque / stiffness. The load torque comes from
    the URDF's link masses at the goal pose (LampTwin.gravity_torque); the stiffness is unknown, so each
    joint is given its sag MAGNITUDE at the reference pose (`levels`, SDK units: an ASSUMPTION, swept by
    twin/run.py --sag-sweep) and scaled from there: deflection = -torque(pose) / |torque(reference)| x level.
    Only base_pitch, elbow_pitch and wrist_pitch carry a load (base_yaw turns about the vertical, and
    wrist_roll's torque is ignored: the head's centre of mass is close to its axis, ASSUMPTION)."""

    LOADED = ("base_pitch", "elbow_pitch", "wrist_pitch")

    def __init__(self, kin: "LampTwin", levels: dict, reference: Units | None = None):
        self.kin = kin
        self.levels = {j: float(levels.get(j, 0.0)) for j in JOINTS}
        self.reference = dict(reference) if reference is not None else kin.sag_reference_pose()
        tau = kin.gravity_torque(self.reference)
        self.reference_torque_nm = {j: tau[j] for j in self.LOADED}
        # units of sag per N m of load, per joint; a joint with no load at the reference cannot be scaled
        self.gain = np.zeros(len(JOINTS))
        for j in self.LOADED:
            if self.levels[j] and abs(tau[j]) < 1e-6:
                raise ValueError(f"{j} carries no load at the reference pose: its sag cannot be scaled")
            if self.levels[j]:
                self.gain[JOINTS.index(j)] = self.levels[j] / abs(tau[j])

    def __call__(self, goal: np.ndarray) -> np.ndarray:
        if not np.any(self.gain):
            return np.zeros(len(JOINTS))
        tau = self.kin.gravity_torque({j: float(v) for j, v in zip(JOINTS, goal, strict=True)})
        return -np.array([tau[j] for j in JOINTS]) * self.gain

    def at(self, units: Units) -> dict:
        """The deflection at a pose, by joint (for reports and tests)."""
        return {j: float(v) for j, v in zip(JOINTS, self(np.array([units[j] for j in JOINTS])), strict=True)}

    def describe(self) -> dict:
        return {"levels_at_reference_units": {j: v for j, v in self.levels.items() if v},
                "reference_pose_units": {j: round(v, 2) for j, v in self.reference.items()},
                "reference_point_m": list(SAG_REFERENCE_POINT),
                "model": "deflection = -gravity torque(pose) / |gravity torque(reference)| x level "
                         "(URDF link masses at run time)"}


def _quat_from_axes(x, y, z) -> list[float]:
    """Quaternion (w, x, y, z) of the rotation whose columns are the given frame axes (right-handed)."""
    rot = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)])
    if abs(np.linalg.det(rot) - 1.0) > 1e-6:
        raise ValueError("axes must form a right-handed orthonormal frame")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat.tolist()


def _quat_facing(direction: np.ndarray) -> np.ndarray:
    """Quaternion turning +x onto `direction`, keeping the frame's y horizontal (no roll)."""
    x = direction / np.linalg.norm(direction)
    y = np.cross([0.0, 0.0, 1.0], x)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.array(_quat_from_axes(x, y, z))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

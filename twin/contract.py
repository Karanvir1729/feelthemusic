"""Shared types for the lamp twin. Every module in twin/ speaks these; nothing here imports mujoco.

Frames and units (the same everywhere in twin/):
  * Base frame, metres: x right, y forward (where the lamp faces at base_yaw neutral), z up.
    z = 0 is the TABLE: the underside of the lamp's base plate. (The vendor's self-collision cylinder
    reaches z = -0.10; that is not the table.)
  * Joint values are the vendor SDK's units: -100..100 over each joint's calibrated range, keyed by
    the five names in JOINTS. Never degrees, never servo ticks, except inside model.py.
  * Time is float seconds on one simulation clock that starts at 0. The music's presentation time
    ("masterTs + L" on the real system) is expressed on the same clock.
  * Picture coordinates are 0..1 with (0, 0) the top-left of the head camera image, x to the right,
    y down. Sizes in the picture are fractions of the picture WIDTH.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
Units = dict  # joint name -> float, SDK units


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class Pose:
    """The head camera: optical centre and unit axes, base frame."""
    position: np.ndarray
    forward: np.ndarray      # optical axis
    down: np.ndarray         # picture +y
    right: np.ndarray        # picture +x


@dataclass(frozen=True)
class Check:
    """Is a pose allowed? Distances are signed metres between real meshes (negative = overlap)."""
    ok: bool
    reasons: list[str]
    min_table_m: float       # lowest point of any moving part above the table
    min_base_m: float        # head/links to the lamp's own base
    min_self_m: float        # moving links to each other (non-adjacent pairs)


class Kinematics(Protocol):
    neutral: Units
    limits: dict             # joint -> (lo, hi) in units, already including a safety margin

    def head(self, units: Units) -> Pose: ...
    def check(self, units: Units) -> Check: ...
    def look_at(self, point: np.ndarray, seed: Units | None = None, **kw) -> tuple[Units, dict]:
        """A whole-arm pose facing `point` that passes check(); report has at least 'aim_error_deg'."""
        ...


# ------------------------------------------------------------------ world and perception
@dataclass(frozen=True)
class Person:
    id: str
    head: np.ndarray                 # centre of the head, base frame
    facing: np.ndarray               # unit vector the face points along
    head_radius: float = 0.09


@dataclass(frozen=True)
class Detection:
    t_capture: float                 # when the camera exposed the frame: ground truth, for scoring only
    t_delivered: float               # when the tracker may use it (capture + pipeline latency)
    person_id: str                   # ground truth, for scoring only; a tracker must not rely on it
    x: float
    y: float
    size: float                      # face width in picture widths
    confidence: float
    t_stamp: float | None = None     # the frame's timestamp as the client sees it: the lamp stamps a frame
                                     # after reading and encoding it, and the client maps that stamp onto
                                     # its own clock. What a tracker must use. None = exact (t_capture).


# ------------------------------------------------------------------ motion (the vendor SDK, modelled)
@dataclass(frozen=True)
class ActionResult:
    accepted: bool
    action_id: str
    reason: str = ""                 # why refused, or ""
    planned_duration_s: float = 0.0
    clip_id: str = ""                # clip.play: the clip it uploaded first (stays on the lamp until deleted)
    t_posted: float | None = None    # when the action POST went out, if later than the call (clip.play
                                     # follows its upload's reply); None = at the call


@dataclass
class MotionState:
    commanded: Units                 # what the servos are being told now
    measured: Units                  # what /joints would report now (lag, sag)
    active_action: str | None
    idle_playing: bool
    finished: list[tuple[str, str]] = field(default_factory=list)   # (action_id, outcome) since last step


class MotionLike(Protocol):
    def submit_move(self, t: float, target: Units) -> ActionResult: ...
    def submit_clip(self, t: float, frames: list[tuple[float, Units]]) -> ActionResult: ...
    def cancel(self, t: float, action_id: str) -> None: ...
    def step(self, t: float) -> MotionState: ...


# ------------------------------------------------------------------ tracking
@dataclass(frozen=True)
class Command:
    """What the tracker asks the lamp to do. The run loop turns it into a MotionLike call."""
    kind: str                        # "move" | "clip" | "cancel"
    target: Units | None = None
    frames: list[tuple[float, Units]] | None = None
    action_id: str | None = None
    why: str = ""


# ------------------------------------------------------------------ music, haptics, light
@dataclass(frozen=True)
class MusicEvent:
    """One FTM event, on the presentation clock: the moment it must be FELT and SEEN."""
    t: float
    kind: str                        # CLICK | KICK | SNARE | BASS | BUILD | DROP (FTM kinds)
    intensity: float                 # 0..1
    sharpness: float = 0.5
    duration_ms: int = 60


@dataclass(frozen=True)
class LightFrame:
    """What the 93-pixel panel shows at time t. rgb is linear 0..1, one row per pixel."""
    t: float
    rgb: np.ndarray                  # shape (93, 3)

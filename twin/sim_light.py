"""The lamp's head light as the vendor SDK really drives it: `light.glow`, simulated at the command level.

Used by twin/sim_sdk.py (the simulated SDK gateway) so that the team's light code meets the same timing
the real lamp imposes. Nothing here draws pictures (twin/panel.py does that); this module decides WHAT the
93 pixels hold at any moment and WHEN each SDK call would answer.

What is modelled (vendor rules re-derived in our own words; file:line citations are relative to the
vendor's private runtime repository, which is not in this one):

  * colour glow (`color`, optional `luminance`/`intensity`): a smootherstep cross-fade of every pixel and
    of the driver brightness over `transition_ms` (600 by default), serialised FIFO behind one output
    lock; the SDK reply comes when the fade ends (mapper.py:153-184, lights/ambient/manager.py:273-292,
    panel/controller.py:120-150, rendering/transitions.py:9-38);
  * luminance-only glow: a brightness fade of the same length that takes NO lock and cancels whatever
    fade is running (manager.py:303-330, controller.py:152-170);
  * effect glow (`animation`/`effect`, also `{}` and `{"intensity": x}` alone): one of five canned
    effects, drawn from the next loop turn at 20 frames/s, replacing any fade; the reply comes at once,
    before the first frame, and an unknown name still "succeeds" (mapper.py:185-207,
    controller.py:172-202, effects/player.py:13-69, rendering/procedural.py:22-131);
  * a fade cancelled by a later call never answers: its SDK call fails 503 four seconds after the
    cancel ("No lifecycle acknowledgment for light.command", mapper.py:176, core/event_lifecycle.py:189-204);
  * `stop` (only system.stop sends it): brightness fade to 0, then the panel is cleared and the
    background effect forgotten (manager.py:487-500, controller.py:204-212).

State and time: the panel holds 93 pixels of 8-bit colour plus one driver brightness 0..1, exactly like
the vendor controller (controller.py:261-293, drivers/base.py:73-77); what you see is pixel x brightness,
and the NeoPixel object multiplies by a further hardware cap of 0.3 (see twin/panel.py HARDWARE_CAP).
Time is float seconds on the caller's clock. The model is event driven: `update(t)` processes every
lock hand-off, fade end and effect end at its exact time, so the result does not depend on how often the
caller steps. `sample(t)` gives the panel at any t not earlier than the last update.

  light = SimLight(transition_ms=600)
  cmd = light.glow({"color": [0, 120, 255], "luminance": 0.6}, t=1.0)
  light.update(1.7); cmd.status, cmd.reply_at   -> "succeeded", 1.6
  light.sample(1.3)                             -> (93, 3) uint8, pixel x brightness
"""
from __future__ import annotations

import colorsys
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ constants (source per line)
PIXEL_COUNT = 93                 # vendor source config/default.yaml:39; robot simulation.yaml:16
TRANSITION_MS = 600.0            # light_policy.transition_ms: vendor source config/default.yaml:63; the live lamp
                                 # reported 600 (twin-spec light.md section 1, research pi-control-surface.md:156)
FADE_FRAME_S = 1.0 / 60.0        # fades draw at most 60 frames/s, first frame one frame in: transitions.py:9, 28-38
EFFECT_FRAME_HZ = 20.0           # effect players draw 20 frames/s nominal: effects/player.py:48
EFFECT_OVERRUN = 0.05            # ASSUMPTION: each effect frame also pays render + show(), so effects run 3-10 %
                                 # slow (twin-spec light.md section 2); 5 % is the middle, not a measurement
LIFECYCLE_ACK_S = 4.0            # an orphaned colour/brightness call gives up 4 s after its handler ends:
                                 # vendor source sdk_gateway/mapper.py:145, :176; core/event_lifecycle.py:168-172
START_COLOR = (255, 224, 180)    # ambient warm white the policy leaves on the panel: manager.py:143-152, 1057-1068
START_BRIGHTNESS = 0.03          # driver level measured on the live lamp (twin-spec light.md section 8,
                                 # research pi-control-surface.md:158); ambient idle is 0.05/0.06 by config
HARDWARE_CAP = 0.3               # NeoPixel brightness: vendor source
                                 # config/profiles/lights/raspberry_pi_neopixel.yaml:29

# The five canned effects (vendor source rendering/effects/player.py:13-44; drawn by procedural.py:22-131).
# kind, colours, speed in cycles per second, spread, seed. Intensity is 1.0 for all of them.
EFFECTS = {
    "rainbow": ("rainbow", ((255, 0, 0),), 0.35, 1.0, 0),
    "breathing": ("breathe", ((255, 180, 50), (255, 224, 180)), 0.42, 1.0, 0),
    "flowing": ("chase", ((0, 255, 255),), 0.35, 0.8, 0),
    "sparkle": ("sparkle", ((70, 80, 100), (255, 255, 255)), 2.0, 0.8, 41),
    "police": ("chase", ((255, 0, 0), (0, 0, 255)), 2.5, 1.0, 911),
}


class GlowInvalid(ValueError):
    """A light.glow payload the vendor mapper refuses with 400 invalid_request (mapper.py:103-132)."""


class GlowCrash(RuntimeError):
    """A payload the vendor mapper crashes on (a non-numeric `timeout`): 500 action_failed (service.py:492-497)."""


def smootherstep(u: float) -> float:
    """6u^5 - 15u^4 + 10u^3 on 0..1, zero slope at both ends (vendor source rendering/transitions.py:12-15)."""
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def fade_progress(elapsed_s: float, transition_s: float) -> float:
    """How far a light.glow fade has drawn after `elapsed_s`, 0..1: frames land every 1/60 s from one frame in,
    each at smootherstep(frame time / T), and between frames the panel holds the last one
    (vendor source rendering/transitions.py:9-38). The one rule twin/show.py SdkLamp and SimLight share."""
    elapsed = max(0.0, float(elapsed_s))
    if transition_s <= 0 or elapsed >= transition_s - 1e-12:
        return 1.0
    return smootherstep(math.floor(elapsed / FADE_FRAME_S + 1e-9) * FADE_FRAME_S / transition_s)


def _optional_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _channel(value) -> int:
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError):
        return 0


def rgb_from_payload(payload: dict):
    """The colour a glow names, or None: '#RRGGBB'/'RRGGBB', a list of 3, or r/g/b keys (mapper.py:550-571)."""
    color = payload.get("color")
    if isinstance(color, str):
        value = color.strip().lstrip("#")
        if len(value) == 6:
            try:
                return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
            except ValueError:
                return None
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return _channel(color[0]), _channel(color[1]), _channel(color[2])
    if all(key in payload for key in ("r", "g", "b")):
        return _channel(payload["r"]), _channel(payload["g"]), _channel(payload["b"])
    return None


def parse_glow(payload: dict) -> dict:
    """Validate a light.glow payload and pick its branch, in the vendor mapper's order (mapper.py:95-207).

    Returns {"kind": "solid"|"brightness"|"effect", ...}. Raises GlowInvalid (400) or GlowCrash (500)."""
    payload = payload if isinstance(payload, dict) else {}
    for name in ("luminance", "intensity"):
        if name in payload:
            value = payload[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise GlowInvalid(f"{name} must be a number between 0 and 1")
    if "color" in payload:
        raw = payload["color"]
        valid_hex = isinstance(raw, str) and len(raw.strip().lstrip("#")) == 6 and rgb_from_payload(payload) is not None
        valid_list = (isinstance(raw, (list, tuple)) and len(raw) == 3
                      and all(type(v) is int and 0 <= v <= 255 for v in raw))
        if not (valid_hex or valid_list):
            raise GlowInvalid("color must be three whole numbers from 0 to 255")
    color = rgb_from_payload(payload)
    effect = payload.get("animation") or payload.get("effect")
    if "luminance" in payload and color is None:
        # Brightness first; an effect named alongside is started after the fade (mapper.py:134-152, :185).
        return {"kind": "brightness", "luminance": float(payload["luminance"]),
                "effect": str(effect) if effect else None,
                "duration": _optional_float(payload.get("duration", payload.get("duration_s")))}
    try:
        timeout = float(payload.get("timeout") or (4.0 if color is not None else 1.0))
    except (TypeError, ValueError) as exc:
        raise GlowCrash(str(exc)) from None
    if color is not None:
        level = payload.get("luminance", payload.get("intensity"))
        return {"kind": "solid", "color": tuple(color), "luminance": None if level is None else float(level),
                "timeout": timeout}
    return {"kind": "effect", "effect": str(effect or "breathing"), "timeout": timeout,
            "duration": _optional_float(payload.get("duration", payload.get("duration_s")))}


def effect_frame(name: str, elapsed_s: float, count: int = PIXEL_COUNT) -> np.ndarray:
    """The 8-bit pixels one effect frame writes after `elapsed_s` of effect time: colour x the effect's own
    brightness, rounded (vendor source effects/player.py:56-69; procedural.py:22-131, in our own words)."""
    kind, colours, speed, spread, seed = EFFECTS[name]
    phase = max(0.0, float(elapsed_s)) * speed
    idx = np.arange(count, dtype=float)
    palette = np.array(colours, dtype=float)
    if kind == "breathe":
        wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase - math.pi / 2.0)     # starts at its minimum
        colour = palette[0] + (palette[-1] - palette[0]) * wave
        rgb = np.tile(colour, (count, 1))
        level = 0.2 + 0.8 * wave
    elif kind == "rainbow":
        offset = (seed % 997) / 997.0
        rgb = np.array([[round(c * 255) for c in colorsys.hsv_to_rgb((phase + offset + spread * i / count) % 1.0,
                                                                      1.0, 1.0)] for i in range(count)], float)
        level = 1.0
    elif kind == "chase":
        head = phase * count
        trail = max(1.0, count * max(0.08, spread) * 0.35)
        distance = (head - idx) % count
        strength = np.where(distance > trail, 0.0, 1.0 - distance / trail)
        colour = palette[int(math.floor(head / count)) % len(palette)]
        rgb = np.round(strength[:, None] * colour[None, :])
        level = 1.0
    else:  # sparkle: a dim base with random twinkles re-drawn 8 times per cycle (our own hash, same statistics)
        frame = int(math.floor(phase * 8.0))
        rng = np.random.default_rng([seed, frame])
        density = 0.04 + spread * 0.28
        base = np.round(palette[0] * 0.12)
        hit, pick, amount = rng.random(count), rng.random(count), rng.random(count)
        chosen = palette[np.minimum(len(palette) - 1, (pick * len(palette)).astype(int))]
        strength = (0.55 + 0.45 * amount)[:, None]
        rgb = np.where((hit >= 1.0 - density)[:, None], np.round(base + (chosen - base) * strength), base)
        level = 1.0
    return np.clip(np.round(rgb * level), 0, 255)


# ------------------------------------------------------------------ one SDK call
@dataclass
class LightCommand:
    """One light command as the lamp handled it. Times are on the caller's clock; None = not reached."""
    id: int
    kind: str                      # solid | brightness | effect | stop
    payload: dict
    requested_at: float
    started_at: float | None = None       # fade (or effect) began
    ended_at: float | None = None         # fade finished, effect RUNNING, or the moment it was cancelled
    reply_at: float | None = None         # when the SDK call answers
    status: str = "queued"                # queued | running | succeeded | orphaned
    detail: str = ""
    color: tuple | None = None
    level: float | None = None            # target driver brightness
    effect: str | None = None
    duration: float | None = None
    session: str = ""
    result: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in ("succeeded", "orphaned")

    def log_row(self) -> dict:
        return {"id": self.id, "kind": self.kind, "payload": self.payload, "requested_at": self.requested_at,
                "started_at": self.started_at, "ended_at": self.ended_at, "reply_at": self.reply_at,
                "status": self.status, "detail": self.detail}


class SimLight:
    """The head panel behind `light.glow`, as the vendor light manager and panel controller behave."""

    def __init__(self, transition_ms: float = TRANSITION_MS, *, pixel_count: int = PIXEL_COUNT,
                 start_color=START_COLOR, start_brightness: float = START_BRIGHTNESS,
                 effect_overrun: float = EFFECT_OVERRUN, t0: float = 0.0):
        self.transition_s = max(0.0, float(transition_ms) / 1000.0)
        self.count = int(pixel_count)
        self.effect_period_s = (1.0 / EFFECT_FRAME_HZ) * (1.0 + float(effect_overrun))
        self._pixels = np.tile(np.array(start_color, float), (self.count, 1))
        self._brightness = float(start_brightness)
        self._color = tuple(start_color) if start_color is not None else None   # controller state colour
        self._animation: str | None = None                                       # controller state animation
        self._fade: dict | None = None
        self._effect: dict | None = None
        self._background: dict | None = None     # the last endless effect: resumes after a timed one ends
        self._holder: LightCommand | None = None  # the solid glow that holds the output lock
        self._queue: deque[LightCommand] = deque()
        self._t = float(t0)
        self._next_id = 0
        self.log: list[LightCommand] = []
        self.manual_override_until = None         # every SDK light call holds ambient off for 900 s
        self.MANUAL_OVERRIDE_S = 900.0            # vendor source config/default.yaml:73; manager.py:920-931

    # ---------------------------------------------------------------- the SDK entry point
    def glow(self, payload: dict, *, t: float, session: str = "") -> LightCommand:
        """Handle one light.glow at time t. Returns the command; its reply_at/status fill in as update() runs.
        Raises GlowInvalid / GlowCrash for payloads the mapper refuses (nothing changes on the panel)."""
        spec = parse_glow(payload)
        self.update(t)
        self.manual_override_until = t + self.MANUAL_OVERRIDE_S
        cmd = self._new(spec["kind"], payload, t, session)
        if spec["kind"] == "solid":
            # The mapper reads the current driver level at request time when no luminance/intensity is given
            # (mapper.py:154-157), so a colour-only glow keeps whatever dim level ambient left.
            cmd.color = spec["color"]
            cmd.level = spec["luminance"] if spec["luminance"] is not None else self._brightness_at(t)
            self._queue.append(cmd)
            if self._holder is None:
                self._grant_lock(t)
        elif spec["kind"] == "brightness":
            cmd.level, cmd.effect, cmd.duration = spec["luminance"], spec["effect"], spec["duration"]
            self._start_brightness(cmd, t)
        else:
            cmd.effect, cmd.duration = spec["effect"], spec["duration"]
            self._start_effect_request(cmd, t)
        return cmd

    def stop(self, *, t: float) -> LightCommand:
        """The light part of system.stop: fade to 0 (no lock), then clear everything (manager.py:487-500)."""
        self.update(t)
        self.manual_override_until = t + self.MANUAL_OVERRIDE_S
        cmd = self._new("stop", {}, t, "")
        cmd.level = 0.0
        self._cancel_fade(t)
        cmd.started_at, cmd.status = t, "running"
        self._fade = {"cmd": cmd, "t0": t, "start_pixels": None, "start_b": self._brightness_at(t),
                      "target": None, "target_b": 0.0}
        self._grant_lock(t)          # a colour glow waiting behind a cancelled fade takes the lock (and wins)
        if self.transition_s <= 0 and self._fade is not None and self._fade["cmd"] is cmd:
            self._handle_event(t)
        return cmd

    # ---------------------------------------------------------------- time
    def update(self, t: float) -> None:
        """Advance to time t, handling every lock hand-off, fade end and effect end at its exact time."""
        t = float(t)
        while True:
            when = self.next_event(after=self._t)
            if when is None or when > t:
                break
            self._t = max(self._t, when)
            self._handle_event(when)
        self._t = max(self._t, t)

    def next_event(self, after: float | None = None) -> float | None:
        """The next moment the panel's program changes by itself (fade end, timed effect end), or None."""
        times = []
        if self._fade is not None:
            times.append(self._fade["t0"] + self.transition_s)
        if self._effect is not None and self._effect["duration"] is not None:
            times.append(self._effect_end(self._effect))
        return min(times) if times else None

    # ---------------------------------------------------------------- what the panel shows
    def pixels(self, t: float) -> np.ndarray:
        """The 8-bit colours written to the 93 pixels at time t (before driver brightness)."""
        return self._state_at(t)[0]

    def brightness(self, t: float) -> float:
        return self._brightness_at(t)

    def sample(self, t: float) -> np.ndarray:
        """(93, 3) uint8: pixel x driver brightness, i.e. the SDK-level output (the 0.3 NeoPixel cap is extra)."""
        px, level = self._state_at(t)
        return np.clip(np.round(px * level), 0, 255).astype(np.uint8)

    def linear(self, t: float, *, hardware_cap: bool = True) -> np.ndarray:
        """(93, 3) float LED drive fraction 0..1 (contract.LightFrame convention, twin/panel.py): pixel/255 x
        brightness, times the NeoPixel cap unless hardware_cap is False."""
        px, level = self._state_at(t)
        return px / 255.0 * level * (HARDWARE_CAP if hardware_cap else 1.0)

    def state(self, t: float) -> dict:
        """A snapshot like the vendor controller's (colour, animation, brightness_level), for status and traces."""
        self.update(t)
        animation = self._animation
        if self._fade is not None:
            animation = "transition"
        return {"color": list(self._color) if self._color else None, "animation": animation,
                "brightness_level": round(self._brightness_at(t), 6),
                "effect": self._effect["name"] if self._effect else None,
                "background_effect": self._background["name"] if self._background else None,
                "lock_holder": self._holder.id if self._holder else None, "queued": len(self._queue)}

    # ---------------------------------------------------------------- internals
    def _new(self, kind: str, payload: dict, t: float, session: str) -> LightCommand:
        self._next_id += 1
        cmd = LightCommand(self._next_id, kind, dict(payload) if isinstance(payload, dict) else {}, float(t),
                           session=session)
        self.log.append(cmd)
        return cmd

    def _effect_end(self, effect: dict) -> float:
        # The effect loop checks elapsed >= duration before each frame (controller.py:242-252): it stops at the
        # first frame time at or past the duration.
        duration = max(0.0, effect["duration"])
        frames = math.ceil(duration / self.effect_period_s - 1e-9)
        return effect["t0"] + frames * self.effect_period_s

    def _effect_pixels(self, effect: dict, t: float) -> np.ndarray:
        k = int(math.floor((t - effect["t0"]) / self.effect_period_s + 1e-9))
        if effect["duration"] is not None:
            last = max(0, math.ceil(max(0.0, effect["duration"]) / self.effect_period_s - 1e-9) - 1)
            k = min(k, last)
        if k < 0:
            return effect["base_pixels"]
        return effect_frame(effect["name"], (effect["frame0"] + k) / EFFECT_FRAME_HZ, self.count)

    def _fade_progress(self, fade: dict, t: float) -> float:
        return fade_progress(t - fade["t0"], self.transition_s)

    def _state_at(self, t: float) -> tuple[np.ndarray, float]:
        px = self._effect_pixels(self._effect, t) if self._effect is not None else self._pixels
        level = self._brightness
        fade = self._fade
        if fade is not None:
            e = self._fade_progress(fade, t)
            level = fade["start_b"] + (fade["target_b"] - fade["start_b"]) * e
            if fade["target"] is not None:       # a colour fade moves every pixel; brightness fades do not
                target = np.array(fade["target"], float)[None, :]
                px = np.round(fade["start_pixels"] + (target - fade["start_pixels"]) * e)
        return px, level

    def _brightness_at(self, t: float) -> float:
        return float(self._state_at(t)[1])

    def _freeze(self, t: float) -> None:
        """Make the state at t the new static base (used before any interruption)."""
        px, level = self._state_at(t)
        self._pixels, self._brightness = np.array(px, float), float(level)

    def _cancel_fade(self, t: float) -> None:
        """A fade cancelled mid-way leaves the panel where it was; its SDK call never gets a result and
        fails 4 s later (controller.py:147-150, 233-240; manager.py:622-632; mapper.py:176)."""
        if self._fade is None:
            return
        self._freeze(t)
        cmd = self._fade["cmd"]
        self._fade = None
        self._animation = None
        if not cmd.done:
            cmd.status, cmd.ended_at, cmd.reply_at = "orphaned", t, t + LIFECYCLE_ACK_S
            cmd.detail = "fade cancelled by a later light call: no lifecycle acknowledgment"
        if self._holder is cmd:
            self._holder = None          # the cancelled set_solid leaves the output lock

    def _cancel_effect(self, t: float) -> None:
        if self._effect is None:
            return
        self._freeze(t)
        effect = self._effect
        if effect is self._background:
            # the endless player object keeps its frame counter (controller.py:188-202)
            k = int(math.floor((t - effect["t0"]) / self.effect_period_s + 1e-9)) + 1
            effect["frame0"] += max(0, k)
        self._effect = None

    def _grant_lock(self, t: float) -> None:
        """Hand the output lock to the next queued colour glow and start its fade (manager.py:273-292)."""
        while self._holder is None and self._queue:
            cmd = self._queue.popleft()
            self._holder = cmd
            # transition_to_solid cancels any effect and any running fade first (controller.py:128-131)
            self._cancel_effect(t)
            self._cancel_fade(t)
            cmd.started_at, cmd.status = t, "running"
            self._animation = "transition"
            self._fade = {"cmd": cmd, "t0": t, "start_pixels": self._pixels.copy(), "start_b": self._brightness,
                          "target": cmd.color, "target_b": max(0.0, min(1.0, cmd.level))}
            if self.transition_s <= 0:
                self._handle_event(t)

    def _start_brightness(self, cmd: LightCommand, t: float) -> None:
        # set_brightness: if nothing is showing, snap to white first; then fade, cancelling any fade
        # (manager.py:303-330, controller.py:152-170). No output lock.
        if cmd.level > 0 and self._color is None and self._effect is None and self._fade is None:
            self._pixels = np.tile(np.array((255.0, 255.0, 255.0)), (self.count, 1))
            self._color = (255, 255, 255)
        self._cancel_fade(t)
        cmd.started_at, cmd.status = t, "running"
        # a brightness fade leaves the pixels (and any running effect) alone; only the driver level moves
        self._fade = {"cmd": cmd, "t0": t, "start_pixels": None, "start_b": self._brightness_at(t),
                      "target": None, "target_b": cmd.level}
        self._grant_lock(t)                      # a holder cancelled above hands the lock on at once
        if self.transition_s <= 0 and self._fade is not None and self._fade["cmd"] is cmd:
            self._handle_event(t)

    def _start_effect_request(self, cmd: LightCommand, t: float) -> None:
        cmd.started_at = t
        name = str(cmd.effect or "breathing")
        if name.lower() not in EFFECTS:
            # RUNNING is published before the controller looks the name up; the lookup then raises and the
            # panel is left alone (manager.py:677-695, controller.py:172-176).
            cmd.status, cmd.ended_at, cmd.reply_at = "succeeded", t, t
            cmd.detail = f"unknown effect {name!r}: nothing drawn (the SDK still reports success)"
            cmd.result = {"light": "play_animation", "animation": name, "state": "RUNNING"}
            return
        self._start_effect(name.lower(), cmd.duration, t)
        cmd.status, cmd.ended_at, cmd.reply_at = "succeeded", t, t
        cmd.result = {"light": "play_animation", "animation": name, "state": "RUNNING"}
        self._grant_lock(t)                      # a colour glow queued behind a cancelled fade starts now

    def _start_effect(self, name: str, duration: float | None, t: float) -> None:
        self._cancel_effect(t)
        self._cancel_fade(t)
        effect = {"name": name, "t0": t, "duration": duration, "frame0": 0, "base_pixels": self._pixels.copy()}
        self._effect = effect
        self._animation = name
        if duration is None:
            self._background = effect

    def _handle_event(self, t: float) -> None:
        fade_end = self._fade["t0"] + self.transition_s if self._fade is not None else None
        effect_end = (self._effect_end(self._effect)
                      if self._effect is not None and self._effect["duration"] is not None else None)
        if fade_end is not None and (effect_end is None or fade_end <= effect_end):
            self._end_fade(fade_end)
        elif effect_end is not None:
            self._end_effect(effect_end)

    def _end_fade(self, t: float) -> None:
        fade = self._fade
        cmd = fade["cmd"]
        self._fade = None
        self._brightness = fade["target_b"]      # the fade's last frame lands exactly on target
        cmd.status, cmd.ended_at, cmd.reply_at = "succeeded", t, t
        if cmd.kind == "solid":
            self._color, self._animation = tuple(cmd.color), None
            self._pixels = np.tile(np.array(cmd.color, float), (self.count, 1))
            self._brightness = fade["target_b"]
            cmd.result = {"light": "set_solid", "state": "SUCCEEDED"}
            if self._holder is cmd:
                self._holder = None
            self._grant_lock(t)
        elif cmd.kind == "brightness":
            self._brightness = fade["target_b"]
            cmd.result = {"state": "SUCCEEDED"}
            if self._animation == "transition":
                self._animation = None
            if cmd.effect:
                # the mapper then publishes play_animation; it answers at RUNNING (mapper.py:147-148, :185-207)
                name = str(cmd.effect)
                if name.lower() in EFFECTS:
                    self._start_effect(name.lower(), cmd.duration, t)
                else:
                    cmd.detail = f"unknown effect {name!r}: nothing drawn (the SDK still reports success)"
                cmd.result = {"light": "play_animation", "animation": name, "state": "RUNNING"}
                self._grant_lock(t)
        elif cmd.kind == "stop":
            # controller.stop(): cancel effects, forget the background, clear the panel (controller.py:204-212)
            self._effect, self._background = None, None
            self._pixels = np.zeros((self.count, 3))
            self._brightness = 0.0
            self._color, self._animation = None, None
            cmd.result = {"state": "SUCCEEDED"}

    def _end_effect(self, t: float) -> None:
        # A timed effect leaves its last frame, then the endless background effect (if any) resumes with its
        # own frame counter (controller.py:193-202, 254-259).
        self._freeze(t)
        self._effect = None
        self._animation = None
        if self._background is not None:
            background = self._background
            background["t0"], background["base_pixels"] = t, self._pixels.copy()
            self._effect = background
            self._animation = background["name"]

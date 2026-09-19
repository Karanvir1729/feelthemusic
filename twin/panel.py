"""The lamp's 93-pixel head panel: where each pixel sits, how to draw it, what colour the diffuser shows.

  load_geometry()             pixel positions (metres, panel frame) and which ring each pixel is on
  perceived_colour(rgb93)     one colour for the whole diffuser: a weighted mean of the pixels
  luminance(rgb)              relative luminance of LINEAR rgb (Rec. 709 / sRGB primaries)
  draw_panel(rgb93, size_px)  a picture of the panel: glowing dots on dark, uint8 RGB

Colour convention (the same in twin/show.py): "linear" rgb in 0..1 is the LED drive fraction, i.e. the
PWM duty the WS2812 gets. The vendor runtime applies no gamma between an SDK colour and the PWM value
(vendor source, see research report twin-spec/light.md section 4 "gamma"), so SDK colour / 255 is linear
light. A LightFrame (twin/contract.py) holds what the panel physically shows, so its rgb already includes
the NeoPixel hardware cap: at most HARDWARE_CAP per channel.

The panel layout belongs to the lamp's vendor. It is read at run time from the robot description
(FTM_ROBOT_DIR/simulation.yaml, "light:" block) and never copied here. Without that file the panel falls
back to a clearly labelled GENERIC layout (a sunflower disc), good for drawing and for "centre vs outer"
logic, but not the lamp's real ring layout.

Panel frame: x to the right and y up, as seen by someone looking INTO the panel from the front.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PIXEL_COUNT = 93            # contract.LightFrame rgb shape (93, 3); vendor source simulation.yaml:16 pixel_count
HARDWARE_CAP = 0.3          # NeoPixel brightness cap on top of SDK luminance: vendor source
                            # config/profiles/lights/raspberry_pi_neopixel.yaml:29, drivers/blinka_neopixel.py:68-73
DISPLAY_GAMMA = 2.2         # only for drawing pictures; the vendor's own simulator uses 2.2 for emitted light
                            # (vendor source simulation.yaml:20). The real LED path has no gamma.
LUMA = np.array([0.2126, 0.7152, 0.0722])   # relative luminance weights of linear sRGB primaries (WCAG / Rec. 709)

GENERIC_RADIUS_M = 0.048    # ASSUMPTION for the generic fallback only: outer pixel radius of a ~10 cm panel
GENERIC_RINGS = 6           # ASSUMPTION for the generic fallback only: centre pixel + 5 radius bands
ROBOT_DIR_ENV = "FTM_ROBOT_DIR"

# Pixel index order on the real lamp. No vendor code maps an index to a ring; the ring list order and the
# start angle / clockwise fields suggest centre pixel = 0, then each ring outward, each ring starting at
# the start angle and running clockwise. ASSUMPTION (research report twin-spec/light.md section 4 "index
# order"): check it on the lamp by watching the "flowing" chase once.
INDEX_ORDER = "centre = 0, then rings outward, each ring from start_angle running clockwise (ASSUMPTION)"


@dataclass(frozen=True, eq=False)
class PanelGeometry:
    xy: np.ndarray                   # (93, 2) metres, panel frame (x right, y up, seen from the front)
    ring: np.ndarray                 # (93,) ring number per pixel, 0 = centre
    ring_counts: tuple               # pixels per ring, centre first
    pixel_radius_m: float            # radius of one LED's lit spot (for drawing)
    diffuser_radius_m: float         # radius of the diffuser disc (for drawing)
    source: str                      # where the layout came from
    index_order: str                 # what is assumed about the index order

    @property
    def n_rings(self) -> int:
        return len(self.ring_counts)

    def ring_pixels(self, k: int) -> np.ndarray:
        """Indices of the pixels on ring k (0 = centre; negative counts from the outside, -1 = outermost)."""
        k = k % self.n_rings
        return np.flatnonzero(self.ring == k)

    def outer(self, n_rings: int = 1) -> np.ndarray:
        """Indices of the n outermost rings."""
        return np.flatnonzero(self.ring >= self.n_rings - n_rings)


# ------------------------------------------------------------------ geometry
def _parse_value(text: str):
    text = text.split("#", 1)[0].strip()
    if text.startswith("[") and text.endswith("]"):
        return [_parse_value(part) for part in text[1:-1].split(",") if part.strip()]
    if text in ("true", "false"):
        return text == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text.strip("\"'")


def read_vendor_light(robot_dir: Path | str | None = None) -> dict | None:
    """The flat key/value pairs of the "light:" block of the vendor's simulation.yaml, or None.

    A tiny line parser (no YAML dependency): the block is plain "key: value" lines, nested one level
    ("topology:"), with scalar or [list] values. Nested keys are flattened; in that block they are unique.
    """
    robot_dir = Path(robot_dir or os.environ.get(ROBOT_DIR_ENV, ""))
    path = robot_dir / "simulation.yaml"
    if not str(robot_dir) or not path.is_file():
        return None
    out, inside = {}, False
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        top_level = not line[0].isspace()
        if top_level:
            inside = line.split(":", 1)[0].strip() == "light"
            continue
        if inside and ":" in line:
            key, value = line.split(":", 1)
            if value.strip():
                out[key.strip()] = _parse_value(value)
    return out or None


def layout_from_rings(counts, radii, start_deg: float, clockwise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pixel xy and ring number for concentric rings, in the index order described by INDEX_ORDER."""
    xy, ring = [], []
    sign = -1.0 if clockwise else 1.0          # clockwise as seen from the front = decreasing math angle
    for k, (n, r) in enumerate(zip(counts, radii, strict=True)):
        for j in range(int(n)):
            a = math.radians(start_deg + sign * 360.0 * j / n)
            xy.append((r * math.cos(a), r * math.sin(a)))
            ring.append(k)
    return np.array(xy, dtype=float), np.array(ring, dtype=int)


def generic_geometry() -> PanelGeometry:
    """GENERIC fallback (not the lamp): 93 pixels on a sunflower disc, rings = radius bands."""
    i = np.arange(PIXEL_COUNT)
    r = GENERIC_RADIUS_M * np.sqrt(i / (PIXEL_COUNT - 1))
    a = math.radians(90.0) - i * math.pi * (3.0 - math.sqrt(5.0))      # golden angle, pixel 1 near the top
    xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    bands = GENERIC_RINGS - 1
    ring = np.where(i == 0, 0, 1 + np.minimum(bands - 1, (r / GENERIC_RADIUS_M * bands).astype(int)))
    counts = tuple(int(np.sum(ring == k)) for k in range(GENERIC_RINGS))
    return PanelGeometry(xy=xy, ring=ring, ring_counts=counts, pixel_radius_m=0.0025,   # ASSUMPTION: 5 mm LED spot
                         diffuser_radius_m=GENERIC_RADIUS_M + 0.007,                   # ASSUMPTION
                         source="GENERIC fallback layout (ASSUMPTION: vendor simulation.yaml not found; "
                                "not the lamp's real ring layout)",
                         index_order="generic: centre = 0, then outward by radius")


@lru_cache(maxsize=4)
def _load_geometry(robot_dir: str) -> PanelGeometry:
    light = read_vendor_light(robot_dir or None)
    try:
        counts = [int(c) for c in light["ring_counts"]]
        radii = [float(r) for r in light["ring_radii_m"]]
        if sum(counts) != int(light.get("pixel_count", PIXEL_COUNT)) or sum(counts) != PIXEL_COUNT:
            raise ValueError("ring counts do not add up to the pixel count")
        if len(counts) != len(radii) or any(b <= a for a, b in zip(radii, radii[1:], strict=False)):
            raise ValueError("ring radii must match the counts and increase")
        xy, ring = layout_from_rings(counts, radii, float(light.get("start_angle_degrees", 90.0)),
                                     bool(light.get("clockwise", True)))
    except (TypeError, KeyError, ValueError):
        return generic_geometry()
    return PanelGeometry(xy=xy, ring=ring, ring_counts=tuple(counts),
                         pixel_radius_m=float(light.get("pixel_radius_m", 0.0025)),
                         diffuser_radius_m=float(light.get("diffuser_radius_m", radii[-1] + 0.007)),
                         source="vendor robot description simulation.yaml (read at run time from FTM_ROBOT_DIR)",
                         index_order=INDEX_ORDER)


def load_geometry(robot_dir: Path | str | None = None) -> PanelGeometry:
    """The lamp's panel from FTM_ROBOT_DIR (or robot_dir); the generic fallback when it is not there."""
    return _load_geometry(str(robot_dir or os.environ.get(ROBOT_DIR_ENV, "")))


# ------------------------------------------------------------------ colour
def luminance(rgb) -> np.ndarray:
    """Relative luminance of linear rgb (..., 3)."""
    return np.asarray(rgb, dtype=float) @ LUMA


def diffuser_weights(geometry: PanelGeometry | None = None, sigma_m: float | None = None) -> np.ndarray:
    """Weights for the perceived colour. Default: every LED counts the same (ASSUMPTION: the diffuser mixes
    all 93 evenly, and the room light is the sum of all pixels, as research report deaf-hoh-design.md:162
    asks the flash governor to use). sigma_m gives a centre-weighted Gaussian instead (a viewer close to
    the diffuser sees the centre more)."""
    if sigma_m is None:
        return np.full(PIXEL_COUNT, 1.0 / PIXEL_COUNT)
    geometry = geometry or load_geometry()
    w = np.exp(-0.5 * np.sum(geometry.xy ** 2, axis=1) / sigma_m ** 2)
    return w / w.sum()


def perceived_colour(rgb93, weights: np.ndarray | None = None) -> np.ndarray:
    """One linear colour for the whole diffuser: the weighted mean of the 93 pixels."""
    rgb93 = np.asarray(rgb93, dtype=float).reshape(-1, 3)
    w = diffuser_weights() if weights is None else np.asarray(weights, dtype=float)
    return w @ rgb93


def linear_to_srgb(x) -> np.ndarray:
    """Linear 0..1 -> sRGB-encoded 0..1 (IEC 61966-2-1), for flash analysis and pictures."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


# ------------------------------------------------------------------ drawing
def draw_panel(rgb93, size_px: int = 256, *, geometry: PanelGeometry | None = None, gain: float | None = None,
               background=(6, 6, 10), halo: float = 0.18) -> np.ndarray:
    """A picture of the panel: (size_px, size_px, 3) uint8, glowing dots on a dark disc.

    rgb93 is what the panel shows (linear, including the hardware cap). `gain` scales it for the picture;
    the default 1 / HARDWARE_CAP draws a pixel at full SDK drive as full brightness. `halo` adds a soft
    wide glow per pixel, standing in for the diffuser. Pure numpy.
    """
    geometry = geometry or load_geometry()
    rgb = np.clip(np.asarray(rgb93, dtype=float).reshape(-1, 3) * (1.0 / HARDWARE_CAP if gain is None else gain),
                  0.0, 1.0)
    size = int(size_px)
    img = np.zeros((size, size, 3))
    scale = 0.46 * size / geometry.diffuser_radius_m            # metres -> picture pixels
    cx = cy = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (geometry.diffuser_radius_m * scale) ** 2
    img[disc] = 0.012                                            # the unlit diffuser, a little lighter than the table
    core = max(1.0, geometry.pixel_radius_m * scale)             # sigma of a lit LED spot, in picture pixels
    wide = 3.5 * core
    reach = int(math.ceil(3 * wide))
    for (x, y), colour in zip(geometry.xy, rgb, strict=True):
        if not colour.any():
            continue
        px, py = cx + x * scale, cy - y * scale                  # picture y points down
        x0, x1 = max(0, int(px) - reach), min(size, int(px) + reach + 1)
        y0, y1 = max(0, int(py) - reach), min(size, int(py) + reach + 1)
        d2 = (xx[y0:y1, x0:x1] - px) ** 2 + (yy[y0:y1, x0:x1] - py) ** 2
        shape = np.exp(-0.5 * d2 / core ** 2) + halo * np.exp(-0.5 * d2 / wide ** 2)
        img[y0:y1, x0:x1] += shape[..., None] * colour
    img = np.clip(img, 0.0, 1.0) ** (1.0 / DISPLAY_GAMMA)
    out = (img * 255.0 + 0.5).astype(np.uint8)
    dark = np.all(out < np.array(background), axis=2)
    out[dark] = np.array(background, dtype=np.uint8)
    return out

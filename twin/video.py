"""A video of one twin run: what the lamp does, what its camera sees, and whether light and touch line up.

    render_video(result, "video.mp4", fps=15)     # result = twin.run.simulate(...)

1280 x 720, H.264 through ffmpeg (found on PATH, or FTM_FFMPEG). Layout:
  left        the room from a free camera: the lamp in its MEASURED pose (lag, sag and the vendor idle
              included), the people, and the shade glowing with the light the real lamp would show on the
              recommended SDK path (show.BestSdkDesign). Top right of it: the 93-pixel panel, ideal design
              vs that SDK path.
  top right   the simulated head camera (61 x 44 deg), with the face detections as the tracker receives
              them (0.15 s after capture, so they trail the picture), the tracker's 3D estimate of the
              locked head (red cross) and where it is aiming (cyan ring).
  bottom      timelines on the run's clock with a moving cursor: tracker state (and when the vendor idle
              plays), the aim error to the locked person from ground truth, the haptic hits as felt on the
              phone and the TITAN board, the panel's colour and brightness for the ideal design, the literal
              SDK path and the recommended SDK path, and every SDK call.

Drawing needs Pillow. Nothing here changes the run; the video is a view of report.json's numbers.
Display: the shade and the panel pictures are drawn at 1 / hardware cap (0.3) so the SDK colour reads; the
colour strips are normalised to the brightest moment of the run. Both are picture choices, not light levels.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from twin import panel as P
from twin.contract import JOINTS

WIDTH, HEIGHT = 1280, 720
ORBIT_SIZE = (768, 432)
HEAD_SIZE = (512, 351)          # 61 x 44 deg: tan(30.5 deg) / tan(22 deg) = 1.458
ORBIT = (-155.0, -16.0, 2.2)    # azimuth, elevation, distance: from the front left, lamp and a standing guest in view
ORBIT_LOOKAT = (0.1, 0.55, 0.45)
TIMELINE_TOP = 432
LABEL_W = 150
PLOT_RIGHT = WIDTH - 10
BG = (14, 15, 20)
GRID = (48, 50, 60)
TEXT = (225, 228, 235)
DIM = (140, 145, 155)
STATE_COLOURS = {"SEARCH": (70, 110, 170), "ACQUIRE": (230, 150, 40), "LOCK": (60, 185, 90),
                 "LOST": (200, 60, 60), "HOLD": (150, 80, 180)}
KIND_COLOURS = {"KICK": (255, 170, 60), "SNARE": (90, 200, 255), "DROP": (255, 60, 60), "BUILD": (250, 230, 80),
                "BASS": (170, 120, 255), "CLICK": (200, 200, 200)}
ERR_MAX_DEG = 45.0


def _ffmpeg() -> str:
    exe = os.environ.get("FTM_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found: put it on PATH or set FTM_FFMPEG")
    return exe


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def _frame_rgb(frames: list, t: float) -> np.ndarray:
    """The LightFrame (93 x 3, linear, cap included) nearest to t."""
    if not frames:
        return np.zeros((P.PIXEL_COUNT, 3))
    t0, t1 = frames[0].t, frames[-1].t
    k = int(round((t - t0) / max(1e-9, (t1 - t0)) * (len(frames) - 1))) if len(frames) > 1 else 0
    return frames[min(max(k, 0), len(frames) - 1)].rgb


def _strip_colours(frames: list, gain: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, display sRGB 0..255 per frame, luminance 0..1 per frame) of the mean panel colour."""
    t = np.array([f.t for f in frames])
    mean = np.array([P.perceived_colour(f.rgb) for f in frames]) * gain
    y = np.clip(P.luminance(mean), 0.0, 1.0)
    return t, (P.linear_to_srgb(np.clip(mean, 0.0, 1.0)) * 255).astype(np.uint8), y


class _Timeline:
    """The static bottom half, drawn once; each frame only adds the cursor."""

    def __init__(self, result, t_end: float):
        from PIL import Image, ImageDraw
        self.t_end = t_end
        self.x0, self.x1 = LABEL_W, PLOT_RIGHT
        self.img = Image.new("RGB", (WIDTH, HEIGHT - TIMELINE_TOP), BG)
        d = ImageDraw.Draw(self.img)
        f, small = _font(13), _font(11)
        s, show = result.steps, result.show
        y = 6
        self.rows = {}

        def row(name: str, h: int, label: str, sub: str = ""):
            nonlocal y
            self.rows[name] = (y, y + h)
            d.text((8, y + max(0, h // 2 - 8)), label, fill=TEXT, font=f)
            if sub:
                d.text((8, y + max(0, h // 2 - 8) + 13), sub, fill=DIM, font=small)
            y += h + 4

        row("state", 18, "tracker state")
        row("idle", 4, "")
        row("err", 62, "aim error (deg)", "ground truth, 0-45")
        row("phone", 20, "felt: phone")
        row("titan", 20, "felt: TITAN")
        row("ideal", 28, "light: ideal", "per-pixel design")
        row("naive", 22, "light: SDK literal")
        row("best", 28, "light: SDK best", "what the lamp shows")
        row("calls", 18, "SDK calls", "motion / light")
        self.bottom = y

        # time grid
        for sec in range(0, int(t_end) + 1):
            x = self.x(sec)
            d.line([(x, 4), (x, self.bottom - 2)], fill=GRID if sec % 5 == 0 else (30, 32, 40))
            if sec % 5 == 0:
                d.text((x + 2, self.bottom), f"{sec} s", fill=DIM, font=small)

        # tracker state and the vendor idle
        t = s["t"]
        y0, y1 = self.rows["state"]
        for k in range(len(t)):
            xa, xb = self.x(t[k]), self.x(t[k] + (t[1] - t[0] if len(t) > 1 else 0.03))
            d.rectangle([xa, y0, max(xa, xb), y1], fill=STATE_COLOURS.get(s["state"][k], (90, 90, 90)))
            if s["idle_playing"][k]:
                i0, i1 = self.rows["idle"]
                d.rectangle([xa, i0, max(xa, xb), i1], fill=(200, 170, 40))

        # aim error: locked person (red) or the nearest person (grey) when nobody is locked
        y0, y1 = self.rows["err"]
        for deg, colour in ((5, (70, 110, 70)), (10, (110, 110, 60)), (22, (110, 60, 60))):
            yy = self._yv(deg, y0, y1)
            d.line([(self.x0, yy), (self.x1, yy)], fill=colour)
            d.text((self.x0 - 26, yy - 7), f"{deg}", fill=colour, font=small)
        for key, colour in (("err_nearest", (110, 110, 120)), ("err_locked", (240, 80, 80))):
            pts = []
            for k in range(len(t)):
                v = s[key][k]
                if math.isfinite(v):
                    pts.append((self.x(t[k]), self._yv(min(v, ERR_MAX_DEG), y0, y1)))
                else:
                    if len(pts) > 1:
                        d.line(pts, fill=colour, width=2 if key == "err_locked" else 1)
                    pts = []
            if len(pts) > 1:
                d.line(pts, fill=colour, width=2 if key == "err_locked" else 1)

        # felt haptics
        for lane, name in (("phone", "phone"), ("titan", "titan")):
            y0, y1 = self.rows[name]
            times, kinds = result.felt.get(lane, (np.zeros(0), []))
            for tf, kind in zip(times, kinds, strict=True):
                if not math.isfinite(tf):
                    continue
                x = self.x(tf)
                top = y0 if kind in ("KICK", "DROP") else y0 + 7
                d.line([(x, top), (x, y1)], fill=KIND_COLOURS.get(kind, TEXT), width=2 if kind == "DROP" else 1)

        # light: colour strip and a luminance line, one gain for all three so they compare
        sets = {"ideal": show.ideal_frames, "naive": show.naive_frames, "best": show.best_frames}
        peak = max((float(np.max(np.array([P.perceived_colour(fr.rgb) for fr in frs]))) for frs in sets.values()
                    if frs), default=0.0)
        gain = 1.0 / peak if peak > 0 else 1.0
        for name, frs in sets.items():
            if not frs:
                continue
            y0, y1 = self.rows[name]
            ft, colours, lum = _strip_colours(frs, gain)
            xs = np.clip(((ft / self.t_end) * (self.x1 - self.x0) + self.x0).astype(int), self.x0, self.x1)
            last_x, line = -1, []
            for i, x in enumerate(xs):
                if x != last_x:
                    d.line([(x, y0), (x, y1)], fill=tuple(int(c) for c in colours[i]))
                    line.append((x, y1 - lum[i] * (y1 - y0)))
                    last_x = x
            if len(line) > 1:
                d.line(line, fill=(255, 255, 255), width=1)
        for a, _, name in result.song.sections:                        # section starts on the ideal strip
            y0, _ = self.rows["ideal"]
            d.text((self.x(a) + 2, y0 - 1), name, fill=(255, 255, 255), font=small)

        # SDK calls: motion (white accepted, red refused, tall for a clip), light (yellow; red if 429)
        y0, y1 = self.rows["calls"]
        for c in result.commands:
            x = self.x(c["t"])
            colour = (235, 235, 235) if c["accepted"] else (240, 60, 60)
            d.line([(x, y0 if c["kind"] == "clip" else y0 + 5), (x, y1)], fill=colour, width=2)
        for c in result.light_calls:
            x = self.x(c.t_send)
            colour = (250, 210, 60) if not c.status.startswith(("429", "refused")) else (240, 60, 60)
            d.line([(x, y0), (x, y0 + 8)], fill=colour, width=2)

    def x(self, t: float) -> int:
        return int(round(self.x0 + (float(t) / self.t_end) * (self.x1 - self.x0)))

    @staticmethod
    def _yv(deg: float, y0: int, y1: int) -> float:
        return y1 - (deg / ERR_MAX_DEG) * (y1 - y0)

    def with_cursor(self, t: float):
        from PIL import ImageDraw
        img = self.img.copy()
        d = ImageDraw.Draw(img)
        x = self.x(t)
        d.line([(x, 2), (x, self.bottom - 2)], fill=(255, 255, 255), width=2)
        return img


def render_video(result, path, *, fps: int = 15, crf: int = 23) -> dict:
    """Draw every frame of `result` (a twin.run.RunResult) and encode it to `path` (MP4)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:                          # the run itself does not need Pillow
        raise RuntimeError("the video needs Pillow (uv run --with pillow ...)") from exc
    kin, world, s, show = result.kin, result.world, result.steps, result.show
    t_steps = s["t"]
    dt = float(t_steps[1] - t_steps[0]) if len(t_steps) > 1 else 1.0 / 30
    t_end = float(t_steps[-1])
    timeline = _Timeline(result, t_end)
    f, big = _font(14), _font(18)
    cap = P.HARDWARE_CAP
    run = result.report["run"]
    title = f"{run['scenario']} | {run['strategy']} | {run['song']} | {run['rate_limit_per_min']} actions/min"

    # detections by delivery time, for the boxes
    dets = sorted(result.detections, key=lambda d: d.t_delivered)
    det_t = np.array([d.t_delivered for d in dets]) if dets else np.zeros(0)
    cmd_t = np.array([c["t"] for c in result.commands]) if result.commands else np.zeros(0)
    hx, hy = HEAD_SIZE

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_frames = int(math.floor(t_end * fps + 1e-9)) + 1
    try:
        for i in range(n_frames):
            t = i / fps
            k = min(int(round(t / dt)), len(t_steps) - 1)
            units = {j: float(v) for j, v in zip(JOINTS, s["measured"][k], strict=True)}
            people = world.people_at(t)
            best_rgb = _frame_rgb(show.best_frames, t)
            ideal_rgb = _frame_rgb(show.ideal_frames, t)
            shade = P.perceived_colour(best_rgb) / cap          # display gain, see the module note
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)

            orbit = kin.render(units, people=people, light_rgb=shade, camera="orbit", width=ORBIT_SIZE[0],
                               height=ORBIT_SIZE[1], orbit=ORBIT, lookat=ORBIT_LOOKAT)
            canvas.paste(Image.fromarray(orbit), (0, 0))
            head = kin.render(units, people=people, light_rgb=shade, camera="head", width=hx, height=hy)
            head_img = Image.fromarray(head)
            hd = ImageDraw.Draw(head_img)
            pose = kin.head(units)
            # detections delivered in the last camera frame period
            if det_t.size:
                lo, hi = np.searchsorted(det_t, t - 0.1 + 1e-9), np.searchsorted(det_t, t + 1e-9, side="right")
                for det in dets[lo:hi]:
                    w = det.size * hx
                    cx, cy = det.x * hx, det.y * hy
                    hd.rectangle([cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2], outline=(255, 230, 60), width=2)
                    hd.text((cx - w / 2, cy - w / 2 - 14), f"{det.confidence:.2f}", fill=(255, 230, 60), font=f)
            for key, colour, cross in (("target", (255, 60, 60), True), ("aim", (60, 220, 255), False)):
                p = s[key][k]
                if not np.all(np.isfinite(p)):
                    continue
                v = p - pose.position
                z = float(v @ pose.forward)
                if z <= 1e-6:
                    continue
                px = (0.5 + kin.fx * float(v @ pose.right) / z) * hx
                py = (0.5 + kin.fy * float(v @ pose.down) / z) * hy
                if cross:
                    hd.line([(px - 12, py), (px + 12, py)], fill=colour, width=2)
                    hd.line([(px, py - 12), (px, py + 12)], fill=colour, width=2)
                else:
                    hd.ellipse([px - 7, py - 7, px + 7, py + 7], outline=colour, width=2)
            hd.text((6, 4), "head camera 61x44 deg", fill=(255, 255, 255), font=f)
            canvas.paste(head_img, (WIDTH - hx, 0))

            d = ImageDraw.Draw(canvas)
            # panel pictures, ideal vs the SDK path
            for n, (rgb, label) in enumerate(((ideal_rgb, "ideal"), (best_rgb, "SDK best"))):
                disc = Image.fromarray(P.draw_panel(rgb, 84))
                x = ORBIT_SIZE[0] - 2 * 92 + n * 92
                canvas.paste(disc, (x, 30))
                d.text((x + 4, 116), label, fill=TEXT, font=f)
            d.text((10, 8), title, fill=TEXT, font=f)
            lx = 10                                                    # state legend, bottom of the room view
            for name, colour in STATE_COLOURS.items():
                d.rectangle([lx, ORBIT_SIZE[1] - 20, lx + 10, ORBIT_SIZE[1] - 10], fill=colour)
                d.text((lx + 14, ORBIT_SIZE[1] - 23), name, fill=TEXT, font=f)
                lx += 24 + 9 * len(name)
            state = s["state"][k]
            section = result.song.section_at(t)
            d.text((10, 28), f"t = {t:5.2f} s   {section}", fill=TEXT, font=big)
            d.rectangle([10, 52, 24, 66], fill=STATE_COLOURS.get(state, (90, 90, 90)))
            locked = f" on {s['track'][k]} (person {s['pid'][k]})" if s["pid"][k] else ""
            d.text((30, 51), f"{state}{locked}", fill=TEXT, font=f)

            # text under the head camera
            x0, y0 = WIDTH - hx + 8, hy + 6
            err = s["err_locked"][k]
            d.text((x0, y0), f"aim error to locked head: {err:.1f} deg" if math.isfinite(err)
                   else "aim error: nobody locked", fill=TEXT, font=f)
            j = int(np.searchsorted(cmd_t, t + 1e-9, side="right")) - 1 if cmd_t.size else -1
            if j >= 0:
                c = result.commands[j]
                ok = "accepted" if c["accepted"] else f"REFUSED {c['reason'][:24]}"
                d.text((x0, y0 + 18), f"last call {c['t']:.1f} s: {c['kind']} ({ok})", fill=TEXT, font=f)
                d.text((x0, y0 + 36), c["why"][:62], fill=DIM, font=f)
            d.text((x0, y0 + 54), f"SDK session: {int(s['session_used'][k])}/{run['rate_limit_per_min']} actions "
                   f"in the last 60 s", fill=DIM, font=f)

            canvas.paste(timeline.with_cursor(t), (0, TIMELINE_TOP))
            proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    return {"frames": n_frames, "fps": fps, "size": [WIDTH, HEIGHT],
            "display_note": "shade and panel pictures at 1/0.3 display gain; colour strips normalised to the "
                            "run's brightest moment (picture choices, not light levels)"}

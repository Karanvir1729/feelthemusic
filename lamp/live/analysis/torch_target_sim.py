#!/usr/bin/env python3
"""Synthetic optics experiment: is a flashing phone torch an easier camera target than a pink screen?

Everything here is synthetic. It renders 640x480 frames of a bright hall (reflective clutter, ceiling
lights, windows, laptop screens, glints), a person holding a phone, and either the phone's torch
(a point source far above sensor saturation, with a lens halo) or its pink screen, then runs:

  a  brightest blob            argmax of the blurred grey frame
  b  small saturated blob      round, small, with a halo that falls off
  c  temporal                  frame near a KNOWN flash time minus the frames before and after
  d  pink screen               PhoneTracker from lamp/live/follow.py (imported, not copied)

It talks to nothing: no lamp, no network, no files outside --out. From the repository root:

  python lamp/live/analysis/torch_target_sim.py --run --out /tmp/torch.json      # ~4 min on 8 cores
  python lamp/live/analysis/torch_target_sim.py --report --out /tmp/torch.json

ASSUMPTIONS (not measured): torch 50 cd on axis at level 1.0, linear in level, half intensity at
30 deg, gone by 85 deg; lens PSF = Gaussian core sigma 0.9 px + 3 % halo ~ (1+(r/3px)^2)^-1.5;
auto-exposure puts the mean encoded output at 110/255 with highlights clipped; MJPEG quality 80; hall 300-700 lux.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)
warnings.simplefilter("ignore", RuntimeWarning)          # nanmedian of an empty error list

W, H = 640, 480
HFOV_DEG = 65.0
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))      # 502 px
OMEGA_PX = 1.0 / F_PX ** 2                                 # sr per pixel
PAN_SPAN = 160
CW = W + PAN_SPAN
I_FULL_CD = 50.0
G_HALO, R0_HALO, SIG_CORE, STAMP_R = 0.03, 3.0, 0.9, 48
T_READOUT = 0.025
JPEG_Q = 80
TOL_PX = 8.0

_yy, _xx = np.mgrid[-STAMP_R:STAMP_R + 1, -STAMP_R:STAMP_R + 1].astype(np.float32)
_HALO_K = (1 + (_xx ** 2 + _yy ** 2) / R0_HALO ** 2) ** -1.5
_HALO_K /= _HALO_K.sum()
_GAMMA_LUT = np.clip(np.round(255.0 * (np.arange(4096) / 4095.0) ** (1 / 2.2)), 0, 255).astype(np.uint8)


def psf_stamp(dx: float, dy: float) -> np.ndarray:
    r2 = (_xx - dx) ** 2 + (_yy - dy) ** 2
    core = np.exp(-r2 / (2 * SIG_CORE ** 2))
    core /= core.sum()
    halo = (1 + r2 / R0_HALO ** 2) ** -1.5
    halo /= halo.sum()
    return ((1 - G_HALO) * core + G_HALO * halo).astype(np.float32)


def beam(theta_deg: float) -> float:
    return math.exp(-math.log(2) * (theta_deg / 30.0) ** 2) * float(np.clip((85 - theta_deg) / 20.0, 0, 1))


def lin(srgb):  # sRGB 0..255 (R,G,B) -> linear BGR
    r, g, b = [(c / 255.0) ** 2.2 for c in srgb]
    return np.array([b, g, r], np.float32)


SKIN = np.array([0.22, 0.30, 0.45], np.float32)            # BGR reflectance


# ------------------------------------------------------------------------------------------- scene
class Scene:
    """One 1.8 s sequence: static canvases in nits, dynamic point sources, camera and timing."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        rng = self.rng = np.random.default_rng(cfg["seed"])
        self.fps = cfg.get("fps", 24)
        self.dur = 1.8
        self.n = int(self.dur * self.fps)
        self.exp = rng.uniform(0.008, 0.020)
        self.phase = rng.uniform(0, 1 / self.fps)
        self.t = self.phase + np.arange(self.n) / self.fps                  # start of readout
        lat = rng.uniform(0.0, 0.11)
        self.stamp = self.t + T_READOUT + lat + rng.uniform(0, 0.02, self.n)
        first = 0.30 + rng.uniform(0, 0.1)
        self.flash_len = cfg.get("flash_len", 0.070)
        self.flashes = [first + 0.5 * i for i in range(3)]
        v = cfg.get("pan", 0)
        if v == 0:
            self.offs = np.full(self.n, int(rng.integers(0, PAN_SPAN + 1)))
        else:
            p = (int(rng.integers(0, 2 * PAN_SPAN)) + v * np.arange(self.n)) % (2 * PAN_SPAN)
            self.offs = np.where(p <= PAN_SPAN, p, 2 * PAN_SPAN - p).astype(int)
        self.v = v
        self.E = rng.uniform(300, 700)
        self.clutter = cfg["clutter"]
        self.d = cfg["d"]
        self.points = []            # dicts: x, y (canvas), q(k) -> full-scale*px
        self.flicker = []           # (x0, y0, x1, y1) canvas rects of video-playing laptops
        self.refl = np.zeros((H, CW, 3), np.float32)
        self.emis = np.zeros((H, CW, 3), np.float32)
        self.person = np.zeros((H, CW), np.uint8)
        self._background()
        self._target()
        self._finish()

    # -- background ---------------------------------------------------------------------------
    def _background(self):
        rng, refl, emis = self.rng, self.refl, self.emis
        tint = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.0)], np.float32)
        lf = cv2.resize(rng.normal(0, 1, (6, 10)).astype(np.float32), (CW, H), interpolation=cv2.INTER_CUBIC)
        refl[:] = np.clip(rng.uniform(0.25, 0.6) + 0.12 * lf, 0.03, 0.9)[..., None] * tint
        horizon = int(H * rng.uniform(0.55, 0.7))
        refl[horizon:] *= rng.uniform(0.35, 0.7)
        for _ in range(6 + 10 * self.clutter):                       # banners, clothes, bags, tables
            x, y = int(rng.integers(0, CW)), int(rng.integers(int(H * 0.25), H))
            w, h = int(rng.integers(15, 120)), int(rng.integers(15, 140))
            sat = int(np.clip(255 * rng.beta(1.6, 3.0), 10, 255))     # most things are not saturated colours
            hsv = np.uint8([[[rng.integers(0, 180), sat, 255]]])
            col = [float(c) for c in (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] / 255.0) ** 2.2 * rng.uniform(0.06, 0.6)]
            shape = rng.random()
            if shape < 0.35:
                cv2.rectangle(refl, (x, y), (x + w, y + h), col, -1)
            elif shape < 0.65:
                cv2.ellipse(refl, (x, y), (w // 2 + 2, h // 2 + 2), float(rng.uniform(0, 180)), 0, 360, col, -1)
            else:
                ang = np.sort(rng.uniform(0, 2 * math.pi, int(rng.integers(5, 8))))
                rad = rng.uniform(0.5, 1.0, len(ang))
                pts = np.stack([x + np.cos(ang) * rad * w / 2, y + np.sin(ang) * rad * h / 2], 1).astype(np.int32)
                cv2.fillPoly(refl, [pts], col)
        if self.clutter >= 2:                                        # an explicit magenta poster
            x, y = int(rng.integers(0, CW - 80)), int(rng.integers(120, 300))
            cv2.rectangle(refl, (x, y), (x + int(rng.integers(40, 90)), y + int(rng.integers(60, 130))),
                          [float(c) for c in lin((230, 40, 140)) * 0.8], -1)
        kinds = ["panel", "laptop", "downlight", "window", "downlight"][: self.cfg["n_distract"]]
        for kind in kinds:
            if kind == "panel":
                x, y = int(rng.integers(0, CW - 150)), int(rng.integers(5, 90))
                cv2.rectangle(emis, (x, y), (x + int(rng.integers(60, 140)), y + int(rng.integers(10, 28))),
                              [float(rng.uniform(4000, 8000))] * 3, -1)
            elif kind == "downlight":
                x, y = int(rng.integers(20, CW - 20)), int(rng.integers(10, 130))
                cv2.circle(emis, (x, y), int(rng.integers(3, 9)), [float(rng.uniform(15000, 40000))] * 3, -1, cv2.LINE_AA)
            elif kind == "window":
                x, y = int(rng.integers(0, CW - 260)), int(rng.integers(30, 120))
                w, h = int(rng.integers(150, 260)), int(rng.integers(80, 160))
                cv2.rectangle(emis, (x, y), (x + w, y + h), [float(rng.uniform(2500, 8000))] * 3, -1)
                for i in range(1, 3):
                    cv2.line(emis, (x + i * w // 3, y), (x + i * w // 3, y + h), (0, 0, 0), 3)
            else:                                                     # laptop screen
                x, y = int(rng.integers(0, CW - 100)), int(rng.integers(horizon - 60, H - 70))
                w, h = int(rng.integers(45, 95)), int(rng.integers(30, 60))
                L = float(rng.uniform(150, 350))
                cv2.rectangle(emis, (x, y), (x + w, y + h), [L, L * 0.97, L * 0.9], -1)
                for _ in range(4):
                    bx, by = x + int(rng.integers(0, w - 10)), y + int(rng.integers(0, h - 8))
                    cv2.rectangle(emis, (bx, by), (bx + 10, by + 6), [L * float(rng.uniform(0.1, 0.6))] * 3, -1)
                if rng.random() < 0.5:
                    self.flicker.append((x, y, x + w, y + h))
        n_glint = [1, 4, 8][self.clutter]
        for _ in range(n_glint):                                     # static specular glints
            q = float(10 ** rng.uniform(0.5, 2.5))
            self.points.append(dict(x=rng.uniform(10, CW - 10), y=rng.uniform(H * 0.3, H - 10), kind="glint", q0=q))
        if self.clutter >= 2:
            for _ in range(2):                                       # the lamp's own panel, mirrored in glass
                self.points.append(dict(x=rng.uniform(10, CW - 10), y=rng.uniform(H * 0.3, H - 10), kind="sync",
                                        q0=float(10 ** rng.uniform(0.7, 1.8))))
            self.points.append(dict(x=rng.uniform(10, CW - 10), y=rng.uniform(H * 0.4, H - 10), kind="led",
                                    q0=float(10 ** rng.uniform(0.3, 1.2)), ph=rng.uniform(0, 0.6)))
        self.walker = None
        if self.clutter >= 1 and rng.random() < 0.7:
            self.walker = dict(x0=rng.uniform(-40, CW), vx=rng.choice([-1, 1]) * rng.uniform(3, 7),
                               w=int(rng.integers(35, 70)), top=int(rng.integers(120, 220)),
                               val=np.array(rng.uniform(0.05, 0.5, 3), np.float32))

    # -- the person and their phone -----------------------------------------------------------------
    def _target(self):
        rng, cfg, s = self.rng, self.cfg, F_PX / self.d
        self.tx = rng.uniform(PAN_SPAN + 50, W - 50)
        self.ty = rng.uniform(H * 0.42, H * 0.72)
        px, py = self.tx, self.ty
        cx, cy = px + rng.uniform(-0.10, 0.10) * s, py + rng.uniform(-0.05, 0.1) * s
        if rng.random() < 0.7:                                       # most clothes are near neutral
            cloth = [float(c) for c in rng.uniform(0.03, 0.6) * rng.uniform(0.9, 1.1, 3)]
        else:
            hsv = np.uint8([[[rng.integers(0, 180), rng.integers(60, 230), 255]]])
            cloth = [float(c) for c in (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] / 255.0) ** 2.2 * rng.uniform(0.08, 0.5)]
        for img, col in ((self.refl, cloth), (self.emis, (0, 0, 0))):
            cv2.rectangle(img, (int(cx - 0.23 * s), int(cy - 0.25 * s)), (int(cx + 0.23 * s), int(cy + 0.6 * s)), col, -1)
        cv2.rectangle(self.person, (int(cx - 0.23 * s), int(cy - 0.25 * s)), (int(cx + 0.23 * s), int(cy + 0.6 * s)), 255, -1)
        head = (int(cx), int(cy - 0.42 * s))
        axes = (max(2, int(0.075 * s)), max(3, int(0.11 * s)))
        for img, col in ((self.refl, [float(c) for c in SKIN]), (self.emis, (0, 0, 0))):
            cv2.ellipse(img, head, axes, 0, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(self.person, head, axes, 0, 0, 360, 255, -1)
        self.face = (head[0], head[1], 2 * axes[0])
        phi = rng.uniform(-20, 20)
        self.truth_size = 0.0
        if cfg["mode"] == "pink":
            self._screen(px, py, s, phi)
        else:
            self._phone_back(px, py, s, phi)
        cv2.circle(self.person, (int(px), int(py)), int(0.12 * s) + 4, 255, -1)

    def _patch(self, patch, mask, px, py, phi):
        """Rotate an upright patch (BGR nits, mask 0..1) and composite it as an emitter."""
        h, w = mask.shape
        R = int(math.hypot(w, h) / 2) + 4
        x0, y0 = int(round(px)) - R, int(round(py)) - R
        M = cv2.getRotationMatrix2D((w / 2, h / 2), phi, 1.0)
        M[0, 2] += R - w / 2 + (px - round(px))
        M[1, 2] += R - h / 2 + (py - round(py))
        P = cv2.warpAffine(patch, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)
        A = cv2.warpAffine(mask, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)[..., None]
        ys, xs = slice(max(0, y0), min(H, y0 + 2 * R)), slice(max(0, x0), min(CW, x0 + 2 * R))
        sy, sx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        self.emis[ys, xs] = self.emis[ys, xs] * (1 - A[sy, sx]) + P[sy, sx] * A[sy, sx]
        self.refl[ys, xs] = self.refl[ys, xs] * (1 - A[sy, sx]) + 0.04 * A[sy, sx]

    def _screen(self, px, py, s, phi):
        rng, cfg = self.rng, self.cfg
        yaw = rng.uniform(0, 35)
        ss = 4                                                      # supersample
        w = max(2, 0.065 * s * math.cos(math.radians(yaw)))
        h = max(3, 0.140 * s)
        Wp, Hp = int(round(w * ss)), int(round(h * ss))
        Lw = rng.uniform(250, 600)
        big = np.zeros((Hp + 8 * ss, Wp + 8 * ss, 3), np.float32)     # bezel margin
        m = np.zeros(big.shape[:2], np.float32)
        o = 4 * ss
        if cfg.get("style", "solid") == "solid":
            big[o:o + Hp, o:o + Wp] = lin((255, 45, 150)) * Lw
        else:
            a, b = lin((110, 70, 230)) * Lw, lin((240, 80, 170)) * Lw
            g = np.linspace(0, 1, Hp, dtype=np.float32)[:, None, None]
            big[o:o + Hp, o:o + Wp] = a * (1 - g) + b * g
            for i in range(5):                                       # white UI text
                yb = o + int(Hp * (0.15 + 0.15 * i))
                big[yb:yb + max(1, int(0.004 * s * ss)), o + int(Wp * 0.15):o + int(Wp * rng.uniform(0.5, 0.85))] = Lw * 0.9
        bez = int(0.003 * s * ss)
        rad = max(1, int(0.008 * s * ss))
        cv2.rectangle(m, (o - bez + rad, o - bez), (o + Wp + bez - rad, o + Hp + bez), 1.0, -1)
        cv2.rectangle(m, (o - bez, o - bez + rad), (o + Wp + bez, o + Hp + bez - rad), 1.0, -1)
        for qx in (o - bez + rad, o + Wp + bez - rad):
            for qy in (o - bez + rad, o + Hp + bez - rad):
                cv2.circle(m, (qx, qy), rad, 1.0, -1)
        self.glare = bool(self.clutter >= 1 and rng.random() < 0.4)
        if self.glare:                                               # a ceiling light mirrored in the glass
            gx, gy = o + int(rng.uniform(0.2, 0.8) * Wp), o + int(rng.uniform(0.15, 0.85) * Hp)
            cv2.ellipse(big, (gx, gy), (max(1, int(0.006 * s * ss)), max(1, int(0.02 * s * ss))), float(rng.uniform(0, 180)),
                        0, 360, [2500.0] * 3, -1)
        self.fingers = bool(rng.random() < 0.7)
        if self.fingers:                                             # fingers over one edge, thumb over the other
            skin = [float(c) for c in SKIN * self.E / math.pi]
            side = o - bez if rng.random() < 0.5 else o + Wp + bez
            for i in range(3):
                fy = o + int(Hp * (0.45 + 0.13 * i))
                cv2.ellipse(big, (side, fy), (int(0.007 * s * ss), int(0.008 * s * ss)), 0, 0, 360, skin, -1)
                cv2.ellipse(m, (side, fy), (int(0.007 * s * ss), int(0.008 * s * ss)), 0, 0, 360, 1.0, -1)
            other = o + Wp + bez if side == o - bez else o - bez
            cv2.ellipse(big, (other, o + int(Hp * 0.6)), (int(0.012 * s * ss), int(0.010 * s * ss)), 0, 0, 360, skin, -1)
            cv2.ellipse(m, (other, o + int(Hp * 0.6)), (int(0.012 * s * ss), int(0.010 * s * ss)), 0, 0, 360, 1.0, -1)
        size = (big.shape[1] // ss, big.shape[0] // ss)
        self._patch(cv2.resize(big, size, interpolation=cv2.INTER_AREA), cv2.resize(m, size, interpolation=cv2.INTER_AREA), px, py, phi)
        self.truth_size = w

    def _phone_back(self, px, py, s, phi):
        rng, cfg = self.rng, self.cfg
        w, h = max(2, int(0.072 * s)), max(3, int(0.148 * s))
        patch = np.zeros((h, w, 3), np.float32)
        m = np.ones((h, w), np.float32)
        self._patch(patch, m, px, py, phi)
        case = rng.uniform(0.03, 0.3, 3).astype(np.float32)
        R = int(math.hypot(w, h) / 2) + 4
        ys, xs = slice(max(0, int(py) - R), int(py) + R), slice(max(0, int(px) - R), int(px) + R)
        sel = (np.abs(self.refl[ys, xs] - 0.04) < 1e-6).all(axis=2)
        self.refl[ys, xs][sel] = case
        c, sn = math.cos(math.radians(-phi)), math.sin(math.radians(-phi))
        ox, oy = -0.018 * s, -0.055 * s
        self.tx, self.ty = px + c * ox - sn * oy, py + sn * ox + c * oy
        theta = cfg.get("theta", 0.0)
        self.torch = dict(x=self.tx, y=self.ty, kind="torch", level=cfg.get("level", 1.0), beam=beam(theta),
                          beacon=cfg.get("beacon", 0.0), visible=cfg.get("visible", True))
        self.points.append(self.torch)
        if cfg.get("second_phone"):
            self.points.append(dict(x=float(np.clip(self.tx + rng.choice([-1, 1]) * rng.uniform(90, 200), 170, W - 10)),
                                    y=self.ty + rng.uniform(-40, 40), kind="torch", level=cfg.get("level", 1.0),
                                    beam=beam(rng.uniform(0, 50)), beacon=0.0, visible=True, d=self.d * rng.uniform(0.8, 1.5)))

    def _finish(self):
        nits = self.refl * (self.E / math.pi)
        view0 = nits[:, self.offs[0]:self.offs[0] + W] + self.emis[:, self.offs[0]:self.offs[0] + W]
        lum = (view0 @ np.array([0.07, 0.72, 0.21], np.float32))[::4, ::4]
        lo, hi = 1.0, 1e5                        # AE: mean encoded output 110/255, highlights clip
        for _ in range(30):
            mid = math.sqrt(lo * hi)
            if float((np.clip(lum / mid, 0, 1) ** (1 / 2.2)).mean()) > 110 / 255:
                lo = mid
            else:
                hi = mid
        self.L_sat = math.sqrt(lo * hi) * self.cfg.get("ae", 1.0)

        def psf(img):
            return (1 - G_HALO) * cv2.GaussianBlur(img, (0, 0), SIG_CORE) + G_HALO * cv2.filter2D(img, -1, _HALO_K, borderType=cv2.BORDER_REFLECT)

        self.refl_psf = psf(nits) / self.L_sat
        self.emis_psf = psf(self.emis) / self.L_sat
        del self.refl, self.emis

    # -- time -----------------------------------------------------------------------------------
    def on_fraction(self, k: int, y: float) -> float:
        t1 = self.t[k] + T_READOUT * y / H
        t0 = t1 - self.exp
        return sum(max(0.0, min(t1, f + self.flash_len) - max(t0, f)) for f in self.flashes) / self.exp

    def pulse(self, t):
        out = 0.0
        for f in self.flashes:
            tau = np.maximum(t - f, 0.0)
            out = out + (1 - np.exp(-tau / 0.015)) * np.exp(-tau / 0.2) * (t > f)
        return out

    def truth(self, k: int):
        return self.tx - self.offs[k], self.ty

    def render(self, k: int) -> np.ndarray:
        rng, off = self.rng, int(self.offs[k])
        rows_t = self.t[k] + T_READOUT * np.arange(H) / H - self.exp / 2
        gain = np.ones(H, np.float32)
        if self.clutter >= 2:
            gain = (1 + 0.10 * self.pulse(rows_t)).astype(np.float32)
        view = self.refl_psf[:, off:off + W] * gain[:, None, None] + self.emis_psf[:, off:off + W]
        for (x0, y0, x1, y1) in self.flicker:
            a, b = max(0, x0 - off), min(W, x1 - off)
            if b > a:
                view[y0:y1, a:b] *= float(np.clip(1 + 0.3 * rng.normal(), 0.3, 1.8))
        if self.walker is not None:
            wk = self.walker
            xw = int(wk["x0"] + wk["vx"] * k) - off
            a, b = max(0, xw), min(W, xw + wk["w"])
            if b > a:
                free = self.person[wk["top"]:, off + a:off + b] == 0
                view[wk["top"]:, a:b][free] = wk["val"] * (self.E / math.pi) / self.L_sat
        cfg = self.cfg
        if cfg["mode"] == "torch" and not self.torch["visible"]:      # torch faces away: only a soft lit patch
            f = self.on_fraction(k, self.ty)
            if f > 0:
                s = F_PX / self.d
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                g = np.exp(-((xx - (self.tx - off)) ** 2 + (yy - (self.ty + 0.5 * s)) ** 2) / (2 * (0.25 * s) ** 2))
                view = view * (1 + 0.5 * self.torch["level"] * f * g)[..., None]
        for p in self.points:
            x, y = p["x"] - off, p["y"]
            if not (-STAMP_R < x < W + STAMP_R):
                continue
            kind = p["kind"]
            if kind == "glint":
                q = p["q0"]
            elif kind == "sync":
                q = p["q0"] * (0.3 + float(self.pulse(np.array(self.t[k] + T_READOUT * y / H))))
            elif kind == "led":
                q = p["q0"] * (((self.t[k] + p["ph"]) % 0.6) < 0.3)
            else:
                if not p["visible"]:
                    continue
                d = p.get("d", self.d)
                full = I_FULL_CD * p["beam"] / (d * d * OMEGA_PX * self.L_sat)
                q = full * (p["level"] * self.on_fraction(k, y) + p["beacon"])
            if q <= 0:
                continue
            ix, iy = int(math.floor(x)), int(math.floor(y))
            st = psf_stamp(x - ix, y - iy) * q
            ax, bx = max(0, ix - STAMP_R), min(W, ix + STAMP_R + 1)
            ay, by = max(0, iy - STAMP_R), min(H, iy + STAMP_R + 1)
            if bx > ax and by > ay:
                view[ay:by, ax:bx] += st[ay - iy + STAMP_R:by - iy + STAMP_R, ax - ix + STAMP_R:bx - ix + STAMP_R, None]
        if self.v:
            nb = int(round(abs(self.v) * self.exp * self.fps))
            if nb >= 2:
                view = cv2.blur(view, (nb, 1))
        noise = np.empty_like(view)
        cv2.randn(noise, 0, 1)
        np.clip(view, 0, 4, out=view)
        view += noise * np.sqrt(0.0003 * view + 1.6e-5)
        q = np.clip(view * 4095, 0, 4095).astype(np.uint16)
        img = _GAMMA_LUT[q]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------------------- detectors
def det_brightest(gray: np.ndarray):
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mx, _, loc = cv2.minMaxLoc(b)
    return (float(loc[0]), float(loc[1])) if mx >= 200 else None


_RING_CACHE: dict = {}


def _rings(R: int):
    if R not in _RING_CACHE:
        n = R + 15
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        r = np.hypot(xx, yy)
        _RING_CACHE[R] = (n, (r >= R + 2) & (r <= R + 5), (r >= R + 9) & (r <= R + 14))
    return _RING_CACHE[R]


def det_shape(gray: np.ndarray):
    """Small, round, saturated, and surrounded by a halo that falls off."""
    mask = (gray >= 245).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not 2 <= area <= 1500 or not 0.6 <= w / h <= 1.7 or area / (w * h) < 0.6:
            continue
        R = int(round(math.sqrt(area / math.pi)))
        m, inner, outer = _rings(R)
        cx, cy = int(round(cents[i][0])), int(round(cents[i][1]))
        if cx - m < 0 or cy - m < 0 or cx + m + 1 > gray.shape[1] or cy + m + 1 > gray.shape[0]:
            continue
        win = gray[cy - m:cy + m + 1, cx - m:cx + m + 1]
        a, b = float(win[inner].mean()), float(win[outer].mean())
        if a >= 240 or a - b < 12:
            continue
        score = (a - b) * math.sqrt(area)
        if score > best_score:
            best, best_score = (float(cents[i][0]), float(cents[i][1])), score
    return best


_K13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_HANN = cv2.createHanningWindow((W // 4, H // 4), cv2.CV_32F)


def small(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32)


def aligned(ref: np.ndarray, small_ref: np.ndarray, small_on: np.ndarray) -> np.ndarray:
    """Shift a reference frame onto the flash frame: the camera sits on a head that moves."""
    (dx, dy), _ = cv2.phaseCorrelate(small_ref, small_on, _HANN)
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return ref
    M = np.float32([[1, 0, 4 * dx], [0, 1, 4 * dy]])
    return cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flash_map(g_on, refs, kernel=None):
    ref = refs[0] if len(refs) == 1 else cv2.max(refs[0], refs[1])
    ref = cv2.dilate(ref, _K13 if kernel is None else kernel)      # anything bright within 6 px cancels
    d = cv2.subtract(g_on, ref).astype(np.float32)                 # saturating: negatives -> 0
    dog = cv2.GaussianBlur(d, (0, 0), 1.5) - cv2.GaussianBlur(d, (0, 0), 6.0)
    _, mx, _, loc = cv2.minMaxLoc(dog)
    return d, mx, loc


_R13 = np.ones((13, 13), np.uint8)


def flash_map_fast(g_on, refs):
    """Same idea in 8-bit: rectangular dilation (separable), wide blur at quarter scale, early exit."""
    ref = refs[0] if len(refs) == 1 else cv2.max(refs[0], refs[1])
    d = cv2.subtract(g_on, cv2.dilate(ref, _R13))
    if cv2.minMaxLoc(d)[1] < 40:
        return d, 0.0, (0, 0)
    a = cv2.GaussianBlur(d, (0, 0), 1.5)
    b = cv2.GaussianBlur(cv2.resize(d, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    dog = cv2.subtract(a, cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR))
    _, mx, _, loc = cv2.minMaxLoc(dog)
    return d, float(mx), loc


def det_temporal(grays, smalls, ks, m, causal=False, comp=True, thr=35.0, kernel=None, max_area=1000, fast=False):
    """Best small flash over candidate frames ks; references are m frames either side."""
    best = None
    for k in ks:
        if k - m < 0 or (not causal and k + m >= len(grays)):
            continue
        refs = [k - m] if causal else [k - m, k + m]
        refs = [aligned(grays[r], smalls[r], smalls[k]) if comp else grays[r] for r in refs]
        d, mx, loc = flash_map_fast(grays[k], refs) if fast else flash_map(grays[k], refs, kernel)
        if mx < thr or (best is not None and mx <= best[0]):
            continue
        x, y = loc
        x0, y0 = max(0, x - 32), max(0, y - 32)
        win = d[y0:y + 33, x0:x + 33].astype(np.float32)
        _, lab = cv2.connectedComponents((win > 0.5 * float(win.max())).astype(np.uint8), connectivity=8)
        blob = lab == lab[y - y0, x - x0] if lab[y - y0, x - x0] else None
        if blob is None:                                            # DoG peak beside the blob: nearest component
            blob = lab > 0
        area = int(blob.sum())
        ys, xs = np.nonzero(blob)
        if area > max_area or (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) > 3.0 * area + 12:
            continue                                                # a lit region or a streak, not a point
        wgt = win[ys, xs]
        best = (mx, k, float(x0 + (wgt * xs).sum() / wgt.sum()), float(y0 + (wgt * ys).sum() / wgt.sum()), area)
    return best


def load_follow(path: str, name: str):
    here = os.path.dirname(path)
    for mod in ("sdk", "spatial"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, here)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(here)
    return mod


FOLLOW = {}


def trackers():
    if not FOLLOW:
        here = os.path.dirname(os.path.abspath(__file__))
        main = os.environ.get("FOLLOW_MAIN") or os.path.join(here, "..", "follow.py")
        FOLLOW["main"] = load_follow(main, "follow_main")
        if os.environ.get("FOLLOW_PR32"):                    # a second follow.py to compare, optional
            FOLLOW["pr32"] = load_follow(os.environ["FOLLOW_PR32"], "follow_pr32")
    return FOLLOW


# ---------------------------------------------------------------------------------------- one run
def classify(res, truth, present, tol):
    if res is None:
        return "none", None
    err = math.hypot(res[0] - truth[0], res[1] - truth[1])
    if present and err <= tol:
        return "ok", err
    return "wrong", err


def run_sequence(cfg: dict) -> dict:
    sc = Scene(cfg)
    frames = [sc.render(k) for k in range(sc.n)]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    smalls = [small(g) for g in grays]
    out: dict = {"cfg": cfg, "res": {}, "cost": {}}

    def add(name, outcome, err):
        r = out["res"].setdefault(name, {"ok": 0, "wrong": 0, "none": 0, "err": []})
        r[outcome] += 1
        if outcome == "ok":
            r["err"].append(err)

    def cost(name, dt):
        out["cost"].setdefault(name, []).append(dt * 1000)

    torch_mode = cfg["mode"] == "torch"
    present = torch_mode and sc.torch["visible"]
    steady = torch_mode and sc.torch["beacon"] > 0
    m = max(1, math.ceil((sc.flash_len + 0.020) * sc.fps))
    captured = 0
    for f in sc.flashes:
        fr = [sc.on_fraction(k, sc.ty) for k in range(sc.n)]
        cand = [k for k in range(sc.n) if f - 0.03 <= sc.t[k] <= f + sc.flash_len + 0.03]
        kbest = max(cand, key=lambda k: fr[k]) if cand else 0
        captured += int(fr[kbest] > 0)
        if torch_mode:
            for name, fn in (("a_brightest", det_brightest), ("b_shape", det_shape)):
                t0 = time.perf_counter()
                res = fn(grays[kbest])
                cost(name, time.perf_counter() - t0)
                o, e = classify(res, sc.truth(kbest), present and (fr[kbest] > 0 or steady), TOL_PX)
                add(name + "@on", o, e)
                koff = max(0, kbest - m)
                res = fn(grays[koff])
                o, e = classify(res, sc.truth(koff), present and steady, TOL_PX)
                add(name + "@off", o, e)
        # temporal, with the stamp uncertainty the lamp really has
        win = [k for k in range(sc.n) if f <= sc.stamp[k] <= f + sc.flash_len + 0.13 + 1 / sc.fps + 0.03]
        t0 = time.perf_counter()
        best = det_temporal(grays, smalls, win, m)
        cost("c_window", time.perf_counter() - t0)
        res = None if best is None else (best[2], best[3])
        kk = kbest if best is None else best[1]
        o, e = classify(res, sc.truth(kk), present, TOL_PX)
        add("c_window", o, e)
        if best is not None:
            out.setdefault("cdbg", []).append((o, round(float(best[0]), 1), int(best[4])))
        t0 = time.perf_counter()
        best = det_temporal(grays, smalls, win, m, fast=True)
        cost("c_fast", time.perf_counter() - t0)
        out.setdefault("ncand", []).append(len(win))
        res = None if best is None else (best[2], best[3])
        kk = kbest if best is None else best[1]
        o, e = classify(res, sc.truth(kk), present, TOL_PX)
        add("c_fast", o, e)
        best = det_temporal(grays, smalls, win, m, kernel=_K5)
        res = None if best is None else (best[2], best[3])
        kk = kbest if best is None else best[1]
        o, e = classify(res, sc.truth(kk), present, TOL_PX)
        add("c_dilate5", o, e)
        best = det_temporal(grays, smalls, win, m, causal=True)
        res = None if best is None else (best[2], best[3])
        kk = kbest if best is None else best[1]
        o, e = classify(res, sc.truth(kk), present, TOL_PX)
        add("c_causal", o, e)
        best = det_temporal(grays, smalls, win, m, comp=False)
        res = None if best is None else (best[2], best[3])
        kk = kbest if best is None else best[1]
        o, e = classify(res, sc.truth(kk), present, TOL_PX)
        add("c_nocomp", o, e)
        knear = int(np.argmin(np.abs(sc.stamp - (f + sc.flash_len / 2))))
        t0 = time.perf_counter()
        best = det_temporal(grays, smalls, [knear], m)
        cost("c_naive", time.perf_counter() - t0)
        res = None if best is None else (best[2], best[3])
        o, e = classify(res, sc.truth(knear), present, TOL_PX)
        add("c_naive", o, e)
    out["captured"] = captured
    if cfg["mode"] == "pink":
        out["fingers"], out["glare"] = sc.fingers, sc.glare
    # pink detectors on every frame in order (TargetLock keeps state), scored at 6 instants
    score_at = set()
    for f in sc.flashes:
        k = int(np.argmin(np.abs(sc.t - f)))
        score_at.update((k, max(0, k - m)))
    for name, mod in trackers().items():
        clock = [0.0]
        tr = mod.PhoneTracker(clock=lambda: clock[0])
        for k in range(sc.n):
            clock[0] = float(sc.stamp[k])
            t0 = time.perf_counter()
            r = tr.locate(frames[k])
            cost("d_pink_" + name, time.perf_counter() - t0)
            if k in score_at:
                res = None if r is None else (r[0] * W, r[1] * H)
                tol = max(TOL_PX, 0.5 * sc.truth_size)
                o, e = classify(res, sc.truth(k), cfg["mode"] == "pink", tol)
                add("d_pink_" + name, o, e)
    if cfg.get("save"):
        k = max(range(sc.n), key=lambda k: sc.on_fraction(k, sc.ty)) if torch_mode else sc.n // 2
        cv2.imwrite(cfg["save"], frames[k], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        out["saved"] = dict(path=cfg["save"], truth=sc.truth(k), frame=k, L_sat=sc.L_sat, exp_ms=sc.exp * 1000)
    out["cost"] = {k: float(np.median(v)) for k, v in out["cost"].items()}
    return out


def summarise(results, keys):
    table: dict = {}
    for r in results:
        key = tuple(r["cfg"].get(k) for k in keys)
        for name, v in r["res"].items():
            t = table.setdefault(key, {}).setdefault(name, {"ok": 0, "wrong": 0, "none": 0, "err": []})
            for f in ("ok", "wrong", "none"):
                t[f] += v[f]
            t["err"] += v["err"]
    return table


def fmt(t):
    n = t["ok"] + t["wrong"] + t["none"]
    e = np.array(t["err"]) if t["err"] else np.array([np.nan])
    return f"det {100 * t['ok'] / n:5.1f}%  false {100 * t['wrong'] / n:5.1f}%  none {100 * t['none'] / n:5.1f}%  err med {np.nanmedian(e):4.1f} p95 {np.nanpercentile(e, 95):4.1f} px  n={n}"


def make_plan() -> list[dict]:
    """The 1952 sequences behind the numbers in the commit message (seeds are fixed)."""
    import itertools
    plan: list[dict] = []
    sid = [1000]

    def add(n, **kw):
        for i in range(n):
            sid[0] += 1
            c = dict(seed=sid[0], pan=[0, 0, 4, 12][i % 4], ae=[0.6, 1.0, 1.6][i % 3])
            c.update(kw)
            plan.append(c)

    nd = {0: 1, 1: 3, 2: 5}
    conds = {"A_full_aimed": dict(level=1.0, theta=10.0), "B_0.3_aimed": dict(level=0.3, theta=10.0),
             "C_0.3_tilt45": dict(level=0.3, theta=45.0), "D_0.3_edge70": dict(level=0.3, theta=70.0),
             "E_away": dict(level=1.0, theta=0.0, visible=False)}
    for d, cl in itertools.product([1.0, 2.0, 3.0, 4.0], [0, 1, 2]):
        for name, c in conds.items():
            add(16, exp="E1", mode="torch", d=d, clutter=cl, n_distract=nd[cl], cond=name, **c)
        for style in ("solid", "app"):
            add(16, exp="E2", mode="pink", d=d, clutter=cl, n_distract=nd[cl], style=style, cond="pink_" + style)
    for d in (2.0, 4.0):
        for fl in (0.07, 0.12, 0.22):
            add(24, exp="E3", mode="torch", d=d, clutter=1, n_distract=3, cond="C_0.3_tilt45", level=0.3, theta=45.0,
                fps=10, flash_len=fl)
    add(24, exp="E3", mode="torch", d=2.0, clutter=1, n_distract=3, cond="C_0.3_tilt45", level=0.3, theta=45.0, fps=24,
        flash_len=0.22)
    for pan in (0, 4, 8, 12, 20):
        for _ in range(24):
            sid[0] += 1
            plan.append(dict(seed=sid[0], exp="E4", mode="torch", d=2.0, clutter=2, n_distract=5, cond="C_0.3_tilt45",
                             level=0.3, theta=45.0, pan=pan, ae=1.0))
    for d, cl in itertools.product([1.0, 2.0, 3.0, 4.0], [0, 1, 2]):
        add(12, exp="E5", mode="torch", d=d, clutter=cl, n_distract=nd[cl], cond="beacon0.1_only", level=0.0, theta=45.0,
            beacon=0.1)
        add(12, exp="E5", mode="torch", d=d, clutter=cl, n_distract=nd[cl], cond="beacon0.1+flash0.3", level=0.3,
            theta=45.0, beacon=0.1)
    add(32, exp="E6", mode="torch", d=2.0, clutter=1, n_distract=3, cond="two_phones", level=0.3, theta=45.0,
        second_phone=True)
    return plan


def report(results: list[dict]) -> None:
    def show(title, rows, names):
        print(f"\n## {title}  ({len(rows)} seq)")
        for name, t in summarise(rows, ()).get((), {}).items():
            if name in names:
                print(f"  {name:16s} {fmt(t)}")

    def sel(**kw):
        return [r for r in results if all(r["cfg"].get(k) == v for k, v in kw.items())]

    temporal = ["a_brightest@on", "b_shape@on", "b_shape@off", "c_window", "c_fast", "c_causal", "c_nocomp", "c_naive"]
    pink = ["d_pink_main", "d_pink_pr32"]
    vis = [r for r in sel(exp="E1") if r["cfg"]["cond"] != "E_away"]
    show("E1 torch visible, all", vis, temporal)
    for d in (1.0, 2.0, 3.0, 4.0):
        show(f"E1 torch visible d={d}", [r for r in vis if r["cfg"]["d"] == d], temporal)
    for cl in (0, 1, 2):
        show(f"E1 torch visible clutter={cl}", [r for r in vis if r["cfg"]["clutter"] == cl], temporal)
    for cond in sorted({r["cfg"]["cond"] for r in sel(exp="E1")}):
        show(f"E1 cond={cond}", sel(exp="E1", cond=cond), temporal)
    for pan in (0, 4, 12):
        show(f"E1 torch visible pan={pan}", [r for r in vis if r["cfg"]["pan"] == pan], temporal)
    for st in ("solid", "app"):
        for d in (1.0, 2.0, 3.0, 4.0):
            show(f"E2 pink style={st} d={d}", sel(exp="E2", style=st, d=d), pink)
    for cl in (0, 1, 2):
        show(f"E2 pink clutter={cl}", sel(exp="E2", clutter=cl), pink)
    show("pink detectors on torch scenes (no screen present)", sel(exp="E1"), pink)
    show("temporal detectors on pink scenes (no torch present)", sel(exp="E2"), temporal)
    for fps in (10, 24):
        for fl in (0.07, 0.12, 0.22):
            rows = sel(exp="E3", fps=fps, flash_len=fl)
            if rows:
                cap = sum(r["captured"] for r in rows) / (3 * len(rows))
                show(f"E3 fps={fps} flash={fl * 1000:.0f} ms, captured by a frame {100 * cap:.0f}%", rows, temporal)
    for pan in (0, 4, 8, 12, 20):
        show(f"E4 pan={pan} px/frame", sel(exp="E4", pan=pan), temporal)
    for cond in ("beacon0.1_only", "beacon0.1+flash0.3"):
        show(f"E5 {cond}", sel(exp="E5", cond=cond), temporal)
    show("E6 two phones flashing in sync", sel(exp="E6"), temporal)
    cost: dict = {}
    for r in results:
        for k, v in r["cost"].items():
            cost.setdefault(k, []).append(v)
    print("\n## cost, ms per call (median of per-sequence medians, all workers busy)")
    for k, v in cost.items():
        print(f"  {k:14s} {np.median(v):6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results JSON (written by --run, read by --report)")
    ap.add_argument("--run", action="store_true", help="render and evaluate the built-in plan")
    ap.add_argument("--report", action="store_true", help="print the tables from --out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sequences (smoke test)")
    args = ap.parse_args()
    if args.run:
        plan = make_plan()[: args.limit or None]
        t0 = time.time()
        with Pool(args.jobs) as pool:
            results = pool.map(run_sequence, plan, chunksize=4)
        json.dump(results, open(args.out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"{len(results)} sequences in {time.time() - t0:.0f} s")
    if args.report:
        report(json.load(open(args.out)))

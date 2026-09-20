#!/usr/bin/env python3
"""The Cha Cha Slide as twenty calls the lamp can dance -- and a DEAF follower can follow.

The lamp is the instructor, not the decoration. Someone who cannot hear "to the left" has only the
lamp to go on, so the move itself has to carry the call: one move, one unmistakable shape, big enough
to read from the far side of the room. That requirement, and not taste, decides every number here:

  * Each call is ONE clip and the clip is exactly as long as the call, so a cue sheet can put
    `cha_look_left` on the bar where the caller says it and nothing drifts.
  * Every clip starts and ends exactly at bc.START, so any call may follow any other with no snap --
    the same contract the dance clips keep (Validator rejects a clip whose ends are not START).
  * Where two calls could look alike the difference is EXAGGERATED, because a follower who has to
    decide between "look left" and "slide left" in a quarter of a second cannot afford subtlety.
    look/slide differ in profile (snap-and-hold vs. constant-speed glide, banked vs. levelled head),
    hop/reverse in the direction the head travels, knees/how-low in depth and in tempo.
  * Where a move has to be READ rather than admired it ends in a HOLD. A held pose is also the only
    pose the servos deliver in full: a beat-rate step lands at roughly half the commanded excursion
    (the gain table in beat_clips, GAINS, is 1.8 on base_yaw for exactly this reason), but give the
    joint a quarter of a second of stillness and it arrives. Holds are legibility AND amplitude.

Values are COMMANDED units against this lamp's own servo calibration, like bc.START/BOTTOM/TOP and
like wave_clip.py; no gain is applied. The timeline is plain beats -> seconds, and every clip goes
through bc.clamp_envelope (hard envelope) and bc.Validator (joint box, table, the lamp's own base,
peak speed and BOTH ZMP tipping bounds) before it is written. A move that fails is not written: see
the --report table, which is what the operator reads to judge whether a move is big enough to follow.

    python3 cha_moves.py --out /tmp/cha --report      # build all 20, print the table
    python3 cha_moves.py --out /tmp/cha --only cha_turn
    python3 cha_moves.py --list                       # the clip names, in cue order

Install them the way install_clips.py installs the dance clips (tmp -> fsync -> os.replace into the
vendor pack dir), then play one with move.sh.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_clips as bc  # noqa: E402
from spatial import JOINTS, LampModel  # noqa: E402

FPS = bc.FPS

# ----------------------------------------------------------------------------- tempo
# The Cha Cha Slide runs at about 124 bpm. EVERY duration below is written in BEATS and multiplied by
# BEAT here, so changing this one constant retimes the whole vocabulary -- including the sleep table
# in move.sh, which derives its seconds the same way from the same number.
TEMPO_BPM = 124.0
BEAT = 60.0 / TEMPO_BPM                 # 0.4839 s; a four-beat call is 1.935 s
GLIDE_RAMP = 0.22                       # of a glide spent getting up to speed (and the same slowing
                                        # down). A glide's peak speed is 1/(1 - 0.22) = 1.28x its
                                        # average, against 1.57x for a cosine ease, so the smooth
                                        # moves are also the cheap ones against bc.SPEED_LIMIT.

# ----------------------------------------------------------------------------- poses
# Measured with spatial.LampModel on this calibration (head positions in metres, base frame: +y is
# FORWARD, toward the audience; +z is up; +x is the lamp's right, which is +base_yaw). bc.START puts
# the head at (0.014, 0.083, 0.324). The two axes worth knowing before reading any pose below:
#   base_pitch up (toward +10) swings the head FORWARD and DOWN   -- -49 -> -10 is +0.10 m of y
#   elbow up    (toward +94) lifts the head and pulls it BACK     -- -22 -> +20 is +0.07 m of z
# so a move that travels forward without dropping, or rises without retreating, needs BOTH.
#
# wrist_pitch does not move the head at all (it is the last joint in the chain): it aims the shade.
# +wp is nose-DOWN (the shade's forward vector goes from +0.67 z at wp -60 to -0.22 z at wp +55).
# It is free travel -- no reach, no tipping moment worth the name -- and it is the joint that reads
# best at a distance, so nearly every move carries a wrist_pitch that agrees with what the arm does.
# ----------------------------------------------------------------------------- the yaw centre
# Measured 2026-09-20: base_yaw stops dead at about -4.5 units under power (servo EEPROM limit or a
# mechanical stop, unresolved) while the right side is free to at least +73. So the choreography is
# built around CENTRE: every clip starts and ends with the head turned +35, and "left" and "right" are
# 35 units either side of THAT (0 and +70). Turn the lamp on the table about 26 deg to the left (seen
# from above, counter-clockwise) and the resting head faces the audience again. The dance library keeps
# its own home at yaw 0; cha_run.py walks the arm between the two.
CENTRE = 35.0
YAW_REACH = 35.0       # the most a look, a slide, a sweep or a flick may turn from CENTRE

# ----------------------------------------------------------------------------- the runtime's rules
# The vendor runtime plays a clip whole ONLY when its first three frames are stationary at the pose the
# arm is actually in (fusion.py's skip rule: frame0..2 velocity < 5 units/s and every joint within 2.0
# units); otherwise it drops the first ~1.5 s of motion. So every call begins with LEAD_BEATS of
# stillness at home. The runtime also stretches time wherever a joint moves faster than 297 units/s
# after its 5-frame moving average (limit_motion_velocity), which would push every later call off the
# bar -- so the ceiling here is 290, checked on the smoothed clip, and it is a FAIL, not a warning.
LEAD_BEATS = 0.2       # 0.098 s: frames 0, 1, 2 at home (the third at 0.067 s); frame 3 moves
CHA_SPEED_LIMIT = 290.0

LOOK_YAW = YAW_REACH   # a look: 35 units is 26 deg on this calibration, 52 deg between left and right
TURN_YAW = YAW_REACH   # the sweep's ends: 0 and +70 on the servo, 26 deg either side of centre
CROSS_YAW = YAW_REACH  # the criss-cross flicks: 70 units across in one beat is 144 units/s average
SLIDE_YAW = YAW_REACH
STOMP_YAW = 16.0       # "a little +yaw": enough to say which foot, not enough to read as a turn
CB_YAW = YAW_REACH     # charlie brown's left/right pulses: half a beat each, 251 units/s peak at 35

LEAN = [0.0, -10.0, 22.0, 0.0, 18.0]      # head (0.008, 0.179, 0.322): +0.096 m FORWARD, dead level.
                                          # The lean is pure travel toward the audience because the
                                          # elbow's +44 exactly pays back the height base_pitch spends.
BACK = [0.0, -46.0, 18.0, 0.0, -12.0]     # head (0.017, 0.036, 0.399): -0.047 m back, +0.075 m up,
                                          # nose tipped UP. Backward travel is scarce (head_y may not
                                          # go under 0.020 and START is only at 0.083), so the retreat
                                          # buys its legibility by rearing UP as well as back -- the
                                          # opposite corner of the room from LEAN, not a smaller LEAN.
BACK_HALF = [0.0, -47.5, -2.0, 0.0, 0.0]  # the first of the two steps back
HOP_TOP = [0.0, -35.0, 20.0, 0.0, 8.0]    # head (0.014, 0.080, 0.386): +0.062 m straight UP, y within
                                          # 3 mm of home. Nose comes UP on the leap.
HOP_LAND = [0.0, -45.0, -30.0, 0.0, 45.0]  # the squash: 0.034 m below home with the nose down. A hop
                                          # that stops dead at home reads as a twitch; the landing
                                          # compression is what makes the eye see weight.
REV_TOP = [0.0, -46.0, 14.0, 0.0, -15.0]  # head (0.017, 0.041, 0.393): up AND 0.042 m BACK, nose up
                                          # 33 deg. It was (bp -47, el 20), 0.059 m back and 0.011 m
                                          # higher, until the validator priced it: stopping that much
                                          # mass that far back drove zmp_y to -0.041 against the
                                          # -0.030 bound (the dip is the DECELERATION at the top of
                                          # the leap -- measured at frame 6 of 30, right at the apex).
                                          # Backward travel is the one thing this move may not trade
                                          # away, so the slowdown below pays for it instead.
REV_LAND = [0.0, -52.0, -24.0, 0.0, 8.0]  # lands BEHIND home (y 0.074 against home's 0.083). -52 is
                                          # the base_pitch floor while the elbow is above -45; going
                                          # further back here would be clamped, not danced.
STOMP = [0.0, -40.0, -58.0, 0.0, 6.0]     # head (0.012, 0.120, 0.211): -0.113 m DOWN in 0.4 beat.
                                          # The elbow does the dropping (it is the joint that folds
                                          # the arm down) and base_pitch only keeps the head from
                                          # falling back into the lamp on the way.
KNEE = [0.0, -20.0, -50.0, 0.0, 12.0]     # head (0.009, 0.159, 0.174): -0.149 m down, deeper than the
                                          # stomp and four times slower -- the pair must not be
                                          # confusable, so they differ in both depth and tempo.
ROCK = [0.0, -28.0, 2.0, 0.0, 24.0]       # the cha-cha-cha rock: about 0.055 m forward, level
REACH = [0.0, -22.0, -2.0, 0.0, 20.0]     # head (0.010, 0.158, 0.303): the arm out over the table at
                                          # home HEIGHT. Sideways travel is r*sin(yaw) and r is the
                                          # head's distance from the yaw axis: 0.084 m at START and
                                          # 0.158 m here, so reaching out DOUBLES what a given turn is
                                          # worth to the eye. This is why the slides reach and the
                                          # snap-and-hold looks do not (see cha_look_right).
TURN_REACH = [0.0, -30.0, -12.0, 0.0, 24.0]   # a smaller reach for the 8-beat sweep: r 0.146 m, and
                                          # head_y still 0.097 at 65 units of yaw
CROSS_REACH = [0.0, -35.0, -14.0, 0.0, 25.0]  # r 0.128 m; the flicks are fast, so the lean is modest
CLAP_DOWN = [0.0, -49.0, -22.0, 0.0, 55.0]    # the clap: the shade snaps nose-DOWN, 0.74 deg a unit,
CLAP_UP = [0.0, -49.0, -22.0, 0.0, 5.0]       # so the nod is 37 deg of visible tip, twice a beat
CLAP_STROKE = 0.5                             # beats per stroke. The clap is the move that reaches
                                              # bc.SPEED_LIMIT first: 50 units in half a beat is
                                              # 322 units/s of the 380 allowed, so this vocabulary
                                              # retimes safely to about 146 bpm and past that the nod
                                              # has to shrink. A dancer's hands pause between claps
                                              # and an earlier draft held 0.10 beat at the top for it,
                                              # but 0.10 beat is 1.5 frames at 30 fps -- a pause
                                              # nobody can see -- and it cost 9 % of the speed budget,
                                              # which is amplitude, which IS visible. So: no pause.


def _at(pose, **joints) -> list[float]:
    """A copy of `pose` with named joints replaced: _at(LEAN, base_yaw=-40)."""
    out = list(np.asarray(pose, dtype=float))
    for name, value in joints.items():
        out[JOINTS.index(name)] = float(value)
    return out


def _yawed(pose, yaw: float, level: bool = False) -> list[float]:
    """`pose` turned by `yaw`. With level=True the wrist rolls -0.4 units per unit of yaw, which is
    the counter-tilt the dance library uses to keep the shade level through a swing instead of letting
    it bank over -- the difference a follower reads as "gliding sideways" rather than "turning"."""
    return _at(pose, base_yaw=yaw, wrist_roll=(-0.4 * yaw if level else 0.0))


def _toward(pose, fraction: float) -> list[float]:
    """Part of the way from START to `pose`, for the quick quotes of a big move (charlie brown)."""
    return list(bc.START + float(fraction) * (np.asarray(pose, dtype=float) - bc.START))


# ----------------------------------------------------------------------------- the timeline
def _glide(u: np.ndarray, ramp: float = GLIDE_RAMP) -> np.ndarray:
    """Position profile on 0..1 whose SPEED ramps up over the first `ramp` of the segment, holds flat
    through the middle and ramps down over the last `ramp`. A cosine ease is always accelerating or
    decelerating, which the eye reads as a gesture; constant speed is what it reads as a slide, and as
    the inexorable descent of "how low can you go"."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    r = min(0.49, max(1e-6, float(ramp)))
    edge = lambda x: r * (0.5 * x - np.sin(np.pi * x) / (2 * np.pi))   # noqa: E731  integral of ease
    p = np.where(u < r, edge(np.clip(u, 0.0, r) / r),
                 np.where(u <= 1.0 - r, 0.5 * r + (u - r),
                          (1.0 - r) - edge(np.clip(1.0 - u, 0.0, r) / r)))
    return p / (1.0 - r)


PROFILES = {"ease": bc.ease, "glide": _glide, "hold": lambda u: np.zeros_like(np.asarray(u, float))}
START = np.array(_at(bc.START, base_yaw=CENTRE), dtype=float)   # where every cha clip starts and ends


class Call:
    """One call of the song, as keyframes in BEATS from the top of the call.

    The length is declared up front and the keyframes must add up to it exactly (rows() refuses
    otherwise): a call that runs half a beat long would push every later call off the bar, and a cue
    sheet cannot see that by looking. The frames are then sampled on the 30 fps grid the runtime plays
    at, with the beat timeline stretched by at most half a frame so the LAST frame lands exactly on
    the final keyframe -- which is START, always.
    """

    def __init__(self, beats: float, *, says: str = "") -> None:
        self.beats = float(beats)
        self.says = says
        self.keys: list[tuple[float, np.ndarray, str]] = [(0.0, np.array(bc.START, dtype=float), "ease")]
        self.bite = 0.0                  # how far clamp_envelope had to move the trajectory

    @property
    def here(self) -> np.ndarray:
        return self.keys[-1][1]

    def to(self, pose, beats: float, style: str = "ease") -> "Call":
        if style not in PROFILES:
            raise ValueError(f"unknown segment style {style!r}")
        self.keys.append((self.keys[-1][0] + float(beats), np.asarray(pose, dtype=float), style))
        return self

    def glide(self, pose, beats: float) -> "Call":
        return self.to(pose, beats, "glide")

    def hold(self, beats: float) -> "Call":
        return self.to(self.here, beats, "hold")

    def home(self, beats: float, style: str = "ease") -> "Call":
        return self.to(bc.START, beats, style)

    def rows(self) -> np.ndarray:
        end = self.keys[-1][0]
        if abs(end - self.beats) > 1e-9:
            raise ValueError(f"keyframes fill {end:g} beats of a {self.beats:g}-beat call")
        if not np.allclose(self.keys[-1][1], bc.START, atol=1e-9):
            raise ValueError("a call must end at bc.START so the next call can follow it with no snap")
        # The lead-in (LEAD_BEATS at home, see the runtime's rules above) is paid for by the call's own
        # closing stillness when it has one, so the move itself keeps its designed speed; a call that
        # ends in motion is instead played LEAD_BEATS/beats faster, and the speed check below sees that.
        keys = self.keys
        t_last0, t_last1, style_last = keys[-2][0], keys[-1][0], keys[-1][2]
        if style_last == "hold" and t_last1 - t_last0 >= LEAD_BEATS - 1e-9:
            keys = ([keys[0], (LEAD_BEATS, keys[0][1], "hold")]
                    + [(t + LEAD_BEATS, pose, style) for t, pose, style in keys[1:-1]]
                    + [keys[-1]])
        else:
            scale = (self.beats - LEAD_BEATS) / self.beats
            keys = ([keys[0], (LEAD_BEATS, keys[0][1], "hold")]
                    + [(LEAD_BEATS + t * scale, pose, style) for t, pose, style in keys[1:]])
        n = int(round(self.beats * BEAT * FPS)) + 1
        t = np.linspace(0.0, self.beats, n)
        U = np.tile(bc.START.astype(float), (n, 1))
        for (t0, a, _), (t1, b, style) in zip(keys, keys[1:]):
            span = max(t1 - t0, 1e-12)
            where = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
            U[where] = a + (b - a) * PROFILES[style]((t[where] - t0) / span)[:, None]
        U[:, JOINTS.index("base_yaw")] += CENTRE      # designed about 0, danced about CENTRE
        V = bc.clamp_envelope(U)
        self.bite = float(np.abs(V - U).max())      # > 0 means a pose above was written unsafely and
        V[0] = START                                # the envelope had to argue with it: a design bug,
        V[-1] = START                               # reported by --report, not something to ignore
        return V


# ----------------------------------------------------------------------------- the twenty calls
def cha_look_right() -> Call:
    """"to the right" -- a human turns their whole body right and stays there for the call.
    The lamp turns its head +55 units (41 deg) and HOLDS it there for 2.5 of the 4 beats. The hold is
    the move: a follower has to see where it is looking, and a swing that comes straight back reads as
    a wobble. It is also the excursion the servo actually delivers, because it has time to arrive.
    It turns from home, without the reach the slides use, and that is the point of the pair: a look
    pivots on the spot (head 0.051 m, the lit end of the shade 0.128 m) while a slide travels."""
    return (Call(4, says="to the right")
            .to(_at(bc.START, base_yaw=LOOK_YAW), 0.75)
            .hold(2.5)
            .home(0.75))


def cha_look_left() -> Call:
    """"to the left" -- the mirror of cha_look_right, yaw -55. Kept as its own clip rather than a
    flipped copy at play time so the cue sheet names the direction it wants."""
    return (Call(4, says="to the left")
            .to(_at(bc.START, base_yaw=-LOOK_YAW), 0.75)
            .hold(2.5)
            .home(0.75))


def cha_clap() -> Call:
    """"everybody clap your hands" -- a human claps on each of the 8 beats.
    The lamp has no second hand to clap against, so the clap becomes the nod that a clap makes you do
    anyway: the shade snaps nose-down to +55 and back up to +5, eight times, base_yaw dead still so
    nothing competes with it. 50 units is 37 deg of tip, which swings the lit end of the shade through
    0.046 m twice a beat. That is as big as a clap can be at this tempo, not a round number: half a
    beat buys 58 units at bc.SPEED_LIMIT and this nod spends 50 of them.
    The stroke STARTS on the beat and lands half a beat later. Landing exactly ON the beat is not on
    offer: the runtime's own start latency is 250-450 ms (measured, see wave.sh), which is most of a
    beat at 124 bpm, so the clip is rhythmic rather than phase-locked and the cue sheet places it."""
    call = Call(8, says="everybody clap your hands")
    for nod in range(8):
        call.to(CLAP_DOWN, CLAP_STROKE)                              # down: the clap
        call.to(CLAP_UP if nod < 7 else bc.START, CLAP_STROKE)       # up, and the last one comes home
    return call


def cha_lean_forward() -> Call:
    """"cha cha now y'all" / step forward -- a human steps toward you and steps back.
    The lamp cannot step, so the HEAD does the travelling: base_pitch swings the arm out over the
    table and the elbow pays the height back, which carries the head 0.096 m at the audience without
    dropping it. A tilt alone would be invisible edge-on; 10 cm of level travel is not."""
    return (Call(4, says="step forward")
            .to(LEAN, 1.25)
            .hold(1.0)
            .home(1.25)
            .hold(0.5))


def cha_take_it_back() -> Call:
    """"take it back now y'all" -- a human walks backward, several steps.
    The lamp rears back and UP in TWO visible steps (there is only 0.06 m of backward travel before
    head_y hits its 0.020 floor, so the retreat borrows the vertical to make its point) and tips its
    nose up on the way. Against cha_lean_forward it differs in direction, in height, in nose angle and
    in rhythm -- four cues, because forward and back are the pair a follower most needs to get right."""
    return (Call(4, says="take it back now y'all")
            .to(BACK_HALF, 0.75).hold(0.25)
            .to(BACK, 0.75).hold(0.25)
            .home(1.5)
            .hold(0.5))


def _hop(call: Call, top, land, rise: float = 0.55, fall: float = 0.6, recover: float = 0.4) -> Call:
    """One hop into a 2-beat call: up fast, down past home into a squash, recover, then stand still.
    Whatever is left of the two beats is the stillness -- and the stillness is what makes the eye call
    the hop sharp, so the default leaves 0.75 beat of it."""
    still = 2.0 - (rise + fall + recover)
    if still <= 0:
        raise ValueError(f"a hop of {rise + fall + recover:g} beats leaves no stillness in a 2-beat call")
    return call.to(top, rise).to(land, fall).home(recover).hold(still)


def cha_hop() -> Call:
    """"one hop this time" -- a human jumps once, straight up, and lands.
    The lamp throws the whole arm up 0.062 m in 0.4 beat (0.19 s), drops through home into a 0.034 m
    squash and settles: rise, land, absorb. Vertical only -- the head's y moves under 3 mm -- so that
    cha_reverse's backward hop cannot be mistaken for it."""
    return _hop(Call(2, says="one hop this time"), HOP_TOP, HOP_LAND)


def cha_hop_two() -> Call:
    """"two hops this time" -- two of the above, back to back, one per 2 beats. Built from the same
    _hop() so the second is identical to the first: a follower counting hops must not have to wonder
    whether a difference in the shape meant something."""
    call = Call(4, says="two hops this time")
    return _hop(_hop(call, HOP_TOP, HOP_LAND), HOP_TOP, HOP_LAND)


def cha_stomp_right() -> Call:
    """"right foot let's stomp" -- a human drives one foot into the floor and lifts it again.
    The lamp jabs the head 0.113 m DOWN in 0.55 beat and back up in 0.55, then stands dead still for
    the rest (0.4 each way was simulated at 44 % delivery on the elbow; 0.55 is what the servo can follow). Fast down, fast up, then nothing: the stillness either side is what makes it
    a stamp and not a bob. The +16 units of yaw say WHICH foot -- small on purpose, because a bigger
    turn would read as "to the right" instead."""
    return (Call(2, says="right foot let's stomp")
            .to(_at(STOMP, base_yaw=STOMP_YAW), 0.55)
            .home(0.55)
            .hold(0.9))


def cha_stomp_left() -> Call:
    """"left foot let's stomp" -- the mirror, yaw -16."""
    return (Call(2, says="left foot let's stomp")
            .to(_at(STOMP, base_yaw=-STOMP_YAW), 0.55)
            .home(0.55)
            .hold(0.9))


def cha_cha() -> Call:
    """"cha cha real smooth" -- a human does three small quick steps in place, weight forward and back.
    The lamp rocks the head about 0.055 m out and home three times, one rock a beat, every segment
    eased so the three rocks join into one continuous swell (an eased pair out-and-back IS a raised
    cosine). Small and smooth is the instruction: this call exists to say "keep dancing, nothing new",
    and it is deliberately the quietest move in the set so the loud ones stay loud."""
    call = Call(4, says="cha cha real smooth")
    for _ in range(3):
        call.to(ROCK, 0.5).home(0.5)
    return call.hold(1.0)


def cha_turn() -> Call:
    """"turn it out" / "right foot two stomps, turn it out" -- a human turns a full 360.
    The lamp cannot: base_yaw is safe to about +-70 units, a fifth of a turn. So the 360 is replaced
    by the largest sweep it owns -- out to +65, all the way across to -65, and home -- with the arm
    reached out first, because sideways travel is r*sin(yaw) and reaching takes r from 0.084 m to
    0.146 m. That is 0.207 m of head travel across the room and 0.369 m of the lit shade, the widest
    gesture in this file, and the half-beat holds at each end stop it reading as a wobble. An honest
    stand-in for the 360, and it is the call a follower is most likely to try to copy literally.
    8 beats, so the sweep is slow enough for the servos to deliver all 130 units of it."""
    return (Call(8, says="turn it out")
            .to(_yawed(TURN_REACH, TURN_YAW), 1.5).hold(0.5)
            .to(_yawed(TURN_REACH, -TURN_YAW), 3.0).hold(0.5)
            .home(2.0).hold(0.5))


def cha_slide_left() -> Call:
    """"slide to the left" -- a human glides sideways, body facing front the whole way.
    The lamp reaches out (r 0.158 m, so the turn is worth twice the travel it is worth at home) and
    swings left at CONSTANT SPEED, never stopping, with wrist_roll counter-tilting -0.4 units per unit
    of yaw to hold the shade LEVEL instead of letting it bank. Everything that makes cha_look_left
    look like a look is deliberately absent: no snap, no hold, no bank. The way out is slower than the
    way home (2.25 beats against 1.75), so the eye takes the leftward half as the move."""
    return (Call(4, says="slide to the left")
            .glide(_yawed(REACH, -SLIDE_YAW, level=True), 2.25)
            .home(1.75, "glide"))


def cha_slide_right() -> Call:
    """"slide to the right" -- the mirror."""
    return (Call(4, says="slide to the right")
            .glide(_yawed(REACH, SLIDE_YAW, level=True), 2.25)
            .home(1.75, "glide"))


def cha_criss_cross() -> Call:
    """"criss cross" -- a human jumps their feet apart and crosses them, twice, fast.
    The lamp flicks +45, crosses THROUGH the middle to -45 in a single beat (292 units/s of the 380
    the runtime allows) and flicks home. Two flicks, one crossing, no hold longer than a quarter beat:
    against cha_turn it is half the width and four times the speed, which is the difference the eye
    uses. The last beat is still, so the count stays visible."""
    return (Call(4, says="criss cross")
            .to(_yawed(CROSS_REACH, CROSS_YAW), 0.75).hold(0.25)
            .to(_yawed(CROSS_REACH, -CROSS_YAW), 1.0).hold(0.25)
            .home(0.75).hold(1.0))


def cha_how_low() -> Call:
    """"how low can you go" -- a human sinks toward the floor for the whole call.
    The lamp descends from home to bc.BOTTOM, the lowest pose it has (head 0.086 m, 3.3 cm above the
    table, on the table margin itself), at CONSTANT speed for 6 beats: about 26 units/s, the slowest
    thing the lamp does. Slowness is the message, so no easing -- an ease would arrive early and hang
    about, and "how low" is a question you answer gradually.
    It cannot arrive at the very end of the call and stay there: every clip must end at bc.START or
    the next call starts with a snap. So the descent owns 6 of the 8 beats, the bottom is HELD for
    half a beat (the punchline, and the half beat is what makes the servos actually reach it), and the
    stand-up is the 1.5-beat tail. The tail is brisk on purpose: it must not be mistaken for a move."""
    return (Call(8, says="how low can you go")
            .glide(bc.BOTTOM, 6.0)
            .hold(0.5)
            .home(1.5))


def cha_to_the_top() -> Call:
    """"bring it to the top" -- a human reaches up over the whole call.
    The lamp rises from home to bc.TOP (head 0.435 m, its highest, elbow straight up) at constant
    speed over 6 beats, holds the top half a beat and comes down over 1.5. Same shape as cha_how_low
    and deliberately so: they are a matched pair and the follower reads the DIRECTION, which is the
    only thing that differs."""
    return (Call(8, says="bring it to the top")
            .glide(bc.TOP, 6.0)
            .hold(0.5)
            .home(1.5))


def cha_freeze() -> Call:
    """"FREEZE" -- a human stops dead.
    The lamp holds bc.START for the whole call and moves nothing: peak speed 0 on every joint. It is a
    clip and not a gap because the cue sheet schedules clips, and because a freeze that is a GAP would
    let whatever ran last keep its pose; this one puts the lamp at home, which is where the next call
    expects to start. Stillness only reads as a freeze if the moves around it are big -- which is the
    other reason every move in this file ends in a hold at home."""
    return Call(4, says="FREEZE").hold(4.0)


def cha_knees() -> Call:
    """"hands on your knees" -- a human bends at the knees, twice, hands sliding down.
    The lamp bobs the head 0.149 m down and back up twice, a beat down and a beat up, with a 0.2-beat
    punctuation at the top so two bobs read as two. Deeper than cha_stomp_right (0.149 m against
    0.113) and two and a half times slower: the depth says "knees", the tempo says it is not a stamp."""
    call = Call(4, says="hands on your knees")
    for _ in range(2):
        call.to(KNEE, 0.9).home(0.9).hold(0.2)
    return call


def cha_charlie_brown() -> Call:
    """"charlie brown" -- a human kicks a foot forward and slides the other back, four quick times.
    The lamp quotes four of its own moves in four beats -- forward, back, left, right -- at 70 % of
    the lean and the retreat and 40 units of yaw, one pulse a beat, home between each. The point of
    this call is the SEQUENCE, so each pulse is a half-beat out and a half-beat back: a follower
    counts four directions, and any one of them alone would be a different call."""
    call = Call(4, says="charlie brown")
    for pose in (_toward(LEAN, 0.7), _toward(BACK, 0.7),
                 _at(bc.START, base_yaw=-CB_YAW), _at(bc.START, base_yaw=CB_YAW)):
        call.to(pose, 0.45).home(0.45).hold(0.10)
    return call


def cha_reverse() -> Call:
    """"reverse, reverse" -- a human hops backward.
    Same shape as cha_hop and the opposite travel: the head goes up AND 0.042 m BACK with the nose
    flicked 33 deg UP (cha_hop's lands nose DOWN), then comes down BEHIND home (head_y 0.074 against
    home's 0.083) before recovering. Hop and reverse are a matched pair a follower sees seconds apart,
    so they differ in the one thing that matters -- which way the lamp went -- and in nothing else
    except this: the strokes are 0.5/0.55 beat against the hop's 0.4/0.5, because backward is the
    expensive direction. Stopping the arm at the top of a backward leap swings the ZMP toward the back
    of the base, and the sharp version measured -0.041 against the -0.030 bound. Slowing the leap by a
    quarter buys it back (-0.025) with the travel intact; shrinking the travel instead would have cost
    the only thing the move is for."""
    return _hop(Call(2, says="reverse"), REV_TOP, REV_LAND, rise=0.55, fall=0.6, recover=0.4)


# ----------------------------------------------------------------------------- the short versions
# DJ Casper does not give every call the same room: in the basic step "to the left" and "take it back
# now y'all" are two beats apart, "slide to the left / slide to the right" likewise, and a clip that
# outlasts its call is blended away by the runtime when the next one is posted. cha_sheet.py picks
# the longest version that fits the gap to the next call.
def cha_look_right_2() -> Call:
    """"to the right", two beats: the same look, a shorter hold."""
    return Call(2, says="to the right").to(_at(bc.START, base_yaw=LOOK_YAW), 0.6).hold(0.9).home(0.5)


def cha_look_left_2() -> Call:
    return Call(2, says="to the left").to(_at(bc.START, base_yaw=-LOOK_YAW), 0.6).hold(0.9).home(0.5)


def cha_take_it_back_2() -> Call:
    """"take it back now y'all", two beats: one step back instead of two."""
    return Call(2, says="take it back now y'all").to(BACK, 0.8).hold(0.4).home(0.6).hold(0.2)


def cha_slide_left_2() -> Call:
    return Call(2, says="slide to the left").glide(_yawed(REACH, -SLIDE_YAW, level=True), 1.1).home(0.9, "glide")


def cha_slide_right_2() -> Call:
    return Call(2, says="slide to the right").glide(_yawed(REACH, SLIDE_YAW, level=True), 1.1).home(0.9, "glide")


def cha_clap_4() -> Call:
    call = Call(4, says="clap your hands")
    for nod in range(4):
        call.to(CLAP_DOWN, CLAP_STROKE).to(CLAP_UP if nod < 3 else bc.START, CLAP_STROKE)
    return call


def cha_turn_4() -> Call:
    """"turn it out", four beats: the same sweep, right then left, at twice the pace."""
    return (Call(4, says="turn it out")
            .to(_yawed(TURN_REACH, TURN_YAW), 0.9).hold(0.3)
            .to(_yawed(TURN_REACH, -TURN_YAW), 1.5).hold(0.3)
            .home(0.8).hold(0.2))


def cha_how_low_4() -> Call:
    return Call(4, says="how low can you go").glide(bc.BOTTOM, 2.4).hold(0.3).home(1.3)


def cha_to_the_top_4() -> Call:
    return Call(4, says="bring it to the top").glide(bc.TOP, 2.3).hold(0.2).home(1.5)


# The cue order is the song's order, near enough: this list is what --list prints and what move.sh
# knows about, and it is the order the report reads best in.
MOVES = {fn.__name__: fn for fn in (
    cha_look_right, cha_look_left, cha_clap, cha_lean_forward, cha_take_it_back,
    cha_hop, cha_hop_two, cha_stomp_right, cha_stomp_left, cha_cha,
    cha_turn, cha_slide_left, cha_slide_right, cha_criss_cross, cha_how_low,
    cha_to_the_top, cha_freeze, cha_knees, cha_charlie_brown, cha_reverse,
    cha_look_right_2, cha_look_left_2, cha_take_it_back_2, cha_slide_left_2, cha_slide_right_2,
    cha_clap_4, cha_turn_4, cha_how_low_4, cha_to_the_top_4,
)}


# ----------------------------------------------------------------------------- build and report
def lamp_model(robotdesc: Path | None = None) -> LampModel:
    """The measured lamp: a robotdesc dir (the Mac's pi5_feetech_r1/ + lelamp-calibration.json), or
    FTM_ROBOT_DIR / LELAMP_CALIBRATION_PATH, or the vendor checkout and the lamp's own calibration.
    The servo calibration is what makes a unit a real angle, so a wrong one silently mis-sizes every
    head-travel number below: LampModel.scale_source says which was used and --report prints it."""
    if robotdesc is not None:
        return LampModel(Path(robotdesc) / "pi5_feetech_r1",
                         calibration=Path(robotdesc) / "lelamp-calibration.json")
    from spatial import DEFAULT_ROBOT_DIR
    robot_dir = Path(os.environ.get("FTM_ROBOT_DIR") or DEFAULT_ROBOT_DIR)
    return LampModel(robot_dir, calibration=Path(os.environ.get("LELAMP_CALIBRATION_PATH")
                                                 or bc.LAMP_CALIBRATION))


def runtime_peak_speed(U: np.ndarray) -> float:
    """Peak joint speed the runtime sees: after trajectory_processing.smooth_motion_plan's 5-frame
    moving average (first and last three frames kept), which is what limit_motion_velocity measures."""
    V = U.copy()
    for j in range(U.shape[1]):
        x = U[:, j]
        c = np.convolve(np.pad(x, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
        c[:3] = x[:3]
        c[-3:] = x[-3:]
        V[:, j] = c
    return float(np.max(np.abs(np.diff(V, axis=0)) * FPS))


def measure(name: str, model: LampModel, validator: bc.Validator) -> dict:
    """Build one move and everything the operator needs to judge it. Nothing is written here."""
    call = MOVES[name]()
    U = call.rows()
    heads = np.array([model.head(dict(zip(JOINTS, u)))["position"] for u in U])
    report = validator.validate(U)
    ok, reasons = report["ok"], list(report["reasons"])
    rt = runtime_peak_speed(U)
    if rt > CHA_SPEED_LIMIT:
        ok = False
        reasons.append(f"runtime would stretch time: {rt:.0f} units/s after its smoothing, ceiling {CHA_SPEED_LIMIT:.0f}")
    lead = U[:3]
    if np.abs(lead - lead[0]).max() > 1e-6:
        ok = False
        reasons.append("no stationary lead-in: the runtime would blend this clip and drop its start")
    return {"name": name, "says": call.says, "beats": call.beats, "frames": len(U), "speed_rt": rt,
            "seconds": len(U) / FPS, "rows": U, "bite": call.bite,
            "speed": bc.peak_speed(U), "travel": heads.max(axis=0) - heads.min(axis=0),
            "head_y_min": report["head_y_min"], "zmp": (report["zmp_y_min"], report["zmp_y_max"]),
            "ok": ok, "reasons": reasons}


def print_report(rows: list[dict], model: LampModel) -> None:
    print(f"the Cha Cha Slide at {TEMPO_BPM:g} bpm: a beat is {BEAT:.3f} s, a four-beat call {4 * BEAT:.3f} s")
    print(f"geometry: {model.scale_source}")
    print(f"limits: speed {CHA_SPEED_LIMIT:.0f} units/s per joint after the runtime's smoothing (%lim), head_y >= {bc.HEAD_Y_MIN:+.3f} m, "
          f"zmp_y {bc.ZMP_Y_MIN:+.3f}..{bc.ZMP_Y_MAX:+.3f} m (the lamp tips at +-0.100)")
    print()
    print(f"{'move':<18}{'beats':>6}{'sec':>6}{'frames':>7} | "
          f"{'yaw':>5}{'bpit':>5}{'elb':>5}{'roll':>5}{'wpit':>5}{'%lim':>6} | "
          f"{'dx':>6}{'dy':>6}{'dz':>6} | {'head_y':>7}{'zmp_lo':>8}{'zmp_hi':>8} | result")
    print("-" * 122)
    for r in rows:
        s, t = r["speed"], r["travel"]
        result = "PASS" if r["ok"] else "FAIL"
        print(f"{r['name']:<18}{r['beats']:>6.1f}{r['seconds']:>6.2f}{r['frames']:>7} | "
              f"{s[0]:>5.0f}{s[1]:>5.0f}{s[2]:>5.0f}{s[3]:>5.0f}{s[4]:>5.0f}"
              f"{100 * r['speed_rt'] / CHA_SPEED_LIMIT:>5.0f}% | "
              f"{t[0]:>6.3f}{t[1]:>6.3f}{t[2]:>6.3f} | "
              f"{r['head_y_min']:>7.3f}{r['zmp'][0]:>+8.3f}{r['zmp'][1]:>+8.3f} | {result}")
        for why in r["reasons"]:
            print(f"{'':<18}  -> {why}")
        if r["bite"] > 1e-6:
            print(f"{'':<18}  -> clamp_envelope moved this trajectory by {r['bite']:.2f} units: "
                  f"a pose in this move is outside the envelope and was rewritten, not danced")
    print("-" * 122)
    print("dx/dy/dz: how far the HEAD travels on each axis, in metres (+y toward the audience, +z up,")
    print("  +x the lamp's right). This follows the head's own centre; the lit shade reaches 0.11 m")
    print("  beyond it, so on a turn its far end travels about twice as far (cha_look_right: head")
    print("  0.051 m, shade nose 0.128 m; cha_turn: head 0.207 m, shade nose 0.369 m).")
    print("sec: what the runtime plays. The last frame is the home frame the next call starts from,")
    print(f"  so a call fills sec - 1/{FPS:.0f} s of the bar -- a four-beat call is {4 * BEAT:.3f} s.")
    print("Judge legibility on dx/dy/dz and on wpit, which is the shade's own nod: it moves no mass,")
    print("  so it never shows up in the ZMP columns, and it is the joint that reads furthest away.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="directory to write the clips into")
    ap.add_argument("--report", action="store_true", help="print the per-move table")
    ap.add_argument("--only", action="append", help="build just this move (repeatable)")
    ap.add_argument("--list", action="store_true",
                    help="print name:beats for every call, in cue order -- move.sh's own table, so "
                         "`cha_moves.py --list | tr '\\n' ' '` regenerates it after a change")
    ap.add_argument("--robotdesc", default=None, help="vendor robot description dir, for the geometry")
    args = ap.parse_args()

    if args.list:
        for name, make in MOVES.items():
            print(f"{name}:{make().beats:g}")
        return 0
    if not args.out and not args.report:
        ap.error("nothing to do: give --out, --report or --list")

    names = args.only or list(MOVES)
    unknown = [n for n in names if n not in MOVES]
    if unknown:
        ap.error(f"unknown move(s): {', '.join(unknown)}")

    model = lamp_model(Path(args.robotdesc) if args.robotdesc else None)
    validator = bc.Validator(model=model, robot_dir=model.robot_dir)
    rows = [measure(name, model, validator) for name in names]

    if args.report:
        print_report(rows, model)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        # Not one file is written when any move fails. The vocabulary is a SET: a cue sheet that calls
        # a name the pack does not hold leaves the lamp standing still on that call, which to a deaf
        # follower is indistinguishable from "FREEZE".
        print(f"\n{len(bad)} of {len(rows)} moves FAILED validation and NOTHING was written: "
              + ", ".join(r["name"] for r in bad))
        for r in bad:
            print(f"  {r['name']}: " + "; ".join(r["reasons"]))
        return 2

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for r in rows:
            # write_clip_atomic takes the rows and does csv_text itself (double-encoding is a TypeError)
            md5 = bc.write_clip_atomic(out / f"{r['name']}.csv", r["rows"])
            print(f"wrote {out / (r['name'] + '.csv')}  md5 {md5}")
    print(f"\n{len(rows)} move{'s' if len(rows) > 1 else ''}, all PASS"
          + (f", {sum(r['frames'] for r in rows) / FPS:.1f} s of choreography" if len(rows) > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

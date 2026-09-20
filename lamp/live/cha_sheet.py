#!/usr/bin/env python3
"""Turn a word-timed transcript of the Cha Cha Slide into a cue sheet for cha_run.py.

The song is a list of spoken CALLS ("to the left", "one hop this time", "criss cross") and each call
has a move in cha_moves.py. A cue sheet line is `<beat> <clip> # <what was said>`: the beat is an
ABSOLUTE index on the measured grid (123.05 bpm, beat 0 at 0.336 s), which is what cha_run.py fires on.

Where the beat comes from. The recogniser gives every word a start time; the call's first keyword
word is snapped to the nearest EVEN beat (see ONSET_LAG below: the calls sit on half-bars and the
recogniser's word times wander half a beat); the measured grid is exact to a few ms over the whole
track (grid.py). cha_run.py then subtracts its lead (LEAD_BEATS + the runtime's start latency), so the lamp
is already moving as the call lands and a deaf follower reads the move, not the words.

Overlap rule. Every clip owns the beats it lasts (a 4-beat look, a 2-beat hop, an 8-beat turn). A call
that lands inside the previous clip's beats is dropped with a note, because posting a clip while the
arm is mid-move makes the runtime blend it -- and the blend throws the first 1.5 s of the new clip away
(ground_truth_2026-09-20.md). The one exception is a repeat of the same call ("cha cha again",
"one hop this time, one hop this time"): the repeat starts the moment the previous copy ends.

    whisper-cli -m ggml-base.en.bin -f cha16k.wav -oj -ml 1 -sow -of cha_words   # cha_words.json
    python3 cha_sheet.py cha_words.json --out cha.sheet                            # the cue sheet
    python3 cha_sheet.py cha_words.json --out cha.sheet --transcript              # also print the words
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BPM, BEAT0 = 123.05, 0.336                  # measured on this file (grid.py); cha_run.py has the same
PERIOD = 60.0 / BPM

# Call patterns, in the order they are tried at every word. Each is (regex over the lower-cased word
# stream, clip, beats the clip lasts). The regex is matched at word boundaries in a running window of
# the next six words, so "to the left" is found even if the recogniser split it "to the, left".
CALLS: list[tuple[str, str, int]] = [
    # The recogniser (whisper base.en on a track with a beat under it) mishears the calls the same
    # way every time: "hop" -> "half"/"halves", "cha cha" -> "shot shot", "criss" -> "chris",
    # "how low" -> "hello", "pump it up" -> "hop it out". Those spellings are matched on purpose.
    (r"\b(everybody )?clap( clap)* your hands\b", "cha_clap", 8),
    (r"\bslide to the left\b", "cha_slide_left", 4),
    (r"\bslide to the right\b", "cha_slide_right", 4),
    (r"\bto the left\b", "cha_look_left", 4),
    (r"\bto the right\b", "cha_look_right", 4),
    (r"\btake it back\b", "cha_take_it_back", 4),
    (r"\b(two|three|four|five|six|seven|eight) (hops?|hal(f|ves)|hots?)\b", "cha_hop_two", 4),
    (r"\b(one|turn it off) (hops?|half|hot)\b", "cha_hop", 2),
    (r"\bright foot\b", "cha_stomp_right", 2),
    (r"\bleft foot\b", "cha_stomp_left", 2),
    (r"\b(cha cha|shot shot|cha shot|shot cha|chacha)\b", "cha_cha", 4),
    (r"\b(turn it out|go to work)\b", "cha_turn", 8),
    (r"\b(criss|chris|kris) cross\b", "cha_criss_cross", 4),
    (r"\b(how low|hello|how long) can you go\b", "cha_how_low", 8),
    (r"\bcan you go down low\b", "cha_how_low", 8),
    (r"\b(bring it )?to the top\b", "cha_to_the_top", 8),
    (r"\bfreeze\b", "cha_freeze", 4),
    (r"\bhands on your knees\b", "cha_knees", 4),
    (r"\bknees\b", "cha_knees", 4),
    (r"\bcharlie brown\b", "cha_charlie_brown", 4),
    (r"\breverse\b", "cha_reverse", 2),
    (r"\b(hop|pump) it (up|out)\b", "cha_to_the_top", 8),
    (r"\bget funky with it\b", "cha_knees", 4),
]
# Calls that are talk, not moves, and would otherwise steal the beats of the move that follows:
# "now it's time to get funky" comes 2 beats before "to the right now".
SKIP = re.compile(r"\b(time to get funky|funky funky)\b")
WINDOW = 8                                  # words of look-ahead when matching a phrase
MAX_DELAY = 4                               # beats a colliding call may slip (2 s): late beats never, since the loud moves are the point
# Shorter versions of a move, by length in beats, for when the next call comes sooner than the full
# move lasts. DJ Casper's calls are NOT all one length: in the basic step "to the left" and "take it
# back now y'all" are two beats apart, "slide to the left / slide to the right" likewise, while the
# claps and "how low" own eight. cha_moves.py builds every variant listed here.
VARIANTS = {
    "cha_clap": {4: "cha_clap_4"},
    "cha_turn": {4: "cha_turn_4"},
    "cha_how_low": {4: "cha_how_low_4"},
    "cha_to_the_top": {4: "cha_to_the_top_4"},
    "cha_look_left": {2: "cha_look_left_2"},
    "cha_look_right": {2: "cha_look_right_2"},
    "cha_take_it_back": {2: "cha_take_it_back_2"},
    "cha_slide_left": {2: "cha_slide_left_2"},
    "cha_slide_right": {2: "cha_slide_right_2"},
    "cha_hop_two": {2: "cha_hop"},
}


def load_words(path: Path) -> list[tuple[float, str]]:
    """(start_seconds, word) from whisper-cli's -oj output (one word per segment with -ml 1 -sow)."""
    data = json.loads(path.read_text())
    words = []
    for seg in data["transcription"]:
        text = re.sub(r"[^a-z' ]", " ", seg["text"].lower()).strip()
        if not text:
            continue
        t = seg["offsets"]["from"] / 1000.0
        for w in text.split():
            words.append((t, w))
    return words


# Calls sit on the HALF-BAR: every call in this song starts on an even beat of the grid (the flux
# onsets in phrases.npy fall 0.25-1.0 beat after an even beat, 91 %; none sit near an odd one). The
# recogniser's word starts wander +-0.55 beat around those onsets, so snapping to the nearest single
# beat put half the cues a beat off. Quantising to even beats after subtracting the 0.5-beat onset
# lag puts them where the DJ put them.
ONSET_LAG = 0.5


# Where the audio itself marks a call (cha_onsets.npy: the 42 loudest vocal onsets found by spectral
# flux in grid.py, accurate to a frame), that onset outranks the recogniser's word time. The word
# times are only trusted where no onset was detected within ONSET_REACH beats.
ONSET_REACH = 1.25


def _onsets() -> list[float]:
    here = Path(__file__).resolve().parent / "cha_onsets.npy"
    if not here.exists():
        return []
    try:
        import numpy as np
        return [float(v) for v in np.load(here)]
    except Exception:
        return []


ONSETS = _onsets()


def to_beat(seconds: float) -> int:
    x = (seconds - BEAT0) / PERIOD
    near = [(abs((o - BEAT0) / PERIOD - x), o) for o in ONSETS]
    if near:
        d, o = min(near)
        if d <= ONSET_REACH:
            x = (o - BEAT0) / PERIOD
    return 2 * int(round((x - ONSET_LAG) / 2.0))


def build(words: list[tuple[float, str]]) -> tuple[list[tuple[int, str, str]], list[str]]:
    """Pass 1 finds every call; pass 2 resolves overlaps.

    Overlap: a call that lands while the previous clip still owns the beat is moved LATER to the first
    free beat if that is at most MAX_DELAY beats away (the follower sees the move a beat late rather
    than not at all), a repeat of the same call queues right behind its twin, anything else is dropped
    with a note. Never earlier: the clip would then start before the call and read as a different move.
    """
    found: list[tuple[int, str, str]] = []
    i = 0
    while i < len(words):
        window = " ".join(w for _, w in words[i:i + WINDOW])
        if SKIP.match(window):
            i += 1
            continue
        hit = None
        for pattern, clip, beats in CALLS:
            m = re.match(pattern, window)   # anchored: the phrase must START at word i
            if m:
                hit = (clip, beats, m.group(0))
                break
        if hit is None:
            i += 1
            continue
        clip, beats, said = hit
        found.append((to_beat(words[i][0]), clip, said, beats))
        i += len(said.split())

    cues: list[tuple[int, str, str]] = []
    notes: list[str] = []
    busy_until = -1
    last_clip = None
    for k, (beat, clip, said, beats) in enumerate(found):
        if beat <= busy_until:
            if clip == last_clip:
                # "two hops, two hops" said twice inside one four-beat call, "reverse reverse": the
                # same move again before the first copy has finished is the SAME move, not a queue.
                continue
            first_free = busy_until + 1
            if first_free - beat <= MAX_DELAY:
                notes.append(f"'{said}' moved {first_free - beat} beat(s) later, {beat} -> {first_free}: {last_clip} owned the beat")
                beat = first_free
            else:
                notes.append(f"dropped '{said}' at beat {beat}: {last_clip} owns beats to {busy_until}")
                continue
        # A move whose next call comes sooner than it lasts plays its longest shorter version, so the
        # next call is not blended away (see VARIANTS).
        nxt = next((b for b, c, _, _ in found[k + 1:] if c != clip), None)
        if nxt is not None and nxt - beat < beats and clip in VARIANTS:
            fits = [(n, name) for n, name in VARIANTS[clip].items() if n <= nxt - beat]
            if fits:
                n, name = max(fits)
                notes.append(f"{clip} at {beat} shortened to {name}: next call {nxt - beat} beats later")
                clip, beats = name, n
        cues.append((beat, clip, said))
        # A clip owns its beats PLUS one: the runtime starts it 250-450 ms after the POST (measured), so
        # a 2-beat clip is still playing when the call 2 beats later lands, and a clip posted while
        # another plays at the same priority is refused (2026-09-20 04:26: "take it back" and a
        # "cha cha" never played).
        busy_until = beat + beats
        last_clip = clip
    return cues, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", type=Path, help="whisper-cli JSON (-oj -ml 1 -sow)")
    ap.add_argument("--out", type=Path, required=True, help="cue sheet to write")
    ap.add_argument("--transcript", action="store_true", help="also print the timed words")
    args = ap.parse_args()

    words = load_words(args.words)
    if args.transcript:
        for t, w in words:
            print(f"{t:7.2f}  beat {to_beat(t):4d}  {w}")
    cues, notes = build(words)
    lines = [f"# Cha Cha Slide cue sheet: {len(cues)} cues from {len(words)} recognised words",
             f"# beat = round((word start - {BEAT0}) / {PERIOD:.5f});  {BPM} bpm"]
    for beat, clip, said in cues:
        lines.append(f"{beat:<5d} {clip:<18s} # {said}  ({BEAT0 + beat * PERIOD:6.1f} s)")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

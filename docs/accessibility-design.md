# Accessibility Design & Multi-Sensory Music Experience

Status: Canonical specification, written 2026-09-19 for Hack the North 2026.

---

## 1. Philosophical Foundations & Terminology

### Respectful, Accurate Language
- **Deaf and hard-of-hearing**: Always use "Deaf and hard-of-hearing". Never use "hearing impaired".
- **Enriching Experience, Not "Fixing"**: We are creating a rich, multi-sensory way to perceive, embody, and share music. We do not view deafness as something to "fix" or "cure".

### Multi-Sensory Embodiment
Music is fundamentally acoustic vibrations in space. For Deaf and hard-of-hearing listeners, the musical information is communicated across three coordinated sensory channels:
1. **Touch (Haptic Sensations)**: Tactile vibration patterns mapped directly to musical anatomy (bass resonance, percussive transients).
2. **Sight (Robot Motion & Safe Lighting)**: A dedicated performance robot facing the listener, delivering smooth illumination pulses and expressive posture.
3. **Togetherness (Shared Shared Clock)**: Zero-drift synchronization ensures everyone in the room experiences the beat simultaneously.

---

## 2. Haptic Architecture & Mapping

Musical frequencies and dynamics are split between physical actuators based on their mechanical capabilities:

### A. Continuous Bass Texture (50–100 Hz)
- **Actuator**: TITAN Core Channels L & R (PAM8403 Class-D Audio Amplifier driving voice-coil haptic motors) and iPhone continuous low-frequency hums.
- **Signal**: Continuous `bass_envelope` extracted from the 50–100 Hz frequency band.
- **Perception**: Felt as deep, sustained rumble against the palms or torso, communicating pitch movement and bassline presence.

### B. Percussive Transients (Kicks & Snares)
- **Actuator**: TITAN Core Channel M (DRV8212 H-Bridge driver) and iPhone Taptic Engine via native Core Haptics.
- **Signal**: Sharp impact events (`Tick`, `Transient`) stamped with energy and sharp attack profiles.
- **Perception**: High-acceleration physical impact that communicates the exact rhythm, tempo, and groove of the drum kit.

---

## 3. Visual & Robot Performance (`lamp/`)

### Eye Safety & Photo-Sensitivity (AGENTS.md Rule 6)
Visual perception is critical for Deaf audiences. Intense flashing or uncontrolled strobing can cause severe visual discomfort or trigger photo-sensitive seizures.
- **WCAG 2.2 SC 2.3.1 Compliance**: Strictly enforced by `safety.flash.FlashLimiter`:
  - No more than 3 flashes in any rolling one-second window.
  - Saturated red flashes ($R/(R+G+B) \ge 0.8$ with $\Delta u'v' > 0.2$) are strictly prohibited.
  - Light transitions smoothly fade rather than cutting abruptly dark.
- **Light Color Palette**: Ambient rests use gentle, cool illumination (e.g. cyan/deep blue); musical energy and bass modulation transition toward warm violet/magenta accents, avoiding high-contrast flashing.

### Bodily Expression & Posture
- The LeLamp robot functions as a performer and companion:
  - Continuously tracks the listener's head/face to maintain visual connection.
  - Subtly nods in time with the rhythmic cadence.
  - Expresses musical tension and release (stretching upward on musical builds, dancing energetically on beat drops).

---

## 4. Evaluation & User Feedback

- **Acceptance Criteria**: The ultimate test of FeelTheMusic is qualitative feedback from Deaf and hard-of-hearing individuals trying the installation.
- **Key Validation Questions**:
  1. Does the haptic separation between bass texture and sharp kicks feel natural and intelligible?
  2. Does the robot's physical presence and light enhance the musical connection without causing visual fatigue?
  3. Is the synchronization between touch and sight perceived as instantaneous and unified?

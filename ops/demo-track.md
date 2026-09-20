# Demo track

Created 2026-09-19. Task #7; owner: @codexfranklin.

Status: waiting for the chosen source audio and the host's permission record.
No track has been selected, exported, analysed or tested on hardware by this task.

## Source and permission record

Fill this record before distributing the audio. Keep private correspondence,
personal details and credentials outside the repository.

| Field | Value |
|---|---|
| Track title and version | Pending |
| Public creator credit | Pending |
| Source filename and SHA-256 | Pending |
| Public source/license URL or non-private permission reference | Pending |
| Booth playback permission confirmed by host | Pending |
| Recording/publication permission, if the demo is recorded | Pending |
| Redistribution permission, if the audio is committed or shared | Pending |
| Required attribution | Pending |
| Selected excerpt start/end in the original source | Pending |

The host supplies the chosen track and confirms the permitted use. A streaming
link alone does not supply the local audio needed for offline playback.

## Export procedure

Use an installed offline audio editor or converter. Record its exact version and
export settings below so another operator can reproduce the file.

1. Preserve the source. Work on a separate export of the agreed excerpt.
2. Confirm the playback conductor's supported sample rate and channel count.
   Export a plain RIFF/WAVE file with integer PCM, format tag 1. Use 16-bit PCM
   as a proposed compatibility default; confirm it with the conductor owner.
   Do not substitute floating-point WAV or WAVE_FORMAT_EXTENSIBLE silently.
3. Prepend digital silence. A 2-second lead-in is a proposed starting value for
   rehearsal, not a measured startup requirement. Record the chosen duration
   and exact frame count; extend it if offline startup requires more time.
4. Analyse the final exported file, including its silence. Play that same file.
   Keep analysis timestamps on its sample timeline; do not remove the lead-in
   from only one path or add it to event timestamps a second time.
5. Keep the shared-clock rule: output at `pts + L - output_latency`, initially
   `L = 300 ms`. The file's silence does not replace the room latency budget.
6. Copy the permitted asset to the demo machine before the event. Verify its
   hash there and rehearse with the network's internet uplink disconnected.

## Export and verification record

| Check | Result |
|---|---|
| Converter/editor and exact version | Not run |
| Export filename and SHA-256 | Not run |
| RIFF/WAVE format tag, PCM bit depth | Not run |
| Sample rate, channels, total frames, duration | Not run |
| Leading silent frames, all channels verified zero | Not run |
| Export decodes completely in the selected conductor | Not run |
| Analysis uses the exact exported hash | Not run |
| Kick/snare/bass detection checked on this actual track | Not run |
| Offline playback after a cold start | Not run |
| Hardware rehearsal: device class, conditions, measured timing | Not run |

Task #7 remains open until the source, permission record, export and verification
are complete. Synthetic fixtures can exercise the parser but do not establish
that detection works on the demo track.

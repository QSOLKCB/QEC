# Acoustic offline evidence viewer · v177.5

`docs/labs/acoustic-evidence.html` is a bounded, dependency-free local inspection
surface. Open that file in a browser with `acoustic-evidence.js` beside it. No
server, package installation, CDN, account, upload, or network request is needed.

**Browser visualization and audible demonstration only.** Imported hashes,
receiver decisions, independent-reference gates and logical outcomes remain
unverified claims. Canonical evidence requires independent Python replay against
externally retained source/reference pins. This viewer does not rerun the
receiver, generate canonical receipts, evolve quantum states, establish physical
quantum protection, or integrate the SPECTRAL repository or UI.

## Load actual artifacts

Generate an export using the procedure in
[the comparative evidence contract](ACOUSTIC_COMPARATIVE_EVIDENCE.md). Then:

1. Open `demonstration.json` to inspect the six actual roles from the broadband
   demonstration. Original/observed/filtered waveform samples are embedded in
   `roles`; candidate/rejected observations and the reference-gated bridge are
   displayed separately.
2. Open one `trials/NN_qD_CASE.json` to switch between its seven imported method
   summaries. `battery.json` additionally enables trial selection. The viewer
   does not recompute metrics or aggregate populations from this finite battery.
3. Open `demonstration_source.json`, an individual `_source.json`, or
   `battery_source.json` to inspect original PCM only. Unavailable observed and
   filtered stages remain empty. No replacement signal or outcomes are generated.
4. Stage pickers accept actual `lab/original.json`, `lab/observed.json`, and
   `lab/filtered.json`, or the corresponding WAV files. Select the intended role
   explicitly. Filename extensions do not establish the format: WAV signatures
   and JSON structures are parsed from the supplied bytes.
5. The main picker also accepts standalone native acoustic source,
   noise-observation, cancellation-receipt, harmonic-observation, or syndrome-
   bridge JSON. A standalone receiver or bridge receipt has no embedded stage
   PCM; attach the actual stage separately if you want to listen.

Standalone PCM JSON belongs in a stage picker, not the main receipt picker.
Native Digital/Packet transport receipts belong in their existing validators;
the acoustic viewer does not claim native transport replay.

## Read the three decision panels

- **Receiver candidate:** the imported raw receiver acceptance and estimated
  symbols, before independent reference authority. A coherent wrong-word
  substitution can be accepted here.
- **Rejections / vetoes:** rejected observation reasons, per-block candidates,
  and imported false-accept/reference-veto information where supplied. A rejected
  block may have a candidate symbol without an accepted estimate.
- **Reference-gated decision:** the caller-declared measured reference, imported
  gate, decoder decision and, for comparative methods, finite-model outcome.
  Importing an `accepted` flag or digest does not authenticate these claims.

The direct-syndrome control is labeled as having no acoustic receiver. The
exact-clean-reference control is labeled as privileged clean PCM. Their reference
costs and resource accounting are shown separately in the imported-method details.
Full native observations are available for the demonstration and standalone
receiver/gate artifacts. Comparative trial methods contain summaries and replay
hashes; the viewer does not invent absent per-block observations.

All receipt outcomes remain tied to the receipt's inputs. Manual stage imports
are visibly identified. An exact sample/rate comparison can show that an attachment
matches an embedded waveform; a mismatch or missing embedded waveform is explicitly
marked as unbound to the displayed receiver/gate claims. That local byte/sample
comparison does not establish canonical replay or reference provenance.

## Waveforms and actual audio

- Every trace uses the same signed PCM16 full-scale axis. Up to 800 plotted
  columns retain each interval's minimum and maximum; this is a display envelope,
  not a regenerated waveform. Very short inputs also connect their actual samples.
- Playback reads actual signed samples into a mono Web Audio buffer at their
  declared 24 kHz or 48 kHz sample rate. It applies a fixed 25% output gain only.
  There is no oscillator, generated replacement, denoising, adaptive fitting,
  normalization, looping, microphone capture, or autoplay.
- Playback is created only by a Play click. Stop, reset, changing trials/methods,
  importing evidence/stages, and page exit stop the current playback. Empty PCM
  cannot be played. Late asynchronous resumes cannot start obsolete audio or
  stop a newer playback.
- Browser/device resampling and device volume affect what you hear. Listening
  cannot certify recovery or physical protection. Start with low device volume.

## Bounded import and privacy contract

| Input | Limit |
|---|---|
| Main JSON file | 8 MiB UTF-8 |
| Stage JSON/WAV file | 1 MiB before reading |
| PCM | 73,728 mono signed 16-bit samples; 24 or 48 kHz |
| WAV parsing | At most 147,500 bytes including headers/chunks; at most 64 chunks |
| Battery | 1–38 trials/sources |
| Methods / receiver symbols / blocks | 7 / 32 / 3 |
| JSON tree | 32 nesting levels, 200,000 visited values, 128 entries per object/array |
| JSON strings | 294,912 characters each; stage filename 256 characters |

PCM JSON requires native `qec.acoustic-pcm.v1` fields, lowercase complete hex,
matching byte/sample counts, mono layout, allowed rate, and correctly shaped
claimed digests. Digests are **not** recomputed or authenticated in the browser.
The viewer checks recognized display structure, not the complete native canonical
schema. Imported Python integer fields outside the displayed bounded symbol and
counter fields can exceed JavaScript's exact integer range; the viewer must not
be used to verify arithmetic, compare canonical JSON bytes, or re-export evidence.

WAV interoperability accepts uncompressed mono PCM16 RIFF/WAVE with one format
chunk (16 bytes, or 18 with zero extension size), one data chunk, valid declared
lengths/byte rate/block alignment, and bounded optional metadata chunks. Stereo,
float, compressed, extensible, unsupported-rate, truncated, duplicate-format or
multiple-data files are rejected. Canonical QEC WAV files work unchanged. A
standard WAV accepted here need not satisfy Python's stricter canonical WAV reader.

Malformed and oversized files are rejected without constructing replacement
signals. A new evidence import clears previous receipt state and manual stage
attachments immediately. A failed stage replacement removes the previous manual
attachment; any visible fallback is explicitly the receipt's embedded PCM.
Pending older reads cannot restore data after reset, a newer same-role import,
a new evidence selection or a trial switch. Imported text is written using safe
text nodes, never parsed as HTML. There is no persistence or network use.

The supported SPECTRAL interoperability is **ordinary local PCM16 WAV and JSON
file interchange only**. No external SPECTRAL source, assets, UI or integration
is edited or asserted by this work.

## Canonical replay and test boundaries

Use the independent validators in
[the comparative evidence contract](ACOUSTIC_COMPARATIVE_EVIDENCE.md),
[the PCM contract](ACOUSTIC_CHANNEL_CONTRACT.md), and
[the syndrome bridge contract](ACOUSTIC_SYNDROME_BRIDGE.md). Keep source and
measured-reference pins outside the artifact under validation. The browser's
“structure inspection” label never substitutes for successful Python replay.

Run focused inspector checks from the repository root:

```sh
PYTHONPATH=src python -m pytest -q tests/acoustic/test_lab.py
node --check docs/labs/acoustic-evidence.js
```

The Python wrapper generates current native source/trial/battery/demonstration
artifacts in a temporary directory and passes them to `test_lab.cjs`. The Node
suite uses dependency-free DOM and audio stubs to exercise actual sample identity,
WAV parsing, separate claims, source-only/missing-stage behavior, hostile text,
malformed/oversized input, stale reads, resets, and interrupted/repeated playback.
It also reads the protected original clean fixtures without changing them.

These tests do **not** establish rendered-browser layout, acoustic quality,
physical-hardware behavior, canonical receipt acceptance, or full-suite status.
Rendered-browser and audible QA must be reported separately if actually performed.

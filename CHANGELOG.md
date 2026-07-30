# Changelog

All notable changes to Flux Fidelity are recorded here. Versions follow the
`manifest.json` and `package.json` version, which are validated to agree by
`npm run check`.

This file starts at 0.50.0. Earlier history is in the Git log; only one release
tag (`v0.49.0`) predates it.

## Unreleased

### Fixed

- Frame interpolation no longer stalls silently on unmuted playback. The display
  takeover gate now names the specific reason it refuses a frame, and the audio
  delay preparation failure behind the observed stall is fixed.
- The presentation watchdog reports the true cumulative stall duration instead of
  resetting its own clock, so a permanently stalled scheduler is no longer
  reported as a series of short self-healing stalls.
- Playback rate is honoured. Presentation deadlines previously mapped source time
  to wall clock 1:1, so any rate other than 1.0 desynchronised video from audio.
- Requesting a neural model that is not in the bundled manifest now fails
  explicitly instead of silently substituting a different model with a different
  scale.
- Settings survive a schema change. Records are migrated forward one version at a
  time rather than being discarded as corrupt.
- The neural "Native model scale" policy is reachable from the popup.
- Transparent images keep their alpha through the image upscaler.
- A transient neural manifest fetch failure no longer disables neural inference
  for the rest of the session.
- Returning to a page through the back/forward cache no longer discards the
  loaded model and all recurrent temporal state.
- Command failures report their specific cause rather than a single generic
  message, and a setting the runtime refused is no longer reported as merely
  pending.

### Added

- `Alt+Shift+U` toggles enhancement on the active tab, which also makes the
  feature reachable while a page is fullscreen.
- The popup reports source and output resolution, links the packaged self-test
  page, and can copy a diagnostics report.
- "Forget this site" clears stored settings for the current origin.

### Changed

- Keyboard focus is restored after a popup command instead of being dropped.
- Extension pages declare an explicit `default-src`; previously omitted CSP
  directives were unrestricted.
- Legacy hostname-keyed settings are removed once they have been migrated.
- Setting contracts live in one module, and CI fails if the content script's
  validation gate drifts from it.
- Image upscaling warns once on a wide-gamut display rather than being skipped.

### Performance

- The temporal tile plan is memoized instead of being recomputed every frame.
- Video access probing is cached, removing a synchronous readback from several
  per-frame and per-tick paths.
- An open popup no longer forces a full document and shadow-DOM walk every
  second.

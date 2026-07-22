# Browser video fixtures

These development-only files exercise the extension against a real top-level
`HTMLVideoElement`. The page exposes `window.__FSRCNNX_VIDEO_FIXTURE__` with
`ready()`, `loadSource(key)`, `pause()`, `play()`, and `snapshot()` methods.

The four WebM files are deterministic synthetic VP9 streams. Their dimensions,
decoded WebCodecs color tuple, byte length, and SHA-256 are pinned in
`fixture-manifest.json`. They are intentionally excluded from the extension
package.

Exact regeneration requires ffmpeg n7.1.1 with libvpx 1.15.0:

```sh
node tools/generate-browser-fixtures.mjs --write
node tools/check-browser-fixtures.mjs
```

The generator prints the complete ffmpeg commands, ffprobe results, sizes, and
hashes. Review that output and update the manifest deliberately after any
authorized regeneration.

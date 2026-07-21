# Shader sources

`FSRCNNX_x2_56-16-4-1.glsl` is the checked-in source for the high-quality FSRCNNX model. Run `npm run fetch:shader-sources` to obtain checksum-pinned upstream sources for standard FSRCNNX x2 and the three ArtCNN variants. Those sources are stored under `shaders/upstream/` and reproduce the corresponding checked-in JSON/WGSL assets byte-for-byte.

The legacy x3 and x4 FSRCNNX generated assets remain usable runtime inputs, but their original GLSL source revision could not be established from the repository history. Do not regenerate or redistribute those two assets as independently sourced models until provenance is established.

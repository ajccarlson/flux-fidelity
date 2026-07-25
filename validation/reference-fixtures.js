// Canonical numerical-reference cases and deterministic input samples.
//
// This module intentionally uses no Node APIs so the fixture generator and the
// extension-owned validation page consume the same input definitions.

export const REFERENCE_FIXTURE_SCHEMA_VERSION = 1;
export const REFERENCE_INPUT_VERSION = "structured-16-v1";

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

// These values are an audited reproduction policy, not generated metadata.
// Updating a fixture therefore requires a deliberate review of both this trust
// root and validation/reference-fixtures.json.
export const REFERENCE_GENERATOR_POLICY = deepFreeze({
  command: "npm run reference:generate",
  inputEncoding: "16-bit big-endian PGM/PPM",
  capture: "mpv window screenshot",
  outputEncoding: "raw little-endian RGB16",
  mpvBaseOptions: [
    "--no-config",
    "--vo=gpu-next",
    "--gpu-api=opengl",
    "--gpu-context=x11",
    "--gpu-sw=yes",
    "--hwdec=no",
    "--force-window=immediate",
    "--keep-open=yes",
    "--image-display-duration=10",
    "--frames=1",
    "--osc=no",
    "--osd-level=0",
    "--border=no",
    "--keepaspect-window=no",
    "--dither=no",
    "--screenshot-format=png",
    "--screenshot-high-bit-depth=yes",
  ],
  ssimOptions: [
    "--linear-downscaling=no",
    "--correct-downscaling=yes",
    "--dscale=catmull_rom",
    "--cscale=catmull_rom",
  ],
});

export const REFERENCE_TOOLCHAIN = deepFreeze({
  mpv: "v0.40.0",
  libplacebo: "v7.351.0",
  ffmpeg: "n7.1.1",
  renderer: "llvmpipe (LLVM 20.1.8, 256 bits)",
  mesa: "25.1.7-arch1.1",
  accelerated: false,
});

export const REFERENCE_INPUTS = deepFreeze({
  model: {
    id: "model",
    width: 37,
    height: 23,
    channels: 1,
    encoding: "gray16",
    formula: "structured-gray-v1",
  },
  ssimds: {
    id: "ssimds",
    width: 62,
    height: 38,
    channels: 3,
    encoding: "rgb16",
    formula: "structured-rgb-v1",
  },
  sharpen: {
    id: "sharpen",
    width: 37,
    height: 23,
    channels: 3,
    encoding: "rgb16",
    formula: "structured-rgb-v1",
  },
  "color-source": {
    id: "color-source",
    width: 13,
    height: 9,
    channels: 3,
    encoding: "rgb16",
    formula: "structured-rgb-v1",
  },
  "color-luma": {
    id: "color-luma",
    width: 25,
    height: 17,
    channels: 1,
    encoding: "gray16",
    formula: "structured-gray-v1",
  },
});

const MODEL_TOLERANCE = Object.freeze({ rmse: 0.008, p99: 0.02, max: 0.05 });

export const REFERENCE_CASES = deepFreeze([
  {
    id: "FSRCNNX_x2_16-0-4-1",
    label: "FSRCNNX standard x2 numerical reference",
    source: {
      path: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
      sha256: "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
    },
    inputs: [{ role: "source", id: "model" }],
    output: {
      kind: "fixture",
      path: "validation/references/FSRCNNX_x2_16-0-4-1.rgb16le",
      width: 74,
      height: 46,
      format: "rgb16le",
      comparison: "red",
      sha256: "b628c6b8eff2f77c8bec19ea8485fcee8a1b4c3e3af9b1482c3e30b2a76dcafa",
    },
    // The upstream hook can emit luma outside [0,1]; the pinned window
    // screenshot and production display path both clamp at presentation.
    oracle: { kind: "mpv-libplacebo-upstream", presentationClamp: true },
    tolerances: { rmse: 0.006, p99: 0.015, max: 0.03 },
  },
  {
    id: "FSRCNNX_x2_56-16-4-1",
    label: "FSRCNNX High x2 numerical reference",
    source: {
      path: "shaders/FSRCNNX_x2_56-16-4-1.glsl",
      sha256: "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
    },
    inputs: [{ role: "source", id: "model" }],
    output: {
      kind: "fixture",
      path: "validation/references/FSRCNNX_x2_56-16-4-1.rgb16le",
      width: 74,
      height: 46,
      format: "rgb16le",
      comparison: "red",
      sha256: "491301777449cb9310e19db4687a74900785b3f91867bead371458229d0fe6c9",
    },
    oracle: { kind: "mpv-libplacebo-upstream", presentationClamp: true },
    tolerances: { rmse: 0.006, p99: 0.015, max: 0.03 },
  },
  {
    id: "ArtCNN_C4F32",
    label: "ArtCNN C4F32 numerical reference",
    source: {
      path: "shaders/upstream/ArtCNN_C4F32.glsl",
      sha256: "f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3",
    },
    inputs: [{ role: "source", id: "model" }],
    output: {
      kind: "fixture",
      path: "validation/references/ArtCNN_C4F32.rgb16le",
      width: 74,
      height: 46,
      format: "rgb16le",
      comparison: "red",
      sha256: "140264cffd5bb8072c4743b375adbf1fb230bf440eba77401740194d0d8d4055",
    },
    oracle: { kind: "mpv-libplacebo-artcnn-f32", normalizedMacroBlocks: 8 },
    tolerances: MODEL_TOLERANCE,
  },
  {
    id: "ArtCNN_C4F32_DN",
    label: "ArtCNN C4F32 denoise numerical reference",
    source: {
      path: "shaders/upstream/ArtCNN_C4F32_DN.glsl",
      sha256: "6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c",
    },
    inputs: [{ role: "source", id: "model" }],
    output: {
      kind: "fixture",
      path: "validation/references/ArtCNN_C4F32_DN.rgb16le",
      width: 74,
      height: 46,
      format: "rgb16le",
      comparison: "red",
      sha256: "e1b74b647fbdc5f57e7f0bd10e9b08aed52ace5555546c09f743bea4b391dd8b",
    },
    oracle: { kind: "mpv-libplacebo-artcnn-f32", normalizedMacroBlocks: 8 },
    tolerances: MODEL_TOLERANCE,
  },
  {
    id: "ArtCNN_C4F32_DS",
    label: "ArtCNN C4F32 deblur numerical reference",
    source: {
      path: "shaders/upstream/ArtCNN_C4F32_DS.glsl",
      sha256: "a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e",
    },
    inputs: [{ role: "source", id: "model" }],
    output: {
      kind: "fixture",
      path: "validation/references/ArtCNN_C4F32_DS.rgb16le",
      width: 74,
      height: 46,
      format: "rgb16le",
      comparison: "red",
      sha256: "032f6acc8a5f731722be4a9bc492091fba0db9c5ba0a9354b51950a7ca31dac6",
    },
    oracle: { kind: "mpv-libplacebo-artcnn-f32", normalizedMacroBlocks: 8 },
    tolerances: MODEL_TOLERANCE,
  },
  {
    id: "filter:ssimds-reference",
    label: "SSimDownscaler numerical reference",
    source: {
      path: "shaders/upstream/SSimDownscaler.glsl",
      sha256: "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
    },
    inputs: [{ role: "source", id: "ssimds" }],
    output: {
      kind: "fixture",
      path: "validation/references/ssimds.rgb16le",
      width: 31,
      height: 19,
      format: "rgb16le",
      comparison: "rgb",
      sha256: "8f231ea725e6d5d341884b3a2e788435242cab94f742793b463a60d9c173e7b0",
    },
    oracle: { kind: "mpv-libplacebo-upstream" },
    tolerances: { rmse: 0.003, p99: 0.006, max: 0.01 },
  },
  {
    id: "filter:sharpen-reference",
    label: "Adaptive sharpen numerical reference",
    source: {
      path: "shaders/upstream/adaptive-sharpen.glsl",
      sha256: "827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4",
    },
    inputs: [{ role: "source", id: "sharpen" }],
    output: {
      kind: "fixture",
      path: "validation/references/sharpen.rgb16le",
      width: 37,
      height: 23,
      format: "rgb16le",
      comparison: "rgb",
      sha256: "4b7ad31ecb5e0d6c0a361ecde95af1ab70a65e1c574748b4a6d6d107edc7211e",
    },
    // rgba16float preserves overshoot, while the mpv window screenshot is
    // display-clamped; compare the same presented range.
    oracle: { kind: "mpv-libplacebo-upstream", presentationClamp: true },
    tolerances: { rmse: 0.012, p99: 0.04, max: 0.08 },
  },
  {
    id: "color:extract-reference",
    label: "BT.709 luma extraction numerical reference",
    source: null,
    inputs: [{ role: "source", id: "color-source" }],
    output: {
      kind: "computed",
      width: 13,
      height: 9,
      format: "rgba16float",
      comparison: "red",
    },
    oracle: { kind: "cpu-bt709-f32-f16", operation: "extract" },
    tolerances: { rmse: 0.0005, max: 0.0015 },
  },
  {
    id: "color:recombine-reference",
    label: "BT.709 luma recombination numerical reference",
    source: null,
    inputs: [
      { role: "source", id: "color-source" },
      { role: "luma", id: "color-luma" },
    ],
    output: {
      kind: "computed",
      width: 25,
      height: 17,
      format: "rgba16float",
      comparison: "rgb",
    },
    oracle: { kind: "cpu-bt709-f32-f16", operation: "recombine" },
    // Native D3D12 filtering can differ from the f32 oracle by one 1/256
    // interpolation step; keep the full-frame RMSE bound strict.
    tolerances: { rmse: 0.001, max: 0.004 },
  },
]);

const REFERENCE_CASE_INDEX = new Map(REFERENCE_CASES.map((entry) => [entry.id, entry]));

export function getReferenceCase(id) {
  const entry = REFERENCE_CASE_INDEX.get(id);
  if (!entry) throw new RangeError(`Unknown reference case: ${id}`);
  return entry;
}

function graySample(x, y) {
  const checker = (((x >> 2) + (y >> 1)) & 1) * 8191;
  const structure = x * 1489 + y * 2539 + x * x * 97 + y * y * 193 + x * y * 389;
  return 4096 + ((structure + checker) % 57344);
}

function rgbSample(x, y, channel) {
  if (channel === 0) return ((31 * x + 17 * y + 7 * x * y + 23) & 255) * 257;
  if (channel === 1) return ((11 * x + 47 * y + 5 * x * y + 71) & 255) * 257;
  return ((((x >> 2) + (y >> 1)) & 1) ? 218 : 37) * 257;
}

export function createReferenceInput(id) {
  const spec = REFERENCE_INPUTS[id];
  if (!spec) throw new RangeError(`Unknown reference input: ${id}`);
  const data = new Uint16Array(spec.width * spec.height * spec.channels);
  let offset = 0;
  for (let y = 0; y < spec.height; y++) {
    for (let x = 0; x < spec.width; x++) {
      if (spec.channels === 1) {
        data[offset++] = graySample(x, y);
      } else {
        for (let channel = 0; channel < spec.channels; channel++) {
          data[offset++] = rgbSample(x, y, channel);
        }
      }
    }
  }
  return Object.freeze({
    id: spec.id,
    width: spec.width,
    height: spec.height,
    channels: spec.channels,
    encoding: spec.encoding,
    data,
  });
}

export function encodeReferenceInput(id) {
  const input = createReferenceInput(id);
  const magic = input.channels === 1 ? "P5" : "P6";
  const header = new TextEncoder().encode(`${magic}\n${input.width} ${input.height}\n65535\n`);
  const bytes = new Uint8Array(header.length + input.data.length * 2);
  bytes.set(header);
  const view = new DataView(bytes.buffer);
  let offset = header.length;
  for (const value of input.data) {
    view.setUint16(offset, value, false);
    offset += 2;
  }
  return bytes;
}

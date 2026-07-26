// fsrcnnx-cda-priors.js — decoded-frame priors for a CDA-style temporal model.
//
// Chromium does not expose decoder motion vectors or residual coefficients for
// an existing HTMLVideoElement. This module builds codec-like substitutes on
// the ORT WebGPU device:
//   1. block SAD search against the previous decoded frame;
//   2. dense NCHW motion-vector planes;
//   3. a luma residual map after motion compensation.
//
// The outputs stay in GPUBuffer objects suitable for ort.Tensor.fromGpuBuffer().
// The first frame after a reset deliberately returns zero priors with valid=false
// and snapshots the current frame for the next recurrent step.

const DEFAULT_BLOCK_SIZE = 16;
const DEFAULT_SEARCH_RADIUS = 8;
const DEFAULT_SAMPLE_STRIDE = 4;
const DEFAULT_HISTORY_FRAMES = 25;
const MIN_BLOCK_SIZE = 4;
const MAX_BLOCK_SIZE = 32;
const MAX_SEARCH_RADIUS = 32;
const MAX_SAMPLE_STRIDE = 8;
const MAX_TIMESTAMP_GAP_SECONDS = 0.25;

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value;
}

function boundedInteger(value, fallback, minimum, maximum, label) {
  const selected = value ?? fallback;
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new Error(`${label} must be an integer from ${minimum} to ${maximum}`);
  }
  return selected;
}

function checkedProduct(label, ...values) {
  let result = 1;
  for (const value of values) {
    positiveSafeInteger(value, label);
    if (result > Math.floor(Number.MAX_SAFE_INTEGER / value)) {
      throw new Error(`${label} exceeds the safe integer range`);
    }
    result *= value;
  }
  return result;
}

function deviceLimit(device, name, fallback) {
  const value = Number(device?.limits?.[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function normalizeCdaPriorOptions(options = {}) {
  if (!options || typeof options !== "object" || Array.isArray(options)) {
    throw new TypeError("CDA prior options must be an object");
  }
  const blockSize = boundedInteger(
    options.blockSize,
    DEFAULT_BLOCK_SIZE,
    MIN_BLOCK_SIZE,
    MAX_BLOCK_SIZE,
    "CDA blockSize",
  );
  const searchRadius = boundedInteger(
    options.searchRadius,
    DEFAULT_SEARCH_RADIUS,
    0,
    MAX_SEARCH_RADIUS,
    "CDA searchRadius",
  );
  const sampleStride = boundedInteger(
    options.sampleStride,
    DEFAULT_SAMPLE_STRIDE,
    1,
    MAX_SAMPLE_STRIDE,
    "CDA sampleStride",
  );
  if (sampleStride > blockSize) {
    throw new Error("CDA sampleStride cannot exceed blockSize");
  }
  return Object.freeze({ blockSize, searchRadius, sampleStride });
}

export function planCdaPriorBuffers(width, height, options = {}) {
  positiveSafeInteger(width, "CDA prior width");
  positiveSafeInteger(height, "CDA prior height");
  const config = normalizeCdaPriorOptions(options);
  const blocksX = Math.ceil(width / config.blockSize);
  const blocksY = Math.ceil(height / config.blockSize);
  const pixels = checkedProduct("CDA prior pixel count", width, height);
  const blocks = checkedProduct("CDA prior block count", blocksX, blocksY);
  const motionBytes = checkedProduct("CDA dense motion bytes", pixels, 2, 4);
  const residualBytes = checkedProduct("CDA residual bytes", pixels, 4);
  const blockMotionBytes = checkedProduct("CDA block motion bytes", blocks, 2, 4);
  return Object.freeze({
    ...config,
    width,
    height,
    blocksX,
    blocksY,
    pixels,
    blocks,
    motionBytes,
    residualBytes,
    blockMotionBytes,
  });
}

function safeSourceKey(value) {
  if (value == null) return "";
  const text = String(value);
  return text.length <= 512 ? text : text.slice(0, 512);
}

// Pure state machine for recurrent history boundaries. It has no source-size
// policy ceiling: resets are driven by media continuity, dimensions, and the
// model's bounded training horizon.
export class CdaTemporalTracker {
  constructor({ maxHistoryFrames = DEFAULT_HISTORY_FRAMES } = {}) {
    this.maxHistoryFrames = boundedInteger(
      maxHistoryFrames,
      DEFAULT_HISTORY_FRAMES,
      1,
      10_000,
      "CDA maxHistoryFrames",
    );
    this.reset("initial");
  }

  reset(reason = "explicit") {
    this.initialized = false;
    this.lastMediaTime = null;
    this.lastPresentedFrames = null;
    this.width = 0;
    this.height = 0;
    this.sourceKey = "";
    this.framesSinceReset = 0;
    this.lastResetReason = String(reason || "explicit").slice(0, 80);
  }

  observe({
    mediaTime,
    presentedFrames,
    width,
    height,
    sourceKey = "",
    forceReset = false,
  } = {}) {
    positiveSafeInteger(width, "CDA frame width");
    positiveSafeInteger(height, "CDA frame height");
    if (!Number.isFinite(mediaTime) || mediaTime < 0) {
      throw new Error("CDA mediaTime must be a finite non-negative number");
    }
    if (!Number.isSafeInteger(presentedFrames) || presentedFrames < 0) {
      throw new Error("CDA presentedFrames must be a non-negative safe integer");
    }
    const nextSourceKey = safeSourceKey(sourceKey);
    let reason = null;
    if (forceReset) reason = "explicit";
    else if (!this.initialized) reason = this.lastResetReason || "initial";
    else if (width !== this.width || height !== this.height) reason = "dimensions";
    else if (nextSourceKey !== this.sourceKey) reason = "source";
    else if (mediaTime <= this.lastMediaTime) reason = "timestamp-backward";
    else if (mediaTime - this.lastMediaTime > MAX_TIMESTAMP_GAP_SECONDS) reason = "timestamp-gap";
    // A busy neural renderer intentionally skips presented video frames. Its
    // previous state still corresponds to the last frame it processed, so a
    // forward counter gap is not itself a discontinuity. The timestamp bound
    // above catches gaps that are too large for the decoded-prior search.
    else if (presentedFrames <= this.lastPresentedFrames) reason = "frame-counter-backward";
    else if (this.framesSinceReset >= this.maxHistoryFrames - 1) {
      reason = "history-window";
    }

    if (reason) this.framesSinceReset = 0;
    else this.framesSinceReset++;
    this.initialized = true;
    this.lastMediaTime = mediaTime;
    this.lastPresentedFrames = presentedFrames;
    this.width = width;
    this.height = height;
    this.sourceKey = nextSourceKey;
    this.lastResetReason = reason;
    return Object.freeze({
      reset: !!reason,
      reason,
      frameIndex: this.framesSinceReset,
    });
  }
}

export function buildCdaPriorShaders(options = {}) {
  const { blockSize, searchRadius, sampleStride } = normalizeCdaPriorOptions(options);
  const common = `
struct Config {
  width: u32,
  height: u32,
  blocks_x: u32,
  blocks_y: u32,
};
fn luma(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}`;
  const search = `${common}
@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var previous_frame: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> block_motion: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cfg.blocks_x || gid.y >= cfg.blocks_y) { return; }
  let origin = vec2<u32>(gid.xy) * ${blockSize}u;
  var best_score = 1e30;
  var best_motion = vec2<i32>(0, 0);
  var best_magnitude = 0x7fffffffi;
  for (var dy = -${searchRadius}i; dy <= ${searchRadius}i; dy = dy + 1i) {
    for (var dx = -${searchRadius}i; dx <= ${searchRadius}i; dx = dx + 1i) {
      var sad = 0.0;
      var count = 0u;
      for (var sy = 0u; sy < ${blockSize}u; sy = sy + ${sampleStride}u) {
        for (var sx = 0u; sx < ${blockSize}u; sx = sx + ${sampleStride}u) {
          let p = origin + vec2<u32>(sx, sy);
          if (p.x >= cfg.width || p.y >= cfg.height) { continue; }
          let q = vec2<i32>(p) + vec2<i32>(dx, dy);
          if (q.x < 0i || q.y < 0i ||
              q.x >= i32(cfg.width) || q.y >= i32(cfg.height)) {
            sad = sad + 1.0;
          } else {
            let a = luma(textureLoad(current_frame, vec2<i32>(p), 0).rgb);
            let b = luma(textureLoad(previous_frame, q, 0).rgb);
            sad = sad + abs(a - b);
          }
          count = count + 1u;
        }
      }
      let score = sad / max(1.0, f32(count));
      let magnitude = abs(dx) + abs(dy);
      if (score < best_score - 1e-7 ||
          (abs(score - best_score) <= 1e-7 && magnitude < best_magnitude)) {
        best_score = score;
        best_motion = vec2<i32>(dx, dy);
        best_magnitude = magnitude;
      }
    }
  }
  let index = gid.y * cfg.blocks_x + gid.x;
  block_motion[index] = vec2<f32>(best_motion);
}`;

  const dense = `${common}
@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var previous_frame: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> block_motion: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> dense_motion: array<f32>;
@group(0) @binding(4) var<storage, read_write> residual: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Config;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cfg.width || gid.y >= cfg.height) { return; }
  let pixel = gid.y * cfg.width + gid.x;
  let plane = cfg.width * cfg.height;
  let block = (gid.y / ${blockSize}u) * cfg.blocks_x + (gid.x / ${blockSize}u);
  let mv = block_motion[block];
  dense_motion[pixel] = mv.x;
  dense_motion[plane + pixel] = mv.y;
  let q_unclamped = vec2<i32>(gid.xy) + vec2<i32>(round(mv));
  let q = clamp(
    q_unclamped,
    vec2<i32>(0, 0),
    vec2<i32>(i32(cfg.width) - 1i, i32(cfg.height) - 1i),
  );
  let a = luma(textureLoad(current_frame, vec2<i32>(gid.xy), 0).rgb);
  let b = luma(textureLoad(previous_frame, q, 0).rgb);
  residual[pixel] = abs(a - b);
}`;

  const snapshot = `${common}
@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var previous_frame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cfg.width || gid.y >= cfg.height) { return; }
  textureStore(
    previous_frame,
    vec2<i32>(gid.xy),
    textureLoad(current_frame, vec2<i32>(gid.xy), 0),
  );
}`;
  return Object.freeze({ search, dense, snapshot });
}

function requireGpuConstants() {
  const buffer = globalThis.GPUBufferUsage;
  const texture = globalThis.GPUTextureUsage;
  if (!buffer || !texture) {
    throw new Error("WebGPU usage constants are unavailable");
  }
  return { buffer, texture };
}

function safeDestroy(resource) {
  try { resource?.destroy?.(); } catch {}
}

export class CdaPriorGenerator {
  constructor(device, options = {}) {
    if (!device || typeof device.createBuffer !== "function" ||
        typeof device.createTexture !== "function" ||
        typeof device.createComputePipeline !== "function") {
      throw new TypeError("CDA priors require a WebGPU device");
    }
    this.device = device;
    this.options = normalizeCdaPriorOptions(options);
    this.shaders = buildCdaPriorShaders(this.options);
    this.searchPipeline = null;
    this.densePipeline = null;
    this.snapshotPipeline = null;
    this.uniform = null;
    this.previous = null;
    this.blockMotion = null;
    this.motion = null;
    this.residual = null;
    this.plan = null;
    this.hasPrevious = false;
    this.disposed = false;
  }

  _ensurePipelines() {
    if (this.searchPipeline && this.densePipeline && this.snapshotPipeline) return;
    const create = (label, code) => this.device.createComputePipeline({
      label,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ label: `${label}-shader`, code }),
        entryPoint: "main",
      },
    });
    const searchPipeline = create("cda-prior-search", this.shaders.search);
    const densePipeline = create("cda-prior-dense", this.shaders.dense);
    const snapshotPipeline = create("cda-prior-snapshot", this.shaders.snapshot);
    this.searchPipeline = searchPipeline;
    this.densePipeline = densePipeline;
    this.snapshotPipeline = snapshotPipeline;
  }

  _validatePlan(plan) {
    const maxDimension = deviceLimit(this.device, "maxTextureDimension2D", 8192);
    const maxBuffer = deviceLimit(this.device, "maxBufferSize", 256 * 1024 * 1024);
    const maxBinding = deviceLimit(
      this.device,
      "maxStorageBufferBindingSize",
      128 * 1024 * 1024,
    );
    const maxGroups = deviceLimit(this.device, "maxComputeWorkgroupsPerDimension", 65_535);
    if (plan.width > maxDimension || plan.height > maxDimension) {
      throw new Error(`CDA prior dimensions exceed the device texture limit ${maxDimension}`);
    }
    if (Math.ceil(plan.width / 8) > maxGroups ||
        Math.ceil(plan.height / 8) > maxGroups ||
        Math.ceil(plan.blocksX / 8) > maxGroups ||
        Math.ceil(plan.blocksY / 8) > maxGroups) {
      throw new Error(`CDA prior dispatch exceeds the device workgroup limit ${maxGroups}`);
    }
    for (const [label, bytes] of [
      ["block motion", plan.blockMotionBytes],
      ["dense motion", plan.motionBytes],
      ["residual", plan.residualBytes],
    ]) {
      if (bytes > maxBuffer || bytes > maxBinding) {
        throw new Error(`CDA ${label} buffer exceeds the device binding limit`);
      }
    }
  }

  _releaseFrameResources() {
    for (const resource of [
      this.previous,
      this.blockMotion,
      this.motion,
      this.residual,
      this.uniform,
    ]) {
      safeDestroy(resource);
    }
    this.previous = null;
    this.blockMotion = null;
    this.motion = null;
    this.residual = null;
    this.uniform = null;
    this.plan = null;
    this.hasPrevious = false;
  }

  _ensureResources(width, height) {
    const plan = planCdaPriorBuffers(width, height, this.options);
    this._validatePlan(plan);
    if (this.plan?.width === width && this.plan?.height === height) return plan;
    this._releaseFrameResources();
    const { buffer, texture } = requireGpuConstants();
    try {
      this.uniform = this.device.createBuffer({
        label: "cda-prior-config",
        size: 16,
        usage: buffer.UNIFORM | buffer.COPY_DST,
      });
      this.blockMotion = this.device.createBuffer({
        label: "cda-prior-block-motion",
        size: plan.blockMotionBytes,
        usage: buffer.STORAGE | buffer.COPY_DST,
      });
      this.motion = this.device.createBuffer({
        label: "cda-prior-dense-motion",
        size: plan.motionBytes,
        usage: buffer.STORAGE | buffer.COPY_DST | buffer.COPY_SRC,
      });
      this.residual = this.device.createBuffer({
        label: "cda-prior-residual",
        size: plan.residualBytes,
        usage: buffer.STORAGE | buffer.COPY_DST | buffer.COPY_SRC,
      });
      this.previous = this.device.createTexture({
        label: `cda-prior-previous-${width}x${height}`,
        size: { width, height },
        format: "rgba16float",
        usage: texture.TEXTURE_BINDING | texture.STORAGE_BINDING,
      });
      this.plan = plan;
      this.hasPrevious = false;
      return plan;
    } catch (error) {
      this._releaseFrameResources();
      throw error;
    }
  }

  reset() {
    this.hasPrevious = false;
  }

  generate(currentTexture, width, height, { reset = false } = {}) {
    if (this.disposed) throw new Error("CDA prior generator is disposed");
    if (!currentTexture || typeof currentTexture.createView !== "function") {
      throw new TypeError("CDA priors require a GPU texture source");
    }
    this._ensurePipelines();
    const plan = this._ensureResources(width, height);
    if (reset) this.hasPrevious = false;
    const valid = this.hasPrevious;
    const currentView = currentTexture.createView();
    const previousView = this.previous.createView();
    this.device.queue.writeBuffer(
      this.uniform,
      0,
      new Uint32Array([plan.width, plan.height, plan.blocksX, plan.blocksY]),
    );
    const encoder = this.device.createCommandEncoder({ label: "cda-prior-frame" });
    if (valid) {
      const searchGroup = this.device.createBindGroup({
        label: "cda-prior-search-bindings",
        layout: this.searchPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: currentView },
          { binding: 1, resource: previousView },
          { binding: 2, resource: { buffer: this.blockMotion } },
          { binding: 3, resource: { buffer: this.uniform } },
        ],
      });
      const searchPass = encoder.beginComputePass({ label: "cda-prior-search-pass" });
      searchPass.setPipeline(this.searchPipeline);
      searchPass.setBindGroup(0, searchGroup);
      searchPass.dispatchWorkgroups(Math.ceil(plan.blocksX / 8), Math.ceil(plan.blocksY / 8));
      searchPass.end();

      const denseGroup = this.device.createBindGroup({
        label: "cda-prior-dense-bindings",
        layout: this.densePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: currentView },
          { binding: 1, resource: previousView },
          { binding: 2, resource: { buffer: this.blockMotion } },
          { binding: 3, resource: { buffer: this.motion } },
          { binding: 4, resource: { buffer: this.residual } },
          { binding: 5, resource: { buffer: this.uniform } },
        ],
      });
      const densePass = encoder.beginComputePass({ label: "cda-prior-dense-pass" });
      densePass.setPipeline(this.densePipeline);
      densePass.setBindGroup(0, denseGroup);
      densePass.dispatchWorkgroups(Math.ceil(plan.width / 8), Math.ceil(plan.height / 8));
      densePass.end();
    } else {
      encoder.clearBuffer(this.motion);
      encoder.clearBuffer(this.residual);
      encoder.clearBuffer(this.blockMotion);
    }

    const snapshotGroup = this.device.createBindGroup({
      label: "cda-prior-snapshot-bindings",
      layout: this.snapshotPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: currentView },
        { binding: 1, resource: previousView },
        { binding: 2, resource: { buffer: this.uniform } },
      ],
    });
    const snapshotPass = encoder.beginComputePass({ label: "cda-prior-snapshot-pass" });
    snapshotPass.setPipeline(this.snapshotPipeline);
    snapshotPass.setBindGroup(0, snapshotGroup);
    snapshotPass.dispatchWorkgroups(Math.ceil(plan.width / 8), Math.ceil(plan.height / 8));
    snapshotPass.end();
    this.device.queue.submit([encoder.finish()]);
    this.hasPrevious = true;
    return Object.freeze({
      valid,
      provider: "decoded-cda-v1",
      width: plan.width,
      height: plan.height,
      motion: this.motion,
      residual: this.residual,
      motionDims: Object.freeze([1, 2, plan.height, plan.width]),
      residualDims: Object.freeze([1, 1, plan.height, plan.width]),
    });
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this._releaseFrameResources();
    this.searchPipeline = null;
    this.densePipeline = null;
    this.snapshotPipeline = null;
    this.device = null;
  }
}

export function createCdaPriorGenerator(device, options) {
  return new CdaPriorGenerator(device, options);
}

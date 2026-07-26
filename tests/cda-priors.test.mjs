import assert from "node:assert/strict";
import test from "node:test";

import {
  CdaPriorGenerator,
  CdaSceneCutDetector,
  CdaTemporalTracker,
  buildCdaPriorShaders,
  normalizeCdaSceneCutOptions,
  normalizeCdaPriorOptions,
  planCdaPriorBuffers,
} from "../src/core/fsrcnnx-cda-priors.js";

function rgbaFrame(width, height, pixel) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [red, green, blue] = pixel(x, y);
      const offset = (y * width + x) * 4;
      data[offset] = red;
      data[offset + 1] = green;
      data[offset + 2] = blue;
      data[offset + 3] = 255;
    }
  }
  return data;
}

test("CDA prior planning has no policy resolution ceiling and emits exact NCHW sizes", () => {
  assert.deepEqual(normalizeCdaPriorOptions(), {
    blockSize: 16,
    searchRadius: 8,
    sampleStride: 4,
  });
  const plan = planCdaPriorBuffers(1920, 1080);
  assert.equal(plan.blocksX, 120);
  assert.equal(plan.blocksY, 68);
  assert.equal(plan.motionBytes, 1920 * 1080 * 2 * 4);
  assert.equal(plan.residualBytes, 1920 * 1080 * 4);
  assert.equal(plan.blockMotionBytes, 120 * 68 * 2 * 4);

  const large = planCdaPriorBuffers(12_000, 7_000, {
    blockSize: 32,
    searchRadius: 4,
    sampleStride: 8,
  });
  assert.equal(large.width, 12_000);
  assert.equal(large.height, 7_000);
  assert.throws(() => normalizeCdaPriorOptions({ sampleStride: 8, blockSize: 4 }),
    /cannot exceed/);
});

test("CDA scene-cut signatures are bounded, conservative, and resettable", () => {
  assert.deepEqual(normalizeCdaSceneCutOptions(), {
    sampleColumns: 64,
    sampleRows: 36,
    sampleDelta: 0.14,
    strongHistogramDistance: 0.42,
    histogramDistance: 0.28,
    meanDistance: 0.18,
    changedRatio: 0.65,
    strongChangedRatio: 0.45,
  });
  assert.throws(
    () => normalizeCdaSceneCutOptions({ sampleColumns: 2 }),
    /sampleColumns/,
  );

  const width = 128;
  const height = 72;
  const black = rgbaFrame(width, height, () => [0, 0, 0]);
  const white = rgbaFrame(width, height, () => [255, 255, 255]);
  const detector = new CdaSceneCutDetector();
  const first = detector.prepare(black, width, height);
  assert.equal(first.sceneCut, false);
  assert.equal(first.sampleCount, 64 * 36);
  detector.commit(first);
  const identical = detector.prepare(black, width, height);
  assert.equal(identical.sceneCut, false);
  detector.commit(identical);
  const cut = detector.prepare(white, width, height);
  assert.equal(cut.sceneCut, true);
  assert.equal(cut.histogramDistance, 1);
  assert.ok(cut.meanDistance > 0.45);
  assert.equal(cut.changedRatio, 1);
  detector.commit(cut);
  assert.throws(() => detector.commit(cut), /already consumed/);
  const reset = detector.prepare(black, width, height, { reset: true });
  assert.equal(reset.sceneCut, false);
  detector.commit(reset);
  assert.throws(
    () => detector.prepare(new Uint8Array(3), 1, 1),
    /expected 4/,
  );
});

test("CDA scene-cut signatures reject compression noise and camera-like translation", () => {
  const width = 128;
  const height = 72;
  const base = rgbaFrame(width, height, (x, y) => {
    const value = (x + y) % 2 ? 224 : 32;
    return [value, value, value];
  });
  const noisy = rgbaFrame(width, height, (x, y) => {
    const value = (x + y) % 2 ? 228 : 28;
    return [value, value, value];
  });
  const shifted = rgbaFrame(width, height, (x, y) => {
    const value = (x + y + 1) % 2 ? 224 : 32;
    return [value, value, value];
  });
  const detector = new CdaSceneCutDetector();
  const baseline = detector.prepare(base, width, height);
  detector.commit(baseline);
  const compression = detector.prepare(noisy, width, height);
  assert.equal(compression.sceneCut, false);
  detector.discard(compression);
  const translation = detector.prepare(shifted, width, height);
  assert.equal(translation.sceneCut, false);
  assert.ok(translation.meanDistance > 0.35);
  assert.ok(translation.changedRatio > 0.9);
  assert.ok(translation.histogramDistance < 0.02);
  detector.discard(translation);
});

test("CDA scene-cut signatures ignore a small overlay", () => {
  const width = 128;
  const height = 72;
  const black = rgbaFrame(width, height, () => [0, 0, 0]);
  const withOverlay = rgbaFrame(width, height, (x, y) => {
    const overlay = x >= 52 && x < 76 && y >= 28 && y < 44;
    return overlay ? [255, 255, 255] : [0, 0, 0];
  });
  const detector = new CdaSceneCutDetector();
  const baseline = detector.prepare(black, width, height);
  detector.commit(baseline);
  const overlay = detector.prepare(withOverlay, width, height);
  assert.equal(overlay.sceneCut, false);
  assert.ok(overlay.changedRatio > 0.9);
  assert.ok(overlay.histogramDistance < 0.07);
  detector.discard(overlay);
});

test("CDA scene-cut signatures establish fresh baselines after resize and reset", () => {
  const detector = new CdaSceneCutDetector();
  const darkLarge = rgbaFrame(128, 72, () => [0, 0, 0]);
  const lightSmall = rgbaFrame(96, 54, () => [255, 255, 255]);
  const darkSmall = rgbaFrame(96, 54, () => [0, 0, 0]);

  const baseline = detector.prepare(darkLarge, 128, 72);
  detector.commit(baseline);
  const resized = detector.prepare(lightSmall, 96, 54);
  assert.equal(resized.sceneCut, false);
  assert.equal(resized.histogramDistance, 0);
  detector.commit(resized);

  const changed = detector.prepare(darkSmall, 96, 54);
  assert.equal(changed.sceneCut, true);
  detector.discard(changed);
  const explicitReset = detector.prepare(darkSmall, 96, 54, { reset: true });
  assert.equal(explicitReset.sceneCut, false);
  assert.equal(explicitReset.reset, true);
  detector.commit(explicitReset);

  detector.reset();
  const coldStart = detector.prepare(lightSmall, 96, 54);
  assert.equal(coldStart.sceneCut, false);
  detector.commit(coldStart);
});

test("CDA history resets on actual temporal boundaries and its trained horizon", () => {
  const tracker = new CdaTemporalTracker({ maxHistoryFrames: 3 });
  const frame = (mediaTime, presentedFrames, extra = {}) => tracker.observe({
    mediaTime,
    presentedFrames,
    width: 320,
    height: 180,
    sourceKey: "video-a",
    ...extra,
  });
  assert.deepEqual(frame(0, 1), { reset: true, reason: "initial", frameIndex: 0 });
  assert.deepEqual(frame(1 / 30, 2), { reset: false, reason: null, frameIndex: 1 });
  assert.deepEqual(frame(2 / 30, 3), { reset: false, reason: null, frameIndex: 2 });
  assert.deepEqual(frame(3 / 30, 4), {
    reset: true,
    reason: "history-window",
    frameIndex: 0,
  });
  assert.deepEqual(frame(4 / 30, 7), {
    reset: false,
    reason: null,
    frameIndex: 1,
  }, "forward frame drops continue from the last processed frame");
  assert.equal(frame(5 / 30, 7).reason, "frame-counter-backward");
  assert.equal(frame(0.5, 8).reason, "timestamp-gap");
  assert.equal(frame(0.4, 9).reason, "timestamp-backward");
  assert.equal(frame(0.5, 10, { sourceKey: "video-b" }).reason, "source");
  assert.equal(frame(0.6, 11, { sourceKey: "video-b", width: 640 }).reason, "dimensions");
  assert.equal(frame(0.7, 12, {
    sourceKey: "video-b",
    width: 640,
    forceReset: true,
    forceResetReason: "seek",
  }).reason, "seek");
  tracker.rebase("scene-cut");
  assert.deepEqual(frame(0.8, 13, {
    sourceKey: "video-b",
    width: 640,
  }), {
    reset: false,
    reason: null,
    frameIndex: 1,
  });
});

test("CDA default history starts initializer windows at frames 0, 25, and 50", () => {
  const tracker = new CdaTemporalTracker();
  const resetFrames = [];
  for (let frame = 0; frame <= 50; frame++) {
    const boundary = tracker.observe({
      mediaTime: frame / 60,
      presentedFrames: frame + 1,
      width: 1280,
      height: 720,
      sourceKey: "history-fixture",
    });
    if (boundary.reset) resetFrames.push(frame);
  }
  assert.deepEqual(resetFrames, [0, 25, 50]);
});

test("CDA shaders use bounded block search and produce dense motion and residual planes", () => {
  const shaders = buildCdaPriorShaders({
    blockSize: 8,
    searchRadius: 3,
    sampleStride: 2,
  });
  assert.match(shaders.search, /dy = -3i/);
  assert.match(shaders.search, /sy < 8u; sy = sy \+ 2u/);
  assert.match(shaders.dense, /dense_motion\[plane \+ pixel\] = mv\.y/);
  assert.match(shaders.dense, /residual\[pixel\] = abs\(a - b\)/);
  assert.match(shaders.snapshot, /texture_storage_2d<rgba16float, write>/);
  for (const source of Object.values(shaders)) {
    assert.doesNotMatch(source, /while\s*\(/);
  }
});

function fakeGpuDevice() {
  const resources = [];
  const submissions = [];
  const writes = [];
  const passes = [];
  const makeResource = (descriptor, extra = {}) => {
    const resource = {
      descriptor,
      destroyed: false,
      destroy() { this.destroyed = true; },
      ...extra,
    };
    resources.push(resource);
    return resource;
  };
  const device = {
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: {
      writeBuffer(...args) { writes.push(args); },
      submit(commandBuffers) { submissions.push(commandBuffers); },
    },
    createShaderModule(descriptor) { return { descriptor }; },
    createComputePipeline(descriptor) {
      return {
        descriptor,
        getBindGroupLayout(index) { return { index, label: descriptor.label }; },
      };
    },
    createBuffer(descriptor) {
      return makeResource(descriptor, { size: descriptor.size });
    },
    createTexture(descriptor) {
      const texture = makeResource(descriptor);
      texture.createView = () => ({ texture });
      return texture;
    },
    createBindGroup(descriptor) { return { descriptor }; },
    createCommandEncoder(descriptor) {
      const operations = [];
      return {
        clearBuffer(buffer) { operations.push({ kind: "clear", buffer }); },
        beginComputePass(passDescriptor) {
          const pass = { descriptor: passDescriptor, dispatches: [] };
          passes.push(pass);
          operations.push({ kind: "pass", pass });
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup(index, group) { pass.group = { index, group }; },
            dispatchWorkgroups(x, y) { pass.dispatches.push([x, y]); },
            end() { pass.ended = true; },
          };
        },
        finish() { return { descriptor, operations }; },
      };
    },
  };
  return { device, resources, submissions, writes, passes };
}

test("CDA generator zero-initializes history, then runs search entirely on GPU", (t) => {
  const previousBufferUsage = globalThis.GPUBufferUsage;
  const previousTextureUsage = globalThis.GPUTextureUsage;
  globalThis.GPUBufferUsage = {
    COPY_SRC: 1,
    COPY_DST: 2,
    UNIFORM: 4,
    STORAGE: 8,
  };
  globalThis.GPUTextureUsage = {
    TEXTURE_BINDING: 1,
    STORAGE_BINDING: 2,
  };
  t.after(() => {
    globalThis.GPUBufferUsage = previousBufferUsage;
    globalThis.GPUTextureUsage = previousTextureUsage;
  });

  const gpu = fakeGpuDevice();
  const generator = new CdaPriorGenerator(gpu.device, {
    blockSize: 8,
    searchRadius: 2,
    sampleStride: 2,
  });
  const current = { createView: () => ({ current: true }) };
  const first = generator.generate(current, 32, 24);
  assert.equal(first.valid, false);
  assert.equal(first.provider, "decoded-cda-v1");
  assert.deepEqual(first.motionDims, [1, 2, 24, 32]);
  assert.equal(gpu.submissions.length, 1);
  assert.equal(gpu.submissions[0][0].operations.filter(({ kind }) => kind === "clear").length, 3);
  assert.equal(gpu.passes.length, 1, "the reset frame only snapshots");

  const second = generator.generate(current, 32, 24);
  assert.equal(second.valid, true);
  assert.equal(gpu.submissions.length, 2);
  assert.equal(gpu.passes.length, 4, "search, dense pack, and snapshot are submitted");
  assert.equal(gpu.writes.length, 2);
  assert.strictEqual(first.motion, second.motion);

  const oldResources = gpu.resources.slice();
  const resized = generator.generate(current, 64, 24);
  assert.equal(resized.valid, false, "a resize starts a new recurrent sequence");
  assert.ok(oldResources.every((resource) =>
    !resource.destroy || resource.destroyed || !resource.descriptor?.size),
  );
  generator.dispose();
  assert.throws(() => generator.generate(current, 64, 24), /disposed/);
});

test("CDA pipeline construction publishes atomically and retries after failure", (t) => {
  const previousBufferUsage = globalThis.GPUBufferUsage;
  const previousTextureUsage = globalThis.GPUTextureUsage;
  globalThis.GPUBufferUsage = {
    COPY_SRC: 1,
    COPY_DST: 2,
    UNIFORM: 4,
    STORAGE: 8,
  };
  globalThis.GPUTextureUsage = {
    TEXTURE_BINDING: 1,
    STORAGE_BINDING: 2,
  };
  t.after(() => {
    globalThis.GPUBufferUsage = previousBufferUsage;
    globalThis.GPUTextureUsage = previousTextureUsage;
  });

  const gpu = fakeGpuDevice();
  const createPipeline = gpu.device.createComputePipeline;
  let remainingFailure = 1;
  gpu.device.createComputePipeline = (descriptor) => {
    if (descriptor.label === "cda-prior-dense" && remainingFailure-- > 0) {
      throw new Error("injected dense pipeline failure");
    }
    return createPipeline(descriptor);
  };
  const generator = new CdaPriorGenerator(gpu.device);
  const current = { createView: () => ({ current: true }) };
  assert.throws(
    () => generator.generate(current, 16, 16),
    /injected dense pipeline failure/,
  );
  assert.equal(generator.searchPipeline, null);
  assert.equal(generator.densePipeline, null);
  assert.equal(generator.snapshotPipeline, null);

  const result = generator.generate(current, 16, 16);
  assert.equal(result.valid, false);
  assert.ok(generator.searchPipeline);
  assert.ok(generator.densePipeline);
  assert.ok(generator.snapshotPipeline);
  generator.dispose();
});

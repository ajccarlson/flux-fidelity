import assert from "node:assert/strict";
import test from "node:test";

import {
  CdaPriorGenerator,
  CdaTemporalTracker,
  buildCdaPriorShaders,
  normalizeCdaPriorOptions,
  planCdaPriorBuffers,
} from "../src/core/fsrcnnx-cda-priors.js";

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
  }).reason, "explicit");
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

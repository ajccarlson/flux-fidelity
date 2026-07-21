import test from "node:test";
import assert from "node:assert/strict";

import { GpuInterp } from "../fsrcnnx-rife-gpu.js";

function makeResource(description, kind) {
  const extent = Array.isArray(description.size) ? description.size : null;
  return {
    ...description,
    kind,
    width: extent?.[0],
    height: extent?.[1],
    destroyed: false,
    destroy() { this.destroyed = true; },
    createView() { return { texture: this }; },
  };
}

function makeDevice(name) {
  const events = {
    calls: 0,
    destroys: 0,
    fences: 0,
    writes: [],
    submissions: [],
  };
  let pipelineId = 0;
  const queue = {
    writeBuffer(buffer, _offset, data) {
      events.calls++;
      events.writes.push({ buffer, data: data.slice(0) });
    },
    submit(commands) {
      events.calls++;
      events.submissions.push(commands);
    },
    onSubmittedWorkDone() {
      events.fences++;
      return Promise.resolve();
    },
  };
  const makePipeline = (kind) => {
    const pipeline = {
      name: `${name}-${kind}-${++pipelineId}`,
      getBindGroupLayout: () => pipeline,
    };
    return pipeline;
  };
  return {
    name,
    events,
    queue,
    destroy() { events.destroys++; },
    createSampler() { return { name: `${name}-sampler` }; },
    createShaderModule({ code }) { return { code }; },
    createRenderPipeline() { return makePipeline("render"); },
    createComputePipeline() { return makePipeline("compute"); },
    createBuffer(description) { return makeResource(description, "buffer"); },
    createTexture(description) { return makeResource(description, "texture"); },
    createBindGroup(description) { events.calls++; return description; },
    createCommandEncoder() {
      events.calls++;
      const command = {};
      return {
        beginComputePass() {
          return {
            setPipeline(pipeline) { command.pipeline = pipeline; },
            setBindGroup(_slot, bindGroup) { command.bindGroup = bindGroup; },
            dispatchWorkgroups(...dimensions) { command.dispatch = dimensions; },
            end() {},
          };
        },
        finish() { return command; },
      };
    },
  };
}

function installWebGpuGlobals(t) {
  const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const bufferUsageDescriptor = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const textureUsageDescriptor = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");

  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu: { getPreferredCanvasFormat: () => "rgba8unorm" } },
  });
  globalThis.GPUBufferUsage = {
    UNIFORM: 1,
    COPY_DST: 2,
    STORAGE: 4,
    COPY_SRC: 8,
  };
  globalThis.GPUTextureUsage = {
    TEXTURE_BINDING: 1,
    RENDER_ATTACHMENT: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    STORAGE_BINDING: 16,
  };

  t.after(() => {
    if (navigatorDescriptor) Object.defineProperty(globalThis, "navigator", navigatorDescriptor);
    else delete globalThis.navigator;
    if (bufferUsageDescriptor) Object.defineProperty(globalThis, "GPUBufferUsage", bufferUsageDescriptor);
    else delete globalThis.GPUBufferUsage;
    if (textureUsageDescriptor) Object.defineProperty(globalThis, "GPUTextureUsage", textureUsageDescriptor);
    else delete globalThis.GPUTextureUsage;
  });
}

test("RIFE GPU interpolation survives capture resize while inference is pending", async (t) => {
  installWebGpuGlobals(t);

  const originalDevice = makeDevice("original");
  const poisonDevice = makeDevice("replacement");
  let inputDisposed = 0;
  let tensorInputBuffer = null;
  let tensorInputDimensions = null;
  const ort = {
    Tensor: {
      fromGpuBuffer(buffer, options) {
        tensorInputBuffer = buffer;
        tensorInputDimensions = options.dims;
        return { dispose() { inputDisposed++; } };
      },
    },
  };
  const gpu = new GpuInterp({
    log() {},
    warn(...parts) { throw new Error(parts.join(" ")); },
  });
  assert.equal(await gpu.init(originalDevice, ort), true);

  gpu._size(80, 72, 8, 7, 1);
  gpu._frames = 2;
  const oldGeneration = {
    prevTex: gpu.prevTex,
    curTex: gpu.curTex,
    inBuf: gpu.inBuf,
  };
  const capturedResources = {
    compPipe: gpu.compPipe,
    compParams: gpu.compParams,
    sampler: gpu.sampler,
  };

  let enterRun;
  let resolveRun;
  const runEntered = new Promise((resolve) => { enterRun = resolve; });
  let outputDisposed = 0;
  const session = {
    run(feeds) {
      assert.ok(feeds.input);
      enterRun();
      return new Promise((resolve) => { resolveRun = resolve; });
    },
  };

  const tween = gpu.interpolateToPooledTex(
    session,
    { inputName: "input", outputName: "output" },
    0,
    0,
    0.5,
    true,
  );
  await runEntered;
  assert.equal(gpu._activeOps, 1);
  assert.equal(tensorInputBuffer, oldGeneration.inBuf);
  assert.deepEqual(tensorInputDimensions, [1, 7, 72, 80]);

  // Model a capture observing a new source size while ORT still owns the old
  // input generation. Retirement must not fence or destroy it yet.
  gpu._size(128, 96, 8, 7, 1);
  assert.equal(originalDevice.events.fences, 0);
  assert.equal(oldGeneration.prevTex.destroyed, false);
  assert.equal(oldGeneration.curTex.destroyed, false);
  assert.equal(oldGeneration.inBuf.destroyed, false);

  const replacementState = {
    device: gpu.device,
    ort: gpu.ort,
    compPipe: gpu.compPipe,
    compParams: gpu.compParams,
    sampler: gpu.sampler,
    inBuf: gpu.inBuf,
  };
  // Poison all mutable post-run references. Correct code uses only the resource
  // snapshot captured before awaiting session.run().
  gpu.device = poisonDevice;
  gpu.ort = { Tensor: { fromGpuBuffer() { throw new Error("mutable ORT used"); } } };
  gpu.compPipe = { getBindGroupLayout() { throw new Error("mutable pipeline used"); } };
  gpu.compParams = makeResource({ label: "replacement-comp-params" }, "buffer");
  gpu.sampler = { name: "replacement-sampler" };
  gpu.inBuf = makeResource({ label: "replacement-input" }, "buffer");

  resolveRun({
    output: {
      gpuBuffer: makeResource({ label: "ort-output" }, "buffer"),
      dispose() { outputDisposed++; },
    },
  });
  const result = await tween;

  assert.ok(result);
  assert.equal(result._w, 80);
  assert.equal(result._h, 72);
  assert.equal(result._gpuInterpDevice, originalDevice);
  assert.equal(poisonDevice.events.calls, 0);

  const compositeWrite = originalDevice.events.writes.find(
    ({ buffer }) => buffer === capturedResources.compParams,
  );
  assert.ok(compositeWrite);
  const dimensions = new DataView(compositeWrite.data);
  assert.deepEqual([
    dimensions.getUint32(0, true),
    dimensions.getUint32(4, true),
    dimensions.getUint32(8, true),
    dimensions.getUint32(12, true),
    dimensions.getUint32(16, true),
    dimensions.getUint32(20, true),
  ], [80, 72, 80, 72, 80, 72]);

  const compositeSubmission = originalDevice.events.submissions
    .flat()
    .find(({ pipeline }) => pipeline === capturedResources.compPipe);
  assert.ok(compositeSubmission);
  assert.equal(compositeSubmission.bindGroup.entries[0].resource, capturedResources.sampler);
  assert.equal(compositeSubmission.bindGroup.entries[1].resource.texture, oldGeneration.prevTex);
  assert.equal(compositeSubmission.bindGroup.entries[2].resource.texture, oldGeneration.curTex);

  await Promise.allSettled([...gpu._retirements]);
  assert.equal(oldGeneration.prevTex.destroyed, true);
  assert.equal(oldGeneration.curTex.destroyed, true);
  assert.equal(oldGeneration.inBuf.destroyed, true);
  assert.ok(originalDevice.events.fences >= 1);
  assert.equal(inputDisposed, 1);
  assert.equal(outputDisposed, 1);

  Object.assign(gpu, replacementState);
  gpu.releaseTex(result);
  await gpu.destroy();
});

test("RIFE GPU interpolation rejects an explicit pair from mixed resize generations", async (t) => {
  installWebGpuGlobals(t);

  const device = makeDevice("pair-guard");
  const gpu = new GpuInterp({
    log() {},
    warn(...parts) { throw new Error(parts.join(" ")); },
  });
  const ort = {
    Tensor: {
      fromGpuBuffer() { throw new Error("mismatched pair reached tensor creation"); },
    },
  };
  assert.equal(await gpu.init(device, ort), true);
  gpu._size(128, 96, 8, 7, 1);

  const oldFrame = device.createTexture({ size: [80, 72] });
  oldFrame._w = 80;
  oldFrame._h = 72;
  const newFrame = device.createTexture({ size: [128, 96] });
  newFrame._w = 128;
  newFrame._h = 96;
  let runs = 0;
  const result = await gpu.interpolateToPooledTex(
    { run() { runs++; return {}; } },
    { inputName: "input", outputName: "output" },
    0,
    0,
    0.5,
    true,
    oldFrame,
    newFrame,
  );

  assert.equal(result, null);
  assert.equal(runs, 0);
  assert.equal(gpu._activeOps, 0);
  await gpu.destroy();
});

test("GpuInterp destroys only a standalone device it requested itself", async (t) => {
  installWebGpuGlobals(t);

  const sharedDevice = makeDevice("shared");
  const shared = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await shared.init(sharedDevice, null), true);
  await shared.destroy();
  assert.equal(sharedDevice.events.destroys, 0);

  const ownedDevice = makeDevice("owned");
  navigator.gpu.requestAdapter = async () => ({
    requestDevice: async () => ownedDevice,
  });
  const standalone = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await standalone.init(null, null), true);
  await standalone.destroy();
  assert.equal(ownedDevice.events.destroys, 1);
});

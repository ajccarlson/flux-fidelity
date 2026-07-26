import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_GPU_FRAME_BUDGET_BYTES,
  DEFAULT_GPU_INPUT_BUDGET_BYTES,
  DEFAULT_GPU_POOL_BUDGET_BYTES,
  GpuInterp,
  GpuResourceLimitError,
} from "../src/core/fsrcnnx-rife-gpu.js";

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

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

function makeDevice(name, { limits = {}, fence = null } = {}) {
  const lost = deferred();
  const events = {
    calls: 0,
    destroys: 0,
    fences: 0,
    writes: [],
    submissions: [],
    textures: [],
    buffers: [],
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
      return fence?.promise || Promise.resolve();
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
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
      ...limits,
    },
    lost: lost.promise,
    lose(info = { reason: "unknown", message: "test loss" }) { lost.resolve(info); },
    events,
    queue,
    destroy() { events.destroys++; lost.resolve({ reason: "destroyed", message: "intentional test destroy" }); },
    createSampler() { return { name: `${name}-sampler` }; },
    createShaderModule({ code }) { return { code }; },
    createRenderPipeline() { return makePipeline("render"); },
    createComputePipeline() { return makePipeline("compute"); },
    createBuffer(description) {
      const resource = makeResource(description, "buffer");
      events.buffers.push(resource);
      return resource;
    },
    createTexture(description) {
      const resource = makeResource(description, "texture");
      events.textures.push(resource);
      return resource;
    },
    importExternalTexture({ source }) { return { source }; },
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
        beginRenderPass() {
          return {
            setPipeline(pipeline) { command.pipeline = pipeline; },
            setBindGroup(_slot, bindGroup) { command.bindGroup = bindGroup; },
            draw(...dimensions) { command.draw = dimensions; },
            end() {},
          };
        },
        copyTextureToTexture(source, destination, size) { command.copy = { source, destination, size }; },
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

test("GPU source flush forgets the retained pair without reallocating textures", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("flush");
  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(device, { Tensor: {} }), true);
  gpu._ensureFrameSize(64, 48, 8, 7);
  const before = [gpu.prevTex, gpu.curTex];
  gpu._frames = 3;
  assert.equal(gpu.hasPrev(), true);

  gpu.resetFrames();

  assert.equal(gpu.hasPrev(), false);
  assert.deepEqual([gpu.prevTex, gpu.curTex], before);
  await gpu.destroy();
});

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
  let sharedLosses = 0;
  const shared = new GpuInterp({
    log() {}, warn() {}, onDeviceLost() { sharedLosses++; },
  });
  assert.equal(await shared.init(sharedDevice, null), true);
  const lossObserver = shared._deviceLossObserver;
  assert.equal(lossObserver.target, shared);
  const checkedOut = shared._acquireTex(32, 24);
  assert.equal(checkedOut._gpuInterpOwner, shared);
  await shared.destroy();
  assert.equal(sharedDevice.events.destroys, 0);
  assert.equal(checkedOut.destroyed, true);
  assert.equal(checkedOut._gpuInterpOwner, null,
    "a late pooled-texture reference must not retain the retired helper");
  assert.equal(checkedOut._gpuInterpDevice, null,
    "a late pooled-texture reference must not retain the retired shared device");
  assert.equal(checkedOut._refs, 0);
  assert.equal(lossObserver.target, null,
    "the never-settled shared-device loss promise must detach from the helper");
  assert.equal(shared._deviceLossObserver, null);
  assert.equal(shared.onDeviceLost, null);
  sharedDevice.lose({ message: "late shared loss" });
  await Promise.resolve();
  assert.equal(sharedLosses, 0, "a retired helper must not receive late shared-device loss");
  for (const field of [
    "sampler", "blitPipe", "blit2dPipe", "packPipe", "compPipe", "blendPipe",
    "presentPipe", "canvasCtx", "_presentCanvas", "device", "ort",
  ]) assert.equal(shared[field], null, `${field} should be released`);

  const ownedDevice = makeDevice("owned");
  navigator.gpu.requestAdapter = async () => ({
    requestDevice: async () => ownedDevice,
  });
  const standalone = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await standalone.init(null, null), true);
  await standalone.destroy();
  assert.equal(ownedDevice.events.destroys, 1);
});

test("GpuInterp standalone initialization is single-flight", async (t) => {
  installWebGpuGlobals(t);
  const adapterGate = deferred();
  const deviceGate = deferred();
  const ownedDevice = makeDevice("single-flight");
  let adapterRequests = 0;
  let deviceRequests = 0;
  navigator.gpu.requestAdapter = () => {
    adapterRequests++;
    return adapterGate.promise;
  };

  const gpu = new GpuInterp({ log() {}, warn() {} });
  const first = gpu.init(null, null);
  const second = gpu.init(null, null);
  assert.equal(adapterRequests, 1);

  adapterGate.resolve({
    requestDevice() {
      deviceRequests++;
      return deviceGate.promise;
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(deviceRequests, 1);
  deviceGate.resolve(ownedDevice);

  assert.deepEqual(await Promise.all([first, second]), [true, true]);
  assert.equal(gpu.device, ownedDevice);
  await gpu.destroy();
  assert.equal(ownedDevice.events.destroys, 1);
});

test("GpuInterp destruction fences a pending owned-device request", async (t) => {
  installWebGpuGlobals(t);
  const deviceGate = deferred();
  const lateDevice = makeDevice("late-owned-device");
  let deviceRequests = 0;
  navigator.gpu.requestAdapter = async () => ({
    requestDevice() {
      deviceRequests++;
      return deviceGate.promise;
    },
  });

  const gpu = new GpuInterp({ log() {}, warn() {} });
  const initialization = gpu.init(null, null);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(deviceRequests, 1);

  let retirementSettled = false;
  const retirement = gpu.destroy().then(() => { retirementSettled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(retirementSettled, false);

  deviceGate.resolve(lateDevice);
  assert.equal(await initialization, false);
  await retirement;
  assert.equal(lateDevice.events.destroys, 1);
  assert.equal(gpu.device, null);
  assert.equal(gpu.ready, false);
});

test("GpuInterp rejects an already-lost device before publishing readiness", async (t) => {
  installWebGpuGlobals(t);
  const lostDevice = makeDevice("already-lost");
  lostDevice.lost = Promise.resolve({ reason: "destroyed", message: "already lost" });
  navigator.gpu.requestAdapter = async () => ({ requestDevice: async () => lostDevice });

  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(null, null), false);
  await gpu.destroy();
  assert.equal(gpu.ready, false);
  assert.equal(gpu.device, null);
  assert.equal(lostDevice.events.destroys, 1,
    "the unpublished standalone device remains owned and is destroyed once");
});

test("standalone blend capture never allocates a RIFE inference buffer", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("blend-only", {
    limits: { maxStorageBufferBindingSize: 64, maxBufferSize: 64 },
  });
  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(device, null), true);
  const uniformBuffers = device.events.buffers.length;

  const captured = gpu.captureToPooled({ videoWidth: 96, videoHeight: 54 }, 8, 7);
  assert.ok(captured);
  assert.equal(gpu.inBuf, undefined);
  assert.equal(device.events.buffers.length, uniformBuffers);
  assert.equal(device.events.buffers.some(({ label }) => label?.startsWith("rife-inBuf-")), false);

  gpu.releaseTex(captured);
  await gpu.destroy();
});

test("GpuInterp resize is atomic across frame and inference allocation failures", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("atomic-resize");
  const ort = { Tensor: { fromGpuBuffer() {} } };
  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(device, ort), true);
  gpu._size(80, 72, 8, 7, 1);
  const old = {
    prev: gpu.prevTex,
    cur: gpu.curTex,
    input: gpu.inBuf,
    dimensions: [gpu._w, gpu._h, gpu._padW, gpu._padH],
  };

  const createTexture = device.createTexture.bind(device);
  let resizeTextureCalls = 0;
  device.createTexture = (description) => {
    resizeTextureCalls++;
    if (resizeTextureCalls === 2) throw new Error("injected frame allocation failure");
    return createTexture(description);
  };
  const failedFrameTextureStart = device.events.textures.length;
  assert.throws(() => gpu._size(128, 96, 8, 7, 1), /injected frame allocation failure/);
  const partialFrameCandidates = device.events.textures.slice(failedFrameTextureStart);
  assert.equal(partialFrameCandidates.length, 1);
  assert.equal(partialFrameCandidates[0].destroyed, true);
  assert.equal(gpu.prevTex, old.prev);
  assert.equal(gpu.curTex, old.cur);
  assert.equal(gpu.inBuf, old.input);
  assert.deepEqual([gpu._w, gpu._h, gpu._padW, gpu._padH], old.dimensions);
  device.createTexture = createTexture;

  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (description) => {
    if (description.label?.startsWith("rife-inBuf-")) throw new Error("injected input allocation failure");
    return createBuffer(description);
  };
  const textureCount = device.events.textures.length;
  assert.throws(() => gpu._size(128, 96, 8, 7, 1), /injected input allocation failure/);

  const candidates = device.events.textures.slice(textureCount);
  assert.equal(candidates.length, 2);
  assert.ok(candidates.every(({ destroyed }) => destroyed));
  assert.equal(gpu.prevTex, old.prev);
  assert.equal(gpu.curTex, old.cur);
  assert.equal(gpu.inBuf, old.input);
  assert.deepEqual([gpu._w, gpu._h, gpu._padW, gpu._padH], old.dimensions);
  assert.equal(old.prev.destroyed, false);
  assert.equal(old.cur.destroyed, false);
  assert.equal(old.input.destroyed, false);

  device.createBuffer = createBuffer;
  await gpu.destroy();
});

test("failed standalone initialization destroys its requested device", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("failed-init");
  device.createComputePipeline = () => { throw new Error("injected pipeline failure"); };
  navigator.gpu.requestAdapter = async () => ({ requestDevice: async () => device });
  const gpu = new GpuInterp({ log() {}, warn() {} });

  assert.equal(await gpu.init(null, null), false);
  assert.equal(device.events.destroys, 1);
  assert.equal(gpu.device, null);
  assert.equal(gpu.ready, false);
});

test("GpuInterp validates texture and storage limits before allocation", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("limits", {
    limits: {
      maxTextureDimension2D: 128,
      maxBufferSize: 4096,
      maxStorageBufferBindingSize: 4096,
    },
  });
  const ort = { Tensor: { fromGpuBuffer() {} } };
  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(device, ort), true);
  const textureCount = device.events.textures.length;
  const bufferCount = device.events.buffers.length;

  assert.throws(() => gpu._size(129, 64, 8, 7, 1), GpuResourceLimitError);
  assert.throws(() => gpu._size(64, 64, 8, 7, 1), GpuResourceLimitError);
  assert.equal(device.events.textures.length, textureCount);
  assert.equal(device.events.buffers.length, bufferCount);
  assert.deepEqual([gpu._w, gpu._h, gpu._padW, gpu._padH], [0, 0, 0, 0]);
  await gpu.destroy();
});

test("GpuInterp defaults admit high-resolution sources within advertised device limits", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("high-resolution-defaults", {
    limits: {
      maxBufferSize: 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
    },
  });
  const gpu = new GpuInterp({ log() {}, warn() {} });
  assert.equal(await gpu.init(device, { Tensor: { fromGpuBuffer() {} } }), true);

  const plan = gpu._size(8192, 8192, 8, 7, 0.5);
  assert.deepEqual(plan, { padW: 4096, padH: 4096 });
  assert.equal(gpu._activeFrameBytes, 8192 * 8192 * 4 * 2,
    "the persistent frame pair exceeds the former fixed default");
  assert.equal(gpu._activeInputBytes, 7 * 4096 * 4096 * 4,
    "the model input exceeds the former fixed default");

  const pooled = [
    gpu._acquireTex(8192, 8192),
    gpu._acquireTex(8192, 8192),
    gpu._acquireTex(8192, 8192),
  ];
  assert.ok(gpu._pooledBytesInUse() > 512 * 1024 * 1024,
    "the live pool exceeds the former fixed default");
  for (const texture of pooled) gpu.releaseTex(texture);
  await gpu.destroy();
});

test("GpuInterp bounds pooled textures by aggregate bytes before allocation", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("pool-budget");
  const frameBytes = 64 * 48 * 4;
  const gpu = new GpuInterp({ log() {}, warn() {}, maxPoolBytes: frameBytes * 2 });
  assert.equal(await gpu.init(device, null), true);

  const first = gpu._acquireTex(64, 48);
  const second = gpu._acquireTex(64, 48);
  const allocated = device.events.textures.length;
  assert.throws(() => gpu._acquireTex(64, 48), (error) =>
    error instanceof GpuResourceLimitError && error.details.resource === "texture-pool");
  assert.equal(device.events.textures.length, allocated, "a rejected pool growth must not create a texture");

  gpu.releaseTex(first);
  assert.equal(gpu._acquireTex(64, 48), first, "a released same-size texture should be recycled");
  gpu.releaseTex(first);
  gpu.releaseTex(second);
  await gpu.destroy();

  assert.throws(() => new GpuInterp({ maxPoolBytes: 0 }), RangeError);
  assert.throws(() => new GpuInterp({ maxFrameBytes: 0 }), RangeError);
  assert.throws(() => new GpuInterp({ maxInputBytes: 0 }), RangeError);
  assert.equal(DEFAULT_GPU_POOL_BUDGET_BYTES, Number.MAX_SAFE_INTEGER);
  assert.equal(DEFAULT_GPU_FRAME_BUDGET_BYTES, Number.MAX_SAFE_INTEGER);
  assert.equal(DEFAULT_GPU_INPUT_BUDGET_BYTES, Number.MAX_SAFE_INTEGER);
});

test("GpuInterp evicts every stale free texture and retries after fenced retirement", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("pool-resize");
  const gpu = new GpuInterp({ log() {}, warn() {}, maxPoolBytes: 100 });
  assert.equal(await gpu.init(device, null), true);

  const stale = [gpu._acquireTex(4, 2), gpu._acquireTex(4, 2), gpu._acquireTex(4, 2)];
  for (const texture of stale) gpu.releaseTex(texture);
  assert.equal(gpu._pool.length, 3);

  assert.throws(() => gpu._acquireTex(10, 2), (error) =>
    error instanceof GpuResourceLimitError && error.details.requested === 176 && error.details.transient === true);
  assert.equal(gpu._pool.length, 0, "all incompatible zero-reference textures should retire together");
  assert.equal(gpu._retiringPooledBytes, 96);
  for (const texture of stale) {
    assert.equal(texture._gpuInterpOwner, null);
    assert.equal(texture._gpuInterpDevice, null);
    assert.equal(texture._refs, 0);
  }
  gpu.retainTex(stale[0]);
  gpu.releaseTex(stale[0]);
  assert.equal(stale[0]._refs, 0, "late references cannot revive a retired texture");
  assert.equal(gpu._pool.includes(stale[0]), false,
    "a texture awaiting physical destruction cannot re-enter the live pool");
  await new Promise((resolve) => setImmediate(resolve));

  const replacement = gpu._acquireTex(10, 2);
  assert.equal(replacement._gpuInterpBytes, 80);
  assert.equal(stale.every((texture) => texture.destroyed), true);
  gpu.releaseTex(replacement);
  await gpu.destroy();
});

test("GpuInterp counts textures behind unresolved fences against the pool budget", async (t) => {
  installWebGpuGlobals(t);
  const fence = deferred();
  const device = makeDevice("pool-fence", { fence });
  const gpu = new GpuInterp({ log() {}, warn() {}, maxPoolBytes: 40 });
  assert.equal(await gpu.init(device, null), true);

  const first = gpu._acquireTex(4, 2); // 32 bytes
  gpu.releaseTex(first);
  for (let attempt = 0; attempt < 20; attempt++) {
    assert.throws(() => gpu._acquireTex(5, 2), GpuResourceLimitError);
  }
  assert.equal(device.events.textures.length, 1, "pending destruction must not create budget headroom");
  assert.equal(gpu._retiringPooledBytes, 32);
  assert.equal(first.destroyed, false);

  fence.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  const second = gpu._acquireTex(5, 2);
  assert.equal(device.events.textures.length, 2);
  gpu.releaseTex(second);
  await gpu.destroy();
});

test("GpuInterp rejects the persistent frame pair before texture allocation", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("frame-budget");
  const gpu = new GpuInterp({ log() {}, warn() {}, maxFrameBytes: 100 });
  assert.equal(await gpu.init(device, null), true);
  const before = device.events.textures.length;

  assert.throws(() => gpu._ensureFrameSize(4, 4, 8, 7), (error) =>
    error instanceof GpuResourceLimitError && error.details.resource === "frame-textures" && error.details.transient === false);
  assert.equal(device.events.textures.length, before);
  assert.deepEqual([gpu._w, gpu._h, gpu.prevTex, gpu.curTex], [0, 0, undefined, undefined]);
  await gpu.destroy();
});

test("GpuInterp counts retired frame pairs until their queue fence resolves", async (t) => {
  installWebGpuGlobals(t);
  const fence = deferred();
  const device = makeDevice("frame-fence", { fence });
  const gpu = new GpuInterp({ log() {}, warn() {}, maxFrameBytes: 160 });
  assert.equal(await gpu.init(device, null), true);

  gpu._ensureFrameSize(4, 2, 8, 7); // 64 active bytes
  gpu._ensureFrameSize(5, 2, 8, 7); // 80 active + 64 retiring
  assert.equal(gpu._activeFrameBytes, 80);
  assert.equal(gpu._retiringFrameBytes, 64);
  const allocated = device.events.textures.length;

  for (let attempt = 0; attempt < 20; attempt++) {
    assert.throws(() => gpu._ensureFrameSize(4, 2, 8, 7), (error) =>
      error instanceof GpuResourceLimitError && error.details.requested === 208 && error.details.transient === true);
  }
  assert.equal(device.events.textures.length, allocated);
  assert.equal(device.events.textures.filter((texture) => !texture.destroyed).length, 4);

  fence.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  gpu._ensureFrameSize(4, 2, 8, 7);
  assert.equal(gpu._activeFrameBytes, 64);
  await gpu.destroy();
});

test("GpuInterp counts retired inference inputs until their queue fence resolves", async (t) => {
  installWebGpuGlobals(t);
  const fence = deferred();
  const device = makeDevice("input-fence", { fence });
  const gpu = new GpuInterp({
    log() {},
    warn() {},
    maxFrameBytes: 1024,
    maxInputBytes: 100000,
  });
  assert.equal(await gpu.init(device, { Tensor: { fromGpuBuffer() {} } }), true);

  // _inferencePlan clamps each axis to 64, so vary channels to obtain distinct
  // 16 KiB and 32 KiB buffers under a compact deterministic test budget.
  gpu._size(4, 2, 1, 1, 1); // 16,384 active bytes
  gpu._size(5, 2, 1, 2, 1); // 32,768 active + 16,384 retiring
  assert.equal(gpu._activeInputBytes, 32768);
  assert.equal(gpu._retiringInputBytes, 16384);
  const allocated = device.events.buffers.length;

  assert.throws(() => gpu._size(4, 2, 1, 4, 1), (error) =>
    error instanceof GpuResourceLimitError && error.details.resource === "inference-inputs" && error.details.transient === true);
  assert.equal(device.events.buffers.length, allocated);

  fence.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  gpu._size(4, 2, 1, 4, 1);
  assert.equal(gpu._activeInputBytes, 65536);
  await gpu.destroy();
});

test("GpuInterp device-loss notification is identity guarded", async (t) => {
  installWebGpuGlobals(t);
  const device = makeDevice("loss");
  const losses = [];
  const gpu = new GpuInterp({
    log() {},
    warn() {},
    onDeviceLost: (owner, info) => losses.push({ info, owner }),
  });
  assert.equal(await gpu.init(device, null), true);
  gpu._ensureFrameSize(64, 48, 8, 7);
  gpu._handleDeviceLost({}, { message: "stale" });
  assert.equal(losses.length, 0);

  const info = { reason: "unknown", message: "adapter reset" };
  device.lose(info);
  await Promise.resolve();
  await gpu.destroy();
  assert.deepEqual(losses, [{ info, owner: device }]);
  assert.equal(gpu.ready, false);
  assert.equal(device.events.destroys, 0, "shared device ownership remains external");
});

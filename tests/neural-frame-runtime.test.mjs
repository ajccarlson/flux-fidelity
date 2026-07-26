import assert from "node:assert/strict";
import test from "node:test";

import {
  createNeuralFrameSession,
  NEURAL_FRAME_CHANNEL,
  serializeNeuralFrameError,
  startNeuralFrameRuntime,
} from "../src/frame/neural-frame-runtime.js";

const FRAME_CAPABILITY = "1234567890abcdef1234567890abcdef1234567890abcdef";

function capabilityRuntime(messages, {
  parentOrigin = "https://video.example",
  opaqueParent = false,
  ok = true,
} = {}) {
  return {
    sendMessage(message, callback) {
      messages.push(message);
      callback?.(ok
        ? { ok: true, parentOrigin, opaqueParent }
        : { ok: false });
    },
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    },
    removeEventListener(type, listener) {
      listeners.get(type)?.delete(listener);
    },
    emit(type, event) {
      for (const listener of [...(listeners.get(type) || [])]) listener(event);
    },
  };
}

function texture(label, events) {
  return {
    label,
    destroyed: 0,
    createView() { return { texture: label }; },
    destroy() {
      this.destroyed++;
      events.push(`destroy:${label}`);
    },
  };
}

function uploadCanvasHarness({
  width = 1,
  height = 1,
  pixels,
} = {}) {
  const calls = [];
  const context = {
    globalCompositeOperation: "source-over",
    drawImage(...args) {
      calls.push({ type: "draw-image", args });
    },
    getImageData(...args) {
      calls.push({ type: "get-image-data", args });
      const [, , requestedWidth, requestedHeight] = args;
      return {
        data: pixels ??
          new Uint8ClampedArray(requestedWidth * requestedHeight * 4),
      };
    },
  };
  const canvas = {
    width,
    height,
    getContext(kind, options) {
      calls.push({ type: "get-context", kind, options });
      return context;
    },
  };
  return { canvas, context, calls };
}

function gpuHarness() {
  const events = [];
  const lost = deferred();
  let textureNumber = 0;
  const renderPass = () => ({
    setPipeline(pipeline) { events.push(`pipeline:${pipeline.label}`); },
    setBindGroup() { events.push("bind"); },
    draw(vertices) { events.push(`draw:${vertices}`); },
    end() { events.push("pass:end"); },
  });
  const device = {
    limits: { maxTextureDimension2D: 16384 },
    lost: lost.promise,
    queue: {
      writeTexture(destination, data, layout, size) {
        events.push({
          type: "write-texture",
          destination,
          data,
          layout,
          size,
        });
      },
      copyExternalImageToTexture(source, destination, size) {
        events.push({
          type: "copy",
          source,
          destination,
          size,
        });
      },
      submit(commands) { events.push({ type: "submit", commands }); },
      onSubmittedWorkDone() {
        events.push("queue:fence");
        return Promise.resolve();
      },
    },
    createTexture(descriptor) {
      events.push({ type: "texture", descriptor });
      return texture(descriptor.label || `texture-${++textureNumber}`, events);
    },
    createSampler(descriptor) {
      events.push({ type: "sampler", descriptor });
      return { label: "linear-sampler" };
    },
    createShaderModule(descriptor) {
      events.push({ type: "shader", descriptor });
      return { code: descriptor.code };
    },
    createRenderPipeline(descriptor) {
      const label = descriptor.vertex.module.code.startsWith("sharp:")
        ? "sharpen"
        : "blit";
      events.push({ type: "render-pipeline", label, descriptor });
      return {
        label,
        getBindGroupLayout() { return { label: `${label}-layout` }; },
      };
    },
    createBindGroup(descriptor) {
      events.push({ type: "bind-group", descriptor });
      return descriptor;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(descriptor) {
          events.push({ type: "render-pass", descriptor });
          return renderPass();
        },
        copyTextureToBuffer(source, destination, size) {
          events.push({
            type: "copy-texture-to-buffer",
            source,
            destination,
            size,
          });
        },
        finish() {
          events.push("encoder:finish");
          return { label: "commands" };
        },
      };
    },
  };
  return { device, events, lost };
}

async function readbackLifecycleHarness(createImageBitmapImpl) {
  const { device, events, lost } = gpuHarness();
  device.limits.maxBufferSize = 256;
  const mappedBytes = new Uint8Array(256);
  mappedBytes.set([1, 2, 3, 255]);
  const readbackBuffer = {
    destroyed: 0,
    maps: 0,
    unmaps: 0,
    async mapAsync() {
      this.maps++;
      events.push("buffer:map");
    },
    getMappedRange() {
      return mappedBytes.buffer;
    },
    unmap() {
      this.unmaps++;
      events.push("buffer:unmap");
    },
    destroy() {
      this.destroyed++;
      events.push("buffer:destroy");
    },
  };
  device.createBuffer = (descriptor) => {
    events.push({ type: "buffer", descriptor });
    return readbackBuffer;
  };

  const modelTexture = texture("model-output", events);
  let ready = false;
  let invalidations = 0;
  const engine = {
    async init() {
      ready = true;
      return { key: "model", label: "Model", scale: 1 };
    },
    async run() {
      return { tex: modelTexture, outW: 1, outH: 1 };
    },
    ready: () => ready,
    device: () => device,
    stats: () => ({}),
    async invalidateDevice(candidate) {
      assert.strictEqual(candidate, device);
      invalidations++;
      ready = false;
      return true;
    },
    async dispose() {
      ready = false;
    },
  };
  const context = {
    configure() {},
    unconfigure() {},
  };
  const canvas = {
    width: 1,
    height: 1,
    getContext(kind) {
      assert.equal(kind, "webgpu");
      return context;
    },
  };
  const upload = uploadCanvasHarness({
    pixels: new Uint8ClampedArray([9, 8, 7, 255]),
  });
  const inputBitmap = {
    width: 1,
    height: 1,
    closes: 0,
    close() { this.closes++; },
  };
  class FakeImageData {
    constructor(data, width, height, options) {
      this.data = data;
      this.width = width;
      this.height = height;
      this.options = options;
    }
  }
  const session = createNeuralFrameSession({
    loadDependencies: async () => ({
      createNeuralEngine: () => engine,
      SsimDownscaler: class {},
      buildSharpenShader: () => "",
    }),
    textureUsage: {
      TEXTURE_BINDING: 1,
      COPY_DST: 2,
      RENDER_ATTACHMENT: 4,
      COPY_SRC: 8,
    },
    bufferUsage: {
      COPY_DST: 16,
      MAP_READ: 32,
    },
    mapMode: {
      READ: 64,
    },
    ImageDataCtor: FakeImageData,
    createOffscreenCanvas: () => canvas,
    createUploadCanvas: () => upload.canvas,
    createImageBitmapImpl,
    isImageBitmap: (candidate) =>
      candidate === inputBitmap || candidate?.testOutputBitmap === true,
    log: () => {},
    warn: () => {},
  });
  await session.handle("attachCanvas", {});
  await session.handle("init", { modelKey: "model" });

  return {
    session,
    inputBitmap,
    readbackBuffer,
    lost,
    invalidations: () => invalidations,
    run: () => session.handle("run", {
      bitmap: inputBitmap,
      srcW: 1,
      srcH: 1,
      presentation: {},
    }),
  };
}

test("extension-frame session copies, infers, SSim-downscales, sharpens, and presents", async () => {
  const { device, events } = gpuHarness();
  const uploadPixels = Uint8ClampedArray.from(
    { length: 4 * 2 * 4 },
    (_, index) => index + 1,
  );
  const upload = uploadCanvasHarness({ pixels: uploadPixels });
  const modelTexture = texture("model-output", events);
  const downscaledTexture = texture("ssim-output", events);
  let engineReady = false;
  const engineCalls = [];
  const engine = {
    async init(key) {
      engineCalls.push(["init", key]);
      engineReady = true;
      return { key, label: "Test neural", scale: 2 };
    },
    async run(source, width, height) {
      engineCalls.push(["run", source, width, height]);
      return { tex: modelTexture, outW: width * 2, outH: height * 2 };
    },
    async stop() { engineCalls.push(["stop"]); },
    async dispose() {
      engineCalls.push(["dispose"]);
      engineReady = false;
    },
    ready: () => engineReady,
    device: () => device,
    stats: () => ({ n: 1, lastTiles: 1 }),
    invalidateDevice: async () => true,
  };
  const ssimCalls = [];
  class FakeSsimDownscaler {
    constructor(owner) {
      assert.strictEqual(owner, device);
      ssimCalls.push(["construct"]);
    }
    prepare(...args) {
      ssimCalls.push(["prepare", ...args]);
      return true;
    }
    run(encoder, input) {
      ssimCalls.push(["run", encoder, input]);
      return downscaledTexture;
    }
    destroy() { ssimCalls.push(["destroy"]); }
  }
  const context = {
    configurations: [],
    unconfigures: 0,
    configure(descriptor) { this.configurations.push(descriptor); },
    unconfigure() { this.unconfigures++; },
    getCurrentTexture() { return texture("canvas-current", events); },
  };
  const canvas = {
    width: 1,
    height: 1,
    transferToImageBitmap() {
      events.push("canvas:fallback-snapshot");
      return outputBitmap;
    },
    getContext(kind) {
      assert.equal(kind, "webgpu");
      return context;
    },
  };
  const bitmap = {
    width: 4,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const outputBitmap = {
    width: 4,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const strengths = [];
  let clock = 10;
  const session = createNeuralFrameSession({
    loadDependencies: async () => ({
      createNeuralEngine: () => engine,
      SsimDownscaler: FakeSsimDownscaler,
      buildSharpenShader: (strength) => {
        strengths.push(strength);
        return `sharp:${strength}`;
      },
    }),
    gpu: { getPreferredCanvasFormat: () => "bgra8unorm" },
    textureUsage: {
      TEXTURE_BINDING: 1,
      COPY_DST: 2,
      RENDER_ATTACHMENT: 4,
    },
    createOffscreenCanvas: () => canvas,
    createUploadCanvas: (width, height) => {
      events.push({ type: "upload-canvas", width, height });
      return upload.canvas;
    },
    createImageBitmapImpl: async (source, x, y, width, height) => {
      assert.strictEqual(source, canvas);
      assert.deepEqual([x, y, width, height], [0, 0, 4, 2]);
      events.push("canvas:snapshot");
      return outputBitmap;
    },
    isImageBitmap: (candidate) => candidate === bitmap || candidate === outputBitmap,
    now: () => clock += 4,
    log: () => {},
    warn: () => {},
  });

  const attached = await session.handle("attachCanvas", {});
  assert.equal(attached.attached, true);
  const initialized = await session.handle("init", { modelKey: "test-model" });
  assert.deepEqual(initialized.model, {
    key: "test-model",
    label: "Test neural",
    scale: 2,
  });

  const result = await session.handle("run", {
    bitmap,
    srcW: 4,
    srcH: 2,
    presentation: {
      width: 4,
      height: 2,
      ssimdsEnabled: true,
      sharpenEnabled: true,
      sharpenStrength: 1.25,
    },
  });
  assert.equal(bitmap.closes, 1, "the transferred ImageBitmap is always released");
  assert.deepEqual(engineCalls.slice(0, 2).map((call) => call[0]), ["init", "run"]);
  assert.deepEqual(ssimCalls[1].slice(0, 5), [
    "prepare",
    8,
    4,
    4,
    2,
  ]);
  assert.strictEqual(ssimCalls[1][5], modelTexture);
  assert.equal(ssimCalls[2][0], "run");
  assert.deepEqual(strengths, [1.25]);
  assert.deepEqual(result.presentation, {
    source: { width: 4, height: 2 },
    output: { width: 4, height: 2 },
    ssimds: {
      source: { width: 8, height: 4 },
      output: { width: 4, height: 2 },
    },
    sharpen: {
      source: { width: 4, height: 2 },
      output: { width: 4, height: 2 },
      strength: 1.25,
    },
  });
  assert.equal(canvas.width, 4);
  assert.equal(canvas.height, 2);
  assert.deepEqual(context.configurations, [{
    device,
    format: "bgra8unorm",
    colorSpace: "srgb",
    alphaMode: "opaque",
    usage: 4,
  }]);
  assert.deepEqual(events.find((entry) => entry?.type === "upload-canvas"), {
    type: "upload-canvas",
    width: 4,
    height: 2,
  });
  assert.deepEqual(upload.calls[0], {
    type: "get-context",
    kind: "2d",
    options: {
      alpha: false,
      colorSpace: "srgb",
      willReadFrequently: true,
    },
  });
  assert.equal(upload.canvas.width, 4);
  assert.equal(upload.canvas.height, 2);
  assert.equal(upload.context.globalCompositeOperation, "copy");
  assert.deepEqual(upload.calls[1], {
    type: "draw-image",
    args: [bitmap, 0, 0, 4, 2],
  });
  assert.deepEqual(upload.calls[2], {
    type: "get-image-data",
    args: [0, 0, 4, 2, { colorSpace: "srgb" }],
  });
  const uploadCall = events.find((entry) => entry?.type === "write-texture");
  assert.deepEqual(uploadCall.destination, {
    texture: engineCalls[1][1].tex,
  });
  assert.strictEqual(uploadCall.data, uploadPixels);
  assert.deepEqual(uploadCall.layout, {
    bytesPerRow: 4 * 4,
    rowsPerImage: 2,
  });
  assert.deepEqual(uploadCall.size, { width: 4, height: 2 });
  assert.equal(
    events.some((entry) => entry?.type === "copy"),
    false,
    "the production upload must not import the transferred bitmap directly",
  );
  assert.ok(events.some((entry) => entry?.type === "submit"));
  assert.ok(
    events.indexOf("queue:fence") < events.indexOf("canvas:snapshot"),
    "the submitted presentation must finish before the canvas is snapshotted",
  );
  assert.equal(events.includes("canvas:fallback-snapshot"), false);
  assert.equal(result.stats.ssimdsRuns, 1);
  assert.equal(result.stats.sharpenRuns, 1);
  assert.strictEqual(result.bitmap, outputBitmap);
  assert.equal(outputBitmap.closes, 0, "the caller owns the exported output bitmap");
  result.bitmap.close();

  const stopped = await session.handle("stop", {});
  assert.equal(stopped.stopped, true);
  assert.equal(context.unconfigures, 1);
  await assert.rejects(
    session.handle("run", { bitmap, srcW: 4, srcH: 2 }),
    /not initialized/,
  );
  assert.equal(bitmap.closes, 2, "precondition failures also close transferred bitmaps");
  await session.handle("dispose", {});
  assert.deepEqual(engineCalls.at(-1), ["dispose"]);
});

test("rgba8 presentation readback strips padded MAP_READ rows and releases GPU resources", async () => {
  const { device, events } = gpuHarness();
  device.limits.maxBufferSize = 512;

  const firstRow = Uint8Array.from([
    1, 2, 3, 255,
    4, 5, 6, 255,
    7, 8, 9, 255,
  ]);
  const secondRow = Uint8Array.from([
    11, 12, 13, 255,
    14, 15, 16, 255,
    17, 18, 19, 255,
  ]);
  const mappedBytes = new Uint8Array(512);
  mappedBytes.fill(0xee);
  mappedBytes.set(firstRow, 0);
  mappedBytes.set(secondRow, 256);
  const readbackBuffer = {
    destroyed: 0,
    maps: [],
    unmaps: 0,
    async mapAsync(mode) {
      this.maps.push(mode);
      events.push({ type: "map-read", mode });
    },
    getMappedRange() {
      events.push("buffer:mapped-range");
      return mappedBytes.buffer;
    },
    unmap() {
      this.unmaps++;
      events.push("buffer:unmap");
    },
    destroy() {
      this.destroyed++;
      events.push("buffer:destroy");
    },
  };
  const bufferDescriptors = [];
  device.createBuffer = (descriptor) => {
    bufferDescriptors.push(descriptor);
    events.push({ type: "buffer", descriptor });
    return readbackBuffer;
  };

  const modelTexture = texture("model-output", events);
  let ready = false;
  const engine = {
    async init() {
      ready = true;
      return { key: "model", label: "Model", scale: 1 };
    },
    async run(_source, width, height) {
      assert.deepEqual([width, height], [3, 2]);
      return { tex: modelTexture, outW: width, outH: height };
    },
    ready: () => ready,
    device: () => device,
    stats: () => ({}),
    async dispose() {
      ready = false;
    },
  };
  const context = {
    unconfigures: 0,
    configure() {
      assert.fail("readback presentation must not configure a canvas swapchain");
    },
    unconfigure() {
      this.unconfigures++;
    },
    getCurrentTexture() {
      assert.fail("readback presentation must not acquire a canvas texture");
    },
  };
  const canvas = {
    width: 1,
    height: 1,
    getContext(kind) {
      assert.equal(kind, "webgpu");
      return context;
    },
  };
  const uploadPixels = new Uint8ClampedArray(3 * 2 * 4);
  const upload = uploadCanvasHarness({ pixels: uploadPixels });
  const bitmap = {
    width: 3,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const oversizedBitmap = {
    width: 3,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const outputBitmap = {
    width: 3,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const imageDataCalls = [];
  class FakeImageData {
    constructor(data, width, height, options) {
      this.data = data;
      this.width = width;
      this.height = height;
      this.options = options;
      imageDataCalls.push(this);
      events.push("image-data:create");
    }
  }
  const imageBitmapSources = [];
  const session = createNeuralFrameSession({
    loadDependencies: async () => ({
      createNeuralEngine: () => engine,
      SsimDownscaler: class {},
      buildSharpenShader: () => "",
    }),
    textureUsage: {
      TEXTURE_BINDING: 1,
      COPY_DST: 2,
      RENDER_ATTACHMENT: 4,
      COPY_SRC: 8,
    },
    bufferUsage: {
      COPY_DST: 16,
      MAP_READ: 32,
    },
    mapMode: {
      READ: 64,
    },
    ImageDataCtor: FakeImageData,
    createOffscreenCanvas: () => canvas,
    createUploadCanvas: () => upload.canvas,
    createImageBitmapImpl: async (source) => {
      imageBitmapSources.push(source);
      events.push("image-bitmap:create");
      return outputBitmap;
    },
    isImageBitmap: (candidate) =>
      candidate === bitmap ||
      candidate === oversizedBitmap ||
      candidate === outputBitmap,
    log: () => {},
    warn: () => {},
  });

  await session.handle("attachCanvas", {});
  await session.handle("init", { modelKey: "model" });
  const result = await session.handle("run", {
    bitmap,
    srcW: 3,
    srcH: 2,
    presentation: {},
  });

  assert.equal(bitmap.closes, 1);
  const readbackCreation = events.find(
    (entry) =>
      entry?.type === "texture" &&
      entry.descriptor.label === "neural-frame-readback-3x2",
  );
  assert.deepEqual(readbackCreation.descriptor, {
    label: "neural-frame-readback-3x2",
    size: { width: 3, height: 2 },
    format: "rgba8unorm",
    usage: 4 | 8,
  });
  assert.deepEqual(bufferDescriptors, [{
    label: "neural-frame-readback-512",
    size: 512,
    usage: 16 | 32,
  }]);
  const readbackPipeline = events.find(
    (entry) => entry?.type === "render-pipeline" && entry.label === "blit",
  );
  assert.equal(
    readbackPipeline.descriptor.fragment.targets[0].format,
    "rgba8unorm",
  );
  const copy = events.find(
    (entry) => entry?.type === "copy-texture-to-buffer",
  );
  assert.strictEqual(copy.source.texture.label, "neural-frame-readback-3x2");
  assert.strictEqual(copy.destination.buffer, readbackBuffer);
  assert.deepEqual(copy.destination, {
    buffer: readbackBuffer,
    bytesPerRow: 256,
    rowsPerImage: 2,
  });
  assert.deepEqual(copy.size, { width: 3, height: 2 });
  assert.ok(
    events.findIndex((entry) => entry?.type === "submit") <
      events.findIndex((entry) => entry?.type === "map-read"),
    "the GPU copy must be submitted before the readback buffer is mapped",
  );
  assert.deepEqual(readbackBuffer.maps, [64]);
  assert.equal(readbackBuffer.unmaps, 1);
  assert.equal(imageDataCalls.length, 1);
  assert.ok(imageDataCalls[0].data instanceof Uint8ClampedArray);
  assert.deepEqual(
    [...imageDataCalls[0].data],
    [...firstRow, ...secondRow],
    "256-byte GPU row padding must not reach ImageData",
  );
  assert.deepEqual({
    width: imageDataCalls[0].width,
    height: imageDataCalls[0].height,
    options: imageDataCalls[0].options,
  }, {
    width: 3,
    height: 2,
    options: { colorSpace: "srgb" },
  });
  assert.deepEqual(imageBitmapSources, [imageDataCalls[0]]);
  assert.strictEqual(result.bitmap, outputBitmap);
  assert.equal(outputBitmap.closes, 0, "the caller owns the readback bitmap");
  result.bitmap.close();

  await assert.rejects(
    session.handle("run", {
      bitmap: oversizedBitmap,
      srcW: 3,
      srcH: 2,
      presentation: { width: 3, height: 3 },
    }),
    (error) => {
      assert.equal(error.code, "resource-limit");
      assert.match(error.message, /readback exceeds the GPU buffer limit/);
      return true;
    },
  );
  assert.equal(oversizedBitmap.closes, 1);
  assert.equal(
    bufferDescriptors.length,
    1,
    "a readback over maxBufferSize must fail before allocating another buffer",
  );

  const readbackTexture = copy.source.texture;
  await session.handle("dispose", {});
  assert.equal(readbackBuffer.destroyed, 1);
  assert.equal(readbackTexture.destroyed, 1);
  assert.equal(context.unconfigures, 1);
});

test("device loss during readback bitmap creation closes the stale bitmap and unmaps", async () => {
  const creationStarted = deferred();
  const creationResult = deferred();
  const staleBitmap = {
    testOutputBitmap: true,
    width: 1,
    height: 1,
    closes: 0,
    close() { this.closes++; },
  };
  const harness = await readbackLifecycleHarness(async (image) => {
    creationStarted.resolve(image);
    return creationResult.promise;
  });

  const running = harness.run();
  await creationStarted.promise;
  assert.equal(harness.readbackBuffer.maps, 1);
  assert.equal(harness.readbackBuffer.unmaps, 0);

  harness.lost.resolve({
    reason: "unknown",
    message: "injected loss during bitmap creation",
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(harness.invalidations(), 1);
  creationResult.resolve(staleBitmap);

  await assert.rejects(running, (error) => {
    assert.equal(error.code, "device-lost");
    assert.equal(error.retryable, true);
    assert.match(error.message, /changed while exporting the output frame/);
    return true;
  });
  assert.equal(staleBitmap.closes, 1);
  assert.equal(harness.inputBitmap.closes, 1);
  assert.equal(harness.readbackBuffer.unmaps, 1);
  assert.equal(
    harness.readbackBuffer.destroyed,
    1,
    "device-loss cleanup destroys the mapped readback buffer",
  );
  await harness.session.handle("dispose", {});
  assert.equal(harness.readbackBuffer.destroyed, 1);
});

test("logical cancellation closes a created output bitmap before publication", async () => {
  const creationStarted = deferred();
  const creationResult = deferred();
  const staleBitmap = {
    testOutputBitmap: true,
    width: 1,
    height: 1,
    closes: 0,
    close() { this.closes++; },
  };
  const harness = await readbackLifecycleHarness(async (image) => {
    creationStarted.resolve(image);
    return creationResult.promise;
  });

  const running = harness.run();
  await creationStarted.promise;
  assert.equal(harness.session.cancel(), true);
  assert.equal(
    harness.readbackBuffer.destroyed,
    0,
    "logical cancellation does not destroy GPU resources concurrently",
  );
  creationResult.resolve(staleBitmap);

  await assert.rejects(running, (error) => {
    assert.equal(error.code, "cancelled");
    assert.equal(error.retryable, true);
    assert.match(error.message, /cancelled/);
    return true;
  });
  assert.equal(staleBitmap.closes, 1);
  assert.equal(harness.inputBitmap.closes, 1);
  assert.equal(harness.readbackBuffer.unmaps, 1);
  assert.equal(harness.readbackBuffer.destroyed, 0);

  await harness.session.handle("stop", {});
  assert.equal(harness.readbackBuffer.destroyed, 1);
  await harness.session.handle("dispose", {});
});

test("readback bitmap creation rejection still unmaps and disposes its buffer", async () => {
  const rejection = new Error("injected bitmap creation failure");
  let creationCalls = 0;
  const harness = await readbackLifecycleHarness(async () => {
    creationCalls++;
    throw rejection;
  });

  await assert.rejects(harness.run(), (error) => {
    assert.equal(error.code, "presentation-failed");
    assert.equal(error.retryable, true);
    assert.equal(error.cause, rejection);
    assert.match(error.message, /output readback failed/);
    return true;
  });
  assert.equal(creationCalls, 1);
  assert.equal(harness.inputBitmap.closes, 1);
  assert.equal(harness.readbackBuffer.maps, 1);
  assert.equal(harness.readbackBuffer.unmaps, 1);
  assert.equal(harness.readbackBuffer.destroyed, 0);

  await harness.session.handle("dispose", {});
  assert.equal(harness.readbackBuffer.destroyed, 1);
});

test("extension-frame handshake binds source, origin, fragment nonce, and one MessagePort", async () => {
  const nonce = "abcdefghijklmnop";
  const windowEvents = eventTarget();
  const parentMessages = [];
  const parent = {
    postMessage(message, targetOrigin) {
      parentMessages.push({ message, targetOrigin });
    },
  };
  const windowObject = {
    ...windowEvents,
    parent,
  };
  const portEvents = eventTarget();
  const portMessages = [];
  const portTransfers = [];
  const port = {
    ...portEvents,
    starts: 0,
    closes: 0,
    postMessage(message, transfer = []) {
      portMessages.push(message);
      portTransfers.push({ message, transfer });
    },
    start() { this.starts++; },
    close() { this.closes++; },
  };
  const runGate = deferred();
  const handled = [];
  let sessionDisposals = 0;
  const session = {
    async handle(method, payload) {
      handled.push([method, payload]);
      if (method === "run") return runGate.promise;
      if (method === "init") throw new Error("x".repeat(1000));
      return { [method === "dispose" ? "disposed" : "ok"]: true };
    },
    async dispose() { sessionDisposals++; },
  };
  const capabilityMessages = [];
  const runtime = await startNeuralFrameRuntime({
    windowObject,
    documentObject: {
      referrer: "https://video.example/watch?v=1",
    },
    locationObject: {
      href: `chrome-extension://extension-id/src/frame/neural-frame.html` +
        `#instanceNonce=${nonce}&frameCapability=${FRAME_CAPABILITY}`,
    },
    runtime: capabilityRuntime(capabilityMessages),
    createSession: () => session,
  });

  assert.deepEqual(capabilityMessages, [{
    type: "FSRCNNX_NEURAL_FRAME_CAPABILITY_CONSUME",
    capability: FRAME_CAPABILITY,
    instanceNonce: nonce,
  }]);
  assert.deepEqual(parentMessages, [{
    message: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "ready",
    },
    targetOrigin: "https://video.example",
  }]);
  windowObject.emit("message", {
    source: {},
    origin: "https://video.example",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  windowObject.emit("message", {
    source: parent,
    origin: "https://attacker.example",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  windowObject.emit("message", {
    source: parent,
    origin: "null",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  assert.equal(runtime.connected(), false);

  windowObject.emit("message", {
    source: parent,
    origin: "https://video.example",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  assert.equal(runtime.connected(), true);
  assert.equal(runtime.connectedOrigin(), "https://video.example");
  assert.equal(port.starts, 1);
  assert.deepEqual(portMessages, [{
    channel: NEURAL_FRAME_CHANNEL,
    kind: "connected",
    instanceNonce: nonce,
  }]);

  const firstBitmap = { close() {} };
  const busyBitmap = { closes: 0, close() { this.closes++; } };
  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "run-1",
      method: "run",
      payload: { bitmap: firstBitmap },
    },
  });
  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "run-2",
      method: "run",
      payload: { bitmap: busyBitmap },
    },
  });
  assert.equal(busyBitmap.closes, 1);
  assert.deepEqual(portMessages.at(-1), {
    channel: NEURAL_FRAME_CHANNEL,
    kind: "response",
    instanceNonce: nonce,
    id: "run-2",
    ok: false,
    error: {
      code: "run-busy",
      message: "A neural frame run is already pending",
      retryable: true,
    },
  });

  const outputBitmap = {
    width: 4,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  runGate.resolve({
    presented: true,
    bitmap: outputBitmap,
    presentation: { output: { width: 4, height: 2 } },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(portMessages.find((message) => message.id === "run-1"), {
    channel: NEURAL_FRAME_CHANNEL,
    kind: "response",
    instanceNonce: nonce,
    id: "run-1",
    ok: true,
    result: {
      presented: true,
      bitmap: outputBitmap,
      presentation: { output: { width: 4, height: 2 } },
    },
  });
  assert.deepEqual(
    portTransfers.find(({ message }) => message.id === "run-1").transfer,
    [outputBitmap],
  );
  assert.equal(outputBitmap.closes, 0, "postMessage owns the transferred output");

  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "init-1",
      method: "init",
      payload: {},
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  const failed = portMessages.find((message) => message.id === "init-1");
  assert.equal(failed.ok, false);
  assert.equal(failed.error.code, "init-failed");
  assert.equal(failed.error.message.length, 320);
  assert.equal(failed.error.retryable, false);

  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "dispose-1",
      method: "dispose",
      payload: {},
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(portMessages.find((message) => message.id === "dispose-1").ok, true);
  assert.equal(port.closes, 1, "the port closes only after the dispose response");
  assert.equal(sessionDisposals, 0, "the dispose request owns session teardown exactly once");
});

test("private cancel envelopes bypass the serialized command tail with exact-key validation", async () => {
  const nonce = "abcdefghijklmnop";
  const windowEvents = eventTarget();
  const parent = { postMessage() {} };
  const windowObject = { ...windowEvents, parent };
  const portEvents = eventTarget();
  const portMessages = [];
  const port = {
    ...portEvents,
    postMessage(message) { portMessages.push(message); },
    start() {},
    close() {},
  };
  const runGate = deferred();
  const events = [];
  const session = {
    async handle(method) {
      events.push(`handle:${method}`);
      if (method === "run") return runGate.promise;
      return { stopped: true };
    },
    cancel() {
      events.push("cancel");
      return true;
    },
    async dispose() {
      events.push("dispose");
    },
  };
  const runtime = await startNeuralFrameRuntime({
    windowObject,
    documentObject: { referrer: "https://video.example/watch" },
    locationObject: {
      href: `chrome-extension://extension-id/src/frame/neural-frame.html` +
        `#instanceNonce=${nonce}&frameCapability=${FRAME_CAPABILITY}`,
    },
    runtime: capabilityRuntime([]),
    createSession: () => session,
  });
  windowObject.emit("message", {
    source: parent,
    origin: "https://video.example",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });

  const unexpectedBitmap = {
    closes: 0,
    close() { this.closes++; },
  };
  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "cancel",
      instanceNonce: nonce,
      payload: { bitmap: unexpectedBitmap },
    },
  });
  assert.equal(unexpectedBitmap.closes, 1);
  assert.deepEqual(events, [], "an extended cancel envelope is rejected");

  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "run-1",
      method: "run",
      payload: { bitmap: { close() {} } },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(events, ["handle:run"]);

  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "cancel",
      instanceNonce: nonce,
    },
  });
  assert.deepEqual(
    events,
    ["handle:run", "cancel"],
    "logical cancellation runs synchronously outside commandTail",
  );

  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "stop-1",
      method: "stop",
      payload: {},
    },
  });
  await Promise.resolve();
  assert.deepEqual(
    events,
    ["handle:run", "cancel"],
    "physical cleanup remains queued behind the active run",
  );

  const cancellation = new Error("injected cancellation");
  cancellation.code = "cancelled";
  cancellation.retryable = true;
  runGate.reject(cancellation);
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(events, ["handle:run", "cancel", "handle:stop"]);
  assert.equal(
    portMessages.find(({ id }) => id === "run-1").error.code,
    "cancelled",
  );
  assert.equal(portMessages.find(({ id }) => id === "stop-1").ok, true);
  await runtime.close();
});

test("frame authorization fails before session creation or the ready announcement", async () => {
  const parentMessages = [];
  const parent = {
    postMessage(message, targetOrigin) {
      parentMessages.push({ message, targetOrigin });
    },
  };
  const windowObject = { ...eventTarget(), parent };
  let sessionCreations = 0;
  const locationObject = {
    href: "chrome-extension://extension-id/src/frame/neural-frame.html" +
      `#instanceNonce=abcdefghijklmnop&frameCapability=${FRAME_CAPABILITY}`,
  };

  await assert.rejects(startNeuralFrameRuntime({
    windowObject,
    documentObject: { referrer: "https://video.example/watch" },
    locationObject,
    runtime: capabilityRuntime([], { ok: false }),
    createSession() {
      sessionCreations++;
      return {};
    },
  }), (error) => error.code === "authorization-denied");
  await assert.rejects(startNeuralFrameRuntime({
    windowObject,
    documentObject: { referrer: "https://attacker.example/watch" },
    locationObject,
    runtime: capabilityRuntime([]),
    createSession() {
      sessionCreations++;
      return {};
    },
  }), (error) => error.code === "authorization-denied");

  assert.equal(sessionCreations, 0);
  assert.deepEqual(parentMessages, []);
});

test("a failed ready announcement disposes the authorized session", async () => {
  let disposals = 0;
  const parent = {
    postMessage() {
      throw new Error("injected detached parent");
    },
  };
  await assert.rejects(startNeuralFrameRuntime({
    windowObject: { ...eventTarget(), parent },
    documentObject: { referrer: "https://video.example/watch" },
    locationObject: {
      href: "chrome-extension://extension-id/src/frame/neural-frame.html" +
        `#instanceNonce=abcdefghijklmnop&frameCapability=${FRAME_CAPABILITY}`,
    },
    runtime: capabilityRuntime([]),
    createSession: () => ({
      async dispose() { disposals++; },
    }),
  }), (error) => error.code === "ready-failed" && error.retryable === true);
  assert.equal(disposals, 1);
});

test("opaque file-parent handshake requires its private fragment capability", async () => {
  const nonce = "abcdefghijklmnop";
  const session = {
    async handle() { return {}; },
    async dispose() {},
  };

  const unflaggedEvents = eventTarget();
  const unflaggedParent = { postMessage() {} };
  const unflaggedWindow = { ...unflaggedEvents, parent: unflaggedParent };
  const unflaggedPort = {
    ...eventTarget(),
    postMessage() {},
    start() {},
    close() {},
  };
  await assert.rejects(startNeuralFrameRuntime({
    windowObject: unflaggedWindow,
    documentObject: { referrer: "file:///C:/video/example.html" },
    locationObject: {
      href: `chrome-extension://extension-id/src/frame/neural-frame.html` +
        `#instanceNonce=${nonce}&frameCapability=${FRAME_CAPABILITY}`,
    },
    runtime: capabilityRuntime([], { parentOrigin: "null", opaqueParent: true }),
    createSession: () => session,
  }), (error) => error.code === "authorization-denied");

  const windowEvents = eventTarget();
  const readyMessages = [];
  const parent = {
    postMessage(message, targetOrigin) {
      readyMessages.push({ message, targetOrigin });
    },
  };
  const windowObject = { ...windowEvents, parent };
  const portMessages = [];
  const port = {
    ...eventTarget(),
    starts: 0,
    closes: 0,
    postMessage(message) { portMessages.push(message); },
    start() { this.starts++; },
    close() { this.closes++; },
  };
  const runtime = await startNeuralFrameRuntime({
    windowObject,
    documentObject: { referrer: "file:///C:/video/example.html" },
    locationObject: {
      href: `chrome-extension://extension-id/src/frame/neural-frame.html` +
        `#instanceNonce=${nonce}&frameCapability=${FRAME_CAPABILITY}&opaqueParent=1`,
    },
    runtime: capabilityRuntime([], { parentOrigin: "null", opaqueParent: true }),
    createSession: () => session,
  });

  assert.equal(readyMessages[0].targetOrigin, "*");
  for (const invalid of [
    { source: {}, origin: "null", instanceNonce: nonce },
    { source: parent, origin: "null", instanceNonce: "wrong-wrong-wrong" },
    { source: parent, origin: "https://video.example", instanceNonce: nonce },
  ]) {
    windowObject.emit("message", {
      source: invalid.source,
      origin: invalid.origin,
      data: {
        channel: NEURAL_FRAME_CHANNEL,
        kind: "connect",
        instanceNonce: invalid.instanceNonce,
      },
      ports: [port],
    });
  }
  assert.equal(runtime.connected(), false);

  windowObject.emit("message", {
    source: parent,
    origin: "null",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  assert.equal(runtime.connected(), true);
  assert.equal(runtime.connectedOrigin(), "null");
  assert.equal(port.starts, 1);
  assert.equal(portMessages[0].kind, "connected");
  await runtime.close();
  assert.equal(port.closes, 1);
});

test("extension-frame runtime closes an output bitmap when its transfer fails", async () => {
  const nonce = "abcdefghijklmnop";
  const windowEvents = eventTarget();
  const parent = { postMessage() {} };
  const windowObject = { ...windowEvents, parent };
  const portEvents = eventTarget();
  const outputBitmap = {
    width: 4,
    height: 2,
    closes: 0,
    close() { this.closes++; },
  };
  const transferAttempts = [];
  const port = {
    ...portEvents,
    closes: 0,
    postMessage(message, transfer = []) {
      transferAttempts.push({ message, transfer });
      if (message.kind === "response" && message.id === "run-1") {
        throw new Error("injected postMessage failure");
      }
    },
    start() {},
    close() { this.closes++; },
  };
  let sessionDisposals = 0;
  const session = {
    async handle(method) {
      assert.equal(method, "run");
      return {
        bitmap: outputBitmap,
        presentation: { output: { width: 4, height: 2 } },
      };
    },
    async dispose() { sessionDisposals++; },
  };
  const runtime = await startNeuralFrameRuntime({
    windowObject,
    documentObject: { referrer: "https://video.example/watch" },
    locationObject: {
      href: `chrome-extension://extension-id/src/frame/neural-frame.html` +
        `#instanceNonce=${nonce}&frameCapability=${FRAME_CAPABILITY}`,
    },
    runtime: capabilityRuntime([]),
    createSession: () => session,
  });

  windowObject.emit("message", {
    source: parent,
    origin: "https://video.example",
    data: { channel: NEURAL_FRAME_CHANNEL, kind: "connect", instanceNonce: nonce },
    ports: [port],
  });
  port.emit("message", {
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "request",
      instanceNonce: nonce,
      id: "run-1",
      method: "run",
      payload: { bitmap: { close() {} } },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  await runtime.close();

  const runAttempt = transferAttempts.find(({ message }) => message.id === "run-1");
  assert.deepEqual(runAttempt.transfer, [outputBitmap]);
  assert.equal(outputBitmap.closes, 1);
  assert.equal(port.closes, 1);
  assert.equal(sessionDisposals, 1);
  assert.equal(runtime.connected(), false);
});

test("device loss unpublishes the frame device, notifies the parent, and remains disposable", async () => {
  const { device, lost } = gpuHarness();
  let ready = false;
  let invalidations = 0;
  let disposals = 0;
  const engine = {
    async init() {
      ready = true;
      return { key: "model", label: "Model", scale: 2 };
    },
    ready: () => ready,
    device: () => device,
    stats: () => ({ n: 0 }),
    async invalidateDevice(candidate) {
      assert.strictEqual(candidate, device);
      invalidations++;
      ready = false;
      return true;
    },
    async dispose() { disposals++; },
  };
  const context = {
    unconfigures: 0,
    configure() {},
    unconfigure() { this.unconfigures++; },
  };
  const canvas = {
    width: 1,
    height: 1,
    getContext: () => context,
  };
  const notifications = [];
  const session = createNeuralFrameSession({
    loadDependencies: async () => ({
      createNeuralEngine: () => engine,
      SsimDownscaler: class {},
      buildSharpenShader: () => "",
    }),
    createOffscreenCanvas: () => canvas,
    isImageBitmap: () => true,
    onDeviceLost: (error, stats) => notifications.push({ error, stats }),
    log: () => {},
    warn: () => {},
  });
  await session.handle("attachCanvas", {});
  await session.handle("init", { modelKey: "model" });

  lost.resolve({ reason: "unknown", message: "injected loss" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(invalidations, 1);
  assert.equal(context.unconfigures, 1);
  assert.equal(notifications.length, 1);
  assert.deepEqual(notifications[0].error, {
    code: "device-lost",
    message: "Neural GPU device was lost: injected loss",
    retryable: true,
  });
  assert.equal(notifications[0].stats.initialized, false);
  assert.equal(notifications[0].stats.deviceLosses, 1);

  const bitmap = { closes: 0, close() { this.closes++; } };
  await assert.rejects(
    session.handle("run", { bitmap, srcW: 2, srcH: 2, presentation: {} }),
    /not initialized/,
  );
  assert.equal(bitmap.closes, 1);
  await session.handle("dispose", {});
  assert.equal(disposals, 1);
});

test("frame dimensions have no policy ceiling below the adapter's reported limit", async () => {
  const { device } = gpuHarness();
  const upload = uploadCanvasHarness();
  const modelTexture = { createView: () => ({}) };
  let ready = false;
  const engine = {
    async init() {
      ready = true;
      return { key: "model", label: "Model", scale: 1 };
    },
    async run(_source, width, height) {
      return { tex: modelTexture, outW: width, outH: height };
    },
    ready: () => ready,
    device: () => device,
    stats: () => ({}),
    async dispose() {},
  };
  const context = {
    configure() {},
    unconfigure() {},
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const canvas = {
    width: 1,
    height: 1,
    transferToImageBitmap: () => outputBitmap,
    getContext: () => context,
  };
  const bitmap = {
    width: 9000,
    height: 1,
    close() {},
  };
  const outputBitmap = {
    width: 9000,
    height: 1,
    close() {},
  };
  const session = createNeuralFrameSession({
    loadDependencies: async () => ({
      createNeuralEngine: () => engine,
      SsimDownscaler: class {},
      buildSharpenShader: () => "",
    }),
    gpu: { getPreferredCanvasFormat: () => "bgra8unorm" },
    textureUsage: {
      TEXTURE_BINDING: 1,
      COPY_DST: 2,
      RENDER_ATTACHMENT: 4,
    },
    createOffscreenCanvas: () => canvas,
    createUploadCanvas: () => upload.canvas,
    isImageBitmap: (candidate) => candidate === bitmap || candidate === outputBitmap,
    log: () => {},
    warn: () => {},
  });
  await session.handle("attachCanvas", {});
  await session.handle("init", { modelKey: "model" });
  const result = await session.handle("run", {
    bitmap,
    srcW: 9000,
    srcH: 1,
    presentation: {},
  });
  assert.deepEqual(result.presentation.output, { width: 9000, height: 1 });
  assert.equal(canvas.width, 9000);
  assert.strictEqual(result.bitmap, outputBitmap);
  result.bitmap.close();
  await session.handle("dispose", {});
});

test("error serialization is bounded and strips unsafe codes and control characters", () => {
  const error = {
    code: "NOT VALID!",
    message: `bad\u0000message${"z".repeat(1000)}`,
    retryable: true,
  };
  const serialized = serializeNeuralFrameError(error, "also invalid!");
  assert.equal(serialized.code, "internal-error");
  assert.equal(serialized.message.length, 320);
  assert.doesNotMatch(serialized.message, /[\u0000-\u001f\u007f]/);
  assert.equal(serialized.retryable, true);
  assert.deepEqual(Object.keys(serialized), ["code", "message", "retryable"]);
});

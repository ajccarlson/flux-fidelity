import test from "node:test";
import assert from "node:assert/strict";

import { GrabResourceLimitError, WebGPUGrabber } from "../src/core/fsrcnnx-grab.js";

function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function makeDevice({ limits = {}, mapGate = null, failPipeline = false } = {}) {
  const lost = deferred();
  const mapEntered = deferred();
  const events = {
    destroys: 0,
    textures: [],
    buffers: [],
    failNextBuffer: false,
    throwOnMappedRange: false,
    mapEntered,
  };
  const device = {
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      ...limits,
    },
    lost: lost.promise,
    events,
    lose(info = { reason: "unknown", message: "test loss" }) { lost.resolve(info); },
    destroy() {
      events.destroys++;
      lost.resolve({ reason: "destroyed", message: "intentional test destroy" });
    },
    createSampler() { return {}; },
    createShaderModule({ code }) { return { code }; },
    createRenderPipeline() {
      if (failPipeline) throw new Error("pipeline construction failed");
      const pipeline = { getBindGroupLayout: () => pipeline };
      return pipeline;
    },
    createTexture(description) {
      const texture = {
        ...description,
        destroyed: 0,
        createView() { return { texture }; },
        destroy() { texture.destroyed++; },
      };
      events.textures.push(texture);
      return texture;
    },
    createBuffer(description) {
      if (events.failNextBuffer) {
        events.failNextBuffer = false;
        throw new Error("buffer allocation failed");
      }
      let mapped = false;
      const buffer = {
        ...description,
        destroyed: 0,
        unmaps: 0,
        async mapAsync() {
          mapEntered.resolve();
          if (mapGate) await mapGate.promise;
          mapped = true;
        },
        getMappedRange() {
          if (!mapped) throw new Error("buffer is not mapped");
          if (events.throwOnMappedRange) throw new Error("mapped range failed");
          return new ArrayBuffer(description.size);
        },
        unmap() {
          if (!mapped) throw new Error("buffer is not mapped");
          mapped = false;
          buffer.unmaps++;
        },
        destroy() { buffer.destroyed++; },
      };
      events.buffers.push(buffer);
      return buffer;
    },
    importExternalTexture({ source }) { return { source }; },
    createBindGroup(description) { return description; },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} };
        },
        copyTextureToBuffer() {},
        finish() { return {}; },
      };
    },
    queue: { submit() {} },
  };
  return device;
}

function installWebGpuGlobals(t, device) {
  const descriptors = new Map([
    ["navigator", Object.getOwnPropertyDescriptor(globalThis, "navigator")],
    ["GPUBufferUsage", Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage")],
    ["GPUTextureUsage", Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage")],
    ["GPUMapMode", Object.getOwnPropertyDescriptor(globalThis, "GPUMapMode")],
  ]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu: { requestAdapter: async () => ({ requestDevice: async () => device }) } },
  });
  globalThis.GPUBufferUsage = { COPY_DST: 1, MAP_READ: 2 };
  globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, COPY_SRC: 2 };
  globalThis.GPUMapMode = { READ: 1 };
  t.after(() => {
    for (const [name, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
}

test("grab allocations replace atomically and preserve the old generation on failure", async (t) => {
  const device = makeDevice();
  installWebGpuGlobals(t, device);
  const grabber = new WebGPUGrabber({ log() {}, warn() {} });
  assert.equal(await grabber.init(), true);

  grabber._alloc(64, 48);
  const oldTexture = grabber.tex;
  const oldBuffer = grabber.readBuf;
  device.events.failNextBuffer = true;
  assert.throws(() => grabber._alloc(96, 54), /buffer allocation failed/);

  const failedTexture = device.events.textures.at(-1);
  assert.notEqual(failedTexture, oldTexture);
  assert.equal(failedTexture.destroyed, 1);
  assert.equal(oldTexture.destroyed, 0);
  assert.equal(oldBuffer.destroyed, 0);
  assert.equal(grabber.tex, oldTexture);
  assert.equal(grabber.readBuf, oldBuffer);
  assert.deepEqual([grabber._w, grabber._h], [64, 48]);

  grabber._alloc(128, 72);
  assert.equal(oldTexture.destroyed, 1);
  assert.equal(oldBuffer.destroyed, 1);
  await grabber.destroy();
  assert.equal(device.events.destroys, 1);
});

test("grab allocation rejects device-limit violations before creating resources", async (t) => {
  const device = makeDevice({ limits: { maxTextureDimension2D: 128, maxBufferSize: 4096 } });
  installWebGpuGlobals(t, device);
  const grabber = new WebGPUGrabber({ log() {}, warn() {} });
  assert.equal(await grabber.init(), true);

  assert.throws(() => grabber._alloc(129, 1), GrabResourceLimitError);
  assert.throws(() => grabber._alloc(64, 17), GrabResourceLimitError);
  assert.equal(device.events.textures.length, 0);
  assert.equal(device.events.buffers.length, 0);
  await grabber.destroy();
});

test("grab always unmaps a successfully mapped staging buffer after a read failure", async (t) => {
  const device = makeDevice();
  installWebGpuGlobals(t, device);
  const warnings = [];
  const grabber = new WebGPUGrabber({ log() {}, warn: (...parts) => warnings.push(parts.join(" ")) });
  assert.equal(await grabber.init(), true);
  device.events.throwOnMappedRange = true;

  assert.equal(await grabber.grab({ videoWidth: 16, videoHeight: 8 }), null);
  assert.equal(grabber.readBuf.unmaps, 1);
  assert.match(warnings.join("\n"), /mapped range failed/);
  await grabber.destroy();
});

test("destroy waits for a pending grab before tearing down its owned device", async (t) => {
  const mapGate = deferred();
  const device = makeDevice({ mapGate });
  installWebGpuGlobals(t, device);
  const grabber = new WebGPUGrabber({ log() {}, warn() {} });
  assert.equal(await grabber.init(), true);

  const pendingGrab = grabber.grab({ videoWidth: 16, videoHeight: 8 });
  await device.events.mapEntered.promise;
  const buffer = grabber.readBuf;
  const destruction = grabber.destroy();
  assert.equal(grabber.ready, false);
  assert.equal(device.events.destroys, 0);
  assert.equal(buffer.destroyed, 0);

  mapGate.resolve();
  assert.equal(await pendingGrab, null);
  await destruction;
  assert.equal(buffer.unmaps, 1);
  assert.equal(buffer.destroyed, 1);
  assert.equal(device.events.destroys, 1);
  assert.equal(grabber.device, null);
});

test("failed initialization destroys its candidate device", async (t) => {
  const device = makeDevice({ failPipeline: true });
  installWebGpuGlobals(t, device);
  const grabber = new WebGPUGrabber({ log() {}, warn() {} });

  assert.equal(await grabber.init(), false);
  assert.equal(grabber.ready, false);
  assert.equal(grabber.device, null);
  assert.equal(device.events.destroys, 1);
});

test("device-loss notification is identity guarded and intentional teardown is silent", async (t) => {
  const device = makeDevice();
  installWebGpuGlobals(t, device);
  const losses = [];
  const grabber = new WebGPUGrabber({
    log() {},
    warn() {},
    onDeviceLost: (owner, info) => losses.push({ info, owner }),
  });
  assert.equal(await grabber.init(), true);
  grabber._handleDeviceLost({}, { message: "stale" });
  assert.equal(losses.length, 0);

  const info = { reason: "unknown", message: "adapter reset" };
  device.lose(info);
  await Promise.resolve();
  await grabber.destroy();
  assert.deepEqual(losses, [{ info, owner: device }]);
  assert.equal(device.events.destroys, 1);

  const intentionalDevice = makeDevice();
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu: { requestAdapter: async () => ({ requestDevice: async () => intentionalDevice }) } },
  });
  let intentionalLosses = 0;
  const intentional = new WebGPUGrabber({ log() {}, warn() {}, onDeviceLost: () => intentionalLosses++ });
  assert.equal(await intentional.init(), true);
  await intentional.destroy();
  await Promise.resolve();
  assert.equal(intentionalLosses, 0);
  assert.equal(intentionalDevice.events.destroys, 1);
});

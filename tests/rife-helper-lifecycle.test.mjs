import test from "node:test";
import assert from "node:assert/strict";

import {
  ORT_WASM_BYTE_LENGTH,
  ORT_WASM_FILE,
  ORT_WASM_MODULE_FILE,
} from "../src/core/fsrcnnx-rife.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
}

test("module GPU-helper lifecycle serializes RIFE, blend, and teardown", async (t) => {
  const chromeDescriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome");
  const fetchDescriptor = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  const stateDescriptor = Object.getOwnPropertyDescriptor(globalThis, "__rifeHelperLifecycle");
  const targetDevice = { id: "ort-device", lost: new Promise(() => {}) };
  const session = {
    inputNames: ["input"],
    outputNames: ["output"],
    releases: 0,
    async release() { this.releases++; },
  };
  const state = {
    env: { wasm: {}, webgpu: { enableFp16: false, device: null } },
    instances: [],
    initGates: [],
    destroyGates: new Map(),
    create() {
      assert.deepEqual(state.env.wasm.wasmPaths, {
        mjs: `https://extension.test/vendor/ort/${ORT_WASM_MODULE_FILE}`,
        wasm: `https://extension.test/vendor/ort/${ORT_WASM_FILE}`,
      });
      assert.equal(state.env.wasm.numThreads, 1);
      state.env.webgpu.device = targetDevice;
      return session;
    },
  };

  const runtimeSource = `
    const state = globalThis.__rifeHelperLifecycle;
    export const env = state.env;
    export class Tensor {
      constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
      static fromGpuBuffer() { return {}; }
    }
    export class InferenceSession {
      static create(url, options) { return state.create(url, options); }
    }
  `;
  const gpuSource = `
    const state = globalThis.__rifeHelperLifecycle;
    export class GpuInterp {
      constructor(options = {}) {
        this.id = state.instances.length + 1;
        this.options = options;
        this.ready = false;
        this.device = null;
        this.ort = null;
        this._rifeCapable = false;
        this.destroyed = false;
        this.destroyCalls = 0;
        this._destroyPromise = null;
        state.instances.push(this);
      }
      async init(device, ort) {
        const gate = state.initGates.shift();
        if (gate) await gate.promise;
        this.device = device || { id: "owned-" + this.id };
        this.ort = ort;
        this._rifeCapable = !!ort;
        this.ready = true;
        return true;
      }
      destroy() {
        if (this._destroyPromise) return this._destroyPromise;
        this.destroyCalls++;
        this.destroyed = true;
        this.ready = false;
        const gate = state.destroyGates.get(this.id);
        this._destroyPromise = (async () => {
          if (gate) await gate.promise;
          this.device = null;
          this.ort = null;
          this._rifeCapable = false;
        })();
        return this._destroyPromise;
      }
    }
  `;
  const runtimeUrl = `data:text/javascript,${encodeURIComponent(runtimeSource)}`;
  const gpuUrl = `data:text/javascript,${encodeURIComponent(gpuSource)}`;
  const testOrtWasm = new Uint8Array(ORT_WASM_BYTE_LENGTH);
  testOrtWasm.set([0x00, 0x61, 0x73, 0x6d]);

  globalThis.__rifeHelperLifecycle = state;
  globalThis.chrome = {
    runtime: {
      getURL(path) {
        if (path === "vendor/ort/ort.webgpu.min.mjs") return runtimeUrl;
        if (path === "src/core/fsrcnnx-rife-gpu.js") return gpuUrl;
        return `https://extension.test/${path}`;
      },
    },
  };
  globalThis.fetch = async (url) => {
    if (String(url).endsWith(ORT_WASM_FILE)) {
      return {
        ok: true,
        status: 200,
        headers: { get: () => String(testOrtWasm.byteLength) },
        arrayBuffer: async () => testOrtWasm.buffer,
      };
    }
    return { ok: true, status: 200 };
  };

  const rife = await import(`../src/core/fsrcnnx-rife.js?helper-lifecycle=${Date.now()}`);
  t.after(async () => {
    try { await rife.disposeRife(); } catch {}
    if (chromeDescriptor) Object.defineProperty(globalThis, "chrome", chromeDescriptor);
    else delete globalThis.chrome;
    if (fetchDescriptor) Object.defineProperty(globalThis, "fetch", fetchDescriptor);
    else delete globalThis.fetch;
    if (stateDescriptor) Object.defineProperty(globalThis, "__rifeHelperLifecycle", stateDescriptor);
    else delete globalThis.__rifeHelperLifecycle;
  });

  assert.equal(await rife.initRife(), true);
  assert.equal(rife.getOrtDevice(), targetDevice);

  // A compatible blend request joins a pending RIFE build and reuses its richer
  // helper instead of racing a standalone candidate onto the same shared device.
  const rifeGate = deferred();
  state.initGates.push(rifeGate);
  const rifeStart = rife.initGpuInterp({ log() {}, warn() {} });
  await waitFor(() => state.instances.length === 1, "RIFE helper initialization did not start");
  const compatibleBlendStart = rife.initGpuBlendStandalone({
    device: targetDevice,
    log() {},
    warn() {},
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.instances.length, 1);
  rifeGate.resolve();
  assert.deepEqual(await Promise.all([rifeStart, compatibleBlendStart]), [true, true]);
  assert.equal(rife.gpuRifeCapable(), true);
  await rife.destroyGpuInterp();

  // Identical standalone calls share one module-level candidate.
  const blendGate = deferred();
  state.initGates.push(blendGate);
  const firstBlend = rife.initGpuBlendStandalone({ log() {}, warn() {} });
  const secondBlend = rife.initGpuBlendStandalone({ log() {}, warn() {} });
  await waitFor(() => state.instances.length === 2, "standalone helper initialization did not start");
  assert.equal(state.instances.length, 2);
  blendGate.resolve();
  assert.deepEqual(await Promise.all([firstBlend, secondBlend]), [true, true]);
  assert.equal(state.instances[1].destroyed, false);
  await rife.destroyGpuInterp();

  // Teardown cancels an unpublished candidate, waits for its physical destroy,
  // and prevents the next generation from allocating until that fence resolves.
  const staleInitGate = deferred();
  state.initGates.push(staleInitGate);
  const staleStart = rife.initGpuBlendStandalone({ log() {}, warn() {} });
  await waitFor(() => state.instances.length === 3, "stale helper initialization did not start");
  const stale = state.instances[2];
  const staleDestroyGate = deferred();
  state.destroyGates.set(stale.id, staleDestroyGate);
  const retirement = rife.destroyGpuInterp();
  const replacementStart = rife.initGpuBlendStandalone({ log() {}, warn() {} });
  staleInitGate.resolve();
  await waitFor(() => stale.destroyCalls === 1, "stale candidate was not destroyed");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.instances.length, 3, "replacement overtook stale physical teardown");
  staleDestroyGate.resolve();
  await retirement;
  assert.equal(await staleStart, false);
  assert.equal(await replacementStart, true);
  assert.equal(state.instances.length, 4);
  assert.equal(rife.gpuActive(), true);

  // An incompatible live standalone helper is also physically retired before
  // the RIFE-capable replacement is constructed.
  const standalone = state.instances[3];
  const transitionDestroyGate = deferred();
  state.destroyGates.set(standalone.id, transitionDestroyGate);
  const transition = rife.initGpuInterp({ log() {}, warn() {} });
  await waitFor(() => standalone.destroyCalls === 1, "standalone transition teardown did not start");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.instances.length, 4, "RIFE replacement overtook standalone teardown");
  transitionDestroyGate.resolve();
  assert.equal(await transition, true);
  assert.equal(state.instances.length, 5);
  assert.equal(rife.gpuRifeCapable(), true);
  assert.equal(state.instances[4].device, targetDevice);

  // Loss while a same-device helper is still unpublished must claim that
  // candidate and include its physical destroy in the session invalidation
  // barrier. Recovery cannot otherwise safely create a new ORT/device generation.
  await rife.destroyGpuInterp();
  const lossInitGate = deferred();
  state.initGates.push(lossInitGate);
  const lossStart = rife.initGpuInterp({ log() {}, warn() {} });
  await waitFor(() => state.instances.length === 6, "loss candidate initialization did not start");
  const lossCandidate = state.instances[5];
  const lossDestroyGate = deferred();
  state.destroyGates.set(lossCandidate.id, lossDestroyGate);
  const invalidation = rife.invalidateDevice(targetDevice, { message: "adapter reset" });
  assert.equal(rife.isReady(), false, "session loss must unpublish readiness synchronously");
  let invalidationFinished = false;
  invalidation.then(() => { invalidationFinished = true; });

  lossInitGate.resolve();
  await waitFor(() => lossCandidate.destroyCalls === 1,
    "lost unpublished candidate did not enter physical teardown");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(invalidationFinished, false,
    "session invalidation must not overtake pending-candidate destruction");

  lossDestroyGate.resolve();
  assert.equal(await lossStart, false);
  assert.equal(await invalidation, true);
  assert.equal(session.releases, 1);

  await rife.disposeRife();
  assert.equal(rife.gpuActive(), false);
  assert.equal(session.releases, 1);
});

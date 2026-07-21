import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../fsrcnnx-main.js", import.meta.url);
let moduleRevision = 0;

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing main-module start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing main-module end marker: ${endMarker}`);
  return source.slice(start, end);
}

async function loadInterpolationLifecycle(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(
    original,
    "let interpolator = null, interpolatorInitPromise = null;",
    "// ===== INVERTED CHAIN",
  );
  const dynamicImport = 'import(chrome.runtime.getURL("fsrcnnx-interpolate.js"))';
  const injected = production.replace(
    dynamicImport,
    "globalThis.__mainLifecycleTestDeps.loadInterpolatorModule()",
  );
  assert.notEqual(injected, production, "interpolator loader injection must match production source");

  const harness = `
    const findVideo = () => null;
    const log = () => {};
    const warn = () => {};
    const chainTap = () => {};
    const chainInfo = () => null;
    const chainAvailable = () => false;
    const chainDevice = () => null;
    const adoptChainDevice = async () => false;
    const chainTargetDims = () => null;
    const chainUpscaleTex = () => false;
    const setChainInverted = () => false;
    const saveSitePrefs = () => {};
    const setTimeout = (...args) => globalThis.__mainLifecycleTestDeps.setTimeout
      ? globalThis.__mainLifecycleTestDeps.setTimeout(...args)
      : globalThis.setTimeout(...args);
    let engine = "fsrcnnx";
    let autoEnableGeneration = 0;
    let interpPausedByNeural = false;
    let interpAutoFallbackPref = true;
    let interpLadderPref = false;
    let interpInvertPref = true;
    function cancelAutoEnable() { autoEnableGeneration++; }
    function pauseInterpolationForNeural() { interpPausedByNeural = true; }
    ${injected}
    export { scheduleInterpolatorGpuRestart };
  `;
  globalThis.__mainLifecycleTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

async function loadAdoptionCoordinator(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(
    original,
    "let adopting = false, adoptionPromise = null, adoptionTarget = null;",
    "async function adoptChainDeviceInternal",
  );
  const internalCall = "adoptChainDeviceInternal(extDevice, isRequestCurrent)";
  const injected = production.replace(
    internalCall,
    "globalThis.__mainLifecycleTestDeps.adoptChainDeviceInternal(extDevice, isRequestCurrent)",
  );
  assert.notEqual(injected, production, "device-adoption injection must match production source");

  const harness = `
    let device = null;
    ${injected}
  `;
  globalThis.__mainLifecycleTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

async function loadModelLifecycle(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(original, "const srcCache =", "// Chains N ArtCnnModel stages");
  const harness = `
    const MODEL_FILES = ["model-a", "model-b", "model-c", "model-d"];
    let device = globalThis.__mainLifecycleTestDeps.device;
    let models = [], activeModel = null;
    let modelsDevice = null, modelLoadPromise = null, modelLoadDevice = null;
    const FsrcnnxModel = globalThis.__mainLifecycleTestDeps.FsrcnnxModel;
    const ArtCnnModel = globalThis.__mainLifecycleTestDeps.ArtCnnModel;
    const fetch = (...args) => globalThis.__mainLifecycleTestDeps.fetch(...args);
    const chrome = { runtime: { getURL: (path) => path } };
    const log = () => {};
    ${production}
    export { loadModels, ensureHiStages, ensureArtStages };
    export function setDevice(next) { device = next; }
    export function state() { return { models, modelsDevice, hiStages, artStages }; }
  `;
  globalThis.__mainLifecycleTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

async function loadAutoEnableLifecycle(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(original, "let autoEnableTimer = null", "export function getStatus");
  const harness = `
    let mode = "off";
    let optInterpolate = false;
    const findVideo = () => globalThis.__mainLifecycleTestDeps.findVideo();
    const probeVideo = () => "ok";
    const setMode = (...args) => globalThis.__mainLifecycleTestDeps.setMode(...args);
    const setInterpolate = (...args) => globalThis.__mainLifecycleTestDeps.setInterpolate(...args);
    const setTimeout = (...args) => globalThis.__mainLifecycleTestDeps.setTimeout(...args);
    const clearTimeout = (...args) => globalThis.__mainLifecycleTestDeps.clearTimeout(...args);
    const log = () => {};
    const warn = () => {};
    const siteHost = () => "test.invalid";
    ${production}
    export { scheduleAutoEnable, cancelAutoEnable };
  `;
  globalThis.__mainLifecycleTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

test("model selection during the shared interpolation import does not lose the pending enable", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });

  const loadStarted = deferred();
  const releaseModule = deferred();
  let instance = null;

  class FakeInterpolator {
    constructor() {
      instance = this;
      this.running = false;
      this.engine = null;
      this.startCalls = 0;
    }
    setInterpEngine(key) { this.engine = key; }
    setAutoFallback() {}
    setLadder() {}
    async start() {
      this.startCalls++;
      this.running = true;
      return { ok: true, running: true };
    }
    stop() { this.running = false; }
    getStats() { return { running: this.running, engine: this.engine }; }
  }

  globalThis.__mainLifecycleTestDeps = {
    loadInterpolatorModule: async () => {
      loadStarted.resolve();
      await releaseModule.promise;
      return { Interpolator: FakeInterpolator };
    },
  };
  const lifecycle = await loadInterpolationLifecycle(globalThis.__mainLifecycleTestDeps);

  const pendingEnable = lifecycle.setInterpolate(true);
  await loadStarted.promise;
  assert.deepEqual(
    await lifecycle.setInterpolateModel("rife_orig"),
    { ok: true, model: "rife_orig", pending: true },
  );
  releaseModule.resolve();

  const enabled = await pendingEnable;
  assert.equal(enabled.ok, true);
  assert.ok(instance, "the shared import should construct an interpolator");
  assert.equal(instance.engine, "rife_orig");
  assert.equal(instance.running, true);
  assert.equal(instance.startCalls, 1);
  assert.deepEqual(lifecycle.getInterpolateStats(), { running: true, engine: "rife_orig" });
});

test("same-target adoption retries with a fresh predicate after a stale coalesced request", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });

  const firstStarted = deferred();
  const releaseFirst = deferred();
  let firstRequestCurrent = true;
  let calls = 0;
  const deps = {
    adoptChainDeviceInternal: async (_target, isRequestCurrent) => {
      calls++;
      if (calls === 1) {
        firstStarted.resolve();
        await releaseFirst.promise;
      }
      return typeof isRequestCurrent !== "function" || isRequestCurrent();
    },
  };
  const { adoptChainDevice } = await loadAdoptionCoordinator(deps);
  const target = {};

  const staleAdoption = adoptChainDevice(target, () => firstRequestCurrent);
  await firstStarted.promise;
  firstRequestCurrent = false;
  const currentAdoption = adoptChainDevice(target, () => true);
  releaseFirst.resolve();

  assert.equal(await staleAdoption, false);
  assert.equal(await currentAdoption, true);
  assert.equal(calls, 2, "the current request must perform a fresh adoption attempt");
});

test("a delayed GPU-error restart cannot resurrect interpolation after disable", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const timers = [];
  let instance;
  class FakeInterpolator {
    constructor() { instance = this; this.running = false; this.startCalls = 0; }
    setAutoFallback() {}
    setLadder() {}
    async start() { this.running = true; this.startCalls++; return { ok: true }; }
    stop() { this.running = false; }
    getStats() { return { running: this.running }; }
  }
  const deps = {
    setTimeout(callback) { timers.push(callback); return timers.length; },
    loadInterpolatorModule: async () => ({ Interpolator: FakeInterpolator }),
  };
  const lifecycle = await loadInterpolationLifecycle(deps);
  assert.equal((await lifecycle.setInterpolate(true)).ok, true);
  assert.equal(instance.startCalls, 1);

  lifecycle.scheduleInterpolatorGpuRestart();
  assert.equal(instance.running, false);
  assert.equal(timers.length, 1);
  await lifecycle.setInterpolate(false);
  await timers[0]();

  assert.equal(instance.running, false);
  assert.equal(instance.startCalls, 1);
});

test("cancelled restored-state polling cannot auto-enable either feature", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const timers = new Map();
  let nextTimer = 0;
  let modeCalls = 0;
  let interpolationCalls = 0;
  const deps = {
    findVideo: () => ({}),
    setMode: async () => { modeCalls++; return { ok: true }; },
    setInterpolate: async () => { interpolationCalls++; return { ok: true }; },
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
  };
  const lifecycle = await loadAutoEnableLifecycle(deps);
  lifecycle.scheduleAutoEnable("upscale", true);
  const delayedCallback = [...timers.values()][0];
  lifecycle.cancelAutoEnable();
  await delayedCallback();

  assert.equal(modeCalls, 0);
  assert.equal(interpolationCalls, 0);

  const source = await readFile(mainUrl, "utf8");
  assert.match(source, /export async function setMode\([^)]*\)[\s\S]{0,250}cancelAutoEnable\(\)/);
  assert.match(source, /export async function setInterpolate\([^)]*\)[\s\S]{0,250}cancelAutoEnable\(\)/);
});

test("base model loading is single-flight, ordered, and atomically device-scoped", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const releaseFetches = deferred();
  const created = [];
  const deviceA = { name: "A" };
  class FakeModel {
    constructor(device, manifest) { this.device = device; this.name = manifest.name; created.push(this); }
    destroy() { this.destroyed = true; }
  }
  const calls = [];
  const deps = {
    device: deviceA,
    FsrcnnxModel: FakeModel,
    ArtCnnModel: FakeModel,
    async fetch(url) {
      calls.push(url);
      await releaseFetches.promise;
      const name = url.match(/model\/([^/.]+)/)?.[1] || url;
      return { ok: true, status: 200, json: async () => ({ name }), text: async () => `wgsl:${name}` };
    },
  };
  const lifecycle = await loadModelLifecycle(deps);
  const first = lifecycle.loadModels();
  const second = lifecycle.loadModels();
  await Promise.resolve();
  assert.equal(calls.length, 8);
  assert.equal(lifecycle.state().models.length, 0);
  releaseFetches.resolve();
  const [a, b] = await Promise.all([first, second]);

  assert.equal(a, b);
  assert.deepEqual(a.map((model) => model.name), ["model-a", "model-b", "model-c", "model-d"]);
  assert.ok(a.every((model) => model.device === deviceA));
  assert.equal(created.length, 4);
  assert.equal(lifecycle.state().modelsDevice, deviceA);
});

test("a device replacement discards a stale base-model load before construction", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const releaseFirstFetch = deferred();
  const deviceA = { name: "A" };
  const deviceB = { name: "B" };
  const created = [];
  let delayFetches = true;
  class FakeModel {
    constructor(owner, manifest) { this.device = owner; this.name = manifest.name; created.push(this); }
    destroy() { this.destroyed = true; }
  }
  const deps = {
    device: deviceA,
    FsrcnnxModel: FakeModel,
    ArtCnnModel: FakeModel,
    async fetch(url) {
      if (delayFetches) await releaseFirstFetch.promise;
      const name = url.match(/model\/([^/.]+)/)?.[1] || url;
      return { ok: true, status: 200, json: async () => ({ name }), text: async () => `wgsl:${name}` };
    },
  };
  const lifecycle = await loadModelLifecycle(deps);
  const staleLoad = lifecycle.loadModels();
  await Promise.resolve();
  lifecycle.setDevice(deviceB);
  delayFetches = false;
  releaseFirstFetch.resolve();

  await assert.rejects(staleLoad, /superseded by device change/);
  assert.equal(created.length, 0);
  assert.equal(lifecycle.state().models.length, 0);

  const current = await lifecycle.loadModels();
  assert.equal(current.length, 4);
  assert.ok(current.every((model) => model.device === deviceB));
  assert.equal(lifecycle.state().modelsDevice, deviceB);
});

test("chained model builders coalesce concurrent requests and capture their device", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const device = { name: "shared" };
  const created = [];
  class FakeModel {
    constructor(owner, manifest) { this.device = owner; this.name = manifest.name; created.push(this); }
    destroy() { this.destroyed = true; }
  }
  const fetchCalls = [];
  const deps = {
    device,
    FsrcnnxModel: FakeModel,
    ArtCnnModel: FakeModel,
    async fetch(url) {
      fetchCalls.push(url);
      const name = url.match(/model\/([^/.]+)/)?.[1] || url;
      return { ok: true, status: 200, json: async () => ({ name }), text: async () => `wgsl:${name}` };
    },
  };
  const lifecycle = await loadModelLifecycle(deps);
  const [highA, highB] = await Promise.all([
    lifecycle.ensureHiStages(2),
    lifecycle.ensureHiStages(2),
  ]);
  assert.equal(highA.length, 2);
  assert.equal(highB.length, 2);
  assert.equal(fetchCalls.length, 2);
  assert.equal(created.length, 2);

  const [artA, artB] = await Promise.all([
    lifecycle.ensureArtStages("ArtCNN_Test", 3),
    lifecycle.ensureArtStages("ArtCNN_Test", 3),
  ]);
  assert.equal(artA.length, 3);
  assert.equal(artB.length, 3);
  assert.equal(fetchCalls.length, 4);
  assert.equal(created.length, 5);
  assert.ok(created.every((model) => model.device === device));
});

test("a chained-model source load cannot commit stages after its device changes", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const releaseSource = deferred();
  const deviceA = { name: "A" };
  const deviceB = { name: "B" };
  const created = [];
  class FakeModel {
    constructor(owner) { this.device = owner; created.push(this); }
    destroy() { this.destroyed = true; }
  }
  const deps = {
    device: deviceA,
    FsrcnnxModel: FakeModel,
    ArtCnnModel: FakeModel,
    async fetch() {
      await releaseSource.promise;
      return { ok: true, status: 200, json: async () => ({ name: "high" }), text: async () => "wgsl" };
    },
  };
  const lifecycle = await loadModelLifecycle(deps);
  const staleBuild = lifecycle.ensureHiStages(2);
  await Promise.resolve();
  lifecycle.setDevice(deviceB);
  releaseSource.resolve();

  await assert.rejects(staleBuild, /superseded by device change/);
  assert.equal(created.length, 0);
  assert.equal(lifecycle.state().hiStages.length, 0);

  const current = await lifecycle.ensureHiStages(2);
  assert.equal(current.length, 2);
  assert.ok(current.every((model) => model.device === deviceB));
});

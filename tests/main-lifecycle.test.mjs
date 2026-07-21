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
  const internalCall = "adoptChainDeviceInternal(extDevice, isRequestCurrent, { preserveModeOnFailure })";
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

async function loadScaleSelection() {
  const original = await readFile(mainUrl, "utf8");
  const production = section(
    original,
    "let _scaleHeld, _scalePending = null, _scalePendingSince = 0;",
    "let sharpenEnabled = false",
  );
  const harness = `
    ${production}
    export { resetScaleSelection };
    export function seed() {
      _scaleHeld = { scale: 4 }; _scalePending = { scale: 2 }; _scalePendingSince = 99;
      _scaleHeldSrcW = 1920; _scaleHeldSrcH = 1080; _scaleLockLogged = true;
    }
    export function state() {
      return { _scaleHeld, _scalePending, _scalePendingSince,
        _scaleHeldSrcW, _scaleHeldSrcH, _scaleLockLogged };
    }
  `;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

async function loadChainUpscaleBoundary(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(original, "export function chainUpscaleTex", "export async function setInterpolateInvert");
  const harness = `
    const deps = globalThis.__mainLifecycleTestDeps;
    let device = {}, mode = "upscale", _texSource = null;
    const ensureTexPipelines = () => deps.ensureTexPipelines();
    const renderUpscale = () => deps.renderUpscale();
    const warn = (...args) => deps.warnings.push(args);
    ${production}
    export function source() { return _texSource; }
  `;
  globalThis.__mainLifecycleTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++moduleRevision}`);
}

async function loadAdoptionInternal(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = section(original, "async function adoptChainDeviceInternal", "function ensureLumaTexture");
  const harness = `
    const deps = globalThis.__mainLifecycleTestDeps;
    const warn = (...args) => deps.warnings.push(args);
    const log = (...args) => deps.logs.push(args);
    let device = deps.oldDevice || null, deviceOwnedByMain = !!deps.oldOwned;
    let adoptionGeneration = 1, adopting = false;
    let optImages = false, mode = "upscale", engine = "fsrcnnx", chainDepth = 1;
    let artVariant = "ArtCNN_C4F32", interpInvertPref = true, chainInverted = false;
    let _gpuErrWinStart = 0, _gpuErrCount = 0, _invRestarts = 0, _invRestartLast = 0;
    let canvas = { style: {} }, ro = {};
    const saveSitePrefs = () => {};
    const scheduleInterpolatorGpuRestart = () => {};
    const invalidateMainDeviceResources = () => { deps.invalidations++; };
    const watchDeviceLoss = (owner) => deps.watchDeviceLoss(owner, {
      replace(next) { adoptionGeneration++; device = next; },
    });
    const buildCore = () => deps.buildCore();
    const loadModels = async () => {};
    const ensureArtStages = async () => {};
    const ensureHiStages = async () => {};
    const ensureImageUpscaler = async () => null;
    const attach = () => {};
    const deactivateRendering = () => { deps.deactivations++; mode = "off"; };
    ${production}
    export function adopt(extDevice, options) {
      return adoptChainDeviceInternal(extDevice, null, options);
    }
    export function state() { return { device, mode, adopting, adoptionGeneration }; }
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

async function loadRendererResourceHelpers(deps) {
  const original = await readFile(mainUrl, "utf8");
  const production = [
    section(original, "function ensureLumaTexture", "// Lazily build the texture-ingest twins"),
    section(original, "function ensureTexPipelines", "function renderUpscale"),
    section(original, "function ensureDebandInter", "function ensureDebandPipelines"),
    section(original, "function ensureHiRGB", "function ensureChainTapTexture"),
    section(original, "function ensureChainTapTexture", "const PASSTHROUGH_WGSL"),
  ].join("\n");
  const harness = `
    let device = globalThis.__mainLifecycleTestDeps.device;
    const textureSizeAllowed = () => true;
    const LUMA_EXTRACT_WGSL = "texture_external textureSampleBaseClampToEdge(s, t)";
    const RECOMBINE_WGSL = "texture_external textureSampleBaseClampToEdge(s, t)";
    const GPUTextureUsage = { STORAGE_BINDING:1, TEXTURE_BINDING:2, RENDER_ATTACHMENT:4, COPY_SRC:8, COPY_DST:16 };
    let format = "rgba8unorm";
    let lumaTexture = null, lumaW = 0, lumaH = 0;
    let hiRGB = null, hiRGBW = 0, hiRGBH = 0;
    let debandInterTex = null, debandInterW = 0, debandInterH = 0;
    let chainTapTex = null;
    let extractPipelineTex = null, recombinePipelineTex = null, recombine16PipelineTex = null;
    ${production}
    export { ensureLumaTexture, ensureHiRGB, ensureDebandInter, ensureChainTapTexture, ensureTexPipelines };
    export function state() { return { lumaTexture, lumaW, lumaH, hiRGB, hiRGBW, hiRGBH,
      debandInterTex, debandInterW, debandInterH, chainTapTex,
      extractPipelineTex, recombinePipelineTex, recombine16PipelineTex }; }
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

test("renderer texture helpers preserve their old generation when replacement allocation fails", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const events = { textures: 0, failTextureAt: -1 };
  const device = {
    createTexture(description) {
      events.textures++;
      if (events.textures === events.failTextureAt) throw new Error("injected texture failure");
      const size = Array.isArray(description.size) ? description.size : [description.size.width, description.size.height];
      return { width: size[0], height: size[1], destroyed: 0, destroy() { this.destroyed++; } };
    },
    createShaderModule() { return {}; },
    createComputePipeline() { return {}; },
    createRenderPipeline() { return {}; },
  };
  const helpers = await loadRendererResourceHelpers({ device });
  const specs = [
    { fn: helpers.ensureLumaTexture, key: "lumaTexture", dims: ["lumaW", "lumaH"] },
    { fn: helpers.ensureHiRGB, key: "hiRGB", dims: ["hiRGBW", "hiRGBH"] },
    { fn: helpers.ensureDebandInter, key: "debandInterTex", dims: ["debandInterW", "debandInterH"] },
  ];
  for (const spec of specs) {
    assert.equal(spec.fn(16, 12), true);
    const old = helpers.state()[spec.key];
    events.failTextureAt = events.textures + 1;
    assert.throws(() => spec.fn(20, 14), /injected texture failure/);
    const failed = helpers.state();
    assert.equal(failed[spec.key], old);
    assert.deepEqual(spec.dims.map((key) => failed[key]), [16, 12]);
    assert.equal(old.destroyed, 0);
    events.failTextureAt = -1;
    assert.equal(spec.fn(20, 14), true);
    assert.equal(old.destroyed, 1);
  }

  const oldTap = helpers.ensureChainTapTexture(16, 12);
  events.failTextureAt = events.textures + 1;
  assert.throws(() => helpers.ensureChainTapTexture(20, 14), /injected texture failure/);
  assert.equal(helpers.state().chainTapTex, oldTap);
  assert.equal(oldTap.destroyed, 0);
  events.failTextureAt = -1;
  assert.equal(helpers.ensureChainTapTexture(20, 14).width, 20);
  assert.equal(oldTap.destroyed, 1);
});

test("texture-ingest pipelines publish only as a complete retryable generation", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  let pipelines = 0;
  let failAt = 2;
  const device = {
    createTexture() { throw new Error("unused"); },
    createShaderModule() { return {}; },
    createComputePipeline() {
      pipelines++;
      if (pipelines === failAt) throw new Error("injected pipeline failure");
      return { kind: "compute" };
    },
    createRenderPipeline() {
      pipelines++;
      if (pipelines === failAt) throw new Error("injected pipeline failure");
      return { kind: "render" };
    },
  };
  const helpers = await loadRendererResourceHelpers({ device });
  assert.throws(() => helpers.ensureTexPipelines(), /injected pipeline failure/);
  assert.deepEqual(
    [helpers.state().extractPipelineTex, helpers.state().recombinePipelineTex, helpers.state().recombine16PipelineTex],
    [null, null, null],
  );
  failAt = -1;
  assert.equal(helpers.ensureTexPipelines(), true);
  assert.ok(helpers.state().extractPipelineTex);
  assert.ok(helpers.state().recombinePipelineTex);
  assert.ok(helpers.state().recombine16PipelineTex);
});

test("scale selection reset clears held models across configuration changes", async () => {
  const selection = await loadScaleSelection();
  selection.seed();
  selection.resetScaleSelection();
  assert.deepEqual(selection.state(), {
    _scaleHeld: undefined,
    _scalePending: null,
    _scalePendingSince: 0,
    _scaleHeldSrcW: 0,
    _scaleHeldSrcH: 0,
    _scaleLockLogged: false,
  });

  const source = await readFile(mainUrl, "utf8");
  for (const [startMarker, endMarker] of [
    ["export function setEngine", "export function setArtVariant"],
    ["export function setArtVariant", "export function setHoverReveal"],
    ["export function setPolicy", "// Restore saved preferences"],
  ]) {
    assert.match(section(source, startMarker, endMarker), /resetScaleSelection\(\)/);
  }
  const renderSelection = section(source, "// Model-owned intermediates", "if (!activeModel) {");
  assert.equal(
    [...renderSelection.matchAll(/modelFitsProcessingBudget\(/g)].length >= 2,
    true,
    "the final hysteresis-selected model must receive a second budget preflight",
  );
});

test("chain texture presentation catches pipeline construction failures and retries", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  let attempts = 0;
  let renders = 0;
  const deps = {
    warnings: [],
    ensureTexPipelines() {
      attempts++;
      if (attempts === 1) throw new Error("transient compile failure");
      return true;
    },
    renderUpscale() { renders++; },
  };
  const boundary = await loadChainUpscaleBoundary(deps);
  const texture = { _w: 16, _h: 9 };

  assert.equal(boundary.chainUpscaleTex(texture), false);
  assert.equal(boundary.source(), null);
  assert.equal(deps.warnings.length, 1);
  assert.equal(boundary.chainUpscaleTex(texture), true);
  assert.equal(renders, 1);
});

test("recovery-owned adoption failure preserves mode for the retry coordinator", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const deps = {
    warnings: [], logs: [], invalidations: 0, deactivations: 0,
    buildCore() { throw new Error("transient pipeline build failure"); },
    watchDeviceLoss() {},
  };
  const adoption = await loadAdoptionInternal(deps);
  const external = { addEventListener() {} };

  assert.equal(await adoption.adopt(external, { preserveModeOnFailure: true }), false);
  assert.equal(adoption.state().mode, "upscale");
  assert.equal(adoption.state().device, null);
  assert.equal(deps.deactivations, 0);
});

test("loss during adoption cannot roll a replacement device back to stale state", async (t) => {
  const previousDeps = globalThis.__mainLifecycleTestDeps;
  t.after(() => { globalThis.__mainLifecycleTestDeps = previousDeps; });
  const replacement = { name: "replacement" };
  const deps = {
    warnings: [], logs: [], invalidations: 0, deactivations: 0,
    buildCore() {},
    watchDeviceLoss(_owner, coordinator) {
      Promise.resolve().then(() => coordinator.replace(replacement));
    },
  };
  const adoption = await loadAdoptionInternal(deps);
  const external = { addEventListener() {} };

  assert.equal(await adoption.adopt(external, { preserveModeOnFailure: true }), false);
  assert.equal(adoption.state().device, replacement);
  assert.equal(adoption.state().mode, "upscale");
  assert.equal(deps.deactivations, 0);
});

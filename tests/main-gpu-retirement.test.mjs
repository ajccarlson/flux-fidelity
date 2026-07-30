import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { CONTRACT_IMPORT } from "./helpers/setting-contract-import.mjs";

const mainUrl = new URL("../src/core/fsrcnnx-main.js", import.meta.url);
let revision = 0;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
}

async function loadRetirementHarness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "function runtimeGpuResourcesRequested()",
    "let webGpuInitPromise = null;",
  );
  const harness = `
    const deps = globalThis.__mainGpuRetirementDeps;
    let pageSuspended = false;
    let device = deps.device, deviceOwnedByMain = deps.owned;
    // This slice includes retirement/adoption, which retire and republish the GPU
    // frame timer. Timing is a diagnostic with no bearing on those lifecycles, so
    // the harness supplies an inert stand-in rather than the real module.
    const GpuFrameTimer = class { destroy() {} };
    let gpuTimer = new GpuFrameTimer();
    const publishGpuTimer = () => { gpuTimer = new GpuFrameTimer(); };
    let gpuResourceGeneration = 4, gpuResourcePhase = "active", gpuResourceReason = null;
    let gpuRetirementTail = Promise.resolve(), gpuRetirementPromise = null;
    let gpuAdapterPhase = "ready", gpuDevicePhase = "ready", gpuRecoveryPhase = "idle", gpuRecoveryAttempt = 0;
    let deviceRecoveryPromise = deps.recoveryDrain?.promise || null;
    let deviceLossInvalidationPromise = deps.lossInvalidationDrain?.promise || null;
    let adoptionGeneration = 2, adopting = false, imageUpscalerInitGeneration = 3;
    let webGpuInitPromise = null, adoptionPromise = null;
    let primaryAllocationRetirementPromise = null;
    let modelLoadPromise = null, fsrcnnxStageBuildPromise = null;
    let highStageBuildPromise = null, artStageBuildPromise = null;
    let imageUpscalerInitPromise = null, interpolatorInitPromise = null;
    const deviceRecoveryRequested = () => deps.demand;
    const boundedRuntimeDetail = (error, fallback = "cleanup failed") =>
      String(error?.message || error || fallback).slice(0, 240);
    const warn = (...args) => deps.events.push(["warn", ...args]);
    const cancelDeviceRecovery = () => deps.events.push("recovery-cancel");
    const pauseDeviceProducers = () => deps.events.push("producers-paused");
    const invalidateImageUpscaler = () => {
      deps.events.push("image-unpublished");
      return deps.imageDrain?.promise;
    };
    const invalidateMainDeviceResources = async () => {
      deps.events.push("main-invalidated");
      return deps.mainCleanup;
    };
    const detachDeviceErrorHandler = () => {};
    const notifyState = () => deps.events.push(["state", gpuResourcePhase, gpuDevicePhase]);
    const neuralEng = {
      stop: () => deps.events.push("neural-stop"),
      quiesce: () => deps.neuralDrain?.promise,
      dispose: async () => {
        deps.events.push("neural-disposed");
        if (deps.neuralDisposeError) throw deps.neuralDisposeError;
      },
    };
    const interpolator = {
      stop: () => deps.events.push("interpolator-stop"),
      retireGpuResources: () => deps.interpolationDrain?.promise,
      releaseModelResources: async () => {
        deps.events.push("interpolator-models-released");
        if (deps.interpolationModelError) throw deps.interpolationModelError;
      },
    };
    ${production}
    export function retire(reason) { return retireGpuResources(reason); }
    export function state() {
      return { device, deviceOwnedByMain, gpuResourceGeneration, gpuResourcePhase,
        gpuResourceReason, gpuDevicePhase, pending: !!gpuRetirementPromise };
    }
  `;
  globalThis.__mainGpuRetirementDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadInitializationHarness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const wrapper = section(source, "let webGpuInitPromise = null;", "function watchDeviceLoss(ownerDevice)");
  const internal = section(
    source,
    "async function initWebGPUInternal(",
    "// Create everything device-bound that initWebGPU used to build inline",
  );
  const harness = `
    const deps = globalThis.__mainGpuRetirementDeps;
    let device = null, deviceOwnedByMain = false;
    // initWebGPU requests timestamp-query when the adapter advertises it and
    // republishes the frame timer on success. Neither affects device publication,
    // which is what this slice tests, so both are stubbed inert.
    const GPU_TIMING_FEATURE = "timestamp-query";
    const publishGpuTimer = () => {};
    let demand = true;
    let gpuResourceGeneration = 0, gpuResourcePhase = "idle", gpuResourceReason = null;
    let gpuLastFailure = null;
    let gpuRetirementPromise = null;
    let gpuAdapterPhase = "unrequested", gpuDevicePhase = "uninitialized";
    const lostDevices = new WeakSet();
    const navigator = { gpu: {
      requestAdapter: (...args) => deps.requestAdapter(...args),
      getPreferredCanvasFormat: () => "rgba8unorm",
    } };
    const waitForGpuRetirement = () => gpuRetirementPromise || Promise.resolve();
    const runtimeGpuResourcesRequested = () => demand;
    const gpuInitializationCurrent = (generation) =>
      generation === gpuResourceGeneration && runtimeGpuResourcesRequested();
    const notifyState = () => deps.events.push(["state", gpuResourcePhase, gpuDevicePhase]);
    const setGpuFailure = (stage, code, error) => {
      gpuDevicePhase = "failed";
      gpuLastFailure = { stage, code, detail: error?.message || String(error) };
      deps.failures.push(gpuLastFailure);
    };
    const setGpuReady = () => {
      gpuAdapterPhase = "ready";
      gpuDevicePhase = "ready";
      gpuResourcePhase = "active";
      deps.events.push("ready");
    };
    const watchDeviceLoss = (owner) => owner.lost?.then(() => {
      lostDevices.add(owner);
      if (device === owner) device = null;
    });
    const ensureCanvas = () => {};
    const buildCore = () => deps.events.push("core-built");
    const invalidateMainDeviceResources = async () => deps.events.push("main-invalidated");
    const warn = (...args) => deps.events.push(["warn", ...args]);
    const log = (...args) => deps.events.push(["log", ...args]);
    ${wrapper}
    ${internal}
    export { initWebGPU };
    export function cancelDemand() { demand = false; gpuResourceGeneration++; }
    export function enableDemand() { demand = true; }
    export function state() {
      return { device, deviceOwnedByMain, gpuResourceGeneration, gpuResourcePhase,
        gpuAdapterPhase, gpuDevicePhase, gpuResourceReason, pending: !!webGpuInitPromise };
    }
  `;
  globalThis.__mainGpuRetirementDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadPrimaryRetirementHarness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "function retirePrimaryGpuAllocations(",
    "function cancelDeviceRecovery()",
  );
  const harness = `
    const deps = globalThis.__mainGpuRetirementDeps;
    const device = deps.device;
    const lostDevices = new WeakSet();
    let primaryAllocationRetirementTail = Promise.resolve();
    let primaryAllocationRetirementPromise = null;
    const sharedStage = { resetAllocation: () => deps.events.push("stage-reset") };
    let models = [sharedStage];
    let highStages = [sharedStage, { resetAllocation: () => deps.events.push("high-reset") }];
    let artStages = { current: [sharedStage, { resetAllocation: () => deps.events.push("art-reset") }] };
    let chainTapTex = deps.texture("chain"), chainTapFrame = 8, chainTapFailed = false;
    let lumaTexture = deps.texture("luma"), lumaW = 10, lumaH = 6;
    let hiRGB = deps.texture("hi"), hiRGBW = 20, hiRGBH = 12;
    let ssimds = { destroy: () => deps.events.push("ssim-reset") };
    let activeModel = sharedStage, chainedFsrcnnx = {}, chainedHigh = {}, chainedArt = {}, _texSource = {};
    let gpuResourceReason = null;
    const resetScaleSelection = () => deps.events.push("scale-reset");
    const resetPresentedRuntime = () => deps.events.push("presentation-reset");
    ${production}
    export function retire(reason) { return retirePrimaryGpuAllocations(reason); }
    export function state() {
      return { chainTapTex, lumaTexture, hiRGB, activeModel, chainedFsrcnnx,
        chainedHigh, chainedArt, texSource: _texSource, gpuResourceReason,
        pending: !!primaryAllocationRetirementPromise };
    }
  `;
  globalThis.__mainGpuRetirementDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadImageInitializationHarness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "async function ensureImageUpscaler(",
    "export async function setImages(",
  );
  const harness = `
    const deps = globalThis.__mainGpuRetirementDeps;
    let gpuRetirementPromise = null, adoptionPromise = null;
    let adopting = false, pageSuspended = false, gpuResourcePhase = "active";
    let optImages = true;
    let device = {}, format = "rgba8unorm", sampler = {};
    let imageUpscaler = null, imageUpscalerRetirementPromise = null;
    let imageUpscalerInitPromise = null, imageUpscalerInitDevice = null;
    let imageUpscalerInitToken = -1, imageUpscalerInitSelection = -1;
    let imageUpscalerInitGeneration = 0, imagesSelectionGeneration = 0;
    const imageUpscalerCurrent = (candidate, selection = imagesSelectionGeneration) =>
      candidate === imageUpscaler && candidate?.device === device && optImages &&
      selection === imagesSelectionGeneration;
    const invalidateImageUpscaler = async () => {
      imageUpscalerInitGeneration++;
      imageUpscaler = null;
      if (deps.invalidateGate) await deps.invalidateGate.promise;
    };
    const initWebGPU = async () => {
      deps.initStarted = true;
      return deps.initGate.promise;
    };
    const createImageUpscaler = async () => {
      deps.createCalls++;
      return { device };
    };
    ${production}
    export { ensureImageUpscaler };
    export function seedStale() { imageUpscaler = { device: {} }; }
    export function disable() {
      optImages = false;
      imagesSelectionGeneration++;
      imageUpscalerInitGeneration++;
    }
  `;
  globalThis.__mainGpuRetirementDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadModeSelectionHarness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(source, "export async function setMode(", "export function setEngine(");
  const harness = `
    const deps = globalThis.__mainGpuRetirementDeps;
    let modeSelectionGeneration = 0, preferenceRestoreGeneration = 0;
    let gpuRetirementPromise = deps.initialRetirement.promise;
    let mode = "off", protectedSource = false, protectedReason = null;
    let primaryController = null, video = null;
    const interpolator = null;
    const cancelPreferenceRestore = () => { preferenceRestoreGeneration++; };
    const deactivateRendering = async () => {
      deps.events.push("deactivate");
      mode = "off";
      const retirement = deps.offRetirement.promise;
      gpuRetirementPromise = retirement;
      await retirement;
      if (gpuRetirementPromise === retirement) gpuRetirementPromise = null;
    };
    const cancelMainLoop = () => deps.events.push("cancel-main-loop");
    const resetScaleSelection = () => deps.events.push("scale-reset");
    const saveSitePrefs = () => deps.events.push("save");
    const updateVideoMonitor = () => deps.events.push("monitor");
    const findVideo = () => ({ id: "video" });
    const queueVideoSelection = async () => {
      deps.events.push("select");
      return true;
    };
    ${production}
    export function state() { return { mode, modeSelectionGeneration }; }
  `;
  globalThis.__mainGpuRetirementDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

test("hard retirement unpublishes immediately, drains work, and destroys only a main-owned device", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const imageDrain = deferred();
  const neuralDrain = deferred();
  const interpolationDrain = deferred();
  const queueFence = deferred();
  const events = [];
  let destroys = 0;
  const device = {
    queue: { onSubmittedWorkDone: () => { events.push("queue-fence"); return queueFence.promise; } },
    destroy: () => { destroys++; events.push("device-destroyed"); },
  };
  const runtime = await loadRetirementHarness({
    device, owned: true, demand: false, events, imageDrain, neuralDrain, interpolationDrain,
  });

  const retirement = runtime.retire("all-features-off");
  assert.equal(runtime.state().device, null, "the retiring device must be unpublished synchronously");
  assert.equal(runtime.state().deviceOwnedByMain, false);
  assert.equal(runtime.state().gpuResourcePhase, "releasing");
  assert.equal(destroys, 0);

  imageDrain.resolve();
  neuralDrain.resolve();
  interpolationDrain.resolve();
  await waitFor(() => events.includes("queue-fence"), "retirement never reached the device fence");
  assert.equal(events.includes("main-invalidated"), false,
    "device-bound caches must survive until submitted work is complete");
  queueFence.resolve();
  assert.deepEqual(await retirement, { ok: true, released: true, reason: "all-features-off" });
  assert.equal(destroys, 1);
  assert.ok(events.indexOf("main-invalidated") < events.indexOf("device-destroyed"));
  assert.ok(events.indexOf("device-destroyed") < events.indexOf("neural-disposed"));
  assert.equal(runtime.state().gpuResourcePhase, "idle");
  assert.equal(runtime.state().gpuDevicePhase, "uninitialized");
  assert.equal(runtime.state().pending, false);
});

test("hard retirement relinquishes an adopted device without destroying it", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const events = [];
  let destroys = 0;
  const device = {
    queue: { onSubmittedWorkDone: async () => events.push("queue-fence") },
    destroy: () => { destroys++; },
  };
  const runtime = await loadRetirementHarness({
    device, owned: false, demand: false, events,
  });

  await runtime.retire("document-hidden");
  assert.equal(destroys, 0, "an ORT/provider-owned device must remain provider-owned");
  assert.equal(events.includes("main-invalidated"), true);
  assert.equal(events.includes("interpolator-models-released"), true);
  assert.equal(events.includes("neural-disposed"), true);
});

test("hard retirement drains an already-running recovery before fencing its device", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const recoveryDrain = deferred();
  const lossInvalidationDrain = deferred();
  const events = [];
  const device = {
    queue: { onSubmittedWorkDone: async () => events.push("queue-fence") },
    destroy: () => events.push("device-destroyed"),
  };
  const runtime = await loadRetirementHarness({
    device,
    owned: true,
    demand: false,
    events,
    recoveryDrain,
    lossInvalidationDrain,
  });

  const retirement = runtime.retire("all-features-off");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(events.includes("queue-fence"), false,
    "a detached recovery continuation may still submit work before it drains");

  recoveryDrain.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(events.includes("queue-fence"), false,
    "hard retirement must also drain the current cross-provider loss barrier");

  lossInvalidationDrain.resolve();
  await retirement;
  assert.ok(events.indexOf("queue-fence") < events.indexOf("device-destroyed"));
});

test("hard retirement reports cleanup failure but still releases every remaining owner", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const neuralDrain = deferred();
  const imageDrain = deferred();
  const events = [];
  const device = {
    queue: { onSubmittedWorkDone: async () => events.push("queue-fence") },
    destroy: () => events.push("device-destroyed"),
  };
  const runtime = await loadRetirementHarness({
    device,
    owned: true,
    demand: false,
    events,
    neuralDrain,
    imageDrain,
    neuralDisposeError: new Error("neural release failed"),
    interpolationModelError: new Error("RIFE release failed"),
    mainCleanup: {
      ok: false,
      errors: [{ resource: "display-texture", detail: "texture destroy failed" }],
    },
  });

  const retirement = runtime.retire("document-hidden");
  neuralDrain.reject(new Error("neural drain failed"));
  imageDrain.reject(new Error("image destroy failed"));
  const result = await retirement;

  assert.equal(result.ok, false);
  assert.equal(result.released, true);
  assert.equal(result.reason, "document-hidden");
  assert.deepEqual(result.errors, [
    { resource: "neural-quiesce", detail: "neural drain failed" },
    { resource: "image-retirement", detail: "image destroy failed" },
    { resource: "display-texture", detail: "texture destroy failed" },
    { resource: "interpolation-models", detail: "RIFE release failed" },
    { resource: "neural-session", detail: "neural release failed" },
  ]);
  assert.equal(events.includes("device-destroyed"), true);
  assert.equal(events.includes("interpolator-models-released"), true);
  assert.equal(events.includes("neural-disposed"), true);
  assert.equal(runtime.state().gpuResourcePhase, "idle");
});

test("a cancelled device request cannot publish, and a fresh generation can initialize", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const firstGate = deferred();
  const events = [];
  const failures = [];
  let firstDestroys = 0;
  let secondDestroys = 0;
  let requests = 0;
  const first = { lost: new Promise(() => {}), destroy: () => { firstDestroys++; } };
  const second = { lost: new Promise(() => {}), destroy: () => { secondDestroys++; } };
  const adapter = {
    features: { has: () => false },
    requestDevice: async () => {
      requests++;
      if (requests === 1) return firstGate.promise;
      return second;
    },
  };
  const runtime = await loadInitializationHarness({
    events, failures, requestAdapter: async () => adapter,
  });

  const stale = runtime.initWebGPU();
  await waitFor(() => requests === 1, "the first device request did not start");
  runtime.cancelDemand();
  firstGate.resolve(first);
  assert.equal(await stale, false);
  assert.equal(firstDestroys, 1);
  assert.equal(runtime.state().device, null);
  assert.equal(runtime.state().gpuResourcePhase, "idle");
  assert.equal(runtime.state().gpuResourceReason, "initialization-cancelled");
  assert.equal(runtime.state().gpuAdapterPhase, "unrequested");
  assert.equal(runtime.state().gpuDevicePhase, "uninitialized");
  assert.deepEqual(failures, [], "intentional cancellation is not a GPU failure");

  runtime.enableDemand();
  assert.equal(await runtime.initWebGPU(), true);
  assert.equal(runtime.state().device, second);
  assert.equal(runtime.state().deviceOwnedByMain, true);
  assert.equal(runtime.state().gpuResourcePhase, "active");
  assert.equal(secondDestroys, 0);
});

test("a request rejection after lifecycle cancellation resolves as cancellation without failure telemetry", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const gate = deferred();
  const events = [];
  const failures = [];
  const adapter = {
    features: { has: () => false },
    requestDevice: () => gate.promise,
  };
  const runtime = await loadInitializationHarness({
    events, failures, requestAdapter: async () => adapter,
  });

  const stale = runtime.initWebGPU();
  await new Promise((resolve) => setImmediate(resolve));
  runtime.cancelDemand();
  gate.reject(new Error("request aborted while hidden"));
  assert.equal(await stale, false);
  assert.deepEqual(failures, []);
  assert.equal(runtime.state().pending, false);
  assert.equal(runtime.state().device, null);
});

test("soft retirement fences source-sized allocations while preserving the healthy device", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const fence = deferred();
  const events = [];
  const device = {
    queue: { onSubmittedWorkDone: () => { events.push("queue-fence"); return fence.promise; } },
  };
  const runtime = await loadPrimaryRetirementHarness({
    device,
    events,
    texture: (name) => ({ destroy: () => events.push(["texture-destroy", name]) }),
  });

  const retirement = runtime.retire("video-off");
  assert.notEqual(runtime.state().lumaTexture, null,
    "allocations remain valid until prior submissions reach the queue fence");
  fence.resolve();
  assert.equal(await retirement, true);
  assert.equal(runtime.state().lumaTexture, null);
  assert.equal(runtime.state().hiRGB, null);
  assert.equal(runtime.state().chainTapTex, null);
  assert.equal(runtime.state().activeModel, null);
  assert.equal(runtime.state().texSource, null);
  assert.equal(runtime.state().gpuResourceReason, "video-off");
  assert.equal(events.filter((event) => event === "stage-reset").length, 1,
    "a stage shared by model pools is reset exactly once");
  // Three, not four: dispRGB was never allocated, so its retirement entry was a
  // phantom rather than a real texture destroy.
  assert.equal(events.filter((event) => Array.isArray(event) && event[0] === "texture-destroy").length, 3);
  assert.equal(runtime.state().pending, false);
});

test("image initialization resolving after Images Off cannot construct or publish a stale worker", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const initGate = deferred();
  const deps = { initGate, initStarted: false, createCalls: 0 };
  const runtime = await loadImageInitializationHarness(deps);

  const pending = runtime.ensureImageUpscaler();
  await waitFor(() => deps.initStarted, "image initialization never requested WebGPU");
  runtime.disable();
  initGate.resolve(true);

  assert.equal(await pending, null);
  assert.equal(deps.createCalls, 0,
    "a stale caller must re-check image demand before allocating device-bound resources");
});

test("image disable during stale-worker retirement cannot publish a replacement", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const invalidateGate = deferred();
  const deps = {
    initGate: { promise: Promise.resolve(true) },
    invalidateGate,
    initStarted: false,
    createCalls: 0,
  };
  const runtime = await loadImageInitializationHarness(deps);
  runtime.seedStale();

  const pending = runtime.ensureImageUpscaler();
  await new Promise((resolve) => setImmediate(resolve));
  runtime.disable();
  invalidateGate.resolve();

  assert.equal(await pending, null);
  assert.equal(deps.createCalls, 0,
    "selection intent must be revalidated after awaited stale-worker cleanup");
});

test("a newer Off request supersedes a mode enable waiting on older retirement", async (t) => {
  const previous = globalThis.__mainGpuRetirementDeps;
  t.after(() => { globalThis.__mainGpuRetirementDeps = previous; });
  const initialRetirement = deferred();
  const offRetirement = deferred();
  const deps = { initialRetirement, offRetirement, events: [] };
  const runtime = await loadModeSelectionHarness(deps);

  const enable = runtime.setMode("upscale", null, { persist: false });
  const disable = runtime.setMode("off", null, { persist: false });
  assert.deepEqual(deps.events, ["deactivate"]);

  initialRetirement.resolve();
  assert.deepEqual(await enable, { ok: false, reason: "superseded" });
  assert.equal(runtime.state().mode, "off");
  assert.equal(deps.events.includes("select"), false,
    "the stale enable request must not reconcile or publish a video renderer");

  offRetirement.resolve();
  assert.deepEqual(await disable, { ok: true });
  assert.equal(runtime.state().mode, "off");
});

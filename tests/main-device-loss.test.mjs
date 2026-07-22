import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../fsrcnnx-main.js", import.meta.url);
let revision = 0;

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function resource(counter) {
  return { destroy() { counter.count++; } };
}

async function flush(turns = 4) {
  for (let turn = 0; turn < turns; turn++) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function loadCoordinator(deps) {
  const source = await readFile(mainUrl, "utf8");
  const start = source.indexOf("function watchDeviceLoss(ownerDevice)");
  const end = source.indexOf("async function initWebGPUInternal(", start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const production = source.slice(start, end);
  const harness = `
    const deps = globalThis.__mainDeviceLossDeps;
    const GPU_RECOVERY_MAX_ATTEMPTS = 3;
    const setTimeout = (...args) => deps.setTimeout
      ? deps.setTimeout(...args)
      : globalThis.setTimeout(...args);
    const clearTimeout = (...args) => deps.clearTimeout
      ? deps.clearTimeout(...args)
      : globalThis.clearTimeout(...args);
    const log = (...args) => deps.logs.push(args);
    const warn = (...args) => deps.warnings.push(args);
    const watchedDeviceLosses = new WeakSet();
    const lostDevices = new WeakSet();
    let device = deps.device, deviceOwnedByMain = true;
    let adoptionGeneration = 0;
    let videoSelectionGeneration = 0;
    let imagesSelectionGeneration = 0;
    let deviceRecoveryGeneration = 0, deviceRecoveryPromise = null, deviceRecoveryTimer = null;
    const deviceLossInvalidations = new WeakMap();
    let deviceLossInvalidationPromise = null;
    let mode = deps.mode || "passthrough", optImages = deps.images === true;
    let optInterpolate = deps.interpolate === true, engine = deps.engine || "fsrcnnx";
    let engineSelectionGeneration = 0, chainDepth = 1, artVariant = "ArtCNN_C4F32";
    let adopting = false, pageSuspended = false;
    let interpolator = {
      _rifeMod: { invalidateDevice: (...args) => deps.invalidateRife(...args) },
      stop: () => { deps.interpolatorStops++; },
    };
    let context = deps.context, format = "rgba8unorm", canvas = deps.canvas;
    let gpuAdapterPhase = "ready", gpuDevicePhase = "ready", gpuRecoveryPhase = "idle";
    let gpuResourcePhase = "active", gpuResourceReason = null;
    let gpuRecoveryAttempt = 0, gpuLastFailure = null, gpuRecoveredAt = null;
    function boundedRuntimeDetail(error, fallback = "Unknown runtime failure") {
      const detail = error?.message || (typeof error === "string" ? error : fallback);
      return String(detail || fallback).replace(/\\s+/g, " ").trim().slice(0, 240);
    }
    function snapshotGpu() {
      return {
        adapter: gpuAdapterPhase,
        device: gpuDevicePhase,
        recovery: gpuRecoveryPhase,
        attempt: gpuRecoveryAttempt,
        lastFailure: gpuLastFailure ? { ...gpuLastFailure } : null,
        recoveredAt: gpuRecoveredAt,
      };
    }
    function notifyState() { deps.notifications.push(snapshotGpu()); }
    function setGpuFailure(stage, code, error,
      { adapter = gpuAdapterPhase, device: devicePhase = "failed" } = {}) {
      gpuAdapterPhase = adapter;
      gpuDevicePhase = devicePhase;
      gpuLastFailure = { stage, code, detail: boundedRuntimeDetail(error), at: Date.now() };
      deps.gpuFailures.push({ ...gpuLastFailure, adapter, device: devicePhase });
      notifyState();
    }
    function setGpuReady({ recovered = false } = {}) {
      gpuAdapterPhase = "ready";
      gpuDevicePhase = "ready";
      gpuResourcePhase = "active";
      gpuResourceReason = null;
      if (recovered || gpuRecoveryPhase === "idle") {
        gpuRecoveryPhase = "idle";
        gpuRecoveryAttempt = 0;
      }
      if (recovered) gpuRecoveredAt = Date.now();
      deps.gpuReadyCalls.push({ recovered });
      notifyState();
    }
    function resetPresentedRuntime() { deps.presentationResets++; return true; }
    let sampler = {}, extractPipeline = {}, recombinePipeline = {}, passthroughPipeline = {};
    let extractPipelineTex = {}, recombinePipelineTex = {}, recombine16PipelineTex = {};
    let recombine16Pipeline = {}, blitPipeline = {}, sharpenPipeline = {}, sharpenStrengthBuilt = 1;
    let chainTapTex = deps.resources.shift(), chainTapFrame = 3;
    let lumaTexture = deps.resources.shift(), lumaW = 2, lumaH = 2;
    let hiRGB = deps.resources.shift(), hiRGBW = 4, hiRGBH = 4;
    let dispRGB = deps.resources.shift(), dispRGBW = 4, dispRGBH = 4;
    let ssimds = { destroy: deps.onSsimDestroy };
    let models = [{ destroy: deps.onModelDestroy }], modelsDevice = device, activeModel = models[0];
    let highStages = [{ destroy: deps.onHighModelDestroy }], artStages = {};
    let chainedFsrcnnx = {}, chainedHigh = {}, chainedArt = {};
    let fsrcnnxLoadPending = true, highLoadPending = true, artLoadPending = true;
    let _scaleHeld = {}, _scalePending = {}, _scalePendingSince = 1, _texSource = {};
    function resetScaleSelection() { _scaleHeld = undefined; _scalePending = null; _scalePendingSince = 0; }
    let neuralEng = { invalidateDevice: (...args) => deps.invalidateNeural(...args) };
    function invalidateImageUpscaler() { deps.imageInvalidations++; }
    function detachDeviceErrorHandler() {}
    function clearMultiTargets() { deps.multiClears++; }
    async function initWebGPU() {
      deps.recoveries++;
      if (deps.recoveryResults) {
        const result = deps.recoveryResults.shift();
        if (result instanceof Error) throw result;
        if (result !== true) return false;
        device = deps.replacement;
        return true;
      }
      await deps.recoveryGate.promise;
      device = deps.replacement;
      return true;
    }
    async function loadModels() {}
    async function ensureFsrcnnxStages() {}
    async function ensureHighStages() {}
    async function ensureArtStages() {}
    async function ensureNeural() { deps.neuralEnsures++; device = deps.replacement; }
    function neuralSelectionCurrent() { return true; }
    async function ensureImageUpscaler() { return null; }
    function startImageUpscalerIfCurrent(upscaler) { upscaler?.start?.(); return !!upscaler; }
    function attach() { deps.attaches++; }
    function scheduleMainLoop() { deps.schedules++; }
    function cancelMainLoop() {}
    function findVideo() { return { id: "selected-video" }; }
    async function queueVideoSelection() {
      deps.selections++;
      if (deps.selectionGate) await deps.selectionGate.promise;
      if (!deviceRecoveryRequested() || pageSuspended) return false;
      canvas.style.display = "block";
      attach(); scheduleMainLoop(); return true;
    }
    function deactivateRendering() { mode = "off"; cancelDeviceRecovery(); }
    ${production}
    export function watch() { watchDeviceLoss(device); }
    export function setCurrent(next) { device = next; }
    export function setDemand(next = {}) {
      if (Object.prototype.hasOwnProperty.call(next, "mode")) mode = next.mode;
      if (Object.prototype.hasOwnProperty.call(next, "images")) optImages = next.images;
      if (Object.prototype.hasOwnProperty.call(next, "interpolate")) optInterpolate = next.interpolate;
      if (Object.prototype.hasOwnProperty.call(next, "engine")) engine = next.engine;
      reconcileDeviceRecoveryDemand();
    }
    export function turnOff() { mode = "off"; reconcileDeviceRecoveryDemand(); }
    export function state() {
      return { device, mode, images: optImages, interpolate: optInterpolate,
        recovering: !!deviceRecoveryPromise || deviceRecoveryTimer != null,
        providerInvalidating: !!deviceLossInvalidationPromise,
        display: canvas.style.display, gpu: snapshotGpu(),
        chainTapTex, lumaTexture, hiRGB, dispRGB, context, models, highStages };
    }
  `;
  globalThis.__mainDeviceLossDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadInitializer(deps) {
  const source = await readFile(mainUrl, "utf8");
  const start = source.indexOf("let webGpuInitPromise = null;");
  const end = source.indexOf("function watchDeviceLoss(ownerDevice)", start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const production = source.slice(start, end);
  const harness = `
    const deps = globalThis.__mainDeviceLossDeps;
    let device = null;
    const lostDevices = new WeakSet();
    let gpuRetirementPromise = null;
    let gpuResourceGeneration = 0;
    let gpuResourcePhase = "idle", gpuResourceReason = null;
    let gpuAdapterPhase = "unrequested", gpuDevicePhase = "uninitialized";
    let gpuRecoveryPhase = "idle", gpuRecoveryAttempt = 0;
    let gpuLastFailure = null, gpuRecoveredAt = null;
    function boundedRuntimeDetail(error, fallback = "Unknown runtime failure") {
      const detail = error?.message || (typeof error === "string" ? error : fallback);
      return String(detail || fallback).replace(/\\s+/g, " ").trim().slice(0, 240);
    }
    function snapshotGpu() {
      return {
        adapter: gpuAdapterPhase,
        device: gpuDevicePhase,
        recovery: gpuRecoveryPhase,
        attempt: gpuRecoveryAttempt,
        lastFailure: gpuLastFailure ? { ...gpuLastFailure } : null,
        recoveredAt: gpuRecoveredAt,
      };
    }
    function notifyState() { deps.notifications.push(snapshotGpu()); }
    function waitForGpuRetirement() { return Promise.resolve(); }
    function gpuInitializationCurrent() { return true; }
    function setGpuFailure(stage, code, error,
      { adapter = gpuAdapterPhase, device: devicePhase = "failed" } = {}) {
      gpuAdapterPhase = adapter;
      gpuDevicePhase = devicePhase;
      gpuLastFailure = { stage, code, detail: boundedRuntimeDetail(error), at: Date.now() };
      deps.gpuFailures.push({ ...gpuLastFailure, adapter, device: devicePhase });
      notifyState();
    }
    function setGpuReady({ recovered = false } = {}) {
      gpuAdapterPhase = "ready";
      gpuDevicePhase = "ready";
      if (recovered || gpuRecoveryPhase === "idle") {
        gpuRecoveryPhase = "idle";
        gpuRecoveryAttempt = 0;
      }
      if (recovered) gpuRecoveredAt = Date.now();
      deps.gpuReadyCalls.push({ recovered });
      notifyState();
    }
    async function initWebGPUInternal() {
      deps.calls++;
      await deps.gate.promise;
      const candidate = deps.device;
      candidate.lost.then(() => {
        lostDevices.add(candidate);
        if (device === candidate) device = null;
      });
      device = candidate;
      return true;
    }
    ${production}
    export { initWebGPU };
    export function state() { return { device, pending: !!webGpuInitPromise, gpu: snapshotGpu() }; }
  `;
  globalThis.__mainDeviceLossDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadGpuReadiness(deps) {
  const source = await readFile(mainUrl, "utf8");
  const start = source.indexOf("function setGpuReady(");
  const end = source.indexOf("function resetPresentedRuntime()", start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const production = source.slice(start, end);
  const harness = `
    const deps = globalThis.__mainDeviceLossDeps;
    let gpuAdapterPhase = deps.adapter;
    let gpuDevicePhase = deps.device;
    let gpuRecoveryPhase = deps.recovery;
    let gpuRecoveryAttempt = deps.attempt;
    let gpuRecoveredAt = deps.recoveredAt || null;
    let gpuResourcePhase = "idle", gpuResourceReason = null;
    const device = {};
    const lostDevices = new WeakSet();
    let deviceRecoveryGeneration = deps.generation || 0;
    let deviceRecoveryTimer = deps.timer || null;
    const clearTimeout = (timer) => {
      deps.clearedTimers.push(timer);
    };
    function notifyState() {
      deps.notifications.push({
        adapter: gpuAdapterPhase,
        device: gpuDevicePhase,
        recovery: gpuRecoveryPhase,
        attempt: gpuRecoveryAttempt,
        recoveredAt: gpuRecoveredAt,
        generation: deviceRecoveryGeneration,
        timer: deviceRecoveryTimer,
      });
    }
    ${production}
    export function publish(options) { setGpuReady(options); }
    export function state() {
      return {
        adapter: gpuAdapterPhase,
        device: gpuDevicePhase,
        recovery: gpuRecoveryPhase,
        attempt: gpuRecoveryAttempt,
        recoveredAt: gpuRecoveredAt,
        generation: deviceRecoveryGeneration,
        timer: deviceRecoveryTimer,
      };
    }
  `;
  globalThis.__mainDeviceLossDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

function setup({ mode = "passthrough", images = false, interpolate = false, engine = "fsrcnnx",
  recoveryResults = null, controlledTimers = false } = {}) {
  const loss = deferred();
  const recoveryGate = deferred();
  const destroyed = { count: 0 };
  const device = { lost: loss.promise };
  const replacement = { lost: new Promise(() => {}) };
  const deps = {
    mode,
    images,
    interpolate,
    engine,
    device,
    replacement,
    loss,
    recoveryGate,
    logs: [],
    warnings: [],
    resources: Array.from({ length: 4 }, () => resource(destroyed)),
    context: { unconfigure() { deps.unconfigured++; } },
    canvas: { style: { display: "block", opacity: "1" } },
    destroyed,
    recoveries: 0,
    selections: 0,
    attaches: 0,
    schedules: 0,
    unconfigured: 0,
    imageInvalidations: 0,
    multiClears: 0,
    modelDestroys: 0,
    highModelDestroys: 0,
    ssimDestroys: 0,
    neuralInvalidations: 0,
    rifeInvalidations: 0,
    neuralEnsures: 0,
    interpolatorStops: 0,
    notifications: [],
    gpuFailures: [],
    gpuReadyCalls: [],
    presentationResets: 0,
    recoveryResults: recoveryResults ? [...recoveryResults] : null,
    timers: [],
    onModelDestroy() { deps.modelDestroys++; },
    onHighModelDestroy() { deps.highModelDestroys++; },
    onSsimDestroy() { deps.ssimDestroys++; },
    async invalidateNeural() {
      deps.neuralInvalidations++;
      await deps.neuralInvalidationGate?.promise;
    },
    async invalidateRife() {
      deps.rifeInvalidations++;
      await deps.rifeInvalidationGate?.promise;
    },
  };
  if (controlledTimers) {
    deps.setTimeout = (callback) => {
      deps.timers.push(callback);
      return callback;
    };
    deps.clearTimeout = (callback) => {
      const index = deps.timers.indexOf(callback);
      if (index !== -1) deps.timers.splice(index, 1);
    };
  }
  return deps;
}

test("concurrent WebGPU initialization reports an immediately-lost device as failure", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const gate = deferred();
  const deps = {
    calls: 0,
    gate,
    device: { lost: Promise.resolve({ message: "already lost" }) },
    notifications: [],
    gpuFailures: [],
    gpuReadyCalls: [],
  };
  const initializer = await loadInitializer(deps);
  const first = initializer.initWebGPU();
  const second = initializer.initWebGPU();
  assert.equal(first, second);
  assert.equal(deps.calls, 1);
  gate.resolve();

  assert.equal(await first, false);
  assert.equal(await second, false);
  assert.equal(initializer.state().device, null);
  assert.equal(initializer.state().pending, false);
  assert.deepEqual(
    {
      adapter: initializer.state().gpu.adapter,
      device: initializer.state().gpu.device,
      recovery: initializer.state().gpu.recovery,
      attempt: initializer.state().gpu.attempt,
    },
    { adapter: "requesting", device: "failed", recovery: "idle", attempt: 0 },
  );
  assert.equal(initializer.state().gpu.lastFailure.stage, "device");
  assert.equal(initializer.state().gpu.lastFailure.code, "device-init-failed");
  assert.equal(deps.gpuFailures.length, 1);
  assert.equal(deps.gpuReadyCalls.length, 0);
  assert.equal(deps.notifications.length >= 2, true,
    "requesting and terminal failure must each be externally observable");
});

test("current device loss retires stale state and single-flights recovery", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  coordinator.watch();
  deps.loss.resolve({ reason: "unknown", message: "adapter reset" });
  await flush();

  assert.equal(deps.recoveries, 1);
  assert.equal(coordinator.state().mode, "passthrough");
  assert.deepEqual(
    {
      adapter: coordinator.state().gpu.adapter,
      device: coordinator.state().gpu.device,
      recovery: coordinator.state().gpu.recovery,
      attempt: coordinator.state().gpu.attempt,
    },
    { adapter: "ready", device: "lost", recovery: "running", attempt: 1 },
  );
  assert.equal(coordinator.state().gpu.lastFailure.code, "device-lost");
  assert.equal(coordinator.state().display, "none");
  assert.equal(deps.destroyed.count, 4);
  assert.equal(deps.modelDestroys, 1);
  assert.equal(deps.highModelDestroys, 1);
  assert.equal(coordinator.state().highStages.length, 0);
  assert.equal(deps.unconfigured, 1);
  assert.equal(deps.imageInvalidations, 1);
  assert.equal(deps.multiClears, 1);
  assert.equal(deps.neuralInvalidations, 1,
    "duplicate loss observation must reuse the neural invalidation barrier");
  assert.equal(deps.rifeInvalidations, 1,
    "the retained RIFE provider must join the same device-loss barrier");
  assert.deepEqual(
    [coordinator.state().chainTapTex, coordinator.state().lumaTexture,
      coordinator.state().hiRGB, coordinator.state().dispRGB],
    [null, null, null, null],
  );

  deps.recoveryGate.resolve();
  await flush();
  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(coordinator.state().mode, "passthrough");
  assert.equal(coordinator.state().display, "block");
  assert.equal(deps.attaches, 1);
  assert.equal(deps.schedules, 1);
  assert.deepEqual(
    {
      adapter: coordinator.state().gpu.adapter,
      device: coordinator.state().gpu.device,
      recovery: coordinator.state().gpu.recovery,
      attempt: coordinator.state().gpu.attempt,
    },
    { adapter: "ready", device: "ready", recovery: "idle", attempt: 0 },
  );
  assert.equal(coordinator.state().gpu.recoveredAt != null, true);
  assert.deepEqual(deps.gpuReadyCalls, [{ recovered: true }]);
});

test("device recovery cannot create either renderer provider until neural and RIFE releases settle", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });

  for (const scenario of [
    { label: "WebGPU", options: {}, starts: "recoveries" },
    {
      label: "neural",
      options: { mode: "upscale", engine: "neural" },
      starts: "neuralEnsures",
    },
  ]) {
    const deps = setup(scenario.options);
    deps.neuralInvalidationGate = deferred();
    deps.rifeInvalidationGate = deferred();
    const coordinator = await loadCoordinator(deps);
    coordinator.watch();
    deps.loss.resolve({ message: `${scenario.label} shared device lost` });
    await flush();

    assert.equal(deps.neuralInvalidations, 1, scenario.label);
    assert.equal(deps.rifeInvalidations, 1, scenario.label);
    assert.equal(deps.recoveries, 0, `${scenario.label} init overtook provider invalidation`);
    assert.equal(deps.neuralEnsures, 0, `${scenario.label} session creation overtook provider invalidation`);
    assert.equal(coordinator.state().providerInvalidating, true, scenario.label);

    deps.neuralInvalidationGate.resolve();
    await flush();
    assert.equal(deps.recoveries, 0,
      `${scenario.label} recovery started after neural cleanup alone`);
    assert.equal(deps.neuralEnsures, 0,
      `${scenario.label} session creation started before deferred RIFE cleanup`);
    assert.equal(coordinator.state().providerInvalidating, true, scenario.label);

    deps.rifeInvalidationGate.resolve();
    await flush();
    assert.equal(deps[scenario.starts], 1,
      `${scenario.label} recovery did not start after the combined barrier settled`);
    assert.equal(coordinator.state().providerInvalidating, false, scenario.label);
    assert.equal(deps.neuralInvalidations, 1,
      `${scenario.label} recovery must reuse, not repeat, neural invalidation`);
    assert.equal(deps.rifeInvalidations, 1,
      `${scenario.label} recovery must reuse, not repeat, RIFE invalidation`);

    deps.recoveryGate.resolve();
    await flush();
  }
});

test("a provider invalidation rejection is observed without a detached loss promise", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup({ mode: "off" });
  deps.invalidateRife = async () => {
    deps.rifeInvalidations++;
    throw new Error("deferred RIFE release failed");
  };
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "loss without recovery demand" });
  await flush();

  assert.equal(deps.neuralInvalidations, 1);
  assert.equal(deps.rifeInvalidations, 1);
  assert.equal(deps.recoveries, 0);
  assert.equal(coordinator.state().providerInvalidating, false);
  assert.equal(deps.warnings.some((args) =>
    args.join(" ").includes("rife-session device-loss invalidation failed")), true);
});

test("interpolation-only device loss remains recovery-eligible and reconciles video selection", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup({ mode: "off", interpolate: true });
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "interpolation device lost" });
  await flush();

  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().interpolate, true);
  assert.equal(deps.recoveries, 1,
    "an enabled interpolation pipeline still requires a replacement GPU device");
  assert.equal(coordinator.state().gpu.recovery, "running");

  deps.recoveryGate.resolve();
  await flush();
  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(deps.selections, 1,
    "post-recovery reconciliation must reselect the interpolation source while video mode is off");
  assert.equal(coordinator.state().gpu.recovery, "idle");
  assert.equal(coordinator.state().gpu.attempt, 0);
});

test("turning video off preserves recovery owned by image upscaling", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup({ mode: "passthrough", images: true });
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "shared device lost" });
  await flush();
  assert.equal(coordinator.state().gpu.recovery, "running");

  coordinator.setDemand({ mode: "off" });
  assert.equal(coordinator.state().images, true);
  assert.equal(coordinator.state().recovering, true,
    "the independent image consumer keeps the recovery attempt alive");
  assert.equal(coordinator.state().gpu.recovery, "running");

  deps.recoveryGate.resolve();
  await flush();
  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(coordinator.state().gpu.recovery, "idle");
  assert.deepEqual(deps.gpuReadyCalls, [{ recovered: true }]);
});

test("removing the last independent GPU consumer retires recovery immediately", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });

  for (const scenario of [
    { options: { mode: "off", images: true }, change: { images: false }, label: "images" },
    {
      options: { mode: "off", interpolate: true },
      change: { engine: "neural" },
      label: "standalone interpolation",
    },
  ]) {
    const deps = setup(scenario.options);
    const coordinator = await loadCoordinator(deps);
    coordinator.watch();
    deps.loss.resolve({ message: `${scenario.label} device lost` });
    await flush();
    assert.equal(coordinator.state().gpu.recovery, "running", scenario.label);

    coordinator.setDemand(scenario.change);
    assert.equal(coordinator.state().recovering, false, scenario.label);
    assert.equal(coordinator.state().gpu.recovery, "idle", scenario.label);
    assert.equal(coordinator.state().gpu.attempt, 0, scenario.label);

    deps.recoveryGate.resolve();
    await flush();
    assert.equal(coordinator.state().gpu.recovery, "idle", scenario.label);
    assert.equal(deps.selections, 0, scenario.label);
    assert.deepEqual(deps.gpuReadyCalls, [], scenario.label);
  }
});

test("healthy GPU publication clears stale scheduled and exhausted recovery telemetry", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });

  for (const recovery of ["scheduled", "exhausted"]) {
    const deps = {
      adapter: "ready",
      device: recovery === "exhausted" ? "failed" : "lost",
      recovery,
      attempt: recovery === "exhausted" ? 3 : 2,
      notifications: [],
      generation: 7,
      timer: { recovery },
      clearedTimers: [],
    };
    const readiness = await loadGpuReadiness(deps);
    readiness.publish();

    assert.deepEqual(
      {
        adapter: readiness.state().adapter,
        device: readiness.state().device,
        recovery: readiness.state().recovery,
        attempt: readiness.state().attempt,
      },
      { adapter: "ready", device: "ready", recovery: "idle", attempt: 0 },
      `a confirmed healthy device must retire stale ${recovery} recovery state`,
    );
    assert.equal(readiness.state().recoveredAt != null, true,
      "retiring deferred loss recovery records the successful healthy publication");
    assert.equal(readiness.state().generation, 8);
    assert.equal(readiness.state().timer, null);
    assert.equal(deps.clearedTimers.length, 1);
    assert.equal(deps.notifications.length, 1);
  }
});

test("loss from a replaced device cannot invalidate the current generation", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  coordinator.setCurrent(deps.replacement);
  deps.loss.resolve({ message: "retired" });
  await flush();

  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(deps.recoveries, 0);
  assert.equal(deps.destroyed.count, 0);
  assert.equal(coordinator.state().gpu.lastFailure, null);
  assert.equal(coordinator.state().gpu.recovery, "idle");
});

test("user off during recovery prevents device-loss resurrection", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "reset" });
  await flush();
  assert.equal(deps.recoveries, 1);
  assert.equal(coordinator.state().gpu.recovery, "running");

  coordinator.turnOff();
  deps.recoveryGate.resolve();
  await flush();
  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().display, "none");
  assert.equal(coordinator.state().gpu.recovery, "idle");
  assert.equal(coordinator.state().gpu.attempt, 0);
  assert.equal(deps.attaches, 0);
  assert.equal(deps.schedules, 0);
});

test("user off while recovery reconciles video cannot publish stale ready telemetry", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  deps.selectionGate = deferred();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "reset during selection" });
  await flush();

  deps.recoveryGate.resolve();
  while (deps.selections === 0) await new Promise((resolve) => setImmediate(resolve));
  coordinator.turnOff();
  deps.selectionGate.resolve();
  await flush();

  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().display, "none");
  assert.equal(coordinator.state().gpu.recovery, "idle");
  assert.equal(deps.attaches, 0);
  assert.equal(deps.schedules, 0);
  assert.deepEqual(deps.gpuReadyCalls, [],
    "the cancelled recovery must revalidate demand after video reconciliation");
});

test("exhausted recovery reports terminal failure without clearing requested features", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup({
    mode: "upscale",
    images: true,
    recoveryResults: [false, false, false],
    controlledTimers: true,
  });
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ reason: "destroyed", message: "adapter reset" });
  await flush();

  assert.equal(deps.recoveries, 1);
  assert.equal(deps.timers.length, 1);
  assert.equal(coordinator.state().gpu.recovery, "scheduled");
  assert.equal(coordinator.state().gpu.attempt, 1);
  assert.equal(coordinator.state().gpu.lastFailure.code, "device-recovery-failed");

  deps.timers.shift()();
  await flush();
  assert.equal(deps.recoveries, 2);
  assert.equal(deps.timers.length, 1);
  assert.equal(coordinator.state().gpu.recovery, "scheduled");
  assert.equal(coordinator.state().gpu.attempt, 2);

  deps.timers.shift()();
  await flush();
  const state = coordinator.state();
  assert.equal(deps.recoveries, 3);
  assert.equal(deps.timers.length, 0);
  assert.equal(state.device, null);
  assert.equal(state.mode, "upscale", "terminal recovery failure must preserve requested video mode");
  assert.equal(state.images, true, "terminal recovery failure must preserve requested image processing");
  assert.equal(state.display, "none");
  assert.equal(state.gpu.recovery, "exhausted");
  assert.equal(state.gpu.attempt, 3);
  assert.equal(state.gpu.device, "failed");
  assert.equal(state.gpu.lastFailure.stage, "recovery");
  assert.equal(state.gpu.lastFailure.code, "device-recovery-exhausted");
  assert.equal(deps.gpuFailures.map(({ code }) => code).filter(
    (code) => code === "device-recovery-failed",
  ).length, 2);
  assert.equal(deps.gpuFailures.at(-1).code, "device-recovery-exhausted");
  assert.equal(deps.neuralInvalidations, 1,
    "retries must await the original neural cleanup barrier without reinvoking it");
  assert.equal(deps.rifeInvalidations, 1,
    "retries must await the original RIFE cleanup barrier without reinvoking it");
  assert.equal(deps.multiClears, 2, "loss cleanup and terminal exhaustion both clear stale targets");
  assert.equal(deps.presentationResets, 2,
    "loss cleanup and terminal exhaustion both invalidate presented runtime state");
});

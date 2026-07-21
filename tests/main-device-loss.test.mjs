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

async function loadCoordinator(deps) {
  const source = await readFile(mainUrl, "utf8");
  const start = source.indexOf("function watchDeviceLoss(ownerDevice)");
  const end = source.indexOf("async function initWebGPUInternal()", start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const production = source.slice(start, end);
  const harness = `
    const deps = globalThis.__mainDeviceLossDeps;
    const log = (...args) => deps.logs.push(args);
    const warn = (...args) => deps.warnings.push(args);
    const watchedDeviceLosses = new WeakSet();
    const lostDevices = new WeakSet();
    let device = deps.device, deviceOwnedByMain = true;
    let adoptionGeneration = 0;
    let deviceRecoveryGeneration = 0, deviceRecoveryPromise = null, deviceRecoveryTimer = null;
    let mode = deps.mode || "passthrough", optImages = false, engine = "fsrcnnx";
    let engineSelectionGeneration = 0, chainDepth = 1, artVariant = "ArtCNN_C4F32";
    let adopting = false;
    let context = deps.context, format = "rgba8unorm", canvas = deps.canvas;
    let sampler = {}, extractPipeline = {}, recombinePipeline = {}, passthroughPipeline = {};
    let extractPipelineTex = {}, recombinePipelineTex = {}, recombine16PipelineTex = {};
    let recombine16Pipeline = {}, blitPipeline = {}, sharpenPipeline = {}, sharpenStrengthBuilt = 1;
    let debandCanvasPipeline = {}, debandFloatPipeline = {}, debandStrengthBuilt = 1;
    let chainTapTex = deps.resources.shift(), chainTapFrame = 3;
    let lumaTexture = deps.resources.shift(), lumaW = 2, lumaH = 2;
    let hiRGB = deps.resources.shift(), hiRGBW = 4, hiRGBH = 4;
    let dispRGB = deps.resources.shift(), dispRGBW = 4, dispRGBH = 4;
    let debandInterTex = deps.resources.shift(), debandInterW = 4, debandInterH = 4;
    let debandTimeBuf = deps.resources.shift(), ssimds = { destroy: deps.onSsimDestroy };
    let models = [{ destroy: deps.onModelDestroy }], modelsDevice = device, activeModel = models[0];
    let hiStages = [], artStages = {}, chainedHi = {}, chainedArt = {};
    let hiLoadPending = true, artLoadPending = true;
    let _scaleHeld = {}, _scalePending = {}, _scalePendingSince = 1, _texSource = {};
    function resetScaleSelection() { _scaleHeld = undefined; _scalePending = null; _scalePendingSince = 0; }
    let neuralEng = { invalidateDevice: (...args) => deps.invalidateNeural(...args) };
    function invalidateImageUpscaler() { deps.imageInvalidations++; }
    function clearMultiTargets() { deps.multiClears++; }
    async function initWebGPU() {
      deps.recoveries++;
      await deps.recoveryGate.promise;
      device = deps.replacement;
      return true;
    }
    async function loadModels() {}
    async function ensureArtStages() {}
    async function ensureHiStages() {}
    async function ensureNeural() {}
    function neuralSelectionCurrent() { return true; }
    async function ensureImageUpscaler() { return null; }
    function attach() { deps.attaches++; }
    function scheduleMainLoop() { deps.schedules++; }
    function deactivateRendering() { mode = "off"; cancelDeviceRecovery(); }
    ${production}
    export function watch() { watchDeviceLoss(device); }
    export function setCurrent(next) { device = next; }
    export function turnOff() { mode = "off"; cancelDeviceRecovery(); }
    export function state() {
      return { device, mode, recovering: !!deviceRecoveryPromise, display: canvas.style.display,
        chainTapTex, lumaTexture, hiRGB, dispRGB, debandInterTex, context, models };
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
    export function state() { return { device, pending: !!webGpuInitPromise }; }
  `;
  globalThis.__mainDeviceLossDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

function setup({ mode = "passthrough" } = {}) {
  const loss = deferred();
  const recoveryGate = deferred();
  const destroyed = { count: 0 };
  const device = { lost: loss.promise };
  const replacement = { lost: new Promise(() => {}) };
  const deps = {
    mode,
    device,
    replacement,
    loss,
    recoveryGate,
    logs: [],
    warnings: [],
    resources: Array.from({ length: 6 }, () => resource(destroyed)),
    context: { unconfigure() { deps.unconfigured++; } },
    canvas: { style: { display: "block", opacity: "1" } },
    destroyed,
    recoveries: 0,
    attaches: 0,
    schedules: 0,
    unconfigured: 0,
    imageInvalidations: 0,
    multiClears: 0,
    modelDestroys: 0,
    ssimDestroys: 0,
    neuralInvalidations: 0,
    onModelDestroy() { deps.modelDestroys++; },
    onSsimDestroy() { deps.ssimDestroys++; },
    async invalidateNeural() { deps.neuralInvalidations++; },
  };
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
});

test("current device loss retires stale state and single-flights recovery", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  coordinator.watch();
  deps.loss.resolve({ reason: "unknown", message: "adapter reset" });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(deps.recoveries, 1);
  assert.equal(coordinator.state().display, "none");
  assert.equal(deps.destroyed.count, 6);
  assert.equal(deps.modelDestroys, 1);
  assert.equal(deps.unconfigured, 1);
  assert.equal(deps.imageInvalidations, 1);
  assert.equal(deps.multiClears, 1);
  assert.equal(deps.neuralInvalidations >= 1, true);
  assert.deepEqual(
    [coordinator.state().chainTapTex, coordinator.state().lumaTexture,
      coordinator.state().hiRGB, coordinator.state().dispRGB,
      coordinator.state().debandInterTex],
    [null, null, null, null, null],
  );

  deps.recoveryGate.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(coordinator.state().display, "block");
  assert.equal(deps.attaches, 1);
  assert.equal(deps.schedules, 1);
});

test("loss from a replaced device cannot invalidate the current generation", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  coordinator.setCurrent(deps.replacement);
  deps.loss.resolve({ message: "retired" });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(coordinator.state().device, deps.replacement);
  assert.equal(deps.recoveries, 0);
  assert.equal(deps.destroyed.count, 0);
});

test("user off during recovery prevents device-loss resurrection", async (t) => {
  const previous = globalThis.__mainDeviceLossDeps;
  t.after(() => { globalThis.__mainDeviceLossDeps = previous; });
  const deps = setup();
  const coordinator = await loadCoordinator(deps);
  coordinator.watch();
  deps.loss.resolve({ message: "reset" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(deps.recoveries, 1);

  coordinator.turnOff();
  deps.recoveryGate.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().display, "none");
  assert.equal(deps.attaches, 0);
  assert.equal(deps.schedules, 0);
});

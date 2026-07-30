import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../src/core/fsrcnnx-main.js", import.meta.url);
const source = await readFile(mainUrl, "utf8");
const interpolateSource = await readFile(
  new URL("../src/core/fsrcnnx-interpolate.js", import.meta.url),
  "utf8",
);
const popupSource = await readFile(new URL("../popup.html", import.meta.url), "utf8");
let revision = 0;

function section(startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

const settingsContract = section(
  "// ---- validated setting contracts",
  "// ---- end validated setting contracts",
);
const policyToDepth = section("function policyToDepth", "// ---- validated setting contracts");

test("performance-driven quality reductions are opt-in by default", () => {
  assert.match(source, /const DEFAULT_INTERPOLATION_RES_MODE = "full";/);
  assert.match(source, /let interpAutoFallbackPref = false;/);
  assert.match(source, /boolean\("interpAutoFallback", false\)/);
  assert.match(interpolateSource, /this\.resMode = "full";/);
  assert.match(interpolateSource, /this\._autoFallback = false;/);
  assert.match(
    popupSource,
    /<input id="interp-autofallback" type="checkbox">/,
  );
  assert.match(popupSource, /<option value="full" selected>Full<\/option>/);
  assert.doesNotMatch(popupSource, /<option value="auto" selected>/);
  assert.doesNotMatch(
    popupSource,
    /<input id="(?:auto-quality-fallback|interp-autofallback)"[^>]*\schecked(?:\s|>)/,
  );
});

// The setting enums now come from a shared contract module instead of being
// declared inline three times, so a sliced harness must import the real module by
// absolute URL. Redeclaring the values here would recreate exactly the drift the
// contract module exists to prevent.
const CONTRACT_IMPORT = [
  "import {",
  "  ENGINES as CONTRACT_ENGINES,",
  "  INTERPOLATION_MODELS as CONTRACT_INTERPOLATION_MODELS,",
  "  INTERPOLATION_RES_MODES as CONTRACT_INTERPOLATION_RES_MODES,",
  "  MODES as CONTRACT_MODES,",
  "  UPSCALE_POLICIES as CONTRACT_UPSCALE_POLICIES,",
  "  upscalePoliciesForEngine,",
  `} from ${JSON.stringify(
    new URL("../src/core/fsrcnnx-setting-contract.js", import.meta.url).href,
  )};`,
  "",
].join("\n");

async function importHarness(code, deps = {}) {
  globalThis.__mainSettingsContractDeps = deps;
  const encoded = Buffer.from(CONTRACT_IMPORT + code).toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${++revision}`);
}

async function loadFsrcnnxPlanHarness() {
  return importHarness(`
    const STANDARD_CASCADE_THRESHOLD = 2.4;
    ${policyToDepth}
    export { fsrcnnxPlan, upscalePresentationPlan, neuralPresentationPlan };
  `);
}

async function loadPolicyHarness({
  engine = "fsrcnnx",
  policy = "display",
  neuralModels = [{ key: "span" }],
} = {}) {
  const engineSetters = section("export function setEngine", "export function setHoverReveal");
  const policySetter = section("export function setPolicy", "// Restore saved preferences");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const ART_FILES = ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"];
    let requestedEngine = ${JSON.stringify(engine)}, engine = ${JSON.stringify(engine)}, upscalePolicy = ${JSON.stringify(policy)};
    const _neuralList = ${JSON.stringify(neuralModels)};
    let engineSelectionGeneration = 0, chainDepth = 1, artVariant = "ArtCNN_C4F32";
    let mode = "off", pageSuspended = false, primaryController = null, video = null;
    let interpPausedByNeural = false, neuralEng = null, artDiagLogged = false, device = null;
    const resetScaleSelection = () => deps.events.push("reset");
    const clearMultiTargets = () => deps.events.push("clear");
    const ensureNeural = async () => {};
    const resumeInterpolationAfterNeural = () => deps.events.push("resume");
    const reconcileDeviceRecoveryDemand = () => true;
    const clearNeuralFallback = () => {};
    const activateNeuralFallback = () => { engine = "fsrcnnx"; };
    const ensureFsrcnnxStages = async () => {};
    const ensureHighStages = async () => {};
    const ensureArtStages = async () => {};
    const cancelPreferenceRestore = () => deps.events.push("fence");
    const saveSitePrefs = () => deps.events.push("save");
    const warn = () => {};
    ${policyToDepth}
    ${settingsContract}
    ${engineSetters}
    ${policySetter}
    export function state() {
      return { requestedEngine, engine, policy: upscalePolicy, chainDepth, artVariant,
        engineSelectionGeneration,
        preferenceFences: deps.events.filter((event) => event === "fence").length };
    }
  `, { events: [] });
}

async function loadSaveHarness() {
  const currentValues = section("function currentSitePreferenceValues()", "function validateSitePreferencePatch");
  const save = section("function saveSitePrefs", "export async function flushPreferenceWrites()");
  const writes = [];
  const module = await importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const DEFAULT_SETTING_FIELDS = ${JSON.stringify([
      "mode", "engine", "artVariant", "policy", "ssimds", "sharpen", "sharpenStrength",
      "hoverReveal", "allVideos", "idlePowerSaving", "autoQualityFallback", "images", "interpolate",
      "interpEngine", "interpResMode", "neuralModel", "interpTargetFps", "interpAvOffsetMs",
      "interpStaticPassthrough", "interpAutoFallback", "interpLadder", "interpInvert",
    ])};
    const siteSettingsStore = { write: async (value) => { deps.writes.push(value); } };
    const warn = () => {};
    const validateSitePreferencePatch = () => new Set();
    const recordPreferenceValidation = () => {};
    let mode = "upscale", requestedEngine = "artcnn", engine = "fsrcnnx", artVariant = "ArtCNN_C4F32_DS";
    let upscalePolicy = "force4", ssimdsEnabled = false, sharpenEnabled = true, sharpenStrength = 1.4;
    let optHoverReveal = true, optAllVideos = true, optIdlePowerSaving = true;
    let optAutoQualityFallback = true;
    let optImages = true, optInterpolate = false, neuralModelKey = "span";
    let pendingEngine = "rife_v4.26_fp16", pendingResMode = "half", pendingTargetFps = 144;
    let pendingAvOffsetMs = 35, interpStaticPassthroughPref = false;
    let interpAutoFallbackPref = false, interpLadderPref = true, interpInvertPref = false;
    ${currentValues}
    ${save}
    export { currentSitePreferenceValues, saveSitePrefs };
  `, { writes });
  return { module, writes };
}

async function loadIdlePowerSavingHarness({ suspended = false } = {}) {
  const setter = section("export async function setIdlePowerSaving", "export function setAllVideos");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    let optIdlePowerSaving = false, pageSuspended = ${JSON.stringify(suspended)};
    const cancelPreferenceRestore = () => { deps.fences++; };
    const saveSitePrefs = (fields) => { deps.saved.push(fields); };
    const notifyState = () => { deps.notifications++; };
    const retireGpuResources = async (reason) => {
      deps.retirements.push(reason);
      return { ok: true, released: true, reason };
    };
    ${setter}
    export function state() { return { optIdlePowerSaving, pageSuspended }; }
  `, { fences: 0, saved: [], notifications: 0, retirements: [] });
}

async function loadAutoQualityFallbackHarness({ selectedEngine = "neural" } = {}) {
  const performanceFallback = section(
    "function resetPlaybackPerformanceMonitoring()",
    "async function ensureNeural",
  );
  const setter = section(
    "export function setAutoQualityFallback",
    "export function setAllVideos",
  );
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    let requestedEngine = ${JSON.stringify(selectedEngine)};
    let engine = requestedEngine;
    let mode = "upscale", upscalePolicy = "display", chainDepth = 1;
    let optAutoQualityFallback = false, rendererFallback = null;
    let neuralLastFailure = null, neuralFail = 0, engineSelectionGeneration = 0;
    let performanceObservationGeneration = 0, performanceQueueSamplePending = false;
    let presentedRuntimeEngine = engine, video = null, device = null;
    const playbackPerformance = {
      reset: () => { deps.monitorResets++; },
      observeFrame: () => null,
      shouldSampleQueue: () => false,
      observeQueueBacklog: () => null,
    };
    const neuralEng = { stop: () => { deps.neuralStops++; } };
    const stopNeuralEngine = () => Promise.resolve(neuralEng?.stop?.());
    const hidePrimaryOverlays = () => {};
    const boundedRuntimeDetail = (error, fallback) => error?.message || fallback;
    const resetScaleSelection = () => { deps.scaleResets++; };
    const clearMultiTargets = () => { deps.clears++; };
    const resumeInterpolationAfterNeural = () => { deps.interpolationResumes++; };
    const ensureFsrcnnxStages = async () => {};
    const warn = () => {};
    const notifyState = () => { deps.notifications++; };
    const cancelPreferenceRestore = () => { deps.fences++; };
    const saveSitePrefs = (fields) => { deps.saved.push(fields); };
    const setEngine = (next) => {
      deps.engineRestores.push(next);
      engineSelectionGeneration++;
      requestedEngine = next;
      engine = next;
      rendererFallback = null;
      resetPlaybackPerformanceMonitoring();
      return { ok: true, pending: next === "neural" };
    };
    ${policyToDepth}
    ${performanceFallback}
    ${setter}
    export function eligible() { return performanceFallbackEligible(); }
    export function lower(signal) { return activatePerformanceFallback(signal); }
    export function installHardFallback() {
      engine = "fsrcnnx";
      rendererFallback = { from: "neural", to: "fsrcnnx", code: "neural-inference-failed" };
    }
    export function state() {
      return { requestedEngine, engine, optAutoQualityFallback, rendererFallback };
    }
  `, {
    monitorResets: 0,
    neuralStops: 0,
    scaleResets: 0,
    clears: 0,
    interpolationResumes: 0,
    notifications: 0,
    fences: 0,
    saved: [],
    engineRestores: [],
  });
}

async function loadRestoreHarness(prefs, { neuralModels = [{ key: "span" }] } = {}) {
  const restore = section("export async function restoreSitePrefs()", "function cancelPreferenceRestore()");
  const validationHelpers = section("function validateSitePreferencePatch", "function saveSitePrefs");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const DEFAULT_SETTING_FIELDS = ${JSON.stringify([
      "mode", "engine", "artVariant", "policy", "ssimds", "sharpen", "sharpenStrength",
      "hoverReveal", "allVideos", "idlePowerSaving", "autoQualityFallback", "images", "interpolate",
      "interpEngine", "interpResMode", "neuralModel", "interpTargetFps", "interpAvOffsetMs",
      "interpStaticPassthrough", "interpAutoFallback", "interpLadder", "interpInvert",
    ])};
    const _neuralList = ${JSON.stringify(neuralModels)};
    const isValidNeuralModelKey = (value) =>
      typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);
    const neuralCatalogReady = Promise.resolve(_neuralList);
    const ART_FILES = ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"];
    let preferenceRestoreGeneration = 0, engineSelectionGeneration = 0;
    let requestedEngine = "fsrcnnx", engine = "fsrcnnx", neuralModelKey = "", artVariant = "ArtCNN_C4F32";
    let upscalePolicy = "display", ssimdsEnabled = true, sharpenEnabled = false, sharpenStrength = 1;
    let optHoverReveal = false, optAllVideos = false, optIdlePowerSaving = false;
    let optAutoQualityFallback = false;
    let chainDepth = 1, pendingEngine = "rife_v4.26", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0;
    let interpStaticPassthroughPref = true, interpAutoFallbackPref = false;
    let interpLadderPref = false, interpInvertPref = true;
    let preferenceValidationFailure = null;
    const invalidPreferenceFields = new Set();
    const loadSitePrefs = async () => deps.prefs;
    const siteSettingsStore = {
      health: () => ({ state: "ready", error: null }),
      write: async (value) => { deps.writes.push(value); },
    };
    const boundedRuntimeDetail = (error) => error?.message || String(error);
    const warn = () => {};
    const clearNeuralFallback = () => {};
    const resetScaleSelection = () => {};
    const setMode = async (value) => { deps.calls.push(["mode", value]); return { ok: true }; };
    const setImages = async (value) => { deps.calls.push(["images", value]); return { ok: true }; };
    const setInterpolate = async (value) => { deps.calls.push(["interpolate", value]); return { ok: true }; };
    ${policyToDepth}
    ${settingsContract}
    ${validationHelpers}
    ${restore}
    export function applyValidation(patch) {
      recordPreferenceValidation(patch, validateSitePreferencePatch(patch));
      return preferenceValidationFailure;
    }
    export function migrationWrites() { return [...deps.writes]; }
    export function state() {
      return { engine, neuralModelKey, artVariant, policy: upscalePolicy, chainDepth,
        ssimdsEnabled, sharpenEnabled, sharpenStrength,
        optIdlePowerSaving, optAutoQualityFallback,
        pendingEngine, pendingResMode, pendingTargetFps, pendingAvOffsetMs,
        interpStaticPassthroughPref, interpAutoFallbackPref, interpLadderPref, interpInvertPref,
        preferenceValidationFailure };
    }
  `, { prefs, calls: [], writes: [] });
}

async function loadInterpolationConfigHarness(instance = null, {
  neuralGate = null,
  neuralModels = [{ key: "span" }],
} = {}) {
  const configure = section("function configureInterpolator", "function scheduleInterpolatorGpuRestart");
  const primarySetters = section("function acceptedPendingInterpolationSetting", "export function listInterpolateModels");
  const secondarySetters = section("export async function setInterpolateInvert", "log(\"pipeline module loaded\")");
  const externalApply = section("async function applyExternalSitePreferences", "export function getStatus");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    ${settingsContract}
    let interpolator = deps.instance;
    let interpolationConfigGeneration = 0;
    let pendingEngine = DEFAULT_INTERPOLATION_MODEL, pendingResMode = DEFAULT_INTERPOLATION_RES_MODE;
    let pendingTargetFps = DEFAULT_INTERPOLATION_TARGET_FPS;
    let pendingAvOffsetMs = DEFAULT_INTERPOLATION_AV_OFFSET_MS;
    let interpStaticPassthroughPref = true, interpAutoFallbackPref = false;
    let interpLadderPref = false, interpInvertPref = true;
    let optInterpolate = true, engine = "fsrcnnx", video = { id: "video" };
    let optAutoQualityFallback = false;
    let interpolationStartFailureStreak = null, chainInverted = false, canvas = null;
    const reviseInterpolationConfiguration = () => ({ generation: ++interpolationConfigGeneration, retry: false });
    const saveSitePrefs = () => deps.saved.push({
      pendingEngine, pendingResMode, pendingTargetFps, pendingAvOffsetMs,
      interpStaticPassthroughPref, interpAutoFallbackPref, interpLadderPref, interpInvertPref,
    });
    const requestInterpolationRetry = async () => false;
    let preferenceRestoreGeneration = 0, preferenceValidationFailure = null;
    const cancelPreferenceRestore = () => { preferenceRestoreGeneration++; deps.fences++; };
    const captureVideoSource = (candidate) => candidate;
    const sameVideoSource = (left, right) => left === right;
    const recordInterpolationStartFailure = (_video, _source, result) => deps.failures.push(result);
    const _neuralList = ${JSON.stringify(neuralModels)};
    const isValidNeuralModelKey = (value) =>
      typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);
    const neuralCatalogReady = Promise.resolve(_neuralList);
    const validateSitePreferencePatch = () => new Set();
    const recordPreferenceValidation = () => {};
    const siteSettingsStore = { write: async (value) => { deps.writes.push(value); } };
    const boundedRuntimeDetail = (error) => error?.message || String(error);
    const warn = () => {};
    let neuralModelKey = "";
    const setNeuralModel = async (key) => {
      deps.neuralCalls++;
      neuralModelKey = key;
      if (deps.neuralGate) await deps.neuralGate.promise;
      return { ok: true };
    };
    const notifyState = () => {};
    const setAutoQualityFallback = (value) => {
      optAutoQualityFallback = value;
      return { ok: true, autoQualityFallback: value };
    };
    const log = () => {};
    ${configure}
    ${primarySetters}
    ${secondarySetters}
    ${externalApply}
    export { configureInterpolator, applyExternalSitePreferences };
    export function setInstance(value) { interpolator = value; }
    export function migrationWrites() { return [...deps.writes]; }
    export function state() {
      return { pendingEngine, pendingResMode, pendingTargetFps, pendingAvOffsetMs,
        interpStaticPassthroughPref, interpAutoFallbackPref, interpLadderPref,
        interpInvertPref, interpolationConfigGeneration, preferenceFences: deps.fences,
        neuralModelKey, optAutoQualityFallback };
    }
  `, { instance, saved: [], failures: [], fences: 0, neuralCalls: 0, neuralGate, writes: [] });
}

async function loadStatusHarness(storeHealth = {
  state: "ready", operation: null, errorOperation: null, pending: 0, error: null,
  schemaVersion: 2, scope: "https://video.example",
}, gpu = { adapter: "unrequested", device: "uninitialized" }, runtime = {}) {
  const getStatus = section("export function getStatus()", "// ---- image upscaling");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const STATUS_VERSION = 1, GPU_RECOVERY_MAX_ATTEMPTS = 3;
    let mode = deps.runtime.mode || "off", primaryController = null, video = null, frameCount = 0;
    let primaryPresentationGeneration = 0;
    let presentedPresentation = deps.runtime.presentation || null;
    let activeModel = null, upscalePolicy = "display", ssimdsEnabled = true;
    let sharpenEnabled = false, sharpenStrength = 1, requestedEngine = "neural", engine = "neural";
    let artVariant = "ArtCNN_C4F32", chainDepth = 1, neuralModelKey = "span-lazy";
    let neuralEng = deps.runtime.neuralEntry ? {
      activeEntry: () => deps.runtime.neuralEntry,
      ready: () => true,
      stats: () => ({ runs: 1, meanRunMs: 10 }),
    } : null;
    let protectedSource = false, protectedReason = null;
    const selectedColorSupport = {
      supported: false,
      code: "color-not-checked",
      detail: "No decoded video is selected.",
      colorSpace: { primaries: null, transfer: null, matrix: null, fullRange: null },
    };
    let optHoverReveal = false, optAllVideos = false, optIdlePowerSaving = false;
    let optAutoQualityFallback = false;
    let optImages = false, imageUpscaledCount = 0;
    let optInterpolate = deps.runtime.interpolate === true, interpPausedByNeural = false;
    let interpolationTerminalQuarantine = deps.runtime.interpFailure || null, interpolator = null;
    if (deps.runtime.interpStats) interpolator = { getStats: () => deps.runtime.interpStats };
    let pendingEngine = "blend", pendingResMode = "quarter", pendingTargetFps = 165;
    let pendingAvOffsetMs = -25, interpStaticPassthroughPref = false;
    let interpAutoFallbackPref = false, interpLadderPref = true, interpInvertPref = false;
    let pageSuspended = deps.runtime.suspended === true;
    let deviceRecoveryPromise = null, deviceRecoveryTimer = null;
    let gpuAdapterPhase = deps.gpu.adapter, gpuDevicePhase = deps.gpu.device;
    let gpuRecoveryPhase = "idle", gpuRecoveryAttempt = 0, gpuLastFailure = null, gpuRecoveredAt = null;
    let gpuResourcePhase = deps.runtime.resourcePhase || "idle";
    let gpuResourceReason = deps.runtime.resourceReason || null;
    let rendererFallback = null, neuralLastFailure = null, neuralFail = 0;
    const frameTimes = deps.frameTimes || [];
    const playbackPerformance = { snapshot: () => ({
      triggered: null,
      lastWindow: null,
      frameIntervalMs: null,
      consecutiveDegradedWindows: 0,
      consecutiveBacklogs: 0,
    }) };
    const primaryPresentationBoundary = {
      pictureInPicture: false,
      directFullscreen: false,
      fullscreenElsewhere: false,
      nativeRequired: false,
    };
    let imageUpscaler = null, imageUpscalerInitPromise = null, imageLastFailure = null;
    let preferenceValidationFailure = null, preferenceApplicationFailure = null;
    let preferenceApplicationLastError = null;
    const multiTargets = new Map();
    const _neuralList = [];
    const navigator = { gpu: {} };
    const findVideo = () => deps.runtime.hasVideo ? {} : null;
    const siteHost = () => "video.example";
    const siteScope = () => "https://video.example";
    const siteSettingsStore = { health: () => deps.storeHealth };
    const currentPresentedRuntime = () => deps.runtime.presented || ({ mode: "off", engine: null });
    const interpolationQuarantineMatches = () => interpolationTerminalQuarantine !== null;
    ${getStatus}
  `, { storeHealth, gpu, runtime });
}

async function loadPreferenceApplicationHarness(
  failures = 0,
  syncPatch = null,
  snapshot = { mode: "upscale" },
) {
  const production = section("export async function syncSitePrefs()", "function sendRuntimeMessage");
  const deps = { listener: null, calls: [], failures, syncPatch, snapshot };
  const module = await importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const DEFAULT_SETTING_FIELDS = ["mode", "images"];
    let externalPreferenceTail = Promise.resolve();
    let preferenceApplicationFailure = null, preferenceValidationFailure = null;
    let preferenceApplicationLastError = null;
    const boundedRuntimeDetail = (error) => error?.message || String(error);
    const warn = () => {};
    const siteSettingsStore = {
      async sync() {
        deps.calls.push("sync");
        if (deps.syncPatch && deps.listener) deps.listener(deps.syncPatch);
        return deps.syncPatch || {};
      },
      snapshot() { deps.calls.push("snapshot"); return { ...deps.snapshot }; },
      health() { return { state: "ready" }; },
      subscribe(listener) { deps.listener = listener; },
    };
    const persistenceStatus = () => ({ state: "ready", operation: null,
      errorOperation: null, pendingWrites: 0, error: null });
    const applyExternalSitePreferences = async (patch) => {
      deps.calls.push(["apply", patch]);
      if (deps.failures === Infinity || deps.failures-- > 0) throw new Error("application exploded");
      return { ok: true };
    };
    ${production}
    export function emit(patch) { deps.listener(patch); }
    export async function drainOnly() { await drainExternalPreferenceApplications(); }
    export function applicationFailure() { return preferenceApplicationFailure; }
    export function lastApplicationError() { return preferenceApplicationLastError; }
  `, deps);
  return { module, deps };
}

async function loadRuntimeKeyHarness() {
  const runtimeKey = section("function interpolationRuntimeConfigKey()", "function interpolationQuarantineMatches");
  return importHarness(`
    let interpolationConfigGeneration = 7, pendingEngine = "rife_v4.26_fp16", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0, interpStaticPassthroughPref = true;
    let interpAutoFallbackPref = false, interpLadderPref = false, interpInvertPref = true;
    ${runtimeKey}
    export const key = interpolationRuntimeConfigKey;
    export function vary(name, value) {
      if (name === "model") pendingEngine = value;
      if (name === "res") pendingResMode = value;
      if (name === "target") pendingTargetFps = value;
      if (name === "offset") pendingAvOffsetMs = value;
      if (name === "static") interpStaticPassthroughPref = value;
      if (name === "fallback") interpAutoFallbackPref = value;
      if (name === "ladder") interpLadderPref = value;
      if (name === "invert") interpInvertPref = value;
    }
  `);
}

test("engine and policy setters reject invalid values and normalize incompatible switches atomically", async () => {
  const settings = await loadPolicyHarness({ engine: "fsrcnnx", policy: "force3" });
  const before = settings.state();
  assert.deepEqual(settings.setEngine("unknown"), {
    ok: false, reason: "invalid engine", engine: "fsrcnnx", activeEngine: "fsrcnnx",
    policy: "force3", chainDepth: 1,
  });
  assert.deepEqual(settings.state(), before, "an invalid engine must not mutate any state");
  const high = await loadPolicyHarness({ engine: "fsrcnnx", policy: "force3" });
  assert.deepEqual(high.setEngine("fsrcnnx-hi"), {
    ok: true, engine: "fsrcnnx-hi", activeEngine: "fsrcnnx-hi",
    policy: "display", chainDepth: 1, pending: false,
  });
  assert.deepEqual(high.setPolicy("force3"), {
    ok: false, reason: "invalid policy", policy: "display", chainDepth: 1,
  });
  assert.deepEqual(high.setPolicy("force8"), {
    ok: true, policy: "force8", chainDepth: 3,
  });

  const noNeuralModels = await loadPolicyHarness({ neuralModels: [] });
  const noNeuralBefore = noNeuralModels.state();
  assert.deepEqual(noNeuralModels.setEngine("neural"), {
    ok: false, reason: "no bundled neural models", engine: "fsrcnnx",
    activeEngine: "fsrcnnx", policy: "display", chainDepth: 1,
  });
  assert.deepEqual(noNeuralModels.state(), noNeuralBefore,
    "an unavailable neural engine must not mutate or persist selection state");

  assert.deepEqual(settings.setEngine("artcnn"), {
    ok: true, engine: "artcnn", activeEngine: "artcnn", policy: "display", chainDepth: 1,
    pending: false,
  });
  assert.equal(settings.state().policy, "display");
  assert.deepEqual(settings.setPolicy("force3"), {
    ok: false, reason: "invalid policy", policy: "display", chainDepth: 1,
  });
  assert.deepEqual(settings.setPolicy("force8"), {
    ok: true, policy: "force8", chainDepth: 3,
  });

  const artBefore = settings.state();
  assert.deepEqual(settings.setArtVariant("../../untrusted"), {
    ok: false, reason: "invalid art variant", artVariant: "ArtCNN_C4F32",
  });
  assert.deepEqual(settings.state(), artBefore);
  assert.equal(settings.state().preferenceFences, 2,
    "only accepted persistent engine and policy changes fence external reconciliation");

  const standard = await loadPolicyHarness({ engine: "fsrcnnx", policy: "display" });
  assert.deepEqual(standard.setPolicy("force3"), {
    ok: true, policy: "force3", chainDepth: 2,
  });
  assert.deepEqual(standard.setPolicy("force4"), {
    ok: true, policy: "force4", chainDepth: 2,
  });

  const neural = await loadPolicyHarness({ engine: "neural", policy: "display" });
  assert.deepEqual(neural.setPolicy("force2"), {
    ok: true, policy: "force2", chainDepth: 1,
  });
  assert.deepEqual(neural.setPolicy("native"), {
    ok: true, policy: "native", chainDepth: 1,
  });
  assert.deepEqual(neural.setPolicy("force4"), {
    ok: false, reason: "invalid policy", policy: "native", chainDepth: 1,
  });
});

test("Neural exact x2 presentation preserves native inference and output dimensions", async () => {
  const { neuralPresentationPlan } = await loadFsrcnnxPlanHarness();
  const expectedExact = {
    modelWidth: 2564, modelHeight: 1436,
    outputWidth: 1282, outputHeight: 718,
    downsample: true,
  };
  assert.deepEqual(neuralPresentationPlan(
    "force2", 641, 359, 4, 3840, { ssimdsEnabled: true },
  ), { ...expectedExact, ssimds: true });
  assert.deepEqual(neuralPresentationPlan(
    "force2", 641, 359, 4, 320, { ssimdsEnabled: false },
  ), { ...expectedExact, ssimds: false },
  "the sampled fallback must retain exact x2 dimensions when SSimDS is off");
  assert.deepEqual(neuralPresentationPlan(
    "native", 641, 359, 4, 320, { ssimdsEnabled: true },
  ), {
    modelWidth: 2564, modelHeight: 1436,
    outputWidth: 2564, outputHeight: 1436,
    downsample: false, ssimds: false,
  });
  assert.deepEqual(neuralPresentationPlan(
    "force2", 641, 359, 2, 320, { ssimdsEnabled: true },
  ), {
    modelWidth: 1282, modelHeight: 718,
    outputWidth: 1282, outputHeight: 718,
    downsample: false, ssimds: false,
  }, "a native x2 model must not be needlessly downscaled");
  assert.deepEqual(neuralPresentationPlan(
    "display", 641, 359, 4, 1923, { ssimdsEnabled: true },
  ), {
    modelWidth: 2564, modelHeight: 1436,
    outputWidth: 1923, outputHeight: 1077,
    downsample: true, ssimds: true,
  }, "display width is already expressed in physical pixels after DPR/fullscreen planning");
});

test("verified standard x2 planning cascades only for explicit or clearly larger targets", async () => {
  const { fsrcnnxPlan, upscalePresentationPlan } = await loadFsrcnnxPlanHarness();
  assert.deepEqual(fsrcnnxPlan("force2", 1), { shouldRun: true, depth: 1 });
  assert.deepEqual(fsrcnnxPlan("force3", 1), { shouldRun: true, depth: 2 });
  assert.deepEqual(fsrcnnxPlan("force4", 1), { shouldRun: true, depth: 2 });
  assert.deepEqual(fsrcnnxPlan("display", 1), { shouldRun: false, depth: 1 });
  assert.deepEqual(fsrcnnxPlan("display", 2.39), { shouldRun: true, depth: 1 });
  assert.deepEqual(fsrcnnxPlan("display", 2.4), { shouldRun: true, depth: 2 });
  assert.deepEqual(fsrcnnxPlan("auto", 1.4), { shouldRun: false, depth: 1 });
  assert.deepEqual(fsrcnnxPlan("auto", 1.41), { shouldRun: true, depth: 1 });
  assert.deepEqual(fsrcnnxPlan("auto", 3), { shouldRun: true, depth: 2 });

  const exactThreeWithSSim = upscalePresentationPlan(
    "force3", 640, 360, 4, 1280, { ssimdsEnabled: true, displaySafe: true },
  );
  const exactThreeWithoutSSim = upscalePresentationPlan(
    "force3", 640, 360, 4, 3840, { ssimdsEnabled: false, displaySafe: true },
  );
  assert.deepEqual(exactThreeWithSSim, {
    modelWidth: 2560, modelHeight: 1440,
    outputWidth: 1920, outputHeight: 1080,
    downsample: true, ssimds: true,
  });
  assert.deepEqual(exactThreeWithoutSSim, {
    modelWidth: 2560, modelHeight: 1440,
    outputWidth: 1920, outputHeight: 1080,
    downsample: true, ssimds: false,
  }, "force3 presentation must not depend on the display size or SSimDS toggle");

  assert.deepEqual(upscalePresentationPlan(
    "force4", 640, 360, 4, 1280, { ssimdsEnabled: false, displaySafe: true },
  ), {
    modelWidth: 2560, modelHeight: 1440,
    outputWidth: 2560, outputHeight: 1440,
    downsample: false, ssimds: false,
  });

  assert.deepEqual(upscalePresentationPlan(
    "force2", 160, 90, 2, 216, { ssimdsEnabled: true, displaySafe: true },
  ), {
    modelWidth: 320, modelHeight: 180,
    outputWidth: 216, outputHeight: 122,
    downsample: true, ssimds: true,
  }, "SSimDS should reduce a 320px model result for a 216-device-pixel target");
  assert.deepEqual(upscalePresentationPlan(
    "force2", 160, 90, 2, 216 * 2, { ssimdsEnabled: true, displaySafe: true },
  ), {
    modelWidth: 320, modelHeight: 180,
    outputWidth: 320, outputHeight: 180,
    downsample: false, ssimds: false,
  }, "CSS width must be converted to physical display pixels before SSimDS planning");
});

test("site persistence records requested intent and writes only selected fields", async () => {
  const { module, writes } = await loadSaveHarness();
  await module.saveSitePrefs();
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0], {
    mode: "upscale", engine: "artcnn", artVariant: "ArtCNN_C4F32_DS", policy: "force4",
    ssimds: false, sharpen: true, sharpenStrength: 1.4,
    hoverReveal: true, allVideos: true, idlePowerSaving: true,
    autoQualityFallback: true,
    images: true, interpolate: false,
    interpEngine: "rife_v4.26_fp16", interpResMode: "half", neuralModel: "span",
    interpTargetFps: 144, interpAvOffsetMs: 35, interpStaticPassthrough: false,
    interpAutoFallback: false, interpLadder: true, interpInvert: false,
  });
  writes.length = 0;
  await module.saveSitePrefs(["engine", "policy"]);
  assert.deepEqual(writes, [{ engine: "artcnn", policy: "force4" }],
    "an effective fallback must not replace durable requested-engine intent");
});

test("idle power saving is opt-in and retires an already-hidden document immediately", async () => {
  const active = await loadIdlePowerSavingHarness();
  assert.deepEqual(await active.setIdlePowerSaving(true), {
    ok: true, idlePowerSaving: true, resourcesReleased: false,
  });
  assert.deepEqual(globalThis.__mainSettingsContractDeps.retirements, []);

  const hidden = await loadIdlePowerSavingHarness({ suspended: true });
  assert.deepEqual(await hidden.setIdlePowerSaving(true), {
    ok: true, idlePowerSaving: true, resourcesReleased: true,
  });
  assert.deepEqual(globalThis.__mainSettingsContractDeps.retirements, ["document-hidden"]);
  assert.deepEqual(globalThis.__mainSettingsContractDeps.saved, [["idlePowerSaving"]]);
  assert.equal(globalThis.__mainSettingsContractDeps.fences, 1);
  assert.equal(globalThis.__mainSettingsContractDeps.notifications, 1);
});

test("automatic quality fallback is opt-in, includes Neural, and restores requested quality when disabled", async () => {
  const quality = await loadAutoQualityFallbackHarness();
  const deps = globalThis.__mainSettingsContractDeps;
  const signal = {
    code: "sustained-frame-drops",
    detail: "Playback pressure persisted.",
    evidence: { dropRatio: 0.25 },
  };

  assert.equal(quality.eligible(), false);
  assert.equal(quality.lower(signal), null,
    "frame pressure cannot lower quality while the preference is disabled");

  assert.deepEqual(quality.setAutoQualityFallback(true), {
    ok: true,
    autoQualityFallback: true,
    restored: false,
  });
  assert.equal(quality.eligible(), true, "Neural is an eligible high-cost requested renderer");
  assert.equal(quality.lower({
    code: "model-render-failed",
    detail: "A single renderer exception is not sustained playback pressure.",
  }), null, "opt-in adaptation must not turn a one-off renderer error into a quality downgrade");
  assert.equal(quality.state().engine, "neural");
  assert.equal(quality.lower(signal).category, "performance");
  assert.deepEqual(quality.state(), {
    requestedEngine: "neural",
    engine: "fsrcnnx",
    optAutoQualityFallback: true,
    rendererFallback: {
      category: "performance",
      from: "neural",
      to: "fsrcnnx",
      code: "sustained-frame-drops",
      detail: "Playback pressure persisted.",
      at: quality.state().rendererFallback.at,
      evidence: { dropRatio: 0.25 },
    },
  });
  assert.equal(deps.neuralStops, 1);
  assert.equal(deps.interpolationResumes, 1);

  assert.deepEqual(quality.setAutoQualityFallback(false), {
    ok: true,
    autoQualityFallback: false,
    restored: true,
    pending: true,
  });
  assert.equal(quality.state().requestedEngine, "neural");
  assert.equal(quality.state().engine, "neural");
  assert.equal(quality.state().rendererFallback, null);
  assert.deepEqual(deps.engineRestores, ["neural"]);
  assert.deepEqual(deps.saved, [["autoQualityFallback"], ["autoQualityFallback"]]);

  quality.installHardFallback();
  assert.deepEqual(quality.setAutoQualityFallback(false), {
    ok: true,
    autoQualityFallback: false,
    restored: false,
  });
  assert.equal(quality.state().rendererFallback.code, "neural-inference-failed",
    "disabling performance adaptation must not erase a hard neural safety fallback");
  assert.deepEqual(deps.engineRestores, ["neural"]);
});

test("automatic quality fallback targets every expensive upscaler but never standard FSRCNNX", async () => {
  const signal = {
    code: "sustained-gpu-backlog",
    detail: "GPU pressure persisted.",
    evidence: { consecutiveBacklogs: 3 },
  };

  for (const selectedEngine of ["fsrcnnx-hi", "artcnn", "neural"]) {
    const quality = await loadAutoQualityFallbackHarness({ selectedEngine });
    quality.setAutoQualityFallback(true);
    assert.equal(quality.eligible(), true, `${selectedEngine} must be eligible after opt-in`);
    const fallback = quality.lower(signal);
    assert.equal(fallback.from, selectedEngine);
    assert.equal(fallback.to, "fsrcnnx");
    assert.equal(fallback.category, "performance");
    assert.equal(quality.state().requestedEngine, selectedEngine,
      "the durable requested model must survive a temporary quality fallback");
    assert.equal(quality.state().engine, "fsrcnnx");
  }

  const standard = await loadAutoQualityFallbackHarness({ selectedEngine: "fsrcnnx" });
  standard.setAutoQualityFallback(true);
  assert.equal(standard.eligible(), false);
  assert.equal(standard.lower(signal), null);
  assert.deepEqual(standard.state(), {
    requestedEngine: "fsrcnnx",
    engine: "fsrcnnx",
    optAutoQualityFallback: true,
    rendererFallback: null,
  });
});

test("preference restore preserves High selections, migrates legacy values, and rejects corrupt fields", async () => {
  const high = await loadRestoreHarness({
    engine: "fsrcnnx-hi", artVariant: "ArtCNN_C4F32", policy: "force4",
  });
  assert.equal((await high.restoreSitePrefs()).ok, true);
  assert.equal(high.state().engine, "fsrcnnx-hi");
  assert.equal(high.state().policy, "force4");
  assert.equal(high.state().chainDepth, 2);
  assert.equal(high.state().preferenceValidationFailure, null);
  assert.deepEqual(high.migrationWrites(), []);

  const highForce8 = await loadRestoreHarness({ engine: "fsrcnnx-hi", policy: "force8" });
  assert.equal((await highForce8.restoreSitePrefs()).ok, true);
  assert.equal(highForce8.state().engine, "fsrcnnx-hi");
  assert.equal(highForce8.state().policy, "force8");
  assert.equal(highForce8.state().chainDepth, 3);
  assert.equal(highForce8.state().preferenceValidationFailure, null);
  assert.deepEqual(highForce8.migrationWrites(), []);

  const legacyInterpolation = await loadRestoreHarness({ interpEngine: "rife_orig" });
  assert.equal((await legacyInterpolation.restoreSitePrefs()).ok, true);
  assert.equal(legacyInterpolation.state().pendingEngine, "rife_v4.26");
  assert.equal(legacyInterpolation.state().preferenceValidationFailure, null);
  assert.deepEqual(legacyInterpolation.migrationWrites(), [{ interpEngine: "rife_v4.26" }]);

  const removedNeuralWhileArt = await loadRestoreHarness({
    engine: "artcnn", neuralModel: "span2x_smoke",
  }, { neuralModels: [] });
  assert.equal((await removedNeuralWhileArt.restoreSitePrefs()).ok, true);
  assert.equal(removedNeuralWhileArt.state().engine, "artcnn");
  assert.equal(removedNeuralWhileArt.state().neuralModelKey, "");
  assert.equal(removedNeuralWhileArt.state().preferenceValidationFailure, null);
  assert.deepEqual(removedNeuralWhileArt.migrationWrites(), [{ neuralModel: null }]);

  const replacedNeuralCatalog = await loadRestoreHarness({
    engine: "neural", neuralModel: "span2x_smoke",
  }, { neuralModels: [{ key: "replacement" }] });
  assert.equal((await replacedNeuralCatalog.restoreSitePrefs()).ok, true);
  assert.equal(replacedNeuralCatalog.state().engine, "neural");
  assert.equal(replacedNeuralCatalog.state().neuralModelKey, "replacement");
  assert.equal(replacedNeuralCatalog.state().preferenceValidationFailure, null);
  assert.deepEqual(replacedNeuralCatalog.migrationWrites(), [{ neuralModel: "replacement" }]);

  const valid = await loadRestoreHarness({
    engine: "artcnn", neuralModel: "span", artVariant: "ArtCNN_C4F32_DN", policy: "force8",
    mode: "upscale", idlePowerSaving: true, autoQualityFallback: true,
    images: true, interpolate: true,
    interpEngine: "blend", interpResMode: "half", interpTargetFps: "144", interpAvOffsetMs: "25",
    interpStaticPassthrough: false, interpAutoFallback: false, interpLadder: true, interpInvert: false,
  });
  assert.equal((await valid.restoreSitePrefs()).ok, true);
  assert.deepEqual(valid.state(), {
    engine: "artcnn", neuralModelKey: "span", artVariant: "ArtCNN_C4F32_DN",
    policy: "force8", chainDepth: 3,
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1,
    optIdlePowerSaving: true, optAutoQualityFallback: true,
    pendingEngine: "blend", pendingResMode: "half", pendingTargetFps: 144, pendingAvOffsetMs: 25,
    interpStaticPassthroughPref: false, interpAutoFallbackPref: false,
    interpLadderPref: true, interpInvertPref: false, preferenceValidationFailure: null,
  });

  const corrupt = await loadRestoreHarness({
    engine: { value: "artcnn" }, neuralModel: "../unknown", artVariant: "../../bad", policy: "force8",
    mode: "bogus", images: "true", interpolate: 1,
    interpEngine: "file:///tmp/model", interpResMode: "max", interpTargetFps: 999,
    interpAvOffsetMs: -101, interpStaticPassthrough: "false",
    interpAutoFallback: "false", interpLadder: 1, interpInvert: null,
    autoQualityFallback: "true",
    sharpenStrength: "2",
  });
  assert.equal((await corrupt.restoreSitePrefs()).ok, true);
  assert.deepEqual(corrupt.state(), {
    engine: "fsrcnnx", neuralModelKey: "", artVariant: "ArtCNN_C4F32",
    policy: "force4", chainDepth: 2,
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1,
    optIdlePowerSaving: false, optAutoQualityFallback: false,
    pendingEngine: "rife_v4.26", pendingResMode: "full",
    pendingTargetFps: "auto", pendingAvOffsetMs: 0,
    interpStaticPassthroughPref: true, interpAutoFallbackPref: false,
    interpLadderPref: false, interpInvertPref: true,
    preferenceValidationFailure: "Invalid stored settings: artVariant, autoQualityFallback, engine, " +
      "images, interpAutoFallback, interpAvOffsetMs, interpEngine, interpInvert, interpLadder, " +
      "interpResMode, interpStaticPassthrough, interpTargetFps, interpolate, mode, neuralModel, " +
      "sharpenStrength",
  });
  assert.deepEqual(corrupt.migrationWrites(), [{ policy: "force4" }]);

  assert.equal(corrupt.applyValidation({ images: true }),
    "Invalid stored settings: artVariant, autoQualityFallback, engine, interpAutoFallback, " +
      "interpAvOffsetMs, interpEngine, interpInvert, interpLadder, interpResMode, " +
      "interpStaticPassthrough, interpTargetFps, interpolate, mode, neuralModel, " +
      "sharpenStrength",
      "a valid unrelated field clears only its own validation error");
});

test("persisted sharpen strength enforces the same inclusive bounds as popup commands", async () => {
  for (const sharpenStrength of [0.1, 2]) {
    const valid = await loadRestoreHarness({ sharpenStrength });
    assert.equal((await valid.restoreSitePrefs()).ok, true);
    assert.equal(valid.state().sharpenStrength, sharpenStrength);
    assert.equal(valid.state().preferenceValidationFailure, null);
  }

  const invalid = await loadRestoreHarness({ sharpenStrength: 2.01 });
  assert.equal((await invalid.restoreSitePrefs()).ok, true);
  assert.equal(invalid.state().sharpenStrength, 2, "unsafe stored sharpen strength is clamped");
  assert.equal(
    invalid.state().preferenceValidationFailure,
    "Invalid stored setting: sharpenStrength",
    "finite out-of-range values remain observable as invalid storage",
  );
});

test("interpolation setters preserve invalid-state immutability and configure a future runtime", async () => {
  const config = await loadInterpolationConfigHarness();
  const initial = config.state();
  assert.equal(config.setInterpolateRes("eighth").ok, false);
  assert.equal(config.setInterpolateAvOffset(Infinity).ok, false);
  assert.equal((await config.setInterpolateModel("unknown")).ok, false);
  assert.equal((await config.setInterpolateModel("rife_orig")).ok, false,
    "the historical key is accepted only by stored-preference migration");
  assert.equal(config.setInterpolateTargetFps(1000).ok, false);
  assert.deepEqual(config.state(), initial);

  assert.equal(config.setInterpolateRes("quarter").ok, true);
  assert.equal(config.setInterpolateAvOffset("-25").ok, true);
  assert.equal((await config.setInterpolateModel("blend")).ok, true);
  assert.equal(config.setInterpolateTargetFps("165").ok, true);
  assert.equal(config.setInterpolateDiag(false).ok, true);
  assert.equal(config.setInterpolateAutoFallback(false).ok, true);
  assert.equal(config.setInterpolateLadder(true).ok, true);
  assert.equal((await config.setInterpolateInvert(false)).ok, true);
  assert.equal(config.state().preferenceFences, 8,
    "each accepted user setting fences older external reconciliation");

  const calls = [];
  const chain = {};
  const runtime = {
    chain,
    _rifeMod: { setStaticPassthrough: (value) => calls.push(["static-cpu", value]) },
    setInterpEngine: (value) => calls.push(["model", value]),
    setResMode: (value) => { calls.push(["res", value]); return value; },
    setTargetFps: (value) => { calls.push(["target", value]); return value; },
    setAvOffset: (value) => { calls.push(["offset", value]); return value; },
    setAutoFallback: (value) => calls.push(["fallback", value]),
    setLadder: (value) => calls.push(["ladder", value]),
  };
  config.configureInterpolator(runtime);
  assert.deepEqual(calls, [
    ["model", "blend"], ["res", "quarter"], ["target", 165], ["offset", -25],
    ["fallback", false], ["ladder", true], ["static-cpu", false],
  ]);
  assert.equal(runtime._staticOn, false);
  assert.equal(chain.invert(), false);
  assert.equal(chain.ladder(), true);
});

test("interpolation runtime failures are returned as explicit command results", async () => {
  const runtime = {
    running: false,
    setResMode() { throw new Error("GPU state unavailable"); },
    setInterpEngine() { throw new Error("model switch failed"); },
  };
  const config = await loadInterpolationConfigHarness(runtime);
  assert.deepEqual(config.setInterpolateRes("full"), {
    ok: true, pending: true, pendingKind: "runtime-error", reason: "runtime failure",
    detail: "GPU state unavailable", resMode: "full",
  });
  assert.deepEqual(await config.setInterpolateModel("rife_v4.26_fp16"), {
    ok: true, pending: true, pendingKind: "runtime-error", model: "rife_v4.26_fp16",
    reason: "runtime failure", detail: "model switch failed",
  });
  assert.equal(config.state().pendingResMode, "full");
  assert.equal(config.state().pendingEngine, "rife_v4.26_fp16");
});

test("a newer local setting fences an older asynchronous external preference apply", async () => {
  let releaseNeural;
  const neuralGate = {
    promise: new Promise((resolve) => { releaseNeural = resolve; }),
  };
  const config = await loadInterpolationConfigHarness(null, { neuralGate });
  const applying = config.applyExternalSitePreferences({
    neuralModel: "span",
    interpEngine: "rife_v4.26_fp16",
  });
  await Promise.resolve();

  assert.equal((await config.setInterpolateModel("blend")).ok, true);
  assert.equal(config.state().pendingEngine, "blend");
  releaseNeural({ ok: true });

  assert.deepEqual(await applying, { ok: false, reason: "superseded" });
  assert.equal(config.state().pendingEngine, "blend",
    "the stale external continuation must not overwrite newer local intent");
  assert.equal(config.state().preferenceFences, 1);
});

test("external preference reconciliation replaces a neural model removed from the catalog", async () => {
  const config = await loadInterpolationConfigHarness();
  assert.deepEqual(await config.applyExternalSitePreferences({ neuralModel: "span2x_smoke" }), {
    ok: true,
    applied: true,
    invalid: [],
  });
  assert.equal(config.state().neuralModelKey, "span");
  assert.deepEqual(config.migrationWrites(), [{ neuralModel: "span" }]);
});

test("external preference reconciliation accepts only a boolean automatic quality fallback", async () => {
  const config = await loadInterpolationConfigHarness();
  assert.deepEqual(await config.applyExternalSitePreferences({ autoQualityFallback: true }), {
    ok: true,
    applied: true,
    invalid: [],
  });
  assert.equal(config.state().optAutoQualityFallback, true);

  assert.deepEqual(await config.applyExternalSitePreferences({ autoQualityFallback: "true" }), {
    ok: false,
    applied: true,
    invalid: ["autoQualityFallback"],
  });
  assert.equal(config.state().optAutoQualityFallback, false,
    "an invalid external value falls back to the safe disabled default");
});

test("preference synchronization retries failed runtime application and reports persistent failure", async () => {
  const initialized = await loadPreferenceApplicationHarness(0, { mode: "passthrough" });
  assert.equal((await initialized.module.syncSitePrefs()).ok, true);
  assert.deepEqual(initialized.deps.calls, [
    "sync",
    ["apply", { mode: "passthrough" }],
  ], "a snapshot recovered by store sync reaches the runtime subscriber");

  const recovered = await loadPreferenceApplicationHarness(1);
  recovered.module.emit({ mode: "passthrough" });
  assert.equal((await recovered.module.syncSitePrefs()).ok, true);
  assert.deepEqual(recovered.deps.calls, [
    "sync",
    ["apply", { mode: "passthrough" }],
    "snapshot",
    ["apply", { mode: "upscale", images: undefined }],
  ]);
  assert.equal(recovered.module.applicationFailure(), null);

  const failed = await loadPreferenceApplicationHarness(Infinity);
  failed.module.emit({ mode: "passthrough" });
  await assert.rejects(failed.module.syncSitePrefs(), /application exploded/);
  // syncSitePrefs clears the blocking flag as it reports, so a failed authoritative
  // replay can no longer block every later command; the message is retained for status.
  assert.equal(failed.module.applicationFailure(), null);
  assert.equal(failed.module.lastApplicationError(), "application exploded");

  const deleted = await loadPreferenceApplicationHarness(1, null, { images: true });
  deleted.module.emit({ mode: undefined });
  deleted.module.emit({ images: true });
  await deleted.module.drainOnly();
  assert.equal(deleted.module.applicationFailure(), "application exploded",
    "an unrelated successful patch cannot hide a failed deletion");

  assert.equal((await deleted.module.syncSitePrefs()).ok, true);
  assert.deepEqual(deleted.deps.calls, [
    ["apply", { mode: undefined }],
    ["apply", { images: true }],
    "sync",
    "snapshot",
    ["apply", { mode: undefined, images: true }],
  ], "authoritative replay includes tombstones for fields absent from the snapshot");
  assert.equal(deleted.module.lastApplicationError(), null);
});

test("status exposes configured interpolation and neural values without live runtimes", async () => {
  const statusModule = await loadStatusHarness();
  const status = statusModule.getStatus();
  assert.equal(status.neural, null);
  assert.equal(status.interpStats, null);
  assert.equal(status.statusVersion, 1);
  assert.equal(status.gpuState, "idle");
  assert.deepEqual(status.runtime.resources, { phase: "idle", reason: null });
  assert.equal(status.idlePowerSaving, false);
  assert.equal(status.autoQualityFallback, false);
  assert.deepEqual(status.persistence, {
    scope: "https://video.example", schemaVersion: 2,
    state: "ready", operation: null, errorOperation: null, pendingWrites: 0, error: null,
  });
  assert.equal(status.renderer.requestedEngine, "neural");
  assert.equal(status.renderer.effectiveEngine, "neural");
  assert.equal(status.renderer.performance.enabled, false);
  assert.deepEqual(status.renderer.nativePresentation, {
    pictureInPicture: false,
    directFullscreen: false,
    fullscreenElsewhere: false,
    nativeRequired: false,
  });
  assert.strictEqual(status.renderer.colorSupport, status.colorSupport);
  assert.equal(status.colorSupport.code, "color-not-checked");
  assert.deepEqual({
    neuralModel: status.neuralModel,
    interpModel: status.interpModel,
    interpResMode: status.interpResMode,
    interpTargetFps: status.interpTargetFps,
    interpAvOffsetMs: status.interpAvOffsetMs,
    interpStaticPassthrough: status.interpStaticPassthrough,
    interpAutoFallback: status.interpAutoFallback,
    interpLadder: status.interpLadder,
    interpInvert: status.interpInvert,
  }, {
    neuralModel: "span-lazy",
    interpModel: "blend",
    interpResMode: "quarter",
    interpTargetFps: 165,
    interpAvOffsetMs: -25,
    interpStaticPassthrough: false,
    interpAutoFallback: false,
    interpLadder: true,
    interpInvert: false,
  });

  const neuralStatus = (await loadStatusHarness(undefined, undefined, {
    mode: "upscale",
    hasVideo: true,
    presented: { mode: "upscale", engine: "neural" },
    neuralEntry: { key: "cda-vsr-4x", label: "CDA-VSR 4x", scale: 4 },
    presentation: {
      source: { width: 641, height: 359 },
      native: { width: 2564, height: 1436 },
      output: { width: 1282, height: 718 },
      ssimds: {
        source: { width: 2564, height: 1436 },
        output: { width: 1282, height: 718 },
      },
    },
  })).getStatus();
  assert.equal(neuralStatus.neural.nativeScale, 4);
  assert.equal(neuralStatus.neural.outputScale, 2);
  assert.equal(neuralStatus.neuralRuntime.nativeScale, 4);
  assert.equal(neuralStatus.neuralRuntime.outputScale, 2);
  assert.deepEqual(neuralStatus.renderer.presentation.native,
    { width: 2564, height: 1436 });

  const failedStatus = (await loadStatusHarness({
    state: "error", operation: null, errorOperation: "syncing", pending: 0,
    error: "opaque storage rejection", schemaVersion: 2, scope: "https://video.example",
  })).getStatus();
  assert.deepEqual(failedStatus.persistence, {
    scope: "https://video.example", schemaVersion: 2,
    state: "error", operation: null, errorOperation: "syncing", pendingWrites: 0,
    error: "opaque storage rejection",
  });

  const unavailableGpu = (await loadStatusHarness(undefined, {
    adapter: "unavailable", device: "uninitialized",
  })).getStatus();
  assert.equal(unavailableGpu.gpuState, "unavailable");
  assert.equal(unavailableGpu.runtime.adapter, "unavailable");

  const releasingGpu = (await loadStatusHarness(undefined, {
    adapter: "ready", device: "releasing",
  }, {
    resourcePhase: "releasing", resourceReason: "document-hidden",
  })).getStatus();
  assert.equal(releasingGpu.gpuState, "releasing");
  assert.deepEqual(releasingGpu.runtime.resources, {
    phase: "releasing", reason: "document-hidden",
  });

  const presentation = {
    committed: true,
    generation: 4,
    mode: "upscale",
    engine: "fsrcnnx",
    source: { width: 640, height: 360 },
    output: { width: 1920, height: 1080 },
    ssimds: {
      source: { width: 2560, height: 1440 },
      output: { width: 1920, height: 1080 },
    },
    sharpen: {
      source: { width: 1920, height: 1080 },
      output: { width: 1920, height: 1080 },
      strength: 1,
    },
    interpolation: { inverted: true },
  };
  const active = (await loadStatusHarness(undefined, undefined, {
    mode: "upscale",
    hasVideo: true,
    presented: { mode: "upscale", engine: "fsrcnnx" },
    presentation,
    interpolate: true,
    interpStats: {
      phase: "running",
      takeoverActive: true,
      presentation: {
        committed: true,
        generation: 3,
        gpu: true,
        sink: "renderer",
        source: { width: 640, height: 360 },
        output: { width: 1920, height: 1080 },
        framesIn: 5,
        framesOut: 8,
      },
    },
  })).getStatus();
  assert.strictEqual(active.presentation, presentation);
  assert.strictEqual(active.renderer.presentation, presentation);
  assert.equal(active.interpolationRuntime.takeoverActive, true);
  assert.strictEqual(active.interpolationRuntime.presentation, active.interpStats.presentation);
  assert.doesNotThrow(() => JSON.stringify(active));

  const inactive = (await loadStatusHarness(undefined, undefined, {
    mode: "upscale",
    presentation,
    presented: { mode: "off", engine: null },
    interpolate: true,
    suspended: true,
    interpStats: active.interpStats,
  })).getStatus();
  assert.equal(inactive.presentation, null, "a non-current renderer cannot leak its last commit");
  assert.equal(inactive.renderer.presentation, null);
  assert.equal(inactive.interpolationRuntime.takeoverActive, false);
  assert.equal(inactive.interpolationRuntime.presentation, null,
    "suspension cannot publish a stale interpolation takeover");

  const failed = (await loadStatusHarness(undefined, undefined, {
    hasVideo: true,
    interpolate: true,
    interpFailure: { stage: "capture", detail: "decoder failed" },
    interpStats: active.interpStats,
  })).getStatus();
  assert.equal(failed.interpolationRuntime.phase, "failed");
  assert.equal(failed.interpolationRuntime.takeoverActive, false);
  assert.equal(failed.interpolationRuntime.presentation, null,
    "terminal quarantine cannot publish a stale takeover while cleanup is queued");

  const uncommitted = (await loadStatusHarness(undefined, undefined, {
    hasVideo: true,
    interpolate: true,
    interpStats: { phase: "running", takeoverActive: true, presentation: null },
  })).getStatus();
  assert.equal(uncommitted.interpolationRuntime.phase, "active");
  assert.equal(uncommitted.interpolationRuntime.takeoverActive, false);
  assert.equal(uncommitted.interpolationRuntime.presentation, null,
    "generated frames are not a committed presentation");
});

test("interpolation quarantine identity includes every runtime-affecting preference", async () => {
  for (const [field, value] of [
    ["model", "blend"], ["res", "half"], ["target", 120], ["offset", 25],
    ["static", false], ["fallback", true], ["ladder", true], ["invert", false],
  ]) {
    const runtime = await loadRuntimeKeyHarness();
    const before = runtime.key();
    runtime.vary(field, value);
    assert.notEqual(runtime.key(), before, `${field} was omitted from the runtime configuration key`);
  }
});

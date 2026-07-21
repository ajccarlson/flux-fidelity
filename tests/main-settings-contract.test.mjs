import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../fsrcnnx-main.js", import.meta.url);
const source = await readFile(mainUrl, "utf8");
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

async function importHarness(code, deps = {}) {
  globalThis.__mainSettingsContractDeps = deps;
  const encoded = Buffer.from(code).toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${++revision}`);
}

async function loadFsrcnnxPlanHarness() {
  return importHarness(`
    const STANDARD_CASCADE_THRESHOLD = 2.4;
    ${policyToDepth}
    export { fsrcnnxPlan, upscalePresentationPlan };
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
      "hoverReveal", "allVideos", "images", "interpolate",
      "interpEngine", "interpResMode", "neuralModel", "interpTargetFps", "interpAvOffsetMs",
      "interpStaticPassthrough", "interpAutoFallback", "interpLadder", "interpInvert",
    ])};
    const siteSettingsStore = { write: async (value) => { deps.writes.push(value); } };
    const warn = () => {};
    const validateSitePreferencePatch = () => new Set();
    const recordPreferenceValidation = () => {};
    let mode = "upscale", requestedEngine = "artcnn", engine = "fsrcnnx", artVariant = "ArtCNN_C4F32_DS";
    let upscalePolicy = "force4", ssimdsEnabled = false, sharpenEnabled = true, sharpenStrength = 1.4;
    let optHoverReveal = true, optAllVideos = true;
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

async function loadRestoreHarness(prefs, { neuralModels = [{ key: "span" }] } = {}) {
  const restore = section("export async function restoreSitePrefs()", "function cancelPreferenceRestore()");
  const validationHelpers = section("function validateSitePreferencePatch", "function saveSitePrefs");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const DEFAULT_SETTING_FIELDS = ${JSON.stringify([
      "mode", "engine", "artVariant", "policy", "ssimds", "sharpen", "sharpenStrength",
      "hoverReveal", "allVideos", "images", "interpolate",
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
    let optHoverReveal = false, optAllVideos = false;
    let chainDepth = 1, pendingEngine = "rife_v4.26", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0;
    let interpStaticPassthroughPref = true, interpAutoFallbackPref = true;
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
    let interpStaticPassthroughPref = true, interpAutoFallbackPref = true;
    let interpLadderPref = false, interpInvertPref = true;
    let optInterpolate = true, engine = "fsrcnnx", video = { id: "video" };
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
        neuralModelKey };
    }
  `, { instance, saved: [], failures: [], fences: 0, neuralCalls: 0, neuralGate, writes: [] });
}

async function loadStatusHarness(storeHealth = {
  state: "ready", operation: null, errorOperation: null, pending: 0, error: null,
  schemaVersion: 2, scope: "https://video.example",
}, gpu = { adapter: "unrequested", device: "uninitialized" }) {
  const getStatus = section("export function getStatus()", "// ---- image upscaling");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const STATUS_VERSION = 1, GPU_RECOVERY_MAX_ATTEMPTS = 3;
    let mode = "off", primaryController = null, video = null, frameCount = 0;
    let primaryPresentationGeneration = 0;
    let activeModel = null, upscalePolicy = "display", ssimdsEnabled = true;
    let sharpenEnabled = false, sharpenStrength = 1, requestedEngine = "neural", engine = "neural";
    let artVariant = "ArtCNN_C4F32", chainDepth = 1, neuralModelKey = "span-lazy";
    let neuralEng = null, protectedSource = false, protectedReason = null;
    const selectedColorSupport = {
      supported: false,
      code: "color-not-checked",
      detail: "No decoded video is selected.",
      colorSpace: { primaries: null, transfer: null, matrix: null, fullRange: null },
    };
    let optHoverReveal = false, optAllVideos = false;
    let optImages = false, imageUpscaledCount = 0, optInterpolate = false, interpPausedByNeural = false;
    let interpolationTerminalQuarantine = null, interpolator = null;
    let pendingEngine = "blend", pendingResMode = "quarter", pendingTargetFps = 165;
    let pendingAvOffsetMs = -25, interpStaticPassthroughPref = false;
    let interpAutoFallbackPref = false, interpLadderPref = true, interpInvertPref = false;
    let pageSuspended = false, deviceRecoveryPromise = null, deviceRecoveryTimer = null;
    let gpuAdapterPhase = deps.gpu.adapter, gpuDevicePhase = deps.gpu.device;
    let gpuRecoveryPhase = "idle", gpuRecoveryAttempt = 0, gpuLastFailure = null, gpuRecoveredAt = null;
    let rendererFallback = null, neuralLastFailure = null, neuralFail = 0;
    let imageUpscaler = null, imageUpscalerInitPromise = null, imageLastFailure = null;
    let preferenceValidationFailure = null, preferenceApplicationFailure = null;
    const multiTargets = new Map();
    const _neuralList = [];
    const navigator = { gpu: {} };
    const findVideo = () => null;
    const siteHost = () => "video.example";
    const siteScope = () => "https://video.example";
    const siteSettingsStore = { health: () => deps.storeHealth };
    const currentPresentedRuntime = () => ({ mode: "off", engine: null });
    const interpolationQuarantineMatches = () => false;
    ${getStatus}
  `, { storeHealth, gpu });
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
  `, deps);
  return { module, deps };
}

async function loadRuntimeKeyHarness() {
  const runtimeKey = section("function interpolationRuntimeConfigKey()", "function interpolationQuarantineMatches");
  return importHarness(`
    let interpolationConfigGeneration = 7, pendingEngine = "rife_v4.26_fp16", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0, interpStaticPassthroughPref = true;
    let interpAutoFallbackPref = true, interpLadderPref = false, interpInvertPref = true;
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
  assert.equal(settings.setEngine("fsrcnnx-hi").ok, false,
    "the removed high engine is accepted only by stored-preference migration");

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
});

test("site persistence records requested intent and writes only selected fields", async () => {
  const { module, writes } = await loadSaveHarness();
  await module.saveSitePrefs();
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0], {
    mode: "upscale", engine: "artcnn", artVariant: "ArtCNN_C4F32_DS", policy: "force4",
    ssimds: false, sharpen: true, sharpenStrength: 1.4,
    hoverReveal: true, allVideos: true,
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

test("preference restore normalizes valid legacy values and rejects corrupt storage fields", async () => {
  const legacyHigh = await loadRestoreHarness({
    engine: "fsrcnnx-hi", artVariant: "ArtCNN_C4F32", policy: "force4",
  });
  assert.equal((await legacyHigh.restoreSitePrefs()).ok, true);
  assert.equal(legacyHigh.state().engine, "fsrcnnx");
  assert.equal(legacyHigh.state().policy, "force4");
  assert.equal(legacyHigh.state().chainDepth, 2);
  assert.equal(legacyHigh.state().preferenceValidationFailure, null);
  assert.deepEqual(legacyHigh.migrationWrites(), [{ engine: "fsrcnnx" }]);

  const legacyForce8 = await loadRestoreHarness({ engine: "fsrcnnx-hi", policy: "force8" });
  assert.equal((await legacyForce8.restoreSitePrefs()).ok, true);
  assert.equal(legacyForce8.state().engine, "fsrcnnx");
  assert.equal(legacyForce8.state().policy, "force4");
  assert.equal(legacyForce8.state().chainDepth, 2);
  assert.equal(legacyForce8.state().preferenceValidationFailure, null);
  assert.deepEqual(legacyForce8.migrationWrites(), [{ engine: "fsrcnnx", policy: "force4" }]);

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
    mode: "upscale", images: true, interpolate: true,
    interpEngine: "blend", interpResMode: "half", interpTargetFps: "144", interpAvOffsetMs: "25",
    interpStaticPassthrough: false, interpAutoFallback: false, interpLadder: true, interpInvert: false,
  });
  assert.equal((await valid.restoreSitePrefs()).ok, true);
  assert.deepEqual(valid.state(), {
    engine: "artcnn", neuralModelKey: "span", artVariant: "ArtCNN_C4F32_DN",
    policy: "force8", chainDepth: 3,
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1,
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
    sharpenStrength: "2",
  });
  assert.equal((await corrupt.restoreSitePrefs()).ok, true);
  assert.deepEqual(corrupt.state(), {
    engine: "fsrcnnx", neuralModelKey: "", artVariant: "ArtCNN_C4F32",
    policy: "force4", chainDepth: 2,
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1,
    pendingEngine: "rife_v4.26", pendingResMode: "auto",
    pendingTargetFps: "auto", pendingAvOffsetMs: 0,
    interpStaticPassthroughPref: true, interpAutoFallbackPref: true,
    interpLadderPref: false, interpInvertPref: true,
    preferenceValidationFailure: "Invalid stored settings: artVariant, engine, " +
      "images, interpAutoFallback, interpAvOffsetMs, interpEngine, interpInvert, interpLadder, " +
      "interpResMode, interpStaticPassthrough, interpTargetFps, interpolate, mode, neuralModel, " +
      "sharpenStrength",
  });
  assert.deepEqual(corrupt.migrationWrites(), [{ policy: "force4" }]);

  assert.equal(corrupt.applyValidation({ images: true }),
    "Invalid stored settings: artVariant, engine, interpAutoFallback, " +
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
    ok: true, pending: true, reason: "runtime failure",
    detail: "GPU state unavailable", resMode: "full",
  });
  assert.deepEqual(await config.setInterpolateModel("rife_v4.26_fp16"), {
    ok: true, pending: true, model: "rife_v4.26_fp16",
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
  assert.equal(failed.module.applicationFailure(), "application exploded");

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
  assert.equal(deleted.module.applicationFailure(), null);
});

test("status exposes configured interpolation and neural values without live runtimes", async () => {
  const statusModule = await loadStatusHarness();
  const status = statusModule.getStatus();
  assert.equal(status.neural, null);
  assert.equal(status.interpStats, null);
  assert.equal(status.statusVersion, 1);
  assert.equal(status.gpuState, "idle");
  assert.deepEqual(status.persistence, {
    scope: "https://video.example", schemaVersion: 2,
    state: "ready", operation: null, errorOperation: null, pendingWrites: 0, error: null,
  });
  assert.equal(status.renderer.requestedEngine, "neural");
  assert.equal(status.renderer.effectiveEngine, "neural");
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
});

test("interpolation quarantine identity includes every runtime-affecting preference", async () => {
  for (const [field, value] of [
    ["model", "blend"], ["res", "half"], ["target", 120], ["offset", 25],
    ["static", false], ["fallback", false], ["ladder", true], ["invert", false],
  ]) {
    const runtime = await loadRuntimeKeyHarness();
    const before = runtime.key();
    runtime.vary(field, value);
    assert.notEqual(runtime.key(), before, `${field} was omitted from the runtime configuration key`);
  }
});

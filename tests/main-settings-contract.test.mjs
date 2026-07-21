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

async function loadPolicyHarness({ engine = "fsrcnnx", policy = "display" } = {}) {
  const engineSetters = section("export function setEngine", "export function setHoverReveal");
  const policySetter = section("export function setPolicy", "// Restore saved preferences");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const ART_FILES = ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"];
    let engine = ${JSON.stringify(engine)}, upscalePolicy = ${JSON.stringify(policy)};
    let engineSelectionGeneration = 0, chainDepth = 1, artVariant = "ArtCNN_C4F32";
    let mode = "off", pageSuspended = false, primaryController = null, video = null;
    let interpPausedByNeural = false, neuralEng = null, artDiagLogged = false, device = null;
    const resetScaleSelection = () => deps.events.push("reset");
    const clearMultiTargets = () => deps.events.push("clear");
    const ensureNeural = async () => {};
    const resumeInterpolationAfterNeural = () => deps.events.push("resume");
    const ensureArtStages = async () => {};
    const ensureHiStages = async () => {};
    const saveSitePrefs = () => deps.events.push("save");
    const warn = () => {};
    ${policyToDepth}
    ${settingsContract}
    ${engineSetters}
    ${policySetter}
    export function state() { return { engine, policy: upscalePolicy, chainDepth, artVariant, engineSelectionGeneration }; }
  `, { events: [] });
}

async function loadSaveHarness() {
  const save = section("async function saveSitePrefs()", "function sendRuntimeMessage");
  const writes = [];
  const module = await importHarness(`
    const chrome = { storage: { local: { set: async (value) => deps.writes.push(value) } } };
    const deps = globalThis.__mainSettingsContractDeps;
    const siteHost = () => "video.example";
    const siteStorageKey = (host) => \`fsrcnnx_site:\${encodeURIComponent(host)}\`;
    let mode = "upscale", engine = "artcnn", artVariant = "ArtCNN_C4F32_DS";
    let upscalePolicy = "force4", ssimdsEnabled = false, sharpenEnabled = true, sharpenStrength = 1.4;
    let optHoverReveal = true, optAllVideos = true, debandEnabled = true, debandStrength = 1.6;
    let optImages = true, optInterpolate = false, neuralModelKey = "span";
    let pendingEngine = "rife_orig", pendingResMode = "half", pendingTargetFps = 144;
    let pendingAvOffsetMs = 35, interpStaticPassthroughPref = false;
    let interpAutoFallbackPref = false, interpLadderPref = true, interpInvertPref = false;
    ${save}
    export { saveSitePrefs };
  `, { writes });
  return { module, writes };
}

async function loadRestoreHarness(prefs) {
  const restore = section("export async function restoreSitePrefs()", "function cancelPreferenceRestore()");
  return importHarness(`
    const deps = globalThis.__mainSettingsContractDeps;
    const _neuralList = [{ key: "span" }];
    const neuralCatalogReady = Promise.resolve(_neuralList);
    const ART_FILES = ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"];
    let preferenceRestoreGeneration = 0, engineSelectionGeneration = 0;
    let engine = "fsrcnnx", neuralModelKey = "", artVariant = "ArtCNN_C4F32";
    let upscalePolicy = "display", ssimdsEnabled = true, sharpenEnabled = false, sharpenStrength = 1;
    let optHoverReveal = false, optAllVideos = false, debandEnabled = false, debandStrength = 1;
    let chainDepth = 1, pendingEngine = "rife_v4.26_fp16", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0;
    let interpStaticPassthroughPref = true, interpAutoFallbackPref = true;
    let interpLadderPref = false, interpInvertPref = true;
    const loadSitePrefs = async () => deps.prefs;
    const resetScaleSelection = () => {};
    const setMode = async (value) => { deps.calls.push(["mode", value]); return { ok: true }; };
    const setImages = async (value) => { deps.calls.push(["images", value]); return { ok: true }; };
    const setInterpolate = async (value) => { deps.calls.push(["interpolate", value]); return { ok: true }; };
    ${policyToDepth}
    ${settingsContract}
    ${restore}
    export function state() {
      return { engine, neuralModelKey, artVariant, policy: upscalePolicy, chainDepth,
        ssimdsEnabled, sharpenEnabled, sharpenStrength, debandStrength,
        pendingEngine, pendingResMode, pendingTargetFps, pendingAvOffsetMs,
        interpStaticPassthroughPref, interpAutoFallbackPref, interpLadderPref, interpInvertPref };
    }
  `, { prefs, calls: [] });
}

async function loadInterpolationConfigHarness(instance = null) {
  const configure = section("function configureInterpolator", "function scheduleInterpolatorGpuRestart");
  const primarySetters = section("function acceptedPendingInterpolationSetting", "export function listInterpolateModels");
  const secondarySetters = section("export async function setInterpolateInvert", "log(\"pipeline module loaded\")");
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
    const captureVideoSource = (candidate) => candidate;
    const sameVideoSource = (left, right) => left === right;
    const recordInterpolationStartFailure = (_video, _source, result) => deps.failures.push(result);
    const log = () => {};
    ${configure}
    ${primarySetters}
    ${secondarySetters}
    export { configureInterpolator };
    export function setInstance(value) { interpolator = value; }
    export function state() {
      return { pendingEngine, pendingResMode, pendingTargetFps, pendingAvOffsetMs,
        interpStaticPassthroughPref, interpAutoFallbackPref, interpLadderPref,
        interpInvertPref, interpolationConfigGeneration };
    }
  `, { instance, saved: [], failures: [] });
}

async function loadStatusHarness() {
  const getStatus = section("export function getStatus()", "// ---- image upscaling");
  return importHarness(`
    let mode = "off", primaryController = null, video = null, frameCount = 0;
    let activeModel = null, upscalePolicy = "display", ssimdsEnabled = true;
    let sharpenEnabled = false, sharpenStrength = 1, engine = "neural";
    let artVariant = "ArtCNN_C4F32", chainDepth = 1, neuralModelKey = "span-lazy";
    let neuralEng = null, protectedSource = false, protectedReason = null;
    let optHoverReveal = false, optAllVideos = false, debandEnabled = false, debandStrength = 1;
    let optImages = false, imageUpscaledCount = 0, optInterpolate = false, interpPausedByNeural = false;
    let interpolationTerminalQuarantine = null, interpolator = null;
    let pendingEngine = "blend", pendingResMode = "quarter", pendingTargetFps = 165;
    let pendingAvOffsetMs = -25, interpStaticPassthroughPref = false;
    let interpAutoFallbackPref = false, interpLadderPref = true, interpInvertPref = false;
    let pageSuspended = false, deviceRecoveryPromise = null, deviceRecoveryTimer = null;
    const multiTargets = new Map();
    const _neuralList = [];
    const navigator = { gpu: {} };
    const findVideo = () => null;
    const siteHost = () => "video.example";
    const interpolationQuarantineMatches = () => false;
    ${getStatus}
  `);
}

async function loadRuntimeKeyHarness() {
  const runtimeKey = section("function interpolationRuntimeConfigKey()", "function interpolationQuarantineMatches");
  return importHarness(`
    let interpolationConfigGeneration = 7, pendingEngine = "rife_orig", pendingResMode = "auto";
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
    ok: false, reason: "invalid engine", engine: "fsrcnnx", policy: "force3", chainDepth: 1,
  });
  assert.deepEqual(settings.state(), before, "an invalid engine must not mutate any state");

  assert.deepEqual(settings.setEngine("artcnn"), {
    ok: true, engine: "artcnn", policy: "display", chainDepth: 1,
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
});

test("site persistence records every authoritative interpolation preference", async () => {
  const { module, writes } = await loadSaveHarness();
  await module.saveSitePrefs();
  assert.equal(writes.length, 1);
  assert.deepEqual(writes[0]["fsrcnnx_site:video.example"], {
    mode: "upscale", engine: "artcnn", artVariant: "ArtCNN_C4F32_DS", policy: "force4",
    ssimds: false, sharpen: true, sharpenStrength: 1.4,
    hoverReveal: true, allVideos: true, deband: true, debandStrength: 1.6,
    images: true, interpolate: false,
    interpEngine: "rife_orig", interpResMode: "half", neuralModel: "span",
    interpTargetFps: 144, interpAvOffsetMs: 35, interpStaticPassthrough: false,
    interpAutoFallback: false, interpLadder: true, interpInvert: false,
  });
});

test("preference restore normalizes valid legacy values and rejects corrupt storage fields", async () => {
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
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1, debandStrength: 1,
    pendingEngine: "blend", pendingResMode: "half", pendingTargetFps: 144, pendingAvOffsetMs: 25,
    interpStaticPassthroughPref: false, interpAutoFallbackPref: false,
    interpLadderPref: true, interpInvertPref: false,
  });

  const corrupt = await loadRestoreHarness({
    engine: { value: "artcnn" }, neuralModel: "unknown", artVariant: "../../bad", policy: "force8",
    mode: "bogus", images: "true", interpolate: 1,
    interpEngine: "file:///tmp/model", interpResMode: "max", interpTargetFps: 999,
    interpAvOffsetMs: -101, interpStaticPassthrough: "false",
    interpAutoFallback: "false", interpLadder: 1, interpInvert: null,
    sharpenStrength: "2", debandStrength: Infinity,
  });
  assert.equal((await corrupt.restoreSitePrefs()).ok, true);
  assert.deepEqual(corrupt.state(), {
    engine: "fsrcnnx", neuralModelKey: "", artVariant: "ArtCNN_C4F32",
    policy: "display", chainDepth: 1,
    ssimdsEnabled: true, sharpenEnabled: false, sharpenStrength: 1, debandStrength: 1,
    pendingEngine: "rife_v4.26_fp16", pendingResMode: "auto",
    pendingTargetFps: "auto", pendingAvOffsetMs: 0,
    interpStaticPassthroughPref: true, interpAutoFallbackPref: true,
    interpLadderPref: false, interpInvertPref: true,
  });
});

test("interpolation setters preserve invalid-state immutability and configure a future runtime", async () => {
  const config = await loadInterpolationConfigHarness();
  const initial = config.state();
  assert.equal(config.setInterpolateRes("eighth").ok, false);
  assert.equal(config.setInterpolateAvOffset(Infinity).ok, false);
  assert.equal((await config.setInterpolateModel("unknown")).ok, false);
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
  assert.deepEqual(await config.setInterpolateModel("rife_orig"), {
    ok: true, pending: true, model: "rife_orig",
    reason: "runtime failure", detail: "model switch failed",
  });
  assert.equal(config.state().pendingResMode, "full");
  assert.equal(config.state().pendingEngine, "rife_orig");
});

test("status exposes configured interpolation and neural values without live runtimes", async () => {
  const statusModule = await loadStatusHarness();
  const status = statusModule.getStatus();
  assert.equal(status.neural, null);
  assert.equal(status.interpStats, null);
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

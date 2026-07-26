import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../src/core/fsrcnnx-main.js", import.meta.url);
const videoControllerUrl = new URL(
  "../src/core/fsrcnnx-video-controller.js",
  import.meta.url,
);
let revision = 0;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function successfulNeuralRun() {
  return {
    presentation: {
      source: { width: 640, height: 360 },
      output: { width: 1280, height: 720 },
      ssimds: null,
      sharpen: null,
    },
  };
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

async function loadSelectionCoordinator(deps) {
  const source = await readFile(mainUrl, "utf8");
  const sourceIdentity = section(
    source,
    "function captureVideoSource(target)",
    "function ensureCanvas()",
  );
  const production = section(
    source,
    "function videoMonitoringNeeded()",
    "export async function suspendDocument()",
  );
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    let mode = deps.mode || "upscale";
    let optInterpolate = deps.optInterpolate !== false;
    let pageSuspended = false;
    let video = deps.initialVideo || null;
    let primaryController = video ? { active: true, video } : null;
    let videoMonitor = null, layoutController = primaryController;
    let videoSelectionGeneration = 0, videoSwitchTail = Promise.resolve();
    let videoSelectionPendingGeneration = 0;
    let videoSelectionPendingRequest = null;
    let selectedVideoSource = null;
    let modeSelectionGeneration = 0;
    let interpolationSelectionGeneration = 0, interpolationConfigGeneration = 0;
    let interpolationTerminalQuarantine = null, interpolationStartFailureStreak = null;
    let pendingEngine = "rife_v4.26", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0;
    let interpAutoFallbackPref = false, interpLadderPref = false, interpInvertPref = true;
    let interpStaticPassthroughPref = true;
    let engineSelectionGeneration = 0;
    let engine = deps.engine || "fsrcnnx";
    let interpolator = deps.interpolator || null;
    let optAllVideos = false;
    let protectedSource = false, protectedReason = null;
    const uncheckedColorSupport = (detail) => ({ supported: false, code: "color-not-checked", detail });
    let selectedColorSupport = uncheckedColorSupport("not checked");
    let chainInverted = false, _texSource = null, activeModel = null;
    let frameTimes = [];
    let primaryAllocationRetirementPromise = null;
    let canvas = { style: {} };
    let neuralEng = { stop: () => deps.events.push(["neural-stop", video]) };
    const stopNeuralEngine = () => Promise.resolve(neuralEng?.stop?.());
    const hidePrimaryOverlays = () => {
      canvas.style.display = "none";
      canvas.style.opacity = "1";
    };
    class VideoSelectionMonitor {}
    const findVideo = () => deps.selected;
    const log = (...args) => deps.events.push(["log", ...args]);
    const warn = (...args) => deps.events.push(["warn", ...args]);
    const configureInterpolator = () => {
      if (deps.configureError) throw deps.configureError;
    };
    const cancelMainLoop = () => deps.events.push(["cancel", primaryController?.video || null]);
    const detach = () => {
      deps.events.push(["detach", primaryController?.video || null]);
      primaryController = null;
      layoutController = null;
    };
    const clearMultiTargets = () => deps.events.push(["clear-multi"]);
    const chainTap = (on) => deps.events.push(["chain-tap", on]);
    const resetScaleSelection = () => deps.events.push(["reset-scale"]);
    const resetPlaybackPerformanceFallback = () => deps.events.push(["reset-performance"]);
    const probeVideo = (candidate, options) => deps.probe(candidate, options);
    const notifyProtected = () => deps.events.push(["notify-protected", protectedReason]);
    const notifyState = () => deps.events.push(["notify-state", mode, !!primaryController]);
    const initWebGPU = () => deps.initWebGPU();
    const loadModels = () => deps.loadModels();
    const ensureNeural = async () => {};
    const resumeInterpolationAfterNeural = () => {};
    const attach = () => {
      primaryController = { active: true, video };
      layoutController = primaryController;
      deps.events.push(["attach", video]);
    };
    const scheduleMainLoop = () => deps.events.push(["schedule", video]);
    const syncMultiTargets = () => {};
    const retirePrimaryGpuAllocations = async (reason) => {
      deps.events.push(["retire-primary", reason]);
      return true;
    };
    ${sourceIdentity}
    selectedVideoSource = captureVideoSource(video);
    ${production}
    export function reconcile(candidate, force = true, sourceBoundary = false) {
      return queueVideoSelection(candidate, { force, sourceBoundary });
    }
    export function invalidateSelection() { videoSelectionGeneration++; }
    export function terminalFailure(failure) { return handleInterpolationTerminalFailure(failure); }
    export function retryInterpolation() { return requestInterpolationRetry("test configuration change"); }
    export function restartInterpolation() {
      return restartInterpolationForVideoSelection(videoSelectionGeneration);
    }
    export function monitorReconcile(candidate = video) {
      return scheduleVideoSelection(candidate, candidate, false);
    }
    export function quarantined(candidate = video) { return interpolationQuarantineMatches(candidate); }
    export function state() {
      return { mode, video, primaryController, protectedSource, protectedReason,
        videoSelectionGeneration, optInterpolate,
        interpolationQuarantined: interpolationQuarantineMatches(video) };
    }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadPresentationBoundary() {
  const source = await readFile(mainUrl, "utf8");
  const targetDimensions = section(source, "export function chainTargetDims()", "let lumaTexture");
  const resetPresentation = section(source, "function resetPresentedRuntime()", "// ---- advanced");
  const presentation = section(source, "function presentationDimensions(", "function positionCanvas");
  const harness = `
    import { videoPresentationState } from ${JSON.stringify(videoControllerUrl.href)};
    let canvas = null, video = null, primaryController = null, renderTargetOwner = null;
    let presentedCanvasVideo = null, presentedSourceW = 0, presentedSourceH = 0;
    let primaryPresentationGeneration = 0;
    let presentedVideoSource = null, presentedRuntimeMode = "off", presentedRuntimeEngine = null;
    let presentedCanvas = null;
    let presentedPresentation = null;
    let pageSuspended = false, device = {}, engine = "fsrcnnx";
    const document = { fullscreenElement: null, pictureInPictureElement: null };
    const lostDevices = new WeakSet();
    let reconcileRequests = 0;
    const captureVideoSource = (candidate) => candidate ? ({ video: candidate, currentSrc: candidate.currentSrc || "" }) : null;
    const sameVideoSource = (left, right) => !!left && !!right && left.video === right.video && left.currentSrc === right.currentSrc;
    const notifyState = () => {};
    const videoPageVisible = (candidate) => candidate?.pageVisible !== false;
    const videoMonitor = { request() { reconcileRequests++; } };
    const hidePrimaryOverlays = () => {
      if (canvas) canvas.style.display = "none";
      presentedCanvas = null;
    };
    const hideNeuralOverlay = () => {};
    const setPrimaryOverlayCanvas = () => {};
    ${targetDimensions}
    ${resetPresentation}
    ${presentation}
    export function setPrimary(nextVideo, nextCanvas, active = true) {
      video = nextVideo; canvas = nextCanvas;
      primaryController = { active, video: nextVideo };
      renderTargetOwner = null;
    }
    export function presentPrimary(mode = "upscale", activeEngine = "fsrcnnx", diagnostics = null) {
      showPresentedCanvas(mode, activeEngine, diagnostics);
    }
    export function presentSecondary(nextVideo, nextCanvas) {
      const saved = { video, canvas, renderTargetOwner };
      video = nextVideo; canvas = nextCanvas; renderTargetOwner = {};
      showPresentedCanvas("passthrough");
      ({ video, canvas, renderTargetOwner } = saved);
    }
    export function generation() { return primaryPresentationGeneration; }
    export function diagnostics() { return presentedPresentation; }
    export function reset() { return resetPresentedRuntime(); }
    export function requests() { return reconcileRequests; }
  `;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadPositionBoundary() {
  const source = await readFile(mainUrl, "utf8");
  const production = section(source, "function inShadowDom(node)", "function showPresentedCanvas(");
  const harness = `
    import { videoPresentationState } from ${JSON.stringify(videoControllerUrl.href)};
    class ShadowRoot {
      constructor(host) { this.host = host; this.fullscreenElement = null; }
      appendChild(node) { node.parentNode = this; }
    }
    const document = {
      body: { appendChild(node) { node.parentNode = this; } },
      fullscreenElement: null,
    };
    const textureSizeAllowed = () => true;
    ${production}
    export { ShadowRoot, document, positionVideoCanvas };
  `;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadEngineSelection(deps) {
  const source = await readFile(mainUrl, "utf8");
  const settingsContract = section(
    source,
    "// ---- validated setting contracts",
    "// ---- end validated setting contracts",
  );
  const production = section(source, "export function setEngine", "export function setArtVariant");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    const _neuralList = deps.neuralModels || [{ key: "span" }];
    let requestedEngine = "fsrcnnx", engine = "fsrcnnx", engineSelectionGeneration = 0, device = null;
    let mode = deps.mode, pageSuspended = !!deps.pageSuspended;
    const video = {};
    let primaryController = deps.active ? { active: true, video } : null;
    let optInterpolate = !!deps.optInterpolate;
    let interpPausedByNeural = false, neuralEng = null, chainDepth = 1;
    let upscalePolicy = "display", artDiagLogged = false, artVariant = "ArtCNN_C4F32";
    const resetScaleSelection = () => deps.events.push("reset");
    const clearMultiTargets = () => deps.events.push("clear-multi");
    const ensureNeural = (selection) => deps.ensureNeural(selection);
    const pauseInterpolationForNeural = () => {
      if (!optInterpolate) return;
      interpPausedByNeural = true;
      deps.events.push("pause-interp");
    };
    const resumeInterpolationAfterNeural = () => {
      if (!interpPausedByNeural) return;
      interpPausedByNeural = false;
      deps.events.push("resume-interp");
    };
    const reconcileDeviceRecoveryDemand = () => {
      deps.events.push("reconcile-recovery");
      return true;
    };
    const retireGpuResourcesIfIdle = async (reason) => {
      deps.events.push(["retire-if-idle", reason]);
      if (deps.retirementGate) await deps.retirementGate.promise;
      return { ok: true, released: true, reason };
    };
    const clearNeuralFallback = () => {};
    const activateNeuralFallback = () => { engine = "fsrcnnx"; };
    const stopNeuralEngine = () => Promise.resolve();
    const hidePrimaryOverlays = () => deps.events.push("hide-primary");
    const policyToDepth = () => 1;
    const ensureFsrcnnxStages = async () => {};
    const ensureHighStages = async () => {};
    const ensureArtStages = async () => {};
    const cancelPreferenceRestore = () => { deps.preferenceFences = (deps.preferenceFences || 0) + 1; };
    const saveSitePrefs = () => deps.events.push("save");
    const warn = (...args) => deps.events.push(["warn", ...args]);
    ${settingsContract}
    ${production}
    export function setSuspended(value) { pageSuspended = value; }
    export function state() { return { engine, engineSelectionGeneration, interpPausedByNeural }; }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadNeuralModelSelection(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(source, "export async function setNeuralModel", "// Shared present tail");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    const _neuralList = deps.neuralModels || [
      { key: "span" }, { key: "span-a" }, { key: "span-b" },
    ];
    const neuralCatalogReady = Promise.resolve(_neuralList);
    let neuralModelKey = "", requestedEngine = "neural", engine = deps.engine || "neural", engineSelectionGeneration = 2;
    let mode = deps.mode, pageSuspended = !!deps.pageSuspended;
    let optInterpolate = !!deps.optInterpolate;
    const video = {};
    let primaryController = deps.active ? { active: true, video } : null;
    const resetScaleSelection = () => {};
    const clearNeuralFallback = () => {};
    const pauseInterpolationForNeural = () => {
      if (optInterpolate) deps.events.push("pause-interp");
    };
    const reconcileDeviceRecoveryDemand = () => {
      deps.events.push("reconcile-recovery");
      return true;
    };
    const retireGpuResourcesIfIdle = async (reason) => {
      deps.events.push(["retire-if-idle", reason]);
      if (deps.retirementGate) await deps.retirementGate.promise;
      return { ok: true, released: true, reason };
    };
    const activateNeuralFallback = () => { engine = "fsrcnnx"; };
    const boundedRuntimeDetail = (error) => error?.message || String(error);
    const ensureNeural = (selection, options) => deps.ensureNeural(selection, options);
    const hidePrimaryOverlays = () => deps.events.push("hide-primary");
    const cancelPreferenceRestore = () => { deps.preferenceFences = (deps.preferenceFences || 0) + 1; };
    const saveSitePrefs = () => deps.events.push("save");
    const warn = (...args) => deps.events.push(["warn", ...args]);
    ${production}
    export function state() { return { neuralModelKey, engineSelectionGeneration }; }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadPreferenceRestore(deps) {
  const source = await readFile(mainUrl, "utf8");
  const settingsContract = section(
    source,
    "// ---- validated setting contracts",
    "// ---- end validated setting contracts",
  );
  const production = section(source, "export async function restoreSitePrefs()", "function cancelPreferenceRestore()");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    const _neuralList = deps.neuralModels || [{ key: "span" }];
    const isValidNeuralModelKey = (value) =>
      typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);
    const neuralCatalogReady = Promise.resolve(_neuralList);
    const ART_FILES = ["ArtCNN_C4F32"];
    let preferenceRestoreGeneration = 0, engineSelectionGeneration = 0;
    let requestedEngine = "fsrcnnx", engine = "fsrcnnx", neuralModelKey = "", artVariant = "ArtCNN_C4F32";
    let upscalePolicy = "display", ssimdsEnabled = true, sharpenEnabled = false, sharpenStrength = 1;
    let optHoverReveal = false, optAllVideos = false;
    let chainDepth = 1, pendingEngine = "rife_v4.26", pendingResMode = "auto";
    let pendingTargetFps = "auto", pendingAvOffsetMs = 0;
    let interpAutoFallbackPref = false, interpLadderPref = false, interpInvertPref = true;
    let interpStaticPassthroughPref = true;
    const loadSitePrefs = async () => deps.prefs;
    const policyToDepth = () => 1;
    const resetScaleSelection = () => {};
    const clearNeuralFallback = () => {};
    const siteSettingsStore = {
      health: () => ({ state: "ready", error: null }),
      write: async (value) => { (deps.writes ||= []).push(value); },
    };
    const validateSitePreferencePatch = () => new Set();
    const recordPreferenceValidation = () => {};
    const ensureNeural = async () => { deps.neuralCalls++; };
    const setMode = async (value) => { deps.events.push(["mode", value, deps.pageSuspended]); return { ok: true }; };
    const setImages = async (value) => { deps.events.push(["images", value]); return { ok: true }; };
    const setInterpolate = async (value) => { deps.events.push(["interpolate", value]); return { ok: true }; };
    ${settingsContract}
    ${production}
    export function state() { return { engine, neuralModelKey, engineSelectionGeneration }; }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadNeuralPresentation(deps) {
  const source = await readFile(mainUrl, "utf8");
  const sourceIdentity = section(source, "function captureVideoSource(target)", "function ensureCanvas()");
  const production = section(source, "function renderNeuralFrame()", "function ensureSharpenPipeline");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    let device = {}, mode = "upscale", requestedEngine = "neural", engine = "neural", adopting = false;
    let neuralBusy = false, neuralFail = 0, videoSelectionGeneration = 3;
    let neuralTemporalResetReason = null;
    let neuralTemporalResetGeneration = 0;
    let engineSelectionGeneration = 0;
    let lastSSimDS = false;
    const video = {
      videoWidth: 640, videoHeight: 360, currentSrc: "a", src: "a",
      seeking: false,
    };
    const primaryController = { active: true, video };
    const outputCanvas = { style: {} };
    const neuralEng = {
      ready: () => true,
      activeEntry: () => ({ scale: 2, padMultiple: 1 }),
      run: (...args) => {
        (deps.runArguments ||= []).push(args);
        return deps.run.promise;
      },
      canvas: () => outputCanvas,
      bumpSkip() {},
      stop() {},
    };
    const neuralFramePresentation = () => ({
      width: 1280, height: 720, ssimdsEnabled: true,
      sharpenEnabled: false, sharpenStrength: 1,
    });
    const positionVideoCanvas = () => true;
    const showPresentedCanvas = (...args) => {
      deps.events.push(["present", ...args]);
      return true;
    };
    const hidePrimaryOverlays = () => deps.events.push(["hide"]);
    const resetPresentedRuntime = () => true;
    const notifyState = () => deps.events.push(["state"]);
    const renderPassthrough = () => {};
    const warn = (...args) => deps.events.push(["warn", ...args]);
    const performanceFallbackEligible = () => false;
    const playbackPerformance = { observeRendererSkip() {} };
    let presentedRuntimeEngine = null;
    const resumeInterpolationAfterNeural = () => {};
    const activateNeuralFallback = () => { engine = "fsrcnnx"; };
    ${sourceIdentity}
    ${production}
    export function render(metadata) { renderNeuralFrame(metadata); }
    export function seek() { return handlePrimarySeek(primaryController); }
    export function setSeeking(value) { video.seeking = value; }
    export function setAdopting(value) { adopting = value; }
    export function changeSource(value) { video.currentSrc = value; video.src = value; }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadRuntimeNotifications(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(source, "function sendRuntimeMessage", "const srcCache");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    const chrome = { runtime: { sendMessage(message) { deps.messages.push(message); return deps.pending(); } } };
    const siteHost = () => "example.test";
    let protectedSource = false, mode = "upscale";
    const video = {
      videoWidth: 640, videoHeight: 360, currentSrc: "video",
      isConnected: true, style: { display: "block" },
    };
    const primaryController = { active: true, video };
    const device = {};
    const lostDevices = new WeakSet();
    const canvas = { isConnected: true, style: { display: "block" } };
    let presentedCanvas = canvas;
    let pageSuspended = false;
    let presentedRuntimeMode = "upscale", presentedRuntimeEngine = "fsrcnnx";
    let presentedCanvasVideo = video, presentedSourceW = 640, presentedSourceH = 360;
    let presentedVideoSource = { video, currentSrc: "video" };
    const captureVideoSource = (candidate) => ({ video: candidate, currentSrc: candidate.currentSrc || "" });
    const sameVideoSource = (left, right) => left?.video === right?.video && left?.currentSrc === right?.currentSrc;
    ${production}
    export { notifyProtected, notifyState };
    export function setPresentationAttachment({ videoConnected = true, canvasConnected = true,
      canvasDisplay = "block" } = {}) {
      video.isConnected = videoConnected;
      canvas.isConnected = canvasConnected;
      canvas.style.display = canvasDisplay;
    }
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadSecondarySourceBoundary(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "function handleSecondarySourceBoundary(target, owner)",
    "// Build per-target model instances",
  );
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    let pageSuspended = false, optAllVideos = true, mode = "upscale", adopting = false;
    let multiTargets = new Map();
    const videoMonitor = { request: () => deps.events.push("request") };
    const invalidateVideoColorSupport = (video) => deps.events.push(["invalidate-color", video]);
    ${production}
    export function register(target) { multiTargets.set(target.video, target); }
    export { handleSecondarySourceBoundary };
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

async function loadVideoSelectors(deps) {
  const source = await readFile(mainUrl, "utf8");
  const primary = section(source, "function findVideo()", "function captureVideoSource(target)");
  const secondary = section(source, "function findAllVideos()", "const MAX_SECONDARY_TARGETS");
  const harness = `
    const deps = globalThis.__videoOwnershipDeps;
    const window = { innerWidth: 1920, innerHeight: 1080 };
    const deepQueryVideos = () => deps.videos;
    const getComputedStyle = (target) => target.computedStyle || {
      display: "block", visibility: "visible", opacity: "1",
    };
    const isTaintedVideo = () => false;
    ${primary}
    ${secondary}
    export { findVideo, findAllVideos };
  `;
  globalThis.__videoOwnershipDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

function setup({
  initialVideo = { id: "A" },
  optInterpolate = true,
  engine = "fsrcnnx",
  startResults = [],
  startErrors = [],
} = {}) {
  const events = [];
  const interpolator = {
    running: optInterpolate,
    video: optInterpolate ? initialVideo : null,
    stop() {
      events.push(["interp-stop", this.video]);
      this.running = false;
      this.video = null;
    },
    async start(candidate) {
      events.push(["interp-start", candidate]);
      if (startErrors.length) throw startErrors.shift();
      const result = startResults.length ? startResults.shift() : { ok: true };
      this.video = result.ok ? candidate : null;
      this.running = !!result.ok;
      return result;
    },
  };
  return {
    initialVideo,
    selected: initialVideo,
    optInterpolate,
    engine,
    interpolator,
    events,
    probe: () => "ok",
    initWebGPU: async () => true,
    loadModels: async () => {},
  };
}

test("SPA replacement transfers renderer and interpolation ownership to one exact video", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup();
  const next = { id: "B" };
  const coordinator = await loadSelectionCoordinator(deps);

  assert.equal(await coordinator.reconcile(next), true);
  const state = coordinator.state();
  assert.equal(state.video, next);
  assert.equal(state.primaryController.video, next);
  assert.equal(deps.interpolator.video, next);
  assert.deepEqual(
    deps.events.filter(([type]) => ["detach", "attach", "interp-start"].includes(type)),
    [["detach", deps.initialVideo], ["attach", next], ["interp-start", next]],
  );
});

test("rapid A to B to C replacement cannot commit deferred B activation", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup({ optInterpolate: false });
  const initStarted = deferred();
  const releaseInit = deferred();
  let initCalls = 0;
  deps.initWebGPU = async () => {
    initCalls++;
    if (initCalls === 1) {
      initStarted.resolve();
      await releaseInit.promise;
    }
    return true;
  };
  const coordinator = await loadSelectionCoordinator(deps);
  const middle = { id: "B" };
  const latest = { id: "C" };

  const stale = coordinator.reconcile(middle);
  await initStarted.promise;
  const current = coordinator.reconcile(latest);
  releaseInit.resolve();

  assert.equal(await stale, false);
  assert.equal(await current, true);
  assert.equal(coordinator.state().video, latest);
  assert.deepEqual(
    deps.events.filter(([type]) => type === "attach").map(([, candidate]) => candidate),
    [latest],
  );
});

test("identical selector scans share one deferred video handoff", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup({ optInterpolate: false });
  const initStarted = deferred();
  const releaseInit = deferred();
  let initCalls = 0;
  deps.initWebGPU = async () => {
    initCalls++;
    initStarted.resolve();
    await releaseInit.promise;
    return true;
  };
  const coordinator = await loadSelectionCoordinator(deps);
  const next = {
    id: "B", currentSrc: "b", src: "b", videoWidth: 1280, videoHeight: 720,
  };

  const handoff = coordinator.reconcile(next, false);
  await initStarted.promise;
  const repeatedScans = Array.from({ length: 8 }, () => coordinator.reconcile(next, false));

  assert.ok(repeatedScans.every((operation) => operation === handoff),
    "an exact in-flight candidate and source must reuse its ownership operation");
  assert.equal(coordinator.state().videoSelectionGeneration, 1,
    "identical scans must not invalidate the generation they are awaiting");

  releaseInit.resolve();
  assert.deepEqual(await Promise.all([handoff, ...repeatedScans]), Array(9).fill(true));
  assert.equal(initCalls, 1);
  assert.equal(coordinator.state().video, next);
  assert.deepEqual(
    deps.events.filter(([type]) => type === "attach").map(([, candidate]) => candidate),
    [next],
  );
});

test("an invalidated deferred handoff is never reused by an identical scan", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup({ optInterpolate: false });
  const initStarted = deferred();
  const releaseInit = deferred();
  let initCalls = 0;
  deps.initWebGPU = async () => {
    initCalls++;
    initStarted.resolve();
    await releaseInit.promise;
    return true;
  };
  const coordinator = await loadSelectionCoordinator(deps);
  const next = {
    id: "B", currentSrc: "b", src: "b", videoWidth: 1280, videoHeight: 720,
  };

  const stale = coordinator.reconcile(next, false);
  await initStarted.promise;
  coordinator.invalidateSelection();
  const current = coordinator.reconcile(next, false);

  assert.notEqual(current, stale,
    "a scan after device/lifecycle invalidation must enqueue fresh ownership work");
  assert.equal(coordinator.state().videoSelectionGeneration, 3);

  releaseInit.resolve();
  assert.equal(await stale, false);
  assert.equal(await current, true);
  assert.equal(initCalls, 2,
    "the fresh operation must perform its own initialization attempt");
  assert.equal(coordinator.state().video, next);
});

test("an immediate queued B to A correction keeps A as the latest owner", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const original = {
    id: "A", currentSrc: "a", src: "a", videoWidth: 640, videoHeight: 360,
  };
  const middle = {
    id: "B", currentSrc: "b", src: "b", videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: original, optInterpolate: false });
  const coordinator = await loadSelectionCoordinator(deps);

  const stale = coordinator.reconcile(middle);
  const latest = coordinator.reconcile(original);

  assert.equal(await stale, false);
  assert.equal(await latest, true);
  assert.equal(coordinator.state().video, original);
  assert.equal(coordinator.state().primaryController.video, original);
  assert.deepEqual(
    deps.events.filter(([type]) => type === "attach").map(([, candidate]) => candidate),
    [original],
  );
});

test("missing and protected videos suspend without forgetting the requested mode, then recover", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup({ optInterpolate: false });
  const coordinator = await loadSelectionCoordinator(deps);

  assert.equal(await coordinator.reconcile(null), true);
  assert.equal(coordinator.state().mode, "upscale");
  assert.equal(coordinator.state().primaryController, null);

  const protectedVideo = { id: "protected" };
  deps.probe = () => "drm";
  assert.equal(await coordinator.reconcile(protectedVideo), true);
  assert.equal(coordinator.state().mode, "upscale");
  assert.equal(coordinator.state().protectedReason, "drm");
  assert.equal(coordinator.state().primaryController, null);

  deps.probe = () => "ok";
  assert.equal(await coordinator.reconcile(protectedVideo), true);
  assert.equal(coordinator.state().protectedSource, false);
  assert.equal(coordinator.state().primaryController.video, protectedVideo);
});

test("an unsupported decoded color space preserves renderer intent and recovers on reprobe", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = setup({ optInterpolate: false });
  const coordinator = await loadSelectionCoordinator(deps);
  const candidate = { id: "hdr" };
  const probeOptions = [];
  deps.probe = (_video, options) => {
    probeOptions.push(options);
    return "color-hdr-unsupported";
  };

  assert.equal(await coordinator.reconcile(candidate), true);
  assert.equal(coordinator.state().mode, "upscale");
  assert.equal(coordinator.state().protectedSource, true);
  assert.equal(coordinator.state().protectedReason, "color-hdr-unsupported");
  assert.equal(coordinator.state().primaryController, null);
  assert.ok(deps.events.some(([type, reason]) =>
    type === "notify-protected" && reason === "color-hdr-unsupported"));
  assert.equal(probeOptions[0]?.forceColor, true, "a new source cannot reuse stale color metadata");

  deps.probe = () => "ok";
  assert.equal(await coordinator.reconcile(candidate), true);
  assert.equal(coordinator.state().protectedSource, false);
  assert.equal(coordinator.state().primaryController.video, candidate);
});

test("standalone interpolation rechecks color support before starting", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const candidate = { id: "standalone-hdr" };
  const deps = setup({ initialVideo: candidate, optInterpolate: true });
  deps.mode = "off";
  deps.probe = () => "color-metadata-unavailable";
  const coordinator = await loadSelectionCoordinator(deps);
  deps.events.length = 0;

  assert.equal(await coordinator.restartInterpolation(), false);
  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().optInterpolate, true, "the user's interpolation preference is durable");
  assert.equal(coordinator.state().protectedReason, "color-metadata-unavailable");
  assert.equal(deps.events.some(([type]) => type === "interp-start"), false);
  assert.ok(deps.events.some(([type, reason]) =>
    type === "notify-protected" && reason === "color-metadata-unavailable"));
});

test("monitor reconciliation stops standalone interpolation after a color transition", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const candidate = { id: "adaptive-stream" };
  const deps = setup({ initialVideo: candidate, optInterpolate: true });
  deps.mode = "off";
  const coordinator = await loadSelectionCoordinator(deps);
  deps.events.length = 0;
  deps.probe = () => "color-wide-gamut-unsupported";

  assert.equal(await coordinator.monitorReconcile(), true);
  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().optInterpolate, true);
  assert.equal(coordinator.state().protectedReason, "color-wide-gamut-unsupported");
  assert.equal(coordinator.state().primaryController, null);
  assert.equal(deps.interpolator.running, false);
  assert.equal(deps.events.some(([type]) => type === "interp-start"), false);

  deps.probe = () => "ok";
  assert.equal(await coordinator.monitorReconcile(), true);
  assert.equal(coordinator.state().protectedSource, false);
  assert.equal(coordinator.state().primaryController.video, candidate);
  assert.equal(deps.interpolator.running, true);
  assert.equal(deps.interpolator.video, candidate);
});

test("a reused video element creates a fresh ownership generation when its media source changes", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const reused = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: reused, optInterpolate: false });
  const coordinator = await loadSelectionCoordinator(deps);

  reused.currentSrc = "https://example.test/b.mp4";
  reused.src = reused.currentSrc;
  reused.videoWidth = 1280;
  reused.videoHeight = 720;

  assert.equal(await coordinator.reconcile(reused, false), true);
  assert.equal(coordinator.state().video, reused);
  assert.equal(coordinator.state().videoSelectionGeneration, 1);
  assert.deepEqual(
    deps.events.filter(([type]) => ["detach", "attach"].includes(type)),
    [["detach", reused], ["attach", reused]],
  );
});

test("terminal interpolation quarantine suppresses monitor restarts but releases on source change", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const reused = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: reused, optInterpolate: true });
  const coordinator = await loadSelectionCoordinator(deps);

  assert.equal(coordinator.terminalFailure({
    video: reused,
    stage: "capture",
    detail: "deterministic decoder failure",
    source: {
      video: reused,
      currentSrc: reused.currentSrc,
      src: reused.src,
      srcObject: null,
    },
  }), true);
  deps.interpolator.stop();
  const startsBeforeReconcile = deps.events.filter(([type]) => type === "interp-start").length;

  assert.equal(await coordinator.reconcile(reused, false), undefined,
    "the normal monitor pass treats the exact quarantined tuple as settled");
  assert.equal(deps.events.filter(([type]) => type === "interp-start").length, startsBeforeReconcile);
  assert.equal(coordinator.state().optInterpolate, true, "runtime failure preserves requested intent");
  assert.equal(coordinator.state().interpolationQuarantined, true);
  assert.equal(await coordinator.reconcile(reused, true), true);
  assert.equal(deps.events.filter(([type]) => type === "interp-start").length, startsBeforeReconcile,
    "forced ownership reconciliation also honors the exact quarantine");

  reused.currentSrc = "https://example.test/b.mp4";
  reused.src = reused.currentSrc;
  reused.videoWidth = 1280;
  reused.videoHeight = 720;
  assert.equal(await coordinator.reconcile(reused, false), true);
  assert.equal(deps.events.filter(([type]) => type === "interp-start").length, startsBeforeReconcile + 1);
  assert.equal(deps.interpolator.video, reused);
  assert.equal(coordinator.state().interpolationQuarantined, false);
  assert.equal(coordinator.terminalFailure({
    video: reused,
    stage: "capture",
    source: {
      video: reused,
      currentSrc: "https://example.test/a.mp4",
      src: "https://example.test/a.mp4",
      srcObject: null,
    },
  }), false, "a stale same-element callback cannot quarantine the replacement source");
  assert.equal(coordinator.state().interpolationQuarantined, false);
});

test("deterministic interpolation start failure is quarantined without clearing intent", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const selected = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({
    initialVideo: selected,
    startResults: [{ ok: false, reason: "unsupported" }],
  });
  const coordinator = await loadSelectionCoordinator(deps);
  deps.interpolator.stop();

  assert.equal(await coordinator.reconcile(selected, true), true);
  const starts = deps.events.filter(([type]) => type === "interp-start").length;
  assert.equal(starts, 1);
  assert.equal(coordinator.state().interpolationQuarantined, true);
  assert.equal(coordinator.state().optInterpolate, true);
  await coordinator.reconcile(selected, false);
  assert.equal(deps.events.filter(([type]) => type === "interp-start").length, starts);
});

test("thrown interpolation setup failures enter the normal quarantine circuit", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const selected = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: selected });
  deps.configureError = new Error("invalid runtime configuration");
  const coordinator = await loadSelectionCoordinator(deps);
  deps.interpolator.stop();

  assert.equal(await coordinator.reconcile(selected, true), true);
  assert.equal(coordinator.state().interpolationQuarantined, true,
    "deterministic configuration exceptions quarantine immediately");
  assert.equal(coordinator.state().optInterpolate, true);
  const starts = deps.events.filter(([type]) => type === "interp-start").length;
  await coordinator.reconcile(selected, false);
  assert.equal(deps.events.filter(([type]) => type === "interp-start").length, starts);
});

test("configuration recovery restarts an already-running interpolator", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const selected = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: selected });
  deps.configureError = new Error("live setter still rejects the desired value");
  const coordinator = await loadSelectionCoordinator(deps);

  assert.equal(deps.interpolator.running, true);
  assert.equal(await coordinator.retryInterpolation(), true);
  assert.equal(coordinator.state().interpolationQuarantined, true,
    "a forced configuration retry must not skip an already-running instance");
  assert.ok(deps.events.some(([type]) => type === "interp-stop"));
});

test("thrown interpolation starts quarantine after the existing retry threshold", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const selected = {
    id: "A", currentSrc: "https://example.test/a.mp4", src: "https://example.test/a.mp4",
    videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({
    initialVideo: selected,
    startErrors: [new Error("transient start exception"), new Error("transient start exception")],
  });
  const coordinator = await loadSelectionCoordinator(deps);
  deps.interpolator.stop();

  assert.equal(await coordinator.reconcile(selected, true), true);
  assert.equal(coordinator.state().interpolationQuarantined, false);
  assert.equal(await coordinator.reconcile(selected, false), true);
  assert.equal(coordinator.state().interpolationQuarantined, true);
  assert.equal(coordinator.state().optInterpolate, true);
});

test("superseded interpolation start results remain retryable", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  for (const reason of ["cancelled", "superseded", "no video", "source-active"]) {
    const selected = {
      id: reason, currentSrc: `https://example.test/${encodeURIComponent(reason)}.mp4`,
      src: `https://example.test/${encodeURIComponent(reason)}.mp4`,
      videoWidth: 640, videoHeight: 360,
    };
    const deps = setup({
      initialVideo: selected,
      startResults: [{ ok: false, reason }, { ok: true }],
    });
    const coordinator = await loadSelectionCoordinator(deps);
    deps.interpolator.stop();

    assert.equal(await coordinator.reconcile(selected, true), true, reason);
    assert.equal(coordinator.state().interpolationQuarantined, false, reason);
    assert.equal(await coordinator.reconcile(selected, false), true, reason);
    assert.equal(deps.events.filter(([type]) => type === "interp-start").length, 2, reason);
    assert.equal(deps.interpolator.running, true, reason);
  }
});

test("standalone interpolation retains a direct source-boundary owner", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const candidate = {
    id: "standalone", currentSrc: "a", src: "a", videoWidth: 640, videoHeight: 360,
  };
  const deps = setup({ initialVideo: null, optInterpolate: true });
  deps.mode = "off";
  const coordinator = await loadSelectionCoordinator(deps);

  assert.equal(await coordinator.reconcile(candidate), true);
  assert.equal(coordinator.state().mode, "off");
  assert.equal(coordinator.state().primaryController.video, candidate,
    "the non-rendering controller owns loadstart/emptied/loadedmetadata events");
  assert.equal(deps.interpolator.video, candidate);

  candidate.currentSrc = "b";
  candidate.src = "b";
  assert.equal(await coordinator.reconcile(candidate, true, true), true);
  assert.equal(deps.interpolator.video, candidate);
  assert.ok(deps.events.filter(([type]) => type === "interp-stop").length >= 2);
});

test("a secondary source boundary retires the exact target before deferred reconciliation", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = { events: [] };
  const boundary = await loadSecondarySourceBoundary(deps);
  const video = {};
  const owner = {};
  const target = {
    video,
    controller: owner,
    canvas: { style: { display: "block" } },
    destroy() { deps.events.push("destroy"); },
  };
  boundary.register(target);

  assert.equal(boundary.handleSecondarySourceBoundary(target, {}), false);
  assert.equal(boundary.handleSecondarySourceBoundary(target, owner), true);
  assert.equal(target.failedReason, "source-changed");
  assert.equal(target.canvas.style.display, "none");
  assert.deepEqual(deps.events, [["invalidate-color", video], "destroy", "request"]);
  assert.equal(boundary.handleSecondarySourceBoundary(target, owner), false,
    "a retired target cannot process a second stale source callback");
});

test("primary and secondary selectors exclude CSS-hidden media despite positive geometry", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const makeVideo = (id, area, computedStyle = {}) => ({
    id,
    videoWidth: 1280,
    videoHeight: 720,
    paused: false,
    ended: false,
    readyState: 4,
    computedStyle: { display: "block", visibility: "visible", opacity: "1", ...computedStyle },
    getBoundingClientRect: () => ({ left: 0, top: 0, right: area, bottom: area, width: area, height: area }),
    closest: () => null,
  });
  const hiddenLarge = makeVideo("hidden-large", 900, { visibility: "hidden" });
  const transparent = makeVideo("transparent", 800, { opacity: "0" });
  const hiddenByApi = makeVideo("check-hidden", 700);
  hiddenByApi.checkVisibility = () => false;
  const visible = makeVideo("visible", 300);
  const deps = { videos: [hiddenLarge, transparent, hiddenByApi, visible] };
  const selectors = await loadVideoSelectors(deps);

  assert.equal(selectors.findVideo(), visible);
  assert.deepEqual(selectors.findAllVideos(), [visible]);
});

test("neural configuration stays lazy while rendering is off or document-suspended", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  for (const scenario of [
    { mode: "off", pageSuspended: false, active: false },
    { mode: "upscale", pageSuspended: true, active: true },
  ]) {
    let calls = 0;
    const deps = {
      ...scenario,
      events: [],
      ensureNeural: async () => { calls++; },
    };
    const selection = await loadEngineSelection(deps);
    assert.deepEqual(selection.setEngine("neural"), {
      ok: true,
      engine: "neural",
      activeEngine: "neural",
      policy: "display",
      chainDepth: 1,
      pending: false,
    });
    await Promise.resolve();
    assert.equal(calls, 0, `neural initialized eagerly for ${JSON.stringify(scenario)}`);

    const model = await loadNeuralModelSelection(deps);
    assert.deepEqual(await model.setNeuralModel("span"), { ok: true, model: "span" });
    assert.equal(calls, 0, "selecting a model records intent without starting an inactive engine");
  }
});

test("neural engine selection is rejected when the bundled catalog is empty", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = {
    mode: "off",
    active: false,
    neuralModels: [],
    events: [],
    ensureNeural: async () => assert.fail("empty catalog must not initialize neural runtime"),
  };
  const selection = await loadEngineSelection(deps);
  assert.deepEqual(selection.setEngine("neural"), {
    ok: false,
    reason: "no bundled neural models",
    engine: "fsrcnnx",
    activeEngine: "fsrcnnx",
    policy: "display",
    chainDepth: 1,
  });
  assert.deepEqual(selection.state(), {
    engine: "fsrcnnx",
    engineSelectionGeneration: 0,
    interpPausedByNeural: false,
  });
});

test("selecting neural while upscaling is inactive pauses standalone interpolation", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  let calls = 0;
  const deps = {
    mode: "off",
    active: false,
    optInterpolate: true,
    events: [],
    ensureNeural: async () => { calls++; },
  };
  const selection = await loadEngineSelection(deps);

  assert.equal(selection.setEngine("neural").pending, false);
  assert.equal(calls, 0, "inactive neural selection remains lazy");
  assert.equal(selection.state().interpPausedByNeural, true);
  assert.ok(deps.events.includes("pause-interp"));
  assert.ok(deps.events.some((event) => Array.isArray(event) &&
    event[0] === "retire-if-idle" && event[1] === "all-features-off"));

  // Switching back immediately is allowed while physical retirement drains;
  // the runtime start/device coordinators provide the actual reuse barrier.
  selection.setEngine("fsrcnnx");
  assert.equal(selection.state().interpPausedByNeural, false);
  assert.ok(deps.events.includes("resume-interp"),
    "leaving neural resumes the preserved interpolation request");
});

test("changing the neural model from fallback pauses inactive standalone interpolation", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  let calls = 0;
  const deps = {
    mode: "off",
    active: false,
    engine: "fsrcnnx",
    optInterpolate: true,
    events: [],
    ensureNeural: async () => { calls++; },
  };
  const model = await loadNeuralModelSelection(deps);

  assert.deepEqual(await model.setNeuralModel("span"), { ok: true, model: "span" });
  assert.equal(calls, 0, "inactive model selection remains lazy");
  assert.ok(deps.events.includes("pause-interp"));
  assert.ok(deps.events.includes("reconcile-recovery"));
  assert.ok(deps.events.some((event) => Array.isArray(event) &&
    event[0] === "retire-if-idle" && event[1] === "all-features-off"));
});

test("overlapping neural model switches use distinct generations and preserve the newest intent", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const pending = [];
  const deps = {
    mode: "upscale",
    active: true,
    events: [],
    ensureNeural(selection, options) {
      const operation = deferred();
      pending.push({ selection, options, operation });
      return operation.promise;
    },
  };
  const model = await loadNeuralModelSelection(deps);
  const first = model.setNeuralModel("span-a");
  const second = model.setNeuralModel("span-b");
  await Promise.resolve();

  assert.deepEqual(pending.map(({ selection, options }) => [selection, options.modelKey]), [
    [3, "span-a"],
    [4, "span-b"],
  ]);
  const superseded = new Error("neural activation superseded");
  superseded.code = "NEURAL_SUPERSEDED";
  pending[0].operation.reject(superseded);
  pending[1].operation.resolve({ key: "span-b" });

  assert.deepEqual(await first, { ok: true, model: "span-b", pending: true });
  assert.deepEqual(await second, { ok: true, model: "span-b" });
  assert.deepEqual(model.state(), { neuralModelKey: "span-b", engineSelectionGeneration: 4 });
});

test("neural model selection rejects keys outside the loaded catalog without mutating intent", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  let calls = 0;
  const model = await loadNeuralModelSelection({
    mode: "upscale",
    active: true,
    events: [],
    neuralModels: [{ key: "span" }],
    ensureNeural: async () => { calls++; },
  });

  assert.deepEqual(await model.setNeuralModel("unknown"), {
    ok: false,
    reason: "invalid neural model",
    model: "span",
  });
  assert.deepEqual(model.state(), { neuralModelKey: "", engineSelectionGeneration: 2 });
  assert.equal(calls, 0);
});

test("active neural initialization cancellation on page suspension preserves engine intent", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const init = deferred();
  const deps = {
    mode: "upscale",
    active: true,
    events: [],
    ensureNeural: () => init.promise,
  };
  const selection = await loadEngineSelection(deps);
  selection.setEngine("neural");
  selection.setSuspended(true);
  init.resolve(Promise.reject(new Error("neural initialization cancelled by stop")));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(selection.state().engine, "neural");
});

test("restoring saved neural preferences while off or hidden does not initialize or downgrade them", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  for (const [mode, pageSuspended] of [["off", false], ["upscale", true]]) {
    const deps = {
      pageSuspended,
      neuralCalls: 0,
      events: [],
      prefs: { engine: "neural", neuralModel: "span", mode, images: false, interpolate: false },
    };
    const restore = await loadPreferenceRestore(deps);
    assert.equal((await restore.restoreSitePrefs()).ok, true);
    assert.equal(deps.neuralCalls, 0);
    assert.equal(restore.state().engine, "neural");
    assert.equal(restore.state().neuralModelKey, "span");
  }
});

test("restoring neural intent with an empty catalog migrates to the standard renderer", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = {
    pageSuspended: false,
    neuralCalls: 0,
    neuralModels: [],
    events: [],
    writes: [],
    prefs: {
      engine: "neural",
      neuralModel: "span",
      mode: "off",
      images: false,
      interpolate: false,
    },
  };
  const restore = await loadPreferenceRestore(deps);
  assert.equal((await restore.restoreSitePrefs()).ok, true);
  assert.equal(deps.neuralCalls, 0);
  assert.equal(restore.state().engine, "fsrcnnx");
  assert.equal(restore.state().neuralModelKey, "");
  assert.deepEqual(deps.writes, [{ engine: "fsrcnnx", neuralModel: null }]);
});

test("neural completion cannot present across adoption or same-element source replacement", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });

  for (const invalidate of [
    (renderer) => renderer.setAdopting(true),
    (renderer) => renderer.changeSource("b"),
  ]) {
    const run = deferred();
    const deps = { run, events: [] };
    const renderer = await loadNeuralPresentation(deps);
    renderer.render();
    invalidate(renderer);
    run.resolve({
      presentation: {
        source: { width: 640, height: 360 },
        output: { width: 1280, height: 720 },
        ssimds: null,
        sharpen: null,
      },
    });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(deps.events.some(([type]) => type === "present"), false);
  }
});

test("a media seek resets the next non-busy neural run exactly once", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const run = deferred();
  const deps = { run, runArguments: [], events: [] };
  const renderer = await loadNeuralPresentation(deps);

  renderer.render({ mediaTime: 1, presentedFrames: 10 });
  assert.equal(deps.runArguments.length, 1);
  assert.deepEqual(deps.runArguments[0][4], {
    mediaTime: 1,
    presentedFrames: 10,
  });

  assert.equal(renderer.seek(), true);
  assert.deepEqual(
    deps.events.slice(-2).map(([type]) => type),
    ["hide", "state"],
    "a seek must relinquish the previously presented overlay immediately",
  );
  renderer.render({ mediaTime: 4, presentedFrames: 11 });
  assert.equal(
    deps.runArguments.length,
    1,
    "a dropped busy frame must not consume the pending seek reset",
  );

  run.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    deps.events.some(([type]) => type === "present"),
    false,
    "a pre-seek run must not present after the seek boundary",
  );

  renderer.render({ mediaTime: 4, presentedFrames: 12 });
  assert.deepEqual(deps.runArguments[1][4], {
    mediaTime: 4,
    presentedFrames: 12,
    reset: true,
    resetReason: "seek",
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    deps.events.filter(([type]) => type === "present").length,
    1,
    "the first generation-matched post-seek result may present",
  );

  renderer.render({ mediaTime: 4.04, presentedFrames: 13 });
  assert.deepEqual(deps.runArguments[2][4], {
    mediaTime: 4.04,
    presentedFrames: 13,
  });
});

test("neural capture stays hidden throughout the seeking-to-seeked window", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const run = deferred();
  const deps = { run, runArguments: [], events: [] };
  const renderer = await loadNeuralPresentation(deps);

  renderer.render({ mediaTime: 1, presentedFrames: 10 });
  renderer.setSeeking(true);
  assert.equal(renderer.seek(), true);
  run.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));

  renderer.render({ mediaTime: 3, presentedFrames: 11 });
  assert.equal(
    deps.runArguments.length,
    1,
    "no neural capture may start while the media element is seeking",
  );
  assert.equal(
    deps.events.some(([type]) => type === "present"),
    false,
    "a completion during seeking must remain hidden",
  );

  renderer.setSeeking(false);
  assert.equal(renderer.seek(), true);
  renderer.render({ mediaTime: 4, presentedFrames: 12 });
  assert.deepEqual(deps.runArguments[1][4], {
    mediaTime: 4,
    presentedFrames: 12,
    reset: true,
    resetReason: "seek",
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    deps.events.filter(([type]) => type === "present").length,
    1,
    "a fresh post-seek result may reveal the overlay",
  );
});

test("a seek reset survives failed capture and failed child inference", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });

  for (const phase of ["capture", "child-run"]) {
    await t.test(phase, async () => {
      const failed = deferred();
      const deps = { run: failed, runArguments: [], events: [] };
      const renderer = await loadNeuralPresentation(deps);

      renderer.seek();
      renderer.render({ mediaTime: 7, presentedFrames: 20 });
      assert.equal(deps.runArguments[0][4].resetReason, "seek");

      failed.reject(new Error(`injected ${phase} failure`));
      await new Promise((resolve) => setImmediate(resolve));

      const retry = deferred();
      deps.run = retry;
      renderer.render({ mediaTime: 7.04, presentedFrames: 21 });
      assert.deepEqual(deps.runArguments[1][4], {
        mediaTime: 7.04,
        presentedFrames: 21,
        reset: true,
        resetReason: "seek",
      });
      retry.resolve(successfulNeuralRun());
      await new Promise((resolve) => setImmediate(resolve));
    });
  }
});

test("a seek reset survives an empty child response", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const empty = deferred();
  const deps = { run: empty, runArguments: [], events: [] };
  const renderer = await loadNeuralPresentation(deps);

  renderer.seek();
  renderer.render({ mediaTime: 5, presentedFrames: 25 });
  empty.resolve(null);
  await new Promise((resolve) => setImmediate(resolve));

  const retry = deferred();
  deps.run = retry;
  renderer.render({ mediaTime: 5.04, presentedFrames: 26 });
  assert.equal(deps.runArguments[1][4].resetReason, "seek");
  retry.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));
});

test("a second seek during a successful reset run remains pending", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const first = deferred();
  const deps = { run: first, runArguments: [], events: [] };
  const renderer = await loadNeuralPresentation(deps);

  renderer.seek();
  renderer.render({ mediaTime: 2, presentedFrames: 30 });
  assert.equal(deps.runArguments[0][4].resetReason, "seek");
  renderer.seek();

  first.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));

  const second = deferred();
  deps.run = second;
  renderer.render({ mediaTime: 8, presentedFrames: 31 });
  assert.deepEqual(deps.runArguments[1][4], {
    mediaTime: 8,
    presentedFrames: 31,
    reset: true,
    resetReason: "seek",
  });
  second.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));

  const third = deferred();
  deps.run = third;
  renderer.render({ mediaTime: 8.04, presentedFrames: 32 });
  assert.deepEqual(deps.runArguments[2][4], {
    mediaTime: 8.04,
    presentedFrames: 32,
  });
  third.resolve(successfulNeuralRun());
  await new Promise((resolve) => setImmediate(resolve));
});

test("renderer runtime notifications observe rejected extension-message promises", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  let observed = 0;
  const deps = {
    messages: [],
    pending: () => ({ catch(handler) { observed++; handler(new Error("context closed")); } }),
  };
  const notifications = await loadRuntimeNotifications(deps);

  assert.doesNotThrow(() => notifications.notifyState());
  assert.doesNotThrow(() => notifications.notifyProtected());
  assert.equal(observed, 2);
  assert.deepEqual(deps.messages.map(({ type }) => type), ["FSRCNNX_STATE", "FSRCNNX_PROTECTED"]);
});

test("renderer notifications fail closed when a committed presentation detaches or hides", async (t) => {
  const previous = globalThis.__videoOwnershipDeps;
  t.after(() => { globalThis.__videoOwnershipDeps = previous; });
  const deps = { messages: [], pending: () => undefined };
  const notifications = await loadRuntimeNotifications(deps);

  notifications.notifyState();
  assert.equal(deps.messages.at(-1).mode, "upscale");

  notifications.setPresentationAttachment({ videoConnected: false });
  notifications.notifyState();
  assert.equal(deps.messages.at(-1).mode, "off", "a detached source cannot remain presented");

  notifications.setPresentationAttachment({ canvasConnected: false });
  notifications.notifyState();
  assert.equal(deps.messages.at(-1).mode, "off", "a detached sink cannot remain presented");

  notifications.setPresentationAttachment({ canvasDisplay: "none" });
  notifications.notifyState();
  assert.equal(deps.messages.at(-1).mode, "off", "a hidden sink cannot remain presented");
});

test("main source keeps secondary mutable state and neural completion target-scoped", async () => {
  const source = await readFile(mainUrl, "utf8");
  const targetSwap = section(source, "function withTarget(t, fn)", "// Find all qualifying videos");
  for (const fragment of [
    "layoutController = t.controller",
    "renderTargetOwner = t",
    "activeModel = t.activeModel",
    "dispRGB = t.dispRGB",
    "chainedFsrcnnx = t.chainedFsrcnnx",
    "highStages = t.highStages",
    "chainedHigh = t.chainedHigh",
    "_scaleHeld = t.scaleHeld",
    "t.activeModel = activeModel",
    "t.chainedHigh = chainedHigh",
    "t.scaleHeld = _scaleHeld",
  ]) {
    assert.match(targetSwap, new RegExp(fragment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
  const neural = section(source, "function renderNeuralFrame()", "function ensureSharpenPipeline");
  assert.match(neural, /video === runVideo/);
  assert.match(neural, /primaryController === runController/);
  assert.match(neural, /runVideoGeneration === videoSelectionGeneration/);
  assert.match(source, /const MAX_SECONDARY_TARGETS = 2/);
  assert.doesNotMatch(source, /\.rvfc\b/);
  for (const [startMarker, endMarker] of [
    ["export function setEngine", "export function setArtVariant"],
    ["export function setArtVariant", "export function setHoverReveal"],
    ["export function setPolicy", "// Restore saved preferences"],
  ]) {
    assert.match(section(source, startMarker, endMarker), /clearMultiTargets\(\)/,
      `${startMarker} must retire secondary target-local model selection state`);
  }
});

test("interpolation dimensions require a successful frame from the current primary source", async () => {
  const boundary = await loadPresentationBoundary();
  const primaryVideo = { videoWidth: 640, videoHeight: 360 };
  const primaryCanvas = { width: 1280, height: 720, style: { display: "none" } };
  boundary.setPrimary(primaryVideo, primaryCanvas);

  assert.equal(boundary.chainTargetDims(), null, "a default or hidden canvas is not a presentation");
  boundary.presentPrimary("upscale", "fsrcnnx", {
    source: { width: 640, height: 360 },
    ssimds: {
      source: { width: 2560, height: 1440 },
      output: { width: 1280, height: 720 },
    },
    sharpen: {
      source: { width: 1280, height: 720 },
      output: { width: 1280, height: 720 },
      strength: 0.8,
    },
    interpolation: { inverted: true },
  });
  assert.deepEqual(boundary.chainTargetDims(), { w: 1280, h: 720 });
  assert.deepEqual(boundary.diagnostics(), {
    committed: true,
    generation: 1,
    mode: "upscale",
    engine: "fsrcnnx",
    source: { width: 640, height: 360 },
    output: { width: 1280, height: 720 },
    ssimds: {
      source: { width: 2560, height: 1440 },
      output: { width: 1280, height: 720 },
    },
    sharpen: {
      source: { width: 1280, height: 720 },
      output: { width: 1280, height: 720 },
      strength: 0.8,
    },
    interpolation: { inverted: true },
  });
  const primaryGeneration = boundary.generation();

  boundary.presentSecondary(
    { videoWidth: 320, videoHeight: 180 },
    { width: 640, height: 360, style: { display: "none" } },
  );
  assert.equal(boundary.generation(), primaryGeneration,
    "secondary presentation cannot replace primary presentation identity");
  assert.equal(boundary.diagnostics().generation, primaryGeneration,
    "secondary presentation cannot replace primary submitted diagnostics");
  assert.deepEqual(boundary.chainTargetDims(), { w: 1280, h: 720 });

  primaryVideo.videoWidth = 1920;
  primaryVideo.videoHeight = 1080;
  assert.equal(boundary.chainTargetDims(), null,
    "same-element source geometry changes invalidate stale dimensions");
  primaryCanvas.width = 3840;
  primaryCanvas.height = 2160;
  boundary.presentPrimary();
  assert.deepEqual(boundary.chainTargetDims(), { w: 3840, h: 2160 });
  assert.deepEqual(boundary.diagnostics().output, { width: 3840, height: 2160 });
  assert.equal(boundary.diagnostics().ssimds, null);
  assert.equal(boundary.diagnostics().sharpen, null);

  primaryVideo.pageVisible = false;
  boundary.presentPrimary();
  assert.equal(primaryCanvas.style.display, "none");
  assert.equal(boundary.chainTargetDims(), null);
  assert.equal(boundary.requests(), 1);

  const hiddenSecondary = {
    videoWidth: 320, videoHeight: 180, pageVisible: false,
  };
  const hiddenSecondaryCanvas = { width: 640, height: 360, style: { display: "block" } };
  boundary.presentSecondary(hiddenSecondary, hiddenSecondaryCanvas);
  assert.equal(hiddenSecondaryCanvas.style.display, "none",
    "a secondary canvas cannot publish over page-hidden media");
  assert.equal(boundary.requests(), 2);

  assert.equal(boundary.reset(), true);
  assert.equal(boundary.diagnostics(), null, "detach/off/suspension reset submitted diagnostics");
  assert.equal(boundary.chainTargetDims(), null);
});

test("production overlays expose stable primary, secondary, and interpolation markers", async () => {
  const mainSource = await readFile(mainUrl, "utf8");
  const interpolationSource = await readFile(
    new URL("../src/core/fsrcnnx-interpolate.js", import.meta.url),
    "utf8",
  );
  assert.match(
    section(mainSource, "function ensureCanvas()", "// True if"),
    /setAttribute\?\.\("data-fsrcnnx-overlay", "primary"\)/,
  );
  assert.match(
    section(mainSource, "class MultiTarget", "let multiTargets"),
    /setAttribute\?\.\("data-fsrcnnx-overlay", "secondary"\)/,
  );
  assert.match(
    interpolationSource,
    /setAttribute\?\.\("data-fsrcnnx-overlay", "interpolation"\)/,
  );
});

test("Neural activation resolves an empty saved selection to the bundled default", async () => {
  const source = await readFile(mainUrl, "utf8");
  const activation = section(
    source,
    "async function ensureNeural(",
    "export async function setNeuralModel",
  );

  assert.match(
    activation,
    /modelKey \|\| neuralModelKey \|\| _neuralList\[0\]\?\.key/,
  );
  assert.match(activation, /neuralEng\.init\(requestedModelKey\)/);
});

test("main canvas mounting respects direct and container fullscreen inside shadow DOM", async () => {
  const boundary = await loadPositionBoundary();
  const host = {};
  const root = new boundary.ShadowRoot(host);
  const parent = {
    appendChild(node) { node.parentNode = this; },
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
  };
  const video = {
    parentElement: parent,
    getRootNode: () => root,
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 300, height: 200 }),
  };
  const canvas = { style: { display: "block" }, width: 0, height: 0, parentNode: null };
  boundary.document.fullscreenElement = host;

  root.fullscreenElement = video;
  assert.equal(boundary.positionVideoCanvas(video, canvas, null, 600, 400), false);
  assert.equal(canvas.style.display, "none");

  const player = {
    contains: (candidate) => candidate === video,
    appendChild(node) { node.parentNode = this; },
  };
  root.fullscreenElement = player;
  assert.equal(boundary.positionVideoCanvas(video, canvas, null, 600, 400), true);
  assert.equal(canvas.parentNode, player);

  root.fullscreenElement = null;
  assert.equal(boundary.positionVideoCanvas(video, canvas, null, 600, 400), true);
  assert.equal(canvas.parentNode, root,
    "a retargeted fullscreen shadow host mounts into the rendered shadow root");
});

test("site preferences use field-level writes and restore lifecycle false values without echoing", async () => {
  const source = await readFile(mainUrl, "utf8");
  const save = section(source, "function currentSitePreferenceValues()", "export async function flushPreferenceWrites()");
  assert.match(save, /siteSettingsStore\.write\(patch\)/);
  assert.match(save, /for \(const field of fields\)/);
  assert.doesNotMatch(save, /chrome\.storage\.local\.(?:get|set)/,
    "main delegates granular writes without a cross-tab read-modify-write record");

  const restore = section(source, "export async function restoreSitePrefs()", "function cancelPreferenceRestore()");
  assert.match(restore, /setMode\(savedMode, restoreToken, \{ persist: false \}\)/);
  assert.match(restore, /setImages\(wantImages, \{ persist: false \}\)/);
  assert.match(restore, /setInterpolate\(wantInterp, restoreToken, \{ persist: false \}\)/);
  assert.doesNotMatch(restore, /scheduleAutoEnable\(/,
    "restored intent should remain monitored beyond the old polling window");
});

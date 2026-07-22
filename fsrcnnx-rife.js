// fsrcnnx-rife.js — RIFE frame interpolation via ONNX Runtime Web (WebGPU EP).
// Loads the bundled ORT runtime and one of the bundled RIFE models, and exposes
// interpolate(frameA, frameB, t) -> ImageBitmap-able result for the tween.
//
// The adapters below cover the bundled seven-channel timestep-aware v4.26 models.
// A missing or unsupported model falls back to blend.
// Input is a single NCHW tensor containing two RGB frames and, when supported, a
// timestep plane. Output is NCHW RGB in the 0..1 range.
// Padding: RIFE needs H,W divisible by a factor (usually 32). We pad to the next
// multiple and crop the result back.

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";

let ort = null;             // the ORT module
let session = null;         // InferenceSession
let modelLoadTried = false;
let modelAvailable = false;
let modelGeneration = 0;
let initPromise = null;
let initPromiseGeneration = -1;
let pendingGpuTeardown = Promise.resolve();
let sessionModelKey = null;
let deferredDeviceGuards = [];
let deferredGuardReleaseTail = Promise.resolve();
let activeCpuRuns = 0;
let cpuIdleResolvers = [];
let cpuInterpolateTail = Promise.resolve();
let ortLoadPromise = null;
let ortSessionCreateTail = Promise.resolve();
const ortSessionDevices = new WeakMap();
const deviceLossListeners = new Set();
let deviceInvalidationTail = Promise.resolve();
let deviceInvalidationPromise = null;
const deviceInvalidations = new WeakMap();
const activeDeviceInvalidations = new Set();
const deviceInvalidationFailures = new Set();
const watchedOrtDevices = new WeakSet();
const lostOrtDevices = new WeakSet();
let runtimeDisposalPromise = null;

// Consumers which own presentation state (notably Interpolator) need to leave
// GPU-present mode as soon as either ORT's shared device or a standalone blend
// device is lost.  Keep this deliberately tiny: session/resource invalidation
// remains owned by this module and listeners only coordinate their lifecycle.
export function addDeviceLossListener(listener) {
  if (typeof listener !== "function") return () => {};
  deviceLossListeners.add(listener);
  return () => deviceLossListeners.delete(listener);
}

function emitDeviceLoss(device, info) {
  for (const listener of [...deviceLossListeners]) {
    try { listener(device, info); } catch {}
  }
}

// The GPU-resident helper also observes device loss, but it is deliberately torn
// down when WebGPU canvas presentation is unavailable.  The CPU/readback fallback
// still uses the committed ORT WebGPU session, so the session's device needs its
// own watcher or that path can keep reporting a dead session as ready forever.
function watchOrtDevice(ownerDevice) {
  if (!ownerDevice || watchedOrtDevices.has(ownerDevice) || !ownerDevice.lost?.then) return;
  watchedOrtDevices.add(ownerDevice);
  const handleLoss = (info) => {
    lostOrtDevices.add(ownerDevice);
    // The identity-aware invalidator preserves a healthy current session while
    // still releasing this device's non-current adoption guard, if one exists.
    invalidateDevice(ownerDevice, info).catch((error) =>
      console.warn("[RIFE] ORT device-loss invalidation failed:", error.message));
  };
  ownerDevice.lost.then(
    handleLoss,
    (error) => handleLoss({ reason: "unknown", message: error?.message || String(error) }),
  );
}

function refreshDeviceInvalidationBarrier() {
  if (!activeDeviceInvalidations.size && !deviceInvalidationFailures.size) {
    deviceInvalidationPromise = null;
    return null;
  }
  const operations = [...activeDeviceInvalidations];
  const latchedFailures = [...deviceInvalidationFailures];
  const barrier = Promise.allSettled(operations).then((results) => {
    const failures = new Set(latchedFailures);
    for (const result of results) {
      if (result.status === "rejected") addDeviceInvalidationFailure(failures, result.reason);
    }
    if (failures.size) {
      throw new AggregateError([...failures], "RIFE device invalidation barrier failed");
    }
  });
  barrier.catch(() => {});
  deviceInvalidationPromise = barrier;
  return barrier;
}

function addDeviceInvalidationFailure(failures, error) {
  if (error instanceof AggregateError && error.errors) {
    for (const cause of error.errors) addDeviceInvalidationFailure(failures, cause);
    return;
  }
  failures.add(error);
}

function exposeDeviceInvalidation(ownerDevice, cleanup, previousBarrier) {
  const operation = Promise.allSettled([
    previousBarrier || Promise.resolve(),
    cleanup,
  ]).then(([previous, current]) => {
    const failures = new Set();
    if (previous.status === "rejected") {
      addDeviceInvalidationFailure(failures, previous.reason);
    }
    if (current.status === "rejected") {
      addDeviceInvalidationFailure(failures, current.reason);
    }
    if (failures.size) {
      throw new AggregateError([...failures], "RIFE device invalidation failed");
    }
    return current.value;
  });
  deviceInvalidations.set(ownerDevice, operation);
  activeDeviceInvalidations.add(operation);
  operation.then(
    () => {
      activeDeviceInvalidations.delete(operation);
      refreshDeviceInvalidationBarrier();
    },
    (error) => {
      activeDeviceInvalidations.delete(operation);
      addDeviceInvalidationFailure(deviceInvalidationFailures, error);
      refreshDeviceInvalidationBarrier();
    },
  );
  refreshDeviceInvalidationBarrier();
  return operation;
}

function whenCpuRunsIdle() {
  if (activeCpuRuns === 0) return Promise.resolve();
  return new Promise((resolve) => cpuIdleResolvers.push(resolve));
}

function endCpuRun() {
  activeCpuRuns = Math.max(0, activeCpuRuns - 1);
  if (activeCpuRuns === 0 && cpuIdleResolvers.length) {
    for (const resolve of cpuIdleResolvers.splice(0)) resolve();
  }
}

// Adapter describing the model's I/O. The verified timestep-aware exports use
// seven channels: two concatenated RGB frames plus a plane filled with t. For the v4.25/
// 4.26 rife_v2 exports, channels 0-2 = frame A, 3-5 = frame B, 6 = timestep plane,
// and t drives interpolation position (t=0→frameA, 1→frameB, 0.5→midpoint).
const MODELS = {
  "rife_v4.26":      { file: "model/rife_v4.26.onnx",      channels: 7, timestepPlane: true,  label: "4.26 (default; may wave on bright motion)",
                       dims: (w, h) => ({ batch: 1, height: h, width: w }) },
  "rife_v4.26_fp16": { file: "model/rife_v4.26_fp16.onnx", channels: 7, timestepPlane: true,  fp16: true, label: "4.26 FP16 (experimental)",
                       dims: (w, h) => ({ batch: 1, height: h, width: w }) },
};
let currentModelKey = "rife_v4.26"; // default: verified FP32 model with broad WebGPU support

const MODEL_IO = {
  url: () => chrome.runtime.getURL(MODELS[currentModelKey].file),
  channels: () => MODELS[currentModelKey].channels,
  timestepPlane: () => MODELS[currentModelKey].timestepPlane,
  inputName: null,
  outputName: null,
  layout: "nchw",
  concatFrames: true,
  input2Name: null,
  padTo: 8,
  normalize: 1.0 / 255.0,
  denormalize: 255.0,
};

export function setStaticPassthrough(on) { MODEL_IO.staticPassthrough = !!on; return MODEL_IO.staticPassthrough; }
export function listModels() {
  return Object.entries(MODELS).map(([k, v]) => ({ key: k, label: v.label, current: k === currentModelKey }));
}
export function setModel(key) {
  if (!MODELS[key]) return false;
  if (key === currentModelKey) return true;

  currentModelKey = key;
  modelGeneration++;
  modelLoadTried = false; modelAvailable = false;
  pinnedDims = null; captureActive = false; captureBroken = false; _skipOutDispose = false;
  MODEL_IO.inputName = null; MODEL_IO.outputName = null;
  _bufW = 0; _bufH = 0; // force buffer realloc (channel count may change)
  _tsFilled = null; usingWasmEp = false; fp16Active = false; lastError = null;

  // Stop exposing the old model's GPU pipeline immediately, but deliberately keep
  // its ORT session referenced. The next init creates and commits a replacement
  // session first, then releases the old session after this teardown and all CPU
  // users finish. This preserves ORT's device reference across model switches.
  const gpuDone = destroyGpuInterp();
  pendingGpuTeardown = pendingGpuTeardown.catch(() => {}).then(async () => {
    try { await gpuDone; } catch {}
  });
  return true;
}

// EXPERIMENT #2 — JSPI: the asyncify build instruments EVERY function for stack
// unwinding (24.3MB wasm vs 15.0MB jspi — the 9MB delta IS the instrumentation)
// and stack-copies on every GPU suspend point. JSPI uses the browser's native
// WebAssembly.Suspending instead. Same ORT 1.27.0, same WebGPU EP, same API;
// only the suspend mechanism differs. Falls back to asyncify automatically if the
// bundle fails to load OR the first session creation fails under it.
// EXPERIMENT #2 VERDICT: JSPI measured 20% SLOWER than asyncify at the clean
// comparator (480p@100%: mu7.8 vs mu6.5ms; 1080p directionally agreed). Thousands
// of SHALLOW suspends per inference: asyncify's memcpy beats JSPI's per-suspend
// promise/microtask round-trip on this workload. Concluded; jspi files removed.
const ENABLE_JSPI_EXPERIMENT = false;
let ortIsJspi = false;
export function usingJspi() { return ortIsJspi; }

async function loadORTImpl(forceAsyncify = false) {
  if (ort) return ort;
  const jspiSupported = typeof WebAssembly !== "undefined" && "Suspending" in WebAssembly;
  const tryOrder = (!forceAsyncify && ENABLE_JSPI_EXPERIMENT && jspiSupported)
    ? ["ort.jspi.min.mjs", "ort.webgpu.min.mjs"]
    : ["ort.webgpu.min.mjs"];
  let lastErr = null;
  for (const file of tryOrder) {
    try {
      const mod = await import(chrome.runtime.getURL("vendor/ort/" + file));
      if (!mod.InferenceSession) throw new Error("module missing InferenceSession");
      ort = { InferenceSession: mod.InferenceSession, Tensor: mod.Tensor, env: mod.env };
      ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/ort/");
      ort.env.wasm.numThreads = 1;
      // Session creation temporarily overrides this through createOrtSession().
      // Keep the shared process-wide baseline deterministic between creations.
      try { ort.env.webgpu.enableFp16 = false; } catch {}
      ortIsJspi = file.includes("jspi");
      console.log(`[RIFE] ORT loaded: ${file}${ortIsJspi ? " (JSPI suspend)" : " (asyncify suspend)"}`);
      return ort;
    } catch (e) { lastErr = e; console.warn(`[RIFE] ORT bundle ${file} failed to load:`, e.message); }
  }
  throw lastErr || new Error("no ORT bundle loaded");
}

async function loadORT(forceAsyncify = false) {
  if (ort) return ort;
  if (ortLoadPromise) return ortLoadPromise;
  const promise = loadORTImpl(forceAsyncify).finally(() => {
    if (ortLoadPromise === promise) ortLoadPromise = null;
  });
  ortLoadPromise = promise;
  return promise;
}

// v0.49.0: the neural upscaler engine shares this ORT instance — one env, one
// WebGPU device across the RIFE and neural sessions (device lifetime follows
// total session refcount; see fsrcnnx-neural.js init-before-release ordering).
export async function ensureOrt() { return loadORT(true); }

// ORT's WebGPU FP16 switch is process-global, not session-local. Every RIFE and
// neural session creation goes through this queue so one engine cannot change the
// flag while the other is compiling. The previous baseline is restored even when
// creation rejects, which also makes fallback attempts deterministic.
export function createOrtSession(url, options, { enableFp16 = false } = {}) {
  const operation = ortSessionCreateTail.catch(() => {}).then(async () => {
    const runtime = await loadORT(true);
    const webgpuEnv = runtime?.env?.webgpu;
    const previous = webgpuEnv ? webgpuEnv.enableFp16 : undefined;
    try {
      if (webgpuEnv) webgpuEnv.enableFp16 = !!enableFp16;
      const created = await runtime.InferenceSession.create(url, options);
      const createdDevice = runtime?.env?.webgpu?.device;
      const providers = options?.executionProviders || [];
      const usesWebgpu = providers.some((provider) => provider === "webgpu" || provider?.name === "webgpu");
      if (created && createdDevice && usesWebgpu) ortSessionDevices.set(created, createdDevice);
      return created;
    } finally {
      try { if (webgpuEnv) webgpuEnv.enableFp16 = previous === undefined ? false : previous; } catch {}
    }
  });
  ortSessionCreateTail = operation.then(() => undefined, () => undefined);
  return operation;
}

export function getOrtSessionDevice(createdSession) {
  return createdSession ? (ortSessionDevices.get(createdSession) || null) : null;
}

function releaseSessionGuards(guards) {
  if (!guards.length) return deferredGuardReleaseTail;
  const previous = deferredGuardReleaseTail;
  deferredGuardReleaseTail = (async () => {
    const failures = [];
    try { await previous; } catch (error) { failures.push(error); }
    for (const guard of guards) {
      try { await guard.session?.release?.(); }
      catch (error) { failures.push(error); }
    }
    if (failures.length) throw new AggregateError(failures, "RIFE session-guard release failed");
  })();
  return deferredGuardReleaseTail;
}

// A replacement session can be backed by a different GPUDevice. In that case,
// keep the old session as a device-lifetime guard until the upscaler confirms it
// has rebuilt on the committed device. Without this handshake, releasing the old
// session can invalidate an adopted renderer between initRife() and chain.adopt().
export function confirmOrtDeviceAdopted(adoptedDevice) {
  const committedDevice = getOrtSessionDevice(session);
  if (!adoptedDevice || adoptedDevice !== committedDevice) return Promise.resolve(false);
  const guards = deferredDeviceGuards;
  deferredDeviceGuards = [];
  return releaseSessionGuards(guards).then(() => true);
}

export let lastError = null; // surfaced to UI so failures aren't silent

// Graph capture remains disabled after its v0.45 evaluation. These flags keep the
// dormant experiment isolated without describing it as an active runtime feature.
let pinnedDims = null;      // {w,h} of the pinned session, null = dynamic
let captureActive = false;
let captureBroken = false;  // capture attempt failed once — don't retry this run
let _skipOutDispose = false; // set when a graph-capture session owns its output buffer
let usingWasmEp = false;    // wasm fallback EP: capture not applicable
export function graphCaptureActive() { return captureActive; }
export async function initRife(pinW = 0, pinH = 0) {
  if (runtimeDisposalPromise) {
    await runtimeDisposalPromise;
    return initRife(pinW, pinH);
  }
  // A loss invalidation retains sole ownership of the unpublished session until
  // all active work and release() finish. Allocate the recovery generation only
  // after that physical cleanup so a new ORT session cannot overlap it.
  if (deviceInvalidationPromise) {
    await deviceInvalidationPromise;
    return initRife(pinW, pinH);
  }
  const generation = modelGeneration;
  if (initPromise) {
    if (initPromiseGeneration === generation) return initPromise;
    try { await initPromise; } catch {}
    return initRife(pinW, pinH);
  }
  initPromiseGeneration = generation;
  const promise = initRifeGeneration(pinW, pinH, generation, currentModelKey)
    .finally(() => {
      if (initPromise === promise) {
        initPromise = null;
        initPromiseGeneration = -1;
      }
    });
  initPromise = promise;
  return promise;
}

async function initRifeGeneration(pinW, pinH, generation, modelKey) {
  const model = MODELS[modelKey];
  const modelUrl = chrome.runtime.getURL(model.file);
  const isCurrent = () => generation === modelGeneration && modelKey === currentModelKey;
  const assertCurrent = () => {
    if (!isCurrent()) {
      const error = new Error("model selection changed during initialization");
      error.staleModelInit = true;
      throw error;
    }
  };

  // Dynamic sessions can be reused across restarts. Dimensions matter only for the
  // dormant graph-capture experiment, where a captured graph is shape-specific.
  const pinOk = sessionModelKey === modelKey
    && (!captureActive || (pinW && pinH && pinnedDims?.w === pinW && pinnedDims?.h === pinH));
  if (modelLoadTried && session && pinOk) return modelAvailable;

  // Keep the committed session intact while constructing a replacement locally.
  // New inference is gated, while already-running GPU/CPU work retains immutable
  // snapshots and is drained at the eventual handoff.
  modelAvailable = false;
  modelLoadTried = true;
  let candidateSession = null;
  let nextPinnedDims = null;
  let nextCaptureActive = false;
  let nextCaptureBroken = captureBroken;
  let nextSkipOutputDispose = false;
  let nextUsingWasmEp = false;
  let nextFp16Active = false;
  let nextLastError = null;
  let stage = "start";
  try {
    stage = "load-ort";
    await loadORT();
    assertCurrent();
    stage = "fetch-model";
    const res = await fetch(modelUrl, { method: "HEAD" });
    assertCurrent();
    if (!res.ok) { lastError = `model HTTP ${res.status}`; modelAvailable = false; return false; }
    stage = "create-session-webgpu";
    // Experiment: ask the WebGPU EP to run in fp16 where it can. Support for this
    // varies by ORT build — if the build ignores or rejects it, we fall back to a
    // plain fp32 session (no risk). We attempt fp16 first, then plain, then wasm.
    // fp16 is DISABLED by default: it gave no measurable speedup on tested
    // hardware AND is the likely cause of wave/banding artifacts on bright regions
    // (fp16 has coarse precision near 1.0, exactly where the artifacts appear, and
    // it's the one thing distinguishing this pipeline from full-precision showcases
    // that don't show the waves). Opt back in via MODEL_IO.tryFp16 = true.
    const wantFp16 = MODEL_IO.tryFp16 === true;
    const buildOpts = (fp16) => {
      const o = { executionProviders: [], graphOptimizationLevel: "all" };
      o.executionProviders = [{ name: "webgpu" }];
      // Keep model OUTPUT on the GPU (preferredOutputLocation is a session-creation
      // option in ORT web — it throws if passed to run()). The GPU-resident path
      // consumes output.gpuBuffer directly (no readback). The CPU fallback path
      // downloads it via await outTensor.getData(). This one setting serves both.
      o.preferredOutputLocation = "gpu-buffer";
      if (fp16) {
        o.optimizedModelFilePath = undefined;
        o.enableGraphCapture = false;
      }
      return o;
    };
    try {
      if (wantFp16) {
        try {
          candidateSession = await createOrtSession(modelUrl, buildOpts(true), { enableFp16: true });
          assertCurrent();
          nextFp16Active = true;
          console.log("[RIFE] session created (fp16 execution requested)");
        } catch (f16err) {
          if (f16err.staleModelInit || !isCurrent()) throw f16err;
          console.warn("[RIFE] fp16 attempt failed, falling back to fp32:", f16err.message);
          nextFp16Active = false;
          candidateSession = await createOrtSession(modelUrl, buildOpts(false), { enableFp16: false });
          assertCurrent();
        }
      } else {
        // EXPERIMENT #1 VERDICT (v0.45.0-0.45.2): graph capture is NOT VIABLE on
        // this stack (ORT 1.27 asyncify WebGPU EP + external gpu-buffer I/O on a
        // shared device). Measured NO speedup with honest timing (1080p 27.7 vs
        // 25.4ms baseline; 480p 11.7 vs 10.7) and produced three correctness
        // failures (cross-device on recreation; submit-only pacing collapse;
        // destroyed-buffer replays → embossed edges + black flashing). Code kept
        // for a future ORT that fixes the buffer-lifetime semantics.
        const ENABLE_GRAPH_CAPTURE_EXPERIMENT = false;
        const wantCapture = ENABLE_GRAPH_CAPTURE_EXPERIMENT && pinW && pinH && !captureBroken && model.dims;
        if (wantCapture) {
          try {
            const o = buildOpts(false);
            o.freeDimensionOverrides = model.dims(pinW, pinH);
            o.enableGraphCapture = true;
            candidateSession = await createOrtSession(modelUrl, o, { enableFp16: false });
            assertCurrent();
            nextPinnedDims = { w: pinW, h: pinH };
            nextCaptureActive = true;
            nextSkipOutputDispose = true;
            console.log(`[RIFE] session created PINNED ${pinW}x${pinH} + graph capture`);
          } catch (gcErr) {
            if (gcErr.staleModelInit || !isCurrent()) throw gcErr;
            console.warn("[RIFE] graph-capture creation failed — plain session:", gcErr.message);
            nextCaptureBroken = true;
            candidateSession = await createOrtSession(modelUrl, buildOpts(false), { enableFp16: false });
            assertCurrent();
          }
        } else {
          candidateSession = await createOrtSession(modelUrl, buildOpts(false), { enableFp16: false });
          assertCurrent();
        }
      }
    } catch (epErr) {
      if (epErr.staleModelInit || !isCurrent()) throw epErr;
      // JSPI runtime failure (bundle loaded but sessions won't create/run under
      // it) → reload the proven asyncify build and retry ONCE before any other
      // fallback. Load-time reload is safe: nothing is bound to a device yet.
      if (ortIsJspi) {
        console.warn("[RIFE] session failed under JSPI — reloading asyncify build:", epErr.message);
        ort = null; ortIsJspi = false;
        await loadORT(true);
        assertCurrent();
        try {
          candidateSession = await createOrtSession(modelUrl, buildOpts(false), { enableFp16: false });
          assertCurrent();
          console.log("[RIFE] session created on asyncify fallback");
        } catch (e2) {
          if (e2.staleModelInit || !isCurrent()) throw e2;
          console.warn("[RIFE] asyncify retry also failed:", e2.message);
        }
      }
      // WebGPU EP failed entirely — retry on wasm so we at least learn if it's an
      // EP issue vs a model/runtime issue. (wasm will be slow but proves the path.)
      if (!candidateSession) {
        console.warn("[RIFE] WebGPU EP failed, trying wasm:", epErr.message);
        stage = "create-session-wasm";
        nextFp16Active = false;
        candidateSession = await createOrtSession(modelUrl, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        }, { enableFp16: false });
        assertCurrent();
        nextLastError = "webgpu-unavailable-using-wasm";
        nextUsingWasmEp = true;
      }
    }

    if (!candidateSession) throw new Error("session creation returned no session");
    const nextInputName = candidateSession.inputNames[0];
    const nextOutputName = candidateSession.outputNames[0];
    assertCurrent();

    // Candidate creation establishes the replacement device/session reference.
    // Drain the old GPU and CPU users while the old committed session is still
    // retained, then atomically publish the candidate before releasing the old one.
    const oldSession = session;
    const oldInputTensor = _inTensor;
    const gpuDone = destroyGpuInterp();
    pendingGpuTeardown = pendingGpuTeardown.catch(() => {}).then(async () => {
      try { await gpuDone; } catch {}
    });
    await pendingGpuTeardown;
    await whenCpuRunsIdle();
    assertCurrent();

    session = candidateSession;
    candidateSession = null;
    const committedSession = session;
    const committedDevice = getOrtSessionDevice(committedSession);
    const oldDevice = getOrtSessionDevice(oldSession);
    let oldGuard = null;
    // Transfer the predecessor out of this initializer's local ownership before
    // yielding to the device-loss watcher. Disposal or an already-settled loss
    // can otherwise clear the replacement and make this continuation return
    // before the predecessor reaches a release path.
    if (oldSession && oldSession !== committedSession) {
      oldGuard = { session: oldSession, device: oldDevice };
      deferredDeviceGuards.push(oldGuard);
    }
    sessionModelKey = modelKey;
    watchOrtDevice(committedDevice);
    MODEL_IO.inputName = nextInputName;
    MODEL_IO.outputName = nextOutputName;
    pinnedDims = nextPinnedDims;
    captureActive = nextCaptureActive;
    captureBroken = nextCaptureBroken;
    _skipOutDispose = nextSkipOutputDispose;
    usingWasmEp = nextUsingWasmEp;
    fp16Active = nextFp16Active;
    lastError = nextLastError;
    modelAvailable = true;
    modelLoadTried = true;

    // A device can already be lost when ORT returns it. Let the watcher claim
    // the committed session, then make initRife() report the resulting state
    // truthfully instead of returning true for an already-unpublished session.
    await Promise.resolve();
    if (committedDevice && lostOrtDevices.has(committedDevice)) {
      await invalidateDevice(committedDevice, {
        reason: "unknown",
        message: "ORT device was already lost during RIFE initialization",
      });
      return false;
    }
    if (session !== committedSession) return false;

    try { oldInputTensor?.dispose?.(); } catch {}
    if (_inTensor === oldInputTensor) {
      _inTensor = null; _inBuf = null; _outImg = null;
      _ca = null; _cb = null; _cout = null; _ctxA = null; _ctxB = null; _octx = null;
    }
    _bufW = 0; _bufH = 0; _tsFilled = null; _tsPlaneW = 0; _tsPlaneH = 0;

    // The committed replacement now owns the runtime/device reference. Only now
    // may releasing a same-device old session decrement ORT's shared refcount. A
    // different-device session remains a guard until chain adoption is confirmed.
    const nowGuardedByCommittedSession = deferredDeviceGuards.filter(
      (guard) => (guard.device && guard.device === committedDevice) ||
        (guard === oldGuard && (!oldDevice || oldDevice === committedDevice)),
    );
    if (nowGuardedByCommittedSession.length) {
      deferredDeviceGuards = deferredDeviceGuards.filter(
        (guard) => !nowGuardedByCommittedSession.includes(guard),
      );
      await releaseSessionGuards(nowGuardedByCommittedSession);
    }
    if (!isCurrent()) return false;
    console.log(`[RIFE] ready (in=${MODEL_IO.inputName} out=${MODEL_IO.outputName})`);
    return true;
  } catch (e) {
    const failedCandidate = candidateSession;
    candidateSession = null;
    try { await failedCandidate?.release?.(); } catch {}
    if (e.staleModelInit || !isCurrent()) {
      return false;
    }
    lastError = `${stage}: ${e.message}`;
    console.error(`[RIFE] init FAILED at ${stage}:`, e.message, e);
    modelAvailable = false;
    return false;
  }
}
export function getLastError() { return lastError; }

// --- GPU-resident interpolation path (optional; falls back to CPU interpolate) ---
let gpuInterp = null;
let gpuInterpDescriptor = null;
let gpuFailedDescriptor = null;
let gpuInitPromise = null;
let gpuInitDescriptor = null;
let gpuLifecycleGeneration = 0;
let lastGpuDestroyPromise = Promise.resolve();
// Expose ORT's device so the GPU path can build buffers on the same device (required
// for Tensor.fromGpuBuffer). Available after session creation in ORT WebGPU builds.
export function getOrtDevice() {
  if (usingWasmEp || !session) return null;
  try { return getOrtSessionDevice(session) || (ort?.env?.webgpu?.device ?? null); } catch { return null; }
}
export function getOrt() { return ort; }

function gpuDescriptorsEqual(left, right) {
  if (!left || !right || left.kind !== right.kind) return false;
  if (left.kind === "rife") {
    return left.session === right.session
      && left.modelGeneration === right.modelGeneration
      && left.device === right.device;
  }
  return left.device === right.device;
}

function gpuDescriptorCurrent(descriptor, generation = gpuLifecycleGeneration) {
  if (!descriptor || descriptor.lifecycleGeneration !== generation ||
      generation !== gpuLifecycleGeneration) return false;
  if (descriptor.kind === "blend") return true;
  return session === descriptor.session
    && modelGeneration === descriptor.modelGeneration
    && getOrtSessionDevice(descriptor.session) === descriptor.device
    && isReady();
}

function activeGpuSatisfies(descriptor) {
  if (!gpuInterp?.ready || !descriptor) return false;
  if (descriptor.kind === "rife") {
    return gpuInterpDescriptor?.kind === "rife"
      && gpuInterpDescriptor.session === descriptor.session
      && gpuInterpDescriptor.modelGeneration === descriptor.modelGeneration
      && gpuInterp.device === descriptor.device
      && !!gpuInterp._rifeCapable;
  }
  // An ORT-capable helper also contains the blend pipeline. A device-less blend
  // request may reuse either kind; a chained request requires exact ownership.
  return !descriptor.device || gpuInterp.device === descriptor.device;
}

async function waitForRuntimeDisposal() {
  while (runtimeDisposalPromise) {
    const disposal = runtimeDisposalPromise;
    await disposal;
    if (runtimeDisposalPromise === disposal) break;
  }
}

async function initializeGpuCandidate(descriptor, generation, { log, warn } = {}) {
  let candidate = null;
  let published = false;
  const isCurrent = () => gpuDescriptorCurrent(descriptor, generation);
  const warning = warn || console.warn;
  try {
    if (!isCurrent()) return false;
    if (descriptor.kind === "rife" && !descriptor.device) {
      warning("[RIFE] no ORT device for GPU path");
      return false;
    }
    const mod = await import(chrome.runtime.getURL("fsrcnnx-rife-gpu.js"));
    if (!isCurrent()) return false;
    candidate = new mod.GpuInterp({
      log,
      warn,
      onDeviceLost: (lostDevice, info) => {
        const label = descriptor.kind === "rife" ? "device-loss" : "blend device-loss";
        invalidateDevice(lostDevice, info).catch((error) =>
          warning(`[RIFE] ${label} invalidation failed:`, error.message));
      },
    });
    const candidateOrt = descriptor.kind === "rife" ? descriptor.ort : null;
    if (!(await candidate.init(descriptor.device || null, candidateOrt))) return false;
    if (!candidate.ready || candidate._destroyRequested || candidate._deviceLost) return false;
    if (!isCurrent()) return false;

    // Only this one module-level operation is allowed to own an unpublished
    // candidate. A defensive identity check keeps an accidental future publisher
    // from being overwritten without retirement.
    if (gpuInterp) return activeGpuSatisfies(descriptor);
    gpuInterp = candidate;
    gpuInterpDescriptor = descriptor;
    candidate = null;
    published = true;
    if (descriptor.kind === "rife") {
      gpuInterp.skipOutputDispose = _skipOutDispose;
      (log || console.log)("[RIFE] GPU-resident path active");
    } else {
      (log || console.log)(`[RIFE] standalone blend GPU path active${descriptor.device ? " (shared device)" : ""}`);
    }
    return true;
  } catch (error) {
    if (isCurrent()) {
      const label = descriptor.kind === "rife" ? "GPU path" : "standalone blend";
      warning(`[RIFE] ${label} init failed:`, error.message);
    }
    return false;
  } finally {
    // Failed, superseded, and teardown-cancelled candidates are never published.
    // Their destruction is part of gpuInitPromise, so the retirement barrier can
    // safely prevent a subsequent generation from allocating alongside them.
    if (candidate && !published) {
      try { await candidate.destroy?.(); } catch {}
    }
  }
}

async function initGpuHelper(descriptor, options = {}) {
  for (;;) {
    await waitForRuntimeDisposal();

    const retirement = lastGpuDestroyPromise;
    try { await retirement; } catch {}
    if (retirement !== lastGpuDestroyPromise || runtimeDisposalPromise) continue;
    if (!gpuDescriptorCurrent(descriptor)) return false;
    if (activeGpuSatisfies(descriptor)) return true;

    // A live but incompatible helper is retired before constructing its
    // replacement. This covers standalone↔RIFE transitions and chain-device
    // changes without ever publishing helpers backed by two devices at once.
    if (gpuInterp) {
      const expectedGeneration = gpuLifecycleGeneration + 1;
      await destroyGpuInterp();
      if (gpuLifecycleGeneration !== expectedGeneration) return false;
      descriptor.lifecycleGeneration = expectedGeneration;
      continue;
    }

    const pending = gpuInitPromise;
    if (pending) {
      if (gpuDescriptorsEqual(gpuInitDescriptor, descriptor)) return pending;
      try { await pending; } catch {}
      continue;
    }
    if (gpuDescriptorsEqual(gpuFailedDescriptor, descriptor)) return false;

    const generation = gpuLifecycleGeneration;
    let operation;
    operation = initializeGpuCandidate(descriptor, generation, options)
      .then((ok) => {
        if (!ok && gpuDescriptorCurrent(descriptor, generation)) {
          gpuFailedDescriptor = descriptor;
        }
        return ok;
      })
      .finally(() => {
        if (gpuInitPromise === operation) {
          gpuInitPromise = null;
          gpuInitDescriptor = null;
        }
      });
    gpuInitDescriptor = descriptor;
    gpuInitPromise = operation;
    return operation;
  }
}

// Try to initialize the GPU-resident path. Returns true if active. Safe to call
// after initRife(). Fails → CPU interpolate() stays in use.
export async function initGpuInterp({ log, warn } = {}) {
  const lifecycleGeneration = gpuLifecycleGeneration;
  await waitForRuntimeDisposal();
  if (lifecycleGeneration !== gpuLifecycleGeneration) return false;
  const targetSession = session;
  const descriptor = {
    kind: "rife",
    lifecycleGeneration,
    session: targetSession,
    modelGeneration,
    device: getOrtSessionDevice(targetSession),
    ort,
  };
  if (!targetSession || !isReady()) return false;
  return initGpuHelper(descriptor, { log, warn });
}

// Standalone blend GPU path: NO RIFE model load. If `device` is provided (e.g. the
// upscaler's, for the upscale→interpolate chain) the pipeline builds on it so
// textures can be shared; otherwise it requests its own device.
export function initGpuBlendStandalone({ log, warn, device } = {}) {
  return initGpuHelper({
    kind: "blend",
    lifecycleGeneration: gpuLifecycleGeneration,
    device: device || null,
  }, { log, warn });
}
export function gpuCaptureTex(tex) { return gpuInterp ? gpuInterp.captureTexToPooled(tex, MODEL_IO.padTo, MODEL_IO.channels()) : null; }
export function gpuActive() { return !!gpuInterp?.ready; }
export function gpuRifeCapable() { return !!(gpuInterp?.ready && gpuInterp._rifeCapable); }

export function destroyGpuInterp() {
  gpuLifecycleGeneration++;
  const old = gpuInterp;
  gpuInterp = null;
  gpuInterpDescriptor = null;
  gpuFailedDescriptor = null;
  const pendingInit = gpuInitPromise;
  let current;
  try { current = old?.destroy?.() || Promise.resolve(); }
  catch { current = Promise.resolve(); }
  const previous = lastGpuDestroyPromise;
  // A pending candidate observes the generation change and destroys itself in
  // its finally block. Include that entire path in the physical retirement fence.
  const combined = Promise.allSettled([
    previous,
    Promise.resolve(current),
    Promise.resolve(pendingInit),
  ]).then(() => undefined);
  lastGpuDestroyPromise = combined;
  return combined;
}

// Full GPU-presentation API (no readback): configure a WebGPU canvas, capture to a
// pooled texture, interpolate to a pooled texture, present a texture, recycle.
export function gpuConfigureCanvas(canvas) { return gpuInterp ? gpuInterp.configureCanvas(canvas) : false; }
export function gpuCapture(video) { return gpuInterp ? gpuInterp.captureToPooled(video, MODEL_IO.padTo, MODEL_IO.channels()) : null; }
export function gpuLastCaptureError() { return gpuInterp?.lastCaptureError || null; }
export function gpuHasPrev() { return gpuInterp ? gpuInterp.hasPrev() : false; }
export function gpuAdvance() { if (gpuInterp) gpuInterp.advance(); }
export function gpuResetFrames() { if (gpuInterp) gpuInterp.resetFrames(); }
export async function gpuTween(w, h, t, useStatic) {
  if (!gpuInterp) return null;
  const t0 = performance.now();
  // setModel() clears the global names immediately, while an old GPU run may still
  // be awaiting session.run(). Give that run an immutable adapter snapshot.
  const runSession = session;
  const runInterp = gpuInterp;
  const runIO = { inputName: MODEL_IO.inputName, outputName: MODEL_IO.outputName };
  const tex = await runInterp.interpolateToPooledTex(runSession, runIO, w, h, t, useStatic);
  const s = runInterp.getSplit ? runInterp.getSplit() : null;
  if (s) { timing.pre = s.pack; timing.infer = s.run; timing.post = s.comp; }
  return tex;
}
export function gpuPresent(tex) { return gpuInterp ? gpuInterp.presentTexture(tex) : false; }
export function gpuRelease(tex) {
  const owner = tex?._gpuInterpOwner || gpuInterp;
  if (owner) owner.releaseTex(tex);
}
export function gpuRetain(tex) {
  const owner = tex?._gpuInterpOwner || gpuInterp;
  if (owner) owner.retainTex(tex);
}
// Pipelined RIFE tween on an EXPLICIT pooled pair — safe to run while the grab loop
// keeps capturing (the shared prev/cur ping-pong may advance mid-inference).
export async function gpuTweenPair(prevTex, curTex, t, useStatic, scale = 1) {
  if (!gpuInterp || !isReady()) return null;
  const runSession = session;
  const runInterp = gpuInterp;
  const runIO = { inputName: MODEL_IO.inputName, outputName: MODEL_IO.outputName };
  const tex = await runInterp.interpolateToPooledTex(runSession, runIO, 0, 0, t, useStatic, prevTex, curTex, scale);
  const s = runInterp.getSplit ? runInterp.getSplit() : null;
  if (s) { timing.pre = s.pack; timing.infer = s.run; timing.post = s.comp + s.read; }
  return tex;
}
// cheap non-AI blend tween (fallback when RIFE can't keep up)
// Ladder blend between two explicit textures (real↔tween adjacency): the blend
// spans only a FRACTION of the source gap, so ghosting shrinks proportionally.
export function gpuBlendPair(texA, texB, t) {
  return gpuInterp ? gpuInterp.blendToPooledTex(t, texA, texB) : null;
}
export function gpuBlend(t) {
  if (!gpuInterp) return null;
  const tex = gpuInterp.blendToPooledTex(t);
  const s = gpuInterp.getSplit ? gpuInterp.getSplit() : null;
  if (s) { timing.pre = 0; timing.infer = 0; timing.post = s.comp; }
  return tex;
}

export function isReady() { return modelAvailable && !!session; }
// Multi-tween RIFE needs a timestep-aware model (7ch): arbitrary t values. The
// 6ch "original" export has t baked to 0.5 — midpoint only.
export function timestepAware() { return !!MODEL_IO.timestepPlane(); }

// Per-tween timing breakdown (ms). On the GPU path: pre = pack submission,
// infer = session.run/backpressure, post = composite submission. The CPU fallback
// includes its upload/download and canvas conversion work in pre/post respectively.
const timing = { pre: 0, infer: 0, post: 0 };
export function getTiming() { return { ...timing }; }
let fp16Active = false;
export function isFp16() { return fp16Active || !!(MODELS[currentModelKey] && MODELS[currentModelKey].fp16); }

// Fill a preallocated planar slice (3 channels, NCHW) from a canvas context's
// pixels, normalized. Writes into `dst` at channel-plane offset `base`.
function fillPlanar(srcCanvasCtx, padW, padH, dst, base, normalize = MODEL_IO.normalize) {
  const { data } = srcCanvasCtx.getImageData(0, 0, padW, padH); // RGBA
  const plane = padW * padH;
  const n = normalize;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    dst[base + p] = data[i] * n;
    dst[base + plane + p] = data[i + 1] * n;
    dst[base + 2 * plane + p] = data[i + 2] * n;
  }
}

// Interpolate the middle frame between two CanvasImageSource frames. `scale`
// runs inference at a fraction of display resolution. Buffers (canvases, contexts,
// the input Float32Array + ORT tensor, and the output ImageData) are preallocated
// and reused across frames — only reallocated when the padded size changes (e.g.
// when Auto shifts resolution) — to avoid per-frame GC churn that causes timing jitter.
let _ca = null, _cb = null, _cout = null, _ctxA = null, _ctxB = null, _octx = null;
let _inBuf = null, _inTensor = null, _outImg = null, _bufW = 0, _bufH = 0;
let _tsFilled = null, _tsPlaneW = 0, _tsPlaneH = 0; // timestep-plane fill cache

// Release the committed model session and every GPU/CPU cache. Logical state is
// cleared synchronously; physical release waits for stale initialization,
// inference, GpuInterp work, and session guards. The selected model key and the
// loaded ORT module remain reusable for a later explicit start.
export function disposeRife() {
  if (runtimeDisposalPromise) return runtimeDisposalPromise;

  modelGeneration++;
  const committedSession = session;
  const initialGuards = deferredDeviceGuards;
  const pendingInit = initPromise;
  const pendingDeviceInvalidation = deviceInvalidationPromise;
  deferredDeviceGuards = [];
  session = null;
  sessionModelKey = null;
  modelLoadTried = false;
  modelAvailable = false;
  pinnedDims = null;
  captureActive = false;
  captureBroken = false;
  _skipOutDispose = false;
  usingWasmEp = false;
  fp16Active = false;
  MODEL_IO.inputName = null;
  MODEL_IO.outputName = null;
  lastError = null;

  const gpuDone = destroyGpuInterp();
  const operation = (async () => {
    const failures = new Set();
    // invalidateDevice() may already own the only reference responsible for
    // releasing the lost session. Disposal is incomplete until that release is
    // done, even though the public session state is already empty.
    try { await pendingDeviceInvalidation; }
    catch (error) { failures.add(error); }
    while (activeDeviceInvalidations.size) {
      try { await refreshDeviceInvalidationBarrier(); }
      catch (error) { failures.add(error); }
    }
    deviceInvalidationFailures.clear();
    refreshDeviceInvalidationBarrier();
    try { await pendingInit; } catch {}
    try { await pendingGpuTeardown; } catch {}
    try { await gpuDone; } catch {}
    await whenCpuRunsIdle();
    try { await cpuInterpolateTail; } catch {}

    const oldInputTensor = _inTensor;
    _inTensor = null; _inBuf = null; _outImg = null;
    _ca = null; _cb = null; _cout = null; _ctxA = null; _ctxB = null; _octx = null;
    _bufW = 0; _bufH = 0; _tsFilled = null; _tsPlaneW = 0; _tsPlaneH = 0;
    try { oldInputTensor?.dispose?.(); } catch {}

    // A replacement init that was already past publication can append a device
    // guard before observing the generation change. Drain both snapshots.
    const guards = [...initialGuards, ...deferredDeviceGuards];
    deferredDeviceGuards = [];
    // releaseSessionGuards([]) returns the existing serialized tail. Await it
    // unconditionally: confirmOrtDeviceAdopted() may already have claimed the
    // only guard while its asynchronous session.release() is still pending.
    try { await releaseSessionGuards(guards); }
    catch (error) { failures.add(error); }
    try { await committedSession?.release?.(); }
    catch (error) { failures.add(error); }
    if (failures.size) {
      throw new AggregateError([...failures], "RIFE session disposal failed");
    }
  })().finally(() => {
    if (runtimeDisposalPromise === operation) runtimeDisposalPromise = null;
  });
  runtimeDisposalPromise = operation;
  return operation;
}

// Invalidate every RIFE object backed by a lost device before any restart can
// mistake the still-referenced session for a healthy reusable one. Calls are
// serialized because the main renderer and GpuInterp can observe the same shared
// loss independently. Identity checks make the second notification a no-op.
export function invalidateDevice(lostDevice, info = null) {
  if (!lostDevice) return Promise.resolve(false);
  const existing = deviceInvalidations.get(lostDevice);
  if (existing) return existing;
  const previousBarrier = deviceInvalidationPromise;

  const committedSession = session;
  const committedDevice = getOrtSessionDevice(committedSession);
  const activeGpu = gpuInterp;
  const affectsSession = !!committedSession && committedDevice === lostDevice;
  const affectsGpu = !!activeGpu && activeGpu.device === lostDevice;
  const affectsPendingGpu = !!gpuInitPromise && gpuInitDescriptor?.device === lostDevice;
  let gpuDone = Promise.resolve();
  const lostGuards = deferredDeviceGuards.filter((guard) => guard.device === lostDevice);
  if (lostGuards.length) {
    // Claim old cross-device guards even when the currently committed session is
    // healthy on another device. Main can lose the device it has not yet handed
    // off while RIFE has already published the replacement session.
    deferredDeviceGuards = deferredDeviceGuards.filter((guard) => guard.device !== lostDevice);
  }

  // Claim and unpublish affected state synchronously. In particular, a loss
  // listener may request recovery immediately; by the time it runs, isReady()
  // and getOrtDevice() must no longer expose the dead session.
  if (affectsSession) {
    modelGeneration++;
    session = null;
    sessionModelKey = null;
    modelLoadTried = false;
    modelAvailable = false;
    pinnedDims = null;
    captureActive = false;
    captureBroken = false;
    _skipOutDispose = false;
    usingWasmEp = false;
    fp16Active = false;
    MODEL_IO.inputName = null;
    MODEL_IO.outputName = null;
    lastError = `device-lost: ${info?.message || info?.reason || "unknown reason"}`;

  }

  if (affectsGpu || affectsPendingGpu) {
    const teardown = destroyGpuInterp();
    pendingGpuTeardown = pendingGpuTeardown.catch(() => {}).then(async () => {
      try { await teardown; } catch {}
    });
    gpuDone = pendingGpuTeardown;
  }

  const claim = (affectsSession || affectsGpu || affectsPendingGpu || lostGuards.length)
    ? { committedSession, affectsSession, gpuDone, lostGuards }
    : null;
  const operation = deviceInvalidationTail.catch(() => {}).then(
    () => finishDeviceInvalidation(claim),
  );
  deviceInvalidationTail = operation.then(() => undefined, () => undefined);
  // Device loss is terminal. Every independent observer must receive the exact
  // claiming cleanup promise rather than a later identity no-op that could mask
  // a session-release failure from the renderer's recovery barrier.
  const exposed = exposeDeviceInvalidation(lostDevice, operation, previousBarrier);

  // Publish the cleanup barrier before callbacks can synchronously re-enter
  // initRife() or disposeRife(). Duplicate notifications share this exact result.
  if (affectsSession || affectsGpu || affectsPendingGpu) emitDeviceLoss(lostDevice, info);
  return exposed;
}

async function finishDeviceInvalidation(claim) {
  if (!claim) return false;
  await claim.gpuDone;
  // Even an empty list returns the existing serialized guard-release tail.
  // Current-session invalidation must not resolve ahead of a guard already
  // claimed by confirmOrtDeviceAdopted().
  const failures = [];
  try { await releaseSessionGuards(claim.lostGuards); }
  catch (error) { failures.push(error); }
  if (!claim.affectsSession) {
    if (failures.length) throw new AggregateError(failures, "RIFE device-loss cleanup failed");
    return true;
  }
  await whenCpuRunsIdle();

  const oldInputTensor = _inTensor;
  _inTensor = null; _inBuf = null; _outImg = null;
  _ca = null; _cb = null; _cout = null; _ctxA = null; _ctxB = null; _octx = null;
  _bufW = 0; _bufH = 0; _tsFilled = null; _tsPlaneW = 0; _tsPlaneH = 0;
  try { oldInputTensor?.dispose?.(); } catch {}

  try { await claim.committedSession.release?.(); }
  catch (error) { failures.push(error); }
  if (failures.length) throw new AggregateError(failures, "RIFE device-loss cleanup failed");
  return true;
}

export async function interpolate(frameA, frameB, w, h, t = 0.5, scale = 1.0) {
  // These canvases and typed arrays are module-global reuse buffers. A stopped
  // Interpolator can be restarted before its previous session.run() settles, so
  // serialize at this exported boundary rather than relying on one lifecycle's
  // local cpuTickBusy flag. Normalize both predecessor and operation failures so
  // one rejected call can never poison the queue for later lifecycles.
  const generation = modelGeneration;
  const operation = cpuInterpolateTail.catch(() => {}).then(
    () => interpolateSerial(frameA, frameB, w, h, t, scale, generation),
  );
  cpuInterpolateTail = operation.then(() => undefined, () => undefined);
  return operation;
}

async function interpolateSerial(frameA, frameB, w, h, t, scale, generation) {
  if (generation !== modelGeneration) return null;
  const runSession = session;
  if (!modelAvailable || !runSession) return null;
  // Model selection may change while session.run() is awaited. Capture the full
  // adapter contract for this run so its continuation never observes the next
  // model's cleared names, channel count, normalization, or disposal policy.
  const runIO = {
    inputName: MODEL_IO.inputName,
    outputName: MODEL_IO.outputName,
    channels: MODEL_IO.channels(),
    timestepPlane: MODEL_IO.timestepPlane(),
    padTo: MODEL_IO.padTo,
    normalize: MODEL_IO.normalize,
    denormalize: MODEL_IO.denormalize,
    staticPassthrough: MODEL_IO.staticPassthrough !== false,
    skipOutputDispose: _skipOutDispose,
  };
  activeCpuRuns++;
  let outTensor = null;
  try {
    const iw = Math.max(runIO.padTo, Math.round(w * scale));
    const ih = Math.max(runIO.padTo, Math.round(h * scale));
    const padW = Math.ceil(iw / runIO.padTo) * runIO.padTo;
    const padH = Math.ceil(ih / runIO.padTo) * runIO.padTo;
    // (re)allocate reusable buffers only when the padded size changes
    if (_bufW !== padW || _bufH !== padH) {
      _ca = new OffscreenCanvas(padW, padH); _cb = new OffscreenCanvas(padW, padH);
      _cout = new OffscreenCanvas(padW, padH);
      _ctxA = _ca.getContext("2d", { colorSpace: SRGB_COLOR_SPACE, willReadFrequently: true });
      _ctxB = _cb.getContext("2d", { colorSpace: SRGB_COLOR_SPACE, willReadFrequently: true });
      _octx = _cout.getContext("2d", { colorSpace: SRGB_COLOR_SPACE });
      const plane = padW * padH;
      const ch = runIO.channels; // 6 (frames only) or 7 (+ timestep plane)
      _inBuf = new Float32Array(ch * plane);
      _inTensor = new ort.Tensor("float32", _inBuf, [1, ch, padH, padW]);
      _outImg = new ImageData(padW, padH);
      _bufW = padW; _bufH = padH;
    }
    const tPre0 = performance.now();
    _ctxA.clearRect(0, 0, padW, padH); _ctxA.drawImage(frameA, 0, 0, iw, ih);
    _ctxB.clearRect(0, 0, padW, padH); _ctxB.drawImage(frameB, 0, 0, iw, ih);

    const plane = padW * padH;
    const feeds = {};
    // fill both frames' planes directly into the reused input buffer
    fillPlanar(_ctxA, padW, padH, _inBuf, 0, runIO.normalize);
    fillPlanar(_ctxB, padW, padH, _inBuf, 3 * plane, runIO.normalize);
    if (runIO.timestepPlane) {
      // 7th channel: a plane filled with the timestep t (verified: channel 6
      // drives interpolation position). Fill only when t changed to save work —
      // but t is constant (0.5) here, so fill once per (re)alloc via _tsFilled.
      const base = 6 * plane;
      if (_tsFilled !== t || _tsPlaneW !== padW || _tsPlaneH !== padH) {
        _inBuf.fill(t, base, base + plane);
        _tsFilled = t; _tsPlaneW = padW; _tsPlaneH = padH;
      }
    }
    feeds[runIO.inputName] = _inTensor;
    timing.pre = performance.now() - tPre0;

    const tInf0 = performance.now();
    const result = await runSession.run(feeds);
    timing.infer = performance.now() - tInf0;
    outTensor = result[runIO.outputName];
    const tPost0 = performance.now();
    // output is on the GPU (preferredOutputLocation:'gpu-buffer'); download it for
    // the CPU passthrough+present path. Accessing .data on a GPU tensor THROWS, so
    // read via getData() (async download) guarded by location.
    let od;
    if (outTensor.location && outTensor.location !== "cpu") {
      od = await outTensor.getData();
    } else {
      od = outTensor.data;
    }
    const dn = runIO.denormalize;
    const img = _outImg;
    // STATIC-REGION PASSTHROUGH: the jitter on static detail comes from RIFE
    // reconstructing (downscale→infer→upscale) still content slightly differently
    // than the pixel-exact real frames it alternates with. Fix: per pixel, measure
    // how much frame A and frame B differ (their motion); where they're nearly
    // identical (static), output the REAL pixel (A/B average) instead of RIFE's
    // reconstruction, so static regions are pixel-stable. Where they differ
    // (motion), use RIFE. Smooth ramp between the two by difference magnitude.
    // A,B live in _inBuf (normalized): A at [0..plane) per channel, B at [3plane..).
    const useStatic = runIO.staticPassthrough;
    const aR = 0, aG = plane, aB = 2 * plane;
    const bR = 3 * plane, bG = 4 * plane, bB = 5 * plane;
    const T_LO = 0.012, T_HI = 0.05; // diff below LO = fully static; above HI = full RIFE
    for (let p = 0, i = 0; p < plane; p++, i += 4) {
      let r = od[p] * dn, g = od[plane + p] * dn, b = od[2 * plane + p] * dn;
      if (useStatic) {
        // per-pixel frame-to-frame difference (0..1 scale)
        const dr = Math.abs(_inBuf[aR + p] - _inBuf[bR + p]);
        const dg = Math.abs(_inBuf[aG + p] - _inBuf[bG + p]);
        const db = Math.abs(_inBuf[aB + p] - _inBuf[bB + p]);
        const d = (dr + dg + db) / 3;
        if (d < T_HI) {
          // real-pixel value = midpoint of A and B (true static content)
          const rr = (_inBuf[aR + p] + _inBuf[bR + p]) * 0.5 * dn;
          const rg = (_inBuf[aG + p] + _inBuf[bG + p]) * 0.5 * dn;
          const rb = (_inBuf[aB + p] + _inBuf[bB + p]) * 0.5 * dn;
          // weight: 0 (all real) when d<=T_LO, 1 (all RIFE) when d>=T_HI
          const wRife = d <= T_LO ? 0 : (d - T_LO) / (T_HI - T_LO);
          const wReal = 1 - wRife;
          r = r * wRife + rr * wReal;
          g = g * wRife + rg * wReal;
          b = b * wRife + rb * wReal;
        }
      }
      img.data[i] = r < 0 ? 0 : r > 255 ? 255 : r;
      img.data[i + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
      img.data[i + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
      img.data[i + 3] = 255;
    }
    _octx.putImageData(img, 0, 0);
    timing.post = performance.now() - tPost0;
    _cout._cropW = iw; _cout._cropH = ih;
    return _cout;
  } catch (e) {
    console.warn("[RIFE] inference failed:", e.message);
    return null;
  } finally {
    // Dynamic sessions return a fresh output tensor. The downloaded data has been
    // fully consumed above, so release its GPU/CPU backing deterministically.
    try { if (outTensor && !runIO.skipOutputDispose) outTensor.dispose?.(); } catch {}
    endCpuRun();
  }
}

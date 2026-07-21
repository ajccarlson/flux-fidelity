// fsrcnnx-rife.js — RIFE frame interpolation via ONNX Runtime Web (WebGPU EP).
// Loads the bundled ORT runtime + a RIFE .onnx model and exposes
// interpolate(frameA, frameB, t) -> ImageBitmap-able result for the tween.
//
// IMPORTANT: the RIFE .onnx model file must be present at model/rife.onnx. It is
// NOT bundled (model files aren't available in the build environment). If absent,
// the caller falls back to the blend placeholder. Different RIFE exports have
// different tensor signatures, so the I/O is described by an adapter (MODEL_IO)
// that can be adjusted to match the specific model without touching the pipeline.
//
// Default adapter targets a common RIFE v4.x export:
//   input: single tensor, two frames concatenated on channels -> [1, 6, H, W],
//          RGB, normalized 0..1, NCHW. (Some exports also take a timestep tensor;
//          set MODEL_IO.timestep to its input name if so.)
//   output: [1, 3, H, W] RGB 0..1 = the interpolated middle frame.
// Padding: RIFE needs H,W divisible by a factor (usually 32). We pad to the next
// multiple and crop the result back.

let ort = null;             // the ORT module
let session = null;         // InferenceSession
let modelLoadTried = false;
let modelAvailable = false;

// Adapter describing the model's I/O. Supports both older 6-channel exports (two
// RGB frames concatenated) and newer timestep-aware exports (7 channels: two RGB
// frames + a timestep plane filled with t). Verified empirically: for the v4.25/
// 4.26 rife_v2 exports, channels 0-2 = frame A, 3-5 = frame B, 6 = timestep plane,
// and t drives interpolation position (t=0→frameA, 1→frameB, 0.5→midpoint).
const MODELS = {
  "rife_v4.26":      { file: "model/rife_v4.26.onnx",      channels: 7, timestepPlane: true,  label: "4.26 (smoother; waves on bright motion)",
                       dims: (w, h) => ({ batch: 1, height: h, width: w }) },
  "rife_v4.26_fp16": { file: "model/rife_v4.26_fp16.onnx", channels: 7, timestepPlane: true,  fp16: true, label: "4.26 FP16 (faster; near-identical output)",
                       dims: (w, h) => ({ batch: 1, height: h, width: w }) },
  "rife_orig":       { file: "model/rife.onnx",            channels: 6, timestepPlane: false, label: "original (no bright-wave artifacts)",
                       dims: (w, h) => ({ dynamic_dim_0: 1, dynamic_dim_1: 6, dynamic_dim_2: h, dynamic_dim_3: w }) },
};
let currentModelKey = "rife_v4.26_fp16"; // default: fp16 (validated near-identical; ~half the inference cost)

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
  if (MODELS[key]) {
    currentModelKey = key;
    modelLoadTried = false; modelAvailable = false; session = null;
    pinnedDims = null; captureActive = false; captureBroken = false; _skipOutDispose = false;
    if (gpuInterp) gpuInterp.skipOutputDispose = false;
    MODEL_IO.inputName = null; MODEL_IO.outputName = null;
    _bufW = 0; _bufH = 0; // force buffer realloc (channel count may change)
    return true;
  }
  return false;
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

async function loadORT(forceAsyncify = false) {
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
      ortIsJspi = file.includes("jspi");
      console.log(`[RIFE] ORT loaded: ${file}${ortIsJspi ? " (JSPI suspend)" : " (asyncify suspend)"}`);
      return ort;
    } catch (e) { lastErr = e; console.warn(`[RIFE] ORT bundle ${file} failed to load:`, e.message); }
  }
  throw lastErr || new Error("no ORT bundle loaded");
}

// v0.49.0: the neural upscaler engine shares this ORT instance — one env, one
// WebGPU device across the RIFE and neural sessions (device lifetime follows
// total session refcount; see fsrcnnx-neural.js init-before-release ordering).
export async function ensureOrt() { return loadORT(true); }

export let lastError = null; // surfaced to UI so failures aren't silent

// GRAPH CAPTURE state: the initial session is created with dynamic dims (startup +
// CPU path unchanged). The first GPU tween learns the real padded dims and REPINS
// the session (freeDimensionOverrides) with enableGraphCapture — the WebGPU EP then
// records the ~180-op dispatch sequence once and replays it per run, cutting per-op
// JS dispatch overhead. Resolution change repins; any failure falls back to a plain
// dynamic session permanently (captureBroken) so the pipeline can't be left broken.
let pinnedDims = null;      // {w,h} of the pinned session, null = dynamic
let captureActive = false;
let captureBroken = false;  // capture attempt failed once — don't retry this run
let _skipOutDispose = false; // set when a graph-capture session owns its output buffer
let usingWasmEp = false;    // wasm fallback EP: capture not applicable
export function graphCaptureActive() { return captureActive; }
export async function initRife(pinW = 0, pinH = 0) {
  // Pin-at-creation: dims are known BEFORE the session exists (chain: upscaler
  // target dims; interp-only: video dims), so the FIRST session is created pinned
  // with graph capture, and every consumer then binds to ITS device. Mid-flight
  // session recreation is forbidden — it stood up a second WebGPU device and
  // cross-device'd every buffer (the v0.45.0 black-screen/freeze).
  const pinOk = pinW && pinH
    ? (pinnedDims && pinnedDims.w === pinW && pinnedDims.h === pinH)
    : !pinnedDims;
  if (modelLoadTried && session && pinOk) return modelAvailable;
  // (re)creating here is safe: init time, nothing from THIS run is bound yet, and
  // chain adoption always re-binds the upscaler to ORT's current device.
  try { session && session.release && session.release(); } catch {}
  session = null; pinnedDims = null; captureActive = false; _skipOutDispose = false;
  modelLoadTried = true;
  let stage = "start";
  try {
    stage = "load-ort";
    await loadORT();
    stage = "fetch-model";
    const res = await fetch(MODEL_IO.url(), { method: "HEAD" });
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
    // explicitly clear the env flag so it can't persist from a prior session
    try { ort.env.webgpu.enableFp16 = wantFp16; } catch {}
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
        try { ort.env.webgpu.enableFp16 = true; } catch {}
      }
      return o;
    };
    try {
      if (wantFp16) {
        try {
          session = await ort.InferenceSession.create(MODEL_IO.url(), buildOpts(true));
          fp16Active = !!(ort.env.webgpu && ort.env.webgpu.enableFp16);
          console.log(`[RIFE] session created (fp16 requested=${fp16Active})`);
        } catch (f16err) {
          console.warn("[RIFE] fp16 attempt failed, falling back to fp32:", f16err.message);
          try { ort.env.webgpu.enableFp16 = false; } catch {}
          fp16Active = false;
          session = await ort.InferenceSession.create(MODEL_IO.url(), buildOpts(false));
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
        const wantCapture = ENABLE_GRAPH_CAPTURE_EXPERIMENT && pinW && pinH && !captureBroken && MODELS[currentModelKey].dims;
        if (wantCapture) {
          try {
            const o = buildOpts(false);
            o.freeDimensionOverrides = MODELS[currentModelKey].dims(pinW, pinH);
            o.enableGraphCapture = true;
            session = await ort.InferenceSession.create(MODEL_IO.url(), o);
            pinnedDims = { w: pinW, h: pinH };
            captureActive = true; _skipOutDispose = true;
            console.log(`[RIFE] session created PINNED ${pinW}x${pinH} + graph capture`);
          } catch (gcErr) {
            console.warn("[RIFE] graph-capture creation failed — plain session:", gcErr.message);
            captureBroken = true;
            session = await ort.InferenceSession.create(MODEL_IO.url(), buildOpts(false));
          }
        } else {
          session = await ort.InferenceSession.create(MODEL_IO.url(), buildOpts(false));
        }
      }
    } catch (epErr) {
      // JSPI runtime failure (bundle loaded but sessions won't create/run under
      // it) → reload the proven asyncify build and retry ONCE before any other
      // fallback. Load-time reload is safe: nothing is bound to a device yet.
      if (ortIsJspi) {
        console.warn("[RIFE] session failed under JSPI — reloading asyncify build:", epErr.message);
        ort = null; ortIsJspi = false;
        await loadORT(true);
        try { ort.env.webgpu.enableFp16 = false; } catch {}
        try {
          session = await ort.InferenceSession.create(MODEL_IO.url(), buildOpts(false));
          if (!MODEL_IO.inputName) MODEL_IO.inputName = session.inputNames[0];
          if (!MODEL_IO.outputName) MODEL_IO.outputName = session.outputNames[0];
          modelAvailable = true;
          console.log(`[RIFE] ready on asyncify fallback (in=${MODEL_IO.inputName} out=${MODEL_IO.outputName})`);
          return true;
        } catch (e2) { console.warn("[RIFE] asyncify retry also failed:", e2.message); }
      }
      // WebGPU EP failed entirely — retry on wasm so we at least learn if it's an
      // EP issue vs a model/runtime issue. (wasm will be slow but proves the path.)
      console.warn("[RIFE] WebGPU EP failed, trying wasm:", epErr.message);
      stage = "create-session-wasm";
      try { ort.env.webgpu.enableFp16 = false; } catch {}
      fp16Active = false;
      session = await ort.InferenceSession.create(MODEL_IO.url(), {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      lastError = "webgpu-unavailable-using-wasm";
      usingWasmEp = true;
    }
    if (!MODEL_IO.inputName) MODEL_IO.inputName = session.inputNames[0];
    if (!MODEL_IO.outputName) MODEL_IO.outputName = session.outputNames[0];
    modelAvailable = true;
    console.log(`[RIFE] ready (in=${MODEL_IO.inputName} out=${MODEL_IO.outputName})`);
    return true;
  } catch (e) {
    lastError = `${stage}: ${e.message}`;
    console.error(`[RIFE] init FAILED at ${stage}:`, e.message, e);
    modelAvailable = false;
    return false;
  }
}
export function getLastError() { return lastError; }

// --- GPU-resident interpolation path (optional; falls back to CPU interpolate) ---
let gpuInterp = null;
let gpuTried = false;
// Expose ORT's device so the GPU path can build buffers on the same device (required
// for Tensor.fromGpuBuffer). Available after session creation in ORT WebGPU builds.
export function getOrtDevice() {
  try { return ort && ort.env && ort.env.webgpu ? ort.env.webgpu.device : null; } catch { return null; }
}
export function getOrt() { return ort; }

// Try to initialize the GPU-resident path. Returns true if active. Safe to call
// after initRife(). Fails → CPU interpolate() stays in use.
export async function initGpuInterp({ log, warn } = {}) {
  if (gpuTried) return !!gpuInterp;
  gpuTried = true;
  try {
    if (!isReady()) return false;
    const device = getOrtDevice();
    if (!device) { (warn||console.warn)("[RIFE] no ORT device for GPU path"); return false; }
    const mod = await import(chrome.runtime.getURL("fsrcnnx-rife-gpu.js"));
    const g = new mod.GpuInterp({ log, warn });
    if (await g.init(device, ort)) {
      gpuInterp = g;
      gpuInterp.skipOutputDispose = _skipOutDispose; // capture session owns its output buffer
      (log||console.log)("[RIFE] GPU-resident path active");
      return true;
    }
    return false;
  } catch (e) { (warn||console.warn)("[RIFE] GPU path init failed:", e.message); return false; }
}
// Standalone blend GPU path: NO RIFE model load. If `device` is provided (e.g. the
// upscaler's, for the upscale→interpolate chain) the pipeline builds on it so
// textures can be shared; otherwise it requests its own device.
export async function initGpuBlendStandalone({ log, warn, device } = {}) {
  if (gpuInterp) return true; // already have a pipeline (RIFE or standalone)
  try {
    const mod = await import(chrome.runtime.getURL("fsrcnnx-rife-gpu.js"));
    const g = new mod.GpuInterp({ log, warn });
    if (await g.init(device || null, null)) { gpuInterp = g; gpuTried = true; (log||console.log)(`[RIFE] standalone blend GPU path active${device ? " (shared device)" : ""}`); return true; }
    return false;
  } catch (e) { (warn||console.warn)("[RIFE] standalone blend init failed:", e.message); return false; }
}
export function gpuCaptureTex(tex) { return gpuInterp ? gpuInterp.captureTexToPooled(tex, MODEL_IO.padTo, MODEL_IO.channels()) : null; }
export function gpuActive() { return !!gpuInterp; }
export function gpuRifeCapable() { return !!(gpuInterp && gpuInterp._rifeCapable); }

export function destroyGpuInterp() { try { gpuInterp && gpuInterp.destroy(); } catch {} gpuInterp = null; gpuTried = false; }

// Full GPU-presentation API (no readback): configure a WebGPU canvas, capture to a
// pooled texture, interpolate to a pooled texture, present a texture, recycle.
export function gpuConfigureCanvas(canvas) { return gpuInterp ? gpuInterp.configureCanvas(canvas) : false; }
export function gpuCapture(video) { return gpuInterp ? gpuInterp.captureToPooled(video, MODEL_IO.padTo, MODEL_IO.channels()) : null; }
export function gpuHasPrev() { return gpuInterp ? gpuInterp.hasPrev() : false; }
export function gpuAdvance() { if (gpuInterp) gpuInterp.advance(); }
export async function gpuTween(w, h, t, useStatic) {
  if (!gpuInterp) return null;
  const t0 = performance.now();
  const tex = await gpuInterp.interpolateToPooledTex(session, MODEL_IO, w, h, t, useStatic);
  const s = gpuInterp.getSplit ? gpuInterp.getSplit() : null;
  if (s) { timing.pre = s.pack; timing.infer = s.run; timing.post = s.comp; }
  return tex;
}
export function gpuPresent(tex) { return gpuInterp ? gpuInterp.presentTexture(tex) : false; }
export function gpuRelease(tex) { if (gpuInterp) gpuInterp.releaseTex(tex); }
export function gpuRetain(tex) { if (gpuInterp) gpuInterp.retainTex(tex); }
// Pipelined RIFE tween on an EXPLICIT pooled pair — safe to run while the grab loop
// keeps capturing (the shared prev/cur ping-pong may advance mid-inference).
export async function gpuTweenPair(prevTex, curTex, t, useStatic, scale = 1) {
  if (!gpuInterp || !isReady()) return null;
  const tex = await gpuInterp.interpolateToPooledTex(session, MODEL_IO, 0, 0, t, useStatic, prevTex, curTex, scale);
  const s = gpuInterp.getSplit ? gpuInterp.getSplit() : null;
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

// per-tween timing breakdown (ms): pre = draw+pack (input transfer), infer =
// session.run (GPU compute), post = output read+putImageData (output transfer).
// Lets us measure how much the CPU round-trip costs vs inference before deciding
// whether the GPU-tensor path is worth its risk.
const timing = { pre: 0, infer: 0, post: 0 };
export function getTiming() { return { ...timing }; }
let fp16Active = false;
export function isFp16() { return fp16Active || !!(MODELS[currentModelKey] && MODELS[currentModelKey].fp16); }

// Fill a preallocated planar slice (3 channels, NCHW) from a canvas context's
// pixels, normalized. Writes into `dst` at channel-plane offset `base`.
function fillPlanar(srcCanvasCtx, padW, padH, dst, base) {
  const { data } = srcCanvasCtx.getImageData(0, 0, padW, padH); // RGBA
  const plane = padW * padH;
  const n = MODEL_IO.normalize;
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
export async function interpolate(frameA, frameB, w, h, t = 0.5, scale = 1.0) {
  if (!isReady()) return null;
  try {
    const iw = Math.max(MODEL_IO.padTo, Math.round(w * scale));
    const ih = Math.max(MODEL_IO.padTo, Math.round(h * scale));
    const padW = Math.ceil(iw / MODEL_IO.padTo) * MODEL_IO.padTo;
    const padH = Math.ceil(ih / MODEL_IO.padTo) * MODEL_IO.padTo;
    // (re)allocate reusable buffers only when the padded size changes
    if (_bufW !== padW || _bufH !== padH) {
      _ca = new OffscreenCanvas(padW, padH); _cb = new OffscreenCanvas(padW, padH);
      _cout = new OffscreenCanvas(padW, padH);
      _ctxA = _ca.getContext("2d", { willReadFrequently: true });
      _ctxB = _cb.getContext("2d", { willReadFrequently: true });
      _octx = _cout.getContext("2d");
      const plane = padW * padH;
      const ch = MODEL_IO.channels(); // 6 (frames only) or 7 (+ timestep plane)
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
    fillPlanar(_ctxA, padW, padH, _inBuf, 0);
    fillPlanar(_ctxB, padW, padH, _inBuf, 3 * plane);
    if (MODEL_IO.timestepPlane()) {
      // 7th channel: a plane filled with the timestep t (verified: channel 6
      // drives interpolation position). Fill only when t changed to save work —
      // but t is constant (0.5) here, so fill once per (re)alloc via _tsFilled.
      const base = 6 * plane;
      if (_tsFilled !== t || _tsPlaneW !== padW || _tsPlaneH !== padH) {
        _inBuf.fill(t, base, base + plane);
        _tsFilled = t; _tsPlaneW = padW; _tsPlaneH = padH;
      }
    }
    feeds[MODEL_IO.inputName] = _inTensor;
    timing.pre = performance.now() - tPre0;

    const tInf0 = performance.now();
    const result = await session.run(feeds);
    timing.infer = performance.now() - tInf0;
    const outTensor = result[MODEL_IO.outputName];
    const tPost0 = performance.now();
    // output is on the GPU (preferredOutputLocation:'gpu-buffer'); download it for
    // the CPU passthrough+present path. Accessing .data on a GPU tensor THROWS, so
    // read via getData() (async download) guarded by location.
    let od;
    if (outTensor.location && outTensor.location !== "cpu") {
      od = await outTensor.getData(true);
    } else {
      od = outTensor.data;
    }
    const dn = MODEL_IO.denormalize;
    const img = _outImg;
    // STATIC-REGION PASSTHROUGH: the jitter on static detail comes from RIFE
    // reconstructing (downscale→infer→upscale) still content slightly differently
    // than the pixel-exact real frames it alternates with. Fix: per pixel, measure
    // how much frame A and frame B differ (their motion); where they're nearly
    // identical (static), output the REAL pixel (A/B average) instead of RIFE's
    // reconstruction, so static regions are pixel-stable. Where they differ
    // (motion), use RIFE. Smooth ramp between the two by difference magnitude.
    // A,B live in _inBuf (normalized): A at [0..plane) per channel, B at [3plane..).
    const useStatic = MODEL_IO.staticPassthrough !== false;
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
  }
}

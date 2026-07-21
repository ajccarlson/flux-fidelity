// fsrcnnx-main.js — real upscaling pipeline (ES module).
// Exported API: setMode(mode), getStatus(). content.js dynamic-imports this and
// relays popup messages. Modes: 'off' | 'passthrough' | 'upscale'.

import { FsrcnnxModel, selectModel } from "./fsrcnnx-runtime.js";
import { allocateModelChain, preflightModelChain } from "./fsrcnnx-model-bundle.js";
import { createNeuralEngine, validateNeuralManifest } from "./fsrcnnx-neural.js";
import { LUMA_EXTRACT_WGSL, RECOMBINE_WGSL } from "./fsrcnnx-color.js";
import { SsimDownscaler } from "./fsrcnnx-ssimds-runtime.js";
import { buildSharpenShader } from "./fsrcnnx-sharpen.js";
import { buildDebandShader } from "./fsrcnnx-deband.js";
import { ArtCnnModel } from "./fsrcnnx-artcnn-runtime.js";

const TAG = "[FSRCNNX]";
const log = (...a) => console.log(TAG, ...a);
const warn = (...a) => console.warn(TAG, ...a);

// Keep a single processed surface within an 8K-class pixel budget even on
// adapters that advertise a larger texture limit.  The pipeline holds several
// rgba16float intermediates (plus model feature maps), so accepting any size the
// adapter can represent is not, by itself, a safe allocation policy.
const MAX_PROCESSING_PIXELS = 7680 * 4320;
const sizeWarningAt = new Map();
function textureSizeAllowed(w, h, label = "texture") {
  const width = Number(w), height = Number(h);
  const limit = Math.max(1, Number(device?.limits?.maxTextureDimension2D) || 8192);
  const valid = Number.isSafeInteger(width) && Number.isSafeInteger(height) &&
    width > 0 && height > 0 && width <= limit && height <= limit &&
    width * height <= Math.min(MAX_PROCESSING_PIXELS, limit * limit);
  if (!valid) {
    const now = performance.now();
    const key = `${label}:${width}x${height}`;
    if (now - (sizeWarningAt.get(key) || -Infinity) > 5000) {
      sizeWarningAt.set(key, now);
      warn(`${label} ${width}x${height} exceeds the safe processing limit ` +
        `(max dimension ${limit}, max pixels ${Math.min(MAX_PROCESSING_PIXELS, limit * limit)})`);
    }
  }
  return valid;
}

function storageBufferSizeAllowed(bytes, label) {
  const size = Number(bytes);
  const maxBuffer = Number(device?.limits?.maxBufferSize) || 256 * 1024 * 1024;
  const maxBinding = Number(device?.limits?.maxStorageBufferBindingSize) || maxBuffer;
  const limit = Math.min(maxBuffer, maxBinding);
  const valid = Number.isSafeInteger(size) && size > 0 && size <= limit;
  if (!valid) {
    const key = `${label}:${size}`;
    const now = performance.now();
    if (now - (sizeWarningAt.get(key) || -Infinity) > 5000) {
      sizeWarningAt.set(key, now);
      warn(`${label} requires ${size} bytes; adapter storage-buffer limit is ${limit}`);
    }
  }
  return valid;
}

function modelFitsProcessingBudget(model, width, height, label) {
  if (!model?.preflight) return true;
  try {
    model.preflight(width, height);
    return true;
  } catch (error) {
    if (!/^MODEL_(?:DIMENSION|DIMENSIONS|WORKING_SET|BINDING)/.test(error?.code || "")) throw error;
    const key = `model:${label}:${width}x${height}:${error.code}`;
    const now = performance.now();
    if (now - (sizeWarningAt.get(key) || -Infinity) > 5000) {
      sizeWarningAt.set(key, now);
      warn(`${label} bypassed: ${error.message}`);
    }
    return false;
  }
}

const MODEL_FILES = ["FSRCNNX_x2_16-0-4-1", "FSRCNNX_x3_16-0-4-1", "FSRCNNX_x4_16-0-4-1"];

let mode = "off";
let modeSelectionGeneration = 0;
let device = null, deviceOwnedByMain = false;
const watchedDeviceLosses = new WeakSet();
const lostDevices = new WeakSet();
let deviceRecoveryGeneration = 0;
let deviceRecoveryPromise = null;
let deviceRecoveryTimer = null;
let context = null, format = null, canvas = null, video = null, sampler = null;
let extractPipeline = null, recombinePipeline = null, passthroughPipeline = null;
// INVERTED CHAIN (#4): tex-ingest twins of the ext-consuming pipelines, plus state.
let extractPipelineTex = null, recombinePipelineTex = null, recombine16PipelineTex = null;
let chainInverted = false;   // interp drives upscales; our per-video-frame loop pauses
let _texSource = null;       // one-shot pooled-frame override for renderUpscale
let interpInvertPref = true; // DEFAULT ON since v0.48.6 (experiment #4 verdict); per-site saved pref overrides
let interpAutoFallbackPref = true; // RIFE→blend performance fallback (persisted per site)
let interpLadderPref = false; // blend ladder (persisted per site)
let _gpuErrWinStart = 0, _gpuErrCount = 0, _invRestarts = 0, _invRestartLast = 0; // present-path breaker state
let recombine16Pipeline = null, blitPipeline = null;
// Interpolation chain tap: the interpolator (blend engine, same device) can consume
// the upscaler's finished frames instead of the raw video.
let chainTapOn = false, chainTapTex = null, chainTapFrame = 0;
export function chainTap(on) {
  chainTapOn = !!on;
  if (!chainTapOn) { try { chainTapTex && chainTapTex.destroy(); } catch {} chainTapTex = null; }
  return chainTapOn;
}
export function chainInfo() {
  // available only when the upscaler is actively rendering the primary video
  if (!device || mode === "off" || !chainTapTex) return null;
  return { device, tex: chainTapTex, w: chainTapTex.width, h: chainTapTex.height, frame: chainTapFrame, format };
}
export function chainAvailable() { return !!(device && mode !== "off"); }
export function chainDevice() { return device || null; }
export function chainTargetDims() {
  // upscaler OUTPUT dims (canvas) — knowable BEFORE the tap produces a frame, so
  // the RIFE session can be dim-pinned at creation (graph capture requirement).
  return (canvas && canvas.width && canvas.height) ? { w: canvas.width, h: canvas.height } : null;
}
let lumaTexture = null, lumaW = 0, lumaH = 0;
let hiRGB = null, hiRGBW = 0, hiRGBH = 0;
let ssimds = null, ssimdsEnabled = true; // on by default; only fires on overshoot
let lastSSimDS = false;
// scale-switch hysteresis state (see renderUpscale)
let _scaleHeld, _scalePending = null, _scalePendingSince = 0;
let _scaleHeldSrcW = 0, _scaleHeldSrcH = 0, _scaleLockLogged = false;
function resetScaleSelection() {
  _scaleHeld = undefined;
  _scalePending = null;
  _scalePendingSince = 0;
  _scaleHeldSrcW = 0;
  _scaleHeldSrcH = 0;
  _scaleLockLogged = false;
}
let sharpenEnabled = false, sharpenStrength = 1.0;
let sharpenPipeline = null, sharpenStrengthBuilt = null;
let debandEnabled = false, debandStrength = 1.0;
let debandCanvasPipeline = null, debandFloatPipeline = null;
let debandStrengthBuilt = null, debandTimeBuf = null;
let debandInterTex = null, debandInterW = 0, debandInterH = 0; // intermediate when chaining deband->sharpen
let dispRGB = null, dispRGBW = 0, dispRGBH = 0; // offscreen display-res for sharpen input
let models = [], activeModel = null;
let modelsDevice = null, modelLoadPromise = null, modelLoadDevice = null;
const ART_FILES = ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"];
let engine = "fsrcnnx"; // "fsrcnnx" | "fsrcnnx-hi" | "artcnn" | "neural"
let engineSelectionGeneration = 0;
let neuralEng = null, neuralModelKey = "", neuralBusy = false, neuralFail = 0; // v0.49.0 ONNX engine
let interpPausedByNeural = false;
let _neuralList = []; // manifest summary for the popup, loaded eagerly
(async () => { try {
  const r = await fetch(chrome.runtime.getURL("model/neural/manifest.json"));
  if (r.ok) {
    _neuralList = validateNeuralManifest(await r.json())
      .map((m) => ({ key: m.key, label: m.label || m.key, scale: m.scale }));
  }
} catch {} })();
let artVariant = "ArtCNN_C4F32";
let chainDepth = 1; // 1 = single 2x, 2 = chained 4x, 3 = chained 8x (2x-only engines)
let artLoadPending = false, artDiagLogged = false;
let rvfc = null, rvfcKind = null, frameCount = 0, lastLog = 0;
let upscalePolicy = "display"; // default: upscale whenever source < display (good for 4K)
let protectedSource = false; // last setMode/recheck found DRM
let protectedReason = null;  // "drm" | "tainted" | null

// ---- advanced / site-specific options (off by default) -------------------
let optHoverReveal = false;   // fade overlay out while cursor is over the player
let optAllVideos = false;     // upscale every qualifying video, not just the main one
let hoverHidden = false;      // current hover-reveal state for the primary overlay

// ---- per-site persistence (chrome.storage.local) -------------------------
const SITE_KEY = "fsrcnnx_sites"; // { "<hostname>": { mode, engine, artVariant, policy, ssimds, sharpen, sharpenStrength } }
function siteHost() { try { return location.hostname || "_"; } catch { return "_"; } }

async function loadSitePrefs() {
  try {
    const all = (await chrome.storage.local.get(SITE_KEY))[SITE_KEY] || {};
    return all[siteHost()] || null;
  } catch { return null; }
}
async function saveSitePrefs() {
  try {
    const all = (await chrome.storage.local.get(SITE_KEY))[SITE_KEY] || {};
    all[siteHost()] = {
      mode, engine, artVariant, policy: upscalePolicy,
      ssimds: ssimdsEnabled, sharpen: sharpenEnabled, sharpenStrength,
      hoverReveal: optHoverReveal, allVideos: optAllVideos,
      deband: debandEnabled, debandStrength,
      images: optImages,
      interpolate: optInterpolate,
      interpEngine: pendingEngine || null,
      neuralModel: neuralModelKey || null,
      interpTargetFps: pendingTargetFps != null ? pendingTargetFps : null,
      interpAutoFallback: interpAutoFallbackPref,
      interpLadder: interpLadderPref,
      interpInvert: interpInvertPref,
    };
    await chrome.storage.local.set({ [SITE_KEY]: all });
  } catch {}
}

function notifyProtected() {
  protectedSource = true;
  try { chrome.runtime.sendMessage({ type: "FSRCNNX_PROTECTED", host: siteHost() }); } catch {}
}

// Tell the service worker the current mode so it can update the toolbar icon
// (color when upscaling, monochrome otherwise) and badge.
function notifyState() {
  try { chrome.runtime.sendMessage({ type: "FSRCNNX_STATE", mode, host: siteHost() }); } catch {}
}

const srcCache = { fsrcnnx: {}, artcnn: {} }; // name -> {manifest, wgsl}

// High-quality FSRCNNX (56-16-4-1, 2x-only). Two instances enable chaining to 4x.
const HI_MODEL = "FSRCNNX_x2_56-16-4-1";
let hiLoadPending = false, chainedHi = null;
let hiSourcePromise = null, hiStageBuildPromise = null;
async function loadHiModelSource() {
  if (srcCache.fsrcnnx[HI_MODEL]) return srcCache.fsrcnnx[HI_MODEL];
  if (hiSourcePromise) return hiSourcePromise;
  const promise = (async () => {
    const base = chrome.runtime.getURL(`model/${HI_MODEL}`);
    const [manifestResponse, wgslResponse] = await Promise.all([
      fetch(`${base}.passes.json`),
      fetch(`${base}.wgsl`),
    ]);
    if (!manifestResponse.ok || !wgslResponse.ok) {
      throw new Error(`${HI_MODEL} fetch failed (${manifestResponse.status}/${wgslResponse.status})`);
    }
    const source = { name: HI_MODEL, manifest: await manifestResponse.json(), wgsl: await wgslResponse.text() };
    srcCache.fsrcnnx[HI_MODEL] = source;
    return source;
  })().finally(() => {
    if (hiSourcePromise === promise) hiSourcePromise = null;
  });
  hiSourcePromise = promise;
  return promise;
}

// Map an upscale policy to a chain depth for 2x-only engines (ArtCNN / FSRCNNX
// high): force2 -> 1 (2x), force4 -> 2 (4x), force8 -> 3 (8x). display/auto run
// a single stage and let the display-fit/SSimDS path handle the rest.
function policyToDepth(p) {
  if (p === "force8") return 3;
  if (p === "force4" || p === "force3") return 2;
  return 1;
}

// Build (and cache) `depth` FSRCNNX-high stage instances. Each stage needs its
// own instance for its own per-size texture cache.
let hiStages = []; // FsrcnnxModel[]
async function ensureHiStages(depth) {
  const requestedDepth = Math.max(1, depth | 0);
  if (hiStages.length >= requestedDepth) return hiStages.slice(0, requestedDepth);
  if (hiStageBuildPromise) {
    try { await hiStageBuildPromise; } catch {}
    return ensureHiStages(requestedDepth);
  }
  const targetDevice = device;
  const baseStages = hiStages;
  if (!targetDevice) throw new Error("cannot build FSRCNNX-high stages without a device");
  const promise = (async () => {
    const source = await loadHiModelSource();
    if (device !== targetDevice || hiStages !== baseStages) {
      throw new Error("FSRCNNX-high stage build superseded by device change");
    }
    const created = [];
    try {
      while (baseStages.length + created.length < requestedDepth) {
        created.push(new FsrcnnxModel(targetDevice, source.manifest, source.wgsl, { expectedName: HI_MODEL }));
      }
    } catch (error) {
      for (const stage of created) { try { stage.destroy?.(); } catch {} }
      throw error;
    }
    if (device !== targetDevice || hiStages !== baseStages) {
      for (const stage of created) { try { stage.destroy?.(); } catch {} }
      throw new Error("FSRCNNX-high stage build superseded by device change");
    }
    hiStages = [...baseStages, ...created];
    return hiStages.slice(0, requestedDepth);
  })().finally(() => {
    if (hiStageBuildPromise === promise) hiStageBuildPromise = null;
  });
  hiStageBuildPromise = promise;
  return promise;
}

// Build (and cache) `depth` ArtCNN stage instances for the current variant.
let artStages = {}; // variant -> ArtCnnModel[]
const artSourcePromises = new Map();
let artStageBuildPromise = null;

async function loadArtModelSource(name) {
  if (srcCache.artcnn[name]) return srcCache.artcnn[name];
  if (artSourcePromises.has(name)) return artSourcePromises.get(name);
  const promise = (async () => {
    const base = chrome.runtime.getURL(`model/${name}.artcnn`);
    const [manifestResponse, wgslResponse] = await Promise.all([
      fetch(`${base}.json`),
      fetch(`${base}.wgsl`),
    ]);
    if (!manifestResponse.ok || !wgslResponse.ok) {
      throw new Error(`${name} fetch failed (${manifestResponse.status}/${wgslResponse.status})`);
    }
    const source = { manifest: await manifestResponse.json(), wgsl: await wgslResponse.text() };
    srcCache.artcnn[name] = source;
    return source;
  })().finally(() => {
    if (artSourcePromises.get(name) === promise) artSourcePromises.delete(name);
  });
  artSourcePromises.set(name, promise);
  return promise;
}

async function ensureArtStages(name, depth) {
  const requestedDepth = Math.max(1, depth | 0);
  if ((artStages[name]?.length || 0) >= requestedDepth) return artStages[name].slice(0, requestedDepth);
  if (artStageBuildPromise) {
    try { await artStageBuildPromise; } catch {}
    return ensureArtStages(name, requestedDepth);
  }
  const targetDevice = device;
  const stageMap = artStages;
  const baseStages = stageMap[name] || [];
  if (!targetDevice) throw new Error("cannot build ArtCNN stages without a device");
  const promise = (async () => {
    const source = await loadArtModelSource(name);
    if (device !== targetDevice || artStages !== stageMap || (stageMap[name] || baseStages) !== baseStages) {
      throw new Error("ArtCNN stage build superseded by device change");
    }
    const created = [];
    try {
      while (baseStages.length + created.length < requestedDepth) {
        created.push(new ArtCnnModel(targetDevice, source.manifest, source.wgsl, { expectedName: name }));
      }
    } catch (error) {
      for (const stage of created) { try { stage.destroy?.(); } catch {} }
      throw error;
    }
    if (device !== targetDevice || artStages !== stageMap || (stageMap[name] || baseStages) !== baseStages) {
      for (const stage of created) { try { stage.destroy?.(); } catch {} }
      throw new Error("ArtCNN stage build superseded by device change");
    }
    stageMap[name] = [...baseStages, ...created];
    return stageMap[name].slice(0, requestedDepth);
  })().finally(() => {
    if (artStageBuildPromise === promise) artStageBuildPromise = null;
  });
  artStageBuildPromise = promise;
  return promise;
}

async function loadModels() {
  const targetDevice = device;
  if (!targetDevice) throw new Error("cannot load models without a WebGPU device");
  if (models.length && modelsDevice === targetDevice) return models;
  if (modelLoadPromise) {
    if (modelLoadDevice === targetDevice) {
      return modelLoadPromise.then(() => (
        models.length && modelsDevice === targetDevice ? models : loadModels()
      ));
    }
    try { await modelLoadPromise; } catch {}
    return loadModels();
  }

  const promise = (async () => {
    // Fetch and validate the complete source set before constructing or exposing
    // any model. Concurrent enables share this transaction, so callers can never
    // observe a partially populated array or reorder models by completion time.
    const sources = await Promise.all(MODEL_FILES.map(async (name) => {
      const base = chrome.runtime.getURL(`model/${name}`);
      const [manifestResponse, wgslResponse] = await Promise.all([
        fetch(`${base}.passes.json`),
        fetch(`${base}.wgsl`),
      ]);
      if (!manifestResponse.ok || !wgslResponse.ok) {
        throw new Error(`model ${name} fetch failed (${manifestResponse.status}/${wgslResponse.status})`);
      }
      return { name, manifest: await manifestResponse.json(), wgsl: await wgslResponse.text() };
    }));
    if (device !== targetDevice) throw new Error("model load superseded by device change");

    const created = [];
    try {
      for (const source of sources) {
        created.push(new FsrcnnxModel(targetDevice, source.manifest, source.wgsl, { expectedName: source.name }));
      }
    } catch (error) {
      for (const model of created) { try { model.destroy?.(); } catch {} }
      throw error;
    }
    if (device !== targetDevice) {
      for (const model of created) { try { model.destroy?.(); } catch {} }
      throw new Error("model load superseded by device change");
    }

    for (const source of sources) {
      srcCache.fsrcnnx[source.name] = { manifest: source.manifest, wgsl: source.wgsl };
    }
    const oldModels = models;
    models = created;
    modelsDevice = targetDevice;
    activeModel = null;
    for (const model of oldModels) { try { model.destroy?.(); } catch {} }
    log(`loaded ${models.length} models`);
    return models;
  })().finally(() => {
    if (modelLoadPromise === promise) {
      modelLoadPromise = null;
      modelLoadDevice = null;
    }
  });
  modelLoadDevice = targetDevice;
  modelLoadPromise = promise;
  return promise;
}

// Chains N ArtCnnModel stages, each a 2x doubler, for a 2^N luma upscale.
// Implements the same interface the render path expects (.scale/.allocate/.run).
class ChainedArtCnn {
  constructor(stages) { this.stages = stages; this.scale = Math.pow(2, stages.length); }
  preflight(lumaW, lumaH) { return preflightModelChain(this.stages, lumaW, lumaH, "ArtCNN"); }
  allocate(lumaW, lumaH, lumaTex) {
    const plan = allocateModelChain(this.stages, lumaW, lumaH, lumaTex, "ArtCNN");
    this.lumaW = lumaW; this.lumaH = lumaH;
    return plan;
  }
  run(encoder, lumaTex) {
    let w = this.lumaW, h = this.lumaH, tex = lumaTex;
    for (const s of this.stages) { s.allocate(w, h, tex); tex = s.run(encoder, tex); w *= 2; h *= 2; }
    return tex;
  }
}

// Same idea for the high-quality FSRCNNX (56-16-4-1 is 2x-only): chain N
// FsrcnnxModel stages for a 2^N luma upscale.
class ChainedFsrcnnx {
  constructor(stages) { this.stages = stages; this.scale = Math.pow(2, stages.length); }
  preflight(lumaW, lumaH) { return preflightModelChain(this.stages, lumaW, lumaH, "FSRCNNX-high"); }
  allocate(lumaW, lumaH, lumaTex) {
    const plan = allocateModelChain(this.stages, lumaW, lumaH, lumaTex, "FSRCNNX-high");
    this.lumaW = lumaW; this.lumaH = lumaH;
    return plan;
  }
  run(encoder, lumaTex) {
    let w = this.lumaW, h = this.lumaH, tex = lumaTex;
    for (const s of this.stages) { s.allocate(w, h, tex); tex = s.run(encoder, tex); w *= 2; h *= 2; }
    return tex;
  }
}
let chainedArt = null;

// Collect <video> elements across the document AND any open shadow roots.
// Plain querySelectorAll does not pierce shadow DOM, so sites that wrap their
// player in a web component (e.g. Reddit's <shreddit-player>) hide their <video>
// inside a shadow tree. We walk shadow roots recursively to find them.
function deepQueryVideos(root = document, acc = [], seen = new Set()) {
  if (!root || seen.has(root)) return acc;
  seen.add(root);
  try {
    for (const v of root.querySelectorAll("video")) acc.push(v);
    // descend into open shadow roots of every element in this tree
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) deepQueryVideos(el.shadowRoot, acc, seen);
    }
  } catch {}
  return acc;
}

function findVideo() {
  const vids = deepQueryVideos()
    .filter((v) => v.videoWidth > 0 && v.videoHeight > 0); // decoded, has dimensions
  if (!vids.length) return null;
  // Prefer a known player container if present (YouTube), else the largest video
  // that is actually on-screen and not tiny (skip hidden/background/ad pixels).
  const inYt = vids.find((v) => v.closest("#movie_player"));
  if (inYt) return inYt;
  const scored = vids
    .map((v) => {
      const r = v.getBoundingClientRect();
      const visible = r.width > 0 && r.height > 0 &&
        r.bottom > 0 && r.right > 0 &&
        r.top < (window.innerHeight || 1e9) && r.left < (window.innerWidth || 1e9);
      const area = r.width * r.height;
      const playing = !v.paused && !v.ended && v.readyState >= 2;
      return { v, area, visible, playing };
    })
    .filter((s) => s.visible && s.area > 64 * 64) // ignore tiny/offscreen videos
    .sort((a, b) => (b.playing - a.playing) || (b.area - a.area)); // playing first, then biggest
  return scored.length ? scored[0].v : null;
}

function ensureCanvas() {
  if (canvas) return;
  canvas = document.createElement("canvas");
  canvas.id = "fsrcnnx-overlay";
  Object.assign(canvas.style, { position: "absolute", top: "0", left: "0", pointerEvents: "none", zIndex: "10", transition: "opacity 0.18s ease" });
}

// True if the node lives inside a shadow tree (its root node is a ShadowRoot).
function inShadowDom(node) {
  const root = node?.getRootNode?.();
  return !!root && root instanceof ShadowRoot;
}

function positionCanvas(outW, outH) {
  if (!canvas || !video) return false;
  if (outW && outH && !textureSizeAllowed(outW, outH, "canvas output")) {
    canvas.style.display = "none"; // expose the original video instead of a stale frame
    return false;
  }
  const v = video.getBoundingClientRect();
  // For videos inside a shadow root (e.g. Reddit's <shreddit-player>), don't try
  // to inject the canvas into the component's shadow tree — it may re-render and
  // remove it, and its styles can clash. Instead overlay at the document body
  // using fixed positioning in viewport coordinates.
  if (inShadowDom(video)) {
    if (canvas.parentElement !== document.body) document.body.appendChild(canvas);
    Object.assign(canvas.style, { position: "fixed", zIndex: "2147483646" });
    canvas.style.left = `${v.left}px`;
    canvas.style.top = `${v.top}px`;
    canvas.style.width = `${v.width}px`;
    canvas.style.height = `${v.height}px`;
  } else {
    const parent = video.parentElement;
    if (!parent) return false;
    if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
    if (canvas.parentElement !== parent) parent.appendChild(canvas);
    canvas.style.position = "absolute";
    const p = parent.getBoundingClientRect();
    canvas.style.left = `${v.left - p.left}px`;
    canvas.style.top = `${v.top - p.top}px`;
    canvas.style.width = `${v.width}px`;
    canvas.style.height = `${v.height}px`;
  }
  if (outW && outH && (canvas.width !== outW || canvas.height !== outH)) {
    canvas.width = outW; canvas.height = outH;
  }
  if (mode !== "off") canvas.style.display = "block";
  return true;
}

let webGpuInitPromise = null;
function initWebGPU() {
  if (device && !lostDevices.has(device)) return Promise.resolve(true);
  if (webGpuInitPromise) return webGpuInitPromise;
  const promise = (async () => {
    const ok = await initWebGPUInternal();
    // GPUDevice.lost can already be settled when watchDeviceLoss() subscribes.
    // Validate publication after that microtask has had a chance to invalidate
    // the candidate, and give every concurrent caller this same truthful result.
    await Promise.resolve();
    return !!(ok && device && !lostDevices.has(device));
  })().finally(() => {
    if (webGpuInitPromise === promise) webGpuInitPromise = null;
  });
  webGpuInitPromise = promise;
  return promise;
}

function watchDeviceLoss(ownerDevice) {
  if (!ownerDevice || watchedDeviceLosses.has(ownerDevice) || !ownerDevice.lost?.then) return;
  watchedDeviceLosses.add(ownerDevice);
  ownerDevice.lost.then((info) => {
    lostDevices.add(ownerDevice);
    if (device !== ownerDevice) {
      log("old device released (expected after ownership transfer)");
      return;
    }
    handleCurrentDeviceLoss(ownerDevice, info);
  }).catch(() => {});
}

function invalidateMainDeviceResources() {
  invalidateImageUpscaler();
  clearMultiTargets();
  const oldModels = new Set([
    ...models,
    ...hiStages,
    ...Object.values(artStages).flat(),
  ]);
  for (const model of oldModels) { try { model?.destroy?.(); } catch {} }
  for (const texture of [chainTapTex, lumaTexture, hiRGB, dispRGB, debandInterTex]) {
    try { texture?.destroy?.(); } catch {}
  }
  try { debandTimeBuf?.destroy?.(); } catch {}
  try { ssimds?.destroy?.(); } catch {}
  try { context?.unconfigure?.(); } catch {}

  chainTapTex = null; chainTapFrame = 0;
  lumaTexture = null; lumaW = 0; lumaH = 0;
  hiRGB = null; hiRGBW = 0; hiRGBH = 0;
  dispRGB = null; dispRGBW = 0; dispRGBH = 0;
  debandInterTex = null; debandInterW = 0; debandInterH = 0;
  debandTimeBuf = null;
  ssimds = null;
  extractPipeline = recombinePipeline = recombine16Pipeline = blitPipeline = null;
  extractPipelineTex = recombinePipelineTex = recombine16PipelineTex = null;
  passthroughPipeline = null;
  sharpenPipeline = null; sharpenStrengthBuilt = null;
  debandCanvasPipeline = debandFloatPipeline = null; debandStrengthBuilt = null;
  sampler = null; context = null; format = null;
  models = []; modelsDevice = null; activeModel = null;
  hiStages = []; artStages = {}; chainedHi = null; chainedArt = null;
  hiLoadPending = false; artLoadPending = false;
  resetScaleSelection();
  _texSource = null;
}

function cancelDeviceRecovery() {
  deviceRecoveryGeneration++;
  if (deviceRecoveryTimer != null) clearTimeout(deviceRecoveryTimer);
  deviceRecoveryTimer = null;
}

function handleCurrentDeviceLoss(lostDevice, info) {
  if (device !== lostDevice) return;
  adoptionGeneration++;
  const generation = ++deviceRecoveryGeneration;
  warn(`device lost: ${info?.message || info?.reason || "unknown reason"}`);
  device = null;
  deviceOwnedByMain = false;
  adopting = false;
  invalidateMainDeviceResources();
  if (canvas) {
    canvas.style.display = "none";
    canvas.style.opacity = "1";
  }
  // Neural keeps a persistent ORT session by design; explicitly invalidate it
  // so recovery cannot hand the renderer the same dead shared device again.
  try { neuralEng?.invalidateDevice?.(lostDevice); } catch {}
  if (mode !== "off" || optImages) scheduleDeviceRecovery(generation, lostDevice, 0);
}

function scheduleDeviceRecovery(generation, lostDevice, attempt) {
  if (generation !== deviceRecoveryGeneration || (mode === "off" && !optImages)) return;
  const promise = recoverDevice(generation, lostDevice, attempt).finally(() => {
    if (deviceRecoveryPromise === promise) deviceRecoveryPromise = null;
  });
  deviceRecoveryPromise = promise;
}

async function recoverDevice(generation, lostDevice, attempt) {
  try {
    try { await neuralEng?.invalidateDevice?.(lostDevice); } catch {}
    if (generation !== deviceRecoveryGeneration || (mode === "off" && !optImages)) return false;

    if (mode === "upscale" && engine === "neural") {
      const neuralSelection = engineSelectionGeneration;
      await ensureNeural(neuralSelection, { preserveModeOnAdoptionFailure: true });
      if (!neuralSelectionCurrent(neuralSelection)) return false;
    } else {
      if (!(await initWebGPU()) || !device) throw new Error("WebGPU reinitialization failed");
      if (mode === "upscale") {
        await loadModels();
        if (engine === "artcnn") await ensureArtStages(artVariant, chainDepth);
        if (engine === "fsrcnnx-hi") await ensureHiStages(chainDepth);
      }
    }

    if (generation !== deviceRecoveryGeneration || (mode === "off" && !optImages)) return false;
    if (optImages) (await ensureImageUpscaler())?.start();
    if (mode !== "off") {
      if (canvas) canvas.style.display = "block";
      attach();
      scheduleMainLoop();
    }
    log(`WebGPU recovered after device loss${attempt ? ` (attempt ${attempt + 1})` : ""}`);
    return true;
  } catch (error) {
    if (generation !== deviceRecoveryGeneration || (mode === "off" && !optImages)) return false;
    warn(`device recovery attempt ${attempt + 1} failed:`, error.message);
    if (attempt < 2) {
      const delay = 250 * Math.pow(2, attempt);
      deviceRecoveryTimer = setTimeout(() => {
        deviceRecoveryTimer = null;
        scheduleDeviceRecovery(generation, lostDevice, attempt + 1);
      }, delay);
    } else if (mode !== "off") {
      warn("device recovery exhausted; restoring the original video");
      deactivateRendering({ persist: false });
    }
    return false;
  }
}

async function initWebGPUInternal() {
  if (device) return true;
  if (!("gpu" in navigator)) { warn("no WebGPU"); return false; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { warn("no adapter"); return false; }
  const feats = [];
  if (adapter.features.has("float32-filterable")) feats.push("float32-filterable");
  const requestedDevice = await adapter.requestDevice({ requiredFeatures: feats });
  // Another provider may have supplied a shared device while the adapter request
  // was pending. Keep that newer device and release the redundant one we own.
  if (device) {
    try { requestedDevice.destroy?.(); } catch {}
    return true;
  }
  device = requestedDevice;
  deviceOwnedByMain = true;
  watchDeviceLoss(requestedDevice);

  try {
    ensureCanvas();
    buildCore();
  } catch (error) {
    if (device === requestedDevice) {
      invalidateMainDeviceResources();
      device = null;
      deviceOwnedByMain = false;
    }
    try { requestedDevice.destroy?.(); } catch {}
    throw error;
  }
  log("WebGPU ready", adapter.info ? `(${adapter.info.vendor} ${adapter.info.architecture})` : "");
  return true;
}

// Create everything device-bound that initWebGPU used to build inline — reusable by
// adoptChainDevice when the upscaler rebuilds on ORT's device for RIFE chaining.
function buildCore() {
  context = canvas.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  // COPY_SRC lets the interpolation chain tap copy the finished upscaled frame.
  context.configure({ device, format, alphaMode: "premultiplied", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  extractPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: LUMA_EXTRACT_WGSL }), entryPoint: "main" },
  });
  const rmod = device.createShaderModule({ code: RECOMBINE_WGSL });
  recombinePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: rmod, entryPoint: "vs" },
    fragment: { module: rmod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  recombine16Pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: rmod, entryPoint: "vs" },
    fragment: { module: rmod, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });
  // INVERTED CHAIN (#4): tex-ingest pipeline twins are built LAZILY on first
  // inverted-mode use (ensureTexPipelines) — building them here added shader
  // compiles to the adoption warmup and tripped the RIFE fallback evaluator in
  // the NORMAL chain (v0.48.0, 1.34x). Normal path stays byte-identical to
  // v0.47.2 behavior; inverted pays a one-time compile hitch at toggle-on.
  const blitMod = device.createShaderModule({ code: BLIT_WGSL });
  blitPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: blitMod, entryPoint: "vs" },
    fragment: { module: blitMod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  ssimds = new SsimDownscaler(device);
}

// STAGE 2 RIFE CHAINING: rebuild the entire upscaler on an external device (ORT's,
// which cannot be replaced) so upscaled textures are consumable by RIFE inference.
// Tears down EVERY device-bound cache (a single missed object from the old device
// throws "resource from different device" at render), swaps devices, rebuilds core,
// and reloads models. Renders are paused via `adopting` during the swap.
let adopting = false, adoptionPromise = null, adoptionTarget = null;
let adoptionGeneration = 0;
export function adoptChainDevice(extDevice, isRequestCurrent = null, { preserveModeOnFailure = false } = {}) {
  if (!extDevice) return Promise.resolve(false);
  if (adoptionPromise) {
    if (adoptionTarget === extDevice) {
      // A same-device caller may be newer than the request which started this
      // adoption. Do not inherit that older caller's cancellation result: once
      // the shared attempt settles, re-check the newer predicate and retry the
      // target once when it was not adopted.
      const pending = adoptionPromise;
      return pending.catch(() => false).then((adopted) => {
        if (adopted) return true;
        if (typeof isRequestCurrent === "function" && !isRequestCurrent()) return false;
        return adoptChainDevice(extDevice, isRequestCurrent, { preserveModeOnFailure });
      });
    }
    const pending = adoptionPromise;
    return pending.catch(() => false).then(() => {
      if (typeof isRequestCurrent === "function" && !isRequestCurrent()) return false;
      return adoptChainDevice(extDevice, isRequestCurrent, { preserveModeOnFailure });
    });
  }
  if (device === extDevice) return Promise.resolve(true); // already unified
  adoptionTarget = extDevice;
  adoptionGeneration++;
  const promise = adoptChainDeviceInternal(extDevice, isRequestCurrent, { preserveModeOnFailure }).finally(() => {
    if (adoptionPromise === promise) {
      adoptionPromise = null;
      adoptionTarget = null;
    }
  });
  adoptionPromise = promise;
  return promise;
}

async function adoptChainDeviceInternal(extDevice, isRequestCurrent, { preserveModeOnFailure = false } = {}) {
  const attemptGeneration = adoptionGeneration;
  const requestCurrent = typeof isRequestCurrent === "function" ? isRequestCurrent : () => true;
  const generationCurrent = () => attemptGeneration === adoptionGeneration;
  const attemptCurrent = () => generationCurrent() && requestCurrent();
  if (!attemptCurrent()) return false;
  const old = device;
  const oldOwnedByMain = deviceOwnedByMain;
  const restartImages = !!optImages;
  let swapped = false;
  adopting = true;
  try {
    // Any work already submitted by the renderer or image helper must finish
    // before their device-bound caches are retired. A stopped image job cannot
    // submit new work after its current synchronous section has yielded.
    try { await old?.queue?.onSubmittedWorkDone?.(); } catch {}
    if (!attemptCurrent() || device !== old) return false;
    invalidateMainDeviceResources();
    // ---- swap + rebuild ----
    device = extDevice;
    swapped = true;
    // External devices are owned by ORT (or another chain provider). They must
    // remain alive for persistent sessions that can be resumed after a mode or
    // model switch; only a device requested by initWebGPU() is ours to destroy.
    deviceOwnedByMain = false;
    watchDeviceLoss(extDevice);
    {
      const d = device;
      // PRESENT-PATH CIRCUIT BREAKER: WebGPU validation errors are ASYNC events —
      // no try/catch sees them, so a poisoned inverted pipeline can storm (the
      // 237-error 1080p reattach). Count uncaptured errors on the adopted device;
      // a burst while inverted triggers ONE clean interpolator restart (the
      // proven toggle path); repeated bursts hard-disable inverted with a loud
      // line. Listener does not preventDefault — ORT's own handler still sees
      // its errors.
      if (!d.__fsrcnnxErrHook) {
        d.__fsrcnnxErrHook = true;
        d.addEventListener("uncapturederror", () => {
          const now = performance.now();
          if (now - (_gpuErrWinStart || 0) > 2000) { _gpuErrWinStart = now; _gpuErrCount = 0; }
          _gpuErrCount++;
          if (_gpuErrCount === 6 && chainInverted) {
            if (now - (_invRestartLast || 0) < 60000 && _invRestarts >= 2) {
              warn("inverted chain: repeated GPU error bursts — DISABLING inverted (re-enable via the toggle)");
              interpInvertPref = false; saveSitePrefs();
              scheduleInterpolatorGpuRestart();
              return;
            }
            if (now - (_invRestartLast || 0) > 60000) _invRestarts = 0;
            _invRestarts++; _invRestartLast = now;
            warn(`inverted chain: GPU error burst — clean restart (${_invRestarts}/2 this minute)`);
            scheduleInterpolatorGpuRestart();
          }
        });
      }
    }
    buildCore();
    // Let an already-settled extDevice.lost callback invalidate this attempt even
    // in passthrough mode, where no model-loading await would otherwise yield.
    await Promise.resolve();
    if (mode === "upscale") await loadModels();
    if (!attemptCurrent() || device !== extDevice || lostDevices.has(extDevice)) {
      const error = new Error("device adoption superseded");
      error.code = "DEVICE_ADOPTION_SUPERSEDED";
      throw error;
    }
    if (engine === "artcnn") { try { await ensureArtStages(artVariant, chainDepth); } catch (e) { warn("art stages rebuild failed:", e.message); } }
    if (engine === "fsrcnnx-hi") { try { await ensureHiStages(chainDepth); } catch (e) { warn("hi stages rebuild failed:", e.message); } }
    if (!attemptCurrent() || device !== extDevice || lostDevices.has(extDevice)) {
      const error = new Error("device adoption superseded");
      error.code = "DEVICE_ADOPTION_SUPERSEDED";
      throw error;
    }
    if (restartImages && optImages) {
      try { (await ensureImageUpscaler())?.start(); }
      catch (e) { warn("image upscaler rebuild failed:", e.message); }
    }
    if (!attemptCurrent() || device !== extDevice || lostDevices.has(extDevice)) {
      const error = new Error("device adoption superseded");
      error.code = "DEVICE_ADOPTION_SUPERSEDED";
      throw error;
    }
    // Free only a device this module requested itself. A previously adopted ORT
    // device may still be referenced by a persistent session and is not ours.
    if (oldOwnedByMain) { try { old?.destroy?.(); } catch {} }
    // A user can turn the extension off while asynchronous adoption is rebuilding
    // pipelines. Respect that newer state instead of resurrecting rendering.
    if (mode !== "off") {
      if (canvas) { canvas.style.display = ""; canvas.style.opacity = "1"; }
      if (!ro) attach(); // detach() nulls ro; re-attach observers exactly once
    }
    log("upscaler ADOPTED shared device (RIFE chain unified)");
    return true;
  } catch (e) {
    warn("adoptChainDevice failed:", e.message);
    // A loss coordinator or newer adoption owns global device state now. Never
    // invalidate or overwrite its replacement from this stale continuation.
    if (!generationCurrent() || (swapped && device !== extDevice)) {
      if (oldOwnedByMain) { try { old?.destroy?.(); } catch {} }
      return false;
    }
    // The old device has not been destroyed until the success path. Rebuild its
    // caches so a failed external-device adoption degrades back to the prior
    // renderer instead of leaving a half-configured device behind.
    try {
      invalidateMainDeviceResources();
      device = old;
      deviceOwnedByMain = oldOwnedByMain;
      if (!device || lostDevices.has(device)) throw new Error("no healthy previous device available");
      buildCore();
      if (mode === "upscale") await loadModels();
      if (engine === "artcnn") await ensureArtStages(artVariant, chainDepth);
      if (engine === "fsrcnnx-hi") await ensureHiStages(chainDepth);
      if (restartImages && optImages) (await ensureImageUpscaler())?.start();
      warn("external device adoption rolled back to the previous renderer");
    } catch (rollbackError) {
      warn("device adoption rollback failed:", rollbackError.message);
      device = null;
      deviceOwnedByMain = false;
      if (!preserveModeOnFailure) deactivateRendering({ persist: false });
    }
    return false;
  } finally {
    adopting = false;
  }
}

function ensureLumaTexture(w, h) {
  if (!textureSizeAllowed(w, h, "luma input")) return false;
  if (lumaTexture && lumaW === w && lumaH === h) return true;
  const candidate = device.createTexture({
    label: `fsrcnnx-luma-${w}x${h}`,
    size: { width: w, height: h }, format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const old = lumaTexture;
  lumaTexture = candidate;
  lumaW = w; lumaH = h;
  try { old?.destroy?.(); } catch {}
  return true;
}

// Lazily build the texture-ingest twins of the ext-consuming pipelines: same
// shaders, sampling a regular texture_2d (pooled rgba8 frame — real or RIFE
// tween) instead of the live video's texture_external. textureSampleLevel(x, 0.0)
// is the compute-legal analog of textureSampleBaseClampToEdge; the pool textures
// are float-filterable rgba8. One-time cost, paid at inverted-mode activation.
function ensureTexPipelines() {
  if (extractPipelineTex && recombinePipelineTex && recombine16PipelineTex) return true;
  if (!device) return false;
  const texVariant = (w) => w
    .replace(/texture_external/g, "texture_2d<f32>")
    .replace(/textureSampleBaseClampToEdge\(([^)]*)\)/g, "textureSampleLevel($1, 0.0)");
  const candidateExtract = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: texVariant(LUMA_EXTRACT_WGSL) }), entryPoint: "main" },
  });
  const rmodT = device.createShaderModule({ code: texVariant(RECOMBINE_WGSL) });
  const candidateRecombine = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: rmodT, entryPoint: "vs" },
    fragment: { module: rmodT, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const candidateRecombine16 = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: rmodT, entryPoint: "vs" },
    fragment: { module: rmodT, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });
  extractPipelineTex = candidateExtract;
  recombinePipelineTex = candidateRecombine;
  recombine16PipelineTex = candidateRecombine16;
  return true;
}

function renderUpscale() {
  // INVERTED CHAIN (#4): when the interpolator hands us a pooled source-res frame
  // (real or tween), upscale THAT instead of the live video. One-shot per call.
  const _ts = _texSource; _texSource = null;
  const srcW = _ts ? _ts.w : video.videoWidth, srcH = _ts ? _ts.h : video.videoHeight;
  if (!srcW || !srcH) return;
  // In the inverted interpolation chain the pooled texture is the authoritative
  // frame. Falling back to the live video would alternate real decoder frames with
  // interpolated ones and break cadence, so every early-out presents that texture.
  const renderFallback = () => _ts
    ? renderTexturePassthrough(_ts.tex, srcW, srcH)
    : renderPassthrough();

  // Determine the target width (device pixels) we're scaling toward.
  // In fullscreen the element rect can lag the transition, so use the screen.
  const dpr = window.devicePixelRatio || 1;
  const fs = document.fullscreenElement != null;
  const targetW = fs
    ? Math.round(window.screen.width * dpr)
    : Math.round(video.getBoundingClientRect().width * dpr);

  // Engine + model selection.
  if (engine === "fsrcnnx-hi") {
    // High-quality FSRCNNX_x2_56-16-4-1 (2x-only). Single 2x, or chained
    // 2x->2x (4x) / 2x->2x->2x (8x). display/auto run a single stage and let the
    // display-fit/SSimDS path scale the rest.
    const want = targetW / srcW;
    const shouldRun =
      upscalePolicy.startsWith("force") ? true :
      upscalePolicy === "auto" ? want > 1.3 :
      want > 1.05; // display
    const depth = upscalePolicy.startsWith("force") ? chainDepth : 1;
    activeModel = null;
    if (shouldRun) {
      const stages = hiStages.slice(0, depth);
      if (stages.length === depth) {
        if (depth === 1) {
          activeModel = stages[0];
        } else {
          if (!chainedHi || chainedHi.stages.length !== depth || chainedHi.stages[0] !== stages[0]) chainedHi = new ChainedFsrcnnx(stages);
          activeModel = chainedHi;
        }
      }
    }
    if (shouldRun && !activeModel) {
      if (!hiLoadPending) {
        hiLoadPending = true;
        ensureHiStages(depth)
          .then(() => { hiLoadPending = false; log(`FSRCNNX-high ${Math.pow(2, depth)}x ready`); })
          .catch((e) => { hiLoadPending = false; warn("FSRCNNX-high load FAILED:", e.message); });
      }
      renderFallback();
      return;
    }
  } else if (engine === "artcnn") {
    // ArtCNN is a fixed 2x luma doubler. force* => always run; display/auto gate on ratio.
    const want = targetW / srcW;
    const shouldRun =
      upscalePolicy.startsWith("force") ? true :
      upscalePolicy === "auto" ? want > 1.3 :
      want > 1.05; // display
    const depth = upscalePolicy.startsWith("force") ? chainDepth : 1;
    activeModel = null;
    if (shouldRun) {
      const stages = (artStages[artVariant] || []).slice(0, depth);
      if (stages.length === depth) {
        if (depth === 1) {
          activeModel = stages[0];
        } else {
          if (!chainedArt || chainedArt.stages.length !== depth || chainedArt.stages[0] !== stages[0]) chainedArt = new ChainedArtCnn(stages);
          activeModel = chainedArt;
        }
      }
    }
    if (shouldRun && !activeModel) {
      if (!artLoadPending) {
        artLoadPending = true;
        ensureArtStages(artVariant, depth)
          .then(() => { artLoadPending = false; log(`ArtCNN ${artVariant} ${Math.pow(2, depth)}x ready`); })
          .catch((e) => { artLoadPending = false; warn("ArtCNN load FAILED:", e.message); });
      }
      renderFallback();
      return;
    }
  } else {
    switch (upscalePolicy) {
      case "force2": activeModel = models[0]; break;
      case "force3": activeModel = models[1]; break;
      case "force4": activeModel = models[2]; break;
      case "display": {
        const want = targetW / srcW;
        activeModel = want > 1.05
          ? (want >= 3.4 ? models[2] : want >= 2.4 ? models[1] : models[0])
          : null;
        break;
      }
      case "auto":
      default:
        activeModel = selectModel(models, targetW, srcW);
    }
  }


  // Model-owned intermediates can be much larger than the source or final
  // output (ArtCNN packs one source pixel into a 4x2 block). Preflight the exact
  // manifest allocation before touching GPU state. Automatic policies may step
  // down to a smaller standard model; explicit force/engine choices bypass.
  if (activeModel && !modelFitsProcessingBudget(activeModel, srcW, srcH,
      `${engine} ${activeModel.scale || "?"}x`)) {
    if (engine === "fsrcnnx" && !upscalePolicy.startsWith("force")) {
      activeModel = models
        .filter((candidate) => candidate !== activeModel && candidate.scale < activeModel.scale)
        .sort((left, right) => right.scale - left.scale)
        .find((candidate) => modelFitsProcessingBudget(candidate, srcW, srcH,
          `${engine} ${candidate.scale}x`)) || null;
    } else {
      resetScaleSelection();
      activeModel = null;
    }
  }

  // SCALE STABILITY: the target width tracks the element rect; near a model
  // threshold, small layout wiggles flip the scale every frame — which resizes the
  // chain tap and thrashes the interpolator (visible size/jitter oscillation).
  // Debounce: a different selection must persist ~1.5s before it applies.
  {
    const now = performance.now();
    if (_scaleHeld === undefined) { _scaleHeld = null; _scalePending = null; _scalePendingSince = 0; }
    if (_scaleHeld === null) { _scaleHeld = activeModel; _scalePending = null; _scaleHeldSrcW = srcW; _scaleHeldSrcH = srcH; }
    else if (activeModel !== _scaleHeld) {
      // While the interpolation chain is tapped, the display rect can flap due to
      // the page's own layout churn (observed: YouTube under load flipping the video
      // rect every few seconds) — LOCK the scale; only a SOURCE dimension change
      // (real quality switch) re-opens selection.
      if (chainTapOn && srcW === _scaleHeldSrcW && srcH === _scaleHeldSrcH) {
        if (!_scaleLockLogged) { log(`scale locked while chained (rect flap suppressed; targetW=${targetW})`); _scaleLockLogged = true; }
        activeModel = _scaleHeld;
      } else {
        if (_scalePending !== activeModel) { _scalePending = activeModel; _scalePendingSince = now; }
        if (now - _scalePendingSince >= 1500 || srcW !== _scaleHeldSrcW || srcH !== _scaleHeldSrcH) {
          log(`scale switch ${_scaleHeld && _scaleHeld.scale ? _scaleHeld.scale + "x" : "off"} → ${activeModel && activeModel.scale ? activeModel.scale + "x" : "off"} (targetW=${targetW}, src=${srcW})`);
          _scaleHeld = activeModel; _scalePending = null; _scaleHeldSrcW = srcW; _scaleHeldSrcH = srcH; _scaleLockLogged = false;
        } else {
          activeModel = _scaleHeld; // hold the current choice until the change persists
        }
      }
    } else { _scalePending = null; _scaleLockLogged = false; }
  }

  // Hysteresis may temporarily restore a held selection. Revalidate the final
  // model so a previously accepted generation can never bypass a newer budget
  // decision while the chain scale is locked.
  if (activeModel && !modelFitsProcessingBudget(activeModel, srcW, srcH,
      `${engine} ${activeModel.scale || "?"}x`)) {
    resetScaleSelection();
    activeModel = null;
  }

  if (!activeModel) {
    renderFallback();
    return;
  }
  const scale = activeModel.scale;
  const outW = srcW * scale, outH = srcH * scale;
  if (!textureSizeAllowed(srcW, srcH, "upscale input") ||
      !textureSizeAllowed(outW, outH, `${engine} output`)) {
    resetScaleSelection();
    activeModel = null;
    lastSSimDS = false;
    renderFallback();
    return;
  }

  if (!positionCanvas(outW, outH) || !ensureLumaTexture(srcW, srcH)) return;
  activeModel.allocate(srcW, srcH, lumaTexture);

  const ext = _ts ? null : safeImportExternal();
  if (!_ts && !ext) return;
  const srcRes = _ts ? _ts.tex.createView() : ext;
  const pExtract = _ts ? extractPipelineTex : extractPipeline;
  const pRecombine = _ts ? recombinePipelineTex : recombinePipeline;
  const pRecombine16 = _ts ? recombine16PipelineTex : recombine16Pipeline;
  const enc = device.createCommandEncoder();

  // 1. extract luma (BT.709)
  {
    const bg = device.createBindGroup({
      layout: pExtract.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: srcRes },
        { binding: 2, resource: lumaTexture.createView() },
      ],
    });
    const cp = enc.beginComputePass();
    cp.setPipeline(pExtract);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(Math.ceil(srcW / 8), Math.ceil(srcH / 8));
    cp.end();
  }

  // 2. FSRCNNX chain -> upscaled luma
  const hiLuma = activeModel.run(enc, lumaTexture);

  // Decide whether to apply SSimDownscaler: only when the FSRCNNX output
  // overshoots the display box (downscaling the oversized result is where
  // SSimDownscaler helps; otherwise the canvas just shows the hi-res directly).
  const fs2 = document.fullscreenElement != null;
  const dispW = Math.max(1, fs2 ? Math.round(window.screen.width * dpr)
                                : Math.round(video.getBoundingClientRect().width * dpr));
  const dispH = Math.max(1, Math.round(dispW * outH / outW));
  const displaySafe = textureSizeAllowed(dispW, dispH, "display output");
  const overshoot = displaySafe && ssimdsEnabled && outW > dispW * 1.05;
  lastSSimDS = overshoot;

  if (overshoot) {
    // recombine -> offscreen hi-res RGB, then SSimDownscaler -> display-res texture
    if (!ensureHiRGB(outW, outH)) return;
    {
      const bg = device.createBindGroup({
        layout: pRecombine16.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: srcRes },
          { binding: 2, resource: hiLuma.createView() },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: hiRGB.createView(), loadOp: "clear", clearValue: { r:0,g:0,b:0,a:1 }, storeOp: "store" }],
      });
      rp.setPipeline(pRecombine16);
      rp.setBindGroup(0, bg);
      rp.draw(3);
      rp.end();
    }
    ssimds.prepare(outW, outH, dispW, dispH, hiRGB);
    const dsOut = ssimds.run(enc, hiRGB);
    if (!positionCanvas(dispW, dispH)) return;
    finalizeToCanvas(enc, dsOut);
  } else if (debandEnabled || sharpenEnabled) {
    // recombine -> offscreen hi-res RGB, then filters -> canvas at hi-res.
    if (!ensureHiRGB(outW, outH)) return;
    const bg = device.createBindGroup({
      layout: pRecombine16.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: srcRes },
        { binding: 2, resource: hiLuma.createView() },
      ],
    });
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: hiRGB.createView(), loadOp: "clear", clearValue: { r:0,g:0,b:0,a:1 }, storeOp: "store" }],
    });
    rp.setPipeline(pRecombine16);
    rp.setBindGroup(0, bg);
    rp.draw(3);
    rp.end();
    if (!positionCanvas(outW, outH)) return;
    finalizeToCanvas(enc, hiRGB);
  } else {
    // no overshoot or filters: recombine straight to canvas at hi-res.
    if (!positionCanvas(outW, outH)) return;
    const bg = device.createBindGroup({
      layout: pRecombine.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: srcRes },
        { binding: 2, resource: hiLuma.createView() },
      ],
    });
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    });
    rp.setPipeline(pRecombine);
    rp.setBindGroup(0, bg);
    rp.draw(3);
    rp.end();
  }

  device.queue.submit([enc.finish()]);
}

// Final stage: take an rgba16float RGB texture and put it on the canvas, applying
// debanding and/or adaptive sharpen as enabled. Order: deband -> sharpen -> canvas
// (smooth banding first, then sharpen detail). Uses an intermediate texture only
// when both run.
function finalizeToCanvas(enc, srcTex) {
  let cur = srcTex;
  // 1. deband (optional) -> intermediate (if sharpen follows) or canvas
  if (debandEnabled) {
    ensureDebandPipelines();
    // update time uniform for grain/temporal variation
    const t = (performance.now() % 100000) / 1000;
    device.queue.writeBuffer(debandTimeBuf, 0, new Float32Array([t]));
    if (sharpenEnabled) {
      if (!ensureDebandInter(canvas.width, canvas.height)) return;
      const bg = device.createBindGroup({
        layout: debandFloatPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: cur.createView() },
          { binding: 2, resource: { buffer: debandTimeBuf } },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: debandInterTex.createView(), loadOp: "clear", clearValue: { r:0,g:0,b:0,a:1 }, storeOp: "store" }],
      });
      rp.setPipeline(debandFloatPipeline); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
      cur = debandInterTex;
    } else {
      // deband straight to canvas
      const bg = device.createBindGroup({
        layout: debandCanvasPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: cur.createView() },
          { binding: 2, resource: { buffer: debandTimeBuf } },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", clearValue: { r:0,g:0,b:0,a:1 }, storeOp: "store" }],
      });
      rp.setPipeline(debandCanvasPipeline); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
      return;
    }
  }
  // 2. sharpen or blit -> canvas
  ensureSharpenPipeline();
  const pipe = sharpenEnabled ? sharpenPipeline : blitPipeline;
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: sampler }, { binding: 1, resource: cur.createView() }],
  });
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", clearValue: { r:0,g:0,b:0,a:1 }, storeOp: "store" }],
  });
  rp.setPipeline(pipe);
  rp.setBindGroup(0, bg);
  rp.draw(3);
  rp.end();
}


// ---- neural engine (v0.49.0) ----------------------------------------------
// ONNX SR models (SPAN/RealPLKSR/DAT2/ATD...) via a second ORT session on the
// shared device. Output is full RGB at model scale in an rgba16float texture;
// present reuses the existing tail (SSimDS overshoot doctrine, deband/sharpen,
// canvas) — recombine is bypassed since chroma is neural.
function pauseInterpolationForNeural() {
  if (!optInterpolate) return;
  interpolationSelectionGeneration++;
  interpPausedByNeural = true;
  if (interpolator) interpolator.stop();
  log("neural engine v1: frame interpolation paused for this session; preference preserved");
}

function resumeInterpolationAfterNeural() {
  if (!interpPausedByNeural) return;
  interpPausedByNeural = false;
  if (!optInterpolate) return;
  setInterpolate(true).catch((e) => warn("interpolation resume after neural failed:", e.message));
}

function neuralSelectionCurrent(expectedSelection) {
  return expectedSelection === engineSelectionGeneration && engine === "neural";
}

function neuralSupersededError() {
  const error = new Error("neural activation superseded");
  error.code = "NEURAL_SUPERSEDED";
  return error;
}

async function ensureNeural(
  expectedSelection = engineSelectionGeneration,
  { preserveModeOnAdoptionFailure = false } = {},
) {
  if (!neuralSelectionCurrent(expectedSelection)) throw neuralSupersededError();
  if (!neuralEng) neuralEng = createNeuralEngine({ log, warn });
  // v1: neural + interpolation are mutually exclusive. This is a runtime pause,
  // not a settings change: leaving neural restores the user's prior preference.
  pauseInterpolationForNeural();
  const entry = await neuralEng.init(neuralModelKey || undefined);
  if (!neuralSelectionCurrent(expectedSelection)) throw neuralSupersededError();
  neuralModelKey = entry.key;
  const d = neuralEng.device();
  if (d && d !== device) {
    log("neural: adopting shared ORT device (upscaler + neural unified)");
    const adopted = await adoptChainDevice(
      d,
      () => neuralSelectionCurrent(expectedSelection),
      { preserveModeOnFailure: preserveModeOnAdoptionFailure },
    );
    if (!neuralSelectionCurrent(expectedSelection)) throw neuralSupersededError();
    if (!adopted) {
      const error = new Error("neural shared-device adoption failed");
      error.code = "NEURAL_ADOPTION_FAILED";
      throw error;
    }
  }
  if (!neuralSelectionCurrent(expectedSelection)) throw neuralSupersededError();
  neuralFail = 0;
  return entry;
}

export async function setNeuralModel(key) {
  neuralModelKey = key || "";
  resetScaleSelection();
  if (engine === "neural") {
    const selectionGeneration = engineSelectionGeneration;
    try { await ensureNeural(selectionGeneration); }
    catch (e) {
      if (e.code !== "NEURAL_SUPERSEDED") warn("neural model switch failed:", e.message);
      return { ok: false, error: e.message };
    }
  }
  saveSitePrefs();
  return { ok: true, model: neuralModelKey };
}

// Shared present tail for any finished rgba16float RGB texture: SSimDS when
// the result overshoots the display box, then deband/sharpen/blit to canvas.
// Additive extraction — renderUpscale keeps its own battle-tested inline copy.
function presentHiRGBTexture(tex, outW, outH) {
  if (!device || !canvas || !context || !textureSizeAllowed(outW, outH, "neural output")) return false;
  const dpr = window.devicePixelRatio || 1;
  const fs2 = document.fullscreenElement != null;
  const dispW = Math.max(1, fs2 ? Math.round(window.screen.width * dpr)
                                : Math.round(video.getBoundingClientRect().width * dpr));
  const dispH = Math.max(1, Math.round(dispW * outH / outW));
  const overshoot = textureSizeAllowed(dispW, dispH, "neural display output") &&
    ssimdsEnabled && ssimds && outW > dispW * 1.05;
  lastSSimDS = overshoot;
  const enc = device.createCommandEncoder();
  if (overshoot) {
    ssimds.prepare(outW, outH, dispW, dispH, tex);
    const dsOut = ssimds.run(enc, tex);
    if (!positionCanvas(dispW, dispH)) return false;
    finalizeToCanvas(enc, dsOut);
  } else {
    if (!positionCanvas(outW, outH)) return false;
    finalizeToCanvas(enc, tex);
  }
  device.queue.submit([enc.finish()]);
  return true;
}

function renderNeuralFrame() {
  if (!neuralEng || !neuralEng.ready()) { renderPassthrough(); return; }
  if (neuralBusy) { neuralEng.bumpSkip(); return; } // queue-of-1: drop, keep last output
  const nd = neuralEng.device();
  if (nd && nd !== device) { // upscaler device predates/postdates the ORT device — unify first
    if (!adopting) adoptChainDevice(nd).catch((e) => warn("neural device adopt failed:", e.message));
    neuralEng.bumpSkip(); return;
  }
  const srcW = video.videoWidth, srcH = video.videoHeight;
  if (!srcW || !srcH) return;
  const entry = neuralEng.activeEntry();
  const scale = Number(entry?.scale) || 1;
  const padMultiple = Math.max(1, Math.trunc(Number(entry?.padMultiple) || 1));
  const padW = Math.ceil(srcW / padMultiple) * padMultiple;
  const padH = Math.ceil(srcH / padMultiple) * padMultiple;
  if (!textureSizeAllowed(srcW, srcH, "neural input") ||
      !textureSizeAllowed(padW, padH, "neural padded input") ||
      !textureSizeAllowed(srcW * scale, srcH * scale, "neural output") ||
      !textureSizeAllowed(padW * scale, padH * scale, "neural padded output") ||
      !storageBufferSizeAllowed(padW * padH * 3 * 4, "neural input buffer") ||
      !storageBufferSizeAllowed(padW * scale * padH * scale * 3 * 4, "neural output buffer")) {
    renderPassthrough();
    return;
  }
  const ext = safeImportExternal();
  if (!ext) return;
  const runDevice = device;
  const runEngine = neuralEng;
  neuralBusy = true;
  runEngine.run(ext, srcW, srcH).then((res) => {
    if (res && device === runDevice && neuralEng === runEngine &&
        mode === "upscale" && engine === "neural") {
      presentHiRGBTexture(res.tex, res.outW, res.outH);
    }
    if (mode === "upscale" && engine === "neural") neuralFail = 0;
  }).catch((e) => {
    // stop(), a mode change, or an engine change deliberately invalidates the
    // run. Its rejection is lifecycle control, not an inference failure.
    if (mode !== "upscale" || engine !== "neural" || /cancelled by stop/i.test(e.message)) return;
    neuralFail++;
    warn(`neural inference failed (${neuralFail}/3):`, e.message);
    if (neuralFail >= 3) {
      warn("neural engine failing repeatedly — reverting to FSRCNNX for this session (site pref NOT overwritten)");
      engineSelectionGeneration++;
      engine = "fsrcnnx";
      try { neuralEng?.stop?.(); } catch {}
      resumeInterpolationAfterNeural();
    }
  }).finally(() => { neuralBusy = false; });
}

function ensureDebandInter(w, h) {
  if (!textureSizeAllowed(w, h, "deband intermediate")) return false;
  if (debandInterTex && debandInterW === w && debandInterH === h) return true;
  const candidate = device.createTexture({
    size: { width: w, height: h }, format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const old = debandInterTex;
  debandInterTex = candidate;
  debandInterW = w; debandInterH = h;
  try { old?.destroy?.(); } catch {}
  return true;
}

function ensureDebandPipelines() {
  if (debandCanvasPipeline && debandFloatPipeline && debandStrengthBuilt === debandStrength) return;
  let candidateTimeBuf = debandTimeBuf;
  let createdTimeBuf = null;
  let candidateCanvas, candidateFloat;
  try {
    if (!candidateTimeBuf) {
      createdTimeBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      candidateTimeBuf = createdTimeBuf;
    }
    const mod = device.createShaderModule({ code: buildDebandShader(debandStrength) });
    const descriptor = (targetFormat) => ({
      layout: "auto",
      vertex: { module: mod, entryPoint: "vs" },
      fragment: { module: mod, entryPoint: "fs", targets: [{ format: targetFormat }] },
      primitive: { topology: "triangle-list" },
    });
    candidateCanvas = device.createRenderPipeline(descriptor(format));
    candidateFloat = device.createRenderPipeline(descriptor("rgba16float"));
  } catch (error) {
    try { createdTimeBuf?.destroy?.(); } catch {}
    throw error;
  }
  debandTimeBuf = candidateTimeBuf;
  debandCanvasPipeline = candidateCanvas;
  debandFloatPipeline = candidateFloat;
  debandStrengthBuilt = debandStrength;
}

function ensureSharpenPipeline() {
  if (sharpenPipeline && sharpenStrengthBuilt === sharpenStrength) return;
  const mod = device.createShaderModule({ code: buildSharpenShader(sharpenStrength, false) });
  sharpenPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  sharpenStrengthBuilt = sharpenStrength;
}

const BLIT_WGSL = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
struct VsOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VsOut{
  var p=array<vec2f,3>(vec2f(-1.,-3.),vec2f(-1.,1.),vec2f(3.,1.));
  var uv=array<vec2f,3>(vec2f(0.,2.),vec2f(0.,0.),vec2f(2.,0.));
  var o:VsOut; o.pos=vec4f(p[i],0.,1.); o.uv=uv[i]; return o;
}
@fragment fn fs(@location(0) uv:vec2f)->@location(0) vec4f{
  return textureSampleLevel(src, samp, uv, 0.0);
}`;

function ensureHiRGB(w, h) {
  if (!textureSizeAllowed(w, h, "RGB intermediate")) return false;
  if (hiRGB && hiRGBW === w && hiRGBH === h) return true;
  const candidate = device.createTexture({
    label: `fsrcnnx-hiRGB-${w}x${h}`,
    size: { width: w, height: h }, format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const old = hiRGB;
  hiRGB = candidate;
  hiRGBW = w; hiRGBH = h;
  try { old?.destroy?.(); } catch {}
  return true;
}

function ensureChainTapTexture(w, h) {
  if (chainTapTex && chainTapTex.width === w && chainTapTex.height === h) return chainTapTex;
  if (!textureSizeAllowed(w, h, "interpolation chain tap")) {
    throw new Error("chain tap dimensions exceed limits");
  }
  const candidate = device.createTexture({
    size: [w, h], format,
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const old = chainTapTex;
  chainTapTex = candidate;
  try { old?.destroy?.(); } catch {}
  return candidate;
}

const PASSTHROUGH_WGSL = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var frame : texture_external;
struct VsOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VsOut{
  var p=array<vec2f,3>(vec2f(-1.,-3.),vec2f(-1.,1.),vec2f(3.,1.));
  var uv=array<vec2f,3>(vec2f(0.,2.),vec2f(0.,0.),vec2f(2.,0.));
  var o:VsOut; o.pos=vec4f(p[i],0.,1.); o.uv=uv[i]; return o;
}
@fragment fn fs(@location(0) uv:vec2f)->@location(0) vec4f{
  return textureSampleBaseClampToEdge(frame, samp, uv);
}`;

function renderPassthrough() {
  if (!passthroughPipeline) {
    const m = device.createShaderModule({ code: PASSTHROUGH_WGSL });
    passthroughPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs" },
      fragment: { module: m, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
  }
  if (!textureSizeAllowed(video.videoWidth, video.videoHeight, "passthrough output")) {
    if (canvas) canvas.style.display = "none";
    return;
  }
  if (!positionCanvas(video.videoWidth, video.videoHeight)) return;
  const ext = safeImportExternal();
  if (!ext) return;
  const enc = device.createCommandEncoder();
  const bg = device.createBindGroup({
    layout: passthroughPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: sampler }, { binding: 1, resource: ext }],
  });
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
  });
  rp.setPipeline(passthroughPipeline);
  rp.setBindGroup(0, bg);
  rp.draw(3);
  rp.end();
  device.queue.submit([enc.finish()]);
}

// Safe presentation fallback for the inverted interpolation chain. The source is
// already a persistent same-device rgba texture, so presenting it directly keeps
// interpolation cadence intact when an upscale model is unavailable or its output
// would exceed adapter/resource limits.
function renderTexturePassthrough(tex, width, height) {
  if (!tex || !textureSizeAllowed(width, height, "interpolated fallback") ||
      !positionCanvas(width, height)) return false;
  const enc = device.createCommandEncoder();
  const bg = device.createBindGroup({
    layout: blitPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: tex.createView() },
    ],
  });
  const rp = enc.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: "store",
    }],
  });
  rp.setPipeline(blitPipeline);
  rp.setBindGroup(0, bg);
  rp.draw(3);
  rp.end();
  device.queue.submit([enc.finish()]);
  return true;
}

let frameTimes = [];
function cancelMainLoop() {
  if (rvfc == null) return;
  const id = rvfc, kind = rvfcKind;
  rvfc = null; rvfcKind = null;
  try {
    if (kind === "video" && video?.cancelVideoFrameCallback) video.cancelVideoFrameCallback(id);
    else cancelAnimationFrame(id);
  } catch {}
}

function scheduleMainLoop() {
  if (mode === "off" || !video || rvfc != null) return;
  if (typeof video.requestVideoFrameCallback === "function") {
    rvfcKind = "video";
    rvfc = video.requestVideoFrameCallback(loop);
  } else {
    rvfcKind = "animation";
    rvfc = requestAnimationFrame(loop);
  }
}

// One shutdown path keeps callback, observer, overlay, and badge state aligned.
// Protected-source callers deliberately do not persist "off", preserving the
// requested site mode for a future processable source.
function deactivateRendering({ persist = true, protectedFailure = false } = {}) {
  cancelDeviceRecovery();
  adoptionGeneration++;
  modeSelectionGeneration++;
  mode = "off";
  cancelMainLoop();
  try { neuralEng?.stop?.(); } catch {}
  chainTap(false);
  chainInverted = false;
  _texSource = null;
  activeModel = null;
  clearMultiTargets();
  detach();
  if (canvas) { canvas.style.display = "none"; canvas.style.opacity = "1"; }
  if (persist) saveSitePrefs();
  if (protectedFailure) notifyProtected(); else notifyState();
}

function loop() {
  // A scheduled callback is single-use. Clear its handle before doing work so
  // scheduleMainLoop() can register exactly one successor.
  rvfc = null; rvfcKind = null;
  if (mode === "off" || !video) return;
  // Mid-playback protection guard: a source can start unprotected then switch to
  // DRM (e.g. navigating to protected content in an SPA). Re-check occasionally.
  if (mode !== "off" && frameCount > 0 && frameCount % 300 === 0) {
    if (isTaintedVideo(video)) {
      warn("source became DRM-protected mid-playback; disabling.");
      protectedSource = true;
      protectedReason = probeVideo(video);
      deactivateRendering({ persist: false, protectedFailure: true });
      return;
    }
  }
  const t0 = performance.now();
  try {
    if (adopting) { // device swap in progress — skip this frame, keep the loop alive
      scheduleMainLoop();
      return;
    }
    if (chainInverted) { // #4: the interpolator drives upscales (chainUpscaleTex);
      // rendering here too would double-draw. Keep re-registering so leaving
      // inverted mode resumes the normal loop instantly.
      scheduleMainLoop();
      return;
    }
    if (!device) { // deviceless window during interp restart — idle, adoption heals
      scheduleMainLoop();
      return;
    }
    if (mode === "upscale" && engine === "neural") renderNeuralFrame();
    else if (mode === "upscale") renderUpscale();
    else renderPassthrough();
    if (mode === "off") return; // a guarded import may have shut rendering down
    // CHAIN TAP: if the interpolation chain is consuming upscaled frames, copy the
    // finished frame (still valid pre-present within this task) into a persistent
    // texture the interpolator (same device) samples from. Only when tapped.
    if (chainTapOn) {
      try {
        const curTex = context.getCurrentTexture();
        ensureChainTapTexture(curTex.width, curTex.height);
        const enc2 = device.createCommandEncoder();
        enc2.copyTextureToTexture({ texture: curTex }, { texture: chainTapTex }, { width: curTex.width, height: curTex.height });
        device.queue.submit([enc2.finish()]);
        chainTapFrame++;
      } catch (e) { /* tap failure must never break upscaling */ }
    }
    // hover-reveal: fade overlay out while cursor is over the player. While the
    // interpolation chain is tapped, the interp overlay IS the output — hide ours.
    if (canvas) canvas.style.opacity = chainTapOn ? "0" : ((optHoverReveal && hoverHidden) ? "0" : "1");
    // multi-video: reconcile the set of secondary videos periodically (cheap,
    // every ~30 frames) since feed videos appear/disappear as you scroll.
    if (optAllVideos && frameCount % 30 === 0) syncMultiTargets();
    frameCount++;
    const dt = performance.now() - t0;
    frameTimes.push(dt);
    if (frameTimes.length > 120) frameTimes.shift();
    const now = performance.now();
    if (now - lastLog > 2000) {
      lastLog = now;
      const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
      const max = Math.max(...frameTimes);
      log(`mode=${mode} frames=${frameCount} src=${video.videoWidth}x${video.videoHeight}` +
          (mode === "upscale" && activeModel ? ` ${engine==="artcnn"?artVariant.replace("ArtCNN_",""):engine==="fsrcnnx-hi"?"FSRCNNX-hi":"FSRCNNX"} ${activeModel.scale}x out=${video.videoWidth*activeModel.scale}x${video.videoHeight*activeModel.scale}` : "") +
          (mode === "upscale" && engine === "neural" && neuralEng && neuralEng.ready() ? ` NEURAL ${neuralEng.activeEntry()?.label || neuralModelKey} ${neuralEng.activeEntry()?.scale}x mu=${neuralEng.stats().mu.toFixed(1)}ms skip:${neuralEng.stats().skip}` : "") +
          (mode === "upscale" && lastSSimDS ? " +SSimDS" : "") +
          (mode === "upscale" && sharpenEnabled ? ` +Sharpen(${sharpenStrength})` : "") +
          (mode === "upscale" && debandEnabled ? ` +Deband(${debandStrength})` : "") +
          ` | CPU encode avg=${avg.toFixed(1)}ms max=${max.toFixed(1)}ms`);
    }
  } catch (e) {
    warn("render error:", e.message, "\n", e.stack);
    deactivateRendering({ persist: true });
  }
  scheduleMainLoop();
}

let ro = null, mo = null, onScroll = null, onResize = null, onFs = null;
let hoverEl = null, onHoverEnter = null, onHoverLeave = null;
let attachedVideo = null;
function attach() {
  if (attachedVideo === video && ro) return;
  detach();
  attachedVideo = video;
  const reposition = () => positionCanvas(canvas.width, canvas.height);
  ro = new ResizeObserver(reposition);
  if (video?.parentElement) ro.observe(video.parentElement);
  onResize = reposition;
  // scroll matters for the fixed-position (shadow-DOM) overlay, which uses
  // viewport coords; also harmless for the absolute case.
  onScroll = reposition;
  onFs = () => setTimeout(reposition, 50);
  window.addEventListener("resize", onResize, { passive: true });
  window.addEventListener("scroll", onScroll, { passive: true, capture: true });
  document.addEventListener("fullscreenchange", onFs);
  mo = new MutationObserver(reposition);
  mo.observe(document.body, { attributes: true, childList: true, subtree: true, attributeFilter: ["class", "style"] });
  attachHover();
}
// Hover-reveal: when enabled, fade the overlay out while the cursor is over the
// player so the site's native controls (and real video) show through. We listen
// on the player region; for shadow-DOM videos we use the shadow host (the
// nearest light-DOM ancestor), since pointer events over the shadow tree surface
// at the host in the light DOM.
function hoverRegion() {
  if (!video) return null;
  if (inShadowDom(video)) {
    const r = video.getRootNode();
    return (r instanceof ShadowRoot ? r.host : null) || video.parentElement;
  }
  return video.parentElement || video;
}
function attachHover() {
  hoverEl = hoverRegion();
  if (!hoverEl) return;
  onHoverEnter = () => { if (optHoverReveal) hoverHidden = true; };
  onHoverLeave = () => { hoverHidden = false; };
  hoverEl.addEventListener("pointerenter", onHoverEnter, { passive: true });
  hoverEl.addEventListener("pointerleave", onHoverLeave, { passive: true });
  hoverEl.addEventListener("pointermove", onHoverEnter, { passive: true });
}
function detachHover() {
  if (hoverEl) {
    if (onHoverEnter) { hoverEl.removeEventListener("pointerenter", onHoverEnter); hoverEl.removeEventListener("pointermove", onHoverEnter); }
    if (onHoverLeave) hoverEl.removeEventListener("pointerleave", onHoverLeave);
  }
  hoverEl = onHoverEnter = onHoverLeave = null;
  hoverHidden = false;
}
function detach() {
  ro?.disconnect(); mo?.disconnect(); ro = mo = null;
  if (onResize) window.removeEventListener("resize", onResize);
  if (onScroll) window.removeEventListener("scroll", onScroll, { capture: true });
  if (onFs) document.removeEventListener("fullscreenchange", onFs);
  onResize = onScroll = onFs = null;
  attachedVideo = null;
  detachHover();
  // remove the body-level overlay if we created one
  if (canvas && canvas.parentElement === document.body) canvas.remove();
}

// A video can be unprocessable for two distinct reasons, both of which make
// importExternalTexture throw or yield nothing:
//   1. DRM/EME  — video.mediaKeys is attached (Netflix, Disney+, …).
//   2. Cross-origin taint — the <video> was loaded from another origin without
//      CORS (e.g. Reddit's v.redd.it). importExternalTexture refuses tainted
//      video just like a 2D canvas does; we can't add crossOrigin after load.
// We can't always tell these apart cheaply, so probeVideo() returns a reason:
//   "ok" | "drm" | "tainted"
function probeVideo(v) {
  try {
    if (v && "mediaKeys" in v && v.mediaKeys) return "drm";
  } catch {}
  // Probe for cross-origin taint via a tiny 2D-canvas read. A SecurityError
  // here means the video is tainted, which also blocks importExternalTexture.
  try {
    const c = document.createElement("canvas");
    c.width = 4; c.height = 4;
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(v, 0, 0, 4, 4);
    cx.getImageData(0, 0, 4, 4); // throws if tainted (cross-origin without CORS)
    return "ok";
  } catch {
    return "tainted";
  }
}
function isProtectedVideo(v) { return probeVideo(v) !== "ok"; }
function isTaintedVideo(v) { return probeVideo(v) !== "ok"; }

// Guarded importExternalTexture. A tainted/DRM video throws here; rather than let
// that error escape into the render loop every frame, catch it, mark the source
// unprocessable, disable, and surface the reason to the popup. Returns null on
// failure (callers must bail).
function safeImportExternal() {
  // TRANSIENT, not DRM: an interp restart destroys the previously-adopted device
  // (ORT creates a fresh one per session) before re-adoption lands. A null device
  // here must idle the frame — latching protected/off on it caused the v0.48.0
  // black screen (null deref classified as "drm" → mode off + canvas hidden).
  if (!device) return null;
  try {
    return device.importExternalTexture({ source: video });
  } catch (e) {
    const reason = (e && /tainted|cross-origin/i.test(e.message)) ? "tainted" : "drm";
    warn(`importExternalTexture failed (${reason}); disabling.`, e.message);
    protectedSource = true; protectedReason = reason;
    deactivateRendering({ persist: false, protectedFailure: true });
    return null;
  }
}

export async function setMode(next, autoEnableToken = null) {
  if (!["off", "passthrough", "upscale"].includes(next)) return { ok: false, reason: "invalid mode" };
  if (autoEnableToken == null) cancelAutoEnable();
  else if (autoEnableToken !== autoEnableGeneration) return { ok: false, reason: "superseded" };
  const selectionGeneration = ++modeSelectionGeneration;
  if (next === mode && (mode === "off" || device)) {
    // A second enable request can arrive while the first is awaiting neural
    // session creation. The newer request owns the lifecycle, so wait for (or
    // retry) that initialization instead of returning with a permanently
    // unready neural engine.
    if (mode === "upscale" && engine === "neural" && !neuralEng?.ready?.()) {
      const neuralSelection = engineSelectionGeneration;
      try { await ensureNeural(neuralSelection); }
      catch (error) {
        if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
        if (neuralSelection === engineSelectionGeneration && engine === "neural") {
          warn("neural activation failed; using FSRCNNX:", error.message);
          engineSelectionGeneration++;
          engine = "fsrcnnx";
          resumeInterpolationAfterNeural();
          await loadModels();
        }
      }
    }
    if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
    if (mode !== "off") { attach(); scheduleMainLoop(); }
    return { ok: true };
  }
  // If interpolation is running, the chain decision (interpolate upscaled frames vs
  // raw video) is made at interp start — restart it after the mode change so either
  // enable order works.
  const restartInterp = async () => {
    if (optInterpolate && interpolator && engine !== "neural") {
      try { interpolator.stop(); await interpolator.start(); log("interp: restarted for upscale-mode change (chain re-evaluated)"); }
      catch (e) { warn("interp restart failed:", e.message); }
    }
  };
  if (next === "off") {
    deactivateRendering({ persist: true });
    await restartInterp();
    return { ok: true };
  }
  protectedSource = false; protectedReason = null;
  const nextVideo = findVideo();
  if (!nextVideo) return { ok: false, reason: "no video" };
  const probe = probeVideo(nextVideo);
  if (probe !== "ok") {
    protectedSource = true;
    protectedReason = probe;
    warn(`source unprocessable (${probe === "drm" ? "DRM/encrypted" : "cross-origin without CORS"}); not enabling.`);
    return { ok: false, reason: probe };
  }
  if (!(await initWebGPU())) return { ok: false, reason: "WebGPU init failed" };
  if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
  if (next === "upscale") await loadModels();
  if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
  cancelMainLoop();
  if (video !== nextVideo) detach();
  video = nextVideo;
  mode = next;
  if (next === "upscale" && engine === "neural") {
    const neuralSelection = engineSelectionGeneration;
    try {
      await ensureNeural(neuralSelection);
    } catch (error) {
      if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
      if (neuralSelection === engineSelectionGeneration && engine === "neural") {
        warn("neural activation failed; using FSRCNNX:", error.message);
        engineSelectionGeneration++;
        engine = "fsrcnnx";
        resumeInterpolationAfterNeural();
        await loadModels();
      }
    }
  }
  if (selectionGeneration !== modeSelectionGeneration) return { ok: false, reason: "superseded" };
  canvas.style.display = "block";
  attach();
  scheduleMainLoop();
  saveSitePrefs();
  notifyState();
  await restartInterp();
  return { ok: true };
}

export function setEngine(e) {
  const selectionGeneration = ++engineSelectionGeneration;
  const wasNeural = engine === "neural";
  engine = e === "artcnn" ? "artcnn" : e === "fsrcnnx-hi" ? "fsrcnnx-hi" : e === "neural" ? "neural" : "fsrcnnx";
  resetScaleSelection();
  if (engine === "neural") {
    ensureNeural(selectionGeneration).catch((er) => {
      if (selectionGeneration !== engineSelectionGeneration || engine !== "neural") return;
      if (mode === "off" && /initialization cancelled/i.test(er.message)) return;
      warn("neural init failed:", er.message);
      engineSelectionGeneration++;
      engine = "fsrcnnx";
      resumeInterpolationAfterNeural();
    });
  } else if (wasNeural || interpPausedByNeural) {
    try { neuralEng?.stop?.(); } catch {}
    resumeInterpolationAfterNeural();
  }
  chainDepth = (engine === "artcnn" || engine === "fsrcnnx-hi") ? policyToDepth(upscalePolicy) : 1;
  artDiagLogged = false;
  if (engine === "artcnn" && device) {
    ensureArtStages(artVariant, chainDepth).catch((er) => warn("ArtCNN preload failed:", er.message));
  }
  if (engine === "fsrcnnx-hi" && device) {
    ensureHiStages(chainDepth).catch((er) => warn("FSRCNNX-high preload failed:", er.message));
  }
  saveSitePrefs();
  return { ok: true, engine };
}
export function setArtVariant(v) {
  if (ART_FILES.includes(v)) {
    artVariant = v;
    resetScaleSelection();
  }
  artDiagLogged = false;
  if (engine === "artcnn" && device) ensureArtStages(artVariant, chainDepth).catch((er) => warn("ArtCNN preload failed:", er.message));
  saveSitePrefs();
  return { ok: true, artVariant };
}

export function setHoverReveal(on) {
  optHoverReveal = !!on;
  if (!optHoverReveal) hoverHidden = false;
  saveSitePrefs();
  return { ok: true, hoverReveal: optHoverReveal };
}
export function setAllVideos(on) {
  optAllVideos = !!on;
  saveSitePrefs();
  // (re)build the multi-video set on next loop tick if active
  if (mode !== "off") syncMultiTargets();
  return { ok: true, allVideos: optAllVideos };
}
export function setDeband(on) {
  debandEnabled = !!on;
  saveSitePrefs();
  return { ok: true, deband: debandEnabled };
}
export function setDebandStrength(v) {
  debandStrength = Math.max(0.3, Math.min(3.0, Number(v) || 1.0));
  saveSitePrefs();
  return { ok: true, debandStrength };
}

// ---- multi-video support (optAllVideos) ----------------------------------
// The render pipeline uses module-level globals (video/canvas/luma textures…).
// To upscale several videos without rewriting every function, each extra video
// gets a MultiTarget holding its own canvas + per-source GPU intermediates. Each
// frame we "swap in" a target's state into the globals, render, then restore.
// This trades a little bookkeeping for not duplicating the render code.
class MultiTarget {
  constructor(vid) {
    this.video = vid;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "fsrcnnx-overlay-multi";
    Object.assign(this.canvas.style, { position: "absolute", top: "0", left: "0", pointerEvents: "none", zIndex: "10", transition: "opacity 0.18s ease" });
    this.lumaTexture = null; this.lumaW = 0; this.lumaH = 0;
    this.hiRGB = null; this.hiRGBW = 0; this.hiRGBH = 0;
    this.debandInterTex = null; this.debandInterW = 0; this.debandInterH = 0;
    this.ssimds = new SsimDownscaler(device);
    this.sharpenPipeline = null; this.sharpenStrengthBuilt = null;
    // Each canvas needs its OWN GPUCanvasContext configured for the device.
    this.context = this.canvas.getContext("webgpu");
    this.context.configure({ device, format, alphaMode: "premultiplied" });
    // Per-target model instances. Models cache GPU textures keyed to one input
    // size, so sharing the global model across differently-sized videos would
    // thrash (realloc every frame). Each target owns its models, built from the
    // already-fetched manifest+wgsl (construction is cheap; the per-size textures
    // must exist per video anyway). Populated lazily by ensureTargetModels().
    this.models = []; this.hiStages = []; this.artStages = {};
    this.rvfc = null; this.hoverHidden = false; this.neuralBypassLogged = false;
    this.attachHover();
  }
  attachHover() {
    const region = inShadowDom(this.video)
      ? ((this.video.getRootNode() instanceof ShadowRoot ? this.video.getRootNode().host : null) || this.video.parentElement)
      : (this.video.parentElement || this.video);
    if (!region) return;
    this._region = region;
    this._enter = () => { if (optHoverReveal) this.hoverHidden = true; };
    this._leave = () => { this.hoverHidden = false; };
    region.addEventListener("pointerenter", this._enter, { passive: true });
    region.addEventListener("pointermove", this._enter, { passive: true });
    region.addEventListener("pointerleave", this._leave, { passive: true });
  }
  destroy() {
    if (this.rvfc != null && typeof this.video.cancelVideoFrameCallback === "function") {
      try { this.video.cancelVideoFrameCallback(this.rvfc); } catch {}
      this.rvfc = null;
    }
    if (this._region) {
      this._region.removeEventListener("pointerenter", this._enter);
      this._region.removeEventListener("pointermove", this._enter);
      this._region.removeEventListener("pointerleave", this._leave);
    }
    const ownedModels = new Set([
      ...this.models,
      ...this.hiStages,
      ...Object.values(this.artStages).flat(),
    ]);
    for (const model of ownedModels) { try { model?.destroy?.(); } catch {} }
    this.models = []; this.hiStages = []; this.artStages = {};
    this.lumaTexture?.destroy?.(); this.hiRGB?.destroy?.();
    this.debandInterTex?.destroy?.(); this.ssimds?.destroy?.();
    try { this.context?.unconfigure?.(); } catch {}
    this.canvas.remove();
  }
}
let multiTargets = new Map(); // video element -> MultiTarget

// Build per-target model instances from cached sources (lazy, cheap).
function ensureTargetModels(t) {
  if (engine === "neural") return false; // one ORT session/queue currently serves only the primary video
  if (engine === "fsrcnnx") {
    if (t.models.length) return true;
    for (const name of MODEL_FILES) {
      const s = srcCache.fsrcnnx[name];
      if (s) t.models.push(new FsrcnnxModel(device, s.manifest, s.wgsl, { expectedName: name }));
    }
  } else if (engine === "fsrcnnx-hi") {
    const s = srcCache.fsrcnnx[HI_MODEL];
    if (!s) return false; // primary path triggers the fetch
    while (t.hiStages.length < chainDepth) {
      t.hiStages.push(new FsrcnnxModel(device, s.manifest, s.wgsl, { expectedName: HI_MODEL }));
    }
  } else if (engine === "artcnn") {
    const s = srcCache.artcnn[artVariant];
    if (!s) return false; // not loaded yet; primary path will have triggered the fetch
    if (!t.artStages[artVariant]) t.artStages[artVariant] = [];
    while (t.artStages[artVariant].length < chainDepth) {
      t.artStages[artVariant].push(new ArtCnnModel(device, s.manifest, s.wgsl, { expectedName: artVariant }));
    }
  }
  return true;
}

// Swap a MultiTarget's state into the globals (returns a restore function).
function withTarget(t, fn) {
  ensureTargetModels(t);
  const saved = { video, canvas, context, lumaTexture, lumaW, lumaH, hiRGB, hiRGBW, hiRGBH, debandInterTex, debandInterW, debandInterH, ssimds, sharpenPipeline, sharpenStrengthBuilt, hoverHidden, models, hiStages, artStages };
  video = t.video; canvas = t.canvas; context = t.context;
  lumaTexture = t.lumaTexture; lumaW = t.lumaW; lumaH = t.lumaH;
  hiRGB = t.hiRGB; hiRGBW = t.hiRGBW; hiRGBH = t.hiRGBH;
  debandInterTex = t.debandInterTex; debandInterW = t.debandInterW; debandInterH = t.debandInterH;
  ssimds = t.ssimds; sharpenPipeline = t.sharpenPipeline; sharpenStrengthBuilt = t.sharpenStrengthBuilt;
  hoverHidden = t.hoverHidden;
  models = t.models; hiStages = t.hiStages; artStages = t.artStages;
  // chained wrappers are rebuilt per-target inside the render branch (cheap)
  chainedHi = null; chainedArt = null;
  try { fn(); }
  finally {
    // persist any lazily-created resources back onto the target
    t.lumaTexture = lumaTexture; t.lumaW = lumaW; t.lumaH = lumaH;
    t.hiRGB = hiRGB; t.hiRGBW = hiRGBW; t.hiRGBH = hiRGBH;
    t.debandInterTex = debandInterTex; t.debandInterW = debandInterW; t.debandInterH = debandInterH;
    t.sharpenPipeline = sharpenPipeline; t.sharpenStrengthBuilt = sharpenStrengthBuilt;
    ({ video, canvas, context, lumaTexture, lumaW, lumaH, hiRGB, hiRGBW, hiRGBH, debandInterTex, debandInterW, debandInterH, ssimds, sharpenPipeline, sharpenStrengthBuilt, hoverHidden, models, hiStages, artStages } = saved);
    chainedHi = null; chainedArt = null;
  }
}

// Find all qualifying videos (same criteria as findVideo, but all of them).
function findAllVideos() {
  return deepQueryVideos()
    .filter((v) => v.videoWidth > 0 && v.videoHeight > 0)
    .filter((v) => {
      const r = v.getBoundingClientRect();
      const visible = r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 &&
        r.top < (window.innerHeight || 1e9) && r.left < (window.innerWidth || 1e9);
      return visible && r.width * r.height > 64 * 64 && !isTaintedVideo(v);
    });
}

// Reconcile multiTargets with the current set of on-screen videos.
function syncMultiTargets() {
  if (!optAllVideos || mode === "off") { clearMultiTargets(); return; }
  const present = new Set(findAllVideos());
  present.delete(video); // primary handled by the main path
  // remove targets whose video is gone/offscreen
  for (const [vid, t] of multiTargets) {
    if (!present.has(vid)) { t.destroy(); multiTargets.delete(vid); }
  }
  // add new ones
  for (const vid of present) {
    if (!multiTargets.has(vid)) {
      const t = new MultiTarget(vid);
      multiTargets.set(vid, t);
      t.canvas.style.display = "block";
      t.rvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype
        ? vid.requestVideoFrameCallback(function step(){ renderMultiOne(vid); if (multiTargets.has(vid) && mode!=="off") multiTargets.get(vid).rvfc = vid.requestVideoFrameCallback(step); })
        : null;
    }
  }
}
function clearMultiTargets() {
  for (const [, t] of multiTargets) t.destroy();
  multiTargets.clear();
}

// Render one secondary video (called from its own rVFC loop).
function renderMultiOne(vid) {
  if (mode === "off" || !optAllVideos) return;
  const t = multiTargets.get(vid);
  if (!t || !vid.videoWidth) return;
  try {
    withTarget(t, () => {
      if (mode === "upscale" && engine === "neural") {
        if (!t.neuralBypassLogged) {
          log("multi-video: neural upscaling is primary-video only; secondary video uses passthrough");
          t.neuralBypassLogged = true;
        }
        renderPassthrough();
      } else if (mode === "upscale") renderUpscale();
      else renderPassthrough();
      if (t.canvas) t.canvas.style.opacity = (optHoverReveal && t.hoverHidden) ? "0" : "1";
    });
  } catch (e) { warn("multi render error:", e.message); }
}

export function setSharpen(on) {
  sharpenEnabled = !!on;
  saveSitePrefs();
  return { ok: true, sharpen: sharpenEnabled };
}
export function setSharpenStrength(v) {
  sharpenStrength = Math.max(0.1, Math.min(2.0, Number(v) || 1.0));
  saveSitePrefs();
  return { ok: true, strength: sharpenStrength };
}

export function setSSimDS(on) {
  ssimdsEnabled = !!on;
  saveSitePrefs();
  return { ok: true, ssimds: ssimdsEnabled };
}

export function setPolicy(p) {
  upscalePolicy = p;
  resetScaleSelection();
  // For 2x-only engines (ArtCNN / FSRCNNX high) the policy encodes the scale via
  // chain depth: force2 -> 1, force4 -> 2, force8 -> 3.
  chainDepth = (engine === "artcnn" || engine === "fsrcnnx-hi") ? policyToDepth(p) : 1;
  artDiagLogged = false;
  if (engine === "artcnn" && device) {
    ensureArtStages(artVariant, chainDepth).catch(() => {});
  }
  if (engine === "fsrcnnx-hi" && device) {
    ensureHiStages(chainDepth).catch(() => {});
  }
  saveSitePrefs();
  return { ok: true, policy: upscalePolicy, chainDepth };
}

// Restore saved preferences for this site (called once on content-script load).
// Applies all settings, and re-enters the saved mode automatically: the saved
// mode IS the per-site memory — "off" stays off, "upscale"/"passthrough"
// re-activates once a playable video is present. No separate toggle.
export async function restoreSitePrefs() {
  const restoreToken = autoEnableGeneration;
  const p = await loadSitePrefs();
  if (!p) return { ok: true, restored: false };
  if (restoreToken !== autoEnableGeneration) return { ok: false, restored: false, reason: "superseded" };
  const selectionGeneration = ++engineSelectionGeneration;
  if (p.engine) engine = p.engine === "artcnn" ? "artcnn" : p.engine === "fsrcnnx-hi" ? "fsrcnnx-hi" : p.engine === "neural" ? "neural" : "fsrcnnx";
  if (typeof p.neuralModel === "string") neuralModelKey = p.neuralModel;
  if (engine === "neural") ensureNeural(selectionGeneration).catch((er) => {
    if (selectionGeneration !== engineSelectionGeneration || engine !== "neural") return;
    if (mode === "off" && /initialization cancelled/i.test(er.message)) return;
    warn("neural restore failed:", er.message);
    engineSelectionGeneration++;
    engine = "fsrcnnx";
    resumeInterpolationAfterNeural();
  });
  if (p.artVariant && ART_FILES.includes(p.artVariant)) artVariant = p.artVariant;
  if (typeof p.policy === "string") upscalePolicy = p.policy;
  if (typeof p.ssimds === "boolean") ssimdsEnabled = p.ssimds;
  if (typeof p.sharpen === "boolean") sharpenEnabled = p.sharpen;
  if (Number.isFinite(p.sharpenStrength)) {
    sharpenStrength = Math.max(0.1, Math.min(2.0, p.sharpenStrength));
  }
  if (typeof p.hoverReveal === "boolean") optHoverReveal = p.hoverReveal;
  if (typeof p.allVideos === "boolean") optAllVideos = p.allVideos;
  if (typeof p.deband === "boolean") debandEnabled = p.deband;
  if (Number.isFinite(p.debandStrength)) {
    debandStrength = Math.max(0.3, Math.min(3.0, p.debandStrength));
  }
  chainDepth = (engine === "artcnn" || engine === "fsrcnnx-hi") ? policyToDepth(upscalePolicy) : 1;

  // image upscaling is independent of the video mode; restore + activate it
  if (typeof p.images === "boolean" && p.images) {
    setImages(true).catch(() => {});
  }

  // interpolation settings: engine/target restore immediately; on/off auto-enables
  // once a video is ready (after the upscale auto-enable, so the chain engages).
  if (typeof p.interpEngine === "string" && p.interpEngine) pendingEngine = p.interpEngine;
  if (p.interpTargetFps != null) pendingTargetFps = p.interpTargetFps;
  if (typeof p.interpAutoFallback === "boolean") interpAutoFallbackPref = p.interpAutoFallback;
  if (typeof p.interpLadder === "boolean") interpLadderPref = p.interpLadder;
  if (typeof p.interpInvert === "boolean") interpInvertPref = p.interpInvert;
  const wantInterp = p.interpolate === true;

  const wasEnabled = p.mode && p.mode !== "off";
  if (wasEnabled) scheduleAutoEnable(p.mode, wantInterp);
  else if (wantInterp) scheduleAutoEnable(null, true);
  return { ok: true, restored: true, wasEnabled, savedMode: p.mode || "off" };
}

// Wait (with backoff) for a playable, non-protected video, then enter `targetMode`.
// Pages often load the player after the content script runs, so we poll briefly.
let autoEnableTimer = null, autoEnableGeneration = 0;
function cancelAutoEnable() {
  autoEnableGeneration++;
  if (autoEnableTimer != null) clearTimeout(autoEnableTimer);
  autoEnableTimer = null;
}
function scheduleAutoEnable(targetMode, alsoInterp = false) {
  if (autoEnableTimer != null) clearTimeout(autoEnableTimer);
  const generation = ++autoEnableGeneration;
  let attempts = 0;
  const tryEnable = async () => {
    if (generation !== autoEnableGeneration) return;
    if (targetMode && mode !== "off") return; // user already acted
    attempts++;
    const v = findVideo();
    if (v && probeVideo(v) === "ok") {
      log(`auto-enabling (${targetMode || "interp-only"}${alsoInterp && targetMode ? "+interp" : ""}) for ${siteHost()}`);
      if (targetMode) await setMode(targetMode, generation);
      if (generation !== autoEnableGeneration) return;
      if (alsoInterp && !optInterpolate) {
        // upscaling (if any) is on first, so the interp start makes the chain decision
        try { await setInterpolate(true, generation); } catch (e) { warn("interp auto-enable failed:", e.message); }
      }
      return;
    }
    if (attempts < 40 && generation === autoEnableGeneration) autoEnableTimer = setTimeout(tryEnable, 500); // up to ~20s
  };
  autoEnableTimer = setTimeout(tryEnable, 600);
}

export function getStatus() {
  return { mode, hasVideo: !!findVideo(), webgpu: "gpu" in navigator, frameCount,
           model: activeModel?.manifest?.name || null, scale: activeModel?.scale || null,
           policy: upscalePolicy, ssimds: ssimdsEnabled,
           sharpen: sharpenEnabled, sharpenStrength,
           engine, artVariant, chainDepth,
           neural: engine === "neural" && neuralEng ? { model: neuralModelKey || (neuralEng.activeEntry()?.key ?? null), label: neuralEng.activeEntry()?.label ?? null, scale: neuralEng.activeEntry()?.scale ?? null, ready: neuralEng.ready(), ...neuralEng.stats() } : null,
           neuralModels: _neuralList,
           protected: protectedSource, protectedReason, host: siteHost(),
           hoverReveal: optHoverReveal, allVideos: optAllVideos,
           deband: debandEnabled, debandStrength,
           images: optImages, imageCount: imageUpscaledCount,
           interpolate: optInterpolate, interpPausedByNeural,
           interpStats: interpolator ? interpolator.getStats() : null,
           interpAutoFallback: interpAutoFallbackPref,
           interpLadder: interpLadderPref, interpInvert: interpInvertPref,
           multiCount: multiTargets.size,
           gpuRecovering: !!(deviceRecoveryPromise || deviceRecoveryTimer) };
}

// ---- image upscaling (advanced option) -----------------------------------
// Provides the image module with the shared GPU device and a freshly-built
// quality-focused FSRCNNX x2 model. It owns a separate per-size texture cache so
// image work cannot reallocate the active video model. Lazily initialized.
let imageUpscaler = null, imageUpscalerInitPromise = null, imageUpscalerInitDevice = null;
let imageUpscalerInitToken = -1;
let imageUpscalerInitGeneration = 0, imagesSelectionGeneration = 0;
let optImages = false, imageUpscaledCount = 0;

function invalidateImageUpscaler() {
  imageUpscalerInitGeneration++;
  const up = imageUpscaler;
  if (!up) return;
  try { up.destroy?.(); } catch {}
  imageUpscaler = null;
  imageUpscaledCount = 0;
}

async function createImageUpscaler(initDevice, initFormat, initSampler, initGeneration) {
  // Images use the stronger, quality-focused FSRCNNX_x2_56-16-4-1 (56 feature
  // maps) once for a 2x upscale. Load its source on demand.
  const IMG_MODEL = "FSRCNNX_x2_56-16-4-1";
  if (!srcCache.fsrcnnx[IMG_MODEL]) {
    const base = chrome.runtime.getURL(`model/${IMG_MODEL}`);
    const [manifest, wgsl] = await Promise.all([
      fetch(`${base}.passes.json`).then((r) => r.json()),
      fetch(`${base}.wgsl`).then((r) => r.text()),
    ]);
    srcCache.fsrcnnx[IMG_MODEL] = { name: IMG_MODEL, manifest, wgsl };
  }
  const mod = await import(chrome.runtime.getURL("fsrcnnx-images.js"));
  if (initGeneration !== imageUpscalerInitGeneration || device !== initDevice) {
    const error = new Error("image upscaler initialization superseded");
    error.code = "IMAGE_INIT_SUPERSEDED";
    throw error;
  }
  const created = new mod.ImageUpscaler({
    device: initDevice, format: initFormat, sampler: initSampler,
    fsrcnnxSource: srcCache.fsrcnnx[IMG_MODEL],
    FsrcnnxModel, SsimDownscaler,
    onCount: (n) => { imageUpscaledCount = n; },
    warn,
  });
  if (initGeneration !== imageUpscalerInitGeneration || device !== initDevice) {
    try { created.destroy?.(); } catch {}
    const error = new Error("image upscaler initialization superseded");
    error.code = "IMAGE_INIT_SUPERSEDED";
    throw error;
  }
  if (imageUpscaler) {
    try { created.destroy?.(); } catch {}
    return imageUpscaler;
  }
  imageUpscaler = created;
  return created;
}

async function ensureImageUpscaler() {
  if (imageUpscaler) return imageUpscaler;
  if (imageUpscalerInitPromise) {
    if (imageUpscalerInitDevice === device && imageUpscalerInitToken === imageUpscalerInitGeneration) {
      return imageUpscalerInitPromise;
    }
    try { await imageUpscalerInitPromise; } catch {}
    return ensureImageUpscaler();
  }
  if (!(await initWebGPU())) return null;
  if (imageUpscaler) return imageUpscaler;
  if (imageUpscalerInitPromise) return imageUpscalerInitPromise;
  const initDevice = device;
  const initFormat = format;
  const initSampler = sampler;
  const initGeneration = imageUpscalerInitGeneration;
  const promise = createImageUpscaler(initDevice, initFormat, initSampler, initGeneration).finally(() => {
    if (imageUpscalerInitPromise === promise) {
      imageUpscalerInitPromise = null;
      imageUpscalerInitDevice = null;
      imageUpscalerInitToken = -1;
    }
  });
  imageUpscalerInitDevice = initDevice;
  imageUpscalerInitToken = initGeneration;
  imageUpscalerInitPromise = promise;
  return promise;
}

export async function setImages(on) {
  const selectionGeneration = ++imagesSelectionGeneration;
  optImages = !!on;
  saveSitePrefs();
  if (optImages) {
    let up;
    try { up = await ensureImageUpscaler(); }
    catch (error) {
      if (selectionGeneration !== imagesSelectionGeneration || !optImages || error.code === "IMAGE_INIT_SUPERSEDED") {
        return { ok: false, images: optImages, reason: "superseded" };
      }
      warn("image upscaler initialization failed:", error.message);
      return { ok: false, images: optImages, reason: error.message };
    }
    if (selectionGeneration !== imagesSelectionGeneration || !optImages) {
      if (!optImages) { try { up?.stop?.(); } catch {} }
      return { ok: false, images: optImages, reason: "superseded" };
    }
    if (up) up.start();
  } else if (imageUpscaler) {
    imageUpscalerInitGeneration++;
    imageUpscaler.stop();
    imageUpscaledCount = 0;
  } else {
    imageUpscalerInitGeneration++;
  }
  return { ok: true, images: optImages };
}

// ---- frame interpolation (experimental, staged) ---------------------------
// Stage 1 is passthrough only (capture -> present -> sync test), no RIFE. This
// is gated, off by default, and NOT persisted per-site yet — it's an opt-in
// experiment while we validate feasibility.
let interpolator = null, interpolatorInitPromise = null;
let interpolationSelectionGeneration = 0, interpolationConfigGeneration = 0;
let optInterpolate = false;
let pendingEngine = null;     // "blend" or a RIFE model key, chosen before start
let pendingTargetFps = null;  // target fps chosen before start

function configureInterpolator(instance) {
  if (pendingEngine && instance.setInterpEngine) instance.setInterpEngine(pendingEngine);
  if (pendingTargetFps != null && instance.setTargetFps) instance.setTargetFps(pendingTargetFps);
  if (instance.setAutoFallback) instance.setAutoFallback(interpAutoFallbackPref);
  if (instance.setLadder) instance.setLadder(interpLadderPref);
}

function scheduleInterpolatorGpuRestart() {
  const instance = interpolator;
  if (!instance?.running || !optInterpolate || engine === "neural") return;
  const selectionGeneration = interpolationSelectionGeneration;
  try { instance.stop(); }
  catch (error) { warn("interpolation GPU-error stop failed:", error.message); return; }
  setTimeout(() => {
    // A delayed recovery must never override a newer user toggle, engine switch,
    // or replacement instance. The current selection owns whether it may start.
    if (selectionGeneration !== interpolationSelectionGeneration ||
        instance !== interpolator || !optInterpolate || engine === "neural") return;
    instance.start().catch((error) => warn("interpolation GPU-error restart failed:", error.message));
  }, 50);
}

async function ensureInterpolatorInstance() {
  if (interpolator) return interpolator;
  if (interpolatorInitPromise) return interpolatorInitPromise;
  const promise = import(chrome.runtime.getURL("fsrcnnx-interpolate.js")).then((mod) => {
    if (!interpolator) {
      interpolator = new mod.Interpolator({ findVideo, log, warn,
        chain: { tap: chainTap, info: chainInfo, available: chainAvailable, device: chainDevice, adopt: adoptChainDevice, targetDims: chainTargetDims,
                 upscaleTex: chainUpscaleTex, setInverted: setChainInverted, invert: () => interpInvertPref,
                 ladder: () => interpLadderPref } });
    }
    return interpolator;
  }).finally(() => {
    if (interpolatorInitPromise === promise) interpolatorInitPromise = null;
  });
  interpolatorInitPromise = promise;
  return promise;
}

export async function setInterpolate(on, autoEnableToken = null) {
  if (autoEnableToken == null) cancelAutoEnable();
  else if (autoEnableToken !== autoEnableGeneration) {
    return { ok: false, interpolate: optInterpolate, reason: "superseded" };
  }
  const selectionGeneration = ++interpolationSelectionGeneration;
  optInterpolate = !!on;
  if (optInterpolate) {
    if (engine === "neural") {
      pauseInterpolationForNeural();
      saveSitePrefs();
      return { ok: true, interpolate: true, running: false, paused: "neural" };
    }
    interpPausedByNeural = false;
    const instance = await ensureInterpolatorInstance();
    if (selectionGeneration !== interpolationSelectionGeneration || !optInterpolate) {
      if (!optInterpolate) instance.stop();
      return { ok: false, interpolate: optInterpolate, reason: "superseded" };
    }
    if (engine === "neural") {
      pauseInterpolationForNeural();
      saveSitePrefs();
      return { ok: true, interpolate: true, running: false, paused: "neural" };
    }
    configureInterpolator(instance);
    const r = await instance.start();
    if (selectionGeneration !== interpolationSelectionGeneration || !optInterpolate || engine === "neural") {
      if (!optInterpolate || engine === "neural") instance.stop();
      if (engine === "neural" && optInterpolate) interpPausedByNeural = true;
      return { ok: false, interpolate: optInterpolate, reason: "superseded" };
    }
    if (!r.ok) { optInterpolate = false; interpPausedByNeural = false; warn("interpolation start failed:", r.reason); }
    saveSitePrefs();
    return r;
  } else if (interpolator) {
    interpPausedByNeural = false;
    interpolator.stop();
  } else {
    interpPausedByNeural = false;
  }
  saveSitePrefs();
  return { ok: true, interpolate: optInterpolate };
}
export function getInterpolateStats() {
  return interpolator ? interpolator.getStats() : { running: false };
}
export function setInterpolateRes(mode) {
  if (interpolator) interpolator.setResMode(mode);
  return { ok: true, resMode: mode };
}
export function setInterpolateAvOffset(ms) {
  if (interpolator) return { ok: true, avOffsetMs: interpolator.setAvOffset(ms) };
  return { ok: false };
}
export async function setInterpolateModel(key) {
  // Model changes are configuration revisions, not lifecycle selections. In
  // particular, they must not cancel an enable that is awaiting the shared
  // module import; configureInterpolator() will apply the newest pending key.
  const configGeneration = ++interpolationConfigGeneration;
  pendingEngine = key; // remember for a future interpolator instance
  saveSitePrefs();
  if (!interpolator) return { ok: true, model: key, pending: true };
  // Apply the engine choice through the proper start path (standalone-blend vs RIFE,
  // chain decision, model init) by restarting a RUNNING interpolator. Mid-run flag
  // flips can't switch pipelines (e.g. RIFE session → standalone blend), which is
  // why Blend used to "stick with RIFE" until a manual off/on.
  const wasRunning = !!interpolator.running;
  if (wasRunning) interpolator.stop();
  if (interpolator.setInterpEngine) interpolator.setInterpEngine(key);
  if (wasRunning) {
    const r = await interpolator.start();
    if (configGeneration !== interpolationConfigGeneration || !optInterpolate || engine === "neural") {
      if (!optInterpolate || engine === "neural") interpolator.stop();
      return { ok: false, model: key, reason: "superseded" };
    }
    return { ok: r.ok, model: key, restarted: true };
  }
  return { ok: true, model: key, ready: true };
}
export function setInterpolateTargetFps(v) {
  pendingTargetFps = v;
  saveSitePrefs();
  if (!interpolator || !interpolator.setTargetFps) return { ok: true, target: v, pending: true };
  return { ok: true, target: interpolator.setTargetFps(v) };
}
export function listInterpolateModels() {
  if (interpolator && interpolator._rifeMod && interpolator._rifeMod.listModels) {
    return interpolator._rifeMod.listModels();
  }
  return [];
}
// ===== INVERTED CHAIN (#4): interpolate-before-upscale =====
export function setChainInverted(on) {
  chainInverted = !!on;
  if (canvas) canvas.style.opacity = "1"; // upscaler canvas IS the display surface
  log(`inverted chain ${chainInverted ? "ON — interp drives upscales per presented frame" : "OFF — normal render loop resumes"}`);
  return chainInverted;
}
export function chainUpscaleTex(tex, w, h) {
  // Present-time upscale of ONE pooled source-res frame (real or RIFE tween).
  // Cost scales with PRESENTED frames; dropped frames never pay. Runs the full
  // existing pass chain (FSRCNNX/ArtCNN, SSimDS, sharpen, deband) through the
  // tex-ingest pipeline twins. Synchronous encode+submit, like a normal frame.
  if (!device || mode !== "upscale" || !tex) return false;
  try {
    if (!ensureTexPipelines()) return false;
    _texSource = { tex, w: w || tex._w, h: h || tex._h };
    renderUpscale();
    return true;
  }
  catch (e) { _texSource = null; warn("chainUpscaleTex failed:", e.message); return false; }
}
export async function setInterpolateInvert(on) {
  interpInvertPref = !!on;
  saveSitePrefs();
  if (!interpolator) return { ok: true, invert: interpInvertPref, pending: true };
  // Mode selection happens at interpolator start (capture path, pin dims, present
  // sink) — mirror the model-change restart so the flip takes effect cleanly.
  const wasRunning = !!interpolator.running;
  if (wasRunning) interpolator.stop();
  if (wasRunning) {
    const r = await interpolator.start();
    return { ok: r.ok, invert: interpInvertPref, restarted: true };
  }
  return { ok: true, invert: interpInvertPref, ready: true };
}
export function setInterpolateAutoFallback(on) {
  interpAutoFallbackPref = !!on;
  saveSitePrefs();
  if (interpolator && interpolator.setAutoFallback) interpolator.setAutoFallback(interpAutoFallbackPref);
  return { ok: true, autoFallback: interpAutoFallbackPref };
}
export function setInterpolateLadder(on) {
  interpLadderPref = !!on;
  saveSitePrefs();
  if (interpolator && interpolator.setLadder) interpolator.setLadder(interpLadderPref);
  return { ok: true, ladder: interpLadderPref };
}
export function setInterpolateDiag(on) {
  // controls the static-region passthrough (jitter fix). Sets both the CPU-path
  // flag (rife module) and the interpolator flag (used by the GPU composite shader).
  let ok = false;
  if (interpolator && interpolator._rifeMod && interpolator._rifeMod.setStaticPassthrough) {
    interpolator._rifeMod.setStaticPassthrough(on); ok = true;
  }
  if (interpolator) { interpolator._staticOn = !!on; ok = true; }
  return { ok, staticPassthrough: !!on };
}

log("pipeline module loaded");

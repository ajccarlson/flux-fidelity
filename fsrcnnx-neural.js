// fsrcnnx-neural.js — ONNX neural upscaler engine (v0.49.0, Tier-B ladder).
//
// Runs community super-resolution models (SPAN / RealPLKSR / DAT2 / ATD, any
// spandrel-exportable arch) as a SECOND ORT WebGPU session alongside RIFE's,
// sharing the same ORT env/device. Unlike the WGSL mpv ports (luma-only), these
// models are RGB: chroma gets neural treatment and the recombine stage is
// bypassed — output composites straight into an rgba16float texture that
// main.js presents through the existing SSimDS/sharpen tail.
//
// Frame flow (mirrors the RIFE GPU-resident pattern exactly):
//   rvfc → pack pass (external texture → padded NCHW fp32 storage buffer,
//   submitted before any await so the external texture is consumed in-task)
//   → await session.run (fromGpuBuffer in, preferredOutputLocation gpu-buffer
//   out — zero readback) → composite pass (output buffer → rgba16float tex,
//   crop replicate-pad) → caller presents.
//
// Models are dropped into model/neural/ with a manifest.json; the export kit
// in tools/neural-export/ produces both. Sessions use DYNAMIC dims (no
// freeDimensionOverrides, no graph capture — experiment #1's verdict stands).

import { createOrtSession, ensureOrt, getOrtSessionDevice } from "./fsrcnnx-rife.js";

const NEURAL_KEY = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const NEURAL_FILE = /^[A-Za-z0-9][A-Za-z0-9._-]*\.onnx$/;
const TENSOR_NAME = /^[A-Za-z_][A-Za-z0-9_.-]{0,127}$/;

export function isValidNeuralModelKey(value) {
  return typeof value === "string" && NEURAL_KEY.test(value);
}

export function validateNeuralManifest(value) {
  const raw = Array.isArray(value) ? value : value?.models;
  if (!Array.isArray(raw)) throw new Error("neural manifest must be an array or {models: array}");
  const keys = new Set();
  const files = new Set();
  return raw.map((entry, index) => {
    const at = `neural manifest entry ${index}`;
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error(`${at} must be an object`);
    if (!isValidNeuralModelKey(entry.key)) throw new Error(`${at} has an invalid key`);
    if (keys.has(entry.key)) throw new Error(`${at} duplicates key '${entry.key}'`);
    if (typeof entry.file !== "string" || !NEURAL_FILE.test(entry.file)) throw new Error(`${at} has an invalid model filename`);
    if (files.has(entry.file)) throw new Error(`${at} duplicates model file '${entry.file}'`);
    if (!Number.isInteger(entry.scale) || entry.scale < 1 || entry.scale > 16) throw new Error(`${at} has an invalid scale`);
    if (entry.padMultiple != null &&
        (!Number.isInteger(entry.padMultiple) || entry.padMultiple < 1 || entry.padMultiple > 256)) {
      throw new Error(`${at} has an invalid padMultiple`);
    }
    if (entry.label != null && (typeof entry.label !== "string" || !entry.label.trim() || entry.label.length > 160)) {
      throw new Error(`${at} has an invalid label`);
    }
    for (const field of ["input", "output"]) {
      if (entry[field] != null && (typeof entry[field] !== "string" || !TENSOR_NAME.test(entry[field]))) {
        throw new Error(`${at} has an invalid ${field} tensor name`);
      }
    }
    if (entry.fp16 != null && typeof entry.fp16 !== "boolean") throw new Error(`${at} has an invalid fp16 flag`);
    keys.add(entry.key);
    files.add(entry.file);
    return Object.freeze({ ...entry });
  });
}

const PACK_EXT_WGSL = `
struct P { padW:u32, padH:u32, w:u32, h:u32 }
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_external;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.padW || gid.y >= u.padH) { return; }
  let xx = min(gid.x, u.w - 1u);
  let yy = min(gid.y, u.h - 1u);
  let uv = (vec2<f32>(f32(xx), f32(yy)) + vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(u.w), f32(u.h));
  let c = textureSampleBaseClampToEdge(src, samp, uv).rgb;
  let plane = u.padW * u.padH;
  let idx = gid.y * u.padW + gid.x;
  dst[idx] = c.r;
  dst[plane + idx] = c.g;
  dst[2u * plane + idx] = c.b;
}`;

// texture_2d twin (same body, sampled with explicit LOD) — the source-is-a-
// parameter doctrine: frames from a decoder and frames synthesized upstream
// go through identical math.
const PACK_TEX_WGSL = PACK_EXT_WGSL
  .replace("texture_external", "texture_2d<f32>")
  .replace("textureSampleBaseClampToEdge(src, samp, uv)", "textureSampleLevel(src, samp, uv, 0.0)");

const COMPOSITE_WGSL = `
struct P { strideW:u32, plane:u32, outW:u32, outH:u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.outW || gid.y >= u.outH) { return; }
  let idx = gid.y * u.strideW + gid.x;
  let r = clamp(src[idx], 0.0, 1.0);
  let g = clamp(src[u.plane + idx], 0.0, 1.0);
  let b = clamp(src[2u * u.plane + idx], 0.0, 1.0);
  textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, 1.0));
}`;

export function createNeuralEngine({ log = console.log, warn = console.warn } = {}) {
  let ort = null;
  let session = null;
  let device = null;
  let manifest = null;          // [{key,label,file,scale,padMultiple?,input?,output?}]
  let active = null;            // manifest entry of the loaded model
  let inputName = "input", outputName = "output";
  let fp16Model = false;
  let initGeneration = 0;
  let initTail = Promise.resolve();
  let latestInitPromise = null;
  let sessionGeneration = 0;
  let lifecycleGeneration = 0;
  let activeRuns = 0;
  let runIdleResolvers = [];
  let runBusy = false;
  let deferredSessionReleases = [];
  const retirements = new Set();
  let disposalPromise = null;
  const watchedDevices = new WeakSet();
  const lostDevices = new WeakSet();
  const deviceInvalidations = new WeakMap();
  const activeDeviceInvalidations = new Set();
  const deviceInvalidationFailures = new Set();
  let deviceInvalidationTail = Promise.resolve();
  let deviceInvalidationPromise = null;
  let sessionReleaseFailures = [];

  // GPU resources (allocated on ORT's device)
  let sampler = null;
  let packExtPipe = null, packTexPipe = null, compPipe = null;
  let inBuf = null, inBufSize = 0;
  let packU = null, compU = null;
  let outTex = null, outTexW = 0, outTexH = 0;

  // instrumentation (mirrors RIFE's readout vocabulary)
  const stats = { last: 0, mu: 0, n: 0, skip: 0, fails: 0 };

  function whenRunsIdle() {
    if (activeRuns === 0) return Promise.resolve();
    return new Promise((resolve) => runIdleResolvers.push(resolve));
  }

  function endRun() {
    activeRuns = Math.max(0, activeRuns - 1);
    if (activeRuns === 0 && runIdleResolvers.length) {
      for (const resolve of runIdleResolvers.splice(0)) resolve();
    }
  }

  function trackRetirement(promise) {
    const tracked = Promise.resolve(promise).catch(() => {});
    retirements.add(tracked);
    tracked.finally(() => retirements.delete(tracked));
    return tracked;
  }

  function afterSubmittedWork(ownerDevice, callback) {
    let fence;
    try { fence = ownerDevice?.queue?.onSubmittedWorkDone?.() || Promise.resolve(); }
    catch { fence = Promise.resolve(); }
    return trackRetirement(Promise.resolve(fence).catch(() => {}).then(callback));
  }

  function retireGpuObjects(objects, ownerDevice = device) {
    const live = objects.filter(Boolean);
    if (!live.length) return Promise.resolve();
    return afterSubmittedWork(ownerDevice, () => {
      for (const object of live) { try { object.destroy?.(); } catch {} }
    });
  }

  function retireTensors(ownerDevice, inputTensor, outputTensor) {
    if (!inputTensor && !outputTensor) return;
    afterSubmittedWork(ownerDevice, () => {
      try { inputTensor?.dispose?.(); } catch {}
      try { outputTensor?.dispose?.(); } catch {}
    });
  }

  async function releaseDeferredSessions() {
    const pending = deferredSessionReleases;
    deferredSessionReleases = [];
    const failures = [];
    for (const oldSession of pending) {
      try { await oldSession?.release?.(); }
      catch (error) {
        failures.push(error);
        sessionReleaseFailures.push(error);
        warn("neural: deferred old session release failed:", error.message);
      }
    }
    if (failures.length) {
      throw new AggregateError(failures, "neural deferred session release failed");
    }
  }

  function watchDevice(ownerDevice) {
    if (!ownerDevice || watchedDevices.has(ownerDevice) || !ownerDevice.lost?.then) return;
    watchedDevices.add(ownerDevice);
    ownerDevice.lost.then((info) => {
      lostDevices.add(ownerDevice);
      if (device === ownerDevice) {
        warn(`neural: GPU device lost: ${info?.message || info?.reason || "unknown reason"}`);
      }
      // A replaced session can still be the lifetime guard for main's old
      // adopted device. Route every watched loss through the identity-aware
      // invalidator so a non-current guard is claimed and released as well.
      invalidateDevice(ownerDevice).catch((error) =>
        warn("neural: device-loss cleanup failed:", error.message));
    }).catch(() => {});
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
        throw new AggregateError([...failures], "neural device invalidation barrier failed");
      }
    });
    // The provider watcher observes each per-device promise. This independent
    // aggregate can reject before init()/dispose() consumes it, so attach a
    // passive observer while preserving the rejection for those callers.
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
        throw new AggregateError([...failures], "neural device invalidation failed");
      }
      return current.value;
    });
    deviceInvalidations.set(ownerDevice, operation);
    activeDeviceInvalidations.add(operation);
    // Register settlement bookkeeping before building allSettled snapshots so
    // a disposal continuation always sees the failure latch populated.
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

  async function loadManifest() {
    if (manifest) return manifest;
    try {
      const r = await fetch(chrome.runtime.getURL("model/neural/manifest.json"));
      if (!r.ok) throw new Error("HTTP " + r.status);
      manifest = validateNeuralManifest(await r.json());
    } catch (e) {
      warn("neural manifest rejected:", e.message);
      manifest = [];
    }
    return manifest;
  }

  function ensurePipelines() {
    if (packExtPipe) return;
    const mk = (code) => device.createShaderModule({ code });
    let nextPackU = null, nextCompU = null;
    try {
      const nextPackExtPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_EXT_WGSL), entryPoint: "main" } });
      const nextPackTexPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_TEX_WGSL), entryPoint: "main" } });
      const nextCompPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(COMPOSITE_WGSL), entryPoint: "main" } });
      const nextSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
      nextPackU = device.createBuffer({ label: "neural-packU", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      nextCompU = device.createBuffer({ label: "neural-compU", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      packExtPipe = nextPackExtPipe; packTexPipe = nextPackTexPipe; compPipe = nextCompPipe;
      sampler = nextSampler; packU = nextPackU; compU = nextCompU;
    } catch (error) {
      try { nextPackU?.destroy?.(); } catch {}
      try { nextCompU?.destroy?.(); } catch {}
      throw error;
    }
  }

  function ensureInBuf(padW, padH) {
    const need = padW * padH * 3 * 4;
    if (inBuf && inBufSize === need) return;
    const old = inBuf;
    const candidate = device.createBuffer({
      label: `neural-in-${padW}x${padH}`,
      size: need,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    inBuf = candidate;
    inBufSize = need;
    if (old) retireGpuObjects([old]);
  }

  function ensureOutTex(w, h) {
    if (outTex && outTexW === w && outTexH === h) return;
    const old = outTex;
    const candidate = device.createTexture({
      label: `neural-out-${w}x${h}`,
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    outTex = candidate;
    outTexW = w; outTexH = h;
    if (old) retireGpuObjects([old]);
  }

  function checkedProduct(label, ...values) {
    let product = 1;
    for (const value of values) {
      if (!Number.isSafeInteger(value) || value <= 0 || product > Number.MAX_SAFE_INTEGER / value) {
        const error = new Error(`${label} exceeds the safe integer range`);
        error.code = "NEURAL_LIMIT";
        throw error;
      }
      product *= value;
    }
    return product;
  }

  function validateAllocationLimits(padW, padH, outW, outH, scale) {
    const maxDimension = Math.max(1, Number(device?.limits?.maxTextureDimension2D) || 8192);
    const maxBuffer = Math.max(1, Math.min(
      Number(device?.limits?.maxBufferSize) || 256 * 1024 * 1024,
      Number(device?.limits?.maxStorageBufferBindingSize) || 128 * 1024 * 1024,
    ));
    const maxGroups = Math.max(1, Number(device?.limits?.maxComputeWorkgroupsPerDimension) || 65535);
    const paddedOutW = checkedProduct("neural padded output width", padW, scale);
    const paddedOutH = checkedProduct("neural padded output height", padH, scale);
    const dimensions = [padW, padH, outW, outH, paddedOutW, paddedOutH];
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1 || value > maxDimension)) {
      const error = new Error(`neural dimensions exceed the device texture limit ${maxDimension}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
    if (Math.ceil(padW / 8) > maxGroups || Math.ceil(padH / 8) > maxGroups ||
        Math.ceil(outW / 8) > maxGroups || Math.ceil(outH / 8) > maxGroups) {
      const error = new Error(`neural dispatch exceeds the device workgroup limit ${maxGroups}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
    const inputBytes = checkedProduct("neural input buffer", padW, padH, 3, 4);
    const outputBytes = checkedProduct("neural output buffer", paddedOutW, paddedOutH, 3, 4);
    if (inputBytes > maxBuffer || outputBytes > maxBuffer) {
      const error = new Error(`neural tensor buffer exceeds the device binding limit ${maxBuffer}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
  }

  async function initOne(key, generation) {
    const list = await loadManifest();
    if (!list.length) throw new Error("no neural models: model/neural/manifest.json missing or empty");
    const entry = list.find((m) => m.key === key) || list[0];
    if (generation !== initGeneration) return null;
    if (session && device && active?.key === entry.key) return active;

    ort = await ensureOrt();
    if (generation !== initGeneration) return null;
    // Create the NEW session before releasing any old one: the shared ORT
    // device's lifetime follows its sessions, and letting the refcount touch
    // zero mid-swap would tear down the device under the upscaler (crash-1's
    // lesson, applied preemptively).
    const url = chrome.runtime.getURL("model/neural/" + entry.file);
    const opts = {
      executionProviders: [{ name: "webgpu" }],
      graphOptimizationLevel: "all",
      enableGraphCapture: false,
      preferredOutputLocation: "gpu-buffer",
    };
    let next;
    let executionFp16 = true;
    try {
      next = await createOrtSession(url, opts, { enableFp16: true });
    } catch (fp16Error) {
      if (generation !== initGeneration) return null;
      executionFp16 = false;
      warn(`neural: FP16 execution session failed for ${entry.file}; retrying FP32: ${fp16Error.message}`);
      try {
        next = await createOrtSession(url, opts, { enableFp16: false });
      } catch (fp32Error) {
        if (generation !== initGeneration) return null;
        throw new Error(`neural session create failed (${entry.file}): ${fp32Error.message}`);
      }
    }
    if (generation !== initGeneration) {
      try { await next?.release?.(); } catch {}
      return null;
    }
    const nextOrt = await ensureOrt();
    const nextDevice = getOrtSessionDevice(next);
    if (!nextDevice) {
      try { await next?.release?.(); } catch {}
      throw new Error("ORT device unavailable after neural session create");
    }
    // Subscribe before publication and yield once. A device can be returned with
    // an already-settled `lost` promise; accepting that session would let init()
    // report success just before the watcher clears it on the next turn.
    watchDevice(nextDevice);
    await Promise.resolve();
    if (generation !== initGeneration) {
      try { await next?.release?.(); } catch {}
      return null;
    }
    if (lostDevices.has(nextDevice)) {
      try { await next?.release?.(); } catch {}
      throw new Error("ORT device was lost during neural session initialization");
    }

    // Do not swap or release the old session while run() is using its names,
    // buffers, output tensor, or device. New runs cannot begin during this task's
    // synchronous commit once the idle promise resolves.
    await whenRunsIdle();
    if (generation !== initGeneration) {
      try { await next?.release?.(); } catch {}
      return null;
    }

    const oldSession = session;
    const oldDevice = device;
    if (oldDevice && oldDevice !== nextDevice) destroyGpuResources(oldDevice);
    device = nextDevice;
    try {
      ensurePipelines();
    } catch (error) {
      destroyGpuResources(nextDevice);
      device = oldDevice;
      try { await next?.release?.(); } catch {}
      throw error;
    }

    ort = nextOrt;
    session = next;
    active = entry;
    inputName = entry.input || (next.inputNames && next.inputNames[0]) || "input";
    outputName = entry.output || (next.outputNames && next.outputNames[0]) || "output";
    fp16Model = /fp16/i.test(entry.file) || entry.fp16 === true;
    sessionGeneration++;
    // Transfer the prior session out of this initializer's local ownership before
    // yielding. A concurrent dispose/device-loss can clear the just-published
    // session at the microtask below; without this handoff, the identity check
    // would throw before the prior session reached either release path.
    if (oldSession && !deferredSessionReleases.includes(oldSession)) {
      deferredSessionReleases.push(oldSession);
    }
    // Re-check after synchronous publication as well. This closes the narrow
    // window between the pre-commit yield and assigning the public session.
    await Promise.resolve();
    const initializationCancelled = generation !== initGeneration;
    if (lostDevices.has(nextDevice) || device !== nextDevice || session !== next) {
      await invalidateDevice(nextDevice);
      throw new Error("ORT device was lost during neural session initialization");
    }
    if (oldSession) {
      if (oldDevice === nextDevice) {
        // The new session retains the same shared device, so the old reference can
        // be released immediately without dropping its device refcount to zero.
        deferredSessionReleases = deferredSessionReleases.filter(
          (candidate) => candidate !== oldSession,
        );
        try { await oldSession.release?.(); }
        catch (error) {
          sessionReleaseFailures.push(error);
          warn("neural: old session release failed:", error.message);
          throw new AggregateError([error], "neural replaced session release failed");
        }
      }
      // For a different device, keep the pre-yield deferred ownership until the
      // first new-device run (whose pack is submitted after main's adoption).
      // This prevents a cross-device swap from orphaning current main pipelines.
    }

    if (initializationCancelled) return null;

    log(`neural: session ready — ${entry.label || entry.key} (${entry.scale}x, ${fp16Model ? "fp16" : "fp32"} weights, ${executionFp16 ? "FP16" : "FP32"} execution, dynamic dims)`);
    return entry;
  }

  function init(key) {
    if (disposalPromise) {
      return disposalPromise.then(() => init(key));
    }
    // A device invalidation owns the old ORT session until its GPU fence and
    // release() have both completed. Allocate the init generation only after
    // that physical cleanup finishes; otherwise invalidateDevice() would both
    // overlap session creation and cancel the recovery request it is gating.
    if (deviceInvalidationPromise) {
      return deviceInvalidationPromise.then(() => init(key));
    }
    const generation = ++initGeneration;
    const raw = initTail.catch(() => {}).then(() => initOne(key, generation));
    initTail = raw.then(() => undefined, () => undefined);
    let exposed;
    exposed = raw.then(
      (entry) => {
        if (entry) return entry;
        const latest = latestInitPromise;
        if (latest && latest !== exposed) return latest;
        throw new Error("neural initialization cancelled");
      },
      (error) => {
        const latest = latestInitPromise;
        if (generation !== initGeneration && latest && latest !== exposed) return latest;
        throw error;
      },
    );
    latestInitPromise = exposed;
    return exposed;
  }

  function destroyGpuResources(ownerDevice = device) {
    const resources = [inBuf, packU, compU, outTex];
    inBuf = null; inBufSize = 0; packU = null; compU = null;
    outTex = null; outTexW = 0; outTexH = 0;
    packExtPipe = null; packTexPipe = null; compPipe = null; sampler = null;
    const cleanup = whenRunsIdle().then(async () => {
      try { await ownerDevice?.queue?.onSubmittedWorkDone?.(); } catch {}
      for (const resource of resources) { try { resource?.destroy?.(); } catch {} }
    });
    return trackRetirement(cleanup);
  }

  function invalidateDevice(lostDevice) {
    if (!lostDevice) return Promise.resolve(false);
    const existing = deviceInvalidations.get(lostDevice);
    if (existing) return existing;
    const previousBarrier = deviceInvalidationPromise;
    const affectsCurrent = device === lostDevice;
    const oldSession = affectsCurrent ? session : null;
    const lostDeferred = deferredSessionReleases.filter(
      (candidate) => getOrtSessionDevice(candidate) === lostDevice,
    );
    if (lostDeferred.length) {
      deferredSessionReleases = deferredSessionReleases.filter(
        (candidate) => getOrtSessionDevice(candidate) !== lostDevice,
      );
    }

    // Claim and unpublish logical ownership in this stack. Recovery and run()
    // must not observe a lost current session during a prior invalidation's
    // queued physical release. A loss of an old cross-device guard leaves the
    // healthy current session published while still claiming that guard once.
    let cleanup = Promise.resolve();
    if (affectsCurrent) {
      ++initGeneration;
      ++lifecycleGeneration;
      ++sessionGeneration;
      session = null;
      device = null;
      runBusy = false;
      cleanup = destroyGpuResources(lostDevice);
    }
    const claimedSessions = new Set([...lostDeferred, oldSession].filter(Boolean));
    const claimed = affectsCurrent || claimedSessions.size > 0;
    const operation = deviceInvalidationTail.catch(() => {}).then(async () => {
      if (!claimed) return false;
      const failures = [];
      try { await cleanup; } catch (error) { failures.push(error); }
      for (const candidate of claimedSessions) {
        try { await candidate?.release?.(); }
        catch (error) {
          failures.push(error);
          warn("neural: device-loss session release failed:", error.message);
        }
      }
      if (failures.length) throw new AggregateError(failures, "neural device-loss cleanup failed");
      return true;
    });
    deviceInvalidationTail = operation.then(() => undefined, () => undefined);
    // A GPUDevice cannot recover after `lost` settles. Retain its exact cleanup
    // promise in a WeakMap so every observer (provider watcher, renderer, and
    // disposal path) sees the claiming operation's result, including failures.
    return exposeDeviceInvalidation(lostDevice, operation, previousBarrier);
  }

  // src: external texture (default) or { tex } for the texture_2d twin.
  // Returns { tex, outW, outH } or throws.
  function validateOutputTensor(tensor, padW, padH, scale, expectedOutputName) {
    if (!tensor?.gpuBuffer) throw new Error(`output not on GPU (${expectedOutputName})`);
    const type = tensor.type || tensor.dataType;
    if (type !== "float32") throw new Error(`unsupported neural output dtype '${type || "unknown"}' (expected float32)`);

    const expectedDims = [1, 3, padH * scale, padW * scale];
    const dims = Array.from(tensor.dims || [], Number);
    if (dims.length !== expectedDims.length || dims.some((value, index) => value !== expectedDims[index])) {
      throw new Error(`neural output shape [${dims.join(",")}] does not match manifest scale/padding [${expectedDims.join(",")}]`);
    }
    const elements = expectedDims.reduce((product, value) => product * value, 1);
    if (!Number.isSafeInteger(elements)) throw new Error("neural output tensor size exceeds safe integer range");
    if (tensor.size != null && Number(tensor.size) !== elements) {
      throw new Error(`neural output element count ${tensor.size} does not match expected ${elements}`);
    }
    const expectedBytes = elements * 4;
    const bufferBytes = Number(tensor.gpuBuffer.size);
    if (!Number.isFinite(bufferBytes) || bufferBytes < expectedBytes) {
      throw new Error(`neural output GPU buffer is ${bufferBytes || 0} bytes; expected at least ${expectedBytes}`);
    }
    return { strideW: expectedDims[3], plane: expectedDims[2] * expectedDims[3] };
  }

  async function run(src, srcW, srcH) {
    if (!session || !device) throw new Error("neural engine not initialized");
    if (runBusy) throw new Error("neural inference already in progress");
    if (!Number.isInteger(srcW) || !Number.isInteger(srcH) || srcW <= 0 || srcH <= 0) {
      throw new Error(`invalid neural input dimensions ${srcW}x${srcH}`);
    }

    const runSession = session;
    const runDevice = device;
    const runOrt = ort;
    const runEntry = active;
    const runInputName = inputName;
    const runOutputName = outputName;
    const runSessionGeneration = sessionGeneration;
    const runLifecycleGeneration = lifecycleGeneration;
    const mult = Math.max(1, runEntry.padMultiple | 0 || 1);
    const padW = Math.ceil(srcW / mult) * mult;
    const padH = Math.ceil(srcH / mult) * mult;
    const scale = runEntry.scale;
    const outW = srcW * scale, outH = srcH * scale;
    validateAllocationLimits(padW, padH, outW, outH, scale);
    const t0 = performance.now();
    runBusy = true;
    activeRuns++;
    let inTensor = null, outT = null;

    try {
      ensurePipelines();
      ensureInBuf(padW, padH);
      ensureOutTex(outW, outH);
      const runInBuf = inBuf;
      const runPackU = packU;
      const runCompU = compU;
      const runPackExtPipe = packExtPipe;
      const runPackTexPipe = packTexPipe;
      const runCompPipe = compPipe;
      const runSampler = sampler;
      const runOutTex = outTex;
      runDevice.queue.writeBuffer(runPackU, 0, new Uint32Array([padW, padH, srcW, srcH]));

      // Pack must be submitted before any await because external textures expire at
      // task end. Every object used after the await is captured for this generation.
      {
        const isTex = src && src.tex;
        const pipe = isTex ? runPackTexPipe : runPackExtPipe;
        const enc = runDevice.createCommandEncoder();
        const bg = runDevice.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: runSampler },
            { binding: 1, resource: isTex ? src.tex.createView() : src },
            { binding: 2, resource: { buffer: runInBuf } },
            { binding: 3, resource: { buffer: runPackU } },
          ],
        });
        const cp = enc.beginComputePass();
        cp.setPipeline(pipe); cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(Math.ceil(padW / 8), Math.ceil(padH / 8));
        cp.end();
        runDevice.queue.submit([enc.finish()]);
      }

      // The external source is now consumed on the new device. It is safe to let
      // sessions retaining a prior device go; main has adopted this device before
      // it can call run().
      await releaseDeferredSessions();
      if (runLifecycleGeneration !== lifecycleGeneration) throw new Error("neural inference cancelled by stop");

      inTensor = runOrt.Tensor.fromGpuBuffer(runInBuf, { dataType: "float32", dims: [1, 3, padH, padW] });
      const result = await runSession.run({ [runInputName]: inTensor });
      outT = result[runOutputName];
      if (runLifecycleGeneration !== lifecycleGeneration) throw new Error("neural inference cancelled by stop");
      if (runSessionGeneration !== sessionGeneration || runSession !== session) throw new Error("neural session changed during inference");
      const shape = validateOutputTensor(outT, padW, padH, scale, runOutputName);

      runDevice.queue.writeBuffer(runCompU, 0, new Uint32Array([shape.strideW, shape.plane, outW, outH]));
      const enc = runDevice.createCommandEncoder();
      const bg = runDevice.createBindGroup({
        layout: runCompPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outT.gpuBuffer } },
          { binding: 1, resource: runOutTex.createView() },
          { binding: 2, resource: { buffer: runCompU } },
        ],
      });
      const cp = enc.beginComputePass();
      cp.setPipeline(runCompPipe); cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(Math.ceil(outW / 8), Math.ceil(outH / 8));
      cp.end();
      runDevice.queue.submit([enc.finish()]);

      const dt = performance.now() - t0;
      stats.last = dt;
      stats.mu = stats.n === 0 ? dt : stats.mu * 0.9 + dt * 0.1;
      stats.n++;
      return { tex: runOutTex, outW, outH };
    } catch (error) {
      stats.fails++;
      throw error;
    } finally {
      // Both wrappers may still back submitted inference/composite commands. Their
      // disposal is deferred to a queue fence; the user-owned input buffer remains
      // owned by this engine and is retired independently on resize/stop.
      retireTensors(runDevice, inTensor, outT);
      runBusy = false;
      endRun();
    }
  }

  function stop() {
    // Release GPU resources but keep the SESSION alive: if this is the only
    // ORT session, releasing it tears down the shared device the upscaler has
    // adopted. Sessions persist until page unload (documented v1 tradeoff:
    // idle model VRAM in exchange for never orphaning the device).
    ++lifecycleGeneration;
    ++initGeneration;
    return destroyGpuResources();
  }

  async function quiesce() {
    // Cancel unpublished initialization and prevent active inference from
    // reaching a newer continuation, while retaining the committed session as
    // a device-lifetime guard until the main renderer releases its resources.
    ++lifecycleGeneration;
    ++initGeneration;
    const pendingInit = initTail;
    const cleanup = destroyGpuResources();
    try { await pendingInit; } catch {}
    await whenRunsIdle();
    try { await cleanup; } catch {}
    while (retirements.size) {
      try { await Promise.allSettled([...retirements]); } catch {}
    }
  }

  function dispose() {
    if (disposalPromise) return disposalPromise;

    // Publish the empty state synchronously. This makes ready()/device() truthful
    // while an active run, queue fence, or stale session creation is draining.
    ++lifecycleGeneration;
    ++initGeneration;
    ++sessionGeneration;
    const oldSession = session;
    const oldDevice = device;
    const oldDeferredSessions = deferredSessionReleases;
    const priorSessionReleaseFailures = sessionReleaseFailures;
    const pendingInit = initTail;
    const pendingDeviceInvalidation = deviceInvalidationPromise;
    session = null;
    device = null;
    active = null;
    inputName = "input";
    outputName = "output";
    fp16Model = false;
    runBusy = false;
    latestInitPromise = null;
    deferredSessionReleases = [];
    sessionReleaseFailures = [];
    const cleanup = destroyGpuResources(oldDevice);

    const operation = (async () => {
      const failures = new Set(priorSessionReleaseFailures);
      // invalidateDevice() may have unpublished the same session and retained
      // sole ownership of releasing it. Disposal is not complete until that
      // release is complete, even when there is nothing left to capture above.
      try { await pendingDeviceInvalidation; }
      catch (error) { failures.add(error); }
      // A watcher can publish another identity-specific invalidation while the
      // captured aggregate drains. Wait until the live set is empty, then consume
      // the settled-failure latch exactly once through this disposal report.
      while (activeDeviceInvalidations.size) {
        try { await refreshDeviceInvalidationBarrier(); }
        catch (error) { failures.add(error); }
      }
      deviceInvalidationFailures.clear();
      refreshDeviceInvalidationBarrier();
      try { await pendingInit; } catch {}
      await whenRunsIdle();
      try { await cleanup; } catch {}
      while (retirements.size) {
        try { await Promise.allSettled([...retirements]); } catch {}
      }
      // A losing initializer may transfer its prior committed session into the
      // replacement array while pendingInit drains. Close disposal over both
      // snapshots and the committed session, releasing each physical guard once.
      const sessionsToRelease = new Set([
        ...oldDeferredSessions,
        ...deferredSessionReleases,
        oldSession,
      ].filter(Boolean));
      deferredSessionReleases = [];
      for (const error of sessionReleaseFailures) failures.add(error);
      sessionReleaseFailures = [];
      for (const old of sessionsToRelease) {
        try { await old?.release?.(); }
        catch (error) {
          failures.add(error);
          warn("neural: session release failed:", error.message);
        }
      }
      if (failures.size) {
        throw new AggregateError([...failures], "neural session disposal failed");
      }
    })().finally(() => {
      if (disposalPromise === operation) disposalPromise = null;
    });
    disposalPromise = operation;
    // New init() calls are serialized behind physical session release.
    initTail = operation.then(() => undefined, () => undefined);
    return operation;
  }

  return {
    models: loadManifest,
    init,
    run,
    stop,
    quiesce,
    dispose,
    invalidateDevice,
    ready: () => !!(session && device),
    activeEntry: () => active,
    device: () => device,
    stats: () => ({ ...stats }),
    bumpSkip: () => { stats.skip++; },
  };
}

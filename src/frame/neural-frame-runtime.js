// Runs Neural ONNX in an extension-page realm. Some page/content-script
// contexts prohibit WebAssembly code generation even when the extension CSP
// permits it, so the page-facing renderer transfers frames through a private
// MessagePort to this frame.

export const NEURAL_FRAME_CHANNEL = "fsrcnnx-neural-frame-v1";

const NONCE_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;
const ERROR_CODE_PATTERN = /^[a-z0-9][a-z0-9-]{0,47}$/;
const MODEL_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const RESET_REASON_PATTERN = /^[a-z][a-z0-9-]{0,47}$/;
const MAX_REQUEST_ID_LENGTH = 96;
const MAX_ERROR_MESSAGE_LENGTH = 320;
const MAX_SEEN_REQUEST_IDS = 256;
const SRGB_COLOR_SPACE = "srgb";
const SSIMDS_THRESHOLD = 1.05;

const BLIT_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>,3>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0)
  );
  var uv = array<vec2<f32>,3>(
    vec2<f32>(0.0,1.0), vec2<f32>(2.0,1.0), vec2<f32>(0.0,-1.0)
  );
  var out: VSOut;
  out.pos = vec4<f32>(p[i],0.0,1.0);
  out.uv = uv[i];
  return out;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, in.uv, 0.0);
}`;

class NeuralFrameError extends Error {
  constructor(code, message, { retryable = false, cause } = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "NeuralFrameError";
    this.code = code;
    this.retryable = retryable;
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new NeuralFrameError("invalid-request", `${label} must be a positive safe integer`);
  }
  return value;
}

function optionalBoolean(value, fallback, label) {
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") {
    throw new NeuralFrameError("invalid-request", `${label} must be a boolean`);
  }
  return value;
}

function requestIdIsValid(value) {
  return (Number.isSafeInteger(value) && value >= 0) ||
    (typeof value === "string" && value.length > 0 &&
      value.length <= MAX_REQUEST_ID_LENGTH);
}

function boundedText(value, fallback, maximum = MAX_ERROR_MESSAGE_LENGTH) {
  let text;
  try {
    text = typeof value === "string" ? value : String(value ?? "");
  } catch {
    text = "";
  }
  text = text.replace(/[\u0000-\u001f\u007f]+/g, " ").trim();
  if (!text) text = fallback;
  return text.slice(0, maximum);
}

function normalizedErrorCode(value, fallback) {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase().replace(/_/g, "-");
    if (ERROR_CODE_PATTERN.test(normalized)) return normalized;
  }
  return fallback;
}

export function serializeNeuralFrameError(error, fallbackCode = "internal-error") {
  const safeFallback = normalizedErrorCode(fallbackCode, "internal-error");
  let code = normalizedErrorCode(error?.code, safeFallback);
  const message = boundedText(error?.message, "Neural frame operation failed.");
  if (code === safeFallback && /device(?: was)? lost/i.test(message)) code = "device-lost";
  const retryable = typeof error?.retryable === "boolean"
    ? error.retryable
    : code === "device-lost" || code === "run-busy" || code === "cancelled";
  return Object.freeze({ code, message, retryable });
}

function assertAllowedKeys(value, allowed, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new NeuralFrameError(
        "invalid-request",
        `${label} contains unsupported field '${boundedText(key, "unknown", 64)}'`,
      );
    }
  }
}

function validateEmptyPayload(value, label) {
  if (value === undefined) return;
  if (!isObject(value)) {
    throw new NeuralFrameError("invalid-request", `${label} payload must be an object`);
  }
  assertAllowedKeys(value, new Set(), `${label} payload`);
}

function safeCloseBitmap(bitmap) {
  try { bitmap?.close?.(); } catch {}
}

function safeDestroy(resource) {
  try { resource?.destroy?.(); } catch {}
}

function safeUnconfigure(context) {
  try { context?.unconfigure?.(); } catch {}
}

function createLocalOffscreenCanvas(width, height) {
  const documentCanvas = globalThis.document?.getElementById?.("neural-frame-output");
  if (documentCanvas && typeof documentCanvas.getContext === "function") {
    documentCanvas.width = width;
    documentCanvas.height = height;
    return documentCanvas;
  }
  const Constructor = globalThis.OffscreenCanvas;
  if (typeof Constructor !== "function") {
    throw new NeuralFrameError(
      "webgpu-unavailable",
      "A WebGPU presentation canvas is unavailable in the extension frame",
    );
  }
  return new Constructor(width, height);
}

function createLocalUploadCanvas(width, height) {
  const Constructor = globalThis.OffscreenCanvas;
  if (typeof Constructor === "function") return new Constructor(width, height);
  const documentCanvas = globalThis.document?.createElement?.("canvas");
  if (!documentCanvas) {
    throw new NeuralFrameError(
      "webgpu-unavailable",
      "A local canvas is unavailable for Neural source uploads",
    );
  }
  documentCanvas.width = width;
  documentCanvas.height = height;
  return documentCanvas;
}

function defaultImageBitmapCheck(value) {
  const Constructor = globalThis.ImageBitmap;
  return typeof Constructor === "function" && value instanceof Constructor;
}

function defaultVideoFrameCheck(value) {
  const Constructor = globalThis.VideoFrame;
  return typeof Constructor === "function" && value instanceof Constructor;
}

async function loadRuntimeDependencies() {
  const [neural, ssimds, sharpen, cdaPriors] = await Promise.all([
    import("../core/fsrcnnx-neural.js"),
    import("../core/fsrcnnx-ssimds-runtime.js"),
    import("../core/fsrcnnx-sharpen.js"),
    import("../core/fsrcnnx-cda-priors.js"),
  ]);
  return {
    createNeuralEngine: neural.createNeuralEngine,
    SsimDownscaler: ssimds.SsimDownscaler,
    buildSharpenShader: sharpen.buildSharpenShader,
    createCdaPriorGenerator: cdaPriors.createCdaPriorGenerator,
    CdaTemporalTracker: cdaPriors.CdaTemporalTracker,
  };
}

function contractUsesProvider(contract, provider) {
  if (contract?.version !== 2 || !contract.graphs ||
      typeof contract.graphs !== "object") return false;
  return Object.values(contract.graphs).some((graph) =>
    Object.values(graph?.inputs || {}).some(
      (descriptor) => descriptor?.provider === provider,
    ));
}

function cdaAuxiliaryBindings(priors) {
  const provider = priors?.provider;
  if (provider !== "decoded-cda-v1") {
    throw new NeuralFrameError(
      "inference-failed",
      "The decoded CDA prior provider returned an incompatible result",
    );
  }
  return Object.freeze({
    motion: Object.freeze({
      gpuBuffer: priors.motion,
      dataType: "float32",
      dims: priors.motionDims,
      provider,
    }),
    residual: Object.freeze({
      gpuBuffer: priors.residual,
      dataType: "float32",
      dims: priors.residualDims,
      provider,
    }),
  });
}

function sanitizeModel(entry) {
  if (!isObject(entry)) return null;
  const model = {
    key: boundedText(entry.key, "unknown", 64),
    label: boundedText(entry.label || entry.key, "Neural", 160),
    scale: Number.isSafeInteger(entry.scale) && entry.scale > 0 ? entry.scale : 1,
  };
  return Object.freeze(model);
}

function sanitizeEngineStats(stats) {
  if (!isObject(stats)) return null;
  const result = {};
  for (const key of [
    "last", "mu", "n", "skip", "fails", "lastTiles", "tileRuns",
    "maxTileW", "maxTileH",
  ]) {
    const value = Number(stats[key]);
    if (Number.isFinite(value) && value >= 0) result[key] = value;
  }
  return Object.freeze(result);
}

function dimensions(width, height) {
  return Object.freeze({ width, height });
}

function validatePresentation(value) {
  if (value === undefined) value = {};
  if (!isObject(value)) {
    throw new NeuralFrameError("invalid-request", "presentation must be an object");
  }
  assertAllowedKeys(value, new Set([
    "width",
    "height",
    "alphaMode",
    "ssimdsEnabled",
    "sharpenEnabled",
    "sharpenStrength",
  ]), "presentation");
  const hasWidth = value.width !== undefined;
  const hasHeight = value.height !== undefined;
  if (hasWidth !== hasHeight) {
    throw new NeuralFrameError(
      "invalid-request",
      "presentation width and height must be supplied together",
    );
  }
  const width = hasWidth ? positiveInteger(value.width, "presentation width") : null;
  const height = hasHeight ? positiveInteger(value.height, "presentation height") : null;
  const alphaMode = value.alphaMode ?? "opaque";
  if (alphaMode !== "opaque" && alphaMode !== "premultiplied") {
    throw new NeuralFrameError(
      "invalid-request",
      "presentation.alphaMode must be 'opaque' or 'premultiplied'",
    );
  }
  const ssimdsEnabled = optionalBoolean(
    value.ssimdsEnabled,
    false,
    "presentation.ssimdsEnabled",
  );
  const sharpenEnabled = optionalBoolean(
    value.sharpenEnabled,
    false,
    "presentation.sharpenEnabled",
  );
  let sharpenStrength = value.sharpenStrength ?? 1;
  if (typeof sharpenStrength !== "number" || !Number.isFinite(sharpenStrength) ||
      sharpenStrength < 0.1 || sharpenStrength > 2) {
    throw new NeuralFrameError(
      "invalid-request",
      "presentation.sharpenStrength must be between 0.1 and 2",
    );
  }
  return Object.freeze({
    width,
    height,
    alphaMode,
    ssimdsEnabled,
    sharpenEnabled,
    sharpenStrength,
  });
}

function validateTemporal(value) {
  if (value === undefined) value = {};
  if (!isObject(value)) {
    throw new NeuralFrameError("invalid-request", "temporal metadata must be an object");
  }
  assertAllowedKeys(
    value,
    new Set(["mediaTime", "presentedFrames", "reset", "resetReason"]),
    "temporal metadata",
  );
  const temporal = {};
  if (value.mediaTime !== undefined) {
    if (!Number.isFinite(value.mediaTime) ||
        value.mediaTime < 0 || value.mediaTime > Number.MAX_SAFE_INTEGER) {
      throw new NeuralFrameError(
        "invalid-request",
        "temporal.mediaTime must be a finite non-negative number",
      );
    }
    temporal.mediaTime = value.mediaTime;
  }
  if (value.presentedFrames !== undefined) {
    if (!Number.isSafeInteger(value.presentedFrames) || value.presentedFrames < 0) {
      throw new NeuralFrameError(
        "invalid-request",
        "temporal.presentedFrames must be a non-negative safe integer",
      );
    }
    temporal.presentedFrames = value.presentedFrames;
  }
  if (value.reset !== undefined) {
    if (typeof value.reset !== "boolean") {
      throw new NeuralFrameError(
        "invalid-request",
        "temporal.reset must be a boolean",
      );
    }
    temporal.reset = value.reset;
  }
  if (value.resetReason !== undefined) {
    if (typeof value.resetReason !== "string" ||
        !RESET_REASON_PATTERN.test(value.resetReason)) {
      throw new NeuralFrameError(
        "invalid-request",
        "temporal.resetReason must be a bounded lowercase identifier",
      );
    }
    temporal.resetReason = value.resetReason;
  }
  return Object.freeze(temporal);
}

export function createNeuralFrameSession({
  loadDependencies = loadRuntimeDependencies,
  gpu = globalThis.navigator?.gpu,
  textureUsage = globalThis.GPUTextureUsage,
  bufferUsage = globalThis.GPUBufferUsage,
  mapMode = globalThis.GPUMapMode,
  ImageDataCtor = globalThis.ImageData,
  createOffscreenCanvas = createLocalOffscreenCanvas,
  createUploadCanvas = createLocalUploadCanvas,
  createImageBitmapImpl = globalThis.createImageBitmap,
  isImageBitmap = defaultImageBitmapCheck,
  isVideoFrame = defaultVideoFrameCheck,
  now = () => globalThis.performance.now(),
  log = (...args) => console.log(...args),
  warn = (...args) => console.warn(...args),
  onDeviceLost = () => {},
} = {}) {
  let dependenciesPromise = null;
  let dependencies = null;
  let engine = null;
  let canvas = null;
  let context = null;
  let initialized = false;
  let disposed = false;
  let device = null;
  let lifecycleGeneration = 0;
  let deviceGeneration = 0;
  let deviceObserver = null;
  let contextDevice = null;
  let canvasFormat = null;
  let canvasAlphaMode = null;
  let presentationDevice = null;
  let presentationFormat = null;
  let sampler = null;
  let blitPipeline = null;
  let sharpenPipeline = null;
  let sharpenStrengthBuilt = null;
  let ssimds = null;
  let sourceTexture = null;
  let sourceTextureWidth = 0;
  let sourceTextureHeight = 0;
  let cdaPriorGenerator = null;
  let cdaTemporalTracker = null;
  let cdaNeedsReset = false;
  let uploadCanvas = null;
  let uploadContext = null;
  let readbackTexture = null;
  let readbackBuffer = null;
  let readbackDevice = null;
  let readbackWidth = 0;
  let readbackHeight = 0;
  let readbackBytesPerRow = 0;
  const retirements = new Set();

  const stats = {
    attaches: 0,
    inits: 0,
    runs: 0,
    stops: 0,
    failures: 0,
    deviceLosses: 0,
    ssimdsRuns: 0,
    ssimdsBypasses: 0,
    sharpenRuns: 0,
    externalCopies: 0,
    stagedUploads: 0,
    externalCopyFailures: 0,
    cdaPriorRuns: 0,
    cdaPriorResets: 0,
    lastRunMs: 0,
    meanRunMs: 0,
  };

  function snapshotStats() {
    return Object.freeze({
      ...stats,
      attached: !!canvas,
      initialized: initialized && !!engine?.ready?.() && !!device,
      disposed,
      engine: sanitizeEngineStats(engine?.stats?.()),
    });
  }

  async function ensureDependencies() {
    if (dependencies) return dependencies;
    if (!dependenciesPromise) {
      dependenciesPromise = Promise.resolve().then(loadDependencies).then((loaded) => {
        if (typeof loaded?.createNeuralEngine !== "function" ||
            typeof loaded?.SsimDownscaler !== "function" ||
            typeof loaded?.buildSharpenShader !== "function") {
          throw new NeuralFrameError(
            "runtime-unavailable",
            "Neural frame dependencies are incomplete",
          );
        }
        dependencies = loaded;
        return loaded;
      }).catch((error) => {
        dependenciesPromise = null;
        throw error;
      });
    }
    return dependenciesPromise;
  }

  function requireLive() {
    if (disposed) throw new NeuralFrameError("disposed", "Neural frame is disposed");
  }

  function cancelledError(message = "Neural frame operation was cancelled") {
    return new NeuralFrameError(
      "cancelled",
      message,
      { retryable: true },
    );
  }

  function cancel() {
    if (disposed) return false;
    ++lifecycleGeneration;
    initialized = false;
    // This hook only advances the core engine's init/run generations. GPU
    // destruction remains owned by the serialized stop/dispose request.
    engine?.cancel?.();
    return true;
  }

  function deviceLimit(ownerDevice = device) {
    const limit = Number(ownerDevice?.limits?.maxTextureDimension2D);
    return Number.isSafeInteger(limit) && limit > 0 ? limit : null;
  }

  function validateDeviceDimensions(width, height, label, ownerDevice = device) {
    const limit = deviceLimit(ownerDevice);
    if (limit !== null && (width > limit || height > limit)) {
      throw new NeuralFrameError(
        "resource-limit",
        `${label} exceeds the device texture limit ${limit}`,
      );
    }
  }

  function retireTexture(texture, ownerDevice = device) {
    if (!texture) return;
    let fence;
    try { fence = ownerDevice?.queue?.onSubmittedWorkDone?.() || Promise.resolve(); }
    catch { fence = Promise.resolve(); }
    const operation = Promise.resolve(fence).catch(() => {}).then(() => safeDestroy(texture));
    retirements.add(operation);
    operation.finally(() => retirements.delete(operation)).catch(() => {});
  }

  async function drainRetirements() {
    while (retirements.size) await Promise.allSettled([...retirements]);
  }

  function detachDeviceObserver() {
    if (deviceObserver) deviceObserver.target = null;
    deviceObserver = null;
  }

  function releaseFrameResourcesSynchronously({ unconfigure = true } = {}) {
    const oldSource = sourceTexture;
    const oldReadbackTexture = readbackTexture;
    const oldReadbackBuffer = readbackBuffer;
    const oldCdaPriorGenerator = cdaPriorGenerator;
    sourceTexture = null;
    sourceTextureWidth = 0;
    sourceTextureHeight = 0;
    cdaPriorGenerator = null;
    cdaTemporalTracker = null;
    cdaNeedsReset = false;
    uploadCanvas = null;
    uploadContext = null;
    readbackTexture = null;
    readbackBuffer = null;
    readbackDevice = null;
    readbackWidth = 0;
    readbackHeight = 0;
    readbackBytesPerRow = 0;
    safeDestroy(oldSource);
    safeDestroy(oldReadbackTexture);
    safeDestroy(oldReadbackBuffer);
    try { oldCdaPriorGenerator?.dispose?.(); } catch {}
    try { ssimds?.destroy?.(); } catch {}
    ssimds = null;
    sampler = null;
    blitPipeline = null;
    sharpenPipeline = null;
    sharpenStrengthBuilt = null;
    presentationDevice = null;
    presentationFormat = null;
    contextDevice = null;
    canvasFormat = null;
    canvasAlphaMode = null;
    if (unconfigure) safeUnconfigure(context);
  }

  async function releaseFrameResources({ unconfigure = true } = {}) {
    const ownerDevice = device;
    try { await ownerDevice?.queue?.onSubmittedWorkDone?.(); } catch {}
    releaseFrameResourcesSynchronously({ unconfigure });
    await drainRetirements();
  }

  function observeDevice(ownerDevice) {
    detachDeviceObserver();
    const generation = ++deviceGeneration;
    if (!ownerDevice?.lost?.then) return;
    const observer = { target: handleDeviceLoss };
    deviceObserver = observer;
    ownerDevice.lost.then(
      (info) => observer.target?.(ownerDevice, generation, info),
      (error) => observer.target?.(ownerDevice, generation, {
        reason: "unknown",
        message: error?.message || String(error),
      }),
    );
  }

  function handleDeviceLoss(ownerDevice, generation, info) {
    if (disposed || generation !== deviceGeneration || ownerDevice !== device) return;
    initialized = false;
    stats.deviceLosses++;
    releaseFrameResourcesSynchronously();
    device = null;
    detachDeviceObserver();
    const detail = serializeNeuralFrameError(
      new NeuralFrameError(
        "device-lost",
        `Neural GPU device was lost: ${info?.message || info?.reason || "unknown reason"}`,
        { retryable: true },
      ),
      "device-lost",
    );
    try { onDeviceLost(detail, snapshotStats()); } catch {}
    Promise.resolve(engine?.invalidateDevice?.(ownerDevice)).catch((error) => {
      warn("neural frame device-loss cleanup failed:", boundedText(error?.message, "unknown"));
    });
  }

  function ensureContextConfigured(ownerDevice, width, height, alphaMode) {
    if (!canvas || !context) {
      throw new NeuralFrameError("not-attached", "No presentation canvas is attached");
    }
    validateDeviceDimensions(width, height, "presentation", ownerDevice);
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    if (contextDevice === ownerDevice && canvasFormat && canvasAlphaMode === alphaMode) {
      return canvasFormat;
    }
    if (!gpu || typeof gpu.getPreferredCanvasFormat !== "function") {
      throw new NeuralFrameError("webgpu-unavailable", "WebGPU canvas support is unavailable");
    }
    if (!textureUsage || !Number.isFinite(textureUsage.RENDER_ATTACHMENT)) {
      throw new NeuralFrameError(
        "webgpu-unavailable",
        "WebGPU texture usage constants are unavailable",
      );
    }
    const format = gpu.getPreferredCanvasFormat();
    context.configure({
      device: ownerDevice,
      format,
      colorSpace: SRGB_COLOR_SPACE,
      alphaMode,
      usage: textureUsage.RENDER_ATTACHMENT,
    });
    contextDevice = ownerDevice;
    canvasFormat = format;
    canvasAlphaMode = alphaMode;
    return format;
  }

  function readbackSupported() {
    return Number.isFinite(textureUsage?.RENDER_ATTACHMENT) &&
      Number.isFinite(textureUsage?.COPY_SRC) &&
      Number.isFinite(bufferUsage?.COPY_DST) &&
      Number.isFinite(bufferUsage?.MAP_READ) &&
      Number.isFinite(mapMode?.READ) &&
      typeof ImageDataCtor === "function" &&
      typeof createImageBitmapImpl === "function";
  }

  function ensureReadbackResources(ownerDevice, width, height) {
    validateDeviceDimensions(width, height, "presentation", ownerDevice);
    const rowBytes = width * 4;
    const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
    const bufferBytes = bytesPerRow * height;
    const maxBufferSize = Number(ownerDevice?.limits?.maxBufferSize);
    if (!Number.isSafeInteger(rowBytes) || !Number.isSafeInteger(bytesPerRow) ||
        !Number.isSafeInteger(bufferBytes) || bytesPerRow > 0xffffffff ||
        (Number.isFinite(maxBufferSize) && bufferBytes > maxBufferSize)) {
      throw new NeuralFrameError(
        "resource-limit",
        "Neural presentation readback exceeds the GPU buffer limit",
      );
    }
    if (readbackTexture && readbackBuffer && readbackDevice === ownerDevice &&
        readbackWidth === width && readbackHeight === height &&
        readbackBytesPerRow === bytesPerRow) {
      return;
    }
    const nextTexture = ownerDevice.createTexture({
      label: `neural-frame-readback-${width}x${height}`,
      size: { width, height },
      format: "rgba8unorm",
      usage: textureUsage.RENDER_ATTACHMENT | textureUsage.COPY_SRC,
    });
    let nextBuffer;
    try {
      nextBuffer = ownerDevice.createBuffer({
        label: `neural-frame-readback-${bufferBytes}`,
        size: bufferBytes,
        usage: bufferUsage.COPY_DST | bufferUsage.MAP_READ,
      });
    } catch (error) {
      safeDestroy(nextTexture);
      throw error;
    }
    safeDestroy(readbackTexture);
    safeDestroy(readbackBuffer);
    readbackTexture = nextTexture;
    readbackBuffer = nextBuffer;
    readbackDevice = ownerDevice;
    readbackWidth = width;
    readbackHeight = height;
    readbackBytesPerRow = bytesPerRow;
  }

  function assertRunGeneration(
    ownerDevice,
    generation,
    runLifecycleGeneration,
    cancellationMessage,
    deviceMessage,
  ) {
    if (disposed || runLifecycleGeneration !== lifecycleGeneration) {
      throw cancelledError(cancellationMessage);
    }
    if (ownerDevice !== device || generation !== deviceGeneration) {
      throw new NeuralFrameError(
        "device-lost",
        deviceMessage,
        { retryable: true },
      );
    }
    if (!initialized) throw cancelledError(cancellationMessage);
  }

  async function captureReadbackBitmap(
    width,
    height,
    ownerDevice,
    generation,
    runLifecycleGeneration,
  ) {
    const buffer = readbackBuffer;
    const bytesPerRow = readbackBytesPerRow;
    try {
      if (!buffer) {
        throw new NeuralFrameError(
          "presentation-failed",
          "Neural readback buffer is unavailable",
        );
      }
      await buffer.mapAsync(mapMode.READ);
      assertRunGeneration(
        ownerDevice,
        generation,
        runLifecycleGeneration,
        "Neural output readback was cancelled",
        "Neural GPU device changed while reading the output frame",
      );
      const mapped = new Uint8Array(buffer.getMappedRange());
      const packed = new Uint8ClampedArray(width * height * 4);
      const rowBytes = width * 4;
      for (let row = 0; row < height; row++) {
        packed.set(
          mapped.subarray(
            row * bytesPerRow,
            row * bytesPerRow + rowBytes,
          ),
          row * rowBytes,
        );
      }
      let image;
      try {
        image = new ImageDataCtor(packed, width, height, {
          colorSpace: SRGB_COLOR_SPACE,
        });
      } catch {
        image = new ImageDataCtor(packed, width, height);
      }
      const bitmap = await createImageBitmapImpl(image);
      if (!isImageBitmap(bitmap) || bitmap.width !== width || bitmap.height !== height) {
        safeCloseBitmap(bitmap);
        throw new NeuralFrameError(
          "presentation-failed",
          "Neural readback produced an invalid output bitmap",
        );
      }
      try {
        assertRunGeneration(
          ownerDevice,
          generation,
          runLifecycleGeneration,
          "Neural output export was cancelled",
          "Neural GPU device changed while exporting the output frame",
        );
      } catch (error) {
        safeCloseBitmap(bitmap);
        throw error;
      }
      return bitmap;
    } catch (error) {
      if (error instanceof NeuralFrameError) throw error;
      throw new NeuralFrameError(
        "presentation-failed",
        "Neural output readback failed",
        { cause: error, retryable: true },
      );
    } finally {
      try { buffer?.unmap?.(); } catch {}
    }
  }

  function ensurePresentationResources(ownerDevice, format) {
    if (sampler && blitPipeline &&
        presentationDevice === ownerDevice && presentationFormat === format) return;
    const candidateSampler = ownerDevice.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    const module = ownerDevice.createShaderModule({ code: BLIT_WGSL });
    const candidateBlit = ownerDevice.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    sampler = candidateSampler;
    blitPipeline = candidateBlit;
    presentationDevice = ownerDevice;
    presentationFormat = format;
    sharpenPipeline = null;
    sharpenStrengthBuilt = null;
    try { ssimds?.destroy?.(); } catch {}
    ssimds = null;
  }

  function ensureSourceTexture(ownerDevice, width, height) {
    if (sourceTexture && sourceTextureWidth === width && sourceTextureHeight === height) {
      return sourceTexture;
    }
    const candidate = ownerDevice.createTexture({
      label: `neural-frame-source-${width}x${height}`,
      size: { width, height },
      format: "rgba8unorm",
      usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST,
    });
    const old = sourceTexture;
    sourceTexture = candidate;
    sourceTextureWidth = width;
    sourceTextureHeight = height;
    retireTexture(old, ownerDevice);
    return candidate;
  }

  function stageSourceFrame(frame, width, height) {
    if (!uploadCanvas || !uploadContext) {
      uploadCanvas = createUploadCanvas(width, height);
      uploadContext = uploadCanvas?.getContext?.("2d", {
        alpha: false,
        colorSpace: SRGB_COLOR_SPACE,
        willReadFrequently: true,
      });
      if (!uploadCanvas || !uploadContext) {
        uploadCanvas = null;
        uploadContext = null;
        throw new NeuralFrameError(
          "presentation-failed",
          "The extension frame could not create its Neural source staging canvas",
        );
      }
    }
    if (uploadCanvas.width !== width) uploadCanvas.width = width;
    if (uploadCanvas.height !== height) uploadCanvas.height = height;
    uploadContext.globalCompositeOperation = "copy";
    uploadContext.drawImage(frame, 0, 0, width, height);
    // This compatibility path materializes child-owned pixels when Chromium
    // cannot import a transferred ImageBitmap or VideoFrame into WebGPU.
    try {
      return uploadContext.getImageData(0, 0, width, height, {
        colorSpace: SRGB_COLOR_SPACE,
      }).data;
    } catch {
      return uploadContext.getImageData(0, 0, width, height).data;
    }
  }

  function uploadSourceFrame(
    ownerDevice,
    frame,
    source,
    width,
    height,
    { directExternalCopy = false } = {},
  ) {
    const queue = ownerDevice?.queue;
    // A transferred ImageBitmap can be accepted here yet import as all-zero
    // pixels across Chromium's OOPIF boundary. Decoder-backed VideoFrame is the
    // only direct path; ImageBitmap retains the child-owned staging path.
    if (directExternalCopy &&
        typeof queue?.copyExternalImageToTexture === "function") {
      try {
        queue.copyExternalImageToTexture(
          { source: frame },
          { texture: source, colorSpace: SRGB_COLOR_SPACE },
          { width, height },
        );
        stats.externalCopies++;
        return;
      } catch {
        // A transferred external image can be unsupported for a particular
        // Chromium/GPU combination even when the queue method exists. Preserve
        // the deterministic child-owned pixel upload as a compatibility path.
        stats.externalCopyFailures++;
      }
    }
    const uploadPixels = stageSourceFrame(frame, width, height);
    queue.writeTexture(
      { texture: source },
      uploadPixels,
      { bytesPerRow: width * 4, rowsPerImage: height },
      { width, height },
    );
    stats.stagedUploads++;
  }

  function ensureSharpenPipeline(ownerDevice, format, strength) {
    if (sharpenPipeline && sharpenStrengthBuilt === strength) return sharpenPipeline;
    const module = ownerDevice.createShaderModule({
      code: dependencies.buildSharpenShader(strength, false),
    });
    const candidate = ownerDevice.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    sharpenPipeline = candidate;
    sharpenStrengthBuilt = strength;
    return candidate;
  }

  function shouldApplySsimds(modelWidth, modelHeight, outputWidth, outputHeight, enabled) {
    return enabled &&
      outputWidth <= modelWidth &&
      outputHeight <= modelHeight &&
      (modelWidth > outputWidth * SSIMDS_THRESHOLD ||
        modelHeight > outputHeight * SSIMDS_THRESHOLD);
  }

  async function capturePresentedBitmap(
    width,
    height,
    ownerDevice,
    generation,
    runLifecycleGeneration,
  ) {
    try {
      await ownerDevice?.queue?.onSubmittedWorkDone?.();
    } catch (error) {
      throw new NeuralFrameError(
        "device-lost",
        "Neural GPU work did not finish before output export",
        { cause: error, retryable: true },
      );
    }
    assertRunGeneration(
      ownerDevice,
      generation,
      runLifecycleGeneration,
      "Neural output export was cancelled",
      "Neural GPU device changed before exporting the output frame",
    );
    let bitmap = null;
    let transferError = null;
    // createImageBitmap works for both the extension document's HTML canvas and
    // the OffscreenCanvas compatibility fallback.
    if (typeof createImageBitmapImpl === "function") {
      try { bitmap = await createImageBitmapImpl(canvas, 0, 0, width, height); }
      catch (error) {
        transferError = error;
      }
    }
    if (!bitmap && typeof canvas?.transferToImageBitmap === "function") {
      try { bitmap = canvas.transferToImageBitmap(); }
      catch (error) { transferError ||= error; }
    }
    if (!bitmap) {
      throw new NeuralFrameError(
        "presentation-failed",
        "The extension frame could not snapshot its Neural output",
        { cause: transferError || undefined },
      );
    }
    if (!isImageBitmap(bitmap) || bitmap.width !== width || bitmap.height !== height) {
      safeCloseBitmap(bitmap);
      throw new NeuralFrameError(
        "presentation-failed",
        "The extension frame produced an invalid Neural output bitmap",
      );
    }
    try {
      assertRunGeneration(
        ownerDevice,
        generation,
        runLifecycleGeneration,
        "Neural output export was cancelled",
        "Neural GPU device changed while exporting the output frame",
      );
    } catch (error) {
      safeCloseBitmap(bitmap);
      throw error;
    }
    return bitmap;
  }

  async function attachCanvas(payload) {
    requireLive();
    if (payload === undefined) payload = {};
    if (!isObject(payload)) {
      throw new NeuralFrameError(
        "invalid-request",
        "attachCanvas payload must be an object",
      );
    }
    assertAllowedKeys(payload, new Set(), "attachCanvas payload");
    if (canvas && context) {
      return Object.freeze({ attached: true, stats: snapshotStats() });
    }
    let nextCanvas;
    let nextContext;
    try {
      nextCanvas = createOffscreenCanvas(1, 1);
      nextContext = nextCanvas?.getContext?.("webgpu");
    }
    catch (error) {
      throw new NeuralFrameError(
        "webgpu-unavailable",
        "Unable to create the extension frame's WebGPU canvas context",
        { cause: error },
      );
    }
    if (!nextContext) {
      throw new NeuralFrameError(
        "webgpu-unavailable",
        "The extension frame's WebGPU canvas context is unavailable",
      );
    }
    canvas = nextCanvas;
    context = nextContext;
    contextDevice = null;
    canvasFormat = null;
    canvasAlphaMode = null;
    stats.attaches++;
    return Object.freeze({ attached: true, stats: snapshotStats() });
  }

  async function initialize(payload) {
    requireLive();
    const initGeneration = lifecycleGeneration;
    if (!canvas || !context) {
      throw new NeuralFrameError("not-attached", "attachCanvas must run before init");
    }
    if (payload === undefined) payload = {};
    if (!isObject(payload)) {
      throw new NeuralFrameError("invalid-request", "init payload must be an object");
    }
    assertAllowedKeys(payload, new Set(["modelKey"]), "init payload");
    const modelKey = payload.modelKey;
    if (modelKey !== undefined &&
        (typeof modelKey !== "string" || !MODEL_KEY_PATTERN.test(modelKey))) {
      throw new NeuralFrameError("invalid-request", "init.modelKey is invalid");
    }
    const loaded = await ensureDependencies();
    if (initGeneration !== lifecycleGeneration) throw cancelledError();
    if (!engine) engine = loaded.createNeuralEngine({ log, warn });
    const entry = await engine.init(modelKey);
    if (initGeneration !== lifecycleGeneration) throw cancelledError();
    const nextDevice = engine.device?.();
    if (!entry || !nextDevice || !engine.ready?.()) {
      throw new NeuralFrameError(
        "initialization-failed",
        "Neural engine did not publish a ready GPU device",
      );
    }
    if (device && device !== nextDevice) {
      await releaseFrameResources();
      if (initGeneration !== lifecycleGeneration) throw cancelledError();
    }
    const nextContract = engine.activeContract?.() ?? null;
    const needsDecodedCdaPriors = contractUsesProvider(
      nextContract,
      "decoded-cda-v1",
    );
    if (needsDecodedCdaPriors) {
      if (typeof loaded.createCdaPriorGenerator !== "function" ||
          typeof loaded.CdaTemporalTracker !== "function") {
        throw new NeuralFrameError(
          "runtime-unavailable",
          "Decoded CDA prior support is unavailable",
        );
      }
      try { cdaPriorGenerator?.dispose?.(); } catch {}
      cdaPriorGenerator = loaded.createCdaPriorGenerator(nextDevice);
      cdaTemporalTracker = new loaded.CdaTemporalTracker();
      cdaNeedsReset = false;
    } else {
      try { cdaPriorGenerator?.dispose?.(); } catch {}
      cdaPriorGenerator = null;
      cdaTemporalTracker = null;
      cdaNeedsReset = false;
    }
    device = nextDevice;
    initialized = true;
    observeDevice(nextDevice);
    stats.inits++;
    return Object.freeze({
      model: sanitizeModel(entry),
      stats: snapshotStats(),
    });
  }

  async function run(payload) {
    requireLive();
    if (!initialized || !engine?.ready?.() || !device) {
      throw new NeuralFrameError("not-initialized", "Neural frame is not initialized");
    }
    if (!isObject(payload)) {
      throw new NeuralFrameError("invalid-request", "run payload must be an object");
    }
    assertAllowedKeys(
      payload,
      new Set(["bitmap", "srcW", "srcH", "presentation", "temporal"]),
      "run payload",
    );
    const inputFrame = payload.bitmap;
    const videoFrame = isVideoFrame(inputFrame);
    const imageBitmap = isImageBitmap(inputFrame);
    if (!videoFrame && !imageBitmap) {
      throw new NeuralFrameError(
        "invalid-bitmap",
        "run requires a transferred ImageBitmap or VideoFrame",
      );
    }
    {
      const srcW = positiveInteger(payload.srcW, "run.srcW");
      const srcH = positiveInteger(payload.srcH, "run.srcH");
      const presentation = validatePresentation(payload.presentation);
      const temporal = validateTemporal(payload.temporal);
      const frameWidth = videoFrame ? inputFrame.displayWidth : inputFrame.width;
      const frameHeight = videoFrame ? inputFrame.displayHeight : inputFrame.height;
      if (frameWidth !== srcW || frameHeight !== srcH) {
        throw new NeuralFrameError(
          "invalid-bitmap",
          `Input frame dimensions ${frameWidth}x${frameHeight} do not match ${srcW}x${srcH}`,
        );
      }

      const runDevice = device;
      const runGeneration = deviceGeneration;
      const runLifecycleGeneration = lifecycleGeneration;
      validateDeviceDimensions(srcW, srcH, "source", runDevice);
      if (!textureUsage ||
          !Number.isFinite(textureUsage.TEXTURE_BINDING) ||
          !Number.isFinite(textureUsage.COPY_DST)) {
        throw new NeuralFrameError(
          "webgpu-unavailable",
          "WebGPU texture usage constants are unavailable",
        );
      }
      const source = ensureSourceTexture(runDevice, srcW, srcH);
      const started = now();
      uploadSourceFrame(runDevice, inputFrame, source, srcW, srcH, {
        directExternalCopy: videoFrame,
      });
      let effectiveTemporal = temporal;
      let engineOptions = { temporal };
      if (cdaPriorGenerator) {
        const recoveringFromFailedRun = cdaNeedsReset;
        // The prior generator snapshots the current frame before inference.
        // Keep this latch set until the output is actually published so any
        // failure cannot leave prior history ahead of recurrent model state.
        cdaNeedsReset = true;
        if (recoveringFromFailedRun) {
          cdaTemporalTracker.reset("previous-run-failed");
        }
        let boundary;
        if (Number.isFinite(temporal.mediaTime) &&
            Number.isSafeInteger(temporal.presentedFrames)) {
          boundary = cdaTemporalTracker.observe({
            mediaTime: temporal.mediaTime,
            presentedFrames: temporal.presentedFrames,
            width: srcW,
            height: srcH,
            forceReset: temporal.reset === true,
          });
        } else {
          cdaTemporalTracker.reset(temporal.resetReason || "metadata-unavailable");
          boundary = Object.freeze({
            reset: true,
            reason: temporal.resetReason || "metadata-unavailable",
            frameIndex: 0,
          });
        }
        const priors = cdaPriorGenerator.generate(source, srcW, srcH, {
          reset: boundary.reset,
        });
        const reset = boundary.reset || !priors.valid;
        const resetReason = reset
          ? (boundary.reason || temporal.resetReason || "prior-initialization")
          : undefined;
        effectiveTemporal = Object.freeze({
          ...temporal,
          reset,
          ...(resetReason ? { resetReason } : {}),
        });
        engineOptions = Object.freeze({
          temporal: effectiveTemporal,
          reset,
          auxiliary: cdaAuxiliaryBindings(priors),
        });
        stats.cdaPriorRuns++;
        if (reset) stats.cdaPriorResets++;
      }
      const rendered = await engine.run(
        { tex: source },
        srcW,
        srcH,
        engineOptions,
      );
      if (disposed || runLifecycleGeneration !== lifecycleGeneration) {
        throw cancelledError();
      }
      if (runDevice !== device || runGeneration !== deviceGeneration) {
        throw new NeuralFrameError(
          "device-lost",
          "Neural GPU device changed during inference",
          { retryable: true },
        );
      }
      if (!initialized) throw cancelledError();
      const modelWidth = positiveInteger(rendered?.outW, "neural output width");
      const modelHeight = positiveInteger(rendered?.outH, "neural output height");
      if (!rendered?.tex || typeof rendered.tex.createView !== "function") {
        throw new NeuralFrameError(
          "inference-failed",
          "Neural inference returned no GPU texture",
        );
      }
      const outputWidth = presentation.width ?? modelWidth;
      const outputHeight = presentation.height ?? modelHeight;
      validateDeviceDimensions(modelWidth, modelHeight, "neural output", runDevice);
      const useReadback = readbackSupported();
      if (useReadback) {
        ensureReadbackResources(runDevice, outputWidth, outputHeight);
      }
      const format = useReadback
        ? "rgba8unorm"
        : ensureContextConfigured(
          runDevice,
          outputWidth,
          outputHeight,
          presentation.alphaMode,
        );
      ensurePresentationResources(runDevice, format);

      const encoder = runDevice.createCommandEncoder({ label: "neural-frame-present" });
      let presentationTexture = rendered.tex;
      let ssimdsApplied = false;
      if (shouldApplySsimds(
        modelWidth,
        modelHeight,
        outputWidth,
        outputHeight,
        presentation.ssimdsEnabled,
      )) {
        if (!ssimds) ssimds = new dependencies.SsimDownscaler(runDevice);
        if (ssimds.prepare(
          modelWidth,
          modelHeight,
          outputWidth,
          outputHeight,
          rendered.tex,
        )) {
          presentationTexture = ssimds.run(encoder, rendered.tex);
          ssimdsApplied = true;
          stats.ssimdsRuns++;
        } else {
          stats.ssimdsBypasses++;
        }
      }

      const finalPipeline = presentation.sharpenEnabled
        ? ensureSharpenPipeline(
          runDevice,
          format,
          presentation.sharpenStrength,
        )
        : blitPipeline;
      const bindGroup = runDevice.createBindGroup({
        layout: finalPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: presentationTexture.createView() },
        ],
      });
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: useReadback
            ? readbackTexture.createView()
            : context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        }],
      });
      renderPass.setPipeline(finalPipeline);
      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(3);
      renderPass.end();
      if (useReadback) {
        encoder.copyTextureToBuffer(
          { texture: readbackTexture },
          {
            buffer: readbackBuffer,
            bytesPerRow: readbackBytesPerRow,
            rowsPerImage: outputHeight,
          },
          { width: outputWidth, height: outputHeight },
        );
      }
      runDevice.queue.submit([encoder.finish()]);

      const outputBitmap = useReadback
        ? await captureReadbackBitmap(
          outputWidth,
          outputHeight,
          runDevice,
          runGeneration,
          runLifecycleGeneration,
        )
        : await capturePresentedBitmap(
          outputWidth,
          outputHeight,
          runDevice,
          runGeneration,
          runLifecycleGeneration,
        );
      try {
        if (disposed || runLifecycleGeneration !== lifecycleGeneration) {
          throw cancelledError();
        }
        if (runDevice !== device || runGeneration !== deviceGeneration) {
          throw new NeuralFrameError(
            "device-lost",
            "Neural GPU device changed before publishing the output frame",
            { retryable: true },
          );
        }
        if (!initialized) throw cancelledError();
        if (presentation.sharpenEnabled) stats.sharpenRuns++;
        const elapsed = Math.max(0, now() - started);
        stats.lastRunMs = elapsed;
        stats.meanRunMs = stats.runs === 0
          ? elapsed
          : stats.meanRunMs * 0.9 + elapsed * 0.1;
        stats.runs++;
        const output = dimensions(outputWidth, outputHeight);
        const sharpenSource = ssimdsApplied
          ? output
          : dimensions(modelWidth, modelHeight);
        const diagnostics = Object.freeze({
          source: dimensions(srcW, srcH),
          output,
          ssimds: ssimdsApplied ? Object.freeze({
            source: dimensions(modelWidth, modelHeight),
            output,
          }) : null,
          sharpen: presentation.sharpenEnabled ? Object.freeze({
            source: sharpenSource,
            output,
            strength: presentation.sharpenStrength,
          }) : null,
        });
        const result = Object.freeze({
          srcW,
          srcH,
          modelWidth,
          modelHeight,
          temporal: effectiveTemporal,
          presentation: diagnostics,
          stats: snapshotStats(),
          bitmap: outputBitmap,
        });
        if (cdaPriorGenerator) cdaNeedsReset = false;
        return result;
      } catch (error) {
        safeCloseBitmap(outputBitmap);
        throw error;
      }
    }
  }

  async function stop() {
    requireLive();
    cancel();
    detachDeviceObserver();
    const failures = [];
    try { await releaseFrameResources(); } catch (error) { failures.push(error); }
    try { await engine?.stop?.(); } catch (error) { failures.push(error); }
    device = null;
    stats.stops++;
    if (failures.length) {
      throw new AggregateError(failures, "Neural frame stop cleanup failed");
    }
    return Object.freeze({ stopped: true, stats: snapshotStats() });
  }

  async function dispose() {
    if (disposed) return Object.freeze({ disposed: true, stats: snapshotStats() });
    cancel();
    disposed = true;
    detachDeviceObserver();
    const failures = [];
    try { await releaseFrameResources(); } catch (error) { failures.push(error); }
    try { await engine?.dispose?.(); } catch (error) { failures.push(error); }
    device = null;
    const oldCanvas = canvas;
    canvas = null;
    context = null;
    try { oldCanvas?.remove?.(); } catch {}
    if (failures.length) {
      throw new AggregateError(failures, "Neural frame disposal failed");
    }
    return Object.freeze({ disposed: true, stats: snapshotStats() });
  }

  async function handle(method, payload) {
    try {
      switch (method) {
        case "attachCanvas": return await attachCanvas(payload);
        case "init": return await initialize(payload);
        case "run":
          try { return await run(payload); }
          finally { safeCloseBitmap(payload?.bitmap); }
        case "stop":
          validateEmptyPayload(payload, "stop");
          return await stop();
        case "dispose":
          validateEmptyPayload(payload, "dispose");
          return await dispose();
        default:
          throw new NeuralFrameError(
            "unknown-method",
            `Unknown neural frame method '${boundedText(method, "unknown", 64)}'`,
          );
      }
    } catch (error) {
      stats.failures++;
      throw error;
    }
  }

  return Object.freeze({
    handle,
    cancel,
    dispose,
    stats: snapshotStats,
  });
}

function referrerIdentity(documentObject) {
  const referrer = documentObject?.referrer;
  if (!referrer) {
    return Object.freeze({ present: false, valid: true, origin: null, protocol: null });
  }
  try {
    const parsed = new URL(referrer);
    return Object.freeze({
      present: true,
      valid: true,
      origin: parsed.origin === "null" ? null : parsed.origin,
      protocol: parsed.protocol,
    });
  } catch {
    return Object.freeze({ present: true, valid: false, origin: null, protocol: null });
  }
}

function validWebOrigin(origin) {
  try {
    if (typeof origin !== "string" || !origin || origin === "null") return false;
    const parsed = new URL(origin);
    return (parsed.protocol === "https:" || parsed.protocol === "http:") &&
      parsed.origin === origin;
  } catch {
    return false;
  }
}

function fragmentCapability(locationObject) {
  try {
    const fragment = new URL(locationObject.href).hash;
    const parameters = new URLSearchParams(
      fragment.startsWith("#") ? fragment.slice(1) : fragment,
    );
    if ([...parameters.keys()].some(
      (key) => key !== "instanceNonce" &&
        key !== "frameCapability" &&
        key !== "opaqueParent",
    )) return null;
    const nonces = parameters.getAll("instanceNonce");
    const frameCapabilities = parameters.getAll("frameCapability");
    const opaqueParents = parameters.getAll("opaqueParent");
    if (nonces.length !== 1 || !NONCE_PATTERN.test(nonces[0]) ||
        frameCapabilities.length !== 1 ||
        !/^[a-f0-9]{48}$/.test(frameCapabilities[0]) ||
        opaqueParents.length > 1 ||
        (opaqueParents.length === 1 && opaqueParents[0] !== "1")) return null;
    return Object.freeze({
      instanceNonce: nonces[0],
      frameCapability: frameCapabilities[0],
      opaqueParent: opaqueParents.length === 1,
    });
  } catch {
    return null;
  }
}

function sendRuntimeMessage(runtime, message) {
  if (typeof runtime?.sendMessage !== "function") {
    return Promise.reject(new Error("chrome.runtime.sendMessage is unavailable"));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const complete = (callback) => (value) => {
      if (settled) return;
      settled = true;
      callback(value);
    };
    const succeed = complete(resolve);
    const fail = complete(reject);
    try {
      const pending = runtime.sendMessage(message, (response) => {
        let runtimeError = null;
        try { runtimeError = runtime.lastError; } catch {}
        if (runtimeError) {
          fail(new Error(runtimeError.message || "Extension message failed"));
        } else {
          succeed(response);
        }
      });
      if (pending && typeof pending.then === "function") {
        pending.then(succeed, fail);
      }
    } catch (error) {
      fail(error);
    }
  });
}

export async function startNeuralFrameRuntime({
  windowObject = globalThis.window,
  documentObject = globalThis.document,
  locationObject = globalThis.location,
  runtime = globalThis.chrome?.runtime,
  createSession = createNeuralFrameSession,
  sessionOptions = {},
  queueMicrotaskImpl = globalThis.queueMicrotask,
} = {}) {
  if (!windowObject || !documentObject || !locationObject) return null;
  if (windowObject.parent === windowObject) {
    throw new NeuralFrameError(
      "invalid-parent",
      "Neural frame must run as an embedded extension page",
    );
  }
  const capability = fragmentCapability(locationObject);
  if (!capability) {
    throw new NeuralFrameError(
      "invalid-nonce",
      "Neural frame URL fragment requires one valid private capability",
    );
  }
  const { instanceNonce, frameCapability, opaqueParent } = capability;

  const referrer = referrerIdentity(documentObject);
  let authorization;
  try {
    authorization = await sendRuntimeMessage(runtime, {
      type: "FSRCNNX_NEURAL_FRAME_CAPABILITY_CONSUME",
      capability: frameCapability,
      instanceNonce,
    });
  } catch (error) {
    throw new NeuralFrameError(
      "authorization-unavailable",
      "Neural frame authorization is unavailable",
      { cause: error, retryable: true },
    );
  }
  if (!authorization || typeof authorization !== "object" ||
      Object.keys(authorization).length !== 3 ||
      authorization.ok !== true ||
      typeof authorization.opaqueParent !== "boolean" ||
      authorization.opaqueParent !== opaqueParent ||
      (opaqueParent
        ? authorization.parentOrigin !== "null" ||
          (referrer.present &&
            (!referrer.valid || referrer.protocol !== "file:"))
        : !validWebOrigin(authorization.parentOrigin) ||
          (referrer.present &&
            (!referrer.valid || referrer.origin !== authorization.parentOrigin)))) {
    throw new NeuralFrameError(
      "authorization-denied",
      "Neural frame authorization was denied",
    );
  }
  const authorizedParentOrigin = authorization.parentOrigin;
  let connectedOrigin = null;
  let port = null;
  let closed = false;
  let closePromise = null;
  let commandTail = Promise.resolve();
  let runReserved = false;
  const seenIds = new Set();
  const seenIdOrder = [];

  function postPort(message, transfer = []) {
    if (closed || !port) return false;
    try {
      port.postMessage(message, transfer);
      return true;
    } catch {
      return false;
    }
  }

  function response(id, ok, value, transfer = []) {
    const message = {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "response",
      instanceNonce,
      id,
      ok,
    };
    if (ok) message.result = value;
    else message.error = value;
    return postPort(message, transfer);
  }

  const session = createSession({
    ...sessionOptions,
    onDeviceLost(error, stats) {
      try { sessionOptions.onDeviceLost?.(error, stats); } catch {}
      postPort({
        channel: NEURAL_FRAME_CHANNEL,
        kind: "event",
        instanceNonce,
        event: "device-lost",
        error,
        stats,
      });
    },
  });

  function rememberId(id) {
    if (seenIds.has(id)) return false;
    seenIds.add(id);
    seenIdOrder.push(id);
    if (seenIdOrder.length > MAX_SEEN_REQUEST_IDS) {
      seenIds.delete(seenIdOrder.shift());
    }
    return true;
  }

  function rejectRequest(id, error, fallback = "invalid-request") {
    response(id, false, serializeNeuralFrameError(error, fallback));
  }

  function receivePortMessage(event) {
    const data = event?.data;
    if (!isObject(data) ||
        data.channel !== NEURAL_FRAME_CHANNEL ||
        data.instanceNonce !== instanceNonce) {
      safeCloseBitmap(data?.payload?.bitmap);
      return;
    }
    if (data.kind === "cancel") {
      try {
        assertAllowedKeys(
          data,
          new Set(["channel", "kind", "instanceNonce"]),
          "cancel",
        );
      } catch {
        safeCloseBitmap(data?.payload?.bitmap);
        return;
      }
      // Cancellation is intentionally outside commandTail. It only invalidates
      // logical generations; the already-queued stop/dispose command performs
      // physical cleanup after the active run unwinds.
      try { session.cancel?.(); }
      catch { close(); }
      return;
    }
    if (data.kind !== "request") {
      safeCloseBitmap(data?.payload?.bitmap);
      return;
    }
    try {
      assertAllowedKeys(
        data,
        new Set(["channel", "kind", "instanceNonce", "id", "method", "payload"]),
        "request",
      );
    } catch (error) {
      safeCloseBitmap(data.payload?.bitmap);
      const id = requestIdIsValid(data.id) ? data.id : null;
      rejectRequest(id, error);
      return;
    }
    const id = requestIdIsValid(data.id) ? data.id : null;
    if (id === null) {
      safeCloseBitmap(data.payload?.bitmap);
      rejectRequest(
        null,
        new NeuralFrameError("invalid-request", "Request id is invalid"),
      );
      return;
    }
    if (!rememberId(id)) {
      safeCloseBitmap(data.payload?.bitmap);
      rejectRequest(
        id,
        new NeuralFrameError("duplicate-request", "Request id was already used"),
      );
      return;
    }
    if (typeof data.method !== "string" || !data.method ||
        data.method.length > 64) {
      safeCloseBitmap(data.payload?.bitmap);
      rejectRequest(
        id,
        new NeuralFrameError("invalid-request", "Request method is invalid"),
      );
      return;
    }
    if (data.method === "run" && runReserved) {
      safeCloseBitmap(data.payload?.bitmap);
      rejectRequest(
        id,
        new NeuralFrameError(
          "run-busy",
          "A neural frame run is already pending",
          { retryable: true },
        ),
        "run-busy",
      );
      return;
    }
    if (data.method === "run") runReserved = true;

    const operation = commandTail.catch(() => {}).then(() =>
      session.handle(data.method, data.payload));
    commandTail = operation.then(() => undefined, () => undefined);
    operation.then(
      (result) => {
        if (data.method === "run") {
          const bitmap = result?.bitmap;
          const output = result?.presentation?.output;
          if (!bitmap || typeof bitmap !== "object" ||
              typeof bitmap.close !== "function" ||
              !Number.isSafeInteger(bitmap.width) || bitmap.width <= 0 ||
              !Number.isSafeInteger(bitmap.height) || bitmap.height <= 0 ||
              bitmap.width !== output?.width || bitmap.height !== output?.height) {
            safeCloseBitmap(bitmap);
            rejectRequest(
              id,
              new NeuralFrameError(
                "presentation-failed",
                "Neural frame produced no valid transferable output bitmap",
              ),
              "presentation-failed",
            );
            return;
          }
          if (!response(id, true, result, [bitmap])) {
            safeCloseBitmap(bitmap);
            close();
            return;
          }
        } else if (!response(id, true, result)) {
          close();
          return;
        }
        if (data.method === "dispose") {
          queueMicrotaskImpl(() => close({ disposeSession: false }));
        }
      },
      (error) => {
        rejectRequest(id, error, `${data.method.toLowerCase()}-failed`);
      },
    ).finally(() => {
      if (data.method === "run") runReserved = false;
    });
  }

  function receivePortMessageError(event) {
    safeCloseBitmap(event?.data?.payload?.bitmap);
    close();
  }

  function receiveWindowMessage(event) {
    if (closed || port || event?.source !== windowObject.parent) return;
    const data = event.data;
    const parentOriginAccepted = opaqueParent
      ? event.origin === "null" &&
        (!referrer.present ||
          (referrer.valid && referrer.protocol === "file:"))
      : event.origin === authorizedParentOrigin &&
        validWebOrigin(event.origin) &&
        (!referrer.present ||
          (referrer.valid && referrer.origin === event.origin));
    if (!isObject(data) ||
        data.channel !== NEURAL_FRAME_CHANNEL ||
        data.kind !== "connect" ||
        data.instanceNonce !== instanceNonce ||
        !parentOriginAccepted ||
        event.ports?.length !== 1) {
      return;
    }
    const candidatePort = event.ports[0];
    if (!candidatePort ||
        typeof candidatePort.postMessage !== "function" ||
        typeof candidatePort.addEventListener !== "function") {
      return;
    }
    connectedOrigin = event.origin;
    port = candidatePort;
    windowObject.removeEventListener("message", receiveWindowMessage);
    port.addEventListener("message", receivePortMessage);
    port.addEventListener("messageerror", receivePortMessageError);
    port.start?.();
    if (!postPort({
      channel: NEURAL_FRAME_CHANNEL,
      kind: "connected",
      instanceNonce,
    })) {
      close();
    }
  }

  function close({ disposeSession = true } = {}) {
    if (closePromise) return closePromise;
    closed = true;
    try { session.cancel?.(); } catch {}
    windowObject.removeEventListener("message", receiveWindowMessage);
    windowObject.removeEventListener?.("pagehide", receivePageHide);
    const ownedPort = port;
    port = null;
    try { ownedPort?.removeEventListener?.("message", receivePortMessage); } catch {}
    try { ownedPort?.removeEventListener?.("messageerror", receivePortMessageError); } catch {}
    try { ownedPort?.close?.(); } catch {}
    closePromise = disposeSession
      ? commandTail.catch(() => {}).then(() => session.dispose?.()).catch(() => {})
      : Promise.resolve();
    return closePromise;
  }

  function receivePageHide() {
    close();
  }

  windowObject.addEventListener("message", receiveWindowMessage);
  windowObject.addEventListener?.("pagehide", receivePageHide, { once: true });
  const readyTarget = opaqueParent ? "*" : authorizedParentOrigin;
  try {
    windowObject.parent.postMessage({
      channel: NEURAL_FRAME_CHANNEL,
      kind: "ready",
    }, readyTarget);
  } catch (error) {
    await close();
    throw new NeuralFrameError(
      "ready-failed",
      "Neural frame could not announce readiness",
      { cause: error, retryable: true },
    );
  }

  return Object.freeze({
    instanceNonce,
    close,
    connected: () => !!port && !closed,
    connectedOrigin: () => connectedOrigin,
  });
}

if (typeof window !== "undefined" &&
    typeof document !== "undefined" &&
    typeof location !== "undefined" &&
    !globalThis.__FSRCNNX_DISABLE_NEURAL_FRAME_AUTOSTART__) {
  void startNeuralFrameRuntime().catch(() => {});
}

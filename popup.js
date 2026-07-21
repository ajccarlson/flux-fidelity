const STATUS_COMMAND = "FSRCNNX_STATUS";
const STATUS_TIMEOUT_MS = 3_000;
const COMMAND_TIMEOUT_MS = 30_000;

export const POLICY_OPTIONS = Object.freeze({
  fsrcnnx: Object.freeze([
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv thresholds)"],
    ["force2", "Always ×2"],
    ["force3", "Always ×3"],
    ["force4", "Always ×4"],
  ]),
  artcnn: Object.freeze([
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv thresholds)"],
    ["force2", "Always ×2"],
    ["force4", "Always ×4"],
    ["force8", "Always ×8"],
  ]),
});

const STATIC_INTERPOLATION_MODELS = Object.freeze([
  ["rife_v4.26", "RIFE 4.26 (default; may wave on bright motion)"],
  ["rife_v4.26_fp16", "RIFE 4.26 FP16 (experimental)"],
  ["rife_orig", "RIFE original"],
  ["blend", "Blend (no AI)"],
]);

function objectRecord(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function errorText(error) {
  if (error && typeof error.message === "string" && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "Unknown extension error";
}

export function isSupportedPageUrl(value) {
  try {
    const protocol = new URL(value).protocol;
    return protocol === "http:" || protocol === "https:" || protocol === "file:";
  } catch {
    return false;
  }
}

function timeout(promise, milliseconds, setTimer, clearTimer) {
  let timer = null;
  const expired = new Promise((_, reject) => {
    timer = setTimer(() => reject(new Error(`No response after ${milliseconds} ms`)), milliseconds);
  });
  return Promise.race([promise, expired]).finally(() => clearTimer(timer));
}

export class PopupTransport {
  constructor(chromeApi, {
    setTimer = globalThis.setTimeout?.bind(globalThis),
    clearTimer = globalThis.clearTimeout?.bind(globalThis),
  } = {}) {
    this.chrome = chromeApi;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
  }

  async send(type, payload = {}) {
    let activeTab;
    try {
      const tabs = await this.chrome.tabs.query({ active: true, currentWindow: true });
      activeTab = Array.isArray(tabs) ? tabs[0] : null;
    } catch (error) {
      return { ok: false, error: "tab-query-failed", reason: errorText(error) };
    }

    if (!activeTab || !isSupportedPageUrl(activeTab.url || "")) {
      return {
        ok: false,
        error: "unsupported-page",
        reason: "The extension cannot run on this browser page.",
      };
    }

    try {
      const waitMs = type === STATUS_COMMAND ? STATUS_TIMEOUT_MS : COMMAND_TIMEOUT_MS;
      const response = await timeout(
        Promise.resolve(this.chrome.tabs.sendMessage(activeTab.id, { type, ...payload })),
        waitMs,
        this.setTimer,
        this.clearTimer,
      );
      if (!objectRecord(response)) {
        return { ok: false, error: "invalid-response", reason: "The page returned an invalid response." };
      }
      if (type !== STATUS_COMMAND && typeof response.ok !== "boolean") {
        return { ok: false, error: "invalid-response", reason: "The command response was incomplete." };
      }
      return response;
    } catch (error) {
      const reason = errorText(error);
      return {
        ok: false,
        error: /No response after/.test(reason) ? "response-timeout" : "no-content-script",
        reason,
      };
    }
  }
}

// Coalesces interval and post-command refreshes into one request at a time. A
// mutation invalidates an older response even if it was already in flight.
export class StatusCoordinator {
  constructor(load, apply) {
    this.load = load;
    this.apply = apply;
    this.requested = 0;
    this.mutation = 0;
    this.drain = null;
  }

  invalidate() {
    this.mutation++;
  }

  refresh() {
    this.requested++;
    if (!this.drain) {
      this.drain = this.run().finally(() => { this.drain = null; });
    }
    return this.drain;
  }

  async run() {
    let completed = 0;
    while (completed < this.requested) {
      const request = this.requested;
      const mutation = this.mutation;
      let status;
      try {
        status = await this.load();
      } catch (error) {
        status = { ok: false, error: "status-failed", reason: errorText(error) };
      }
      completed = request;
      if (request === this.requested && mutation === this.mutation) this.apply(status);
    }
  }
}

function optionSignature(items) {
  return items.map(({ value, label, disabled }) => `${value}\u0000${label}\u0000${disabled ? 1 : 0}`).join("\u0001");
}

export function reconcileSelectOptions(documentRef, select, rawItems, selected = null) {
  const items = rawItems.map((item) => Array.isArray(item)
    ? { value: String(item[0]), label: String(item[1]), disabled: false }
    : {
        value: String(item.value ?? ""),
        label: String(item.label ?? item.value ?? ""),
        disabled: item.disabled === true,
      });
  const signature = optionSignature(items);
  if (select.dataset.optionSignature !== signature) {
    const options = items.map((item) => {
      const option = documentRef.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      option.disabled = item.disabled;
      return option;
    });
    select.replaceChildren(...options);
    select.dataset.optionSignature = signature;
  }
  if (selected != null && items.some((item) => item.value === String(selected))) {
    select.value = String(selected);
  }
  return signature;
}

export function describeCommandFailure(result) {
  const code = result?.error || result?.reason || "command-failed";
  const messages = {
    "unsupported-page": "The extension cannot run on this browser page.",
    "tab-query-failed": "The active tab could not be inspected.",
    "no-content-script": "The page is not connected to the extension. Reload the page and try again.",
    "response-timeout": "The page did not answer the command in time.",
    "invalid-response": "The page returned an invalid extension response.",
    "invalid-input": "That setting is not valid.",
    "invalid mode": "That mode is not valid.",
    "invalid engine": "That engine is not valid.",
    "invalid policy": "That upscale policy is not valid for this engine.",
    "no video": "No playable video is available yet. The requested setting will be retried when one appears.",
    drm: "This video is DRM-protected, so its frames cannot be processed.",
    tainted: "This cross-origin video does not permit the extension to read its pixels.",
    protected: "This protected video cannot be processed.",
    "renderer unavailable": "The WebGPU renderer is unavailable on this page.",
    "lifecycle-pending": "The page changed while the renderer was starting. The request remains pending.",
    superseded: "A newer page or setting change replaced this request.",
    "startup-failed": "The extension could not start on this page.",
    "status-failed": "The page status could not be read.",
    "command-failed": "The page could not apply that setting.",
  };
  if (messages[code]) return messages[code];
  if (typeof result?.message === "string" && result.message) return result.message;
  if (typeof result?.reason === "string" && result.reason) return result.reason;
  return messages["command-failed"];
}

function setBooleanStatus(element, value, yes = "yes", no = "no") {
  element.textContent = value ? yes : no;
  element.className = `v ${value ? "ok" : "no"}`;
}

function failureDetail(failure) {
  if (typeof failure === "string") return failure.trim();
  if (!objectRecord(failure)) return "";
  for (const value of [failure.detail, failure.message, failure.reason, failure.error?.message]) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function joinFailureMessage(summary, failure, prefixes = []) {
  const base = String(summary || "").trim().replace(/[.:]+$/, "");
  let detail = failureDetail(failure);
  for (const prefix of prefixes) detail = detail.replace(prefix, "").trim();
  detail = detail.replace(/^[:\s-]+/, "").trim();
  if (!detail) return `${base}.`;
  const normalizedBase = base.toLocaleLowerCase();
  const normalizedDetail = detail.replace(/[.:]+$/, "").toLocaleLowerCase();
  if (normalizedDetail === normalizedBase || normalizedBase.includes(normalizedDetail)) {
    return `${base}.`;
  }
  return `${base}: ${detail}`;
}

function isPreferenceFailure(status) {
  const code = String(status?.error || "").toLocaleLowerCase();
  return /preference|settings|storage|application/.test(code) ||
    objectRecord(status?.persistence) &&
      (status.persistence.state === "error" || !!status.persistence.error);
}

function preferenceFailureMessage(status) {
  const code = String(status?.error || "").toLocaleLowerCase();
  if (/validation|invalid|corrupt|schema/.test(code)) {
    return joinFailureMessage("Stored settings are invalid", status?.reason, [
      /^invalid stored settings?\s*/i,
      /^settings validation failed\s*/i,
    ]);
  }
  if (/application|apply/.test(code)) {
    return joinFailureMessage("Settings could not be applied", status?.reason, [
      /^preference application failed\s*/i,
      /^settings could not be applied\s*/i,
    ]);
  }
  if (/sync/.test(code)) {
    return joinFailureMessage("Settings could not be synchronized", status?.reason, [
      /^preference (?:sync|synchronization) failed\s*/i,
      /^settings could not be synchronized\s*/i,
    ]);
  }
  if (/write|save/.test(code)) {
    return joinFailureMessage("Settings could not be saved", status?.reason, [
      /^settings could not be saved\s*/i,
    ]);
  }
  if (/read|load|storage/.test(code)) {
    return joinFailureMessage("Settings could not be loaded", status?.reason, [
      /^settings could not be (?:read|loaded)\s*/i,
    ]);
  }
  return joinFailureMessage("Settings could not be accessed", status?.reason);
}

function gpuDisplayState(status) {
  const direct = {
    unavailable: "unavailable",
    idle: "available",
    available: "available",
    initializing: "starting",
    starting: "starting",
    ready: "ready",
    recovering: "recovering",
    failed: "failed",
  }[status.gpuState];
  if (direct) return direct;

  const runtime = objectRecord(status.runtime) ? status.runtime : null;
  if (runtime) {
    if (runtime.api === "unavailable") return "unavailable";
    const recovery = runtime.recovery?.phase;
    if (recovery === "scheduled" || recovery === "running" || recovery === "recovering") {
      return "recovering";
    }
    if (recovery === "exhausted") return "failed";
    if (runtime.device === "ready") return "ready";
    if (runtime.adapter === "requesting" || runtime.device === "requesting" ||
        runtime.device === "initializing") return "starting";
    if (runtime.adapter === "failed" || runtime.device === "failed" || runtime.device === "lost") {
      return "failed";
    }
    if (runtime.api === "available") return "available";
  }
  return status.webgpu === true ? "available" : "unavailable";
}

function setGpuStatus(element, state) {
  const labels = {
    unavailable: "unavailable",
    available: "available",
    starting: "starting…",
    ready: "ready",
    recovering: "recovering…",
    failed: "failed",
  };
  element.textContent = labels[state] || labels.unavailable;
  element.className = `v ${state === "available" || state === "ready"
    ? "ok"
    : state === "unavailable" || state === "failed" ? "no" : ""}`.trimEnd();
}

function setVisible(element, visible, display = "block") {
  const hidden = !visible;
  const nextDisplay = visible ? display : "none";
  if (element.hidden !== hidden) element.hidden = hidden;
  if (element.style.display !== nextDisplay) element.style.display = nextDisplay;
}

function setText(element, value) {
  const next = String(value ?? "");
  if (element.textContent !== next) element.textContent = next;
}

function rendererFallback(status) {
  const renderer = objectRecord(status.renderer) ? status.renderer : null;
  if (objectRecord(renderer?.fallback)) return renderer.fallback;
  const requested = renderer?.requestedEngine || status.engine;
  const effective = renderer?.effectiveEngine || status.activeEngine;
  if (requested === "neural" && effective && effective !== "neural") {
    return { from: "neural", to: effective };
  }
  return null;
}

function effectiveEngine(status) {
  const renderer = objectRecord(status.renderer) ? status.renderer : null;
  return renderer?.effectiveEngine || status.activeEngine || status.engine;
}

function engineLabel(engine) {
  return ({
    fsrcnnx: "FSRCNNX standard",
    artcnn: "ArtCNN",
    neural: "Neural",
  })[engine] || "Renderer";
}

function formatEngineModel(status, engine) {
  if (engine === "neural") {
    const neural = status.neural;
    const key = neural?.label || neural?.model || status.neuralModel;
    if (!key) return "—";
    return `${neural?.scale || "?"}× ${key}`;
  }
  if (!status.model) return "—";
  if (engine === "artcnn") return String(status.model).replace("ArtCNN_", "ArtCNN ");
  const scale = status.scale ? `${status.scale}× ` : "";
  return scale + String(status.model)
    .replace("FSRCNNX_", "")
    .replace("_16-0-4-1", "")
    .replace("_56-16-4-1", " high");
}

function formatModel(status) {
  const activeEngine = effectiveEngine(status);
  const fallback = rendererFallback(status);
  if (fallback) {
    const fallbackEngine = fallback.to || activeEngine || "fsrcnnx";
    const model = formatEngineModel(status, fallbackEngine);
    const label = `${engineLabel(fallbackEngine)} fallback`;
    return model === "—" ? label : `${model} (${label})`;
  }
  return formatEngineModel(status, activeEngine);
}

function formatInterpolationStats(status) {
  const stats = status.interpStats;
  if (!status.interpolate) return "";
  if (status.interpPausedByNeural) return "Paused while the neural upscaler is selected.";
  if (status.interpQuarantined) return "Stopped after a repeated interpolation failure.";
  if (!stats) return "Configured; waiting for the interpolation runtime.";
  const parts = [`input ${stats.fpsIn ?? 0} → output ${stats.fpsOut ?? 0} fps`];
  if (stats.interpMode === "blend") parts.push(`blend → ${stats.effectiveTargetFps ?? "?"} fps`);
  else if (stats.rife) parts.push(`RIFE ${stats.inferMs ?? 0} ms @ ${Math.round((stats.scale || 1) * 100)}%`);
  if (Number.isFinite(stats.maxGapMs)) parts.push(`max gap ${stats.maxGapMs} ms`);
  return parts.join(" · ");
}

function commandSucceeded(result) {
  return objectRecord(result) && result.ok === true;
}

export function createPopupController({
  document: documentRef,
  chrome: chromeApi,
  setInterval: setIntervalRef = globalThis.setInterval?.bind(globalThis),
  clearInterval: clearIntervalRef = globalThis.clearInterval?.bind(globalThis),
  transport = new PopupTransport(chromeApi),
} = {}) {
  if (!documentRef) throw new TypeError("A document is required");
  if (!chromeApi && !transport) throw new TypeError("The Chrome extension API is required");

  const $ = (id) => documentRef.getElementById(id);
  const modeButtons = [...documentRef.querySelectorAll(".modes button")];
  const controls = [...documentRef.querySelectorAll("button, input, select")];
  let ready = false;
  let commandBusy = 0;
  let currentStatus = null;
  let operationSerial = 0;
  let commandTail = Promise.resolve();
  let refreshTimer = null;
  let feedbackScope = "context";
  let forceControlSync = false;

  function feedback(message = "", tone = "", scope = "operation") {
    const element = $("operation-status");
    setText(element, message);
    if (element.dataset.tone !== tone) element.dataset.tone = tone;
    feedbackScope = scope;
  }

  function contextFeedback(message = "", tone = "") {
    if (feedbackScope === "operation") return;
    feedback(message, tone, "context");
  }

  function clearPageState() {
    $("s-video").textContent = "—";
    $("s-video").className = "v";
    $("s-model").textContent = "—";
    $("s-model").className = "v";
    $("s-frames").textContent = "0";
    setText($("runtime-status"), "");
    setVisible($("drm-banner"), false);
  }

  function updateAvailability() {
    const globallyDisabled = !ready || commandBusy > 0;
    for (const control of controls) control.disabled = globallyDisabled;
    if (globallyDisabled || !currentStatus) return;

    const neural = currentStatus.engine === "neural";
    $("policy").disabled = neural;
    $("all-videos").disabled = neural;
    $("artvariant").disabled = currentStatus.engine !== "artcnn";
    $("neural-model").disabled = !neural || !Array.isArray(currentStatus.neuralModels) ||
      currentStatus.neuralModels.length === 0;

    const interpolationConfigured = currentStatus.interpolate === true;
    for (const id of [
      "interp-model", "interp-res", "interp-target", "interp-avoff", "interp-diag",
      "interp-ladder", "interp-invert", "interp-autofallback",
    ]) $(id).disabled = !interpolationConfigured;
  }

  function unavailable(status) {
    ready = false;
    currentStatus = null;
    clearPageState();
    $("s-webgpu").className = "v no";
    const code = status?.error;
    if (code === "unsupported-page") {
      $("s-webgpu").textContent = "unavailable";
      feedback("Open an http, https, or permitted local file page to use the extension.", "notice", "context");
    } else if (code === "no-content-script") {
      $("s-webgpu").textContent = "disconnected";
      feedback("Reload this page so it can connect to the extension.", "error", "context");
    } else {
      $("s-webgpu").textContent = "error";
      feedback(describeCommandFailure(status), "error", "context");
    }
    updateAvailability();
  }

  function loading() {
    ready = false;
    currentStatus = null;
    clearPageState();
    $("s-webgpu").textContent = "checking…";
    $("s-webgpu").className = "v";
    feedback("The extension is starting on this page…", "notice", "context");
    updateAvailability();
  }

  function failed(status) {
    ready = false;
    currentStatus = null;
    clearPageState();
    const preferenceFailure = isPreferenceFailure(status);
    if (preferenceFailure) {
      setGpuStatus($("s-webgpu"), gpuDisplayState(status));
    } else {
      $("s-webgpu").textContent = "failed";
      $("s-webgpu").className = "v no";
    }
    const message = preferenceFailure
      ? preferenceFailureMessage(status)
      : joinFailureMessage(describeCommandFailure(status), status.reason);
    feedback(message, "error", "context");
    updateAvailability();
  }

  function persistenceHealth(status) {
    const candidates = [
      status.persistence,
      status.settingsPersistence,
      status.settings?.persistence,
      status.settings,
    ];
    return candidates.find((candidate) => objectRecord(candidate) && (
      typeof candidate.state === "string" ||
      typeof candidate.phase === "string" ||
      typeof candidate.operation === "string" ||
      typeof candidate.errorOperation === "string" ||
      candidate.error || candidate.lastFailure ||
      Number.isFinite(candidate.pendingWrites) || Number.isFinite(candidate.pending)
    )) || null;
  }

  function persistenceMessage(status, persistence) {
    if (!persistence) return "";
    const detail = failureDetail(persistence.error || persistence.lastFailure);
    const pending = Number(persistence.pendingWrites ?? persistence.pending ?? 0);
    const signal = [
      persistence.operation,
      persistence.errorOperation,
      persistence.phase,
      persistence.state,
      persistence.error?.code,
      persistence.lastFailure?.code,
      detail,
    ].filter(Boolean).join(" ").toLocaleLowerCase();
    const failed = persistence.state === "error" || !!persistence.error ||
      /(?:^|[-\s])(?:error|failed|failure)(?:$|[-\s])/.test(signal);

    // A BFCache synchronization can make the store report a generic active
    // write state while it is actually reading and applying external changes.
    if (!failed && status.runtime?.phase === "syncing") return "Synchronizing settings…";
    if (!failed) {
      if (/sync/.test(signal)) return "Synchronizing settings…";
      if (/(?:^|[-\s])(?:read|reading|load|loading)(?:$|[-\s])/.test(signal)) {
        return "Loading settings…";
      }
      if (/validat/.test(signal)) return "Validating settings…";
      if (/application|apply/.test(signal)) return "Applying settings…";
      if (/writ|sav/.test(signal)) return "Saving settings…";
      return "";
    }

    if (/validat|invalid|corrupt|schema/.test(signal)) {
      return joinFailureMessage("Stored settings are invalid", detail, [
        /^invalid stored settings?\s*/i,
        /^settings validation failed\s*/i,
      ]);
    }
    if (/application|apply/.test(signal)) {
      return joinFailureMessage("Settings could not be applied", detail, [
        /^preference application failed\s*/i,
        /^settings could not be applied\s*/i,
      ]);
    }
    if (/sync/.test(signal)) {
      return joinFailureMessage("Settings could not be synchronized", detail, [
        /^preference (?:sync|synchronization) failed\s*/i,
        /^settings could not be synchronized\s*/i,
      ]);
    }
    if (/read|load|get/.test(signal)) {
      return joinFailureMessage("Settings could not be loaded", detail, [
        /^settings could not be (?:read|loaded)\s*/i,
      ]);
    }
    if (pending > 0 || /writ|sav|quota|storage set/.test(signal)) {
      return joinFailureMessage("Settings could not be saved", detail, [
        /^settings could not be saved\s*/i,
      ]);
    }
    return joinFailureMessage("Settings could not be loaded or applied", detail);
  }

  function requestedFeaturesNeedGpu(status) {
    const renderer = objectRecord(status.renderer) ? status.renderer : null;
    const images = objectRecord(status.imagesRuntime) ? status.imagesRuntime : null;
    const interpolation = objectRecord(status.interpolationRuntime)
      ? status.interpolationRuntime
      : null;
    const requestedMode = renderer?.requestedMode || status.mode || "off";
    return requestedMode !== "off" || images?.requested === true || status.images === true ||
      interpolation?.requested === true || status.interpolate === true;
  }

  function runtimeMessage(status) {
    const persistence = persistenceHealth(status);
    const settingsMessage = persistenceMessage(status, persistence);
    if (settingsMessage) return settingsMessage;

    const gpuState = gpuDisplayState(status);
    if (requestedFeaturesNeedGpu(status)) {
      if (gpuState === "recovering" || status.gpuRecovering) {
        return "Recovering the WebGPU device…";
      }
      if (gpuState === "unavailable") {
        return "WebGPU is unavailable, so the requested features cannot start.";
      }
      if (gpuState === "failed") {
        const detail = failureDetail(status.runtime?.lastFailure);
        return detail
          ? `WebGPU failed: ${detail}`
          : "WebGPU failed, so the requested features cannot start.";
      }
    }

    const renderer = objectRecord(status.renderer) ? status.renderer : null;
    const requestedMode = renderer?.requestedMode || status.mode || "off";
    const activeMode = renderer?.activeMode || status.activeMode || "off";
    if (renderer?.phase === "blocked") {
      return "The requested renderer is blocked by this video source.";
    }
    if (renderer?.phase === "waiting-video") {
      return "The requested mode will activate when a playable video appears.";
    }
    if (renderer?.phase === "starting") return "The requested renderer is starting…";
    if (renderer?.phase === "failed") return "The requested renderer could not start.";

    const fallback = rendererFallback(status);
    if (fallback) {
      const from = fallback.from === "neural" ? "Neural upscaling" : engineLabel(fallback.from);
      const to = engineLabel(fallback.to || effectiveEngine(status));
      const detail = failureDetail(fallback);
      return `${from} fell back to ${to}${detail ? `: ${detail}` : "."}`;
    }

    if (status.documentSuspended || renderer?.phase === "suspended") {
      return "Rendering is suspended while this page is hidden.";
    }

    const neural = objectRecord(status.neuralRuntime) ? status.neuralRuntime : null;
    if (neural?.requested === true && neural.phase === "failed") {
      return joinFailureMessage("Neural upscaling failed", neural.lastFailure);
    }

    const images = objectRecord(status.imagesRuntime) ? status.imagesRuntime : null;
    if (images?.requested === true && images.phase === "failed") {
      return joinFailureMessage("Image upscaling failed", images.lastFailure);
    }

    const interpolation = objectRecord(status.interpolationRuntime)
      ? status.interpolationRuntime
      : null;
    const interpFailure = interpolation?.lastFailure || status.interpFailure;
    if (interpFailure) {
      const detail = failureDetail(interpFailure);
      const suffix = detail ? `: ${detail}` : ".";
      return `Interpolation stopped after repeated failures${suffix}`;
    }

    // Terminal optional-feature failures are reported above as a group so a
    // lower-severity startup or wait state cannot hide them.
    if (images?.requested === true) {
      if (images.phase === "recovering") return "Image upscaling is recovering…";
      if (images.phase === "starting") return "Image upscaling is starting…";
      if (images.phase !== "active") return "Image upscaling is waiting to start.";
    }

    if (status.interpPausedByNeural ||
        (interpolation?.phase === "paused" && interpolation.pauseReason === "neural")) {
      return "Frame interpolation is paused while the neural upscaler is selected.";
    }
    if (requestedMode !== "off" && activeMode === "off") {
      return status.hasVideo
        ? "The requested mode is waiting for the renderer."
        : "The requested mode will activate when a playable video appears.";
    }
    return "";
  }

  function render(status) {
    if (!objectRecord(status)) {
      unavailable({ error: "invalid-response" });
      return;
    }
    if (status.error && !status.failed) {
      unavailable(status);
      return;
    }
    if (status.loading) {
      loading();
      return;
    }
    if (status.failed) {
      failed(status);
      return;
    }

    ready = true;
    currentStatus = status;
    const forceSync = forceControlSync;
    forceControlSync = false;
    if (feedbackScope === "context") contextFeedback("", "");
    setGpuStatus($("s-webgpu"), gpuDisplayState(status));
    setBooleanStatus($("s-video"), status.hasVideo === true);
    const modelText = formatModel(status);
    $("s-model").textContent = modelText;
    $("s-model").className = `v ${modelText === "—" ? "" : "ok"}`.trimEnd();
    $("s-frames").textContent = String(status.frameCount ?? 0);
    setText($("runtime-status"), runtimeMessage(status));

    const protectedMessage = status.protected
      ? (status.protectedReason === "tainted"
          ? "This cross-origin video does not permit the extension to read its pixels."
          : "This DRM-protected video does not permit the extension to read its frames.")
      : "";
    setText($("drm-banner"), protectedMessage);
    setVisible($("drm-banner"), !!protectedMessage);

    const engine = ["fsrcnnx", "artcnn", "neural"].includes(status.engine)
      ? status.engine
      : "fsrcnnx";
    if (forceSync || documentRef.activeElement !== $("engine")) $("engine").value = engine;
    setVisible($("artvariant"), engine === "artcnn");
    setVisible($("neuralrow"), engine === "neural");
    if (status.artVariant && (forceSync || documentRef.activeElement !== $("artvariant"))) {
      $("artvariant").value = status.artVariant;
    }

    const policyItems = POLICY_OPTIONS[engine] || POLICY_OPTIONS.fsrcnnx;
    reconcileSelectOptions(
      documentRef,
      $("policy"),
      policyItems,
      forceSync || documentRef.activeElement !== $("policy") ? status.policy || "display" : $("policy").value,
    );

    const neuralItems = Array.isArray(status.neuralModels) && status.neuralModels.length
      ? status.neuralModels.map((model) => ({
          value: model.key,
          label: `${model.label || model.key} (${model.scale}×)`,
        }))
      : [{ value: "", label: "No bundled neural models", disabled: true }];
    const selectedNeural = status.neural?.model || status.neuralModel || "";
    reconcileSelectOptions(
      documentRef,
      $("neural-model"),
      neuralItems,
      forceSync || documentRef.activeElement !== $("neural-model") ? selectedNeural : $("neural-model").value,
    );
    if (engine === "neural") {
      const fallback = rendererFallback(status);
      if (fallback) {
        const detail = failureDetail(fallback);
        $("neural-note").textContent = `Using ${engineLabel(fallback.to || effectiveEngine(status))} fallback` +
          (detail ? ` · ${detail}` : ".");
      } else {
        $("neural-note").textContent = status.neural?.ready
          ? `mean ${(status.neural.mu || 0).toFixed(1)} ms · skipped ${status.neural.skip || 0}`
          : (selectedNeural ? "Selected; initializes when video upscaling is active." : "No model selected.");
      }
    } else {
      $("neural-note").textContent = "";
    }

    for (const button of modeButtons) {
      const selected = button.dataset.mode === status.mode;
      button.dataset.active = selected ? "1" : "0";
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    }

    const syncCheckbox = (id, value) => {
      if (typeof value === "boolean") $(id).checked = value;
    };
    syncCheckbox("ssimds", status.ssimds);
    syncCheckbox("sharpen", status.sharpen);
    syncCheckbox("deband", status.deband);
    syncCheckbox("hover-reveal", status.hoverReveal);
    syncCheckbox("all-videos", status.allVideos);
    syncCheckbox("images", status.images);
    syncCheckbox("interpolate", status.interpolate);
    syncCheckbox("interp-diag", status.interpStaticPassthrough);
    syncCheckbox("interp-ladder", status.interpLadder);
    syncCheckbox("interp-invert", status.interpInvert);
    syncCheckbox("interp-autofallback", status.interpAutoFallback);

    setVisible($("sharpen-row"), status.sharpen === true);
    setVisible($("deband-row"), status.deband === true);
    setVisible($("interp-res-row"), status.interpolate === true);
    $("multi-count").textContent = status.allVideos && status.multiCount
      ? `(${status.multiCount} active)` : "";
    $("image-count").textContent = status.images && status.imageCount
      ? `(${status.imageCount} processed)` : "";

    const syncRange = (id, outputId, value, digits = null) => {
      if (!Number.isFinite(value) || (!forceSync && documentRef.activeElement === $(id))) return;
      $(id).value = String(value);
      $(outputId).textContent = digits == null ? String(value) : value.toFixed(digits);
    };
    syncRange("sharpen-str", "sharpen-val", status.sharpenStrength, 1);
    syncRange("deband-str", "deband-val", status.debandStrength, 1);

    const interpModels = Array.isArray(status.interpStats?.models) && status.interpStats.models.length
      ? status.interpStats.models.map((model) => [model.key, model.label])
      : STATIC_INTERPOLATION_MODELS;
    const selectedInterp = status.interpModel ||
      status.interpStats?.models?.find((model) => model.current)?.key || "rife_v4.26";
    if (forceSync || documentRef.activeElement !== $("interp-model")) {
      reconcileSelectOptions(documentRef, $("interp-model"), interpModels, selectedInterp);
    }

    const resMode = status.interpResMode || status.interpStats?.resMode;
    if (resMode && (forceSync || documentRef.activeElement !== $("interp-res"))) $("interp-res").value = resMode;
    const target = status.interpTargetFps ?? status.interpStats?.targetFps;
    if (target != null && (forceSync || documentRef.activeElement !== $("interp-target"))) {
      $("interp-target").value = String(target);
    }
    const detectedHz = status.interpStats?.detectedHz;
    $("interp-target-hz").textContent = target === "auto"
      ? (detectedHz ? `detected ${detectedHz} Hz` : "detecting display refresh…")
      : (status.interpStats?.effectiveTargetFps ? `effective ${status.interpStats.effectiveTargetFps} fps` : "");
    const avOffset = status.interpAvOffsetMs ?? status.interpStats?.avOffsetMs;
    syncRange("interp-avoff", "interp-avoff-val", avOffset);
    $("interp-stats").textContent = formatInterpolationStats(status);

    updateAvailability();
  }

  const coordinator = new StatusCoordinator(
    () => transport.send(STATUS_COMMAND),
    render,
  );

  function runCommand(label, invoke) {
    const operation = ++operationSerial;
    coordinator.invalidate();
    commandBusy++;
    feedback(`${label}…`, "pending", "operation");
    updateAvailability();

    const task = commandTail.catch(() => {}).then(async () => {
      let result;
      try {
        result = await invoke();
      } catch (error) {
        result = { ok: false, error: "command-failed", reason: errorText(error) };
      }
      const current = operation === operationSerial;
      if (current) {
        if (commandSucceeded(result)) {
          const pending = result.pending === true || result.running === false || result.paused;
          feedback(pending ? "Setting saved; activation is pending." : "Setting applied.", "success", "operation");
        } else {
          feedback(describeCommandFailure(result), result?.pending ? "notice" : "error", "operation");
        }
        coordinator.invalidate();
        forceControlSync = true;
        await coordinator.refresh();
      }
      return result;
    }).finally(() => {
      commandBusy = Math.max(0, commandBusy - 1);
      updateAvailability();
    });
    commandTail = task;
    return task;
  }

  function command(id, event, label, type, payload) {
    $(id).addEventListener(event, () => runCommand(label, () => transport.send(type, payload())));
  }

  function bind() {
    command("engine", "change", "Changing upscaler engine", "FSRCNNX_SETENGINE", () => ({ engine: $("engine").value }));
    command("artvariant", "change", "Changing ArtCNN variant", "FSRCNNX_SETARTVARIANT", () => ({ variant: $("artvariant").value }));
    command("policy", "change", "Changing upscale policy", "FSRCNNX_SETPOLICY", () => ({ policy: $("policy").value }));
    command("ssimds", "change", "Updating downscaling", "FSRCNNX_SETSSIMDS", () => ({ on: $("ssimds").checked }));
    command("sharpen", "change", "Updating sharpening", "FSRCNNX_SETSHARPEN", () => ({ on: $("sharpen").checked }));
    command("sharpen-str", "change", "Changing sharpen strength", "FSRCNNX_SETSHARPENSTR", () => ({ strength: Number($("sharpen-str").value) }));
    command("deband", "change", "Updating debanding", "FSRCNNX_SETDEBAND", () => ({ on: $("deband").checked }));
    command("deband-str", "change", "Changing deband strength", "FSRCNNX_SETDEBANDSTR", () => ({ strength: Number($("deband-str").value) }));
    command("interpolate", "change", "Updating frame interpolation", "FSRCNNX_SETINTERPOLATE", () => ({ on: $("interpolate").checked }));
    command("interp-res", "change", "Changing inference resolution", "FSRCNNX_SETINTERPRES", () => ({ mode: $("interp-res").value }));
    command("interp-avoff", "change", "Changing audio sync trim", "FSRCNNX_SETINTERPAVOFFSET", () => ({ ms: Number($("interp-avoff").value) }));
    command("interp-diag", "change", "Updating static-detail stabilization", "FSRCNNX_SETINTERPDIAG", () => ({ on: $("interp-diag").checked }));
    command("interp-ladder", "change", "Updating blend ladder", "FSRCNNX_SETLADDER", () => ({ on: $("interp-ladder").checked }));
    command("interp-autofallback", "change", "Updating automatic fallback", "FSRCNNX_SETAUTOFALLBACK", () => ({ on: $("interp-autofallback").checked }));
    command("interp-invert", "change", "Changing interpolation order", "FSRCNNX_SETINVERT", () => ({ on: $("interp-invert").checked }));
    command("interp-model", "change", "Changing interpolation model", "FSRCNNX_SETINTERPMODEL", () => ({ key: $("interp-model").value }));
    command("interp-target", "change", "Changing blend target", "FSRCNNX_SETINTERPTARGETFPS", () => ({
      value: $("interp-target").value === "auto" ? "auto" : Number($("interp-target").value),
    }));
    command("images", "change", "Updating image enhancement", "FSRCNNX_SETIMAGES", () => ({ on: $("images").checked }));
    command("hover-reveal", "change", "Updating hover reveal", "FSRCNNX_SETHOVERREVEAL", () => ({ on: $("hover-reveal").checked }));
    command("all-videos", "change", "Updating multi-video processing", "FSRCNNX_SETALLVIDEOS", () => ({ on: $("all-videos").checked }));
    command("neural-model", "change", "Changing neural model", "FSRCNNX_SETNEURALMODEL", () => ({ model: $("neural-model").value }));

    $("sharpen-str").addEventListener("input", () => {
      $("sharpen-val").textContent = Number($("sharpen-str").value).toFixed(1);
    });
    $("deband-str").addEventListener("input", () => {
      $("deband-val").textContent = Number($("deband-str").value).toFixed(1);
    });
    $("interp-avoff").addEventListener("input", () => {
      $("interp-avoff-val").textContent = $("interp-avoff").value;
    });

    for (const button of modeButtons) {
      button.addEventListener("click", () => runCommand(
        `Switching to ${button.textContent.trim().toLowerCase()} mode`,
        () => transport.send("FSRCNNX_SETMODE", { mode: button.dataset.mode }),
      ));
    }
  }

  function start() {
    bind();
    loading();
    void coordinator.refresh();
    refreshTimer = setIntervalRef(() => { void coordinator.refresh(); }, 1_000);
    return api;
  }

  function stop() {
    if (refreshTimer != null) clearIntervalRef(refreshTimer);
    refreshTimer = null;
  }

  const api = { start, stop, refresh: () => coordinator.refresh(), runCommand, render };
  return api;
}

if (typeof document !== "undefined" && globalThis.chrome?.tabs) {
  createPopupController({ document, chrome: globalThis.chrome }).start();
}

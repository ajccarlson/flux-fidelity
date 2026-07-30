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
  "fsrcnnx-hi": Object.freeze([
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv thresholds)"],
    ["force2", "Always ×2"],
    ["force4", "Always ×4"],
    ["force8", "Always ×8"],
  ]),
  artcnn: Object.freeze([
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv thresholds)"],
    ["force2", "Always ×2"],
    ["force4", "Always ×4"],
    ["force8", "Always ×8"],
  ]),
  neural: Object.freeze([
    ["display", "Fit display with SSimDS (recommended)"],
    ["force2", "Exact ×2 output"],
    ["native", "Native model scale"],
  ]),
});

const STATIC_INTERPOLATION_MODELS = Object.freeze([
  ["rife_v4.26", "RIFE 4.26 (default; may wave on bright motion)"],
  ["rife_v4.26_fp16", "RIFE 4.26 FP16 (experimental)"],
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

    if (!activeTab || (typeof activeTab.url === "string" && !isSupportedPageUrl(activeTab.url))) {
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

// Builds an SVG path for a series scaled to the viewBox, plus a dashed reference
// line at `budget`. Returned as data rather than applied to the DOM so it can be
// unit-tested without a document.
export function encodeSparkline(samples, budgetMs, { width = 240, height = 48 } = {}) {
  const series = (Array.isArray(samples) ? samples : [])
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (series.length < 2) return { line: "", budget: "", peak: 0, scaleMs: 0 };
  const peak = Math.max(...series);
  const budget = Number.isFinite(budgetMs) && budgetMs > 0 ? budgetMs : 0;
  // Scale to whichever is larger so the budget line is always visible and a series
  // that never approaches it does not look alarming.
  const scaleMs = Math.max(peak, budget) * 1.1 || 1;
  const x = (index) => (index / (series.length - 1)) * width;
  const y = (value) => height - Math.min(1, value / scaleMs) * height;
  const line = series
    .map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)} ${y(value).toFixed(1)}`)
    .join(" ");
  const budgetY = budget ? y(budget).toFixed(1) : null;
  return {
    line,
    budget: budgetY == null ? "" : `M0 ${budgetY} L${width} ${budgetY}`,
    peak,
    scaleMs,
  };
}

// Summarizes the same series as text. The chart is aria-hidden, so this is the
// accessible representation rather than a caption for it.
export function describeEncodeSeries(samples, budgetMs) {
  const series = (Array.isArray(samples) ? samples : [])
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (!series.length) return "No frames measured yet.";
  const mean = series.reduce((total, value) => total + value, 0) / series.length;
  const peak = Math.max(...series);
  const budget = Number.isFinite(budgetMs) && budgetMs > 0 ? budgetMs : null;
  const over = budget ? series.filter((value) => value > budget).length : 0;
  const parts = [
    `${series.length} frames`,
    `mean ${mean.toFixed(1)} ms`,
    `peak ${peak.toFixed(1)} ms`,
  ];
  if (budget) {
    parts.push(`budget ${budget.toFixed(1)} ms`);
    // The number that actually matters: encode time above the frame interval is
    // when frames start being missed.
    parts.push(over ? `${over} over budget` : "none over budget");
  }
  return `${parts.join(" · ")}.`;
}

// Turns the status payload into a pasteable report. getStatus() already computes
// far more than the popup shows; without this, a user whose video looks wrong has
// nothing to hand over and no way to reach validate.html.
export function formatDiagnostics(status, { version = null } = {}) {
  if (!status || typeof status !== "object") return "Flux Fidelity diagnostics: status unavailable";
  const renderer = status.renderer || {};
  const lines = [
    `Flux Fidelity diagnostics${version ? ` (v${version})` : ""}`,
    `mode: ${status.mode ?? "unknown"} (requested ${status.requestedMode ?? "unknown"})`,
    `engine: ${status.engine ?? "unknown"} (active ${status.activeEngine ?? "unknown"})`,
    `policy: ${status.policy ?? "unknown"}  chainDepth: ${status.chainDepth ?? "?"}`,
    `video present: ${status.hasVideo === true}  protected: ${status.protected === true}` +
      `${status.protectedReason ? ` (${status.protectedReason})` : ""}`,
    `presentation: ${formatPresentation(status)}`,
    `renderer phase: ${renderer.phase ?? "unknown"}  attempts: ${status.renderAttempts ?? 0}`,
    `frames: ${status.frameCount ?? 0}`,
    `gpu: ${status.gpuState ?? "unknown"}`,
  ];
  if (renderer.fallback) {
    lines.push(`fallback: ${renderer.fallback.from} -> ${renderer.fallback.to}` +
      `${renderer.fallback.code ? ` (${renderer.fallback.code})` : ""}`);
  }
  if (status.colorSupport) {
    lines.push(`color: ${status.colorSupport.code ?? "unknown"}`);
  }
  if (status.persistence) {
    lines.push(`persistence: ${status.persistence.state ?? "unknown"}` +
      `${status.persistence.error ? ` (${status.persistence.error})` : ""}`);
  }
  return lines.join("\n");
}

export function formatPresentation(status) {
  const presentation = status?.renderer?.presentation || status?.presentation;
  const source = presentation?.source;
  const output = presentation?.output;
  const dimension = (value) => (Number.isFinite(value) && value > 0 ? Math.round(value) : null);
  const sw = dimension(source?.width), sh = dimension(source?.height);
  const ow = dimension(output?.width), oh = dimension(output?.height);
  if (!sw || !sh || !ow || !oh) return "—";
  const scale = ow / sw;
  const factor = Number.isFinite(scale) && scale > 0
    ? `${Math.round(scale * 100) / 100}×`
    : "";
  return `${sw}×${sh} → ${ow}×${oh}${factor ? ` (${factor})` : ""}`;
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
    "color-hdr-unsupported": "HDR video remains on the browser's native renderer because enhancement supports SDR only.",
    "color-wide-gamut-unsupported": "Wide-gamut video remains on the browser's native renderer because its color cannot be preserved.",
    "color-space-unsupported": "This video's color space is outside the validated BT.709 SDR boundary.",
    "color-metadata-unavailable": "The video's decoded color metadata could not be verified safely.",
    "renderer unavailable": "The WebGPU renderer is unavailable on this page.",
    "lifecycle-pending": "The page changed while the renderer was starting. The request remains pending.",
    superseded: "A newer page or setting change replaced this request.",
    "startup-failed": "The extension could not start on this page.",
    "status-failed": "The page status could not be read.",
    "command-failed": "The page could not apply that setting.",
  };
  const generic = messages["command-failed"];
  // content.js always sets error:"command-failed" and puts the specific cause in
  // `reason`. Checking `error` first therefore matched the generic entry every
  // time and discarded the real reason for every content-level failure, so the
  // user only ever saw "The page could not apply that setting." Prefer a specific
  // reason over the catch-all; named codes still win.
  if (messages[code] && messages[code] !== generic) return messages[code];
  const specific = [result?.message, result?.reason]
    .find((value) => typeof value === "string" && value.trim() && value !== code);
  if (specific) return `${generic} ${specific.trim()}`;
  if (messages[code]) return messages[code];
  return generic;
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
    releasing: "releasing",
    recovering: "recovering",
    failed: "failed",
  }[status.gpuState];
  if (direct) return direct;

  const runtime = objectRecord(status.runtime) ? status.runtime : null;
  if (runtime) {
    if (runtime.api === "unavailable") return "unavailable";
    if (runtime.resources?.phase === "releasing") return "releasing";
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
    releasing: "releasing…",
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

function sourceBlockMessage(reason) {
  return ({
    drm: "This DRM-protected video does not permit the extension to read its frames.",
    tainted: "This cross-origin video does not permit the extension to read its pixels.",
    "color-hdr-unsupported": "HDR video is unsupported; the browser's native renderer remains active.",
    "color-wide-gamut-unsupported": "Wide-gamut video is unsupported because enhancement cannot preserve its color.",
    "color-space-unsupported": "This video's color space is outside the validated BT.709 SDR boundary.",
    "color-metadata-unavailable": "Decoded color metadata is unavailable or incomplete, so the native renderer remains active.",
  })[reason] || "The requested renderer is blocked by this video source.";
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
    "fsrcnnx-hi": "FSRCNNX high",
    artcnn: "ArtCNN",
    neural: "Neural",
  })[engine] || "Renderer";
}

function formatNeuralLabel(label, scale, option = false) {
  const text = String(label || "");
  const includesScale = Number.isInteger(scale) && scale > 0 && new RegExp(
    `(?:^|[^0-9])(?:${scale}\\s*[x×]|[x×]\\s*${scale})(?:$|[^0-9])`,
    "i",
  ).test(text);
  if (!Number.isInteger(scale) || scale < 1 || includesScale) {
    return text;
  }
  return option ? `${text} (${scale}×)` : `${scale}× ${text}`;
}

function isCdaNeuralModel(status, selectedKey) {
  const catalogLabel = Array.isArray(status.neuralModels)
    ? status.neuralModels.find(({ key }) => key === selectedKey)?.label
    : null;
  return [selectedKey, status.neural?.label, catalogLabel]
    .some((value) => typeof value === "string" && /\bcda(?:-vsr)?\b/i.test(value));
}

function formatReadyNeuralNote(neural, warnForCda) {
  const pipelineMean = Number(neural.meanRunMs);
  const pipelineRuns = Number(neural.runs);
  const hasPipelineMean = Number.isFinite(pipelineMean) && pipelineMean >= 0 &&
    (pipelineMean > 0 || (Number.isFinite(pipelineRuns) && pipelineRuns > 0));
  const coreMean = Number(neural.mu);
  const mean = hasPipelineMean
    ? pipelineMean
    : Number.isFinite(coreMean) && coreMean >= 0 ? coreMean : 0;
  const count = (value) => {
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? Math.trunc(number) : 0;
  };
  const tiles = count(neural.lastTiles);
  const nativeScale = Number(neural.nativeScale ?? neural.scale);
  const outputScale = Number(neural.outputScale);
  const parts = [];
  if (Number.isFinite(nativeScale) && nativeScale > 0 &&
      Number.isFinite(outputScale) && outputScale > 0 &&
      Math.abs(nativeScale - outputScale) > 0.01) {
    parts.push(`native ${nativeScale}× → output ${Number(outputScale.toFixed(2))}×`);
  }
  parts.push(
    `${hasPipelineMean ? "total" : "core"} mean ${mean.toFixed(1)} ms`,
    `${tiles} ${tiles === 1 ? "tile" : "tiles"}`,
    `skipped ${count(neural.skip)}`,
  );
  if (warnForCda) parts.push("high GPU/memory use");
  return parts.join(" · ");
}

function formatEngineModel(status, engine) {
  if (engine === "neural") {
    const neural = status.neural;
    const key = neural?.label || neural?.model || status.neuralModel;
    if (!key) return "—";
    const scale = Number(neural?.scale);
    return Number.isInteger(scale) && scale > 0
      ? formatNeuralLabel(key, scale)
      : `?× ${key}`;
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
  // Scoped by group: a second .modes row for interpolation would otherwise be
  // swept into the upscaler mode handler and send the wrong command.
  const modeButtons = [...documentRef.querySelectorAll('.modes[data-group="video"] button')];
  const interpModeButtons = [...documentRef.querySelectorAll('.modes[data-group="interpolation"] button')];
  const tabButtons = [...documentRef.querySelectorAll('[role="tab"]')];
  let activePanel = "panel-video";
  // Tab buttons are navigation, not settings, so they are excluded from the
  // blanket disable applied while a command is in flight or the page is not
  // connected. Including them meant the tabs went dead in exactly the situation
  // where a user most wants to reach the Performance panel — and a disabled
  // button fires no click, which is how the browser probe caught it.
  const controls = [...documentRef.querySelectorAll("button, input, select")]
    .filter((element) => element.getAttribute?.("role") !== "tab");
  let ready = false;
  let commandBusy = 0;
  let currentStatus = null;
  let operationSerial = 0;
  let commandTail = Promise.resolve();
  let refreshTimer = null;
  let feedbackScope = "context";
  let forceControlSync = false;
  let pendingRefocus = null;

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
    // aria-busy tells assistive technology the form is intentionally inert rather
    // than broken, which matters most on the 30 s timeout path.
    documentRef.body?.setAttribute?.("aria-busy", globallyDisabled ? "true" : "false");
    if (globallyDisabled || !currentStatus) return;
    if (pendingRefocus && !pendingRefocus.disabled) {
      const target = pendingRefocus;
      pendingRefocus = null;
      try { target.focus?.(); } catch {}
    }

    const neural = currentStatus.engine === "neural";
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
      return sourceBlockMessage(renderer.blockedReason || status.protectedReason);
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
    if (interpolation?.phase === "blocked") {
      return sourceBlockMessage(interpolation.blockedReason || status.protectedReason);
    }
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
    // getStatus() has always reported presentation dimensions and the popup threw
    // them away, so the one question an upscaler exists to answer — what goes in
    // and what comes out — was unanswerable from the UI. "Frames: N" is a counter,
    // not a quality signal.
    renderTabIndicators(status);
    renderPerformance(status);
    const resolutionText = formatPresentation(status);
    $("s-resolution").textContent = resolutionText;
    $("s-resolution").className = `v ${resolutionText === "—" ? "" : "ok"}`.trimEnd();
    setText($("runtime-status"), runtimeMessage(status));

    const protectedMessage = status.protected ? sourceBlockMessage(status.protectedReason) : "";
    setText($("drm-banner"), protectedMessage);
    setVisible($("drm-banner"), !!protectedMessage);

    const engine = ["fsrcnnx", "fsrcnnx-hi", "artcnn", "neural"].includes(status.engine)
      ? status.engine
      : "fsrcnnx";
    const hasNeuralModels = Array.isArray(status.neuralModels) && status.neuralModels.length > 0;
    if (forceSync || documentRef.activeElement !== $("engine")) {
      reconcileSelectOptions(documentRef, $("engine"), [
        ["fsrcnnx", "FSRCNNX standard"],
        ["fsrcnnx-hi", "FSRCNNX high"],
        ["artcnn", "ArtCNN"],
        { value: "neural", label: "Neural (ONNX)", disabled: !hasNeuralModels },
      ], engine);
    }
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
          label: formatNeuralLabel(model.label || model.key, model.scale, true),
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
        const warnForCda = isCdaNeuralModel(status, selectedNeural);
        $("neural-note").textContent = status.neural?.ready
          ? formatReadyNeuralNote(status.neural, warnForCda)
          : (selectedNeural
              ? "Selected; initializes when video upscaling is active." +
                (warnForCda ? " High GPU/memory use." : "")
              : "No model selected.");
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
    syncCheckbox("hover-reveal", status.hoverReveal);
    syncCheckbox("all-videos", status.allVideos);
    syncCheckbox("idle-power-saving", status.idlePowerSaving);
    syncCheckbox("auto-quality-fallback", status.autoQualityFallback);
    syncCheckbox("images", status.images);
    // Mirrors the upscaler's Mode row: the pressed button is the current state, so
    // the control and the state are the same thing rather than a checkbox whose
    // meaning has to be read from its position.
    for (const button of interpModeButtons) {
      const on = button.dataset.mode === "on";
      const pressed = on === (status.interpolate === true);
      button.setAttribute("aria-pressed", pressed ? "true" : "false");
      button.dataset.active = pressed ? "1" : "0";
    }
    syncCheckbox("interp-diag", status.interpStaticPassthrough);
    syncCheckbox("interp-ladder", status.interpLadder);
    syncCheckbox("interp-invert", status.interpInvert);
    syncCheckbox("interp-autofallback", status.interpAutoFallback);

    setVisible($("sharpen-row"), status.sharpen === true);

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
    // Every control is disabled while a command is in flight, and disabling the
    // focused element moves focus to <body>. With no restoration a keyboard or
    // screen-reader user was ejected to the top of the popup after every single
    // toggle, for up to COMMAND_TIMEOUT_MS. Remember what had focus so
    // updateAvailability() can hand it back when the controls re-enable.
    const refocusTarget = documentRef.activeElement;
    pendingRefocus = controls.includes(refocusTarget) ? refocusTarget : null;
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
          // A stored preference the runtime actively refused is not the same as
          // one that is merely waiting for a runtime, and reporting both as
          // "activation is pending" hid every rejection behind a normal wait.
          const refused = result.pendingKind === "runtime-rejected" ||
            result.pendingKind === "runtime-error";
          if (refused) {
            feedback(
              `Setting saved, but the video runtime did not accept it${
                result.reason ? ` (${result.reason})` : ""}.`,
              "notice",
              "operation",
            );
          } else if (pending) {
            feedback("Setting saved; activation is pending.", "success", "operation");
          } else {
            // Nothing to report: the control now shows the new value, and a banner
            // for every routine change made the alert region constant noise.
            feedback("", "success", "operation");
          }
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

  // Tabs are wired manually rather than with <details> so the panels carry real
  // tab semantics: the previous disclosure summary was not a heading, which left
  // its fourteen controls unreachable by heading navigation. Arrow keys move
  // between tabs and only the selected tab sits in the tab order, per the ARIA
  // tabs pattern.
  function selectTab(panelId, { focus = false } = {}) {
    for (const tab of tabButtons) {
      const selected = tab.dataset.panel === panelId;
      tab.setAttribute("aria-selected", selected ? "true" : "false");
      tab.tabIndex = selected ? 0 : -1;
      const panel = $(tab.dataset.panel);
      if (panel) panel.hidden = !selected;
      if (selected && focus) { try { tab.focus?.(); } catch {} }
    }
    activePanel = panelId;
  }

  // A dot on the tab shows at a glance whether each feature is running, so the
  // state is visible without opening the panel. The dot is paired with hidden text
  // because colour alone is not an accessible signal, and the text becomes part of
  // the tab's accessible name.
  function renderTabIndicators(status) {
    // Three states rather than present/absent:
    //   off     — the mode is not enabled
    //   on      — enabled and actually running
    //   pending — enabled, but not running: no video playing, or it is suspended,
    //             recovering, or has failed
    //
    // A dot is always rendered and only its colour changes. The previous version
    // toggled the hidden attribute, which an author `display: inline-block` rule on
    // .tab-dot silently overrode — so the dot was permanently visible and
    // permanently green. The tests asserted the attribute rather than the rendered
    // result, so nothing caught it.
    //
    // "running" for upscaling is activeMode, which comes from
    // currentPresentedRuntime() and reports a mode only while frames are being
    // presented for the current video on a live device.
    const features = [
      ["tab-video", status?.mode === "upscale", status?.activeMode === "upscale"],
      [
        "tab-interpolation",
        status?.interpolate === true,
        status?.interpolationRuntime?.phase === "active",
      ],
    ];
    for (const [tabId, enabled, running] of features) {
      const state = !enabled ? "off" : running ? "on" : "pending";
      const dot = $(`${tabId}-dot`);
      if (dot) dot.dataset.state = state;
      // Written out as well as coloured: the state must not depend on hue alone.
      setText($(`${tabId}-state`), state === "on"
        ? " (on)"
        : state === "pending"
          ? " (on, not running)"
          : " (off)");
    }
  }

  // Only reads fields getStatus() already publishes. Everything drawn here also
  // appears as text, because the chart is aria-hidden.
  function renderPerformance(status) {
    const renderer = status?.renderer || {};
    const perf = renderer.performance || {};
    const samples = Array.isArray(renderer.encodeMs) ? renderer.encodeMs : [];
    const budgetMs = Number(perf.frameIntervalMs) || 0;
    const active = samples.length > 0;
    setVisible($("perf-idle"), !active);
    setVisible($("perf-body"), active);
    if (!active) return;

    const spark = encodeSparkline(samples, budgetMs);
    $("perf-encode-line")?.setAttribute("d", spark.line);
    $("perf-encode-budget")?.setAttribute("d", spark.budget);
    setText($("perf-encode-summary"), describeEncodeSeries(samples, budgetMs));

    // Decoded-frame counts come from the media element, so they populate whatever
    // the automatic-quality-fallback setting is. Reading them from the guard's
    // window meant they stayed blank for every default configuration.
    const quality = renderer.playbackQuality || null;
    const total = Number(quality?.totalFrames);
    const dropped = Number(quality?.droppedFrames);
    $("perf-presented").textContent = Number.isFinite(total) ? String(total) : "—";
    // A ratio rather than a bare count: 40 dropped frames means nothing without
    // knowing how many arrived alongside them.
    $("perf-dropped").textContent = Number.isFinite(dropped)
      ? (Number.isFinite(total) && total > 0
        ? `${dropped} (${Math.round((dropped / total) * 100)}%)`
        : String(dropped))
      : "—";
    // Requested versus effective engine: after a fallback these differ, and that
    // divergence was previously visible only in the console.
    const requested = status?.engine ?? "—";
    const effective = renderer.effectiveEngine ?? status?.activeEngine ?? requested;
    $("perf-engine").textContent = requested === effective
      ? String(requested)
      : `${requested} → ${effective}`;
    $("perf-output").textContent = formatPresentation(status);

    const interp = status?.interpStats || null;
    const interpActive = !!interp && interp.running === true;
    setVisible($("perf-interp-group"), interpActive);
    if (interpActive) {
      $("perf-interp-fps").textContent = `${interp.fpsIn ?? "—"} → ${interp.fpsOut ?? "—"} fps`;
      // framesOut counts enqueued frames and framesPresented counts committed ones;
      // they diverge under presentation backpressure, and the auto-fallback watches
      // the former. Showing both makes that difference observable.
      $("perf-interp-frames").textContent =
        `${interp.framesOut ?? 0} out / ${interp.framesPresented ?? 0} shown`;
      $("perf-interp-infer").textContent = interp.inferMeanMs
        ? `${interp.inferMs ?? 0} / ${interp.inferMeanMs} ms`
        : `${interp.inferMs ?? 0} ms`;
      $("perf-interp-stutter").textContent =
        `${interp.stutters ?? 0} (max gap ${interp.maxGapMs ?? 0} ms)`;
    }

    const neural = status?.neural || null;
    setVisible($("perf-neural-group"), !!neural);
    if (neural) {
      $("perf-neural-infer").textContent = Number.isFinite(neural.meanRunMs)
        ? `${(neural.lastRunMs ?? 0).toFixed?.(1) ?? neural.lastRunMs} / ${neural.meanRunMs.toFixed(1)} ms`
        : "—";
      $("perf-neural-tiles").textContent = Number.isFinite(neural.lastTiles)
        ? String(neural.lastTiles)
        : "—";
      // Reset runs versus recurrent runs exposes how often temporal state is
      // discarded, which is otherwise invisible and affects perceived quality.
      const resets = Number(neural.temporalResetRuns);
      const recurrent = Number(neural.temporalRecurrentRuns);
      $("perf-neural-temporal").textContent =
        Number.isFinite(resets) && Number.isFinite(recurrent)
          ? `${resets} reset / ${recurrent} recurrent`
          : "—";
      $("perf-neural-skip").textContent = Number.isFinite(neural.skip) ? String(neural.skip) : "—";
    }
  }

  function bindTabs() {
    // Assert the initial selection here rather than trusting the markup's
    // aria-selected, so the controller owns the "exactly one tab selected"
    // invariant and the two cannot disagree.
    if (tabButtons.length) selectTab(activePanel);
    tabButtons.forEach((tab, index) => {
      tab.addEventListener("click", () => selectTab(tab.dataset.panel));
      tab.addEventListener("keydown", (event) => {
        const key = event.key;
        const delta = key === "ArrowRight" ? 1 : key === "ArrowLeft" ? -1 : 0;
        if (!delta && key !== "Home" && key !== "End") return;
        event.preventDefault?.();
        const target = key === "Home"
          ? 0
          : key === "End"
            ? tabButtons.length - 1
            : (index + delta + tabButtons.length) % tabButtons.length;
        selectTab(tabButtons[target].dataset.panel, { focus: true });
      });
    });
  }

  function bind() {
    bindTabs();
    // Destructive, so it confirms first. There is no undo: the tombstones are
    // convergent and will propagate to every other tab for this origin.
    $("forget-site")?.addEventListener("click", () => {
      const proceed = globalThis.confirm?.(
        "Clear all Flux Fidelity settings for this site and return it to defaults?",
      );
      if (proceed === false) return;
      return runCommand("Forgetting this site", () => transport.send("FSRCNNX_FORGETSITE"));
    });
    $("copy-diagnostics")?.addEventListener("click", async () => {
      const report = formatDiagnostics(currentStatus, {
        version: chromeApi?.runtime?.getManifest?.()?.version ?? null,
      });
      try {
        await globalThis.navigator?.clipboard?.writeText(report);
        feedback("Diagnostics copied to the clipboard.", "success", "operation");
      } catch (error) {
        feedback(`Diagnostics could not be copied: ${errorText(error)}`, "error", "operation");
      }
    });
    command("engine", "change", "Changing upscaler engine", "FSRCNNX_SETENGINE", () => ({ engine: $("engine").value }));
    command("artvariant", "change", "Changing ArtCNN variant", "FSRCNNX_SETARTVARIANT", () => ({ variant: $("artvariant").value }));
    command("policy", "change", "Changing upscale policy", "FSRCNNX_SETPOLICY", () => ({ policy: $("policy").value }));
    command("ssimds", "change", "Updating downscaling", "FSRCNNX_SETSSIMDS", () => ({ on: $("ssimds").checked }));
    command("sharpen", "change", "Updating sharpening", "FSRCNNX_SETSHARPEN", () => ({ on: $("sharpen").checked }));
    command("sharpen-str", "change", "Changing sharpen strength", "FSRCNNX_SETSHARPENSTR", () => ({ strength: Number($("sharpen-str").value) }));
    for (const button of interpModeButtons) {
      button.addEventListener("click", () => runCommand(
        button.dataset.mode === "on" ? "Enabling interpolation" : "Disabling interpolation",
        () => transport.send("FSRCNNX_SETINTERPOLATE", { on: button.dataset.mode === "on" }),
      ));
    }
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
    command("idle-power-saving", "change", "Updating idle power saving", "FSRCNNX_SETIDLEPOWERSAVING", () => ({ on: $("idle-power-saving").checked }));
    command("auto-quality-fallback", "change", "Updating automatic quality fallback", "FSRCNNX_SETAUTOQUALITYFALLBACK", () => ({ on: $("auto-quality-fallback").checked }));
    command("neural-model", "change", "Changing neural model", "FSRCNNX_SETNEURALMODEL", () => ({ model: $("neural-model").value }));

    $("sharpen-str").addEventListener("input", () => {
      $("sharpen-val").textContent = Number($("sharpen-str").value).toFixed(1);
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

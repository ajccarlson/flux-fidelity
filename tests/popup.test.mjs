import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import {
  PopupTransport,
  StatusCoordinator,
  createPopupController,
  describeCommandFailure,
  describeEncodeSeries,
  describeAvoidedWork,
  describeGpuSeries,
  encodeSparkline,
  formatDiagnostics,
  formatPresentation,
  isSupportedPageUrl,
  reconcileSelectOptions,
} from "../src/popup.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function flush(turns = 8) {
  for (let i = 0; i < turns; i++) await new Promise((resolve) => setImmediate(resolve));
}

class FakeElement {
  constructor(tagName = "div", id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.attributes = new Map();
    this.checked = false;
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.disabled = false;
    this.hidden = false;
    this.listeners = new Map();
    this.style = { display: "" };
    this.textContent = "";
    this.value = "";
  }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }

  emit(type, event = {}) {
    const emitted = { currentTarget: this, target: this, type, ...event };
    for (const listener of [...(this.listeners.get(type) || [])]) listener(emitted);
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  replaceChildren(...children) {
    this.children = children;
    if (this.tagName === "SELECT") this.value = children[0]?.value ?? "";
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
    this.elements = new Map();
    this.controls = [];
    this.modeButtons = [];
    this.interpModeButtons = [];
    this.tabButtons = [];
  }

  // Mirrors popup.html: a tab button carries role="tab" and a data-panel pointing
  // at the panel it controls.
  addTab(id, panelId) {
    const tab = this.add(id, "button");
    tab.setAttribute("role", "tab");
    tab.dataset.panel = panelId;
    this.tabButtons.push(tab);
    this.add(panelId, "div");
    return tab;
  }

  add(id, tagName = "div", values = {}) {
    const element = Object.assign(new FakeElement(tagName, id), values);
    this.elements.set(id, element);
    if (["BUTTON", "INPUT", "SELECT"].includes(element.tagName)) this.controls.push(element);
    return element;
  }

  createElement(tagName) {
    return new FakeElement(tagName);
  }

  getElementById(id) {
    return this.elements.get(id) ?? null;
  }

  querySelectorAll(selector) {
    if (selector === '.modes[data-group="video"] button') return this.modeButtons;
    if (selector === '.modes[data-group="interpolation"] button') return this.interpModeButtons;
    if (selector === "button, input, select") return this.controls;
    if (selector === '[role="tab"]') return this.tabButtons;
    return [];
  }

  addInterpMode(mode, label) {
    const button = this.add(`interp-mode-${mode}`, "button", { textContent: label });
    button.dataset.mode = mode;
    this.interpModeButtons.push(button);
    return button;
  }

  addMode(mode, label) {
    const button = this.add(`mode-${mode}`, "button", { textContent: label });
    button.dataset.mode = mode;
    button.dataset.active = mode === "off" ? "1" : "0";
    button.setAttribute("aria-pressed", mode === "off" ? "true" : "false");
    this.modeButtons.push(button);
    return button;
  }
}

function popupDocument() {
  const document = new FakeDocument();

  for (const id of [
    "s-webgpu", "s-video", "s-model", "s-frames", "s-resolution", "runtime-status", "drm-banner",
    "copy-diagnostics", "forget-site",
    "perf-idle", "perf-body", "perf-encode-line", "perf-encode-budget", "perf-encode-summary",
    "perf-gpu-summary", "perf-avoided-summary",
    "perf-presented", "perf-dropped", "perf-engine", "perf-output",
    "perf-interp-group", "perf-interp-fps", "perf-interp-frames", "perf-interp-infer",
    "perf-interp-stutter",
    "perf-neural-group", "perf-neural-infer", "perf-neural-tiles", "perf-neural-temporal",
    "perf-neural-skip",
    "operation-status", "neuralrow", "neural-note", "sharpen-row", "sharpen-val",
    "multi-count", "image-count", "interp-res-row",
    "interp-target-hz", "interp-avoff-val", "interp-stats",
  ]) document.add(id);

  for (const id of [
    "tab-video-dot", "tab-video-state",
    "tab-interpolation-dot", "tab-interpolation-state",
  ]) document.add(id, "span");

  document.addTab("tab-video", "panel-video");
  document.addTab("tab-interpolation", "panel-interpolation");
  document.addTab("tab-advanced", "panel-advanced");
  document.addTab("tab-performance", "panel-performance");

  for (const [id, value] of [
    ["engine", "fsrcnnx"],
    ["artvariant", "ArtCNN_C4F32"],
    ["neural-model", ""],
    ["policy", "display"],
    ["interp-model", "rife_v4.26"],
    ["interp-res", "auto"],
    ["interp-target", "auto"],
  ]) document.add(id, "select", { value });

  for (const [id, value] of [
    ["ssimds", ""],
    ["sharpen", ""],
    ["sharpen-str", "1"],
    ["hover-reveal", ""],
    ["all-videos", ""],
    ["idle-power-saving", ""],
    ["auto-quality-fallback", ""],
    ["images", ""],
    ["interpolate", ""],
    ["interp-avoff", "0"],
    ["interp-diag", ""],
    ["interp-ladder", ""],
    ["interp-invert", ""],
    ["interp-autofallback", ""],
  ]) document.add(id, "input", { value });

  document.getElementById("ssimds").checked = true;
  document.getElementById("interp-diag").checked = true;
  document.getElementById("interp-invert").checked = true;
  document.getElementById("interp-autofallback").checked = false;

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Loading models…";
  document.getElementById("neural-model").children = [placeholder];

  document.addMode("off", "Off");
  document.addMode("upscale", "Upscale");
  document.addInterpMode("off", "Off");
  document.addInterpMode("on", "On");
  return document;
}

function readyStatus(overrides = {}) {
  const runtime = {
    api: "available",
    adapter: "unrequested",
    device: "uninitialized",
    recovery: { phase: "idle", attempt: 0, maxAttempts: 3 },
    lastFailure: null,
    ...(overrides.runtime || {}),
    recovery: {
      phase: "idle", attempt: 0, maxAttempts: 3,
      ...(overrides.runtime?.recovery || {}),
    },
    resources: {
      phase: "idle", reason: null,
      ...(overrides.runtime?.resources || {}),
    },
  };
  const persistence = {
    state: "ready", operation: null, errorOperation: null, pendingWrites: 0, error: null,
    ...(overrides.persistence || {}),
  };
  return {
    activeMode: "off",
    activeEngine: overrides.engine || "fsrcnnx",
    allVideos: false,
    artVariant: "ArtCNN_C4F32",
    engine: "fsrcnnx",
    frameCount: 12,
    gpuState: "idle",
    hasVideo: true,
    hoverReveal: false,
    idlePowerSaving: false,
    autoQualityFallback: false,
    imageCount: 0,
    images: false,
    interpAutoFallback: false,
    interpAvOffsetMs: 0,
    interpInvert: true,
    interpLadder: false,
    interpModel: "rife_v4.26",
    interpResMode: "auto",
    interpStaticPassthrough: true,
    interpTargetFps: "auto",
    interpolate: false,
    mode: "off",
    model: "FSRCNNX_x2_16-0-4-1",
    multiCount: 0,
    neuralModels: [],
    policy: "display",
    protected: false,
    scale: 2,
    sharpen: false,
    sharpenStrength: 1,
    ssimds: true,
    webgpu: true,
    ...overrides,
    runtime,
    persistence,
  };
}

// The tab buttons are real <button> elements, so querySelectorAll returns them —
// but they are navigation rather than settings and are deliberately never disabled.
// Assertions about the blanket disable therefore mean command controls only.
function commandControls(document) {
  return document.controls.filter((control) => control.getAttribute("role") !== "tab");
}

function controllerHarness(options = {}) {
  const document = options.document ?? popupDocument();
  const intervalCallbacks = new Map();
  let intervalId = 0;
  const controller = createPopupController({
    document,
    transport: options.transport ?? { send: async () => readyStatus() },
    setInterval(callback) {
      const id = ++intervalId;
      intervalCallbacks.set(id, callback);
      return id;
    },
    clearInterval(id) { intervalCallbacks.delete(id); },
  });
  return { controller, document, intervalCallbacks };
}

test("page URL support is limited to http, https, and file pages", () => {
  for (const url of [
    "http://example.test/watch",
    "https://example.test/watch",
    "file:///tmp/video.html",
  ]) assert.equal(isSupportedPageUrl(url), true, url);

  for (const url of [
    "chrome://settings/",
    "edge://extensions/",
    "data:text/html,hello",
    "javascript:void(0)",
    "not a URL",
    "",
  ]) assert.equal(isSupportedPageUrl(url), false, url);
});

test("PopupTransport distinguishes tab-query, unsupported-page, and receiver failures", async () => {
  const queryFailure = new PopupTransport({
    tabs: { query: async () => { throw new Error("tabs permission unavailable"); } },
  });
  assert.deepEqual(await queryFailure.send("FSRCNNX_STATUS"), {
    ok: false,
    error: "tab-query-failed",
    reason: "tabs permission unavailable",
  });

  let unsupportedSent = false;
  const unsupported = new PopupTransport({ tabs: {
    query: async () => [{ id: 4, url: "chrome://settings/" }],
    sendMessage: async () => { unsupportedSent = true; },
  } });
  assert.deepEqual(await unsupported.send("FSRCNNX_STATUS"), {
    ok: false,
    error: "unsupported-page",
    reason: "The extension cannot run on this browser page.",
  });
  assert.equal(unsupportedSent, false);

  const missingReceiver = new PopupTransport({ tabs: {
    query: async () => [{ id: 5, url: "https://example.test/video" }],
    sendMessage: async () => { throw new Error("Could not establish connection. Receiving end does not exist."); },
  } });
  assert.deepEqual(await missingReceiver.send("FSRCNNX_SETMODE", { mode: "upscale" }), {
    ok: false,
    error: "no-content-script",
    reason: "Could not establish connection. Receiving end does not exist.",
  });
});

test("PopupTransport sends to http and file tabs and rejects malformed responses", async () => {
  const messages = [];
  let url = "http://example.test/video";
  let response = { loading: true };
  const transport = new PopupTransport({ tabs: {
    query: async () => [{ id: 17, url }],
    sendMessage: async (tabId, message) => {
      messages.push([tabId, message]);
      return response;
    },
  } });

  assert.deepEqual(await transport.send("FSRCNNX_STATUS"), { loading: true });
  url = "file:///tmp/player.html";
  response = { ok: true, mode: "upscale" };
  assert.deepEqual(await transport.send("FSRCNNX_SETMODE", { mode: "upscale" }), response);
  assert.deepEqual(messages, [
    [17, { type: "FSRCNNX_STATUS" }],
    [17, { type: "FSRCNNX_SETMODE", mode: "upscale" }],
  ]);

  response = null;
  assert.deepEqual(await transport.send("FSRCNNX_STATUS"), {
    ok: false,
    error: "invalid-response",
    reason: "The page returned an invalid response.",
  });
  response = { mode: "upscale" };
  assert.deepEqual(await transport.send("FSRCNNX_SETMODE", { mode: "upscale" }), {
    ok: false,
    error: "invalid-response",
    reason: "The command response was incomplete.",
  });
});

test("PopupTransport can use an injected active tab without persistent URL access", async () => {
  const messages = [];
  const transport = new PopupTransport({ tabs: {
    query: async () => [{ id: 23 }],
    sendMessage: async (tabId, message) => {
      messages.push([tabId, message]);
      return { loading: false, failed: false, statusVersion: 1 };
    },
  } });

  assert.deepEqual(await transport.send("FSRCNNX_STATUS"), {
    loading: false, failed: false, statusVersion: 1,
  });
  assert.deepEqual(messages, [[23, { type: "FSRCNNX_STATUS" }]]);
});

test("PopupTransport converts a silent receiver into a bounded timeout result", async () => {
  const timers = new Map();
  let nextTimer = 0;
  const transport = new PopupTransport({ tabs: {
    query: async () => [{ id: 2, url: "https://example.test/" }],
    sendMessage: () => new Promise(() => {}),
  } }, {
    setTimer(callback, delay) {
      const id = ++nextTimer;
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimer(id) { timers.delete(id); },
  });

  const result = transport.send("FSRCNNX_STATUS");
  await flush(1);
  const pending = [...timers.values()];
  assert.equal(pending.length, 1);
  assert.equal(pending[0].delay, 3_000);
  pending[0].callback();
  assert.deepEqual(await result, {
    ok: false,
    error: "response-timeout",
    reason: "No response after 3000 ms",
  });
  assert.equal(timers.size, 0);
});

test("StatusCoordinator serializes bursts, skips superseded data, and applies only the newest response", async () => {
  const requests = [];
  const applied = [];
  let active = 0;
  let maximumActive = 0;
  const coordinator = new StatusCoordinator(() => {
    const request = deferred();
    requests.push(request);
    active++;
    maximumActive = Math.max(maximumActive, active);
    return request.promise.finally(() => { active--; });
  }, (status) => applied.push(status));

  const first = coordinator.refresh();
  const second = coordinator.refresh();
  assert.strictEqual(first, second, "callers in one burst share the drain promise");
  assert.equal(requests.length, 1);
  requests[0].resolve({ frameCount: 1 });
  await flush(2);
  assert.equal(requests.length, 2, "one trailing refresh represents the newer request");
  assert.deepEqual(applied, [], "the superseded first response is never rendered");

  requests[1].resolve({ frameCount: 2 });
  await Promise.all([first, second]);
  assert.equal(maximumActive, 1);
  assert.deepEqual(applied, [{ frameCount: 2 }]);
});

test("StatusCoordinator fences responses captured before a mutation", async () => {
  const requests = [];
  const applied = [];
  const coordinator = new StatusCoordinator(() => {
    const request = deferred();
    requests.push(request);
    return request.promise;
  }, (status) => applied.push(status));

  const stale = coordinator.refresh();
  coordinator.invalidate();
  requests[0].resolve({ policy: "display" });
  await stale;
  assert.deepEqual(applied, []);

  const fresh = coordinator.refresh();
  requests[1].resolve({ policy: "force2" });
  await fresh;
  assert.deepEqual(applied, [{ policy: "force2" }]);
});

test("select reconciliation safely replaces a placeholder with exactly one neural model", () => {
  const document = new FakeDocument();
  const select = document.add("neural-model", "select", { value: "" });
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Loading models…";
  select.children = [placeholder];

  reconcileSelectOptions(document, select, [{
    value: "local2x",
    label: "<img src=x onerror=alert(1)> (2×)",
  }], "local2x");

  assert.equal(select.children.length, 1);
  assert.notStrictEqual(select.children[0], placeholder);
  assert.equal(select.children[0].value, "local2x");
  assert.equal(select.children[0].textContent, "<img src=x onerror=alert(1)> (2×)");
  assert.equal(select.value, "local2x");
});

test("select reconciliation detects same-count catalog changes and preserves identical options", () => {
  const document = new FakeDocument();
  const select = document.add("models", "select");
  reconcileSelectOptions(document, select, [["one", "Model one"], ["two", "Model two"]], "two");
  const firstChildren = [...select.children];
  const firstSignature = select.dataset.optionSignature;

  reconcileSelectOptions(document, select, [["one", "Model one"], ["three", "Model three"]], "three");
  assert.equal(select.children.length, 2);
  assert.notStrictEqual(select.children[0], firstChildren[0]);
  assert.notEqual(select.dataset.optionSignature, firstSignature);
  assert.deepEqual(select.children.map(({ value, textContent }) => [value, textContent]), [
    ["one", "Model one"],
    ["three", "Model three"],
  ]);
  assert.equal(select.value, "three");

  const stableChildren = [...select.children];
  reconcileSelectOptions(document, select, [["one", "Model one"], ["three", "Model three"]], "one");
  assert.strictEqual(select.children[0], stableChildren[0]);
  assert.strictEqual(select.children[1], stableChildren[1]);
  assert.equal(select.value, "one");
});

test("controller replaces the neural placeholder and notices a same-size model catalog update", () => {
  const { controller, document } = controllerHarness();
  const select = document.getElementById("neural-model");
  const placeholder = select.children[0];

  controller.render(readyStatus({
    engine: "neural",
    neuralModel: "span2x_a",
    neuralModels: [{ key: "span2x_a", label: "SPAN A", scale: 2 }],
  }));
  assert.equal(select.children.length, 1);
  assert.notStrictEqual(select.children[0], placeholder);
  assert.equal(select.children[0].textContent, "SPAN A (2×)");
  assert.equal(select.value, "span2x_a");
  const firstModelOption = select.children[0];

  controller.render(readyStatus({
    engine: "neural",
    neuralModel: "span4x_b",
    neuralModels: [{ key: "span4x_b", label: "SPAN B", scale: 4 }],
  }));
  assert.equal(select.children.length, 1);
  assert.notStrictEqual(select.children[0], firstModelOption);
  assert.equal(select.children[0].value, "span4x_b");
  assert.equal(select.children[0].textContent, "SPAN B (4×)");
  assert.equal(select.value, "span4x_b");
});

test("controller deduplicates neural scales and reports CDA total performance", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    engine: "neural",
    activeEngine: "neural",
    neuralModel: "cda-vsr-4x",
    policy: "force2",
    neuralModels: [{
      key: "cda-vsr-4x",
      label: "CDA-VSR 4x (mixed FP16)",
      scale: 4,
    }],
    neural: {
      model: "cda-vsr-4x",
      label: "CDA-VSR 4x (mixed FP16)",
      scale: 4,
      nativeScale: 4,
      outputScale: 2,
      ready: true,
      meanRunMs: 75.25,
      runs: 4,
      mu: 41.5,
      lastTiles: 6,
      skip: 3,
    },
  }));

  assert.equal(
    document.getElementById("neural-model").children[0].textContent,
    "CDA-VSR 4x (mixed FP16)",
  );
  assert.equal(document.getElementById("s-model").textContent, "CDA-VSR 4x (mixed FP16)");
  assert.equal(document.getElementById("policy").value, "force2");
  assert.equal(
    document.getElementById("neural-note").textContent,
    "native 4× → output 2× · total mean 75.3 ms · 6 tiles · skipped 3 · high GPU/memory use",
  );
});

test("controller falls back to core neural timing and warns before CDA initialization", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    engine: "neural",
    activeEngine: "neural",
    neuralModel: "local2x",
    neuralModels: [{ key: "local2x", label: "Local 2x", scale: 2 }],
    neural: {
      model: "local2x",
      label: "Local 2x",
      scale: 2,
      ready: true,
      meanRunMs: 0,
      runs: 0,
      mu: 20.25,
      lastTiles: 1,
      skip: 0,
    },
  }));

  assert.equal(document.getElementById("neural-model").children[0].textContent, "Local 2x");
  assert.equal(
    document.getElementById("neural-note").textContent,
    "core mean 20.3 ms · 1 tile · skipped 0",
  );

  controller.render(readyStatus({
    engine: "neural",
    activeEngine: "neural",
    neuralModel: "cda-vsr-4x",
    neuralModels: [{
      key: "cda-vsr-4x",
      label: "CDA-VSR 4x (mixed FP16)",
      scale: 4,
    }],
    neural: null,
  }));
  assert.equal(
    document.getElementById("neural-note").textContent,
    "Selected; initializes when video upscaling is active. High GPU/memory use.",
  );
});

test("controller presents FSRCNNX High with its fixed-scale policies and model label", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    engine: "fsrcnnx-hi",
    activeEngine: "fsrcnnx-hi",
    model: "FSRCNNX_x2_56-16-4-1",
    policy: "force8",
    scale: 8,
  }));

  const engine = document.getElementById("engine");
  const policy = document.getElementById("policy");
  assert.equal(engine.value, "fsrcnnx-hi");
  assert.equal(engine.children.find(({ value }) => value === "fsrcnnx-hi")?.textContent,
    "FSRCNNX high");
  assert.deepEqual(policy.children.map(({ value }) => value),
    ["display", "auto", "force2", "force4", "force8"]);
  assert.equal(policy.value, "force8");
  assert.equal(document.getElementById("s-model").textContent, "8× x2 high");
});

test("controller disables every command while loading or failed and enables only applicable ready controls", () => {
  const { controller, document } = controllerHarness();

  controller.render({ loading: true });
  assert.equal(document.getElementById("s-webgpu").textContent, "checking…");
  assert.ok(commandControls(document).every((control) => control.disabled));

  controller.render({ failed: true, error: "startup-failed", reason: "GPU adapter rejected" });
  assert.equal(document.getElementById("s-webgpu").textContent, "failed");
  assert.match(document.getElementById("operation-status").textContent, /could not start/i);
  assert.match(document.getElementById("operation-status").textContent, /GPU adapter rejected/);
  assert.ok(commandControls(document).every((control) => control.disabled));

  controller.render(readyStatus({
    engine: "artcnn",
    interpolate: true,
  }));
  assert.equal(document.getElementById("engine").disabled, false);
  assert.equal(document.getElementById("engine").children.find(({ value }) => value === "neural")?.disabled, true,
    "the neural engine option stays unavailable while the bundled catalog is empty");
  assert.equal(document.getElementById("policy").disabled, false);
  assert.equal(document.getElementById("artvariant").disabled, false);
  assert.equal(document.getElementById("neural-model").disabled, true);
  assert.equal(document.getElementById("interp-model").disabled, false);
  assert.equal(document.getElementById("interp-avoff").disabled, false);

  controller.render(readyStatus({
    engine: "neural",
    neuralModel: "local2x",
    neuralModels: [{ key: "local2x", label: "Local 2x", scale: 2 }],
  }));
  assert.equal(document.getElementById("engine").children.find(({ value }) => value === "neural")?.disabled, false);
  assert.equal(document.getElementById("neural-model").disabled, false);
  assert.equal(document.getElementById("policy").disabled, false);
  assert.deepEqual(document.getElementById("policy").children.map(({ value }) => value),
    ["display", "force2", "native"]);
  assert.equal(document.getElementById("all-videos").disabled, true);
  assert.equal(document.getElementById("artvariant").disabled, true);
  assert.equal(document.getElementById("interp-model").disabled, true);
});

test("settings failures preserve the truthful WebGPU label and do not repeat their reason", () => {
  const { controller, document } = controllerHarness();
  const webgpu = document.getElementById("s-webgpu");
  const operation = document.getElementById("operation-status");
  const scenarios = [
    [
      "preference-sync-failed",
      "storage unavailable",
      "Settings could not be synchronized: storage unavailable",
    ],
    [
      "storage-read-failed",
      "profile read rejected",
      "Settings could not be loaded: profile read rejected",
    ],
    [
      "preference-application-failed",
      "Preference application failed: mode rejected",
      "Settings could not be applied: mode rejected",
    ],
  ];

  for (const [error, reason, message] of scenarios) {
    controller.render(readyStatus({ failed: true, error, reason, gpuState: "ready" }));
    assert.equal(webgpu.textContent, "ready", error);
    assert.equal(webgpu.className, "v ok", error);
    assert.equal(operation.textContent, message, error);
    assert.equal(operation.textContent.split(reason).length - 1, error === "preference-application-failed" ? 0 : 1,
      `${error} must not repeat its raw reason`);
    assert.ok(commandControls(document).every((control) => control.disabled), error);
  }
});

test("WebGPU status distinguishes idle capability, initialization, readiness, recovery, and failure", () => {
  const { controller, document } = controllerHarness();
  const webgpu = document.getElementById("s-webgpu");

  for (const [gpuState, label, className] of [
    ["idle", "available", "v ok"],
    ["initializing", "starting…", "v"],
    ["ready", "ready", "v ok"],
    ["releasing", "releasing…", "v"],
    ["recovering", "recovering…", "v"],
    ["failed", "failed", "v no"],
    ["unavailable", "unavailable", "v no"],
  ]) {
    controller.render(readyStatus({ gpuState }));
    assert.equal(webgpu.textContent, label, gpuState);
    assert.equal(webgpu.className, className, gpuState);
  }

  controller.render(readyStatus({
    gpuState: undefined,
    runtime: { api: "available", adapter: "requesting", device: "requesting" },
  }));
  assert.equal(webgpu.textContent, "starting…", "nested runtime state is accepted");

  controller.render(readyStatus({
    gpuState: undefined,
    runtime: { api: "available", resources: { phase: "releasing", reason: "document-hidden" } },
  }));
  assert.equal(webgpu.textContent, "releasing…", "nested resource retirement is accepted");

  const legacy = readyStatus();
  delete legacy.gpuState;
  delete legacy.runtime;
  controller.render(legacy);
  assert.equal(webgpu.textContent, "available", "the legacy boolean remains a safe fallback");
});

test("runtime status explains requested-feature GPU recovery and bounded failure detail", () => {
  const { controller, document } = controllerHarness();

  controller.render(readyStatus({
    mode: "upscale",
    activeMode: "off",
    gpuState: "recovering",
    runtime: {
      api: "available",
      adapter: "ready",
      device: "lost",
      recovery: { phase: "running", attempt: 2, maxAttempts: 3 },
    },
  }));
  assert.equal(document.getElementById("s-webgpu").textContent, "recovering…");
  assert.equal(document.getElementById("runtime-status").textContent,
    "Recovering the WebGPU device…");

  const failed = readyStatus({
    mode: "upscale",
    activeMode: "off",
    gpuState: "failed",
    runtime: {
      api: "available",
      adapter: "failed",
      device: "failed",
      recovery: { phase: "exhausted", attempt: 3, maxAttempts: 3 },
      lastFailure: { code: "adapter-request-failed", detail: "GPU process reset" },
    },
  });
  controller.render(failed);
  assert.equal(document.getElementById("s-webgpu").textContent, "failed");
  assert.equal(document.getElementById("runtime-status").textContent,
    "WebGPU failed: GPU process reset");

  controller.render({ ...failed, mode: "off" });
  assert.equal(document.getElementById("runtime-status").textContent, "",
    "an old GPU failure is not presented as current when no GPU feature is requested");
});

test("settings persistence activity and errors take priority over renderer recovery", () => {
  const { controller, document } = controllerHarness();
  const recovering = {
    mode: "upscale",
    activeMode: "off",
    gpuState: "recovering",
    runtime: {
      api: "available",
      adapter: "ready",
      device: "lost",
      recovery: { phase: "running", attempt: 1, maxAttempts: 3 },
    },
  };

  controller.render(readyStatus({
    ...recovering,
    persistence: { state: "writing", pending: 2, error: null },
  }));
  assert.equal(document.getElementById("runtime-status").textContent, "Saving settings…");

  controller.render(readyStatus({
    ...recovering,
    persistence: { state: "error", pending: 0, error: "Storage quota exceeded" },
  }));
  assert.equal(document.getElementById("runtime-status").textContent,
    "Settings could not be saved: Storage quota exceeded");
});

test("settings runtime wording distinguishes synchronization, reads, validation, and application", () => {
  const { controller, document } = controllerHarness();
  const runtimeStatus = document.getElementById("runtime-status");
  const scenarios = [
    [
      { runtime: { phase: "syncing" }, persistence: { state: "writing", pendingWrites: 0 } },
      "Synchronizing settings…",
    ],
    [
      { persistence: { state: "reading", pendingWrites: 0 } },
      "Loading settings…",
    ],
    [
      {
        persistence: {
          state: "error", errorOperation: "validation", error: "Invalid stored setting: policy",
        },
      },
      "Stored settings are invalid: policy",
    ],
    [
      {
        persistence: {
          state: "error",
          errorOperation: "application",
          error: "Preference application failed: renderer rejected the saved mode",
        },
      },
      "Settings could not be applied: renderer rejected the saved mode",
    ],
    [
      {
        persistence: {
          state: "error", errorOperation: "loading", error: "storage backend rejected",
        },
      },
      "Settings could not be loaded: storage backend rejected",
    ],
    [
      {
        persistence: {
          state: "error", errorOperation: "syncing", error: "opaque storage rejection",
        },
      },
      "Settings could not be synchronized: opaque storage rejection",
    ],
    [
      {
        persistence: {
          state: "error", errorOperation: "writing", error: "opaque storage rejection",
        },
      },
      "Settings could not be saved: opaque storage rejection",
    ],
  ];

  for (const [overrides, expected] of scenarios) {
    controller.render(readyStatus(overrides));
    assert.equal(runtimeStatus.textContent, expected);
  }
});

test("image and direct neural runtime states are visible behind higher-priority failures", () => {
  const { controller, document } = controllerHarness();
  const runtimeStatus = document.getElementById("runtime-status");

  for (const [imagesRuntime, expected] of [
    [
      { requested: true, phase: "failed", lastFailure: { detail: "image pipeline rejected" } },
      "Image upscaling failed: image pipeline rejected",
    ],
    [{ requested: true, phase: "starting" }, "Image upscaling is starting…"],
    [{ requested: true, phase: "recovering" }, "Image upscaling is recovering…"],
    [{ requested: true, phase: "waiting" }, "Image upscaling is waiting to start."],
  ]) {
    controller.render(readyStatus({ images: true, imagesRuntime }));
    assert.equal(runtimeStatus.textContent, expected, imagesRuntime.phase);
  }

  controller.render(readyStatus({
    engine: "neural",
    neuralRuntime: {
      requested: true,
      phase: "failed",
      lastFailure: { code: "neural-init-failed", detail: "ONNX provider unavailable" },
    },
  }));
  assert.equal(runtimeStatus.textContent, "Neural upscaling failed: ONNX provider unavailable");

  controller.render(readyStatus({
    mode: "upscale",
    activeMode: "off",
    images: true,
    imagesRuntime: {
      requested: true,
      phase: "failed",
      lastFailure: { detail: "image pipeline rejected" },
    },
    renderer: { requestedMode: "upscale", activeMode: "off", phase: "failed" },
  }));
  assert.equal(runtimeStatus.textContent, "The requested renderer could not start.",
    "renderer failure remains higher priority than an auxiliary image failure");
});

test("terminal interpolation failure outranks an image runtime wait", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    images: true,
    imagesRuntime: { requested: true, phase: "waiting", lastFailure: null },
    interpolate: true,
    interpolationRuntime: {
      requested: true,
      phase: "failed",
      lastFailure: { stage: "inference", detail: "RIFE failed repeatedly" },
    },
  }));

  assert.equal(document.getElementById("runtime-status").textContent,
    "Interpolation stopped after repeated failures: RIFE failed repeatedly");
});

test("requested neural controls remain selected while model and runtime expose the effective fallback", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    mode: "upscale",
    activeMode: "upscale",
    engine: "neural",
    activeEngine: "fsrcnnx",
    gpuState: "ready",
    runtime: { api: "available", adapter: "ready", device: "ready" },
    model: "FSRCNNX_x2_16-0-4-1",
    neural: null,
    neuralModel: "local2x",
    neuralModels: [{ key: "local2x", label: "Local 2x", scale: 2 }],
    renderer: {
      requestedMode: "upscale",
      activeMode: "upscale",
      requestedEngine: "neural",
      effectiveEngine: "fsrcnnx",
      activeEngine: "fsrcnnx",
      phase: "active",
      fallback: {
        from: "neural",
        to: "fsrcnnx",
        code: "neural-init-failed",
        detail: "ONNX session creation failed",
      },
    },
  }));

  assert.equal(document.getElementById("engine").value, "neural");
  assert.equal(document.getElementById("mode-upscale").getAttribute("aria-pressed"), "true");
  assert.equal(document.getElementById("policy").disabled, false,
    "control applicability follows the requested engine");
  assert.match(document.getElementById("s-model").textContent, /FSRCNNX standard fallback/);
  assert.doesNotMatch(document.getElementById("s-model").textContent, /Local 2x/);
  assert.equal(document.getElementById("runtime-status").textContent,
    "Neural upscaling fell back to FSRCNNX standard: ONNX session creation failed");
  assert.equal(document.getElementById("neural-note").textContent,
    "Using FSRCNNX standard fallback · ONNX session creation failed");
});

test("failed commands roll focused controls back from authoritative status and keep the operation error visible", async () => {
  const authoritative = readyStatus({ sharpen: false });
  let statusCalls = 0;
  const transport = {
    async send(type) {
      if (type === "FSRCNNX_STATUS") {
        statusCalls++;
        return { ...authoritative };
      }
      assert.equal(type, "FSRCNNX_SETSHARPEN");
      return { ok: false, error: "invalid-input" };
    },
  };
  const { controller, document } = controllerHarness({ transport });
  controller.start();
  await flush();

  const sharpen = document.getElementById("sharpen");
  assert.equal(sharpen.checked, false);
  document.activeElement = sharpen;
  sharpen.checked = true;
  sharpen.emit("change");
  assert.ok(commandControls(document).every((control) => control.disabled), "commands lock controls while in flight");
  await flush(16);

  assert.ok(statusCalls >= 2, "a command result is followed by an authoritative status refresh");
  assert.equal(document.getElementById("operation-status").dataset.tone, "error");
  assert.equal(document.getElementById("operation-status").textContent, "That setting is not valid.");
  assert.equal(sharpen.checked, false, "failed optimistic state is rolled back even while the control remains focused");

  await controller.refresh();
  assert.equal(document.getElementById("operation-status").textContent, "That setting is not valid.");
  assert.equal(document.getElementById("operation-status").dataset.tone, "error");
  controller.stop();
});

test("automatic quality fallback is off by default and sends an explicit opt-in command", async () => {
  let enabled = false;
  const messages = [];
  const transport = {
    async send(type, payload = {}) {
      messages.push([type, payload]);
      if (type === "FSRCNNX_SETAUTOQUALITYFALLBACK") {
        enabled = payload.on;
        return { ok: true, autoQualityFallback: enabled };
      }
      return readyStatus({ autoQualityFallback: enabled });
    },
  };
  const { controller, document } = controllerHarness({ transport });
  controller.start();
  await flush();

  const toggle = document.getElementById("auto-quality-fallback");
  assert.equal(toggle.checked, false);
  toggle.checked = true;
  toggle.emit("change");
  await flush(16);

  assert.ok(messages.some(([type, payload]) =>
    type === "FSRCNNX_SETAUTOQUALITYFALLBACK" && payload.on === true));
  assert.equal(toggle.checked, true);
  controller.stop();
});

test("a non-ready page state replaces stale command feedback", async () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus());
  // A plain success now leaves the region empty — the control's own state is the
  // confirmation — so the stale text this test replaces has to come from a result
  // that genuinely needs reporting.
  await controller.runCommand("Applying test setting", async () => ({ ok: true, pending: true }));
  assert.match(
    document.getElementById("operation-status").textContent,
    /activation is pending/,
  );

  controller.render({ ok: false, error: "no-content-script" });
  assert.equal(document.getElementById("operation-status").dataset.tone, "error");
  assert.match(document.getElementById("operation-status").textContent, /Reload this page/);
  assert.ok(commandControls(document).every((control) => control.disabled));
});

test("stable refreshes do not rewrite live-region text", () => {
  const { controller, document } = controllerHarness();
  const runtime = document.getElementById("runtime-status");
  const protectedAlert = document.getElementById("drm-banner");
  let runtimeWrites = 0;
  let alertWrites = 0;
  Object.defineProperty(runtime, "textContent", {
    configurable: true,
    get() { return this._text || ""; },
    set(value) { runtimeWrites++; this._text = String(value); },
  });
  Object.defineProperty(protectedAlert, "textContent", {
    configurable: true,
    get() { return this._text || ""; },
    set(value) { alertWrites++; this._text = String(value); },
  });
  const status = readyStatus({
    mode: "upscale",
    activeMode: "off",
    protected: true,
    protectedReason: "drm",
  });

  controller.render(status);
  const firstRuntimeWrites = runtimeWrites;
  const firstAlertWrites = alertWrites;
  controller.render({ ...status });
  assert.equal(runtimeWrites, firstRuntimeWrites);
  assert.equal(alertWrites, firstAlertWrites);
});

test("color-boundary failures remain precise in commands, runtime status, and the source banner", () => {
  const cases = [
    ["color-hdr-unsupported", /HDR video/],
    ["color-wide-gamut-unsupported", /Wide-gamut video/],
    ["color-space-unsupported", /validated BT\.709 SDR boundary/],
    ["color-metadata-unavailable", /decoded color metadata/i],
  ];

  for (const [reason, expected] of cases) {
    const { controller, document } = controllerHarness();
    controller.render(readyStatus({
      mode: "upscale",
      activeMode: "off",
      protected: true,
      protectedReason: reason,
      renderer: {
        requestedMode: "upscale",
        activeMode: "off",
        phase: "blocked",
        blockedReason: reason,
      },
    }));

    assert.match(describeCommandFailure({ ok: false, reason }), expected);
    assert.match(document.getElementById("runtime-status").textContent, expected);
    assert.match(document.getElementById("drm-banner").textContent, expected);
  }
});

test("a color-blocked standalone interpolator reports its source reason", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    mode: "off",
    activeMode: "off",
    interpolate: true,
    protected: true,
    protectedReason: "color-hdr-unsupported",
    renderer: {
      requestedMode: "off",
      activeMode: "off",
      phase: "off",
      blockedReason: "color-hdr-unsupported",
    },
    interpolationRuntime: {
      requested: true,
      phase: "blocked",
      blockedReason: "color-hdr-unsupported",
      lastFailure: null,
    },
  }));

  assert.match(document.getElementById("runtime-status").textContent, /HDR video/);
  assert.match(document.getElementById("drm-banner").textContent, /HDR video/);
});

test("status refreshes do not overwrite a range while the user is adjusting it", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({ sharpen: true, sharpenStrength: 1 }));
  const range = document.getElementById("sharpen-str");
  const output = document.getElementById("sharpen-val");
  range.value = "1.7";
  output.textContent = "1.7";
  document.activeElement = range;

  controller.render(readyStatus({ sharpen: true, sharpenStrength: 0.4 }));
  assert.equal(range.value, "1.7");
  assert.equal(output.textContent, "1.7");

  document.activeElement = null;
  controller.render(readyStatus({ sharpen: true, sharpenStrength: 0.4 }));
  assert.equal(range.value, "0.4");
  assert.equal(output.textContent, "0.4");
});

test("mode accessibility reflects the requested mode while runtime text explains pending activation", () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus({
    mode: "upscale",
    activeMode: "off",
    hasVideo: true,
  }));

  const off = document.getElementById("mode-off");
  const upscale = document.getElementById("mode-upscale");
  assert.equal(off.getAttribute("aria-pressed"), "false");
  assert.equal(upscale.getAttribute("aria-pressed"), "true");
  assert.equal(upscale.dataset.active, "1");
  assert.equal(document.getElementById("runtime-status").textContent,
    "The requested mode is waiting for the renderer.");

  controller.render(readyStatus({ mode: "upscale", activeMode: "off", hasVideo: false }));
  assert.equal(upscale.getAttribute("aria-pressed"), "true");
  assert.equal(document.getElementById("runtime-status").textContent,
    "The requested mode will activate when a playable video appears.");
});

test("a content-level failure surfaces its specific reason rather than only the generic text", () => {
  // content.js always sets error "command-failed" and puts the cause in `reason`,
  // so checking `error` first matched the generic entry every time and discarded
  // the reason for every content-level failure.
  const described = describeCommandFailure({
    ok: false,
    error: "command-failed",
    reason: "policy must be one of: display, auto, force2",
  });
  assert.match(described, /could not apply that setting/);
  assert.match(described, /policy must be one of/);

  // Named codes still win outright.
  assert.equal(
    describeCommandFailure({ ok: false, error: "response-timeout" }),
    "The page did not answer the command in time.",
  );
  // And a bare generic failure still reads cleanly.
  assert.equal(
    describeCommandFailure({ ok: false, error: "command-failed" }),
    "The page could not apply that setting.",
  );
});

test("diagnostics serialize the status fields the four visible rows cannot show", () => {
  const report = formatDiagnostics({
    mode: "upscale",
    requestedMode: "upscale",
    engine: "neural",
    activeEngine: "fsrcnnx",
    policy: "display",
    chainDepth: 2,
    hasVideo: true,
    protected: false,
    frameCount: 4821,
    renderAttempts: 3,
    gpuState: "ready",
    renderer: {
      phase: "active",
      presentation: { source: { width: 640, height: 360 }, output: { width: 1280, height: 720 } },
      fallback: { from: "neural", to: "fsrcnnx", code: "neural-init-failed" },
    },
    colorSupport: { code: "color-supported" },
    persistence: { state: "ready" },
  }, { version: "0.50.0" });

  assert.match(report, /Flux Fidelity diagnostics \(v0\.50\.0\)/);
  // The engine the user picked and the one actually running can differ after a
  // fallback; a report that showed only one would hide the interesting case.
  assert.match(report, /engine: neural \(active fsrcnnx\)/);
  assert.match(report, /fallback: neural -> fsrcnnx \(neural-init-failed\)/);
  assert.match(report, /presentation: 640×360 → 1280×720 \(2×\)/);
  assert.equal(report.includes("\n"), true, "the report must be multi-line for pasting");

  assert.match(formatDiagnostics(null), /status unavailable/);
});

test("presentation formatting reports source, output, and scale, or nothing at all", () => {
  assert.equal(
    formatPresentation({ renderer: { presentation: {
      source: { width: 1920, height: 1080 }, output: { width: 3840, height: 2160 },
    } } }),
    "1920×1080 → 3840×2160 (2×)",
  );
  // Partial or absent dimensions must not render a misleading half-answer.
  for (const presentation of [
    undefined,
    { source: { width: 640, height: 360 } },
    { source: { width: 0, height: 0 }, output: { width: 1280, height: 720 } },
  ]) {
    assert.equal(formatPresentation({ renderer: { presentation } }), "—");
  }
});

test("the popup viewport is capped so its scroll container can actually overflow", async () => {
  const markup = await readFile(new URL("../popup.html", import.meta.url), "utf8");
  // Extracts a rule whose selector is exactly `selector` on its own line, so a
  // selector list such as `body, button {` cannot be mistaken for it.
  const rule = (selector) => {
    const lines = markup.split("\n");
    const start = lines.findIndex((line) => line.trim() === `${selector} {`);
    if (start < 0) return "";
    const body = [];
    for (let index = start + 1; index < lines.length; index++) {
      if (lines[index].trim() === "}") break;
      body.push(lines[index]);
    }
    return body.join("\n");
  };

  // A browser-action popup is sized from the document. `body` is the scroll
  // container, so if `html` is left unconstrained the document can grow past the
  // popup's height cap: the overflow lands on `html`, which nothing scrolls, while
  // `body` still fits inside `html` and never draws a scrollbar. That is how
  // expanding the Advanced <details> made those settings unreachable.
  //
  // This is asserted against the stylesheet because there is no behavioural hook:
  // the Node suite compiles no CSS, and the browser harness opens the popup as a
  // tab with a fixed viewport rather than a content-sized popup window, so it
  // cannot reproduce the sizing path either. The invariant is the contract.
  const body = rule("body");
  const html = rule("html");
  assert.match(body, /overflow-y:\s*auto/, "body is expected to be the scroll container");
  const bodyCap = /max-height:\s*(\d+)px/.exec(body);
  assert.ok(bodyCap, "body must cap its height for scrolling to engage");
  const htmlCap = /max-height:\s*(\d+)px/.exec(html);
  assert.ok(
    htmlCap,
    "html must also be capped, or the document outgrows the popup and nothing scrolls",
  );
  assert.equal(
    htmlCap[1],
    bodyCap[1],
    "html and body caps must agree so the window never exceeds the scroll container",
  );
  assert.match(html, /overflow:\s*hidden/, "html must not become a second scroll container");
});

test("the encode sparkline scales to the budget and describes itself in text", () => {
  const samples = [4, 6, 5, 20, 7];
  const spark = encodeSparkline(samples, 16.7);
  assert.match(spark.line, /^M0\.0 /, "the path must start at the first sample");
  assert.equal(spark.line.split("L").length, samples.length, "one segment per later sample");
  assert.match(spark.budget, /^M0 [\d.]+ L240 [\d.]+$/, "the budget is a flat reference line");
  // Scaling to max(peak, budget) keeps the budget line on-screen even when the
  // series never approaches it, and keeps a spiky series from clipping.
  assert.ok(spark.scaleMs >= 20, "scale must cover the peak");
  const calm = encodeSparkline([2, 3, 2], 16.7);
  assert.ok(calm.scaleMs >= 16.7, "scale must cover the budget even for a calm series");

  // Too short to plot must not produce a malformed path.
  assert.deepEqual(encodeSparkline([5], 16.7).line, "");
  assert.deepEqual(encodeSparkline(null, 16.7).line, "");
  // Non-finite samples cannot poison the geometry.
  assert.equal(encodeSparkline([1, Number.NaN, 3, Infinity], 16.7).line.includes("NaN"), false);
});

test("the encode summary names the number that actually matters", () => {
  const described = describeEncodeSeries([4, 6, 20, 22], 16.7);
  assert.match(described, /4 frames/);
  assert.match(described, /peak 22\.0 ms/);
  // Frames over the budget are when frames start being missed, so the count is
  // stated rather than left for the reader to infer from the chart.
  assert.match(described, /2 over budget/);
  assert.match(describeEncodeSeries([4, 5], 16.7), /none over budget/);
  assert.match(describeEncodeSeries([], 16.7), /No frames measured/);
  // Without a known budget the summary must not invent one.
  assert.equal(/budget/.test(describeEncodeSeries([4, 5], 0)), false);
});

test("the GPU summary distinguishes unsupported from not-yet-measured", () => {
  // The distinction matters: a dash or a zero would read as "the GPU costs
  // nothing", when in fact nothing was measured. timestamp-query is absent
  // under SwiftShader and behind a flag on several platforms.
  assert.match(describeGpuSeries(undefined, 16.7), /Not available/);
  assert.match(describeGpuSeries({ supported: false }, 16.7), /Not available/);
  assert.match(
    describeGpuSeries({ supported: true, samples: 0, avgMs: null, maxMs: null }, 16.7),
    /Measuring/,
  );
  const described = describeGpuSeries(
    { supported: true, samples: 8, avgMs: 3.5, maxMs: 9.25 }, 16.7,
  );
  assert.match(described, /8 samples/);
  assert.match(described, /mean 3\.5 ms/);
  assert.match(described, /peak 9\.3 ms/);
  assert.match(described, /budget 16\.7 ms/);
  // Without a known frame interval the summary must not invent a budget.
  assert.equal(
    /budget/.test(describeGpuSeries({ supported: true, samples: 8, avgMs: 3.5, maxMs: 9.25 }, 0)),
    false,
  );
});

test("tabs expose one selected panel and keep the rest out of the tab order", () => {
  const { controller, document } = controllerHarness();
  // start() performs the binding; the controller exposes no separate bind.
  controller.start();

  const tabs = document.querySelectorAll('[role="tab"]');
  assert.equal(tabs.length, 4);
  const selected = tabs.filter((tab) => tab.getAttribute("aria-selected") === "true");
  assert.equal(selected.length, 1, "exactly one tab may be selected");
  assert.equal(selected[0].id, "tab-video", "Video is the default panel");
  assert.equal(document.getElementById("panel-video").hidden, false);
  assert.equal(document.getElementById("panel-interpolation").hidden, true);
  assert.equal(document.getElementById("panel-advanced").hidden, true);
  assert.equal(document.getElementById("panel-performance").hidden, true);

  // Only the selected tab is reachable by Tab; the others are arrow-key targets.
  // The previous disclosure had no such semantics, which left its controls
  // unreachable by heading navigation.
  assert.deepEqual(tabs.map((tab) => tab.tabIndex), [0, -1, -1, -1]);

  document.getElementById("tab-performance").emit("click");
  assert.equal(document.getElementById("panel-performance").hidden, false);
  assert.equal(document.getElementById("panel-video").hidden, true);
  assert.equal(document.getElementById("tab-performance").getAttribute("aria-selected"), "true");
  assert.deepEqual(tabs.map((tab) => tab.tabIndex), [-1, -1, -1, 0]);

  // Arrow keys wrap, per the ARIA tabs pattern.
  document.getElementById("tab-performance").emit("keydown", { key: "ArrowRight" });
  assert.equal(document.getElementById("tab-video").getAttribute("aria-selected"), "true");
  document.getElementById("tab-video").emit("keydown", { key: "ArrowLeft" });
  assert.equal(document.getElementById("tab-performance").getAttribute("aria-selected"), "true");
});

test("tabs stay usable while the page is disconnected or a command is in flight", async () => {
  const { controller, document } = controllerHarness({
    transport: { send: async () => ({ ok: false, error: "no-content-script", reason: "not connected" }) },
  });
  controller.start();
  await Promise.resolve();

  // Navigation is not a setting. Sweeping the tabs into the blanket disable made
  // them dead exactly when a user most wants the Performance panel, and a disabled
  // button fires no click at all.
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    assert.equal(tab.disabled, false, `${tab.id} must remain enabled`);
  }
  document.getElementById("tab-performance").emit("click");
  assert.equal(document.getElementById("panel-performance").hidden, false);
});

test("a routine success reports nothing while a refusal still speaks up", async () => {
  const { controller, document } = controllerHarness();
  controller.render(readyStatus());
  const region = document.getElementById("operation-status");

  // Every routine change used to post "Setting applied." into an alert region above
  // the tabs, which made ordinary use look like a stream of events. The control's
  // own state is already the confirmation.
  await controller.runCommand("Applying test setting", async () => ({ ok: true }));
  assert.equal(region.textContent, "");

  // A refusal is not routine and must still be surfaced.
  await controller.runCommand("Applying test setting", async () => ({
    ok: true, pending: true, pendingKind: "runtime-rejected", reason: "runtime rejected resolution",
  }));
  assert.match(region.textContent, /did not accept it/);
  assert.match(region.textContent, /runtime rejected resolution/);
});

test("tab dots report off, pending, and on with matching text", () => {
  const { controller, document } = controllerHarness();
  const state = (id) => document.getElementById(`${id}-dot`).dataset.state;
  const label = (id) => document.getElementById(`${id}-state`).textContent;

  // Neither feature enabled: both read off. This is the case that was broken —
  // the dot toggled the hidden attribute, which `.tab-dot { display: inline-block }`
  // overrode, so a green dot showed permanently no matter the state.
  controller.render(readyStatus({
    mode: "off", activeMode: "off", interpolate: false,
    interpolationRuntime: { requested: false, phase: "off" },
  }));
  assert.equal(state("tab-video"), "off");
  assert.equal(state("tab-interpolation"), "off");
  assert.equal(label("tab-video"), " (off)");

  // Enabled but not running: no video yet.
  controller.render(readyStatus({
    mode: "upscale", activeMode: "off", hasVideo: false, interpolate: true,
    interpolationRuntime: { requested: true, phase: "waiting" },
  }));
  assert.equal(state("tab-video"), "pending");
  assert.equal(state("tab-interpolation"), "pending");
  assert.equal(label("tab-video"), " (on, not running)");

  // Enabled but suspended mid-playback is also pending, not on.
  controller.render(readyStatus({
    mode: "upscale", activeMode: "off", videoSuspended: true, interpolate: true,
    interpolationRuntime: { requested: true, phase: "suspended" },
  }));
  assert.equal(state("tab-video"), "pending");
  assert.equal(state("tab-interpolation"), "pending");

  // Running.
  controller.render(readyStatus({
    mode: "upscale", activeMode: "upscale", interpolate: true,
    interpolationRuntime: { requested: true, phase: "active" },
  }));
  assert.equal(state("tab-video"), "on");
  assert.equal(state("tab-interpolation"), "on");
  assert.equal(label("tab-video"), " (on)");
  assert.equal(label("tab-interpolation"), " (on)");
});

test("the dot is never hidden, so no display rule can strand it on one colour", async () => {
  const markup = await readFile(new URL("../popup.html", import.meta.url), "utf8");
  // `[hidden]` is a UA rule, so any author `display` declaration on .tab-dot beats
  // it. State must therefore be carried by an attribute the stylesheet keys on,
  // never by hiding the element.
  assert.equal(/id="tab-\w+-dot"[^>]*\shidden/.test(markup), false);
  for (const dotState of ["on", "pending", "off"]) {
    assert.ok(
      markup.includes(`.tab-dot[data-state="${dotState}"]`),
      `the stylesheet must give ${dotState} its own colour`,
    );
  }
});

test("interpolation mode buttons mirror the upscaler's mode row", async () => {
  const sent = [];
  const { controller, document } = controllerHarness({
    transport: { send: async (type, payload) => { sent.push([type, payload]); return { ok: true }; } },
  });
  controller.start();
  controller.render(readyStatus({ interpolate: false }));

  const [off, on] = document.interpModeButtons;
  assert.equal(off.getAttribute("aria-pressed"), "true");
  assert.equal(on.getAttribute("aria-pressed"), "false");

  on.emit("click");
  await flush();
  assert.ok(
    sent.some(([type, payload]) => type === "FSRCNNX_SETINTERPOLATE" && payload?.on === true),
    "the On button must request interpolation",
  );

  controller.render(readyStatus({ interpolate: true }));
  assert.equal(on.getAttribute("aria-pressed"), "true");
  assert.equal(off.getAttribute("aria-pressed"), "false");

  off.emit("click");
  await flush();
  assert.ok(
    sent.some(([type, payload]) => type === "FSRCNNX_SETINTERPOLATE" && payload?.on === false),
    "the Off button must disable interpolation",
  );
});

test("the avoided-work summary distinguishes fast from skipped", () => {
  // A user seeing low GPU time must be able to tell "the model is fast" from
  // "most frames were never computed", and a user suspecting a stale image must
  // be able to see whether skipping is happening at all.
  assert.match(describeAvoidedWork({}), /Nothing skipped yet/);
  assert.match(
    describeAvoidedWork({ duplicateFrames: { supported: false } }),
    /repeat detection unavailable/,
  );
  const busy = describeAvoidedWork({
    duplicateFrames: { supported: true, skipped: 412, duplicateRate: 0.38, probing: true },
    offscreenFrames: { skipped: 96, inViewport: false },
  });
  assert.match(busy, /412 repeated frames reused/);
  assert.match(busy, /38% of sampled frames/);
  assert.match(busy, /96 skipped while scrolled out of view/);
  // Backing off is stated, so a zero count reads as "not applicable" rather than
  // "broken".
  assert.match(
    describeAvoidedWork({ duplicateFrames: { supported: true, skipped: 0, duplicateRate: 0, probing: false } }),
    /detection paused/,
  );
});

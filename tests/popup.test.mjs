import assert from "node:assert/strict";
import test from "node:test";

import {
  PopupTransport,
  StatusCoordinator,
  createPopupController,
  describeCommandFailure,
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
    if (selector === ".modes button") return this.modeButtons;
    if (selector === "button, input, select") return this.controls;
    return [];
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
    "s-webgpu", "s-video", "s-model", "s-frames", "runtime-status", "drm-banner",
    "operation-status", "neuralrow", "neural-note", "sharpen-row", "sharpen-val",
    "multi-count", "image-count", "interp-res-row",
    "interp-target-hz", "interp-avoff-val", "interp-stats",
  ]) document.add(id);

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
  document.addMode("passthrough", "Passthrough");
  document.addMode("upscale", "Upscale");
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
  assert.ok(document.controls.every((control) => control.disabled));

  controller.render({ failed: true, error: "startup-failed", reason: "GPU adapter rejected" });
  assert.equal(document.getElementById("s-webgpu").textContent, "failed");
  assert.match(document.getElementById("operation-status").textContent, /could not start/i);
  assert.match(document.getElementById("operation-status").textContent, /GPU adapter rejected/);
  assert.ok(document.controls.every((control) => control.disabled));

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
  assert.equal(document.getElementById("policy").disabled, true);
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
    assert.ok(document.controls.every((control) => control.disabled), error);
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
  assert.equal(document.getElementById("policy").disabled, true,
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
  assert.ok(document.controls.every((control) => control.disabled), "commands lock controls while in flight");
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
  await controller.runCommand("Applying test setting", async () => ({ ok: true }));
  assert.equal(document.getElementById("operation-status").textContent, "Setting applied.");

  controller.render({ ok: false, error: "no-content-script" });
  assert.equal(document.getElementById("operation-status").dataset.tone, "error");
  assert.match(document.getElementById("operation-status").textContent, /Reload this page/);
  assert.ok(document.controls.every((control) => control.disabled));
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
  const passthrough = document.getElementById("mode-passthrough");
  const upscale = document.getElementById("mode-upscale");
  assert.equal(off.getAttribute("aria-pressed"), "false");
  assert.equal(passthrough.getAttribute("aria-pressed"), "false");
  assert.equal(upscale.getAttribute("aria-pressed"), "true");
  assert.equal(upscale.dataset.active, "1");
  assert.equal(document.getElementById("runtime-status").textContent,
    "The requested mode is waiting for the renderer.");

  controller.render(readyStatus({ mode: "upscale", activeMode: "off", hasVideo: false }));
  assert.equal(upscale.getAttribute("aria-pressed"), "true");
  assert.equal(document.getElementById("runtime-status").textContent,
    "The requested mode will activate when a playable video appears.");
});

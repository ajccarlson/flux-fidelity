import assert from "node:assert/strict";
import test from "node:test";

import {
  PopupTransport,
  StatusCoordinator,
  createPopupController,
  isSupportedPageUrl,
  reconcileSelectOptions,
} from "../popup.js";

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
    "deband-row", "deband-val", "multi-count", "image-count", "interp-res-row",
    "interp-target-hz", "interp-avoff-val", "interp-stats",
  ]) document.add(id);

  for (const [id, value] of [
    ["engine", "fsrcnnx"],
    ["artvariant", "ArtCNN_C4F32"],
    ["neural-model", ""],
    ["policy", "display"],
    ["interp-model", "rife_v4.26_fp16"],
    ["interp-res", "auto"],
    ["interp-target", "auto"],
  ]) document.add(id, "select", { value });

  for (const [id, value] of [
    ["ssimds", ""],
    ["sharpen", ""],
    ["sharpen-str", "1"],
    ["deband", ""],
    ["deband-str", "1"],
    ["hover-reveal", ""],
    ["all-videos", ""],
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
  document.getElementById("interp-autofallback").checked = true;

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
  return {
    activeMode: "off",
    allVideos: false,
    artVariant: "ArtCNN_C4F32",
    deband: false,
    debandStrength: 1,
    engine: "fsrcnnx",
    frameCount: 12,
    hasVideo: true,
    hoverReveal: false,
    imageCount: 0,
    images: false,
    interpAutoFallback: true,
    interpAvOffsetMs: 0,
    interpInvert: true,
    interpLadder: false,
    interpModel: "rife_v4.26_fp16",
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
    value: "span2x_smoke",
    label: "<img src=x onerror=alert(1)> (2×)",
  }], "span2x_smoke");

  assert.equal(select.children.length, 1);
  assert.notStrictEqual(select.children[0], placeholder);
  assert.equal(select.children[0].value, "span2x_smoke");
  assert.equal(select.children[0].textContent, "<img src=x onerror=alert(1)> (2×)");
  assert.equal(select.value, "span2x_smoke");
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
  assert.equal(document.getElementById("policy").disabled, false);
  assert.equal(document.getElementById("artvariant").disabled, false);
  assert.equal(document.getElementById("neural-model").disabled, true);
  assert.equal(document.getElementById("interp-model").disabled, false);
  assert.equal(document.getElementById("interp-avoff").disabled, false);

  controller.render(readyStatus({
    engine: "neural",
    neuralModel: "span2x_smoke",
    neuralModels: [{ key: "span2x_smoke", label: "SPAN smoke", scale: 2 }],
  }));
  assert.equal(document.getElementById("neural-model").disabled, false);
  assert.equal(document.getElementById("policy").disabled, true);
  assert.equal(document.getElementById("all-videos").disabled, true);
  assert.equal(document.getElementById("artvariant").disabled, true);
  assert.equal(document.getElementById("interp-model").disabled, true);
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

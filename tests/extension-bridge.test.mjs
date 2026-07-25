import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const contentUrl = new URL("../src/content.js", import.meta.url);

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

class FakeEventTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }
  emit(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) listener(event);
  }
}

async function flush(turns = 8) {
  for (let i = 0; i < turns; i++) await new Promise((resolve) => setImmediate(resolve));
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

async function loadBridge(importModule, { prerendering = false, visibilityState = "visible" } = {}) {
  const original = await readFile(contentUrl, "utf8");
  const source = original.replace(
    "api = await import(url);",
    "api = await globalThis.__importModule(url);",
  );
  assert.notEqual(source, original, "content module-loader injection must match production source");

  const window = new FakeEventTarget();
  const document = new FakeEventTarget();
  document.prerendering = prerendering;
  document.visibilityState = visibilityState;
  const sentMessages = [];
  let messageListener = null;
  const logs = [];
  const context = vm.createContext({
    __importModule: importModule,
    chrome: {
      runtime: {
        getURL: (path) => `chrome-extension://test/${path}`,
        onMessage: { addListener(listener) { messageListener = listener; } },
        sendMessage(message) {
          sentMessages.push(message);
          return Promise.resolve();
        },
      },
    },
    console: {
      log: (...args) => logs.push(["log", ...args]),
      error: (...args) => logs.push(["error", ...args]),
    },
    document,
    navigator: { gpu: {} },
    Promise,
    setTimeout,
    clearTimeout,
    window,
  });
  new vm.Script(source, { filename: "content.js" }).runInContext(context);
  assert.equal(typeof messageListener, "function");

  function message(msg) {
    const responses = [];
    const claimed = messageListener(msg, {}, (response) => responses.push(response));
    return { claimed, responses };
  }

  return { context, document, logs, message, sentMessages, window };
}

function completeApi(overrides = {}) {
  return {
    restoreSitePrefs: async () => ({ ok: true, restored: true }),
    syncSitePrefs: async () => ({ ok: true }),
    flushPreferenceWrites: async () => ({ ok: true }),
    getStatus: () => ({ mode: "upscale", hasVideo: true, webgpu: true, frameCount: 7 }),
    resumeDocument: async () => ({ ok: true }),
    suspendDocument: async () => ({ ok: true }),
    setMode: async (mode) => ({ ok: true, mode }),
    setEngine: () => ({ ok: true }),
    setNeuralModel: () => ({ ok: true }),
    setArtVariant: () => ({ ok: true }),
    setInterpolate: () => ({ ok: true }),
    setInterpolateRes: () => ({ ok: true }),
    setInterpolateAvOffset: () => ({ ok: true }),
    setInterpolateModel: () => ({ ok: true }),
    setInterpolateTargetFps: () => ({ ok: true }),
    setInterpolateLadder: () => ({ ok: true }),
    setInterpolateAutoFallback: () => ({ ok: true }),
    setInterpolateInvert: () => ({ ok: true }),
    setInterpolateDiag: () => ({ ok: true }),
    setImages: () => ({ ok: true }),
    setHoverReveal: () => ({ ok: true }),
    setAllVideos: () => ({ ok: true }),
    setIdlePowerSaving: () => ({ ok: true }),
    setSharpen: () => ({ ok: true }),
    setSharpenStrength: () => ({ ok: true }),
    setSSimDS: () => ({ ok: true }),
    setPolicy: () => ({ ok: true }),
    ...overrides,
  };
}

const booleanCommand = (type, method) => ({
  type,
  method,
  field: "on",
  valid: [true, false],
  invalid: [0, 1, "true", null, undefined],
});

const COMMAND_CASES = [
  {
    type: "FSRCNNX_SETMODE", method: "setMode", field: "mode",
    valid: ["off", "passthrough", "upscale"], invalid: ["", "UPscale", null, 0],
  },
  {
    type: "FSRCNNX_SETENGINE", method: "setEngine", field: "engine",
    valid: ["fsrcnnx", "fsrcnnx-hi", "artcnn", "neural"],
    invalid: ["", "FSRCNNX", "unknown", null],
  },
  {
    type: "FSRCNNX_SETNEURALMODEL", method: "setNeuralModel", field: "model",
    valid: ["local2x", "future4x", "vendor.model-v2"],
    invalid: ["", "../local2x", "key/with/slash", "contains space", null],
  },
  {
    type: "FSRCNNX_SETARTVARIANT", method: "setArtVariant", field: "variant",
    valid: ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"],
    invalid: ["", "ArtCNN_C4F32_UNKNOWN", null],
  },
  booleanCommand("FSRCNNX_SETINTERPOLATE", "setInterpolate"),
  {
    type: "FSRCNNX_SETINTERPRES", method: "setInterpolateRes", field: "mode",
    valid: ["auto", "full", "half", "quarter"], invalid: ["", "eighth", null, 1],
  },
  {
    type: "FSRCNNX_SETINTERPAVOFFSET", method: "setInterpolateAvOffset", field: "ms",
    valid: [-100, 0, 300], invalid: [-100.01, 300.01, "0", NaN, Infinity, null],
  },
  {
    type: "FSRCNNX_SETINTERPMODEL", method: "setInterpolateModel", field: "key",
    valid: ["rife_v4.26_fp16", "rife_v4.26", "blend"],
    invalid: ["", "rife_orig", "rife_v4.25", "../rife_orig", null],
  },
  {
    type: "FSRCNNX_SETINTERPTARGETFPS", method: "setInterpolateTargetFps", field: "value",
    valid: ["auto", 24, 60, 480], invalid: [23.99, 480.01, "60", "AUTO", NaN, Infinity, null],
  },
  booleanCommand("FSRCNNX_SETLADDER", "setInterpolateLadder"),
  booleanCommand("FSRCNNX_SETAUTOFALLBACK", "setInterpolateAutoFallback"),
  booleanCommand("FSRCNNX_SETINVERT", "setInterpolateInvert"),
  booleanCommand("FSRCNNX_SETINTERPDIAG", "setInterpolateDiag"),
  booleanCommand("FSRCNNX_SETIMAGES", "setImages"),
  booleanCommand("FSRCNNX_SETHOVERREVEAL", "setHoverReveal"),
  booleanCommand("FSRCNNX_SETALLVIDEOS", "setAllVideos"),
  booleanCommand("FSRCNNX_SETIDLEPOWERSAVING", "setIdlePowerSaving"),
  booleanCommand("FSRCNNX_SETSHARPEN", "setSharpen"),
  {
    type: "FSRCNNX_SETSHARPENSTR", method: "setSharpenStrength", field: "strength",
    valid: [0.1, 1, 2], invalid: [0.09, 2.01, "1", NaN, Infinity, null],
  },
  booleanCommand("FSRCNNX_SETSSIMDS", "setSSimDS"),
  {
    type: "FSRCNNX_SETPOLICY", method: "setPolicy", field: "policy",
    valid: ["display", "auto", "force2", "force3", "force4", "force8"],
    invalid: ["", "force1", "FORCE2", null],
  },
];

test("status remains loading through restore and explicit restore is single-flight", async () => {
  const imported = deferred();
  const restored = deferred();
  let restoreCalls = 0;
  const bridge = await loadBridge(() => imported.promise);

  assert.deepEqual(plain(bridge.sentMessages), [{
    type: "FSRCNNX_DOCUMENT", state: "active", generation: 1,
  }]);
  const importingStatus = bridge.message({ type: "FSRCNNX_STATUS" });
  assert.equal(importingStatus.claimed, true);
  await flush();
  assert.equal(importingStatus.responses.length, 1);
  assert.equal(importingStatus.responses[0].loading, true);
  assert.equal(importingStatus.responses[0].failed, false);
  assert.equal(importingStatus.responses[0].statusVersion, 1);
  assert.equal(importingStatus.responses[0].gpuState, "idle");
  assert.equal(importingStatus.responses[0].runtime.phase, "loading");
  assert.equal(importingStatus.responses[0].renderer.phase, "loading");

  imported.resolve(completeApi({
    restoreSitePrefs: () => { restoreCalls++; return restored.promise; },
  }));
  await flush();
  assert.equal(restoreCalls, 1);

  const restoringStatus = bridge.message({ type: "FSRCNNX_STATUS" });
  const explicitRestore = bridge.message({ type: "FSRCNNX_RESTORE" });
  await flush();
  assert.equal(restoringStatus.responses[0].loading, true);
  assert.equal(explicitRestore.responses.length, 0, "restore response waits for the shared restore");
  assert.equal(restoreCalls, 1);

  restored.resolve({ ok: true, restored: true, marker: 9 });
  await flush();
  assert.deepEqual(explicitRestore.responses, [{ ok: true, restored: true, marker: 9 }]);
  assert.equal(restoreCalls, 1);

  const repeatedRestore = bridge.message({ type: "FSRCNNX_RESTORE" });
  const readyStatus = bridge.message({ type: "FSRCNNX_STATUS" });
  await flush();
  assert.deepEqual(repeatedRestore.responses, [{ ok: true, restored: true, marker: 9 }]);
  assert.equal(restoreCalls, 1, "restore is never run a second time");
  assert.equal(readyStatus.responses.length, 1);
  assert.equal(readyStatus.responses[0].loading, false);
  assert.equal(readyStatus.responses[0].failed, false);
  assert.equal(readyStatus.responses[0].mode, "upscale");
});

test("the allowlisted dispatcher responds exactly once to success, throws, and rejections", async () => {
  const api = completeApi({
    setMode: async (mode) => ({ ok: true, mode }),
    setEngine: () => { throw new Error("engine exploded"); },
    setImages: async () => { throw new Error("image rejected"); },
  });
  const bridge = await loadBridge(async () => api);
  await flush();

  const success = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  const thrown = bridge.message({ type: "FSRCNNX_SETENGINE", engine: "fsrcnnx" });
  const rejected = bridge.message({ type: "FSRCNNX_SETIMAGES", on: true });
  const unknown = bridge.message({ type: "FSRCNNX_CALL_ANYTHING", method: "setMode" });
  await flush();

  assert.equal(success.claimed, true);
  assert.deepEqual(success.responses, [{ ok: true, mode: "upscale" }]);
  assert.equal(thrown.responses.length, 1);
  assert.deepEqual(plain(thrown.responses[0]), {
    ok: false, error: "command-failed", reason: "engine exploded",
  });
  assert.equal(rejected.responses.length, 1);
  assert.deepEqual(plain(rejected.responses[0]), {
    ok: false, error: "command-failed", reason: "image rejected",
  });
  assert.equal(unknown.claimed, false);
  assert.deepEqual(unknown.responses, []);
});

test("the command contract matrix covers the complete production allowlist", async () => {
  const source = await readFile(contentUrl, "utf8");
  const start = source.indexOf("const COMMANDS = Object.freeze({");
  const end = source.indexOf("\n});\n\nfunction normalizeCommandResponse", start);
  assert.ok(start >= 0 && end > start, "production command table markers must remain discoverable");
  const commandSource = source.slice(start, end);
  const production = [...commandSource.matchAll(/^  (FSRCNNX_[A-Z0-9_]+):/gm)]
    .map((match) => match[1])
    .sort();
  const tested = ["FSRCNNX_RESTORE", ...COMMAND_CASES.map(({ type }) => type)].sort();
  assert.deepEqual(production, tested);
});

test("status accepts no payload fields", async () => {
  const bridge = await loadBridge(async () => completeApi());
  await flush();

  const valid = bridge.message({ type: "FSRCNNX_STATUS" });
  const invalid = bridge.message({ type: "FSRCNNX_STATUS", force: true });
  await flush();

  assert.equal(valid.responses[0].loading, false);
  assert.deepEqual(plain(invalid.responses), [{
    ok: false,
    error: "invalid-input",
    reason: "Unexpected field: force",
    field: "force",
  }]);
});

test("every allowlisted command accepts its complete supported value set", async () => {
  const calls = new Map(COMMAND_CASES.map(({ method }) => [method, []]));
  const overrides = Object.fromEntries(COMMAND_CASES.map(({ method }) => [
    method,
    (value) => {
      calls.get(method).push(value);
      return { ok: true, value };
    },
  ]));
  const bridge = await loadBridge(async () => completeApi(overrides));
  await flush();

  for (const command of COMMAND_CASES) {
    for (const value of command.valid) {
      const request = bridge.message({ type: command.type, [command.field]: value });
      assert.equal(request.claimed, true, command.type);
      await flush();
      assert.deepEqual(plain(request.responses), [{ ok: true, value }],
        `${command.type} should accept ${String(value)}`);
    }
    assert.deepEqual(calls.get(command.method), command.valid, command.type);
  }
});

test("every allowlisted command rejects missing, malformed, and extra payload fields before main", async () => {
  let calls = 0;
  const overrides = Object.fromEntries(COMMAND_CASES.map(({ method }) => [
    method,
    () => { calls++; return { ok: true }; },
  ]));
  const bridge = await loadBridge(async () => completeApi(overrides));
  await flush();
  const attempts = [];

  for (const command of COMMAND_CASES) {
    attempts.push({
      command,
      expectedField: command.field,
      request: bridge.message({ type: command.type }),
    });
    for (const value of command.invalid) {
      attempts.push({
        command,
        expectedField: command.field,
        request: bridge.message({ type: command.type, [command.field]: value }),
      });
    }
    attempts.push({
      command,
      expectedField: "unexpected",
      request: bridge.message({
        type: command.type,
        [command.field]: command.valid[0],
        unexpected: true,
      }),
    });
  }

  await flush(12);
  assert.equal(calls, 0, "invalid payloads must never reach a main-world setter");
  for (const { command, expectedField, request } of attempts) {
    assert.equal(request.claimed, true, command.type);
    assert.equal(request.responses.length, 1, command.type);
    const response = plain(request.responses[0]);
    assert.equal(response.ok, false, command.type);
    assert.equal(response.error, "invalid-input", command.type);
    assert.equal(response.field, expectedField, command.type);
    assert.equal(typeof response.reason, "string", command.type);
    assert.ok(response.reason.length > 0, command.type);
    assert.deepEqual(Object.keys(response).sort(), ["error", "field", "ok", "reason"]);
  }
});

test("restore accepts no payload and malformed commands fail without waiting for startup", async () => {
  const imported = deferred();
  const bridge = await loadBridge(() => imported.promise);

  const malformed = bridge.message({ type: "FSRCNNX_SETMODE", mode: "invalid" });
  const restoreWithPayload = bridge.message({ type: "FSRCNNX_RESTORE", force: true });
  await flush();
  assert.deepEqual(plain(malformed.responses), [{
    ok: false,
    error: "invalid-input",
    reason: "mode must be one of: off, passthrough, upscale",
    field: "mode",
  }]);
  assert.deepEqual(plain(restoreWithPayload.responses), [{
    ok: false,
    error: "invalid-input",
    reason: "Unexpected field: force",
    field: "force",
  }]);

  imported.resolve(completeApi());
  await flush();
  const restore = bridge.message({ type: "FSRCNNX_RESTORE" });
  await flush();
  assert.deepEqual(plain(restore.responses), [{ ok: true, restored: true }]);
});

test("malformed main-world command results become stable invalid-response envelopes", async () => {
  const malformed = [undefined, null, 7, "ok", [], {}, { ok: "yes" }];
  let index = 0;
  const bridge = await loadBridge(async () => completeApi({
    setMode: () => malformed[index++],
    setEngine: () => ({ ok: false, error: "engine-disabled", reason: "not available" }),
    setImages: () => ({ ok: true, images: true }),
  }));
  await flush();

  for (let i = 0; i < malformed.length; i++) {
    const request = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
    await flush();
    assert.deepEqual(plain(request.responses), [{
      ok: false,
      error: "invalid-response",
      reason: "Command response must be an object with a boolean ok field",
    }]);
  }

  const failure = bridge.message({ type: "FSRCNNX_SETENGINE", engine: "neural" });
  const success = bridge.message({ type: "FSRCNNX_SETIMAGES", on: true });
  await flush();
  assert.deepEqual(plain(failure.responses), [{
    ok: false, error: "engine-disabled", reason: "not available",
  }]);
  assert.deepEqual(plain(success.responses), [{ ok: true, images: true }]);
});

test("import and restore failures become truthful status and command responses", async () => {
  for (const scenario of ["import", "restore"]) {
    const bridge = await loadBridge(async () => {
      if (scenario === "import") throw new Error("module unavailable");
      return completeApi({ restoreSitePrefs: async () => { throw new Error("preferences corrupt"); } });
    });
    await flush();

    const status = bridge.message({ type: "FSRCNNX_STATUS" });
    const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
    await flush();
    assert.equal(status.responses.length, 1, scenario);
    assert.equal(status.responses[0].loading, false, scenario);
    assert.equal(status.responses[0].failed, true, scenario);
    assert.equal(status.responses[0].error, "startup-failed", scenario);
    assert.equal(status.responses[0].statusVersion, 1, scenario);
    assert.equal(status.responses[0].gpuState, "idle", scenario);
    assert.equal(status.responses[0].runtime.phase, "failed", scenario);
    assert.equal(status.responses[0].renderer.phase, "failed", scenario);
    assert.match(status.responses[0].reason, scenario === "import" ? /module unavailable/ : /preferences corrupt/);
    assert.equal(command.responses.length, 1, scenario);
    assert.equal(command.responses[0].error, "startup-failed", scenario);
  }
});

test("page lifecycle events coalesce and serialize suspend/resume transitions", async () => {
  const suspend = deferred();
  const calls = [];
  const api = completeApi({
    resumeDocument: async () => { calls.push("resume"); },
    suspendDocument: async () => { calls.push("suspend"); await suspend.promise; },
  });
  const bridge = await loadBridge(async () => api);
  await flush();
  assert.deepEqual(calls, ["resume"], "initial ownership is reconciled after restore");

  bridge.window.emit("pagehide");
  bridge.document.emit("freeze");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend"]);
  assert.deepEqual(
    plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [["active", 1], ["active", 1], ["hidden", 2], ["hidden", 2]],
    "the initial resume is preceded by a confirming ownership handshake and repeated hidden signals reannounce",
  );

  bridge.window.emit("pageshow");
  bridge.document.emit("resume");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend"], "resume waits behind in-flight suspend");
  assert.deepEqual(
    plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [
      ["active", 1], ["active", 1], ["hidden", 2], ["hidden", 2],
      ["active", 3], ["active", 3],
    ],
  );

  suspend.resolve();
  await flush();
  assert.deepEqual(calls, ["resume", "suspend", "resume"]);
});

test("BFCache activation synchronizes preferences once before resuming", async () => {
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    restoreSitePrefs: async () => { calls.push("restore"); return { ok: true }; },
    syncSitePrefs: async () => { calls.push("sync"); return { ok: true }; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }));
  await flush();
  assert.deepEqual(calls, ["restore", "resume"],
    "initial startup restores once without performing a redundant sync");

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  bridge.document.emit("resume");
  await flush();

  assert.deepEqual(calls, ["restore", "resume", "suspend", "sync", "resume"]);
  assert.equal(calls.filter((call) => call === "sync").length, 1,
    "same-generation active signals share one preference sync");
});

test("pagehide during a deferred preference sync suppresses the obsolete resume", async () => {
  const synced = deferred();
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => { calls.push("sync"); return synced.promise; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }));
  await flush();

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend", "sync"]);

  bridge.window.emit("pagehide");
  await flush();
  assert.equal(calls.at(-1), "suspend", "the newly hidden document is quiesced during sync");

  synced.resolve({ ok: true });
  await flush();
  assert.equal(calls.filter((call) => call === "resume").length, 1,
    "the completed stale sync cannot resume the hidden document");
  assert.equal(calls.at(-1), "suspend");
});

test("mutating commands wait behind active sync and flush before responding", async () => {
  const synced = deferred();
  const flushed = deferred();
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => { calls.push("sync"); return synced.promise; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
    setMode: async (mode) => { calls.push(`set:${mode}`); return { ok: true, mode }; },
    flushPreferenceWrites: async () => { calls.push("flush"); return flushed.promise; },
  }));
  await flush();
  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();

  const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();
  assert.equal(command.responses.length, 0);
  assert.equal(calls.some((call) => call.startsWith("set:")), false,
    "the popup mutation cannot race ahead of restored preferences");

  synced.resolve({ ok: true });
  await flush();
  assert.ok(calls.indexOf("sync") < calls.indexOf("set:upscale"));
  assert.ok(calls.indexOf("set:upscale") < calls.indexOf("flush"));
  assert.equal(command.responses.length, 0, "the response waits for durable persistence");

  const status = bridge.message({ type: "FSRCNNX_STATUS" });
  await flush();
  assert.equal(status.responses.length, 1, "status reads do not wait for preference writes");
  assert.equal(command.responses.length, 0);

  flushed.resolve({ ok: true });
  await flush();
  assert.deepEqual(plain(command.responses), [{ ok: true, mode: "upscale" }]);
});

test("resolved false flush results fail the mutation response", async () => {
  let setterCalls = 0;
  const flushResults = [
    { ok: false, reason: "quota exhausted" },
    false,
  ];
  const bridge = await loadBridge(async () => completeApi({
    setMode: async (mode) => { setterCalls++; return { ok: true, mode }; },
    flushPreferenceWrites: async () => flushResults.shift(),
  }));
  await flush();

  const objectFailure = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();
  assert.equal(setterCalls, 1);
  assert.deepEqual(plain(objectFailure.responses), [{
    ok: false,
    error: "command-failed",
    reason: "Preference write flush failed: quota exhausted",
  }]);

  const literalFailure = bridge.message({ type: "FSRCNNX_SETMODE", mode: "passthrough" });
  await flush();
  assert.equal(setterCalls, 2);
  assert.deepEqual(plain(literalFailure.responses), [{
    ok: false,
    error: "command-failed",
    reason: "Preference write flush failed: Preference writes could not be flushed",
  }]);
});

test("ordinary active-page mutations synchronize live external preferences before setting", async () => {
  const synced = deferred();
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => { calls.push("sync"); return synced.promise; },
    setMode: async (mode) => { calls.push(`set:${mode}`); return { ok: true, mode }; },
    flushPreferenceWrites: async () => { calls.push("flush"); return { ok: true }; },
  }));
  await flush();
  calls.length = 0;

  const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();
  assert.deepEqual(calls, ["sync"]);
  assert.equal(command.responses.length, 0);

  synced.resolve({ ok: true });
  await flush();
  assert.deepEqual(calls, ["sync", "set:upscale", "flush"]);
  assert.deepEqual(plain(command.responses), [{ ok: true, mode: "upscale" }]);
});

test("preference sync and flush hooks remain optional for older main APIs", async () => {
  const api = completeApi();
  delete api.syncSitePrefs;
  delete api.flushPreferenceWrites;
  const bridge = await loadBridge(async () => api);
  await flush();

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();

  assert.deepEqual(plain(command.responses), [{ ok: true, mode: "upscale" }]);
});

test("preference sync failures leave the document suspended and retry cleanly", async () => {
  let syncCalls = 0;
  let resumeCalls = 0;
  let setterCalls = 0;
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => {
      syncCalls++;
      if (syncCalls === 1) throw new Error("storage unavailable");
      return { ok: true };
    },
    resumeDocument: async () => { resumeCalls++; return { ok: true }; },
    setMode: async () => { setterCalls++; return { ok: true }; },
  }));
  await flush();
  assert.equal(resumeCalls, 1);

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  assert.equal(syncCalls, 1);
  assert.equal(resumeCalls, 1, "failed synchronization keeps the renderer suspended");

  const status = bridge.message({ type: "FSRCNNX_STATUS" });
  const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();
  assert.equal(status.responses[0].failed, true);
  assert.equal(status.responses[0].error, "preference-sync-failed");
  assert.match(status.responses[0].reason, /storage unavailable/);
  assert.equal(status.responses[0].runtime.phase, "failed");
  assert.equal(status.responses[0].renderer.phase, "suspended");
  assert.deepEqual(plain(command.responses), [{
    ok: false,
    error: "command-failed",
    reason: "storage unavailable",
  }]);
  assert.equal(setterCalls, 0);

  bridge.document.emit("resume");
  await flush();
  assert.equal(syncCalls, 2);
  assert.equal(resumeCalls, 2);
  const recovered = bridge.message({ type: "FSRCNNX_STATUS" });
  await flush();
  assert.equal(recovered.responses[0].failed, false);
  assert.equal(recovered.responses[0].error, undefined);
});

test("a transient active preference sync failure retries without another lifecycle event", async () => {
  let syncCalls = 0;
  let resumeCalls = 0;
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => {
      syncCalls++;
      if (syncCalls === 1) throw new Error("temporary storage failure");
      return { ok: true };
    },
    resumeDocument: async () => { resumeCalls++; return { ok: true }; },
  }));
  await flush();

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  assert.equal(syncCalls, 1);
  assert.equal(resumeCalls, 1);

  await new Promise((resolve) => setTimeout(resolve, 140));
  await flush();
  assert.equal(syncCalls, 2);
  assert.equal(resumeCalls, 2,
    "the successful bounded retry resumes the still-active document");
  const recovered = bridge.message({ type: "FSRCNNX_STATUS" });
  await flush();
  assert.equal(recovered.responses[0].failed, false);
});

test("a page hidden during restore never performs an obsolete initial resume", async () => {
  const imported = deferred();
  const restored = deferred();
  const calls = [];
  const bridge = await loadBridge(() => imported.promise);
  bridge.window.emit("pagehide");
  imported.resolve(completeApi({
    restoreSitePrefs: () => restored.promise,
    resumeDocument: async () => { calls.push("resume"); },
    suspendDocument: async () => { calls.push("suspend"); },
  }));
  await flush();
  assert.deepEqual(calls, ["suspend"], "hidden documents quiesce before restore completes");
  restored.resolve({ ok: true, restored: false });
  await flush();
  assert.deepEqual(calls, ["suspend"]);
  assert.deepEqual(plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [["active", 1], ["hidden", 2]]);
});

test("pageshow cannot overtake an early asynchronous suspension", async () => {
  const restored = deferred();
  const suspended = deferred();
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    restoreSitePrefs: () => restored.promise,
    resumeDocument: async () => { calls.push("resume"); },
    suspendDocument: async () => { calls.push("suspend"); await suspended.promise; },
  }));
  await flush();

  bridge.window.emit("pagehide");
  await flush();
  assert.deepEqual(calls, ["suspend"]);
  bridge.window.emit("pageshow");
  restored.resolve({ ok: true, restored: true });
  await flush();
  assert.deepEqual(calls, ["suspend"], "resume remains serialized behind early suspend");

  suspended.resolve();
  await flush();
  assert.deepEqual(calls, ["suspend", "resume"]);
});

test("failed hidden suspension cannot strand the next active preference gate", async () => {
  const suspended = deferred();
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => { calls.push("sync"); return { ok: true }; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return suspended.promise; },
    setMode: async (mode) => { calls.push(`set:${mode}`); return { ok: true, mode }; },
    flushPreferenceWrites: async () => { calls.push("flush"); return { ok: true }; },
  }));
  await flush();
  assert.deepEqual(calls, ["resume"]);

  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend"]);

  suspended.resolve({ ok: false, reason: "suspension failed" });
  await flush();
  assert.deepEqual(calls, ["resume", "suspend", "sync", "resume"],
    "the active gate must run even when the previously applied state was already active");

  const command = bridge.message({ type: "FSRCNNX_SETMODE", mode: "upscale" });
  await flush();
  assert.deepEqual(plain(command.responses), [{ ok: true, mode: "upscale" }]);
  assert.deepEqual(calls.slice(-3), ["sync", "set:upscale", "flush"]);
});

test("pagehide fences an in-flight resume before it can reactivate a hidden document", async () => {
  const secondResume = deferred();
  const calls = [];
  let resumeCalls = 0;
  const bridge = await loadBridge(async () => completeApi({
    resumeDocument: async () => {
      resumeCalls++;
      calls.push("resume");
      if (resumeCalls === 2) await secondResume.promise;
      return { ok: true };
    },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }));
  await flush();
  bridge.window.emit("pagehide");
  await flush();
  bridge.window.emit("pageshow");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend", "resume"]);

  bridge.window.emit("pagehide");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend", "resume", "suspend"],
    "suspend must run without waiting for an obsolete resume to settle");

  secondResume.resolve();
  await flush();
  assert.equal(calls.at(-1), "suspend",
    "the stale resume completion is followed by a final quiescence fence");
  assert.ok(calls.filter((call) => call === "suspend").length >= 2);
});

test("a failed lifecycle transition is retried by a repeated browser lifecycle signal", async () => {
  let resumes = 0;
  const bridge = await loadBridge(async () => completeApi({
    resumeDocument: async () => {
      resumes++;
      return resumes === 1 ? { ok: false, reason: "temporarily unavailable" } : { ok: true };
    },
  }));
  await flush();
  assert.equal(resumes, 1);

  bridge.document.emit("resume");
  await flush();
  assert.equal(resumes, 2);
  assert.deepEqual(plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [["active", 1], ["active", 1], ["active", 1], ["active", 1]],
    "same-state recovery retries retain one lifecycle generation");
});

test("prerendered documents stay suspended until browser activation", async () => {
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    restoreSitePrefs: async () => { calls.push("restore"); return { ok: true }; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }), { prerendering: true });
  await flush();

  assert.deepEqual(calls, ["suspend", "restore"]);
  assert.deepEqual(plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [["hidden", 1]]);

  bridge.document.prerendering = false;
  bridge.document.emit("prerenderingchange");
  await flush();
  assert.deepEqual(calls, ["suspend", "restore", "resume"]);
  assert.deepEqual(plain(bridge.sentMessages.map(({ state, generation }) => [state, generation])),
    [["hidden", 1], ["active", 2], ["active", 2]]);
});

test("ordinary visibility changes retire and restore document ownership", async () => {
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    restoreSitePrefs: async () => { calls.push("restore"); return { ok: true }; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }), { visibilityState: "hidden" });
  await flush();

  assert.deepEqual(calls, ["suspend", "restore"]);
  bridge.window.emit("pageshow");
  await flush();
  assert.deepEqual(calls, ["suspend", "restore"],
    "pageshow must not reactivate a document that is still visibility-hidden");

  bridge.document.visibilityState = "visible";
  bridge.document.emit("visibilitychange");
  await flush();
  assert.deepEqual(calls, ["suspend", "restore", "resume"]);

  bridge.document.visibilityState = "hidden";
  bridge.document.emit("visibilitychange");
  await flush();
  assert.deepEqual(calls, ["suspend", "restore", "resume", "suspend"]);
});

test("an extension message repairs a missed visible-document activation", async () => {
  const calls = [];
  const bridge = await loadBridge(async () => completeApi({
    syncSitePrefs: async () => { calls.push("sync"); return { ok: true }; },
    resumeDocument: async () => { calls.push("resume"); return { ok: true }; },
    suspendDocument: async () => { calls.push("suspend"); return { ok: true }; },
  }));
  await flush();

  bridge.document.visibilityState = "hidden";
  bridge.window.emit("pagehide");
  await flush();
  assert.deepEqual(calls, ["resume", "suspend"]);

  // Simulate a browser activation that updates visibility but drops the usual
  // pageshow/resume/visibilitychange signal before the content script runs.
  bridge.document.visibilityState = "visible";
  const status = bridge.message({ type: "FSRCNNX_STATUS" });
  await flush();

  assert.equal(status.responses.length, 1);
  assert.deepEqual(calls, ["resume", "suspend", "sync", "resume"]);
});

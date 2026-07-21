import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const contentUrl = new URL("../content.js", import.meta.url);

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

async function loadBridge(importModule, { prerendering = false } = {}) {
  const original = await readFile(contentUrl, "utf8");
  const source = original.replace(
    "api = await import(url);",
    "api = await globalThis.__importModule(url);",
  );
  assert.notEqual(source, original, "content module-loader injection must match production source");

  const window = new FakeEventTarget();
  const document = new FakeEventTarget();
  document.prerendering = prerendering;
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
    setDeband: () => ({ ok: true }),
    setDebandStrength: () => ({ ok: true }),
    setHoverReveal: () => ({ ok: true }),
    setAllVideos: () => ({ ok: true }),
    setSharpen: () => ({ ok: true }),
    setSharpenStrength: () => ({ ok: true }),
    setSSimDS: () => ({ ok: true }),
    setPolicy: () => ({ ok: true }),
    ...overrides,
  };
}

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

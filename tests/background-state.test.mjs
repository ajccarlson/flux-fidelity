import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const backgroundUrl = new URL("../background.js", import.meta.url);
const manifestUrl = new URL("../manifest.json", import.meta.url);

async function flush(turns = 4) {
  for (let i = 0; i < turns; i++) await new Promise((resolve) => setImmediate(resolve));
}

async function loadBackground({ rejectActions = false } = {}) {
  const source = await readFile(backgroundUrl, "utf8");
  const calls = [];
  const caughtRejections = [];
  const listeners = {};
  const action = {};
  for (const method of [
    "setBadgeText", "setBadgeBackgroundColor", "setBadgeTextColor", "setTitle", "setIcon",
  ]) {
    action[method] = (details) => {
      calls.push({ method, details });
      if (!rejectActions) return Promise.resolve();
      return {
        catch(handler) {
          caughtRejections.push(method);
          handler(new Error(`${method} rejected`));
          return this;
        },
      };
    };
  }
  const chrome = {
    action,
    runtime: { onMessage: { addListener(listener) { listeners.message = listener; } } },
    tabs: {
      onUpdated: { addListener(listener) { listeners.updated = listener; } },
      onRemoved: { addListener(listener) { listeners.removed = listener; } },
      onReplaced: { addListener(listener) { listeners.replaced = listener; } },
    },
  };
  const context = vm.createContext({ chrome, Map, Number, Set });
  new vm.Script(source, { filename: "background.js" }).runInContext(context);
  return { calls, caughtRejections, listeners };
}

function sender(tabId, documentId, { frameId = 0, lifecycle = "active" } = {}) {
  return { tab: { id: tabId }, frameId, documentId, documentLifecycle: lifecycle };
}

function documentMessage(state, generation = 1) {
  return { type: "FSRCNNX_DOCUMENT", state, generation };
}

function callsSince(state, index, method) {
  return state.calls.slice(index).filter((call) => call.method === method);
}

test("only an active injected document can own and update a tab badge", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;

  message(documentMessage("active"), sender(4, "cached", { lifecycle: "cached" }));
  message(documentMessage("active"), { tab: { id: 4 }, frameId: 0 });
  message({ type: "FSRCNNX_DOCUMENT", state: "active" }, sender(4, "missing-generation"));
  assert.equal(state.calls.length, 0);

  message(documentMessage("active"), sender(4, "doc-a", { frameId: 9 }));
  assert.equal(state.calls.find((call) => call.method === "setBadgeText")?.details.text, "");
  let mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(4, "doc-a", { frameId: 9 }));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["ON"]);
  assert.equal(callsSince(state, mark, "setIcon")[0].details.path[16], "icons/icon-16.png");

  mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "bogus" }, sender(4, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "off" }, sender(4, "doc-a", { lifecycle: "cached" }));
  assert.equal(state.calls.length, mark, "invalid modes and non-active senders are rejected");
});

test("an active state message can recover ownership after service-worker restart", async () => {
  const state = await loadBackground();
  state.listeners.message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(6, "live-doc"));
  assert.equal(state.calls.filter((call) => call.method === "setBadgeText").at(-1).details.text, "ON");

  const mark = state.calls.length;
  state.listeners.message(
    { type: "FSRCNNX_STATE", mode: "passthrough" },
    sender(6, "cached-doc", { lifecycle: "cached" }),
  );
  assert.equal(state.calls.length, mark);
});

test("replacement, BFCache handshakes, and stale messages preserve current ownership", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(8, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(8, "doc-a"));

  let mark = state.calls.length;
  message(documentMessage("active"), sender(8, "doc-b"));
  assert.equal(state.calls.length, mark, "a new document cannot displace a still-active owner");
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(8, "doc-b"));
  message(documentMessage("active"), sender(8, "doc-a"));
  assert.equal(state.calls.length, mark,
    "a delayed current-owner claim resets fallback confirmations without losing buffered state");
  message(
    documentMessage("hidden", 2),
    sender(8, "doc-a", { frameId: 7, lifecycle: "cached" }),
  );
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["··"],
    "the current owner's explicit hidden signal still promotes that pending document");

  mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(8, "doc-a"));
  message(documentMessage("hidden", 2), sender(8, "doc-a", { lifecycle: "cached" }));
  assert.equal(state.calls.length, mark, "the replaced document cannot update or hide the owner");

  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(8, "doc-b"));
  assert.equal(state.calls.at(-5).details.text, "··");
  mark = state.calls.length;
  message(documentMessage("hidden", 2), sender(8, "doc-b", { lifecycle: "cached" }));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), [""]);

  mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(8, "doc-b"));
  assert.equal(state.calls.length, mark, "hidden owner state is rejected");

  message(documentMessage("active", 3), sender(8, "doc-a"));
  message({ type: "FSRCNNX_PROTECTED" }, sender(8, "doc-a"));
  assert.equal(state.calls.filter((call) => call.method === "setBadgeText").at(-1).details.text, "✕");
});

test("current-owner interleaving requires two fresh fallback confirmations", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(7, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(7, "doc-a"));

  const mark = state.calls.length;
  message(documentMessage("active"), sender(7, "doc-b"));
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(7, "doc-b"));
  message(documentMessage("active"), sender(7, "doc-a"));
  message(documentMessage("active"), sender(7, "doc-b"));
  assert.equal(state.calls.length, mark,
    "one retry after the current owner reaffirms cannot complete fallback");

  message(documentMessage("active"), sender(7, "doc-b"));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["··"]);
});

test("fallback confirmations cannot be combined across lifecycle generations", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(11, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(11, "doc-a"));

  const mark = state.calls.length;
  message(documentMessage("active"), sender(11, "doc-b"));
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(11, "doc-b"));
  message(documentMessage("active", 3), sender(11, "doc-b"));
  assert.equal(state.calls.length, mark,
    "a newer active lifecycle starts a fresh confirmation pair");
  message(documentMessage("active", 3), sender(11, "doc-b"));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["··"]);
});

test("a repeated active handshake recovers when the outgoing hidden signal is lost", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(10, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(10, "doc-a"));

  let mark = state.calls.length;
  message(documentMessage("active"), sender(10, "doc-b"));
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(10, "doc-b"));
  assert.equal(state.calls.length, mark,
    "one incoming handshake only stages ownership and buffers its visible state");

  message(documentMessage("active"), sender(10, "doc-b"));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["··"],
    "the repeated browser-confirmed handshake promotes the waiting document");

  mark = state.calls.length;
  message(documentMessage("active"), sender(10, "doc-a"));
  message(documentMessage("active"), sender(10, "doc-a"));
  message(documentMessage("active"), sender(10, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "protected" }, sender(10, "doc-a"));
  assert.equal(state.calls.length, mark,
    "consecutive active retries and state from the fallback-retired owner cannot reverse ownership");

  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(10, "doc-b"));
  message(documentMessage("hidden", 2), sender(10, "doc-b", { lifecycle: "cached" }));
  mark = state.calls.length;
  message(documentMessage("active"), sender(10, "doc-a"));
  message(documentMessage("active"), sender(10, "doc-a"));
  assert.equal(state.calls.length, mark,
    "a hidden current owner does not make retired same-generation retries authoritative");

  message(
    documentMessage("hidden", 2),
    sender(10, "doc-a", { frameId: 6, lifecycle: "cached" }),
  );
  message(documentMessage("active"), sender(10, "doc-a"));
  message(documentMessage("active"), sender(10, "doc-a"));
  assert.equal(state.calls.length, mark,
    "active messages older than observed hidden evidence remain rejected even when delivered later");

  message(documentMessage("active", 3), sender(10, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(10, "doc-a"));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), ["", "ON"],
    "newer hidden and active generations allow a normal BFCache return");
});

test("a retired document can fallback again after an unobserved hidden-active cycle", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(13, "doc-a"));
  message(documentMessage("active"), sender(13, "doc-b"));
  message(documentMessage("active"), sender(13, "doc-b"));

  const mark = state.calls.length;
  message(documentMessage("active", 3), sender(13, "doc-a"));
  assert.equal(state.calls.length, mark,
    "the newer lifecycle is eligible but still needs its second equal-generation confirmation");
  message(documentMessage("active", 3), sender(13, "doc-a"));
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), [""]);
});

test("fallback safely retires an owner reconstructed only from renderer state", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(14, "doc-a"));
  message(documentMessage("active"), sender(14, "doc-b"));
  message(documentMessage("active"), sender(14, "doc-b"));

  const mark = state.calls.length;
  message(documentMessage("active"), sender(14, "doc-a"));
  message(documentMessage("active"), sender(14, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(14, "doc-a"));
  assert.equal(state.calls.length, mark,
    "missing pre-restart lifecycle state cannot let the displaced owner reverse fallback");
});

test("a pending document that hides before promotion withdraws its claim", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(9, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(9, "doc-a"));

  message(documentMessage("active"), sender(9, "doc-b"));
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(9, "doc-b"));
  message(
    documentMessage("hidden", 2),
    sender(9, "doc-b", { frameId: 5, lifecycle: "cached" }),
  );
  const mark = state.calls.length;
  message(
    documentMessage("hidden", 2),
    sender(9, "doc-a", { frameId: 4, lifecycle: "cached" }),
  );
  assert.deepEqual(callsSince(state, mark, "setBadgeText").map((call) => call.details.text), [""],
    "a withdrawn pending document cannot be promoted later");
});

test("navigation, tab removal, and tab replacement fully reset ownership", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("active"), sender(12, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(12, "doc-a"));

  let mark = state.calls.length;
  state.listeners.updated(12, { status: "loading" });
  assert.equal(callsSince(state, mark, "setBadgeText")[0].details.text, "");
  assert.equal(callsSince(state, mark, "setTitle")[0].details.title, "Video Upscaler");
  assert.equal(callsSince(state, mark, "setIcon")[0].details.path[16], "icons/icon-off-16.png");
  mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(12, "doc-a"));
  message(documentMessage("active"), sender(12, "doc-a"));
  assert.equal(state.calls.length, mark);

  message(documentMessage("active"), sender(12, "doc-b"));
  mark = state.calls.length;
  message(documentMessage("active"), sender(12, "doc-a"));
  message(documentMessage("active"), sender(12, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(12, "doc-a"));
  assert.equal(state.calls.length, mark,
    "the navigation tombstone remains retired after the replacement claims ownership");
  message({ type: "FSRCNNX_STATE", mode: "passthrough" }, sender(12, "doc-b"));
  assert.equal(state.calls.filter((call) => call.method === "setBadgeText").at(-1).details.text, "··");
  state.listeners.removed(12);

  message(documentMessage("active"), sender(15, "old-tab-doc"));
  mark = state.calls.length;
  state.listeners.replaced(16, 15);
  assert.equal(callsSince(state, mark, "setBadgeText")[0].details.tabId, 16);
  assert.equal(callsSince(state, mark, "setBadgeText")[0].details.text, "");
  mark = state.calls.length;
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(16, "unclaimed-added-doc"));
  assert.equal(state.calls.length, mark);
  message(documentMessage("active"), sender(16, "new-tab-doc"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(16, "new-tab-doc"));
  assert.equal(state.calls.filter((call) => call.method === "setBadgeText").at(-1).details.text, "ON");
});

test("navigation retires an inferred active transition after worker-state recovery", async () => {
  const state = await loadBackground();
  const message = state.listeners.message;
  message(documentMessage("hidden"), sender(17, "doc-a", { lifecycle: "cached" }));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(17, "doc-a"));
  state.listeners.updated(17, { status: "loading" });
  message(documentMessage("active"), sender(17, "doc-b"));

  const mark = state.calls.length;
  message(documentMessage("hidden"), sender(17, "doc-a", { lifecycle: "cached" }));
  message(documentMessage("active", 2), sender(17, "doc-a"));
  message(documentMessage("active", 2), sender(17, "doc-a"));
  message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(17, "doc-a"));
  assert.equal(state.calls.length, mark,
    "an older duplicate hidden epoch cannot re-arm the inferred outgoing active generation");
});

test("every action Promise rejection is observed", async () => {
  const state = await loadBackground({ rejectActions: true });
  state.listeners.message(
    documentMessage("active"),
    sender(20, "doc-a"),
  );
  state.listeners.message({ type: "FSRCNNX_STATE", mode: "upscale" }, sender(20, "doc-a"));
  await flush();
  assert.equal(state.caughtRejections.length, state.calls.length);
  assert.ok(state.caughtRejections.includes("setBadgeText"));
  assert.ok(state.caughtRejections.includes("setIcon"));
});

test("manifest pins the browser version that supplies document identity metadata", async () => {
  const manifest = JSON.parse(await readFile(manifestUrl, "utf8"));
  assert.equal(manifest.minimum_chrome_version, "113");
  assert.deepEqual(manifest.permissions, ["storage"], "document ownership adds no permission");
  assert.equal(manifest.content_scripts[0].all_frames ?? false, false,
    "document lifecycle authority depends on outermost-only injection");
});

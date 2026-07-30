import test from "node:test";
import assert from "node:assert/strict";

import { Interpolator } from "../src/core/fsrcnnx-interpolate.js";
import { listModels, setModel } from "../src/core/fsrcnnx-rife.js";

function makeInterpolator() {
  return new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: () => {},
  });
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
}

function installGlobals(values) {
  const originals = new Map();
  for (const [key, value] of Object.entries(values)) {
    originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  return () => {
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  };
}

let cpuGrabModuleSequence = 0;

function installCpuGrabModule(state) {
  const stateKey = `__fsrcnnxCpuGrabState${++cpuGrabModuleSequence}`;
  globalThis[stateKey] = state;
  const source = `
    const state = globalThis[${JSON.stringify(stateKey)}];
    export class WebGPUGrabber {
      constructor(options = {}) {
        this.options = options;
        this.ready = false;
        state.instances.push(this);
        state.events.push("construct");
      }
      async init() {
        state.initCalls++;
        state.events.push("init");
        this.ready = true;
        return true;
      }
      destroy() {
        if (this._destroyPromise) return this._destroyPromise;
        state.destroyCalls++;
        state.events.push("replacement-destroy");
        this.ready = false;
        this._destroyPromise = Promise.resolve();
        return this._destroyPromise;
      }
    }
  `;
  const moduleUrl = `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
  const restoreGlobals = installGlobals({
    chrome: { runtime: { getURL: () => moduleUrl } },
  });
  return () => {
    restoreGlobals();
    delete globalThis[stateKey];
  };
}

test("only verified bundled RIFE models remain selectable", () => {
  assert.deepEqual(listModels().map(({ key }) => key).sort(), ["rife_v4.26", "rife_v4.26_fp16"]);
  assert.equal(setModel("rife_orig"), false);
});

test("interpolator retains engine choices made before runtime import", () => {
  const interpolator = makeInterpolator();

  assert.equal(interpolator.setInterpEngine("rife_v4.26_fp16"), false);
  assert.equal(interpolator._rifeModelKey, "rife_v4.26_fp16");
  assert.equal(interpolator._forceBlend, false);

  assert.equal(interpolator.setInterpEngine("blend"), true);
  assert.equal(interpolator._forceBlend, true);
  assert.equal(interpolator._interpMode, "blend");
});

test("concurrent starts share one lifecycle and failed startup cleans up", async () => {
  const interpolator = makeInterpolator();
  const first = interpolator.start();
  const second = interpolator.start();

  assert.equal(first, second);
  const result = await first;
  assert.equal(result.ok, false);
  assert.equal(result.reason, "unsupported");
  assert.equal(interpolator.running, false);
  assert.equal(interpolator._state, "idle");
  assert.deepEqual(interpolator.stop(), { ok: true, stopped: false });
});

test("resource retirement drains GPU teardown before releasing the model session", async () => {
  const interpolator = makeInterpolator();
  const gpu = deferred();
  const events = [];
  interpolator._rifeMod = {
    async destroyGpuInterp() {
      events.push("gpu-start");
      await gpu.promise;
      events.push("gpu-end");
    },
    async disposeRife() { events.push("model-release"); },
  };

  const gpuRetirement = interpolator.retireGpuResources();
  const fullRetirement = interpolator.releaseModelResources();
  await waitFor(() => events.length === 1, "GPU retirement did not start");
  assert.deepEqual(events, ["gpu-start"]);
  gpu.resolve();
  await gpuRetirement;
  await fullRetirement;
  assert.deepEqual(events, ["gpu-start", "gpu-end", "model-release"]);
});

test("resource retirement is published before stop callbacks can re-enter start", async () => {
  const interpolator = makeInterpolator();
  const gpuRelease = deferred();
  const events = [];
  let destroyPromise = null;
  let restart = null;
  let starts = 0;
  interpolator._rifeMod = {
    destroyGpuInterp() {
      if (!destroyPromise) {
        events.push("gpu-release-start");
        destroyPromise = gpuRelease.promise.then(() => events.push("gpu-release-end"));
      }
      return destroyPromise;
    },
  };
  interpolator.chain = {
    tap(on) {
      if (!on) {
        events.push("tap-off");
        restart = interpolator.start();
      }
    },
  };
  Object.assign(interpolator, {
    running: true,
    _state: "running",
    _stopped: false,
    video: {},
    overlay: { remove() {} },
    queue: [],
    _chainActive: true,
  });
  interpolator._startInternal = async () => {
    starts++;
    events.push("restart");
    return { ok: true };
  };

  const retirement = interpolator.retireGpuResources();
  assert.ok(restart, "chain teardown must have exercised synchronous re-entry");
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(starts, 0, "re-entrant startup must wait behind physical retirement");
  assert.deepEqual(events, ["gpu-release-start", "tap-off"]);

  gpuRelease.resolve();
  await retirement;
  assert.deepEqual(await restart, { ok: true });
  assert.equal(starts, 1);
  assert.deepEqual(events, ["gpu-release-start", "tap-off", "gpu-release-end", "restart"]);
});

test("full model release drops scratch backing stores, run closures, and the RIFE loss listener", async () => {
  const interpolator = makeInterpolator();
  const listenerOwner = new Set();
  const listener = () => interpolator._handleRifeDeviceLoss({}, {});
  listenerOwner.add(listener);
  interpolator._deviceLossUnsubscribe = () => listenerOwner.delete(listener);

  const makeScratch = (width, height) => ({ width, height });
  const grab = makeScratch(1920, 1080);
  const blend = makeScratch(1920, 1080);
  const tween = makeScratch(1920, 1080);
  interpolator._grab = grab;
  interpolator._grabCtx = {};
  interpolator._blend = blend;
  interpolator._blendCtx = {};
  interpolator._tw = tween;
  interpolator._twctx = {};
  interpolator._flush = () => {};
  interpolator._lastTapFrame = { sourceSizedTexture: true };
  interpolator._lastTapW = 1920;
  interpolator._lastTapH = 1080;
  interpolator._lastVW = 1920;
  interpolator._lastVH = 1080;
  interpolator._infWindow = [1, 2, 3];
  interpolator._srcFrameBase = 12;
  interpolator._outFrameBase = 24;
  interpolator._fbWindowStart = 100;
  interpolator._prevTs = 123;
  interpolator._lastPresentedTs = 456;
  interpolator._lastEnqTs = 789;
  interpolator._videoLatencyMs = 10;
  interpolator._rifeMod = {
    async destroyGpuInterp() {},
    async disposeRife() {},
  };

  const release = interpolator.releaseModelResources();
  assert.equal(listenerOwner.size, 0,
    "the external module listener owner must release the instance synchronously");
  await release;

  for (const canvas of [grab, blend, tween]) {
    assert.equal(canvas.width, 0);
    assert.equal(canvas.height, 0);
  }
  for (const key of [
    "_grab", "_grabCtx", "_blend", "_blendCtx", "_tw", "_twctx", "_flush",
    "_lastTapFrame", "_lastVW", "_lastVH", "_infWindow", "_srcFrameBase",
    "_outFrameBase", "_fbWindowStart", "_prevTs", "_lastPresentedTs",
    "_lastEnqTs", "_videoLatencyMs", "_deviceLossUnsubscribe", "_rifeMod",
  ]) assert.equal(interpolator[key], null, `${key} should be released`);
  assert.equal(interpolator._lastTapW, 0);
  assert.equal(interpolator._lastTapH, 0);
});

test("full release drains an active CPU frame before retiring its scratch backing", async () => {
  const interpolator = makeInterpolator();
  const frame = deferred();
  const scratch = { width: 1280, height: 720 };
  const events = [];
  interpolator._activeCpuFrameTasks.add(frame.promise);
  interpolator._grab = scratch;
  interpolator._grabCtx = {};
  interpolator._rifeMod = {
    async destroyGpuInterp() { events.push("gpu-release"); },
    async disposeRife() { events.push("model-release"); },
  };

  const release = interpolator.releaseModelResources();
  await Promise.resolve();
  assert.equal(scratch.width, 1280);
  assert.deepEqual(events, [], "retirement must not overtake the active CPU frame");

  frame.resolve();
  await release;
  assert.deepEqual(events, ["gpu-release", "model-release"]);
  assert.equal(scratch.width, 0);
  assert.equal(scratch.height, 0);
});

test("GPU retirement drains active CPU work, releases per-run scratch, and permits restart", async () => {
  const interpolator = makeInterpolator();
  const frame = deferred();
  const events = [];
  const scratch = {
    grab: { width: 1280, height: 720 },
    grabCtx: {},
    blend: { width: 1280, height: 720 },
    blendCtx: {},
    tween: { width: 1280, height: 720 },
    tweenCtx: {},
    flush: () => {},
  };
  Object.assign(interpolator, {
    _grab: scratch.grab,
    _grabCtx: scratch.grabCtx,
    _blend: scratch.blend,
    _blendCtx: scratch.blendCtx,
    _tw: scratch.tween,
    _twctx: scratch.tweenCtx,
    _flush: scratch.flush,
    _lastTapFrame: { sourceSizedTexture: true },
    _lastTapW: 1280,
    _lastTapH: 720,
    _lastVW: 1280,
    _lastVH: 720,
    _infWindow: [1, 2],
    _srcFrameBase: 10,
    _outFrameBase: 20,
    _fbWindowStart: 30,
    _prevTs: 40,
    _lastPresentedTs: 50,
    _lastEnqTs: 60,
    _videoLatencyMs: 70,
  });
  interpolator._activeCpuFrameTasks.add(frame.promise);
  const rifeModule = {
    async destroyGpuInterp() { events.push("gpu-release"); },
  };
  interpolator._rifeMod = rifeModule;

  const retirement = interpolator.retireGpuResources();
  await Promise.resolve();
  assert.deepEqual(events, []);
  for (const canvas of [scratch.grab, scratch.blend, scratch.tween]) {
    assert.equal(canvas.width, 1280, "scratch must remain valid while CPU work is active");
    assert.equal(canvas.height, 720, "scratch must remain valid while CPU work is active");
  }

  frame.resolve();
  await retirement;
  assert.deepEqual(events, ["gpu-release"]);
  for (const canvas of [scratch.grab, scratch.blend, scratch.tween]) {
    assert.equal(canvas.width, 0);
    assert.equal(canvas.height, 0);
  }
  for (const key of [
    "_grab", "_grabCtx", "_blend", "_blendCtx", "_tw", "_twctx", "_flush",
    "_lastTapFrame", "_lastVW", "_lastVH", "_infWindow", "_srcFrameBase",
    "_outFrameBase", "_fbWindowStart", "_prevTs", "_lastPresentedTs",
    "_lastEnqTs", "_videoLatencyMs",
  ]) assert.equal(interpolator[key], null, `${key} should be released`);
  assert.equal(interpolator._lastTapW, 0);
  assert.equal(interpolator._lastTapH, 0);
  assert.equal(interpolator._activeCpuFrameTasks.size, 0);
  assert.equal(interpolator._rifeMod, rifeModule,
    "GPU retirement must preserve the model module for a later restart");

  const fresh = { width: 320, height: 180 };
  interpolator._startInternal = async () => {
    assert.equal(interpolator._grab, null,
      "restart must construct a new lifecycle after explicit retirement");
    interpolator._grab = fresh;
    interpolator._grabCtx = {};
    return { ok: true };
  };
  assert.deepEqual(await interpolator.start(), { ok: true });
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._state, "running");
  assert.equal(interpolator._grab, fresh);
});

test("ordinary stop keeps reusable scratch state and the shared loss subscription", async () => {
  const interpolator = makeInterpolator();
  const scratch = {
    grab: { width: 640, height: 360 },
    grabCtx: {},
    blend: { width: 640, height: 360 },
    blendCtx: {},
    tween: { width: 640, height: 360 },
    tweenCtx: {},
    flush: () => {},
  };
  Object.assign(interpolator, {
    _grab: scratch.grab,
    _grabCtx: scratch.grabCtx,
    _blend: scratch.blend,
    _blendCtx: scratch.blendCtx,
    _tw: scratch.tween,
    _twctx: scratch.tweenCtx,
    _flush: scratch.flush,
    running: true,
    _state: "running",
    _stopped: false,
    video: {},
    overlay: { remove() {} },
    queue: [],
  });
  let unsubscribes = 0;
  interpolator._deviceLossUnsubscribe = () => { unsubscribes++; };
  interpolator._rifeMod = { destroyGpuInterp() {} };

  assert.deepEqual(interpolator.stop(), { ok: true, stopped: true });
  assert.equal(interpolator._grab, scratch.grab);
  assert.equal(interpolator._grabCtx, scratch.grabCtx);
  assert.equal(interpolator._blend, scratch.blend);
  assert.equal(interpolator._blendCtx, scratch.blendCtx);
  assert.equal(interpolator._tw, scratch.tween);
  assert.equal(interpolator._twctx, scratch.tweenCtx);
  assert.equal(interpolator._flush, scratch.flush);
  assert.equal(interpolator._deviceLossUnsubscribe instanceof Function, true);
  assert.equal(unsubscribes, 0);

  interpolator._startInternal = async () => ({ ok: true });
  assert.deepEqual(await interpolator.start(), { ok: true });
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._state, "running");
});

test("ordinary stop resets ownership synchronously and one restart waits for grabber destruction", async () => {
  const interpolator = makeInterpolator();
  const destruction = deferred();
  let destroyCalls = 0;
  let starts = 0;
  const oldGrabber = {
    ready: true,
    destroy() {
      destroyCalls++;
      this.ready = false;
      return destruction.promise;
    },
  };
  Object.assign(interpolator, {
    running: true,
    _state: "running",
    _stopped: false,
    video: {},
    overlay: { remove() {} },
    queue: [],
    _gpuGrab: oldGrabber,
  });
  interpolator._startInternal = async () => {
    starts++;
    return { ok: true };
  };

  assert.deepEqual(interpolator.stop(), { ok: true, stopped: true });
  assert.equal(interpolator.running, false);
  assert.equal(interpolator._state, "idle");
  assert.equal(interpolator.video, null);
  assert.equal(interpolator._gpuGrab, null);
  assert.equal(destroyCalls, 1);

  const first = interpolator.start();
  const second = interpolator.start();
  assert.equal(second, first, "restart callers must share the teardown-gated start");
  await Promise.resolve();
  assert.equal(starts, 0, "startup must not request replacement resources during physical teardown");

  destruction.resolve();
  assert.deepEqual(await first, { ok: true });
  assert.equal(starts, 1);
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._state, "running");
  assert.equal(interpolator._cpuGrabberTeardowns.size, 0);
});

test("a dimension restart cannot overlap the prior CPU grabber device", async () => {
  const interpolator = makeInterpolator();
  const destruction = deferred();
  const events = [];
  const oldGrabber = {
    ready: true,
    destroy() {
      events.push("destroy-start");
      this.ready = false;
      return destruction.promise.then(() => events.push("destroy-end"));
    },
  };
  Object.assign(interpolator, {
    running: true,
    _state: "running",
    _stopped: false,
    video: {},
    overlay: { remove() {} },
    queue: [],
    _gpuGrab: oldGrabber,
  });
  interpolator._startInternal = async () => {
    events.push("start");
    return { ok: true };
  };

  const generation = interpolator._lifecycleGen;
  assert.equal(interpolator._scheduleDimsRestart(generation, 1920, 1080), true);
  await waitFor(() => events.includes("destroy-start"), "dimension restart did not stop the old grabber");
  assert.deepEqual(events, ["destroy-start"]);

  destruction.resolve();
  await waitFor(() => events.includes("start") && !interpolator._dimsRestarting,
    "dimension restart did not resume after grabber teardown");
  assert.deepEqual(events, ["destroy-start", "destroy-end", "start"]);
  assert.equal(interpolator.running, true);
});

test("an exact source bypasses rescoring and an early failure preserves its visibility", async () => {
  const restore = installGlobals({
    createImageBitmap: () => {},
    OffscreenCanvas: class {},
  });
  try {
    let rescored = 0;
    const source = { style: { visibility: "collapse" } };
    const interpolator = new Interpolator({
      findVideo: () => { rescored++; return { style: { visibility: "visible" } }; },
      log: () => {},
      warn: () => {},
    });

    const result = await interpolator.start(source);

    assert.deepEqual(result, { ok: false, reason: "no-rvfc" });
    assert.equal(rescored, 0);
    assert.equal(source.style.visibility, "collapse");
    assert.equal(interpolator.video, null);
  } finally {
    restore();
  }
});

test("a chain source accessor is authoritative, including a missing source", () => {
  const owned = {};
  let rescored = 0;
  const explicit = new Interpolator({
    sourceVideo: () => owned,
    findVideo: () => { rescored++; return {}; },
    log: () => {},
    warn: () => {},
  });
  assert.equal(explicit._resolveSourceVideo(), owned);
  const interpolator = new Interpolator({
    findVideo: () => { rescored++; return {}; },
    chain: { source: () => owned },
    log: () => {},
    warn: () => {},
  });
  assert.equal(interpolator._resolveSourceVideo(), owned);
  interpolator.chain.source = () => null;
  assert.equal(interpolator._resolveSourceVideo(), null);
  assert.equal(rescored, 0);
});

test("inverted chaining requires explicit capability for the exact source", () => {
  const source = {};
  const other = {};
  const chain = { source: () => source };
  const interpolator = new Interpolator({ chain, log: () => {}, warn: () => {} });

  assert.equal(interpolator._chainCanInvert(source), false, "legacy chains must not imply inversion");
  chain.canInvert = (candidate) => candidate === source;
  assert.equal(interpolator._chainCanInvert(source), true);
  assert.equal(interpolator._chainCanInvert(other), false);
  chain.source = () => other;
  assert.equal(interpolator._chainCanInvert(source), false, "renderer/source mismatch disables inversion");
});

test("committed interpolation diagnostics report the real sink and reset with takeover lifecycle", () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator.video = { videoWidth: 640, videoHeight: 360 };
  interpolator.overlay = { width: 1280, height: 720, style: {}, remove() {} };
  interpolator.queue = [];
  interpolator._takeoverActive = true;
  interpolator.stats = { framesIn: 8, framesOut: 14, started: 0 };

  interpolator._recordCommittedPresentation({ gpu: true, width: 1280, height: 720 });
  assert.deepEqual(interpolator.getStats().presentation, {
    committed: true,
    generation: 1,
    gpu: true,
    sink: "overlay",
    source: { width: 1280, height: 720 },
    output: { width: 1280, height: 720 },
    framesIn: 8,
    framesOut: 14,
  });
  assert.equal(interpolator.getStats().takeoverActive, true);
  assert.equal(interpolator.getStats().framesPresented, 1);
  assert.doesNotThrow(() => JSON.stringify(interpolator.getStats().presentation));

  interpolator.chain = { targetDims: () => ({ w: 2560, h: 1440 }) };
  interpolator._chainInverted = true;
  interpolator._recordCommittedPresentation({ gpu: true, width: 640, height: 360 });
  assert.deepEqual(interpolator.getStats().presentation, {
    committed: true,
    generation: 2,
    gpu: true,
    sink: "renderer",
    source: { width: 640, height: 360 },
    output: { width: 2560, height: 1440 },
    framesIn: 8,
    framesOut: 14,
  });

  interpolator._relinquishPresentation();
  assert.equal(interpolator.getStats().takeoverActive, false);
  assert.equal(interpolator.getStats().presentation, null);
  interpolator.stop();
  assert.equal(interpolator.getStats().framesPresented, 0,
    "source handoff, suspension, and off all stop the lifecycle and reset its counter");
});

test("loss of inverted capability never hides the source behind a rejected upscale", async () => {
  let nextFrame = null;
  let released = 0;
  let upscaleCalls = 0;
  let invertedOff = 0;
  const restore = installGlobals({
    requestAnimationFrame(callback) { nextFrame = callback; return 1; },
    cancelAnimationFrame() {},
    document: { fullscreenElement: null, body: {} },
  });
  try {
    const video = { style: { visibility: "visible" } };
    const chain = {
      source: () => video,
      canInvert: () => false,
      upscaleTex() { upscaleCalls++; return false; },
      setInverted(value) { if (!value) invertedOff++; },
    };
    const interpolator = new Interpolator({ chain, log: () => {}, warn: () => {} });
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = { style: { display: "none" }, remove() {} };
    interpolator._chainInverted = true;
    interpolator._rifeMod = { gpuRelease() { released++; } };
    interpolator._targetInterval = 16.7;
    interpolator.queue = Array.from({ length: 4 }, () => ({
      tex: { _w: 16, _h: 9 },
      ts: 0,
      enq: performance.now(),
    }));
    const generation = interpolator._lifecycleGen;

    interpolator._present(generation);
    assert.equal(typeof nextFrame, "function");
    nextFrame(performance.now());
    await Promise.resolve();

    assert.equal(upscaleCalls, 0);
    assert.equal(video.style.visibility, "visible");
    assert.equal(released, 4);
    assert.equal(invertedOff, 1);
    assert.equal(interpolator.running, false);
  } finally {
    restore();
  }
});

test("DOM and audio takeover keeps the source visible and repairs detached/fullscreen presentation", () => {
  const windowListeners = [];
  const videoListeners = [];
  const documentListeners = new Map();
  let appended = 0;
  let removed = 0;
  let audioSetups = 0;
  let audioTeardowns = 0;
  const document = {
    baseURI: "https://example.test/",
    fullscreenElement: null,
    body: { appendChild(node) { appended++; node.isConnected = true; node.parentNode = this; } },
    addEventListener(type, listener) { documentListeners.set(type, listener); },
    removeEventListener(type, listener) {
      if (documentListeners.get(type) === listener) documentListeners.delete(type);
    },
  };
  const restore = installGlobals({
    document,
    window: {
      addEventListener(type) { windowListeners.push(["add", type]); },
      removeEventListener(type) { windowListeners.push(["remove", type]); },
    },
  });
  try {
    const video = {
      style: { visibility: "collapse" },
      parentElement: null,
      getBoundingClientRect: () => ({ left: 1, top: 2, width: 3, height: 4 }),
      addEventListener(type) { videoListeners.push(["add", type]); },
      removeEventListener(type) { videoListeners.push(["remove", type]); },
    };
    const overlay = {
      isConnected: false,
      parentNode: null,
      style: {},
      remove() { removed++; this.isConnected = false; this.parentNode = null; },
    };
    const interpolator = makeInterpolator();
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = overlay;
    interpolator.queue = [];
    interpolator._flush = () => {};
    interpolator._stageAudioDelay = () => ({ bypass: true, status: "staged", rollback() {} });
    interpolator._commitAudioDelay = (transaction) => {
      audioSetups++;
      transaction.status = "committed";
      return true;
    };
    interpolator._teardownAudioDelay = () => { audioTeardowns++; };
    const generation = interpolator._lifecycleGen;
    interpolator._installMediaBoundaryListeners(video, generation);

    assert.equal(video.style.visibility, "collapse", "pipeline setup alone is passthrough");
    assert.equal(interpolator._activateTakeover(generation), true);
    assert.equal(appended, 1);
    assert.equal(audioSetups, 1);
    assert.equal(video.style.visibility, "collapse",
      "the site-owned source style is never mutated");

    overlay.isConnected = false;
    overlay.parentNode = null;
    assert.equal(interpolator._activateTakeover(generation), true);
    assert.equal(appended, 2, "a page-detached overlay is remounted on the next successful frame");
    assert.equal(audioSetups, 1, "remounting does not duplicate audio routing");

    document.fullscreenElement = video;
    documentListeners.get("fullscreenchange")();
    assert.equal(overlay.isConnected, false,
      "video-element fullscreen safely falls back to the original source");
    assert.equal(audioTeardowns, 1);
    document.fullscreenElement = null;
    assert.equal(interpolator._activateTakeover(generation), true);
    assert.equal(appended, 3);
    assert.equal(audioSetups, 2);

    const shadowHost = {};
    const shadowRoot = {
      host: shadowHost,
      fullscreenElement: null,
      appendChild(node) { node.isConnected = true; node.parentNode = this; },
    };
    video.getRootNode = () => shadowRoot;
    document.fullscreenElement = shadowHost;
    documentListeners.get("fullscreenchange")();
    assert.equal(overlay.parentNode, shadowRoot,
      "shadow-player fullscreen mounts inside the rendered shadow tree, not unslotted light DOM");

    shadowRoot.fullscreenElement = video;
    documentListeners.get("fullscreenchange")();
    assert.equal(overlay.isConnected, false,
      "a directly-fullscreen shadow video also falls back to its original surface");

    const innerPlayer = {
      appendChild(node) { node.isConnected = true; node.parentNode = this; },
    };
    video.parentElement = innerPlayer;
    shadowRoot.fullscreenElement = innerPlayer;
    assert.equal(interpolator._activateTakeover(generation), true);
    assert.equal(overlay.parentNode, innerPlayer,
      "an actual fullscreen container inside a shadow root owns the overlay");

    assert.equal(interpolator.stop().stopped, true);
    assert.equal(video.style.visibility, "collapse");
    assert.equal(audioTeardowns, 3);
    assert.ok(removed >= 2);
    assert.ok(windowListeners.some(([, type]) => type === "scroll"));
    assert.ok(videoListeners.some(([, type]) => type === "seeking"));
  } finally {
    restore();
  }
});

test("a page-hidden source immediately relinquishes interpolation presentation", () => {
  let visible = true;
  let removed = 0;
  let teardowns = 0;
  const restore = installGlobals({
    document: {
      fullscreenElement: null,
      body: { appendChild(node) { node.isConnected = true; node.parentNode = this; } },
      addEventListener() {},
      removeEventListener() {},
    },
    window: { addEventListener() {}, removeEventListener() {} },
  });
  try {
    const video = {
      style: { visibility: "visible" },
      checkVisibility: () => visible,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
      addEventListener() {},
      removeEventListener() {},
    };
    const overlay = {
      isConnected: false,
      parentNode: null,
      style: {},
      remove() { removed++; this.isConnected = false; this.parentNode = null; },
    };
    const interpolator = makeInterpolator();
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = overlay;
    interpolator._stageAudioDelay = () => ({ bypass: true, status: "staged", rollback() {} });
    interpolator._commitAudioDelay = (transaction) => { transaction.status = "committed"; return true; };
    interpolator._teardownAudioDelay = () => { teardowns++; };

    assert.equal(interpolator._activateTakeover(interpolator._lifecycleGen), true);
    visible = false;
    assert.equal(interpolator.refreshLayout(), false);
    assert.equal(overlay.isConnected, false);
    assert.equal(video.style.visibility, "visible");
    assert.equal(teardowns, 1);
    assert.equal(removed, 1);
  } finally {
    restore();
  }
});

test("media boundaries flush before takeover and while presentation is relinquished", () => {
  const videoListeners = new Map();
  const documentListeners = new Map();
  const add = (registry, type, listener) => {
    const listeners = registry.get(type) || new Set();
    listeners.add(listener);
    registry.set(type, listeners);
  };
  const remove = (registry, type, listener) => registry.get(type)?.delete(listener);
  const emit = (registry, type) => {
    for (const listener of registry.get(type) || []) listener();
  };
  const document = {
    fullscreenElement: null,
    body: { appendChild(node) { node.isConnected = true; node.parentNode = this; } },
    addEventListener(type, listener) { add(documentListeners, type, listener); },
    removeEventListener(type, listener) { remove(documentListeners, type, listener); },
  };
  const restore = installGlobals({
    document,
    window: { addEventListener() {}, removeEventListener() {} },
  });
  try {
    const video = {
      currentSrc: "https://example.test/video.mp4",
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
      addEventListener(type, listener) { add(videoListeners, type, listener); },
      removeEventListener(type, listener) { remove(videoListeners, type, listener); },
    };
    const overlay = {
      isConnected: false,
      parentNode: null,
      style: {},
      remove() { this.isConnected = false; this.parentNode = null; },
    };
    const interpolator = makeInterpolator();
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = overlay;
    interpolator.queue = [];
    interpolator._stageAudioDelay = () => ({ bypass: true, status: "staged", rollback() {} });
    interpolator._commitAudioDelay = (transaction) => { transaction.status = "committed"; return true; };
    interpolator._teardownAudioDelay = () => {};
    let flushes = 0;
    interpolator._flush = () => { flushes++; };
    const generation = interpolator._lifecycleGen;

    interpolator._installMediaBoundaryListeners(video, generation);
    assert.equal(interpolator._takeoverActive, false);
    emit(videoListeners, "seeking");
    assert.equal(flushes, 1, "startup-owned history is flushed before first takeover");
    assert.equal(interpolator._activateTakeover(generation), false,
      "takeover remains gated until the seek completes");
    emit(videoListeners, "seeked");

    assert.equal(interpolator._activateTakeover(generation), true);
    document.fullscreenElement = video;
    emit(documentListeners, "fullscreenchange");
    assert.equal(interpolator._takeoverActive, false);
    emit(videoListeners, "seeking");
    assert.equal(flushes, 3, "fullscreen fallback retains media-boundary ownership");
    emit(videoListeners, "seeked");

    document.fullscreenElement = null;
    assert.equal(interpolator._activateTakeover(generation), true);
    assert.equal(videoListeners.get("seeking").size, 1, "reactivation never duplicates media listeners");
    interpolator.stop();
    emit(videoListeners, "seeking");
    assert.equal(flushes, 4, "stop removes the lifecycle-owned listener");
  } finally {
    restore();
  }
});

function makeAudioEventTarget(properties = {}) {
  const listeners = new Map();
  return Object.assign(properties, {
    addEventListener(type, listener) {
      const entries = listeners.get(type) || new Set();
      entries.add(listener);
      listeners.set(type, entries);
    },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    emit(type, detail = {}) {
      const event = { type, target: this, ...detail };
      for (const listener of [...(listeners.get(type) || [])]) listener(event);
    },
    listenerCount(type) { return listeners.get(type)?.size || 0; },
  });
}

function makeCapturedTrack(kind = "audio", {
  enabled = true,
  muted = false,
  readyState = "live",
  cloneable = true,
} = {}) {
  const track = makeAudioEventTarget({
    kind,
    enabled,
    muted,
    readyState,
    stopCalls: 0,
    clones: [],
    stop() {
      this.stopCalls++;
      this.readyState = "ended";
    },
  });
  if (cloneable) {
    track.clone = function clone() {
      const owned = makeCapturedTrack(kind, {
        enabled: this.enabled,
        muted: this.muted,
        readyState: this.readyState,
        cloneable: false,
      });
      owned.sourceTrack = this;
      this.clones.push(owned);
      return owned;
    };
  }
  return track;
}

function makeAudioStream(initialTracks = []) {
  const tracks = [...initialTracks];
  return makeAudioEventTarget({
    getTracks: () => [...tracks],
    getAudioTracks: () => tracks.filter((track) => track.kind === "audio"),
    addTrack(track) {
      tracks.push(track);
      this.emit("addtrack", { track });
    },
    removeTrack(track) {
      const index = tracks.indexOf(track);
      if (index >= 0) tracks.splice(index, 1);
      this.emit("removetrack", { track });
    },
  });
}

async function settleAudioPreparation() {
  for (let turn = 0; turn < 4; turn++) await Promise.resolve();
}

async function stagePrimedTakeover(harness, interpolator = harness.interpolator) {
  const generation = interpolator._lifecycleGen;
  for (let attempt = 0; attempt < 12; attempt++) {
    const takeover = interpolator._stageTakeover(generation);
    if (takeover) return takeover;
    await settleAudioPreparation();
    const transaction = interpolator._audioPreparation?.stagedTransaction;
    if (transaction?.status === "staged") {
      transaction.context.currentTime = Math.max(
        transaction.context.currentTime,
        transaction.route.primeReadyAt,
      );
    }
  }
  assert.fail("audio takeover did not become ready");
}

function makeAudioHarness({
  audioTrackCount = 1,
  audioTrackEnabled = true,
  audioTrackMuted = false,
  captureThrows = false,
  contextState = "running",
  currentSrc = "https://example.test/video.mp4",
  fakeTimeouts = false,
  hidden = false,
  initialMuted = false,
  mountFailure = false,
  nativeMutedShadow = false,
  rawStream = null,
  resume = null,
  setSinkId = null,
  sinkId = "",
  srcObject = null,
  volume = 0.5,
} = {}) {
  const calls = {
    captures: [],
    captureStream: 0,
    clearIntervals: [],
    close: 0,
    contexts: [],
    gainValues: [],
    intervals: [],
    mounts: 0,
    muteWrites: [],
    overlayRemovals: 0,
    resume: [],
    setSinkIds: [],
    timeouts: [],
    trackSources: [],
  };

  function createContext() {
    const index = calls.contexts.length;
    const state = Array.isArray(contextState) ? (contextState[index] || "running") : contextState;
    const context = makeAudioEventTarget({
      state,
      currentTime: 0,
      sinkId: "",
      destination: {},
      resume() {
        calls.resume.push(this);
        if (resume) return resume(this, index);
        this.state = "running";
        return Promise.resolve();
      },
      setSinkId(id) {
        calls.setSinkIds.push({ context: this, id });
        if (setSinkId) return setSinkId(this, id, index);
        this.sinkId = id;
        return Promise.resolve();
      },
      close() {
        calls.close++;
        this.state = "closed";
        return Promise.resolve();
      },
      createDelay() {
        return {
          delayTime: {
            value: 0,
            cancelScheduledValues() {},
            setValueAtTime(value) { this.value = value; },
            setTargetAtTime(value) { this.value = value; },
          },
          connect() {},
          disconnect() {},
        };
      },
      createGain() {
        return {
          gain: {
            value: 1,
            setValueAtTime(value) {
              this.value = value;
              calls.gainValues.push(value);
            },
          },
          connect() {},
          disconnect() {},
        };
      },
      createMediaStreamTrackSource(track) {
        const source = { context: this, track, connect() {}, disconnect() {} };
        calls.trackSources.push(source);
        return source;
      },
    });
    calls.contexts.push(context);
    return context;
  }
  class AudioContext { constructor() { return createContext(); } }

  const body = {
    appendChild(node) {
      calls.mounts++;
      if (mountFailure) throw new Error("mount failed");
      node.isConnected = true;
      node.parentNode = this;
    },
  };
  const document = makeAudioEventTarget({
    baseURI: "https://example.test/",
    fullscreenElement: null,
    pictureInPictureElement: null,
    hidden,
    body,
  });
  const window = makeAudioEventTarget({ AudioContext });
  let intervalId = 0;
  let timeoutId = 0;
  const globals = {
    document,
    window,
    setInterval(callback) {
      const id = ++intervalId;
      calls.intervals.push({ id, callback });
      return id;
    },
    clearInterval(id) { calls.clearIntervals.push(id); },
  };
  if (fakeTimeouts) {
    globals.setTimeout = (callback, ms) => {
      const timer = { id: ++timeoutId, callback, ms, active: true };
      calls.timeouts.push(timer);
      return timer.id;
    };
    globals.clearTimeout = (id) => {
      const timer = calls.timeouts.find((entry) => entry.id === id);
      if (timer) timer.active = false;
    };
  }
  if (nativeMutedShadow) {
    class HTMLMediaElement {}
    const nativeMuted = new WeakMap();
    Object.defineProperty(HTMLMediaElement.prototype, "muted", {
      configurable: true,
      get() { return nativeMuted.get(this) || false; },
      set(value) { nativeMuted.set(this, !!value); },
    });
    globals.HTMLMediaElement = HTMLMediaElement;
  }
  const restore = installGlobals(globals);

  let capture = null;
  const video = makeAudioEventTarget({
    currentSrc,
    src: "",
    srcObject,
    sinkId,
    volume,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    captureStream() {
      calls.captureStream++;
      if (captureThrows) throw new Error("capture failed");
      if (!capture) {
        const audioTracks = Array.from({ length: audioTrackCount }, (_, index) =>
          makeCapturedTrack("audio", {
            enabled: Array.isArray(audioTrackEnabled) ? !!audioTrackEnabled[index] : !!audioTrackEnabled,
            muted: Array.isArray(audioTrackMuted) ? !!audioTrackMuted[index] : !!audioTrackMuted,
          }));
        const videoTrack = makeCapturedTrack("video", { cloneable: false });
        const stream = rawStream || makeAudioStream([...audioTracks, videoTrack]);
        capture = { stream, audioTracks, videoTrack };
        calls.captures.push(capture);
      }
      return capture.stream;
    },
  });
  let muted = !!initialMuted;
  Object.defineProperty(video, "muted", {
    configurable: true,
    get() { return muted; },
    set(value) {
      muted = !!value;
      calls.muteWrites.push(muted);
    },
  });

  const interpolators = [];
  function createInterpolator(chain = null) {
    const overlay = {
      isConnected: false,
      parentNode: null,
      style: {},
      remove() {
        calls.overlayRemovals++;
        this.isConnected = false;
        this.parentNode = null;
      },
    };
    const interpolator = new Interpolator({
      chain,
      findVideo: () => null,
      log: () => {},
      warn: () => {},
    });
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = overlay;
    interpolator.queue = [];
    interpolator._flush = () => {};
    interpolators.push(interpolator);
    return interpolator;
  }
  const interpolator = createInterpolator();
  return {
    calls,
    createInterpolator,
    document,
    generation: interpolator._lifecycleGen,
    interpolator,
    get overlay() { return interpolator.overlay; },
    runActiveTimeouts() {
      for (const timer of calls.timeouts) {
        if (!timer.active) continue;
        timer.active = false;
        timer.callback();
      }
    },
    video,
    cleanup() {
      try {
        for (const instance of [...interpolators].reverse()) instance.stop();
      } finally {
        restore();
      }
    },
  };
}

function makeTakeoverGateProbe() {
  const warnings = [];
  const interpolator = new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: (message) => warnings.push(message),
  });
  const mount = {};
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator.video = {
    currentSrc: "https://example.test/video.mp4",
    src: "",
    srcObject: null,
    sinkId: "",
  };
  interpolator.overlay = {};
  interpolator._sourceCanPresent = () => true;
  interpolator._overlayMountTarget = () => mount;
  interpolator._relinquishPresentation = () => {};
  interpolator._handleAudioCaptureChange = () => {};
  interpolator._handlePipelineFailure = () => {};
  interpolator._setAudioGain = () => {};
  interpolator._stageAudioDelay = () => ({
    bypass: true,
    silent: false,
    rollback() {},
  });
  return { generation: interpolator._lifecycleGen, interpolator, warnings };
}

function installActiveAudioRoute(interpolator) {
  const session = {};
  const route = {
    silent: false,
    muteOwned: true,
    muteAccess: { read: () => true },
    snapshot: {},
    preparation: {},
    context: { state: "running" },
    session,
  };
  interpolator._takeoverActive = true;
  interpolator._audioRoute = route;
  interpolator._audioCaptureSession = session;
  interpolator._audioSourceMatches = () => true;
  interpolator._audioSinkMatches = () => true;
  interpolator._capturedAudioTracksHealthy = () => true;
  interpolator._syncOwnedAudioTracks = () => true;
  return route;
}

test("due-frame takeover diagnostics identify every rejection gate", () => {
  const cases = [
    ["lifecycle-generation-stale", ({ generation }) => generation + 1],
    ["source-video-missing", ({ interpolator }) => { interpolator.video = null; }],
    ["presentation-overlay-missing", ({ interpolator }) => { interpolator.overlay = null; }],
    ["audio-boundary-pending", ({ interpolator }) => { interpolator._audioBoundaryPending = true; }],
    ["source-not-presentable", ({ interpolator }) => { interpolator._sourceCanPresent = () => false; }],
    ["overlay-mount-unavailable", ({ interpolator }) => { interpolator._overlayMountTarget = () => null; }],
    ["silent-audio-route-invalid", ({ interpolator }) => {
      const route = installActiveAudioRoute(interpolator);
      route.silent = true;
      interpolator._silentAudioRouteValid = () => false;
    }],
    ["audio-source-changed", ({ interpolator }) => {
      installActiveAudioRoute(interpolator);
      interpolator._audioSourceMatches = () => false;
    }],
    ["audio-sink-mismatch", ({ interpolator }) => {
      installActiveAudioRoute(interpolator);
      interpolator._audioSinkMatches = () => false;
    }],
    ["audio-context-not-running", ({ interpolator }) => {
      const route = installActiveAudioRoute(interpolator);
      route.context.state = "suspended";
    }],
    ["audio-capture-session-mismatch", ({ interpolator }) => {
      installActiveAudioRoute(interpolator);
      interpolator._audioCaptureSession = {};
    }],
    ["audio-capture-session-tracks-unhealthy", ({ interpolator }) => {
      const route = installActiveAudioRoute(interpolator);
      interpolator._capturedAudioTracksHealthy = (owner) => owner !== route.session;
    }],
    ["audio-route-tracks-unhealthy", ({ interpolator }) => {
      const route = installActiveAudioRoute(interpolator);
      interpolator._capturedAudioTracksHealthy = (owner) => owner === route.session;
    }],
    ["audio-route-track-sync-failed", ({ interpolator }) => {
      installActiveAudioRoute(interpolator);
      interpolator._syncOwnedAudioTracks = () => false;
    }],
    ["media-mute-ownership-lost", ({ interpolator }) => {
      const route = installActiveAudioRoute(interpolator);
      route.muteAccess.read = () => false;
    }],
    ["audio-delay-not-ready", ({ interpolator }) => { interpolator._stageAudioDelay = () => null; }],
    ["staged-audio-context-not-running", ({ interpolator }) => {
      interpolator._stageAudioDelay = () => ({
        bypass: false,
        silent: false,
        context: { state: "suspended" },
        rollback() {},
      });
    }],
    ["inverted-chain-resume-rejected", ({ interpolator }) => {
      interpolator._chainInverted = true;
      interpolator._chainPresentationSuspended = true;
      interpolator.chain = { setInverted: () => false };
    }],
    ["inverted-chain-resume-threw", ({ interpolator }) => {
      interpolator._chainInverted = true;
      interpolator._chainPresentationSuspended = true;
      interpolator.chain = { setInverted: () => { throw new Error("resume failed"); } };
    }],
  ];

  for (const [gate, configure] of cases) {
    const probe = makeTakeoverGateProbe();
    const configuredGeneration = configure(probe);
    const generation = configuredGeneration ?? probe.generation;
    assert.equal(
      probe.interpolator._stageTakeover(generation, { diagnose: true }),
      null,
      `${gate} rejects takeover`,
    );
    assert.equal(probe.warnings.length, 1, `${gate} emits one diagnostic`);
    assert.match(
      probe.warnings[0],
      new RegExp(`^interp: takeover gated at ${gate}(?: \\(.*\\))? — due frame dropped$`),
      `${gate} is named in its diagnostic`,
    );
  }
});

test("takeover gate warnings are opt-in and rate-limited independently", () => {
  const originalNow = Date.now;
  let now = 10_000;
  Date.now = () => now;
  try {
    const { generation, interpolator, warnings } = makeTakeoverGateProbe();
    interpolator._sourceCanPresent = () => false;

    assert.equal(interpolator._stageTakeover(generation), null);
    assert.deepEqual(warnings, [], "ordinary staging remains silent");

    assert.equal(interpolator._stageTakeover(generation, { diagnose: true }), null);
    assert.equal(interpolator._stageTakeover(generation, { diagnose: true }), null);
    assert.equal(warnings.length, 1, "the same gate is suppressed within its interval");

    interpolator._sourceCanPresent = () => true;
    interpolator._audioBoundaryPending = true;
    assert.equal(interpolator._stageTakeover(generation, { diagnose: true }), null);
    assert.equal(warnings.length, 2, "a different gate has an independent limiter");

    interpolator._audioBoundaryPending = false;
    interpolator._sourceCanPresent = () => false;
    now += 4_999;
    assert.equal(interpolator._stageTakeover(generation, { diagnose: true }), null);
    assert.equal(warnings.length, 2);
    now += 1;
    assert.equal(interpolator._stageTakeover(generation, { diagnose: true }), null);
    assert.equal(warnings.length, 3, "the gate reports again once the interval elapses");
  } finally {
    Date.now = originalNow;
  }
});

test("audio preparation is asynchronous and AudioContext time alone opens the priming gate", async () => {
  const harness = makeAudioHarness({ volume: 0.35 });
  try {
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    assert.equal(harness.calls.captureStream, 1);
    assert.equal(harness.interpolator._audioPreparation.status, "pending");
    assert.equal(harness.calls.trackSources.length, 0);
    assert.equal(harness.video.muted, false);

    await settleAudioPreparation();
    assert.equal(harness.interpolator._audioPreparation.status, "ready");
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    const transaction = harness.interpolator._audioPreparation.stagedTransaction;
    const { route } = transaction;
    assert.equal(route.gain.gain.value, 0);
    assert.deepEqual(harness.calls.muteWrites, []);

    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null,
      "wall-clock turns cannot substitute for AudioContext clock progress");
    route.context.currentTime = route.primeReadyAt - 0.001;
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    route.context.currentTime = route.primeReadyAt;
    const staged = harness.interpolator._stageTakeover(harness.generation);
    assert.equal(staged.audioTransaction, transaction);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.video.muted, true);
    assert.equal(harness.calls.gainValues.at(-1), 0.35);
  } finally {
    harness.cleanup();
  }
});

test("audio preparation accepts a running context when the resume promise remains pending", async () => {
  const resumeGate = deferred();
  const harness = makeAudioHarness({
    contextState: "suspended",
    resume(context) {
      context.state = "running";
      return resumeGate.promise;
    },
  });
  try {
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    assert.equal(harness.calls.resume.length, 1);
    await settleAudioPreparation();
    assert.equal(harness.interpolator._audioPreparation.status, "ready");

    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    const transaction = harness.interpolator._audioPreparation.stagedTransaction;
    transaction.context.currentTime = transaction.route.primeReadyAt;
    const staged = harness.interpolator._stageTakeover(harness.generation);
    assert.equal(staged.audioTransaction, transaction);
    assert.equal(harness.interpolator.running, true);
  } finally {
    resumeGate.resolve();
    harness.cleanup();
  }
});

test("audio-delay diagnostics distinguish preparation from a frozen priming clock", async () => {
  const originalNow = Date.now;
  let now = 10_000;
  Date.now = () => now;
  const harness = makeAudioHarness({ volume: 0.35 });
  const warnings = [];
  harness.interpolator.warn = (message) => warnings.push(message);
  try {
    assert.equal(
      harness.interpolator._stageTakeover(harness.generation, { diagnose: true }),
      null,
    );
    assert.match(warnings.at(-1),
      /audio-delay-not-ready \(preparation-pending; context=running\)/);

    await settleAudioPreparation();
    now += 5_000;
    assert.equal(
      harness.interpolator._stageTakeover(harness.generation, { diagnose: true }),
      null,
    );
    const transaction = harness.interpolator._audioPreparation.stagedTransaction;
    assert.match(warnings.at(-1),
      /audio-delay-not-ready \(route-staged; context=running; clock=0\.000s; target=0\.125s; remaining=0\.125s\)/);

    now += 5_000;
    assert.equal(
      harness.interpolator._stageTakeover(harness.generation, { diagnose: true }),
      null,
    );
    assert.match(warnings.at(-1),
      /audio-delay-not-ready \(route-priming; context=running; clock=0\.000s; target=0\.125s; remaining=0\.125s\)/);

    transaction.context.currentTime = transaction.route.primeReadyAt;
    now += 5_000;
    const staged = harness.interpolator._stageTakeover(harness.generation, { diagnose: true });
    assert.equal(staged.audioTransaction, transaction);
    assert.equal(harness.interpolator._audioDelayPendingDetail, null);
  } finally {
    Date.now = originalNow;
    harness.cleanup();
  }
});

test("one persistent URL capture survives seek, context, sink, source, and interpolation restages", async () => {
  const harness = makeAudioHarness();
  const { interpolator, video } = harness;
  interpolator._installMediaBoundaryListeners(video, harness.generation);
  try {
    let staged = await stagePrimedTakeover(harness);
    assert.equal(interpolator._activateTakeover(harness.generation, staged), true);
    const capture = harness.calls.captures[0];
    const original = capture.audioTracks[0];
    const firstClone = original.clones[0];
    assert.equal(capture.videoTrack.stopCalls, 1, "captured video is retired immediately");
    assert.equal(original.stopCalls, 0);
    assert.equal(original.enabled, true);

    video.emit("seeking");
    assert.equal(firstClone.stopCalls, 1);
    assert.equal(original.enabled, true, "a live session keeps the persistent capturer enabled");
    video.emit("seeked");
    staged = await stagePrimedTakeover(harness);
    assert.equal(interpolator._activateTakeover(harness.generation, staged), true);

    const firstContext = harness.calls.contexts[0];
    firstContext.state = "suspended";
    firstContext.emit("statechange");
    assert.equal(video.muted, false);
    assert.equal(interpolator.running, true);
    await settleAudioPreparation();
    staged = await stagePrimedTakeover(harness);
    assert.equal(interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.calls.contexts.length, 1, "context suspension resumes the existing context");

    video.sinkId = "speaker-b";
    assert.equal(interpolator._stageTakeover(harness.generation), null);
    staged = await stagePrimedTakeover(harness);
    assert.equal(interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.calls.contexts.length, 2);
    assert.equal(harness.calls.setSinkIds.at(-1).id, "speaker-b");
    assert.equal(harness.calls.captureStream, 1, "route/context changes reuse the one native capture");

    video.currentSrc = "https://example.test/replacement.mp4";
    video.emit("loadstart");
    assert.equal(original.enabled, false, "source retirement idles rather than stops persistent audio");
    assert.equal(original.stopCalls, 0);
    original.readyState = "ended";
    capture.stream.removeTrack(original);
    const replacement = makeCapturedTrack("audio");
    capture.stream.addTrack(replacement);
    assert.equal(replacement.enabled, false, "new tracks stay idle until a lifecycle owns the record");
    video.emit("loadedmetadata");
    staged = await stagePrimedTakeover(harness);
    assert.equal(interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(replacement.enabled, true);
    assert.equal(harness.calls.captureStream, 1);

    interpolator.stop();
    assert.equal(replacement.stopCalls, 0);
    assert.equal(replacement.enabled, false, "stop releases but does not destroy cached capture audio");
    const restarted = harness.createInterpolator();
    staged = await stagePrimedTakeover(harness, restarted);
    assert.equal(restarted._activateTakeover(restarted._lifecycleGen, staged), true);
    assert.equal(harness.calls.captureStream, 1, "a new interpolation lifecycle reuses the element record");
    assert.equal(replacement.clones.length, 2, "each active route owns a fresh short-lived clone");
    restarted.stop();
    assert.ok(replacement.clones.every((track) => track.stopCalls === 1));
    assert.equal(replacement.stopCalls, 0);
    assert.equal(replacement.enabled, false);
  } finally {
    harness.cleanup();
  }
});

test("a naturally ended cached URL capture cannot silently commit a same-source replay", async () => {
  const harness = makeAudioHarness();
  harness.interpolator._installMediaBoundaryListeners(harness.video, harness.generation);
  try {
    const staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    const original = harness.calls.captures[0].audioTracks[0];

    harness.video.emit("ended");
    harness.video.emit("play");
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null,
      "an exhausted audio-bearing record must not degrade into the video-only bypass");
    assert.equal(harness.interpolator._takeoverActive, false);
    assert.equal(harness.interpolator._audioRoute, null);
    assert.equal(harness.video.muted, false);
    assert.equal(harness.calls.captureStream, 1);
    assert.equal(original.stopCalls, 0);
    await settleAudioPreparation();
    assert.equal(harness.interpolator.running, false,
      "same-source replay remains passthrough until a metadata source boundary");
    assert.equal(original.enabled, false);
  } finally {
    harness.cleanup();
  }
});

test("srcObject audio is routed directly and remains entirely page-owned", async () => {
  const sourceTrack = makeCapturedTrack("audio", { enabled: false });
  const provider = makeAudioStream([sourceTrack]);
  const harness = makeAudioHarness({ currentSrc: "", srcObject: provider });
  try {
    let staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.calls.captureStream, 0);
    assert.equal(harness.calls.trackSources[0].track, sourceTrack);
    assert.equal(sourceTrack.clones.length, 0);
    assert.equal(sourceTrack.enabled, false);

    sourceTrack.enabled = true;
    sourceTrack.muted = true;
    sourceTrack.emit("mute");
    assert.equal(harness.interpolator._takeoverActive, false);
    assert.equal(harness.video.muted, false);
    assert.equal(sourceTrack.enabled, true);
    assert.equal(sourceTrack.stopCalls, 0);
    assert.equal(harness.interpolator.running, true);

    sourceTrack.muted = false;
    sourceTrack.emit("unmute");
    staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    harness.interpolator.stop();
    assert.equal(sourceTrack.stopCalls, 0);
    assert.equal(sourceTrack.enabled, true);
    assert.equal(sourceTrack.clones.length, 0);
  } finally {
    harness.cleanup();
  }
});

test("cached capture tracks mirrored from srcObject are never modified or reused as URL audio", async () => {
  const harness = makeAudioHarness();
  try {
    let staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    harness.interpolator.stop();
    const capture = harness.calls.captures[0];
    const pageTrack = makeCapturedTrack("audio", { enabled: true });
    harness.video.srcObject = makeAudioStream([pageTrack]);
    capture.stream.addTrack(pageTrack);
    assert.equal(pageTrack.enabled, true);
    assert.equal(pageTrack.stopCalls, 0);

    const direct = harness.createInterpolator();
    staged = await stagePrimedTakeover(harness, direct);
    assert.equal(direct._activateTakeover(direct._lifecycleGen, staged), true);
    assert.equal(direct._audioRoute.audioTracks[0], pageTrack);
    direct.stop();
    assert.equal(pageTrack.stopCalls, 0);

    harness.video.srcObject = null;
    harness.video.currentSrc = "https://example.test/again.mp4";
    const pageTrackSources = harness.calls.trackSources.filter(({ track }) => track === pageTrack).length;
    const urlAgain = harness.createInterpolator();
    staged = await stagePrimedTakeover(harness, urlAgain);
    assert.equal(urlAgain._activateTakeover(urlAgain._lifecycleGen, staged), true);
    assert.ok(!urlAgain._audioCaptureSession.audioTracks.includes(pageTrack));
    assert.equal(harness.calls.trackSources.filter(({ track }) => track === pageTrack).length,
      pageTrackSources);
    assert.equal(pageTrack.enabled, true);
    assert.equal(pageTrack.stopCalls, 0);
  } finally {
    harness.cleanup();
  }
});

test("sink preparation mirrors the device and discards a stale asynchronous result", async () => {
  const firstSink = deferred();
  const harness = makeAudioHarness({
    sinkId: "speaker-a",
    setSinkId(context, id, index) {
      if (index === 0) return firstSink.promise.then(() => { context.sinkId = id; });
      context.sinkId = id;
      return Promise.resolve();
    },
  });
  try {
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    assert.equal(harness.calls.setSinkIds[0].id, "speaker-a");
    harness.video.sinkId = "speaker-b";
    firstSink.resolve();
    await settleAudioPreparation();
    assert.equal(harness.calls.contexts[0].state, "closed");
    assert.equal(harness.interpolator.running, true, "a stale sink promise is not terminal");
    assert.equal(harness.calls.captureStream, 1);

    const staged = await stagePrimedTakeover(harness);
    assert.equal(harness.calls.setSinkIds.at(-1).id, "speaker-b");
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.interpolator._audioRoute.context.sinkId, "speaker-b");
    assert.equal(harness.calls.captureStream, 1);
  } finally {
    harness.cleanup();
  }
});

test("hidden and pending AudioContext setup gate production and only visible time can expire", async () => {
  const resumeGate = deferred();
  const harness = makeAudioHarness({
    contextState: "suspended",
    fakeTimeouts: true,
    hidden: true,
    resume: () => resumeGate.promise,
  });
  try {
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    assert.equal(harness.interpolator._audioPreparation.status, "waiting-visible");
    assert.equal(harness.calls.resume.length, 0);
    assert.equal(harness.calls.timeouts.length, 0);
    assert.equal(harness.interpolator._productionEligible(), false);

    harness.document.hidden = false;
    harness.document.emit("visibilitychange");
    assert.equal(harness.interpolator._audioPreparation.status, "pending");
    assert.equal(harness.calls.resume.length, 1);
    assert.equal(harness.calls.timeouts.filter((timer) => timer.active).length, 1);

    harness.document.hidden = true;
    harness.document.emit("visibilitychange");
    harness.runActiveTimeouts();
    await settleAudioPreparation();
    assert.equal(harness.interpolator.running, true, "background time cannot consume the setup deadline");

    harness.document.hidden = false;
    harness.document.emit("visibilitychange");
    harness.runActiveTimeouts();
    await settleAudioPreparation();
    assert.equal(harness.interpolator.running, false, "visible pending setup eventually fails closed");
    assert.equal(harness.video.muted, false);
  } finally {
    resumeGate.resolve();
    harness.cleanup();
  }
});

test("muted and video-only sources bypass Web Audio then transition when audio becomes audible", async () => {
  const mutedHarness = makeAudioHarness({ initialMuted: true });
  try {
    let staged = mutedHarness.interpolator._stageTakeover(mutedHarness.generation);
    assert.equal(staged.audioTransaction.silent, true);
    assert.equal(mutedHarness.interpolator._activateTakeover(mutedHarness.generation, staged), true);
    assert.equal(mutedHarness.calls.captureStream, 0);
    assert.equal(mutedHarness.calls.contexts.length, 0);
    assert.equal(mutedHarness.video.muted, true);
    mutedHarness.video.muted = false;
    mutedHarness.video.emit("volumechange");
    assert.equal(mutedHarness.interpolator._takeoverActive, false);
    staged = await stagePrimedTakeover(mutedHarness);
    assert.equal(mutedHarness.interpolator._activateTakeover(mutedHarness.generation, staged), true);
    assert.equal(mutedHarness.calls.captureStream, 1);
    assert.equal(mutedHarness.calls.contexts.length, 1);
  } finally {
    mutedHarness.cleanup();
  }

  const videoOnly = makeAudioHarness({ audioTrackCount: 0 });
  try {
    let staged = videoOnly.interpolator._stageTakeover(videoOnly.generation);
    assert.equal(staged.audioTransaction.silent, true);
    assert.equal(videoOnly.interpolator._activateTakeover(videoOnly.generation, staged), true);
    assert.equal(videoOnly.calls.contexts.length, 0);
    const added = makeCapturedTrack("audio");
    videoOnly.calls.captures[0].stream.addTrack(added);
    assert.equal(videoOnly.interpolator._takeoverActive, false);
    staged = await stagePrimedTakeover(videoOnly);
    assert.equal(videoOnly.interpolator._activateTakeover(videoOnly.generation, staged), true);
    assert.equal(videoOnly.calls.captureStream, 1);
    assert.equal(added.clones.length, 1);
  } finally {
    videoOnly.cleanup();
  }
});

test("track mute and context suspension recover without terminal interpolation failure", async () => {
  const harness = makeAudioHarness();
  try {
    let staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    const original = harness.calls.captures[0].audioTracks[0];
    const firstClone = original.clones[0];
    original.muted = true;
    original.emit("mute");
    assert.equal(harness.interpolator._takeoverActive, false);
    assert.equal(harness.video.muted, false);
    assert.equal(firstClone.stopCalls, 1);
    assert.equal(original.stopCalls, 0);
    assert.equal(harness.interpolator.running, true);
    assert.equal(harness.interpolator._productionEligible(), false);

    original.muted = false;
    original.emit("unmute");
    staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    const context = harness.interpolator._audioRoute.context;
    context.state = "suspended";
    context.emit("statechange");
    assert.equal(harness.video.muted, false);
    assert.equal(harness.interpolator.running, true);
    await settleAudioPreparation();
    staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.calls.contexts.length, 1);
    assert.equal(harness.interpolator.running, true);
  } finally {
    harness.cleanup();
  }
});

test("page mute reclamation is terminal and closes replacement gain before native audio returns", async () => {
  const harness = makeAudioHarness({ volume: 0.25 });
  try {
    const staged = await stagePrimedTakeover(harness);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    harness.video.volume = 0.75;
    harness.video.emit("volumechange");
    assert.equal(harness.calls.gainValues.at(-1), 0.75);

    harness.video.muted = false;
    harness.video.emit("volumechange");
    assert.equal(harness.interpolator._takeoverActive, false);
    assert.equal(harness.video.muted, false);
    assert.equal(harness.calls.gainValues.at(-1), 0);
    await settleAudioPreparation();
    assert.equal(harness.interpolator.running, false);
  } finally {
    harness.cleanup();
  }
});

test("an own muted shadow is rejected before audio capture or presentation", async () => {
  const harness = makeAudioHarness({ nativeMutedShadow: true });
  try {
    assert.equal(harness.interpolator._stageTakeover(harness.generation), null);
    assert.equal(harness.calls.captureStream, 0);
    assert.equal(harness.calls.contexts.length, 0);
    assert.equal(harness.calls.mounts, 0);
    assert.deepEqual(harness.calls.muteWrites, []);
    await settleAudioPreparation();
    assert.equal(harness.interpolator.running, false);
  } finally {
    harness.cleanup();
  }
});

test("fullscreen, picture-in-picture, and audio blocks hand an inverted producer back to its renderer", () => {
  for (const mode of ["fullscreen", "picture-in-picture", "audio-blocked"]) {
    const harness = makeAudioHarness({ initialMuted: true });
    const transitions = [];
    const chain = { setInverted(value) { transitions.push(value); return true; } };
    harness.interpolator.chain = chain;
    harness.interpolator._chainInverted = true;
    harness.interpolator._chainPresentationSuspended = false;
    try {
      if (mode === "fullscreen") harness.document.fullscreenElement = harness.video;
      if (mode === "picture-in-picture") harness.document.pictureInPictureElement = harness.video;
      if (mode === "audio-blocked") harness.interpolator._setAudioBlocked("test", true);
      assert.equal(harness.interpolator._productionEligible(), false, mode);
      assert.equal(harness.interpolator._chainPresentationSuspended, true, mode);
      assert.deepEqual(transitions, [false], mode);

      harness.document.fullscreenElement = null;
      harness.document.pictureInPictureElement = null;
      harness.interpolator._setAudioBlocked("test", false);
      const staged = harness.interpolator._stageTakeover(harness.generation);
      assert.ok(staged);
      assert.equal(harness.interpolator._chainPresentationSuspended, false);
      assert.deepEqual(transitions, [false, true]);
      staged.audioTransaction.rollback();
    } finally {
      harness.cleanup();
    }
  }
});

test("a PiP event synchronously relinquishes an active inverted takeover", () => {
  const harness = makeAudioHarness({ initialMuted: true });
  const transitions = [];
  harness.interpolator.chain = {
    setInverted(value) { transitions.push(value); return true; },
  };
  harness.interpolator._chainInverted = true;
  harness.interpolator._chainPresentationSuspended = false;
  try {
    const staged = harness.interpolator._stageTakeover(harness.generation);
    assert.ok(staged);
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), true);
    assert.equal(harness.interpolator._takeoverActive, true);

    harness.document.pictureInPictureElement = harness.video;
    harness.video.emit("enterpictureinpicture");
    assert.equal(harness.interpolator._takeoverActive, false);
    assert.equal(harness.interpolator._chainPresentationSuspended, true);
    assert.deepEqual(transitions, [false]);
  } finally {
    harness.cleanup();
  }
});

test("early activation invalidation rolls back the staged route without claiming native audio", async () => {
  const harness = makeAudioHarness();
  try {
    const staged = await stagePrimedTakeover(harness);
    const transaction = staged.audioTransaction;
    const original = harness.calls.captures[0].audioTracks[0];
    const clone = original.clones[0];
    harness.interpolator._lifecycleGen++;
    assert.equal(harness.interpolator._activateTakeover(harness.generation, staged), false);
    assert.equal(transaction.status, "rolled-back");
    assert.equal(clone.stopCalls, 1);
    assert.equal(original.stopCalls, 0);
    assert.equal(harness.video.muted, false);
    assert.deepEqual(harness.calls.muteWrites, []);
    assert.equal(harness.calls.mounts, 0);
  } finally {
    harness.cleanup();
  }
});

test("a takeover-staging exception releases the shifted presentation item exactly once", async () => {
  let nextFrame = null;
  let closes = 0;
  let stageOptions = null;
  const restore = installGlobals({
    requestAnimationFrame(callback) { nextFrame = callback; return 1; },
    cancelAnimationFrame() {},
  });
  try {
    const interpolator = makeInterpolator();
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = {};
    interpolator.overlay = { style: {}, remove() {} };
    interpolator._targetInterval = 100;
    interpolator.queue = [{
      bmp: { width: 16, height: 9, close() { closes++; } },
      ts: 0,
      enq: performance.now(),
    }];
    interpolator._stageTakeover = (_generation, options) => {
      stageOptions = options;
      throw new Error("host getter failed");
    };

    interpolator._present(interpolator._lifecycleGen);
    nextFrame(performance.now());
    await Promise.resolve();

    assert.equal(closes, 1);
    assert.equal(interpolator.running, false);
    assert.deepEqual(stageOptions, { diagnose: true }, "only the due-frame drain opts into diagnostics");
  } finally {
    restore();
  }
});

test("consecutive pipeline failures stop once, while success resets the breaker", async () => {
  const warnings = [];
  const interpolator = new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: (message) => warnings.push(message),
  });
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  let overlayRemoved = 0;
  interpolator.overlay = { remove() { overlayRemoved++; } };
  interpolator._takeoverActive = true;
  interpolator._committedPresentation = { committed: true, generation: 3 };
  let failureSnapshot = null;
  interpolator.onTerminalFailure = () => {
    failureSnapshot = interpolator.getStats();
  };
  const generation = interpolator._lifecycleGen;
  let stops = 0;
  interpolator.stop = () => {
    stops++;
    interpolator._lifecycleGen++;
    interpolator._stopped = true;
  };

  for (let i = 0; i < 4; i++) {
    assert.equal(interpolator._handlePipelineFailure(generation, new Error("tainted"), "capture"), false);
  }
  interpolator._recordPipelineSuccess("capture");
  for (let i = 0; i < 4; i++) {
    assert.equal(interpolator._handlePipelineFailure(generation, new Error("tainted"), "capture"), false);
  }
  assert.equal(stops, 0);
  assert.equal(interpolator._handlePipelineFailure(generation, new Error("tainted"), "capture"), true);
  assert.equal(overlayRemoved, 1, "terminal failure restores the native presentation synchronously");
  assert.equal(interpolator.getStats().takeoverActive, false);
  assert.equal(interpolator.getStats().presentation, null);
  assert.equal(failureSnapshot.takeoverActive, false,
    "terminal diagnostics are emitted only after stale takeover state is cleared");
  assert.equal(failureSnapshot.presentation, null);
  assert.equal(interpolator._handlePipelineFailure(generation, new Error("duplicate"), "capture"), true);
  await Promise.resolve();
  assert.equal(stops, 1);
  assert.equal(warnings.length, 1);
});

test("blend cadence is bounded and derives from source timestamps", () => {
  const interpolator = makeInterpolator();
  interpolator.setTargetFps(120);

  assert.equal(interpolator._tweensForGap(0, 1_000_000 / 30), 3);
  assert.equal(interpolator._tweensForGap(0, 1_000_000), 7);
  assert.equal(interpolator.setTargetFps(10), 24);
  assert.equal(interpolator.setTargetFps(1000), 480);
  assert.equal(interpolator.setTargetFps("auto"), "auto");
});

test("RIFE model selection rejects unknown keys and updates the public inventory", () => {
  const initial = listModels();
  const original = initial.find((model) => model.current)?.key;
  assert.ok(original);
  assert.equal(setModel("missing-model"), false);

  const alternate = initial.find((model) => model.key !== original)?.key;
  assert.ok(alternate);
  assert.equal(setModel(alternate), true);
  assert.equal(listModels().find((model) => model.current)?.key, alternate);
  assert.equal(setModel(original), true);
});

test("stale inverted resize callbacks always release the restart guard", async () => {
  const interpolator = makeInterpolator();
  interpolator._stopped = false;
  const generation = interpolator._lifecycleGen;

  assert.equal(interpolator._scheduleDimsRestart(generation, 1920, 1080), true);
  assert.equal(interpolator._dimsRestarting, true);
  interpolator._lifecycleGen++;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(interpolator._dimsRestarting, false);

  interpolator._dimsRestarting = true;
  assert.deepEqual(interpolator.stop(), { ok: true, stopped: false });
  assert.equal(interpolator._dimsRestarting, false);
});

test("GPU capture resource limits schedule one stop and a newer lifecycle wins", async () => {
  const warnings = [];
  const interpolator = new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: (message) => warnings.push(message),
  });
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const generation = interpolator._lifecycleGen;
  interpolator._rifeMod = {
    gpuLastCaptureError: () => Object.assign(new Error("frame pair too large"), { code: "GPU_RESOURCE_LIMIT" }),
  };
  let stops = 0;
  interpolator.stop = () => { stops++; interpolator._lifecycleGen++; interpolator._stopped = true; };

  assert.equal(interpolator._handleGpuCaptureFailure(generation), true);
  assert.equal(interpolator._handleGpuCaptureFailure(generation), true);
  await Promise.resolve();
  assert.equal(stops, 1);
  assert.equal(warnings.length, 1);

  interpolator._stopped = false;
  interpolator._gpuResourceStopQueued = false;
  const staleGeneration = interpolator._lifecycleGen;
  assert.equal(interpolator._handleGpuCaptureFailure(staleGeneration), true);
  interpolator._lifecycleGen++;
  await Promise.resolve();
  assert.equal(stops, 1, "a user-owned newer lifecycle must cancel the queued stop");
});

test("transient GPU budget pressure drops a capture without stopping", async () => {
  const warnings = [];
  const interpolator = new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: (message) => warnings.push(message),
  });
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._rifeMod = {
    gpuLastCaptureError: () => Object.assign(new Error("waiting for queue fence"), {
      code: "GPU_RESOURCE_LIMIT",
      details: { transient: true },
    }),
  };
  let stops = 0;
  interpolator.stop = () => { stops++; };

  const generation = interpolator._lifecycleGen;
  for (let attempt = 0; attempt < interpolator._pipelineFailureLimit + 2; attempt++) {
    if (!interpolator._handleGpuCaptureFailure(generation)) {
      interpolator._handlePipelineFailure(
        generation,
        interpolator._rifeMod.gpuLastCaptureError(),
        "capture",
      );
    }
  }
  await Promise.resolve();
  assert.equal(stops, 0);
  assert.equal(warnings.length, 0);
  assert.equal(interpolator._pipelineFailureStreaks.capture ?? 0, 0);
  assert.equal(interpolator._gpuResourceStopQueued, false);
});

test("stale CPU tween completion closes every owned bitmap without enqueueing", () => {
  const interpolator = makeInterpolator();
  interpolator._stopped = false;
  interpolator._lifecycleGen = 4;
  let enqueues = 0;
  interpolator._enqueue = () => { enqueues++; };
  const closed = { tween: 0, current: 0, lookahead: 0 };
  const cur = {
    bmp: { close() { closed.current++; } },
    prevBmp: { close() { closed.lookahead++; } },
  };
  const tween = { close() { closed.tween++; } };
  const stats = { framesOut: 7 };

  assert.equal(interpolator._commitCpuTweenBitmap(3, cur, tween, 123, stats), false);
  assert.deepEqual(closed, { tween: 1, current: 1, lookahead: 1 });
  assert.equal(enqueues, 0);
  assert.equal(stats.framesOut, 7);
});

test("a source flush invalidates CPU tween completion within the same lifecycle", () => {
  const interpolator = makeInterpolator();
  interpolator._stopped = false;
  interpolator._lifecycleGen = 4;
  interpolator._flushGen = 9;
  let enqueues = 0;
  interpolator._enqueue = () => { enqueues++; };
  const closed = { tween: 0, current: 0, lookahead: 0 };
  const cur = {
    bmp: { close() { closed.current++; } },
    prevBmp: { close() { closed.lookahead++; } },
  };
  const tween = { close() { closed.tween++; } };
  const stats = { framesOut: 2 };

  assert.equal(interpolator._commitCpuTweenBitmap(4, cur, tween, 123, stats, 8), false);
  assert.deepEqual(closed, { tween: 1, current: 1, lookahead: 1 });
  assert.equal(enqueues, 0);
  assert.equal(stats.framesOut, 2);
});

test("device-loss restart is single-flight and a newer user stop wins", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  let starts = 0;
  let stops = 0;
  interpolator.start = async () => { starts++; return { ok: true }; };
  interpolator.stop = () => {
    stops++;
    interpolator.running = false;
    interpolator._state = "idle";
    interpolator._stopped = true;
    interpolator._lifecycleGen++;
    return { ok: true };
  };

  const lostDevice = {};
  assert.equal(interpolator._handleRifeDeviceLoss(lostDevice, { message: "reset" }), true);
  assert.equal(interpolator._handleRifeDeviceLoss(lostDevice, { message: "duplicate" }), false);
  // Simulate an explicit off request before the queued recovery microtask.
  interpolator.stop();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(stops, 1);
  assert.equal(starts, 0, "the stale loss callback must not resurrect interpolation");
  assert.equal(interpolator._deviceRestarting, false);
});

test("device-loss restart waits for the old CPU grabber device to finish destroying", async () => {
  const interpolator = makeInterpolator();
  const destruction = deferred();
  const events = [];
  Object.assign(interpolator, {
    running: true,
    _state: "running",
    _stopped: false,
    video: {},
    overlay: { remove() {} },
    queue: [],
    _gpuGrab: {
      ready: true,
      destroy() {
        events.push("grab-destroy-start");
        this.ready = false;
        return destruction.promise.then(() => events.push("grab-destroy-end"));
      },
    },
  });
  interpolator._startInternal = async () => {
    events.push("restart");
    return { ok: true };
  };

  assert.equal(interpolator._handleRifeDeviceLoss({}, { message: "GPU reset" }), true);
  await waitFor(() => events.includes("grab-destroy-start"),
    "device-loss recovery did not stop the old CPU grabber");
  assert.deepEqual(events, ["grab-destroy-start"]);

  destruction.resolve();
  await waitFor(() => events.includes("restart") && !interpolator._deviceRestarting,
    "device-loss recovery did not restart after CPU grabber teardown");
  assert.deepEqual(events, ["grab-destroy-start", "grab-destroy-end", "restart"]);
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._state, "running");
});

test("loss of a replacement device queues one subsequent interpolation recovery", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const firstStartEntered = deferred();
  const finishFirstStart = deferred();
  const productionStop = interpolator.stop.bind(interpolator);
  let starts = 0;
  let stops = 0;
  interpolator.stop = (options) => {
    stops++;
    return productionStop(options);
  };
  interpolator.start = async () => {
    starts++;
    const generation = ++interpolator._lifecycleGen;
    interpolator._stopped = false;
    interpolator.running = true;
    interpolator._state = "starting";
    if (starts === 1) {
      firstStartEntered.resolve();
      await finishFirstStart.promise;
    }
    const current = interpolator._isCurrent(generation);
    if (current) interpolator._state = "running";
    return { ok: current };
  };

  const firstDevice = {};
  const replacementDevice = {};
  assert.equal(interpolator._handleRifeDeviceLoss(firstDevice, { message: "first reset" }), true);
  await firstStartEntered.promise;
  assert.equal(interpolator._handleRifeDeviceLoss(replacementDevice, { message: "replacement reset" }), true);
  assert.equal(interpolator._handleRifeDeviceLoss(replacementDevice, { message: "duplicate" }), false);

  finishFirstStart.resolve();
  await waitFor(() => starts === 2 && !interpolator._deviceRestarting, "queued replacement recovery did not finish");
  assert.equal(starts, 2);
  assert.equal(stops, 2);
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._pendingDeviceLoss, null);
});

test("user stop cancels a replacement-device loss queued during recovery", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const firstStartEntered = deferred();
  const finishFirstStart = deferred();
  const productionStop = interpolator.stop.bind(interpolator);
  let starts = 0;
  let stops = 0;
  interpolator.stop = (options) => {
    stops++;
    return productionStop(options);
  };
  interpolator.start = async () => {
    starts++;
    const generation = ++interpolator._lifecycleGen;
    interpolator._stopped = false;
    interpolator.running = true;
    interpolator._state = "starting";
    firstStartEntered.resolve();
    await finishFirstStart.promise;
    const current = interpolator._isCurrent(generation);
    if (current) interpolator._state = "running";
    return { ok: current };
  };

  assert.equal(interpolator._handleRifeDeviceLoss({}, { message: "first reset" }), true);
  await firstStartEntered.promise;
  assert.equal(interpolator._handleRifeDeviceLoss({}, { message: "replacement reset" }), true);
  interpolator.stop();
  finishFirstStart.resolve();

  await waitFor(() => !interpolator._deviceRestarting, "cancelled recovery did not unwind");
  assert.equal(starts, 1);
  assert.equal(stops, 2);
  assert.equal(interpolator.running, false);
  assert.equal(interpolator._pendingDeviceLoss, null);
});

test("CPU grab initialization waits for unavailable-grabber destruction", async () => {
  const state = { instances: [], events: [], initCalls: 0, destroyCalls: 0 };
  const cleanup = installCpuGrabModule(state);
  const destruction = deferred();
  try {
    const interpolator = makeInterpolator();
    const oldGrabber = {
      ready: false,
      destroy() {
        state.events.push("old-destroy-start");
        return destruction.promise.then(() => state.events.push("old-destroy-end"));
      },
    };
    interpolator._gpuGrab = oldGrabber;
    interpolator._stopped = false;
    const generation = interpolator._lifecycleGen;

    const first = interpolator._ensureCpuGrabber(generation);
    const second = interpolator._ensureCpuGrabber(generation);
    assert.equal(second, first, "teardown-gated initialization must stay single-flight");
    await Promise.resolve();
    assert.equal(state.initCalls, 0);
    assert.deepEqual(state.events, ["old-destroy-start"]);

    destruction.resolve();
    assert.equal(await first, true);
    assert.equal(state.initCalls, 1);
    assert.equal(interpolator._gpuGrab, state.instances[0]);
    assert.deepEqual(state.events.slice(0, 4), [
      "old-destroy-start", "old-destroy-end", "construct", "init",
    ]);
  } finally {
    cleanup();
  }
});

test("CPU grab device-loss recovery waits for physical destruction before replacement init", async () => {
  const state = { instances: [], events: [], initCalls: 0, destroyCalls: 0 };
  const cleanup = installCpuGrabModule(state);
  const destruction = deferred();
  try {
    const interpolator = makeInterpolator();
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator._gpuPresent = false;
    interpolator._cpuGrabRecoveryDelays = [0];
    const generation = interpolator._lifecycleGen;
    let destroyCalls = 0;
    const lostGrabber = {
      ready: false,
      destroy() {
        destroyCalls++;
        state.events.push("lost-destroy-start");
        return destruction.promise.then(() => state.events.push("lost-destroy-end"));
      },
    };
    interpolator._gpuGrab = lostGrabber;

    assert.equal(interpolator._handleCpuGrabberDeviceLoss(
      lostGrabber, generation, { message: "GPU reset" },
    ), true);
    assert.equal(interpolator._gpuGrab, null);
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(destroyCalls, 1);
    assert.equal(state.initCalls, 0,
      "recovery must not request a replacement device while the lost device drains");

    destruction.resolve();
    await waitFor(() => interpolator._gpuGrab === state.instances[0],
      "CPU grab recovery did not publish its replacement");
    assert.equal(state.initCalls, 1);
    assert.deepEqual(state.events.slice(0, 4), [
      "lost-destroy-start", "lost-destroy-end", "construct", "init",
    ]);
    assert.equal(interpolator._cpuGrabRecovery, null);
    assert.equal(interpolator._cpuGrabberTeardowns.size, 0);
  } finally {
    cleanup();
  }
});

test("CPU grab device loss replaces only the current grabber in one recovery flight", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  const generation = interpolator._lifecycleGen;
  const current = { ready: true };
  const replacement = { ready: true };
  interpolator._gpuGrab = current;
  let attempts = 0;
  interpolator._ensureCpuGrabber = async (requestedGeneration) => {
    attempts++;
    assert.equal(requestedGeneration, generation);
    interpolator._gpuGrab = replacement;
    return true;
  };

  assert.equal(interpolator._handleCpuGrabberDeviceLoss({}, generation, { message: "stale" }), false);
  assert.equal(interpolator._gpuGrab, current);
  assert.equal(interpolator._handleCpuGrabberDeviceLoss(current, generation, { message: "reset" }), true);
  assert.equal(interpolator._handleCpuGrabberDeviceLoss(current, generation, { message: "duplicate" }), false);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(attempts, 1);
  assert.equal(interpolator._gpuGrab, replacement);
  assert.equal(interpolator._cpuGrabRecovery, null);
});

test("CPU grab recovery is bounded and a newer lifecycle cancels remaining attempts", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  const generation = interpolator._lifecycleGen;
  let attempts = 0;
  interpolator._ensureCpuGrabber = async () => { attempts++; return false; };

  const first = interpolator._scheduleCpuGrabberRecovery(generation);
  assert.equal(interpolator._scheduleCpuGrabberRecovery(generation), first);
  assert.equal(await first, false);
  assert.equal(attempts, 3);

  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  interpolator._ensureCpuGrabber = async () => { attempts++; await gate; return false; };
  const cancelled = interpolator._scheduleCpuGrabberRecovery(generation);
  interpolator._lifecycleGen++;
  interpolator._stopped = true;
  release();
  assert.equal(await cancelled, false);
  assert.equal(attempts, 4, "no retry may survive the lifecycle that scheduled it");
});

test("the presentation watchdog reports a true cumulative stall and never fakes a present", () => {
  let nextFrame = null;
  let clock = 100_000;
  const warnings = [];
  const restore = installGlobals({
    requestAnimationFrame(callback) { nextFrame = callback; return 1; },
    cancelAnimationFrame() {},
    document: { fullscreenElement: null, body: {} },
    performance: { now: () => clock },
  });
  try {
    const video = { style: { visibility: "visible" }, playbackRate: 1 };
    const interpolator = new Interpolator({
      findVideo: () => video,
      log: () => {},
      warn: (message) => warnings.push(String(message)),
    });
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = { style: {}, remove() {} };
    interpolator._rifeMod = { gpuRelease() {} };
    interpolator._targetInterval = 16.7;
    // Reproduces the observed production shape: frames are queued and due, but the
    // display-takeover gate refuses every one of them without raising.
    interpolator._stageTakeover = () => null;
    const enqueue = () => {
      interpolator.queue = Array.from({ length: 4 }, (_unused, index) => ({
        tex: { _w: 16, _h: 9 }, ts: index * 16_700, enq: clock,
      }));
    };

    enqueue();
    interpolator._lastPresentAt = clock - 2000;
    interpolator._present(interpolator._lifecycleGen);

    nextFrame(clock);
    const presentAtAfterFirstStall = interpolator._lastPresentAt;

    enqueue();
    clock += 6000;               // clears the 5 s log throttle
    nextFrame(clock);

    assert.equal(
      interpolator._lastPresentAt,
      presentAtAfterFirstStall,
      "only a committed presentation may advance _lastPresentAt",
    );
    const stalls = warnings.filter((message) => message.includes("present stalled"));
    assert.equal(stalls.length, 2);
    assert.match(stalls[0], /present stalled 2000ms/);
    assert.match(
      stalls[1],
      /present stalled 8000ms/,
      "a continuing stall must report total elapsed time, not the log-throttle interval",
    );
  } finally {
    restore();
  }
});

test("presentation deadlines follow playbackRate so fast playback cannot outrun the schedule", () => {
  let nextFrame = null;
  let clock = 100_000;
  const restore = installGlobals({
    requestAnimationFrame(callback) { nextFrame = callback; return 1; },
    cancelAnimationFrame() {},
    document: { fullscreenElement: null, body: {} },
    performance: { now: () => clock },
  });
  try {
    const presented = [];
    const video = { style: { visibility: "visible" }, playbackRate: 2 };
    const interpolator = new Interpolator({ findVideo: () => video, log: () => {}, warn: () => {} });
    interpolator.running = true;
    interpolator._state = "running";
    interpolator._stopped = false;
    interpolator.video = video;
    interpolator.overlay = { style: {}, remove() {} };
    interpolator._rifeMod = {
      gpuRelease() {},
      gpuPresent(tex) { presented.push(tex.ts); return true; },
    };
    interpolator._targetInterval = 16.7;
    interpolator._stageTakeover = () => ({ mount: {}, audioTransaction: null });
    interpolator._activateTakeover = () => true;
    interpolator._recordCommittedPresentation = () => {};
    // Four frames spanning 150 ms of source time.
    interpolator.queue = [0, 50, 100, 150].map((offsetMs) => ({
      tex: { _w: 16, _h: 9, ts: offsetMs }, ts: offsetMs * 1000, enq: clock,
    }));

    interpolator._present(interpolator._lifecycleGen);
    nextFrame(clock);            // anchors at source 0 ms
    clock += 50;                 // 50 ms of wall clock at rate 2 == 100 ms of source
    nextFrame(clock);

    assert.deepEqual(
      presented,
      [0, 50, 100],
      "at rate 2, 50 ms of wall clock must retire 100 ms of source frames",
    );
  } finally {
    restore();
  }
});

test("a rate change flushes the buffer whose deadlines were mapped at the old rate", () => {
  const listeners = new Map();
  let flushes = 0;
  const video = {
    addEventListener(type, listener) { listeners.set(type, listener); },
    removeEventListener(type) { listeners.delete(type); },
  };
  const interpolator = makeInterpolator();
  interpolator.video = video;
  interpolator._flush = () => { flushes++; };
  interpolator._installMediaBoundaryListeners(video, interpolator._lifecycleGen);

  assert.ok(listeners.has("ratechange"), "ratechange must be a media boundary");
  listeners.get("ratechange")();
  assert.equal(flushes, 1);
});

test("a discontinuity clears the breaker streak and the stall clock together", () => {
  const interpolator = makeInterpolator();
  interpolator._tweenFailStreak = 4;
  interpolator._stallSince = 1234;

  interpolator._resetFailureStreaks();

  assert.equal(interpolator._tweenFailStreak, 0,
    "a stale streak would re-trip the breaker on the first failure after a rebuffer");
  assert.equal(interpolator._stallSince, null);
});

test("restoring RIFE releases the pipelined frame the blend path never advanced", () => {
  const released = [];
  const interpolator = makeInterpolator();
  interpolator._interpMode = "blend";
  interpolator._forceBlend = false;
  interpolator._rifeMod = {
    gpuRelease(tex) { released.push(tex); },
    gpuRifeCapable: () => true,
  };
  const stale = { _w: 16, _h: 9 };
  interpolator._pipePrev = { tex: stale, ts: 0 };

  interpolator.setAutoFallback(false);

  assert.equal(interpolator._interpMode, "rife");
  assert.equal(interpolator._pipePrev, null,
    "a stale pre-blend frame would be paired with a live frame across the mode flip");
  assert.deepEqual(released, [stale]);
});

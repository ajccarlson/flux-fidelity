import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const mainUrl = new URL("../src/core/fsrcnnx-main.js", import.meta.url);
let revision = 0;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

async function flushUntil(predicate, label) {
  for (let turn = 0; turn < 30; turn++) {
    if (predicate()) return;
    await Promise.resolve();
  }
  assert.fail(`timed out waiting for ${label}`);
}

function controlledBridge({
  startGate = null,
  attachGate = null,
  initGate = null,
  runGate = null,
} = {}) {
  const calls = {
    start: 0,
    attach: 0,
    init: 0,
    run: 0,
    cancel: 0,
    stop: 0,
    dispose: 0,
  };
  let callbacks = null;
  const bridge = {
    state: "idle",
    runPending: false,
    start() {
      calls.start++;
      bridge.state = "connecting";
      const operation = startGate?.promise ?? Promise.resolve({ connected: true });
      return operation.then((result) => {
        bridge.state = "ready";
        return result;
      });
    },
    attachCanvas() {
      calls.attach++;
      return attachGate?.promise ?? Promise.resolve({ attached: true });
    },
    init(modelKey) {
      calls.init++;
      const operation = initGate?.promise ?? Promise.resolve({
        model: { key: modelKey, label: modelKey, scale: 2 },
      });
      return operation;
    },
    run() {
      calls.run++;
      return runGate?.promise ?? Promise.resolve({
        presentation: { output: { width: 2, height: 2 } },
      });
    },
    cancel() {
      calls.cancel++;
      return bridge.state !== "failed" && bridge.state !== "disposed";
    },
    stop() {
      calls.stop++;
      return Promise.resolve({ stopped: true, stats: { engine: { n: 0 } } });
    },
    dispose() {
      calls.dispose++;
      bridge.state = "disposed";
      return Promise.resolve({ disposed: true });
    },
  };
  return {
    bridge,
    calls,
    install(nextCallbacks) { callbacks = nextCallbacks; },
    fail(error) {
      bridge.state = "failed";
      callbacks.onStateChange({ previous: "ready", state: "failed", error });
    },
  };
}

async function loadAdapter(control) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "function emptyNeuralStats()",
    "function neuralOutputCanvas()",
  );
  const deps = {
    frameFailures: [],
    deviceLosses: [],
    warnings: [],
    canvasRemovals: 0,
    outputCanvases: [],
    inputCanvases: [],
    bridgeCreations: 0,
    createBridge(callbacks) {
      const controls = Array.isArray(control) ? control : [control];
      const selected = controls[deps.bridgeCreations++];
      assert.ok(selected, "unexpected Neural bridge creation");
      selected.install(callbacks);
      return selected.bridge;
    },
  };
  deps.OffscreenCanvas = class {
    constructor(width, height) {
      this.width = width;
      this.height = height;
      this.context = {
        drawImage() {},
        imageSmoothingEnabled: true,
      };
      deps.inputCanvases.push(this);
    }
    getContext() { return this.context; }
    transferToImageBitmap() {
      return {
        width: this.width,
        height: this.height,
        closes: 0,
        close() { this.closes++; },
      };
    }
  };
  globalThis.__neuralMainAdapterDeps = deps;
  globalThis.OffscreenCanvas = deps.OffscreenCanvas;
  const harness = `
    const deps = globalThis.__neuralMainAdapterDeps;
    const log = () => {};
    const warn = (...args) => deps.warnings.push(args);
    const SRGB_COLOR_SPACE = "srgb";
    const createImageBitmap = async () => ({ close() {} });
    const createNeuralOutputCanvas = () => {
      const canvas = {
        width: 1,
        height: 1,
        remove() { deps.canvasRemovals++; },
      };
      deps.outputCanvases.push(canvas);
      return canvas;
    };
    const createNeuralFrameBridge = (options) => deps.createBridge(options);
    const handleNeuralFrameDeviceLoss = (error) => deps.deviceLosses.push(error);
    const handleNeuralFrameFailure = (error) => deps.frameFailures.push(error);
    ${production}
    export const create = () => createEmbeddedNeuralEngine({ log, warn });
  `;
  const module = await import(
    `data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`
  );
  return { engine: module.create(), deps };
}

test("Neural adapter stop fences handshake, attachment, and remote initialization", async (t) => {
  await t.test("stop before the queued handshake prevents transport creation", async () => {
    const control = controlledBridge();
    const { engine } = await loadAdapter(control);
    const initializing = engine.init("model-a");
    const stopping = engine.stop();

    await assert.rejects(initializing, (error) => error.code === "NEURAL_SUPERSEDED");
    await stopping;
    assert.deepEqual(control.calls, {
      start: 0,
      attach: 0,
      init: 0,
      run: 0,
      cancel: 0,
      stop: 0,
      dispose: 0,
    });
  });

  await t.test("stop during the frame handshake retires the attached remote", async () => {
    const startGate = deferred();
    const attachGate = deferred();
    const control = controlledBridge({ startGate, attachGate });
    const { engine } = await loadAdapter(control);
    const initializing = engine.init("model-b");
    await flushUntil(() => control.calls.start === 1, "frame handshake");

    const stopping = engine.stop();
    startGate.resolve({ connected: true });
    await flushUntil(() => control.calls.attach === 1, "canvas attachment");
    attachGate.resolve({ attached: true });

    await assert.rejects(initializing, (error) => error.code === "NEURAL_SUPERSEDED");
    await stopping;
    assert.equal(control.calls.init, 0);
    assert.equal(control.calls.stop, 1);
    assert.equal(control.calls.dispose, 1);
    assert.equal(engine.ready(), false);
  });

  await t.test("stop during attachment cannot send a later remote init", async () => {
    const attachGate = deferred();
    const control = controlledBridge({ attachGate });
    const { engine } = await loadAdapter(control);
    const initializing = engine.init("model-c");
    await flushUntil(() => control.calls.attach === 1, "canvas attachment");

    const stopping = engine.stop();
    attachGate.resolve({ attached: true });

    await assert.rejects(initializing, (error) => error.code === "NEURAL_SUPERSEDED");
    await stopping;
    assert.equal(control.calls.init, 0);
    assert.equal(control.calls.stop, 1);
    assert.equal(control.calls.dispose, 1);
    assert.equal(engine.ready(), false);
  });

  await t.test("stop during remote init waits and then releases the initialized session", async () => {
    const initGate = deferred();
    const control = controlledBridge({ initGate });
    const { engine } = await loadAdapter(control);
    const initializing = engine.init("model-d");
    await flushUntil(() => control.calls.init === 1, "remote initialization");

    const stopping = engine.stop();
    initGate.resolve({
      model: { key: "model-d", label: "Model D", scale: 2 },
      stats: { engine: { n: 0 } },
    });

    await assert.rejects(initializing, (error) => error.code === "NEURAL_SUPERSEDED");
    await stopping;
    assert.equal(control.calls.stop, 1);
    assert.equal(control.calls.dispose, 1);
    assert.equal(engine.ready(), false);
    assert.equal(engine.activeEntry(), null);
  });
});

test("Neural adapter stop cancels active work and releases both page-side canvas backings", async () => {
  const runGate = deferred();
  const control = controlledBridge({ runGate });
  const { engine, deps } = await loadAdapter(control);
  await engine.init("model-active");
  const outputCanvas = engine.canvas();
  outputCanvas.width = 640;
  outputCanvas.height = 360;

  const running = engine.run({}, 320, 180, {
    width: 640,
    height: 360,
  });
  await flushUntil(() => control.calls.run === 1, "remote run");
  assert.equal(deps.inputCanvases.length, 1);
  const inputCanvas = deps.inputCanvases[0];
  assert.equal(inputCanvas.width, 320);
  assert.equal(inputCanvas.height, 180);

  const stopping = engine.stop();
  assert.equal(control.calls.cancel, 1);
  assert.equal(inputCanvas.width, 0);
  assert.equal(inputCanvas.height, 0);
  assert.equal(outputCanvas.width, 0);
  assert.equal(outputCanvas.height, 0);
  assert.equal(
    control.calls.stop,
    0,
    "remote cleanup waits for the active run to settle",
  );

  const cancellation = new Error("injected remote cancellation");
  cancellation.code = "cancelled";
  runGate.reject(cancellation);
  await assert.rejects(running, (error) => error === cancellation);
  await stopping;

  assert.equal(control.calls.stop, 1);
  assert.equal(control.calls.dispose, 1);
  assert.equal(engine.canvas(), null);
  assert.equal(deps.canvasRemovals, 1);
  assert.equal(engine.ready(), false);
});

test("Neural adapter recreates a fresh frame transport after stop", async () => {
  const first = controlledBridge();
  const second = controlledBridge();
  const { engine, deps } = await loadAdapter([first, second]);

  assert.equal((await engine.init("model-first")).key, "model-first");
  const firstCanvas = engine.canvas();
  await engine.stop();
  assert.equal(first.calls.stop, 1);
  assert.equal(first.calls.dispose, 1);
  assert.equal(firstCanvas.width, 0);
  assert.equal(firstCanvas.height, 0);

  assert.equal((await engine.init("model-second")).key, "model-second");
  assert.equal(deps.bridgeCreations, 2);
  assert.notStrictEqual(engine.canvas(), firstCanvas);
  assert.equal(second.calls.start, 1);
  assert.equal(second.calls.attach, 1);
  assert.equal(second.calls.init, 1);
  assert.equal(engine.ready(), true);

  await engine.dispose();
  assert.equal(second.calls.dispose, 1);
  assert.equal(deps.canvasRemovals, 2);
});

test("a terminal failure from an active Neural frame is surfaced immediately", async () => {
  const control = controlledBridge();
  const { engine, deps } = await loadAdapter(control);
  await engine.init("model-active");
  assert.equal(engine.ready(), true);

  const failure = new Error("Neural frame run timed out");
  control.fail(failure);

  assert.equal(engine.ready(), false);
  assert.equal(engine.activeEntry(), null);
  assert.deepEqual(deps.frameFailures, [failure]);
  await engine.stop();
  assert.equal(control.calls.dispose, 1);
  assert.equal(deps.canvasRemovals, 1);
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  NEURAL_FRAME_CHANNEL,
  NeuralFrameBridgeError,
  createNeuralFrameBridge,
} from "../src/core/fsrcnnx-neural-frame-bridge.js";

const FRAME_CAPABILITY = "1234567890abcdef1234567890abcdef1234567890abcdef";

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }
  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }
  emit(type, event = {}) {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener(event);
  }
}

class FakeStyle {
  constructor() {
    this.values = new Map();
  }
  setProperty(property, value, priority) {
    this.values.set(property, { value, priority });
  }
}

class FakeNode extends FakeEventTarget {
  constructor(tagName, harness) {
    super();
    this.tagName = tagName.toUpperCase();
    this.harness = harness;
    this.attributes = new Map();
    this.children = [];
    this.parentNode = null;
    this.style = new FakeStyle();
    this._documentRoot = false;
  }
  get isConnected() {
    if (this._documentRoot) return true;
    if (this instanceof FakeShadowRoot) return !!this.host?.isConnected;
    return !!this.parentNode?.isConnected;
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    this.harness.notifyMutation({ type: "attributes", target: this, attributeName: name });
  }
  removeAttribute(name) {
    this.attributes.delete(name);
    this.harness.notifyMutation({ type: "attributes", target: this, attributeName: name });
  }
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  set src(value) {
    this.setAttribute("src", value);
  }
  get src() {
    return this.getAttribute("src") ?? "";
  }
  appendChild(child) {
    if (child.parentNode) child.parentNode.removeChild(child);
    this.children.push(child);
    child.parentNode = this;
    this.harness.notifyMutation({
      type: "childList",
      target: this,
      addedNodes: [child],
      removedNodes: [],
    });
    return child;
  }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) {
      this.children.splice(index, 1);
      child.parentNode = null;
      this.harness.notifyMutation({
        type: "childList",
        target: this,
        addedNodes: [],
        removedNodes: [child],
      });
    }
    return child;
  }
  remove() {
    this.parentNode?.removeChild(this);
  }
  attachShadow({ mode }) {
    assert.equal(mode, "closed");
    if (this._shadow) throw new Error("shadow root already attached");
    this._shadow = new FakeShadowRoot(this, this.harness);
    this.harness.shadowRoots.push(this._shadow);
    return this._shadow;
  }
}

class FakeShadowRoot extends FakeNode {
  constructor(host, harness) {
    super("#shadow-root", harness);
    this.host = host;
  }
}

class FakeFrame extends FakeNode {
  constructor(harness) {
    super("iframe", harness);
    this.contentWindow = {
      posts: [],
      postMessage: (data, targetOrigin, transfer) => {
        this.contentWindow.posts.push({ data, targetOrigin, transfer });
        harness.onConnect?.({ data, targetOrigin, transfer, frame: this });
      },
    };
  }
}

function isObservedDescendant(node, target) {
  for (let current = node; current; current = current.parentNode) {
    if (current === target) return true;
  }
  return false;
}

class FakePort {
  constructor() {
    this.peer = null;
    this.onmessage = null;
    this.onmessageerror = null;
    this.posts = [];
    this.closed = false;
  }
  start() {}
  close() {
    this.closed = true;
  }
  postMessage(data, transfer = []) {
    if (this.closed) throw new Error("port closed");
    this.posts.push({ data, transfer });
    const peer = this.peer;
    queueMicrotask(() => {
      if (!peer?.closed) peer?.onmessage?.({ data });
    });
  }
}

class FakeMessageChannel {
  constructor() {
    this.port1 = new FakePort();
    this.port2 = new FakePort();
    this.port1.peer = this.port2;
    this.port2.peer = this.port1;
  }
}

class FakeCanvas {
  constructor({ bitmaprenderer = true } = {}) {
    this.width = 1;
    this.height = 1;
    this.bitmapTransfers = [];
    this.drawCalls = [];
    this.clearCalls = [];
    this._bitmaprenderer = bitmaprenderer;
    this.bitmapContext = {
      transferFromImageBitmap: (bitmap) => this.bitmapTransfers.push(bitmap),
    };
    this.context2d = {
      drawImage: (...args) => this.drawCalls.push(args),
      clearRect: (...args) => this.clearCalls.push(args),
      imageSmoothingEnabled: true,
      globalCompositeOperation: "source-over",
    };
  }
  getContext(kind) {
    if (kind === "bitmaprenderer") {
      return this._bitmaprenderer ? this.bitmapContext : null;
    }
    if (kind === "2d") return this.context2d;
    return null;
  }
}

class FakeBitmap {
  constructor(name, width = 1, height = 1) {
    this.name = name;
    this.width = width;
    this.height = height;
    this.closeCalls = 0;
  }
  close() { this.closeCalls++; }
}

class FakeVideoFrame {
  constructor(name, displayWidth = 1, displayHeight = 1) {
    this.name = name;
    this.displayWidth = displayWidth;
    this.displayHeight = displayHeight;
    this.closeCalls = 0;
  }
  close() { this.closeCalls++; }
}

async function flushMicrotasks(turns = 3) {
  for (let index = 0; index < turns; index++) await Promise.resolve();
}

class Harness {
  constructor({ parentProtocol = "https:" } = {}) {
    this.window = new FakeEventTarget();
    this.window.location = { protocol: parentProtocol };
    this.observers = [];
    this.shadowRoots = [];
    this.frames = [];
    this.requests = [];
    this.controls = [];
    this.capabilityRequests = [];
    this.heldMethods = new Set();
    this.childPort = null;
    this.autoAcknowledge = true;
    this.documentElement = new FakeNode("html", this);
    this.documentElement._documentRoot = true;
    this.document = {
      isConnected: true,
      documentElement: this.documentElement,
      body: null,
      createElement: (tagName) => {
        const node = tagName === "iframe"
          ? new FakeFrame(this)
          : new FakeNode(tagName, this);
        if (tagName === "iframe") this.frames.push(node);
        return node;
      },
    };
    this.documentElement.parentNode = this.document;
    const harness = this;
    this.MutationObserver = class {
      constructor(callback) {
        this.callback = callback;
        this.watches = [];
        this.disconnected = false;
        harness.observers.push(this);
      }
      observe(target, options) {
        this.watches.push({ target, options });
      }
      disconnect() {
        this.disconnected = true;
        this.watches.length = 0;
      }
    };
    this.onConnect = ({ data, targetOrigin, transfer }) => {
      assert.equal(data.channel, NEURAL_FRAME_CHANNEL);
      assert.equal(data.kind, "connect");
      assert.equal(targetOrigin, "chrome-extension://unit-test");
      assert.equal(transfer.length, 1);
      this.childPort = transfer[0];
      this.childPort.onmessage = (event) => this.onRequest(event.data);
      if (this.autoAcknowledge) {
        this.childPort.postMessage({
          channel: NEURAL_FRAME_CHANNEL,
          kind: "connected",
          instanceNonce: data.instanceNonce,
        });
      }
    };
  }

  notifyMutation(record) {
    const recipients = this.observers.filter((observer) => {
      if (observer.disconnected) return false;
      return observer.watches.some(({ target, options }) => {
        if (record.type === "attributes" && !options.attributes) return false;
        if (record.type === "childList" && !options.childList) return false;
        if (record.type === "attributes" &&
            options.attributeFilter &&
            !options.attributeFilter.includes(record.attributeName)) return false;
        return record.target === target ||
          (options.subtree && isObservedDescendant(record.target, target));
      });
    });
    queueMicrotask(() => {
      for (const observer of recipients) {
        if (!observer.disconnected) observer.callback([record]);
      }
    });
  }

  onRequest(message) {
    if (message?.kind === "cancel") {
      this.controls.push(message);
      return;
    }
    this.requests.push(message);
    if (!this.heldMethods.has(message.method)) {
      this.respond(message.id, {
        method: message.method,
        payload: message.payload,
      });
    }
  }

  respond(id, result, transfer = []) {
    this.childPort.postMessage({
      channel: NEURAL_FRAME_CHANNEL,
      kind: "response",
      instanceNonce: this.bridge.instanceNonce,
      id,
      ok: true,
      result,
    }, transfer);
  }

  reject(id, error) {
    this.childPort.postMessage({
      channel: NEURAL_FRAME_CHANNEL,
      kind: "response",
      instanceNonce: this.bridge.instanceNonce,
      id,
      ok: false,
      error,
    });
  }

  createBridge({ useWindowTimers = false, ...options } = {}) {
    const environment = {
      window: this.window,
      document: this.document,
      MessageChannel: FakeMessageChannel,
      MutationObserver: this.MutationObserver,
      HTMLCanvasElement: FakeCanvas,
      ImageBitmap: FakeBitmap,
      VideoFrame: FakeVideoFrame,
      crypto: { getRandomValues: (bytes) => bytes.fill(7) },
      setTimeout,
      clearTimeout,
    };
    if (useWindowTimers) {
      const harness = this;
      this.window.setTimeout = function (...args) {
        assert.equal(this, harness.window);
        return setTimeout(...args);
      };
      this.window.clearTimeout = function (...args) {
        assert.equal(this, harness.window);
        return clearTimeout(...args);
      };
      delete environment.setTimeout;
      delete environment.clearTimeout;
    }
    const harness = this;
    this.bridge = createNeuralFrameBridge({
      instanceNonce: "0123456789abcdef0123456789abcdef",
      runtime: {
        getURL(path) {
          const normalized = String(path).replace(/^\/+/, "");
          const host = normalized ? "unit-test-dynamic" : "unit-test";
          return `chrome-extension://${host}/${normalized}`;
        },
        sendMessage(message, callback) {
          harness.capabilityRequests.push(message);
          const response = options.capabilityResponse ?? {
            ok: true,
            capability: FRAME_CAPABILITY,
          };
          callback?.(response);
        },
      },
      warn: () => {},
      environment,
      ...options,
    });
    return this.bridge;
  }

  ready({ origin = "chrome-extension://unit-test", source, data } = {}) {
    const frame = this.frames[0];
    this.window.emit("message", {
      origin,
      source: source ?? frame.contentWindow,
      data: data ?? { channel: NEURAL_FRAME_CHANNEL, kind: "ready" },
    });
  }
}

async function connect(harness, options) {
  const bridge = harness.createBridge(options);
  const started = bridge.start();
  await flushMicrotasks();
  harness.frames[0].emit("load");
  harness.ready();
  await started;
  return bridge;
}

test("bridge hides an unsandboxed extension frame and authenticates the private port", async () => {
  const harness = new Harness();
  harness.autoAcknowledge = false;
  const states = [];
  const bridge = harness.createBridge({
    onStateChange: (state) => states.push(state.state),
  });
  const started = bridge.start();
  await flushMicrotasks();
  const frame = harness.frames[0];
  const host = bridge.hostElement;

  assert.equal(host.parentNode, harness.documentElement);
  assert.equal(host._shadow.children[0], frame);
  assert.equal(host._shadow.host, host);
  assert.equal(frame.getAttribute("sandbox"), null);
  assert.equal(frame.getAttribute("width"), "1");
  assert.equal(frame.getAttribute("height"), "1");
  assert.equal(host.style.values.get("position").value, "fixed");
  assert.equal(host.style.values.get("left").value, "-10000px");
  assert.equal(
    frame.src,
    "chrome-extension://unit-test-dynamic/src/frame/neural-frame.html" +
      "#instanceNonce=0123456789abcdef0123456789abcdef" +
      `&frameCapability=${FRAME_CAPABILITY}`,
  );
  assert.deepEqual(harness.capabilityRequests, [{
    type: "FSRCNNX_NEURAL_FRAME_CAPABILITY_MINT",
    instanceNonce: bridge.instanceNonce,
  }]);

  // Neither an unrelated WindowProxy nor an unrelated channel can claim the
  // handshake. The public ready hello carries no nonce.
  harness.ready({ source: {}, data: { channel: NEURAL_FRAME_CHANNEL, kind: "ready" } });
  harness.ready({ data: { channel: "other", kind: "ready" } });
  assert.equal(frame.contentWindow.posts.length, 0);

  harness.ready();
  assert.equal(frame.contentWindow.posts.length, 1);
  const connectMessage = frame.contentWindow.posts[0];
  assert.deepEqual(connectMessage.data, {
    channel: NEURAL_FRAME_CHANNEL,
    kind: "connect",
    instanceNonce: bridge.instanceNonce,
  });
  assert.equal(connectMessage.targetOrigin, "chrome-extension://unit-test");
  assert.equal(bridge.state, "connecting",
    "the bridge must wait for the private nonce acknowledgement");

  harness.childPort.postMessage({
    channel: NEURAL_FRAME_CHANNEL,
    kind: "connected",
    instanceNonce: bridge.instanceNonce,
  });
  const result = await started;
  assert.equal(result.connected, true);
  assert.equal(result.frame, frame);
  assert.equal(bridge.connected, true);
  assert.deepEqual(states, ["connecting", "ready"]);

  await bridge.dispose();
});

test("bridge invokes browser-native timers with their Window receiver", async () => {
  const harness = new Harness();
  const bridge = await connect(harness, { useWindowTimers: true });

  assert.equal(bridge.state, "ready");
  await bridge.dispose();
});

test("bridge trusts Chrome's dynamic frame URL but rejects another extension host", () => {
  const harness = new Harness();
  const bridge = harness.createBridge();
  assert.equal(bridge.origin, "chrome-extension://unit-test",
    "the dynamic URL commits under the static extension origin");
  assert.throws(
    () => new Harness().createBridge({
      frameUrl: "chrome-extension://another-extension/src/frame/neural-frame.html",
    }),
    /must belong to this chrome-extension origin/,
  );
});

test("bridge grants the opaque-origin capability only to file parents", async () => {
  const harness = new Harness({ parentProtocol: "file:" });
  const bridge = await connect(harness);

  assert.equal(
    harness.frames[0].src,
    "chrome-extension://unit-test-dynamic/src/frame/neural-frame.html" +
      "#instanceNonce=0123456789abcdef0123456789abcdef" +
      `&frameCapability=${FRAME_CAPABILITY}&opaqueParent=1`,
  );
  await bridge.dispose();
});

test("bridge fails closed when background capability minting is denied", async () => {
  const harness = new Harness();
  const bridge = harness.createBridge({ capabilityResponse: { ok: false } });

  await assert.rejects(
    bridge.start(),
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "capability-denied" &&
      error.retryable === true,
  );
  assert.equal(bridge.state, "failed");
  assert.equal(harness.frames.length, 0);
  assert.equal(hostIsGone(harness), true);
});

test("disposing during capability mint cannot create a late frame", async () => {
  const harness = new Harness();
  let respond;
  const bridge = harness.createBridge({
    runtime: {
      getURL(path) {
        const normalized = String(path).replace(/^\/+/, "");
        const host = normalized ? "unit-test-dynamic" : "unit-test";
        return `chrome-extension://${host}/${normalized}`;
      },
      sendMessage(_message, callback) {
        respond = callback;
      },
    },
  });
  const started = bridge.start();
  const stopped = assert.rejects(
    started,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "disposed",
  );
  await flushMicrotasks();
  assert.equal(typeof respond, "function");
  assert.equal(harness.frames.length, 0);

  await bridge.dispose();
  await stopped;
  respond({ ok: true, capability: FRAME_CAPABILITY });
  await flushMicrotasks();
  assert.equal(bridge.state, "disposed");
  assert.equal(harness.frames.length, 0);
});

test("bridge correlates requests, forwards diagnostics, and transfers each owner explicitly", async () => {
  const harness = new Harness();
  const events = [];
  const bridge = await connect(harness, {
    onEvent: (event) => events.push(event),
  });
  const canvas = new FakeCanvas();

  const attached = await bridge.attachCanvas(canvas);
  assert.equal(attached.method, "attachCanvas");
  const attachPost = harness.childPort.peer.posts.at(-1);
  assert.deepEqual(attachPost.data.payload, {});
  assert.deepEqual(attachPost.transfer, []);
  assert.equal(bridge.canvasAttached, true);

  const initialized = await bridge.init("realesrganv2-animevideo-xsx2");
  assert.equal(initialized.method, "init");
  assert.deepEqual(initialized.payload, {
    modelKey: "realesrganv2-animevideo-xsx2",
  });

  harness.heldMethods.add("run");
  const bitmap = new FakeBitmap("frame-1");
  const run = bridge.run(bitmap, {
    srcW: 640,
    srcH: 360,
    presentation: {
      width: 1280,
      height: 720,
      alphaMode: "opaque",
      ssimdsEnabled: true,
      sharpenEnabled: true,
      sharpenStrength: 1.25,
    },
    temporal: {
      mediaTime: 12.5,
      presentedFrames: 301,
      reset: true,
      resetReason: "source-change",
    },
  });
  await Promise.resolve();
  assert.equal(bridge.runPending, true);
  await assert.rejects(
    bridge.run(new FakeBitmap("frame-2"), { srcW: 640, srcH: 360 }),
    (error) => error instanceof NeuralFrameBridgeError && error.code === "run-busy",
  );
  const runRequest = harness.requests.find((request) => request.method === "run");
  assert.deepEqual(runRequest.payload.presentation, {
    width: 1280,
    height: 720,
    alphaMode: "opaque",
    ssimdsEnabled: true,
    sharpenEnabled: true,
    sharpenStrength: 1.25,
  });
  assert.deepEqual(runRequest.payload.temporal, {
    mediaTime: 12.5,
    presentedFrames: 301,
    reset: true,
    resetReason: "source-change",
  });
  const runPost = harness.childPort.peer.posts.find(
    ({ data }) => data.id === runRequest.id,
  );
  assert.deepEqual(runPost.transfer, [bitmap]);
  const outputBitmap = new FakeBitmap("neural-output", 1280, 720);
  harness.respond(runRequest.id, {
    srcW: 640,
    srcH: 360,
    modelWidth: 1280,
    modelHeight: 720,
    presentation: {
      source: { width: 640, height: 360 },
      output: { width: 1280, height: 720 },
      ssimds: null,
      sharpen: null,
    },
    stats: { frames: 1 },
    bitmap: outputBitmap,
  }, [outputBitmap]);
  assert.deepEqual(await run, {
    srcW: 640,
    srcH: 360,
    modelWidth: 1280,
    modelHeight: 720,
    presentation: {
      source: { width: 640, height: 360 },
      output: { width: 1280, height: 720 },
      ssimds: null,
      sharpen: null,
    },
    stats: { frames: 1 },
  });
  assert.equal(canvas.width, 1280);
  assert.equal(canvas.height, 720);
  assert.deepEqual(canvas.bitmapTransfers, [outputBitmap]);
  assert.equal(outputBitmap.closeCalls, 1,
    "the page presenter explicitly closes the transferred output bitmap");
  assert.equal(bridge.runPending, false);

  harness.childPort.postMessage({
    channel: NEURAL_FRAME_CHANNEL,
    kind: "event",
    instanceNonce: bridge.instanceNonce,
    event: "device-lost",
    error: { code: "device-lost", message: "adapter reset", retryable: true },
    stats: { frames: 1 },
  });
  await flushMicrotasks();
  assert.equal(events.length, 1);
  assert.equal(events[0].event, "device-lost");
  assert.equal(events[0].error.code, "device-lost");
  assert.equal(events[0].error.retryable, true);

  assert.equal((await bridge.stop()).method, "stop");
  const disposed = await bridge.dispose();
  assert.equal(disposed.method, "dispose");
  assert.equal(bridge.state, "disposed");
  assert.equal(bridge.frameElement, null);
  assert.equal(bridge.hostElement, null);
  assert.equal(hostIsGone(harness), true);
});

test("bridge accepts and explicitly transfers a VideoFrame input", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const canvas = new FakeCanvas();
  await bridge.attachCanvas(canvas);
  harness.heldMethods.add("run");

  const frame = new FakeVideoFrame("decoded-frame", 320, 180);
  const running = bridge.run(frame, {
    srcW: 320,
    srcH: 180,
    presentation: { width: 640, height: 360 },
  });
  await flushMicrotasks();
  const request = harness.requests.find(({ method }) => method === "run");
  const post = harness.childPort.peer.posts.find(({ data }) => data.id === request.id);
  assert.strictEqual(request.payload.bitmap, frame);
  assert.deepEqual(post.transfer, [frame]);

  const output = new FakeBitmap("video-frame-output", 640, 360);
  harness.respond(request.id, {
    presentation: { output: { width: 640, height: 360 } },
    bitmap: output,
  }, [output]);
  await running;

  assert.deepEqual(canvas.bitmapTransfers, [output]);
  assert.equal(output.closeCalls, 1);
  await bridge.dispose();
});

test("stop cancels an in-flight run before serialized cleanup and drops stale output", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const canvas = new FakeCanvas();
  await bridge.attachCanvas(canvas);
  harness.heldMethods.add("run");

  const input = new FakeBitmap("stop-input", 320, 180);
  const running = bridge.run(input, {
    srcW: 320,
    srcH: 180,
    presentation: { width: 640, height: 360 },
  });
  const rejectedRun = assert.rejects(
    running,
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "cancelled" && error.method === "run",
  );
  await flushMicrotasks();
  const runRequest = harness.requests.find(({ method }) => method === "run");

  const stopping = bridge.stop();
  await flushMicrotasks();
  assert.deepEqual(harness.controls, [{
    channel: NEURAL_FRAME_CHANNEL,
    kind: "cancel",
    instanceNonce: bridge.instanceNonce,
  }]);
  assert.equal(
    harness.requests.some(({ method }) => method === "stop"),
    false,
    "remote cleanup remains serialized behind the active run",
  );

  const staleOutput = new FakeBitmap("stale-stop-output", 640, 360);
  harness.respond(runRequest.id, {
    presentation: { output: { width: 640, height: 360 } },
    bitmap: staleOutput,
  }, [staleOutput]);
  await rejectedRun;
  await stopping;

  assert.equal(staleOutput.closeCalls, 1);
  assert.deepEqual(canvas.bitmapTransfers, [null]);
  assert.equal(canvas.width, 0);
  assert.equal(canvas.height, 0);
  const outbound = harness.childPort.peer.posts.map(({ data }) => data);
  const cancelIndex = outbound.findIndex(({ kind }) => kind === "cancel");
  const stopIndex = outbound.findIndex(({ method }) => method === "stop");
  assert.ok(cancelIndex >= 0 && stopIndex > cancelIndex);
  await bridge.dispose();
});

test("dispose cancels an in-flight run before releasing the frame transport", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const canvas = new FakeCanvas();
  await bridge.attachCanvas(canvas);
  harness.heldMethods.add("run");

  const input = new FakeBitmap("dispose-input", 320, 180);
  const running = bridge.run(input, {
    srcW: 320,
    srcH: 180,
    presentation: { width: 640, height: 360 },
  });
  const rejectedRun = assert.rejects(
    running,
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "cancelled" && error.method === "run",
  );
  await flushMicrotasks();
  const runRequest = harness.requests.find(({ method }) => method === "run");

  const disposing = bridge.dispose();
  assert.equal(bridge.state, "disposing");
  await flushMicrotasks();
  assert.deepEqual(harness.controls, [{
    channel: NEURAL_FRAME_CHANNEL,
    kind: "cancel",
    instanceNonce: bridge.instanceNonce,
  }]);
  assert.equal(
    harness.requests.some(({ method }) => method === "dispose"),
    false,
    "remote disposal remains serialized behind the active run",
  );

  const staleOutput = new FakeBitmap("stale-dispose-output", 640, 360);
  harness.respond(runRequest.id, {
    presentation: { output: { width: 640, height: 360 } },
    bitmap: staleOutput,
  }, [staleOutput]);
  await rejectedRun;
  await disposing;

  assert.equal(staleOutput.closeCalls, 1);
  assert.deepEqual(canvas.bitmapTransfers, [null]);
  assert.equal(canvas.width, 0);
  assert.equal(canvas.height, 0);
  const outbound = harness.childPort.peer.posts.map(({ data }) => data);
  const cancelIndex = outbound.findIndex(({ kind }) => kind === "cancel");
  const disposeIndex = outbound.findIndex(({ method }) => method === "dispose");
  assert.ok(cancelIndex >= 0 && disposeIndex > cancelIndex);
  assert.equal(bridge.state, "disposed");
  assert.equal(hostIsGone(harness), true);
});

test("bridge safely falls back to an exact-size 2D bitmap presentation", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const canvas = new FakeCanvas({ bitmaprenderer: false });
  await bridge.attachCanvas(canvas);
  harness.heldMethods.add("run");

  const input = new FakeBitmap("input", 320, 180);
  const pending = bridge.run(input, {
    srcW: 320,
    srcH: 180,
    presentation: { width: 640, height: 360 },
  });
  await Promise.resolve();
  const request = harness.requests.find(({ method }) => method === "run");
  const output = new FakeBitmap("output", 640, 360);
  harness.respond(request.id, {
    srcW: 320,
    srcH: 180,
    modelWidth: 640,
    modelHeight: 360,
    presentation: {
      source: { width: 320, height: 180 },
      output: { width: 640, height: 360 },
      ssimds: null,
      sharpen: null,
    },
    stats: {},
    bitmap: output,
  }, [output]);

  const result = await pending;
  assert.equal(result.bitmap, undefined);
  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 360);
  assert.deepEqual(canvas.drawCalls, [[output, 0, 0, 640, 360]]);
  assert.equal(canvas.context2d.imageSmoothingEnabled, false);
  assert.equal(canvas.context2d.globalCompositeOperation, "copy");
  assert.equal(output.closeCalls, 1);
  await bridge.dispose();
});

test("bridge closes and rejects a mismatched output bitmap before presentation", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const canvas = new FakeCanvas();
  await bridge.attachCanvas(canvas);
  harness.heldMethods.add("run");

  const input = new FakeBitmap("input", 320, 180);
  const pending = bridge.run(input, {
    srcW: 320,
    srcH: 180,
    presentation: { width: 640, height: 360 },
  });
  await Promise.resolve();
  const request = harness.requests.find(({ method }) => method === "run");
  const output = new FakeBitmap("wrong-size-output", 639, 360);
  harness.respond(request.id, {
    presentation: {
      output: { width: 640, height: 360 },
    },
    bitmap: output,
  }, [output]);

  await assert.rejects(
    pending,
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "protocol-error" && error.method === "run",
  );
  assert.equal(output.closeCalls, 1);
  assert.deepEqual(canvas.bitmapTransfers, [null],
    "cleanup clears the presenter without displaying the invalid bitmap");
  assert.equal(bridge.state, "failed");
});

function hostIsGone(harness) {
  return harness.documentElement.children.length === 0;
}

test("bridge rejects invalid presentation tails before transferring a bitmap", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const cases = [
    { width: 1280 },
    { width: 1280, height: 720, ssimdsEnabled: 1 },
    { width: 1280, height: 720, sharpenEnabled: "true" },
    { width: 1280, height: 720, sharpenStrength: 0.09 },
    { width: 1280, height: 720, sharpenStrength: 2.01 },
  ];
  for (const presentation of cases) {
    const bitmap = new FakeBitmap("invalid");
    await assert.rejects(
      bridge.run(bitmap, { srcW: 640, srcH: 360, presentation }),
      TypeError,
    );
  }
  assert.equal(harness.requests.some(({ method }) => method === "run"), false);
  await bridge.dispose();
});

test("bridge rejects unbounded temporal metadata before transferring a frame", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  const cases = [
    { mediaTime: -1 },
    { presentedFrames: 1.5 },
    { reset: "true" },
    { resetReason: "Source changed!" },
    { unsupported: true },
  ];
  for (const temporal of cases) {
    const frame = new FakeVideoFrame("invalid-temporal");
    await assert.rejects(
      bridge.run(frame, { srcW: 640, srcH: 360, temporal }),
      TypeError,
    );
    assert.equal(frame.closeCalls, 0, "validation must happen before ownership transfer");
  }
  assert.equal(harness.requests.some(({ method }) => method === "run"), false);
  await bridge.dispose();
});

test("wrong extension origin fails the handshake without transferring a port", async () => {
  const harness = new Harness();
  const bridge = harness.createBridge();
  const started = bridge.start();
  await flushMicrotasks();
  harness.ready({ origin: "https://host.invalid" });
  await assert.rejects(
    started,
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "frame-origin-mismatch",
  );
  assert.equal(bridge.state, "failed");
  assert.equal(harness.frames[0].contentWindow.posts.length, 0);
  const closed = await bridge.closed;
  assert.equal(closed.error.code, "frame-origin-mismatch");
});

test("public nonce disclosure and an incorrect private nonce both fail closed", async () => {
  const publicHarness = new Harness();
  const publicBridge = publicHarness.createBridge();
  const publicStart = publicBridge.start();
  await flushMicrotasks();
  publicHarness.ready({
    data: {
      channel: NEURAL_FRAME_CHANNEL,
      kind: "ready",
      instanceNonce: publicBridge.instanceNonce,
    },
  });
  await assert.rejects(
    publicStart,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "protocol-error",
  );
  assert.equal(publicHarness.frames[0].contentWindow.posts.length, 0);

  const privateHarness = new Harness();
  privateHarness.autoAcknowledge = false;
  const privateBridge = privateHarness.createBridge();
  const privateStart = privateBridge.start();
  await flushMicrotasks();
  privateHarness.ready();
  privateHarness.childPort.postMessage({
    channel: NEURAL_FRAME_CHANNEL,
    kind: "connected",
    instanceNonce: "ffffffffffffffffffffffffffffffff",
  });
  await assert.rejects(
    privateStart,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "protocol-error",
  );
  assert.equal(privateBridge.state, "failed");
});

test("frame removal rejects pending requests even when the host is reinserted", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  harness.heldMethods.add("init");
  const pending = bridge.init("model-v1");
  await Promise.resolve();
  const host = bridge.hostElement;
  host.remove();
  harness.documentElement.appendChild(host);

  await assert.rejects(
    pending,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "frame-removed",
  );
  assert.equal(bridge.state, "failed");
  assert.equal(bridge.frameElement, null);
  assert.equal(hostIsGone(harness), true);
});

test("frame navigation rejects pending requests before the replacement can handshake", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  harness.heldMethods.add("init");
  const pending = bridge.init("model-v1");
  await Promise.resolve();
  harness.frames[0].emit("load");

  await assert.rejects(
    pending,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "frame-navigated",
  );
  assert.equal(bridge.state, "failed");
  assert.equal(hostIsGone(harness), true);
});

test("bounded request timeout rejects work and retires the ambiguous frame instance", async () => {
  const harness = new Harness();
  const bridge = await connect(harness, { timeouts: { run: 10 } });
  harness.heldMethods.add("run");
  const pending = bridge.run(new FakeBitmap("timeout"), {
    srcW: 320,
    srcH: 180,
  });

  await assert.rejects(
    pending,
    (error) => error instanceof NeuralFrameBridgeError &&
      error.code === "request-timeout" && error.method === "run",
  );
  assert.equal(bridge.state, "failed");
  assert.equal(bridge.frameElement, null);
  assert.equal(hostIsGone(harness), true);
});

test("pagehide rejects in-flight work and transitions directly to disposed", async () => {
  const harness = new Harness();
  const bridge = await connect(harness);
  harness.heldMethods.add("run");
  const bitmap = new FakeBitmap("pagehide");
  const pending = bridge.run(bitmap, {
    srcW: 320,
    srcH: 180,
  });
  await Promise.resolve();
  harness.window.emit("pagehide");

  await assert.rejects(
    pending,
    (error) => error instanceof NeuralFrameBridgeError && error.code === "page-unloaded",
  );
  assert.equal(bitmap.closeCalls, 1);
  assert.equal(bridge.state, "disposed");
  assert.equal(hostIsGone(harness), true);
});

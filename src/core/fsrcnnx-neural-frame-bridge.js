// fsrcnnx-neural-frame-bridge.js — isolated-world parent for Neural rendering.
//
// Chromium can apply a page's WebAssembly/JIT policy to an isolated content
// script. Neural inference therefore lives in an unsandboxed extension iframe,
// while this bridge keeps DOM ownership in the existing content-script world.
// It deliberately uses window.postMessage only for an origin/source
// authenticated hello and port handoff; the nonce is verified privately on the
// transferred MessagePort before any commands are allowed.
//
// Integration:
//   const bridge = createNeuralFrameBridge({ onEvent });
//   await bridge.start();
//   await bridge.attachCanvas(htmlCanvas);
//   await bridge.init(modelKey);
//   await bridge.run(bitmap, { srcW, srcH, presentation: { width, height } });
//   await bridge.stop();
//   await bridge.dispose();
//
// The HTMLCanvasElement always remains page-owned. The child renders into its
// own extension-realm OffscreenCanvas and transfers a finished ImageBitmap back;
// this avoids transferring a placeholder-controlled canvas across Chromium's
// OOPIF boundary. A validated run() transfers its input ImageBitmap to the child
// and consumes/closes the returned bitmap through bitmaprenderer (or a 2D
// fallback). run() rejects with
// `run-busy` while a previous run is unresolved, giving the render loop an
// explicit backpressure signal instead of accumulating stale video frames.

export const NEURAL_FRAME_CHANNEL = "fsrcnnx-neural-frame-v1";

const DEFAULT_FRAME_PATH = "src/frame/neural-frame.html";
const NEURAL_CAPABILITY_MINT = "FSRCNNX_NEURAL_FRAME_CAPABILITY_MINT";
const NONCE_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;
const CAPABILITY_PATTERN = /^[a-f0-9]{48}$/;
const ERROR_CODE_PATTERN = /^[a-z][a-z0-9-]{0,63}$/;
const MODEL_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const MAX_ERROR_MESSAGE_LENGTH = 1000;
const MAX_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_PENDING_REQUESTS = 16;
const SRGB_COLOR_SPACE = "srgb";
const DEFAULT_TIMEOUTS = Object.freeze({
  handshake: 8_000,
  attachCanvas: 10_000,
  init: 90_000,
  run: 30_000,
  stop: 10_000,
  dispose: 10_000,
});

function boundedText(value, fallback) {
  if (typeof value !== "string" || !value.trim()) return fallback;
  return value.trim().slice(0, MAX_ERROR_MESSAGE_LENGTH);
}

function safeCloseBitmap(bitmap) {
  try { bitmap?.close?.(); } catch {}
}

function checkedTimeout(value, fallback, label) {
  const timeout = value == null ? fallback : value;
  if (!Number.isFinite(timeout) || !Number.isInteger(timeout) ||
      timeout < 1 || timeout > MAX_TIMEOUT_MS) {
    throw new TypeError(`${label} must be an integer from 1 to ${MAX_TIMEOUT_MS}`);
  }
  return timeout;
}

function positiveDimension(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive safe integer`);
  }
  return value;
}

function makeNonce(cryptoObject) {
  if (!cryptoObject || typeof cryptoObject.getRandomValues !== "function") {
    throw new Error("secure random generation is unavailable");
  }
  const bytes = new Uint8Array(18);
  cryptoObject.getRandomValues(bytes);
  let nonce = "";
  for (const byte of bytes) nonce += byte.toString(16).padStart(2, "0");
  return nonce;
}

function sendRuntimeMessage(runtime, message) {
  if (typeof runtime?.sendMessage !== "function") {
    return Promise.reject(new Error("chrome.runtime.sendMessage is unavailable"));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const complete = (callback) => (value) => {
      if (settled) return;
      settled = true;
      callback(value);
    };
    const succeed = complete(resolve);
    const fail = complete(reject);
    try {
      const pending = runtime.sendMessage(message, (response) => {
        let runtimeError = null;
        try { runtimeError = runtime.lastError; } catch {}
        if (runtimeError) {
          fail(new Error(runtimeError.message || "Extension message failed"));
        } else {
          succeed(response);
        }
      });
      if (pending && typeof pending.then === "function") {
        pending.then(succeed, fail);
      }
    } catch (error) {
      fail(error);
    }
  });
}

function normalizeErrorCode(value, fallback = "remote-error") {
  return typeof value === "string" && ERROR_CODE_PATTERN.test(value)
    ? value
    : fallback;
}

function normalizePresentation(value) {
  if (value == null) return {};
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("neural presentation settings must be an object");
  }
  const presentation = {};
  if (value.width != null) {
    presentation.width = positiveDimension(value.width, "neural presentation width");
  }
  if (value.height != null) {
    presentation.height = positiveDimension(value.height, "neural presentation height");
  }
  if ((value.width == null) !== (value.height == null)) {
    throw new TypeError("neural presentation width and height must be provided together");
  }
  if (value.alphaMode != null) {
    if (value.alphaMode !== "opaque" && value.alphaMode !== "premultiplied") {
      throw new TypeError("neural presentation alphaMode must be 'opaque' or 'premultiplied'");
    }
    presentation.alphaMode = value.alphaMode;
  }
  for (const field of ["ssimdsEnabled", "sharpenEnabled"]) {
    if (value[field] != null) {
      if (typeof value[field] !== "boolean") {
        throw new TypeError(`neural presentation ${field} must be a boolean`);
      }
      presentation[field] = value[field];
    }
  }
  if (value.sharpenStrength != null) {
    if (!Number.isFinite(value.sharpenStrength) ||
        value.sharpenStrength < 0.1 || value.sharpenStrength > 2) {
      throw new TypeError("neural presentation sharpenStrength must be from 0.1 to 2");
    }
    presentation.sharpenStrength = value.sharpenStrength;
  }
  return presentation;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function removedNodesInclude(record, node) {
  if (!node || !record?.removedNodes) return false;
  for (const removed of record.removedNodes) {
    if (removed === node) return true;
  }
  return false;
}

export class NeuralFrameBridgeError extends Error {
  constructor(code, message, {
    cause,
    retryable = false,
    remote = false,
    method = null,
  } = {}) {
    super(message);
    this.name = "NeuralFrameBridgeError";
    this.code = normalizeErrorCode(code, "bridge-error");
    this.retryable = !!retryable;
    this.remote = !!remote;
    if (method != null) this.method = method;
    if (cause !== undefined) this.cause = cause;
  }
}

function bridgeError(code, message, options) {
  return new NeuralFrameBridgeError(code, message, options);
}

function remoteBridgeError(value, method) {
  const raw = value && typeof value === "object" ? value : {};
  return bridgeError(
    normalizeErrorCode(raw.code),
    boundedText(raw.message, `Neural frame ${method} failed`),
    {
      retryable: raw.retryable === true,
      remote: true,
      method,
    },
  );
}

function requireEnvironment(environment = {}) {
  const globalObject = globalThis;
  const parentWindow = environment.window ?? globalObject.window;
  const parentDocument = environment.document ?? globalObject.document;
  const MessageChannelCtor = environment.MessageChannel ?? globalObject.MessageChannel;
  const MutationObserverCtor = environment.MutationObserver ?? globalObject.MutationObserver;
  const HTMLCanvasElementCtor =
    environment.HTMLCanvasElement ?? globalObject.HTMLCanvasElement;
  const ImageBitmapCtor = environment.ImageBitmap ?? globalObject.ImageBitmap;
  const cryptoObject = environment.crypto ?? globalObject.crypto;
  const setTimerSource = environment.setTimeout ??
    parentWindow?.setTimeout ??
    globalObject.setTimeout;
  const clearTimerSource = environment.clearTimeout ??
    parentWindow?.clearTimeout ??
    globalObject.clearTimeout;
  // Chromium's Window timer methods require their Window receiver. Keep
  // injectable timers callable for tests while binding browser-native timers
  // to the realm that owns them.
  const setTimer = environment.setTimeout
    ? (...args) => environment.setTimeout(...args)
    : parentWindow?.setTimeout
      ? (...args) => parentWindow.setTimeout(...args)
      : (...args) => globalObject.setTimeout(...args);
  const clearTimer = environment.clearTimeout
    ? (...args) => environment.clearTimeout(...args)
    : parentWindow?.clearTimeout
      ? (...args) => parentWindow.clearTimeout(...args)
      : (...args) => globalObject.clearTimeout(...args);

  if (!parentWindow?.addEventListener || !parentWindow?.removeEventListener) {
    throw new Error("window event APIs are unavailable");
  }
  if (!parentDocument?.createElement) {
    throw new Error("document element creation is unavailable");
  }
  if (typeof MessageChannelCtor !== "function") {
    throw new Error("MessageChannel is unavailable");
  }
  if (typeof MutationObserverCtor !== "function") {
    throw new Error("MutationObserver is unavailable");
  }
  if (typeof setTimerSource !== "function" ||
      typeof clearTimerSource !== "function") {
    throw new Error("timer APIs are unavailable");
  }

  return {
    window: parentWindow,
    document: parentDocument,
    MessageChannel: MessageChannelCtor,
    MutationObserver: MutationObserverCtor,
    HTMLCanvasElement: HTMLCanvasElementCtor,
    ImageBitmap: ImageBitmapCtor,
    crypto: cryptoObject,
    setTimeout: setTimer,
    clearTimeout: clearTimer,
  };
}

function resolveExtensionFrameUrl(frameUrl, runtime) {
  if (typeof runtime?.getURL !== "function") {
    throw new Error("chrome.runtime.getURL is unavailable");
  }
  const extensionBase = new URL(runtime.getURL(""));
  if (extensionBase.protocol !== "chrome-extension:" || !extensionBase.host ||
      extensionBase.username || extensionBase.password || extensionBase.port) {
    throw new Error("chrome.runtime.getURL returned an invalid extension origin");
  }
  const defaultFrame = new URL(runtime.getURL(DEFAULT_FRAME_PATH), extensionBase);
  if (defaultFrame.protocol !== "chrome-extension:" || !defaultFrame.host ||
      defaultFrame.username || defaultFrame.password || defaultFrame.port) {
    throw new Error("chrome.runtime.getURL returned an invalid frame URL");
  }
  // `use_dynamic_url` may give this packaged resource a session-scoped host.
  // Trust only the static extension host and the host Chrome returned for this
  // exact packaged frame, never an arbitrary chrome-extension URL.
  const allowedHosts = new Set([extensionBase.host, defaultFrame.host]);
  const resolved = frameUrl == null
    ? defaultFrame
    : new URL(frameUrl, extensionBase);
  if (resolved.protocol !== "chrome-extension:" ||
      !allowedHosts.has(resolved.host) ||
      resolved.username || resolved.password || resolved.port) {
    throw new Error("neural frame URL must belong to this chrome-extension origin");
  }
  // A dynamic web-accessible URL redirects and commits under the extension's
  // static origin; only the initial iframe src uses the per-session dynamic host.
  const extensionOrigin = `chrome-extension://${extensionBase.host}`;
  return { resolved, extensionOrigin };
}

function hasOpaqueFileParent(windowObject) {
  try {
    return windowObject?.location?.protocol === "file:";
  } catch {
    return false;
  }
}

/**
 * Creates the isolated-world side of the Neural extension-frame transport.
 *
 * Options intended for callers:
 * - `frameUrl`: optional URL under this extension
 *   (defaults to src/frame/neural-frame.html)
 * - `onEvent(event)`: receives authenticated child events such as device loss
 * - `onStateChange({previous, state, error})`: lifecycle hook for main
 * - `timeouts`: per-method bounded timeouts in milliseconds
 *
 * `environment`, `instanceNonce`, and `runtime` are injectable so the transport
 * and its security checks can be tested without weakening production defaults.
 */
export function createNeuralFrameBridge(options = {}) {
  return new NeuralFrameBridge(options);
}

export class NeuralFrameBridge {
  constructor({
    frameUrl,
    onEvent = () => {},
    onStateChange = () => {},
    warn = (...args) => console.warn(...args),
    timeouts = {},
    environment,
    instanceNonce,
    runtime = globalThis.chrome?.runtime,
  } = {}) {
    if (typeof onEvent !== "function") throw new TypeError("onEvent must be a function");
    if (typeof onStateChange !== "function") {
      throw new TypeError("onStateChange must be a function");
    }
    if (typeof warn !== "function") throw new TypeError("warn must be a function");
    if (timeouts == null || typeof timeouts !== "object" || Array.isArray(timeouts)) {
      throw new TypeError("timeouts must be an object");
    }

    this._environment = requireEnvironment(environment);
    const { resolved, extensionOrigin } = resolveExtensionFrameUrl(frameUrl, runtime);
    resolved.hash = "";
    this._runtime = runtime;
    this._frameBaseUrl = resolved.href;
    this._nonce = instanceNonce ?? makeNonce(this._environment.crypto);
    if (!NONCE_PATTERN.test(this._nonce)) {
      throw new TypeError("instanceNonce must contain 16-128 URL-safe characters");
    }
    this._frameUrl = null;
    this._origin = extensionOrigin;
    this._onEvent = onEvent;
    this._onStateChange = onStateChange;
    this._warn = warn;
    this._timeouts = Object.freeze({
      handshake: checkedTimeout(
        timeouts.handshake,
        DEFAULT_TIMEOUTS.handshake,
        "neural frame handshake timeout",
      ),
      attachCanvas: checkedTimeout(
        timeouts.attachCanvas,
        DEFAULT_TIMEOUTS.attachCanvas,
        "neural frame attachCanvas timeout",
      ),
      init: checkedTimeout(
        timeouts.init,
        DEFAULT_TIMEOUTS.init,
        "neural frame init timeout",
      ),
      run: checkedTimeout(
        timeouts.run,
        DEFAULT_TIMEOUTS.run,
        "neural frame run timeout",
      ),
      stop: checkedTimeout(
        timeouts.stop,
        DEFAULT_TIMEOUTS.stop,
        "neural frame stop timeout",
      ),
      dispose: checkedTimeout(
        timeouts.dispose,
        DEFAULT_TIMEOUTS.dispose,
        "neural frame dispose timeout",
      ),
    });

    this._state = "idle";
    this._host = null;
    this._frame = null;
    this._port = null;
    this._observer = null;
    this._connection = null;
    this._connectionTimer = null;
    this._pending = new Map();
    this._nextRequestId = 1;
    this._initialLoadSeen = false;
    this._connectSent = false;
    this._canvas = null;
    this._canvasAttachStarted = false;
    this._canvasAttached = false;
    this._canvasPresenter = null;
    this._runPromise = null;
    this._runGeneration = 0;
    this._disposePromise = null;
    this._closed = deferred();
    this._closedSettled = false;

    this._handleWindowMessage = (event) => this._onWindowMessage(event);
    this._handleFrameLoad = () => this._onFrameLoad();
    this._handleFrameError = () => this._fail(bridgeError(
      "frame-load-failed",
      "Neural extension frame failed to load",
      { retryable: true },
    ));
    this._handlePageHide = () => this._fail(bridgeError(
      "page-unloaded",
      "The page was unloaded while Neural processing was active",
    ), "disposed");
    this._handlePortMessage = (event) => this._onPortMessage(event);
    this._handlePortMessageError = () => this._fail(bridgeError(
      "port-message-error",
      "Neural frame sent an unreadable channel message",
      { retryable: true },
    ));
  }

  get state() { return this._state; }
  get connected() { return this._state === "ready"; }
  get hostElement() { return this._host; }
  get frameElement() { return this._frame; }
  get canvasElement() { return this._canvas; }
  get canvasAttached() { return this._canvasAttached; }
  get runPending() { return this._runPromise != null; }
  get instanceNonce() { return this._nonce; }
  get origin() { return this._origin; }
  get closed() { return this._closed.promise; }

  _setState(state, error = null) {
    if (this._state === state) return;
    const previous = this._state;
    this._state = state;
    const notification = Object.freeze({ previous, state, error });
    Promise.resolve().then(() => this._onStateChange(notification)).catch((callbackError) => {
      try { this._warn("neural frame state callback failed:", callbackError); }
      catch {}
    });
  }

  _settleClosed(error = null) {
    if (this._closedSettled) return;
    this._closedSettled = true;
    this._closed.resolve(Object.freeze({ state: this._state, error }));
  }

  _newFrame() {
    const { document } = this._environment;
    const mount = document.documentElement ?? document.body;
    if (!mount?.appendChild) {
      throw bridgeError(
        "document-unavailable",
        "The document has no root for the Neural extension frame",
        { retryable: true },
      );
    }

    const host = document.createElement("div");
    if (typeof host.attachShadow !== "function") {
      throw bridgeError(
        "shadow-root-unavailable",
        "A closed shadow root is unavailable for the Neural extension frame",
      );
    }
    host.setAttribute("aria-hidden", "true");
    for (const [property, value] of [
      ["position", "fixed"],
      ["left", "-10000px"],
      ["top", "-10000px"],
      ["width", "1px"],
      ["height", "1px"],
      ["overflow", "hidden"],
      ["pointer-events", "none"],
      ["z-index", "-2147483648"],
    ]) {
      host.style.setProperty(property, value, "important");
    }
    const shadow = host.attachShadow({ mode: "closed" });
    if (!shadow?.appendChild) {
      throw bridgeError(
        "shadow-root-unavailable",
        "The Neural extension frame could not create a closed shadow root",
      );
    }

    const frame = document.createElement("iframe");
    frame.setAttribute("aria-hidden", "true");
    frame.setAttribute("title", "FSRCNNX Neural renderer");
    frame.setAttribute("tabindex", "-1");
    frame.setAttribute("width", "1");
    frame.setAttribute("height", "1");
    // Intentionally unsandboxed: the extension document needs its own WebGPU
    // and WebAssembly execution environment. Origin+source+nonce authenticate it.
    frame.removeAttribute("sandbox");
    for (const [property, value] of [
      ["display", "block"],
      ["position", "fixed"],
      ["left", "-10000px"],
      ["top", "-10000px"],
      ["width", "1px"],
      ["height", "1px"],
      ["border", "0"],
      ["margin", "0"],
      ["padding", "0"],
      ["pointer-events", "none"],
    ]) {
      frame.style.setProperty(property, value, "important");
    }
    frame.src = this._frameUrl;
    frame.addEventListener("load", this._handleFrameLoad);
    frame.addEventListener("error", this._handleFrameError);

    this._host = host;
    this._frame = frame;
    this._observer = new this._environment.MutationObserver((records) => {
      if (this._state === "disposed" || this._state === "failed") return;
      if (this._host?.isConnected === false || this._frame?.isConnected === false) {
        this._fail(bridgeError(
          "frame-removed",
          "Neural extension frame was removed from the document",
          { retryable: true },
        ));
        return;
      }
      for (const record of records ?? []) {
        if (record?.type === "childList" &&
            (removedNodesInclude(record, mount) ||
             removedNodesInclude(record, host) ||
             removedNodesInclude(record, frame))) {
          this._fail(bridgeError(
            "frame-removed",
            "Neural extension frame was removed from its protected host",
            { retryable: true },
          ));
          return;
        }
        if (record?.target !== this._frame || record.type !== "attributes") continue;
        if (record.attributeName === "sandbox" ||
            (record.attributeName === "src" &&
             this._frame.getAttribute("src") !== this._frameUrl)) {
          this._fail(bridgeError(
            "frame-navigated",
            "Neural extension frame was modified or navigated",
            { retryable: true },
          ));
          return;
        }
      }
    });
    // Observe the Document rather than only its current root so replacing
    // documentElement cannot detach the closed-shadow host unnoticed.
    this._observer.observe(document, {
      childList: true,
    });
    // The host is a direct child. Avoid observing an entire dynamic page.
    this._observer.observe(mount, {
      childList: true,
    });
    this._observer.observe(shadow, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["src", "sandbox"],
    });

    shadow.appendChild(frame);
    mount.appendChild(host);
    if (host.isConnected === false || frame.isConnected === false) {
      throw bridgeError(
        "frame-removed",
        "Neural extension frame could not be attached to the document",
        { retryable: true },
      );
    }
    return frame;
  }

  async _prepareFrameUrl() {
    const response = await sendRuntimeMessage(this._runtime, {
      type: NEURAL_CAPABILITY_MINT,
      instanceNonce: this._nonce,
    });
    if (!response || typeof response !== "object" ||
        Object.keys(response).length !== 2 ||
        response.ok !== true ||
        !CAPABILITY_PATTERN.test(response.capability)) {
      throw bridgeError(
        "capability-denied",
        "Background authorization for the Neural extension frame was denied",
        { retryable: true },
      );
    }
    const frameCapability = {
      instanceNonce: this._nonce,
      frameCapability: response.capability,
    };
    if (hasOpaqueFileParent(this._environment.window)) {
      // file: parents serialize postMessage origins as "null". The background
      // binds this flag to the active file document before the child accepts it.
      frameCapability.opaqueParent = "1";
    }
    const resolved = new URL(this._frameBaseUrl);
    resolved.hash = new URLSearchParams(frameCapability).toString();
    return resolved.href;
  }

  async _beginStart() {
    try {
      const frameUrl = await this._prepareFrameUrl();
      if (this._state !== "connecting") return;
      this._frameUrl = frameUrl;
      this._newFrame();
    } catch (error) {
      if (this._state !== "connecting") return;
      this._fail(error instanceof NeuralFrameBridgeError ? error : bridgeError(
        "capability-unavailable",
        boundedText(error?.message, "Neural frame authorization is unavailable"),
        { cause: error, retryable: true },
      ));
    }
  }

  start() {
    if (this._state === "ready") {
      return Promise.resolve(Object.freeze({
        connected: true,
        origin: this._origin,
        frame: this._frame,
      }));
    }
    if (this._state === "connecting") return this._connection.promise;
    if (this._state !== "idle") {
      return Promise.reject(bridgeError(
        this._state === "disposed" ? "disposed" : "bridge-closed",
        `Neural frame bridge is ${this._state}`,
      ));
    }

    this._connection = deferred();
    this._setState("connecting");
    try {
      this._environment.window.addEventListener("message", this._handleWindowMessage);
      this._environment.window.addEventListener("pagehide", this._handlePageHide);
      this._connectionTimer = this._environment.setTimeout(() => {
        this._fail(bridgeError(
          "handshake-timeout",
          "Neural extension frame did not complete its handshake",
          { retryable: true },
        ));
      }, this._timeouts.handshake);
      void this._beginStart();
    } catch (error) {
      this._fail(error instanceof NeuralFrameBridgeError ? error : bridgeError(
        "frame-creation-failed",
        boundedText(error?.message, "Neural extension frame could not be created"),
        { cause: error, retryable: true },
      ));
    }
    return this._connection.promise;
  }

  _onFrameLoad() {
    if (!this._initialLoadSeen) {
      this._initialLoadSeen = true;
      return;
    }
    if (this._state === "connecting" || this._state === "ready" ||
        this._state === "disposing") {
      this._fail(bridgeError(
        "frame-navigated",
        "Neural extension frame navigated after its initial load",
        { retryable: true },
      ));
    }
  }

  _onWindowMessage(event) {
    const frameWindow = this._frame?.contentWindow;
    if (!frameWindow || event?.source !== frameWindow) return;
    if (event.origin !== this._origin) {
      this._fail(bridgeError(
        "frame-origin-mismatch",
        "Neural extension frame reported an unexpected origin",
      ));
      return;
    }
    const data = event.data;
    if (!data || typeof data !== "object" ||
        data.channel !== NEURAL_FRAME_CHANNEL || data.kind !== "ready") return;
    if (this._state !== "connecting") return;
    if (this._connectSent) return;
    if (Object.hasOwn(data, "instanceNonce")) {
      this._fail(bridgeError(
        "protocol-error",
        "Neural frame disclosed its private handshake nonce",
      ));
      return;
    }

    let channel;
    try {
      channel = new this._environment.MessageChannel();
      const port = channel.port1;
      if (!port || !channel.port2 || typeof port.postMessage !== "function") {
        throw new Error("MessageChannel did not provide two usable ports");
      }
      port.onmessage = this._handlePortMessage;
      port.onmessageerror = this._handlePortMessageError;
      port.start?.();
      frameWindow.postMessage({
        channel: NEURAL_FRAME_CHANNEL,
        kind: "connect",
        instanceNonce: this._nonce,
      }, this._origin, [channel.port2]);
      this._port = port;
      this._connectSent = true;
    } catch (error) {
      try { channel?.port1?.close?.(); } catch {}
      try { channel?.port2?.close?.(); } catch {}
      this._fail(bridgeError(
        "handshake-failed",
        boundedText(error?.message, "Neural extension frame channel setup failed"),
        { cause: error, retryable: true },
      ));
      return;
    }

    // No further window-level message participates in the handshake. The
    // child must now echo the fragment nonce over the private port.
    this._environment.window.removeEventListener("message", this._handleWindowMessage);
  }

  _createCanvasPresenter(canvas) {
    let bitmapContext = null;
    try { bitmapContext = canvas.getContext("bitmaprenderer"); }
    catch {}
    if (bitmapContext) {
      if (typeof bitmapContext.transferFromImageBitmap !== "function") {
        throw bridgeError(
          "canvas-context-unavailable",
          "Canvas bitmap presentation is incomplete in this browser",
        );
      }
      return Object.freeze({ kind: "bitmaprenderer", context: bitmapContext });
    }

    let context2d = null;
    try {
      context2d = canvas.getContext("2d", {
        alpha: false,
        colorSpace: SRGB_COLOR_SPACE,
        desynchronized: true,
      });
    } catch {}
    if (!context2d || typeof context2d.drawImage !== "function") {
      throw bridgeError(
        "canvas-context-unavailable",
        "Canvas bitmap presentation is unavailable in this browser",
      );
    }
    try { context2d.imageSmoothingEnabled = false; } catch {}
    return Object.freeze({ kind: "2d", context: context2d });
  }

  _presentRunResult(value) {
    const result = value && typeof value === "object" ? value : null;
    const bitmap = result?.bitmap;
    const output = result?.presentation?.output;
    const BitmapCtor = this._environment.ImageBitmap;
    const validBitmap = bitmap && typeof bitmap === "object" &&
      (typeof BitmapCtor !== "function" || bitmap instanceof BitmapCtor) &&
      typeof bitmap.close === "function" &&
      Number.isSafeInteger(bitmap.width) && bitmap.width > 0 &&
      Number.isSafeInteger(bitmap.height) && bitmap.height > 0;
    if (!validBitmap ||
        !Number.isSafeInteger(output?.width) || output.width <= 0 ||
        !Number.isSafeInteger(output?.height) || output.height <= 0 ||
        bitmap.width !== output.width || bitmap.height !== output.height) {
      safeCloseBitmap(bitmap);
      throw bridgeError(
        "protocol-error",
        "Neural frame returned invalid output bitmap dimensions",
        { retryable: true, method: "run" },
      );
    }
    if (!this._canvas || !this._canvasPresenter) {
      safeCloseBitmap(bitmap);
      throw bridgeError(
        "canvas-not-attached",
        "Neural output canvas is not attached",
        { method: "run" },
      );
    }

    try {
      if (this._canvas.width !== output.width) this._canvas.width = output.width;
      if (this._canvas.height !== output.height) this._canvas.height = output.height;
      if (this._canvasPresenter.kind === "bitmaprenderer") {
        this._canvasPresenter.context.transferFromImageBitmap(bitmap);
      } else {
        const context2d = this._canvasPresenter.context;
        // Assigning canvas.width/height resets all 2D context state.
        try { context2d.imageSmoothingEnabled = false; } catch {}
        try { context2d.globalCompositeOperation = "copy"; } catch {}
        context2d.drawImage(bitmap, 0, 0, output.width, output.height);
      }
    } catch (error) {
      throw bridgeError(
        "presentation-failed",
        boundedText(error?.message, "Neural output bitmap presentation failed"),
        { cause: error, retryable: true, method: "run" },
      );
    } finally {
      // bitmaprenderer consumes ownership; close() is harmless after that
      // transfer and is required for the 2D fallback and all failure paths.
      safeCloseBitmap(bitmap);
    }

    const { bitmap: _ownedBitmap, ...metadata } = result;
    return Object.freeze(metadata);
  }

  _clearCanvasPresenter() {
    if (!this._canvas || !this._canvasPresenter) return;
    try {
      if (this._canvasPresenter.kind === "bitmaprenderer") {
        this._canvasPresenter.context.transferFromImageBitmap(null);
      } else {
        this._canvasPresenter.context.clearRect?.(
          0,
          0,
          this._canvas.width || 0,
          this._canvas.height || 0,
        );
      }
    } catch {}
  }

  _releaseCanvasBacking() {
    this._clearCanvasPresenter();
    if (!this._canvas) return;
    // clearRect/transferFromImageBitmap(null) releases the presented image, while
    // zero-sizing also releases the potentially large canvas backing allocation.
    try { this._canvas.width = 0; } catch {}
    try { this._canvas.height = 0; } catch {}
  }

  _signalCancel() {
    // Invalidate any response already in transit before notifying the child.
    this._runGeneration++;
    if (!this._port) return false;
    try {
      this._port.postMessage({
        channel: NEURAL_FRAME_CHANNEL,
        kind: "cancel",
        instanceNonce: this._nonce,
      });
      return true;
    } catch (error) {
      throw bridgeError(
        "cancel-send-failed",
        boundedText(error?.message, "Neural frame cancellation could not be sent"),
        { cause: error, retryable: true },
      );
    }
  }

  _onPortMessage(event) {
    const data = event?.data;
    const responseBitmap = data?.result?.bitmap;
    if (!data || typeof data !== "object" ||
        data.channel !== NEURAL_FRAME_CHANNEL ||
        data.instanceNonce !== this._nonce) {
      safeCloseBitmap(responseBitmap);
      this._fail(bridgeError(
        "protocol-error",
        "Neural frame sent an invalid channel envelope",
        { retryable: true },
      ));
      return;
    }

    if (data.kind === "connected") {
      safeCloseBitmap(responseBitmap);
      if (this._state !== "connecting" || !this._connectSent) {
        this._fail(bridgeError(
          "protocol-error",
          "Neural frame sent an unexpected connection acknowledgement",
        ));
        return;
      }
      this._environment.clearTimeout(this._connectionTimer);
      this._connectionTimer = null;
      this._setState("ready");
      this._connection.resolve(Object.freeze({
        connected: true,
        origin: this._origin,
        frame: this._frame,
      }));
      return;
    }

    if (data.kind === "event") {
      safeCloseBitmap(responseBitmap);
      if (data.event !== "device-lost") return;
      const notification = Object.freeze({
        event: "device-lost",
        error: remoteBridgeError(data.error, "device-lost"),
        stats: data.stats,
      });
      Promise.resolve().then(() => this._onEvent(notification)).catch((error) => {
        try { this._warn("neural frame event callback failed:", error); }
        catch {}
      });
      return;
    }
    if (data.kind !== "response" ||
        !Number.isSafeInteger(data.id) || data.id <= 0 ||
        typeof data.ok !== "boolean") {
      safeCloseBitmap(responseBitmap);
      this._fail(bridgeError(
        "protocol-error",
        "Neural frame sent a malformed response",
        { retryable: true },
      ));
      return;
    }

    const pending = this._pending.get(data.id);
    // A response can arrive after its bounded timeout fired. Its request ID is
    // no longer live, so it must not settle a newer operation.
    if (!pending) {
      safeCloseBitmap(responseBitmap);
      return;
    }
    this._pending.delete(data.id);
    this._environment.clearTimeout(pending.timer);
    if (!data.ok) {
      safeCloseBitmap(responseBitmap);
      pending.reject(remoteBridgeError(data.error, pending.method));
      return;
    }
    if (pending.method === "run") {
      if (pending.runGeneration !== this._runGeneration) {
        safeCloseBitmap(responseBitmap);
        pending.reject(bridgeError(
          "cancelled",
          "Neural frame run was cancelled",
          { retryable: true, method: "run" },
        ));
        return;
      }
      try {
        pending.resolve(this._presentRunResult(data.result));
      } catch (error) {
        pending.reject(error);
        this._fail(error);
      }
      return;
    }
    if (responseBitmap) {
      safeCloseBitmap(responseBitmap);
      const error = bridgeError(
        "protocol-error",
        `Neural frame attached an output bitmap to ${pending.method}`,
        { retryable: true, method: pending.method },
      );
      pending.reject(error);
      this._fail(error);
      return;
    }
    pending.resolve(data.result);
  }

  async _request(
    method,
    payload,
    transfer = [],
    { allowDisposing = false, runGeneration = null } = {},
  ) {
    if (this._state === "idle" || this._state === "connecting") await this.start();
    const allowed = this._state === "ready" ||
      (allowDisposing && this._state === "disposing");
    if (!allowed || !this._port || this._frame?.isConnected === false) {
      throw bridgeError(
        this._state === "disposed" ? "disposed" : "bridge-closed",
        `Neural frame bridge is ${this._state}`,
      );
    }
    if (this._pending.size >= MAX_PENDING_REQUESTS) {
      throw bridgeError(
        "request-backpressure",
        "Too many Neural frame requests are pending",
        { retryable: true, method },
      );
    }

    const id = this._nextRequestId++;
    if (!Number.isSafeInteger(this._nextRequestId)) this._nextRequestId = 1;
    while (this._pending.has(this._nextRequestId)) this._nextRequestId++;
    const operation = deferred();
    const timeout = this._timeouts[method] ?? DEFAULT_TIMEOUTS.stop;
    const timer = this._environment.setTimeout(() => {
      if (!this._pending.has(id)) return;
      const error = bridgeError(
        "request-timeout",
        `Neural frame ${method} timed out`,
        { retryable: true, method },
      );
      this._fail(error);
    }, timeout);
    this._pending.set(id, {
      method,
      runGeneration,
      resolve: operation.resolve,
      reject: operation.reject,
      timer,
    });

    try {
      // The transfer list is always explicit. Only run transfers an input
      // ImageBitmap; the page-owned output canvas never crosses the OOPIF.
      this._port.postMessage({
        channel: NEURAL_FRAME_CHANNEL,
        kind: "request",
        instanceNonce: this._nonce,
        id,
        method,
        payload,
      }, transfer);
    } catch (error) {
      this._environment.clearTimeout(timer);
      this._pending.delete(id);
      const wrapped = bridgeError(
        "request-send-failed",
        boundedText(error?.message, `Neural frame ${method} could not be sent`),
        { cause: error, retryable: true, method },
      );
      operation.reject(wrapped);
      this._fail(wrapped);
    }
    return operation.promise;
  }

  async attachCanvas(canvas) {
    if (this._canvasAttachStarted) {
      throw bridgeError(
        "canvas-already-attached",
        "This Neural frame bridge already attached an output canvas",
      );
    }
    const CanvasCtor = this._environment.HTMLCanvasElement;
    if ((typeof CanvasCtor === "function" && !(canvas instanceof CanvasCtor)) ||
        !canvas || typeof canvas.getContext !== "function") {
      throw new TypeError("attachCanvas requires an HTMLCanvasElement");
    }

    await this.start();
    this._canvasAttachStarted = true;
    this._canvas = canvas;
    try {
      this._canvasPresenter = this._createCanvasPresenter(canvas);
    } catch (error) {
      const wrapped = error instanceof NeuralFrameBridgeError
        ? error
        : bridgeError(
            "canvas-context-unavailable",
            boundedText(error?.message, "Canvas presentation context is unavailable"),
            { cause: error },
          );
      this._fail(wrapped);
      throw wrapped;
    }

    try {
      const result = await this._request("attachCanvas", {});
      this._canvasAttached = true;
      return result;
    } catch (error) {
      if (this._state !== "failed" && this._state !== "disposed") this._fail(error);
      throw error;
    }
  }

  init(modelKey) {
    if (typeof modelKey !== "string" || !MODEL_KEY_PATTERN.test(modelKey)) {
      return Promise.reject(new TypeError("Neural model key is invalid"));
    }
    if (this._runPromise) {
      return Promise.reject(bridgeError(
        "run-busy",
        "Cannot initialize Neural while a frame is in flight",
        { retryable: true, method: "init" },
      ));
    }
    return this._request("init", { modelKey });
  }

  run(bitmap, { srcW, srcH, presentation } = {}) {
    if (this._runPromise) {
      return Promise.reject(bridgeError(
        "run-busy",
        "A Neural frame is already in flight",
        { retryable: true, method: "run" },
      ));
    }
    const BitmapCtor = this._environment.ImageBitmap;
    if ((typeof BitmapCtor === "function" && !(bitmap instanceof BitmapCtor)) ||
        !bitmap || typeof bitmap !== "object") {
      return Promise.reject(new TypeError("run requires an ImageBitmap"));
    }
    try {
      srcW = positiveDimension(srcW, "neural source width");
      srcH = positiveDimension(srcH, "neural source height");
      presentation = normalizePresentation(presentation);
    } catch (error) {
      return Promise.reject(error);
    }

    const operation = this._request(
      "run",
      { bitmap, srcW, srcH, presentation },
      [bitmap],
      { runGeneration: this._runGeneration },
    );
    this._runPromise = operation;
    operation.then(
      () => {
        if (this._runPromise === operation) this._runPromise = null;
      },
      () => {
        // If connection or posting failed, ownership never reached the child.
        // close() is also safe on an already-transferred (detached) bitmap.
        try { bitmap.close?.(); } catch {}
        if (this._runPromise === operation) this._runPromise = null;
      },
    );
    return operation;
  }

  cancel() {
    if (this._state === "disposed" || this._state === "failed") return false;
    try {
      return this._signalCancel();
    } catch (error) {
      this._fail(error);
      throw error;
    }
  }

  async stop() {
    if (this._disposePromise) {
      throw bridgeError("disposed", "Neural frame bridge is being disposed");
    }
    this.cancel();
    const running = this._runPromise;
    if (running) {
      try { await running; } catch {}
    }
    try {
      return await this._request("stop", {});
    } finally {
      this._releaseCanvasBacking();
    }
  }

  dispose() {
    if (this._disposePromise) return this._disposePromise;
    this._disposePromise = this._dispose();
    return this._disposePromise;
  }

  async _dispose() {
    if (this._state === "disposed") return { disposed: true };
    if (this._state === "idle" || this._state === "connecting" ||
        this._state === "failed") {
      const error = this._state === "connecting"
        ? bridgeError("disposed", "Neural frame bridge was disposed during its handshake")
        : null;
      this._cleanup(error, "disposed");
      return { disposed: true };
    }

    try {
      this.cancel();
    } catch (error) {
      this._cleanup(error, "disposed");
      throw error;
    }
    this._setState("disposing");
    const running = this._runPromise;
    if (running) {
      try { await running; } catch {}
    }
    if (this._state !== "disposing") {
      this._cleanup(null, "disposed");
      return { disposed: true };
    }

    let result;
    let failure;
    try {
      result = await this._request("dispose", {}, [], { allowDisposing: true });
    } catch (error) {
      failure = error;
    }
    this._cleanup(failure, "disposed");
    if (failure) throw failure;
    return result ?? { disposed: true };
  }

  _fail(error, finalState = "failed") {
    if (this._state === "failed" || this._state === "disposed") return;
    const wrapped = error instanceof NeuralFrameBridgeError
      ? error
      : bridgeError(
          "bridge-failed",
          boundedText(error?.message, "Neural frame bridge failed"),
          { cause: error },
        );
    this._cleanup(wrapped, finalState);
  }

  _cleanup(error, finalState) {
    const { window, clearTimeout } = this._environment;
    clearTimeout(this._connectionTimer);
    this._connectionTimer = null;
    window.removeEventListener("message", this._handleWindowMessage);
    window.removeEventListener("pagehide", this._handlePageHide);

    try { this._observer?.disconnect?.(); } catch {}
    this._observer = null;

    if (this._port) {
      this._port.onmessage = null;
      this._port.onmessageerror = null;
      try { this._port.close?.(); } catch {}
      this._port = null;
    }

    const frame = this._frame;
    if (frame) {
      try { frame.removeEventListener("load", this._handleFrameLoad); } catch {}
      try { frame.removeEventListener("error", this._handleFrameError); } catch {}
      try { frame.remove?.(); } catch {}
    }
    this._frame = null;
    this._frameUrl = null;
    const host = this._host;
    if (host) {
      try { host.remove?.(); } catch {}
    }
    this._host = null;
    this._releaseCanvasBacking();
    this._canvasPresenter = null;
    this._canvas = null;
    this._canvasAttached = false;
    this._runPromise = null;

    for (const pending of this._pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error ?? bridgeError(
        "disposed",
        "Neural frame bridge was disposed",
        { method: pending.method },
      ));
    }
    this._pending.clear();

    if (this._state === "connecting" && this._connection) {
      this._connection.reject(error ?? bridgeError(
        "disposed",
        "Neural frame bridge was disposed during its handshake",
      ));
    }
    this._setState(finalState, error);
    this._settleClosed(error);
  }
}

#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { constants as fsConstants } from "node:fs";
import { access, mkdtemp, readFile, realpath, rm } from "node:fs/promises";
import http from "node:http";
import net from "node:net";
import { tmpdir } from "node:os";
import { basename, delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { GENERATED_MODEL_CATALOG } from "../fsrcnnx-model-catalog.js";
import { createValidationPlan, REFERENCE_VALIDATION_CHECKS } from "../fsrcnnx-validation.js";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXPECTED_VALIDATION_IDS = Object.freeze(
  createValidationPlan(GENERATED_MODEL_CATALOG).map(({ id }) => id),
);
const EXPECTED_CHECK_COUNT = EXPECTED_VALIDATION_IDS.length;
const NUMERICAL_VALIDATION_IDS = Object.freeze([
  ...REFERENCE_VALIDATION_CHECKS.map(({ id }) => id),
  ...GENERATED_MODEL_CATALOG.map(({ name }) => `${name}:inference`),
]);
const STARTUP_TIMEOUT_MS = 30_000;
const DISCOVERY_TIMEOUT_MS = 3_000;
const VALIDATION_TIMEOUT_MS = 240_000;
const CDP_TIMEOUT_MS = 15_000;
const HTTP_TIMEOUT_MS = 5_000;
const SHUTDOWN_TIMEOUT_MS = 5_000;
const MAX_PROTOCOL_MESSAGE_BYTES = 16 * 1024 * 1024;
const WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

const BROWSER_CANDIDATES = [
  "microsoft-edge-dev",
  "microsoft-edge-beta",
  "microsoft-edge",
  "google-chrome-stable",
  "google-chrome",
  "chromium",
  "chromium-browser",
  "/usr/bin/microsoft-edge-dev",
  "/usr/bin/microsoft-edge",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

function abortReason(signal) {
  if (signal?.reason instanceof Error) return signal.reason;
  return new Error(signal?.reason ? String(signal.reason) : "operation aborted");
}

function delay(milliseconds, signal = null) {
  if (signal?.aborted) return Promise.reject(abortReason(signal));
  return new Promise((resolveDelay, rejectDelay) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolveDelay();
    }, milliseconds);
    const onAbort = () => {
      clearTimeout(timer);
      rejectDelay(abortReason(signal));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function executable(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function resolveExecutable(candidate) {
  if (!candidate) return null;
  if (candidate.includes("/")) return (await executable(candidate)) ? resolve(candidate) : null;
  for (const directory of (process.env.PATH || "").split(delimiter)) {
    if (!directory) continue;
    const path = join(directory, candidate);
    if (await executable(path)) return path;
  }
  return null;
}

async function findBrowser() {
  const configured = process.env.FSRCNNX_BROWSER?.trim();
  if (configured) {
    const browser = await resolveExecutable(configured);
    if (!browser) throw new Error(`FSRCNNX_BROWSER is not executable: ${configured}`);
    return browser;
  }
  for (const candidate of BROWSER_CANDIDATES) {
    const browser = await resolveExecutable(candidate);
    if (browser) return browser;
  }
  throw new Error("no supported Edge/Chrome/Chromium executable found; set FSRCNNX_BROWSER");
}

function requestJson(url, { method = "GET", timeoutMs = HTTP_TIMEOUT_MS, signal = null } = {}) {
  if (signal?.aborted) return Promise.reject(abortReason(signal));
  return new Promise((resolveRequest, rejectRequest) => {
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", onAbort);
      callback(value);
    };
    const request = http.request(url, { method }, (response) => {
      const chunks = [];
      let size = 0;
      const responseFailure = (error) => finish(rejectRequest,
        error instanceof Error ? error : new Error("DevTools HTTP response ended prematurely"));
      response.on("error", responseFailure);
      response.on("aborted", () => responseFailure(new Error("DevTools HTTP response was aborted")));
      response.on("close", () => {
        if (!response.complete) responseFailure(new Error("DevTools HTTP response ended prematurely"));
      });
      response.on("data", (chunk) => {
        size += chunk.length;
        if (size > MAX_PROTOCOL_MESSAGE_BYTES) {
          request.destroy(new Error("DevTools HTTP response exceeded the size limit"));
          return;
        }
        chunks.push(chunk);
      });
      response.on("end", () => {
        if (response.statusCode == null || response.statusCode < 200 || response.statusCode >= 300) {
          finish(rejectRequest, new Error(`DevTools HTTP ${method} ${url.pathname} returned ${response.statusCode}`));
          return;
        }
        try {
          finish(resolveRequest, JSON.parse(Buffer.concat(chunks).toString("utf8")));
        } catch (error) {
          finish(rejectRequest, new Error(`invalid DevTools JSON response: ${error.message}`));
        }
      });
    });
    const onAbort = () => request.destroy(abortReason(signal));
    signal?.addEventListener("abort", onAbort, { once: true });
    request.setTimeout(timeoutMs, () => request.destroy(new Error(`DevTools HTTP request timed out after ${timeoutMs} ms`)));
    request.on("error", (error) => finish(rejectRequest, error));
    request.end();
  });
}

class RawWebSocket {
  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.fragments = [];
    this.fragmentBytes = 0;
    this.fragmentOpcode = null;
    this.closed = false;
    this.messageListeners = new Set();
    this.closeListeners = new Set();
    socket.on("data", (chunk) => this.receive(chunk));
    socket.on("error", (error) => this.finish(error));
    socket.on("end", () => this.finish(new Error("DevTools WebSocket ended")));
    socket.on("close", () => this.finish(new Error("DevTools WebSocket closed")));
  }

  static connect(address, { timeoutMs = CDP_TIMEOUT_MS, signal = null } = {}) {
    if (signal?.aborted) return Promise.reject(abortReason(signal));
    const endpoint = new URL(address);
    if (endpoint.protocol !== "ws:") throw new Error(`unsupported DevTools protocol ${endpoint.protocol}`);
    return new Promise((resolveSocket, rejectSocket) => {
      const socket = net.createConnection({
        host: endpoint.hostname,
        port: Number(endpoint.port || 80),
      });
      const key = randomBytes(16).toString("base64");
      const expectedAccept = createHash("sha1").update(key + WEBSOCKET_GUID).digest("base64");
      let handshake = Buffer.alloc(0);
      let settled = false;
      const timer = setTimeout(() => fail(new Error(`DevTools WebSocket connection timed out after ${timeoutMs} ms`)), timeoutMs);
      const cleanup = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        socket.removeListener("error", fail);
        socket.removeListener("data", onHandshakeData);
        socket.removeListener("connect", onConnect);
      };
      const fail = (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        socket.destroy();
        rejectSocket(error);
      };
      const onAbort = () => fail(abortReason(signal));
      const onConnect = () => {
        const path = `${endpoint.pathname || "/"}${endpoint.search}`;
        socket.write([
          `GET ${path} HTTP/1.1`,
          `Host: ${endpoint.host}`,
          "Upgrade: websocket",
          "Connection: Upgrade",
          `Sec-WebSocket-Key: ${key}`,
          "Sec-WebSocket-Version: 13",
          "Origin: http://127.0.0.1",
          "",
          "",
        ].join("\r\n"));
      };
      const onHandshakeData = (chunk) => {
        handshake = Buffer.concat([handshake, chunk]);
        if (handshake.length > 64 * 1024) {
          fail(new Error("DevTools WebSocket handshake exceeded the size limit"));
          return;
        }
        const boundary = handshake.indexOf("\r\n\r\n");
        if (boundary < 0) return;
        const header = handshake.subarray(0, boundary).toString("latin1");
        const lines = header.split("\r\n");
        if (!/^HTTP\/1\.[01] 101\b/.test(lines[0])) {
          fail(new Error(`DevTools WebSocket upgrade failed: ${lines[0] || "empty response"}`));
          return;
        }
        const headers = new Map();
        for (const line of lines.slice(1)) {
          const separator = line.indexOf(":");
          if (separator > 0) headers.set(line.slice(0, separator).trim().toLowerCase(), line.slice(separator + 1).trim());
        }
        if (headers.get("sec-websocket-accept") !== expectedAccept) {
          fail(new Error("DevTools WebSocket returned an invalid accept key"));
          return;
        }
        const remainder = handshake.subarray(boundary + 4);
        settled = true;
        cleanup();
        socket.setNoDelay(true);
        const webSocket = new RawWebSocket(socket);
        if (remainder.length) webSocket.receive(remainder);
        resolveSocket(webSocket);
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      socket.once("error", fail);
      socket.once("connect", onConnect);
      socket.on("data", onHandshakeData);
    });
  }

  onMessage(listener) { this.messageListeners.add(listener); }
  onClose(listener) { this.closeListeners.add(listener); }

  send(text) {
    if (this.closed) throw new Error("DevTools WebSocket is closed");
    this.sendFrame(0x1, Buffer.from(text, "utf8"));
  }

  sendFrame(opcode, payload) {
    const mask = randomBytes(4);
    let lengthBytes;
    if (payload.length < 126) {
      lengthBytes = Buffer.from([0x80 | payload.length]);
    } else if (payload.length <= 0xffff) {
      lengthBytes = Buffer.alloc(3);
      lengthBytes[0] = 0x80 | 126;
      lengthBytes.writeUInt16BE(payload.length, 1);
    } else {
      lengthBytes = Buffer.alloc(9);
      lengthBytes[0] = 0x80 | 127;
      lengthBytes.writeBigUInt64BE(BigInt(payload.length), 1);
    }
    const encoded = Buffer.alloc(payload.length);
    for (let index = 0; index < payload.length; index++) encoded[index] = payload[index] ^ mask[index % 4];
    this.socket.write(Buffer.concat([Buffer.from([0x80 | opcode]), lengthBytes, mask, encoded]));
  }

  receive(chunk) {
    if (this.closed) return;
    this.buffer = Buffer.concat([this.buffer, chunk]);
    try {
      while (this.readFrame()) {}
    } catch (error) {
      this.socket.destroy();
      this.finish(error);
    }
  }

  readFrame() {
    if (this.buffer.length < 2) return false;
    const first = this.buffer[0];
    const second = this.buffer[1];
    const final = (first & 0x80) !== 0;
    const opcode = first & 0x0f;
    const masked = (second & 0x80) !== 0;
    let length = second & 0x7f;
    let offset = 2;
    if (length === 126) {
      if (this.buffer.length < offset + 2) return false;
      length = this.buffer.readUInt16BE(offset);
      offset += 2;
    } else if (length === 127) {
      if (this.buffer.length < offset + 8) return false;
      const extended = this.buffer.readBigUInt64BE(offset);
      if (extended > BigInt(MAX_PROTOCOL_MESSAGE_BYTES)) throw new Error("DevTools WebSocket frame exceeded the size limit");
      length = Number(extended);
      offset += 8;
    }
    if (length > MAX_PROTOCOL_MESSAGE_BYTES) throw new Error("DevTools WebSocket frame exceeded the size limit");
    const maskBytes = masked ? 4 : 0;
    if (this.buffer.length < offset + maskBytes + length) return false;
    const mask = masked ? this.buffer.subarray(offset, offset + 4) : null;
    offset += maskBytes;
    const payload = Buffer.from(this.buffer.subarray(offset, offset + length));
    this.buffer = this.buffer.subarray(offset + length);
    if (mask) {
      for (let index = 0; index < payload.length; index++) payload[index] ^= mask[index % 4];
    }
    this.handleFrame(opcode, final, payload);
    return this.buffer.length >= 2;
  }

  handleFrame(opcode, final, payload) {
    if (opcode === 0x8) {
      this.finish(new Error("DevTools WebSocket closed"));
      this.socket.end();
      return;
    }
    if (opcode === 0x9) {
      this.sendFrame(0xA, payload);
      return;
    }
    if (opcode === 0xA) return;
    if (opcode === 0x1) {
      if (this.fragmentOpcode != null) throw new Error("overlapping DevTools WebSocket messages");
      if (final) {
        this.emitMessage(payload.toString("utf8"));
        return;
      }
      this.fragmentOpcode = opcode;
      this.fragments = [payload];
      this.fragmentBytes = payload.length;
      return;
    }
    if (opcode === 0x0) {
      if (this.fragmentOpcode == null) throw new Error("unexpected DevTools WebSocket continuation frame");
      this.fragmentBytes += payload.length;
      if (this.fragmentBytes > MAX_PROTOCOL_MESSAGE_BYTES) throw new Error("DevTools WebSocket message exceeded the size limit");
      this.fragments.push(payload);
      if (final) {
        const message = Buffer.concat(this.fragments, this.fragmentBytes).toString("utf8");
        this.fragmentOpcode = null;
        this.fragments = [];
        this.fragmentBytes = 0;
        this.emitMessage(message);
      }
      return;
    }
    throw new Error(`unsupported DevTools WebSocket opcode ${opcode}`);
  }

  emitMessage(message) {
    for (const listener of this.messageListeners) listener(message);
  }

  finish(error) {
    if (this.closed) return;
    this.closed = true;
    for (const listener of this.closeListeners) listener(error);
    this.messageListeners.clear();
    this.closeListeners.clear();
  }

  close() {
    if (this.closed) return;
    try { this.sendFrame(0x8, Buffer.from([0x03, 0xE8])); } catch {}
    this.socket.end();
    this.finish(new Error("DevTools WebSocket closed by client"));
  }
}

class CdpClient {
  constructor(webSocket) {
    this.webSocket = webSocket;
    this.nextId = 1;
    this.pending = new Map();
    this.eventListeners = new Set();
    webSocket.onMessage((message) => this.receive(message));
    webSocket.onClose((error) => this.closePending(error));
  }

  static async connect(address, options) {
    return new CdpClient(await RawWebSocket.connect(address, options));
  }

  onEvent(listener) { this.eventListeners.add(listener); }

  receive(payload) {
    let message;
    try {
      message = JSON.parse(payload);
    } catch (error) {
      this.closePending(new Error(`invalid DevTools protocol message: ${error.message}`));
      return;
    }
    if (message.id != null) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(`CDP ${pending.method}: ${message.error.message}`));
      else pending.resolve(message.result || {});
      return;
    }
    if (message.method) {
      for (const listener of this.eventListeners) {
        try { listener(message); } catch {}
      }
    }
  }

  send(method, params = {}, timeoutMs = CDP_TIMEOUT_MS) {
    const id = this.nextId++;
    return new Promise((resolveCommand, rejectCommand) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        rejectCommand(new Error(`CDP ${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending.set(id, { method, resolve: resolveCommand, reject: rejectCommand, timer });
      try {
        this.webSocket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        rejectCommand(error);
      }
    });
  }

  closePending(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  close() { this.webSocket.close(); }
}

async function evaluate(client, expression, timeoutMs = CDP_TIMEOUT_MS) {
  const response = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  }, timeoutMs);
  if (response.exceptionDetails) {
    const detail = response.exceptionDetails.exception?.description || response.exceptionDetails.text || "evaluation failed";
    throw new Error(detail);
  }
  return response.result?.value;
}

function spawnBrowser(browser, profile, extensionRoot) {
  const browserArguments = [
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    "--remote-allow-origins=*",
    `--load-extension=${extensionRoot}`,
    `--disable-extensions-except=${extensionRoot}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "--disable-features=msEdgeSidebarV2",
    "--window-size=1280,900",
    "--enable-unsafe-webgpu",
    "--use-angle=swiftshader",
    "--use-vulkan=swiftshader",
    "--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE",
    "--disable-vulkan-surface",
    "about:blank",
  ];
  let command = browser;
  let args = browserArguments;
  if (!process.env.DISPLAY) {
    command = null;
    args = null;
  }

  return { command, args, browserArguments };
}

async function startBrowser(browser, profile, extensionRoot, signal) {
  if (signal?.aborted) throw abortReason(signal);
  const launch = spawnBrowser(browser, profile, extensionRoot);
  let command = launch.command;
  let args = launch.args;
  if (!command) {
    const xvfbRun = await resolveExecutable("xvfb-run");
    if (!xvfbRun) throw new Error("DISPLAY is unset and xvfb-run is unavailable");
    command = xvfbRun;
    args = ["-a", browser, ...launch.browserArguments];
  }
  if (signal?.aborted) throw abortReason(signal);
  const child = spawn(command, args, {
    cwd: PROJECT_ROOT,
    env: process.env,
    detached: process.platform !== "win32",
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  let probe = "";
  let settled = false;
  let resolveEndpoint;
  let rejectEndpoint;
  const endpoint = new Promise((resolvePromise, rejectPromise) => {
    resolveEndpoint = resolvePromise;
    rejectEndpoint = rejectPromise;
  });
  const append = (chunk) => {
    const text = chunk.toString("utf8");
    output = (output + text).slice(-64 * 1024);
    probe = (probe + text).slice(-8 * 1024);
    const match = probe.match(/DevTools listening on (ws:\/\/[^\s]+)/);
    if (match && !settled) {
      settled = true;
      cleanupWait();
      resolveEndpoint(match[1]);
    }
  };
  child.stdout.on("data", append);
  child.stderr.on("data", append);
  const timer = setTimeout(() => failStart(new Error(`browser startup timed out after ${STARTUP_TIMEOUT_MS} ms`)), STARTUP_TIMEOUT_MS);
  const onAbort = () => failStart(abortReason(signal));
  const onExit = (code, exitSignal) => failStart(new Error(
    `browser exited before DevTools became ready (code ${code ?? "none"}, signal ${exitSignal || "none"})`,
  ));
  function cleanupWait() {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    child.removeListener("exit", onExit);
  }
  function failStart(error) {
    if (settled) return;
    settled = true;
    cleanupWait();
    rejectEndpoint(error);
  }
  signal?.addEventListener("abort", onAbort, { once: true });
  child.once("exit", onExit);
  child.once("error", failStart);
  return { child, endpoint, output: () => output };
}

async function terminateBrowser(child) {
  if (!child) return;
  const alive = () => {
    if (process.platform === "win32") return child.exitCode == null && child.signalCode == null;
    if (!Number.isInteger(child.pid)) return false;
    try {
      process.kill(-child.pid, 0);
      return true;
    } catch (error) {
      return error?.code !== "ESRCH";
    }
  };
  const kill = (signal) => {
    try {
      if (process.platform === "win32") child.kill(signal);
      else process.kill(-child.pid, signal);
    } catch {}
  };
  const waitUntilGone = async (timeoutMs) => {
    const deadline = Date.now() + timeoutMs;
    while (alive() && Date.now() < deadline) await delay(50);
    return !alive();
  };
  if (!alive()) return;
  kill("SIGTERM");
  if (await waitUntilGone(SHUTDOWN_TIMEOUT_MS)) return;
  kill("SIGKILL");
  if (!(await waitUntilGone(SHUTDOWN_TIMEOUT_MS))) {
    throw new Error(`browser process group ${child.pid} survived SIGKILL`);
  }
}

function httpBaseFromWebSocket(address) {
  const endpoint = new URL(address);
  return new URL(`http://${endpoint.host}/`);
}

async function listTargets(httpBase, signal) {
  return requestJson(new URL("json/list", httpBase), { signal });
}

function extensionIdFromPath(extensionRoot) {
  const digest = createHash("sha256").update(extensionRoot).digest().subarray(0, 16);
  let id = "";
  for (const byte of digest) {
    id += String.fromCharCode(97 + (byte >>> 4));
    id += String.fromCharCode(97 + (byte & 0x0f));
  }
  return id;
}

async function discoverExtension(httpBase, expectedName, extensionRoot, manifestKey, signal) {
  const deadline = Date.now() + DISCOVERY_TIMEOUT_MS;
  let lastError = null;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw abortReason(signal);
    let targets;
    try {
      targets = await listTargets(httpBase, signal);
    } catch (error) {
      lastError = error;
      await delay(200, signal);
      continue;
    }
    const candidates = targets.filter((target) => target.type === "service_worker" &&
      /^chrome-extension:\/\/[a-p]{32}\/background\.js(?:[?#]|$)/.test(target.url || "") &&
      target.webSocketDebuggerUrl);
    for (const target of candidates) {
      let client = null;
      try {
        const remaining = () => Math.max(1, deadline - Date.now());
        client = await CdpClient.connect(target.webSocketDebuggerUrl, {
          signal,
          timeoutMs: Math.min(CDP_TIMEOUT_MS, remaining()),
        });
        await client.send("Runtime.enable", {}, Math.min(CDP_TIMEOUT_MS, remaining()));
        const name = await evaluate(
          client,
          "chrome.runtime.getManifest().name",
          Math.min(CDP_TIMEOUT_MS, remaining()),
        );
        if (name !== expectedName) continue;
        const extensionId = new URL(target.url).hostname;
        if (!/^[a-p]{32}$/.test(extensionId)) throw new Error(`invalid extension ID ${extensionId}`);
        return { extensionId, source: "service worker" };
      } catch (error) {
        lastError = error;
      } finally {
        client?.close();
      }
    }
    await delay(200, signal);
  }
  // MV3 workers are allowed to stop before the first target query. Chromium
  // deterministically assigns an unpacked extension ID from the canonical
  // extension path when the manifest has no explicit key. The opened extension
  // page independently verifies this fallback ID and the manifest name before
  // any validation state is trusted.
  if (manifestKey) {
    const suffix = lastError ? `: ${lastError.message}` : "";
    throw new Error(`the service worker was unavailable and path-derived IDs do not apply to keyed manifests${suffix}`);
  }
  const extensionId = extensionIdFromPath(extensionRoot);
  if (!/^[a-p]{32}$/.test(extensionId)) {
    const suffix = lastError ? `: ${lastError.message}` : "";
    throw new Error(`could not discover or derive the unpacked extension ID${suffix}`);
  }
  return { extensionId, source: "canonical path" };
}

function remoteObjectText(object) {
  if (Object.prototype.hasOwnProperty.call(object || {}, "value")) {
    if (typeof object.value === "string") return object.value;
    try { return JSON.stringify(object.value); } catch {}
  }
  return object?.unserializableValue || object?.description || object?.type || "unknown";
}

function collectRuntimeEvent(events, message) {
  if (message.method === "Runtime.consoleAPICalled") {
    events.push({
      kind: "console",
      type: message.params.type,
      text: (message.params.args || []).map(remoteObjectText).join(" "),
    });
  } else if (message.method === "Runtime.exceptionThrown") {
    const detail = message.params.exceptionDetails || {};
    events.push({
      kind: "exception",
      type: "exception",
      text: detail.exception?.description || detail.text || "uncaught exception",
    });
  }
  if (events.length > 500) events.shift();
}

function assertValidationResult(state, events) {
  const problems = [];
  if (!state || typeof state !== "object") problems.push("validation did not publish a result object");
  else {
    if (state.done !== true) problems.push("result is not final");
    if (state.complete !== true) problems.push("result is not complete");
    if (state.ok !== true) problems.push("result is not all-pass");
    if (state.total !== EXPECTED_CHECK_COUNT) problems.push(
      `expected ${EXPECTED_CHECK_COUNT} checks, received ${state.total}`,
    );
    if (state.pass !== EXPECTED_CHECK_COUNT) problems.push(
      `expected ${EXPECTED_CHECK_COUNT} passes, received ${state.pass}`,
    );
    for (const [field, value] of [["fail", 0], ["skip", 0], ["pending", 0]]) {
      if (state[field] !== value) problems.push(`expected ${field}=${value}, received ${state[field]}`);
    }
    if (!Array.isArray(state.results) || state.results.length !== EXPECTED_CHECK_COUNT) {
      problems.push(`expected ${EXPECTED_CHECK_COUNT} individual results`);
    } else {
      const ids = new Set();
      for (const result of state.results) {
        if (!result || typeof result.id !== "string" || !result.id) problems.push("result has no stable ID");
        else ids.add(result.id);
        if (result?.status !== "pass") {
          const detail = typeof result?.detail === "string" && result.detail ? `: ${result.detail}` : "";
          problems.push(`${result?.id || "unknown check"} is ${result?.status || "missing"}${detail}`);
        }
      }
      if (ids.size !== EXPECTED_CHECK_COUNT) problems.push("result IDs are not unique and fixed");
      const missing = EXPECTED_VALIDATION_IDS.filter((id) => !ids.has(id));
      const expectedIds = new Set(EXPECTED_VALIDATION_IDS);
      const unexpected = [...ids].filter((id) => !expectedIds.has(id));
      if (missing.length) problems.push(`missing expected result IDs: ${missing.join(", ")}`);
      if (unexpected.length) problems.push(`unexpected result IDs: ${unexpected.join(", ")}`);
    }
  }
  const runtimeFailures = events.filter((event) => event.kind === "exception" ||
    (event.kind === "console" && ["error", "warning", "assert"].includes(event.type)));
  if (runtimeFailures.length) {
    problems.push(`${runtimeFailures.length} console warning/error/uncaught exception event(s) observed`);
  }
  if (problems.length) {
    const error = new Error(`browser validation failed: ${problems.join("; ")}`);
    error.browserEvents = events;
    throw error;
  }
}

async function runValidation(httpBase, extensionId, expectedName, signal) {
  const extensionUrl = `chrome-extension://${extensionId}/validate.html?autorun=1`;
  // Attach before navigation so import failures and synchronous startup
  // exceptions cannot occur before the Runtime event stream is enabled.
  const targetUrl = new URL(`/json/new?${encodeURIComponent("about:blank")}`, httpBase);
  const target = await requestJson(targetUrl, { method: "PUT", signal });
  if (!target.webSocketDebuggerUrl) throw new Error("validation target has no DevTools WebSocket URL");
  const events = [];
  let client = null;
  try {
    client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
    client.onEvent((message) => collectRuntimeEvent(events, message));
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    await client.send("Page.bringToFront");
    const navigation = await client.send("Page.navigate", { url: extensionUrl });
    if (navigation.errorText) throw new Error(`validation navigation failed: ${navigation.errorText}`);
    const deadline = Date.now() + VALIDATION_TIMEOUT_MS;
    let state = null;
    while (Date.now() < deadline) {
      if (signal?.aborted) throw abortReason(signal);
      let probe;
      try {
        probe = await evaluate(client, `(() => ({
          href: location.href,
          title: document.title,
          readyState: document.readyState,
          runtimeId: globalThis.chrome?.runtime?.id || null,
          manifestName: globalThis.chrome?.runtime?.getManifest?.().name || null,
          validation: window.__FSRCNNX_VALIDATION__ || null,
        }))()`);
      } catch (error) {
        if (/execution context|context.*destroyed|cannot find context/i.test(error.message)) {
          await delay(100, signal);
          continue;
        }
        throw error;
      }
      if (probe?.href === "about:blank") {
        await delay(100, signal);
        continue;
      }
      if (!probe?.href?.startsWith(`chrome-extension://${extensionId}/validate.html`)) {
        throw new Error(`validation page failed to load (current URL: ${probe?.href || "unknown"})`);
      }
      if (probe.runtimeId !== extensionId || probe.manifestName !== expectedName) {
        if (probe.readyState === "loading") {
          await delay(100, signal);
          continue;
        }
        throw new Error(
          `extension identity verification failed (ID ${probe.runtimeId || "missing"}, ` +
          `manifest ${probe.manifestName || "missing"})`,
        );
      }
      state = probe.validation;
      if (state?.done === true) break;
      await delay(200, signal);
    }
    if (!state?.done) throw new Error(`browser validation timed out after ${VALIDATION_TIMEOUT_MS} ms`);
    await delay(50, signal);
    try {
      assertValidationResult(state, events);
    } catch (error) {
      error.validationState = state;
      throw error;
    }
    return { state, events };
  } catch (error) {
    if (!error.browserEvents) error.browserEvents = events;
    throw error;
  } finally {
    client?.close();
  }
}

async function runPopupSmoke(httpBase, extensionId, expectedName, signal) {
  const popupUrl = `chrome-extension://${extensionId}/popup.html`;
  const targetUrl = new URL(`/json/new?${encodeURIComponent("about:blank")}`, httpBase);
  const target = await requestJson(targetUrl, { method: "PUT", signal });
  if (!target.webSocketDebuggerUrl) throw new Error("popup smoke target has no DevTools WebSocket URL");
  const events = [];
  let client = null;
  try {
    client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
    client.onEvent((message) => collectRuntimeEvent(events, message));
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    await client.send("Page.bringToFront");
    const navigation = await client.send("Page.navigate", { url: popupUrl });
    if (navigation.errorText) throw new Error(`popup navigation failed: ${navigation.errorText}`);

    const deadline = Date.now() + CDP_TIMEOUT_MS;
    let state = null;
    while (Date.now() < deadline) {
      if (signal?.aborted) throw abortReason(signal);
      try {
        state = await evaluate(client, `(() => {
          const controls = [...document.querySelectorAll("button, input, select")];
          return {
            href: location.href,
            readyState: document.readyState,
            runtimeId: globalThis.chrome?.runtime?.id || null,
            manifestName: globalThis.chrome?.runtime?.getManifest?.().name || null,
            scriptType: document.querySelector('script[src="popup.js"]')?.type || null,
            webgpuText: document.getElementById("s-webgpu")?.textContent || "",
            operationText: document.getElementById("operation-status")?.textContent || "",
            controlCount: controls.length,
            disabledCount: controls.filter((control) => control.disabled).length,
            modeStates: [...document.querySelectorAll(".modes button")].map((button) => ({
              mode: button.dataset.mode,
              pressed: button.getAttribute("aria-pressed"),
            })),
            missingIds: [
              "s-webgpu", "s-video", "s-model", "s-frames", "runtime-status",
              "drm-banner", "operation-status", "engine", "policy", "interpolate",
            ].filter((id) => !document.getElementById(id)),
          };
        })()`);
      } catch (error) {
        if (/execution context|context.*destroyed|cannot find context/i.test(error.message)) {
          await delay(100, signal);
          continue;
        }
        throw error;
      }
      if (state?.href === popupUrl && state.readyState === "complete" && state.operationText) break;
      await delay(100, signal);
    }

    const problems = [];
    if (state?.href !== popupUrl) problems.push(`popup failed to load (${state?.href || "unknown URL"})`);
    if (state?.runtimeId !== extensionId || state?.manifestName !== expectedName) {
      problems.push("popup extension identity is incorrect");
    }
    if (state?.scriptType !== "module") problems.push("popup module script did not load");
    if (state?.webgpuText !== "unavailable") problems.push("popup unsupported-page state was not rendered");
    if (!/open an http, https, or permitted local file page/i.test(state?.operationText || "")) {
      problems.push("popup unsupported-page guidance is missing");
    }
    if (!Number.isInteger(state?.controlCount) || state.controlCount < 20 ||
        state.disabledCount !== state.controlCount) {
      problems.push("popup did not disable all controls on an unsupported active page");
    }
    if (state?.modeStates?.length !== 3 ||
        state.modeStates.filter(({ pressed }) => pressed === "true").length !== 1 ||
        state.modeStates.find(({ pressed }) => pressed === "true")?.mode !== "off") {
      problems.push("popup mode accessibility state is invalid");
    }
    if (!Array.isArray(state?.missingIds) || state.missingIds.length) {
      problems.push(`popup is missing required elements: ${(state?.missingIds || []).join(", ")}`);
    }
    const runtimeFailures = events.filter((event) => event.kind === "exception" ||
      (event.kind === "console" && ["error", "warning", "assert"].includes(event.type)));
    if (runtimeFailures.length) problems.push(`${runtimeFailures.length} popup runtime failure event(s) observed`);
    if (problems.length) {
      const error = new Error(`popup browser smoke failed: ${problems.join("; ")}`);
      error.browserEvents = events;
      throw error;
    }
    return state;
  } catch (error) {
    if (!error.browserEvents) error.browserEvents = events;
    throw error;
  } finally {
    client?.close();
  }
}

function diagnostics(events) {
  if (!Array.isArray(events) || !events.length) return "";
  return events.slice(-20).map((event) => `  ${event.kind}/${event.type}: ${event.text}`).join("\n");
}

function numericalDiagnostics(state) {
  if (!Array.isArray(state?.results)) return "";
  const byId = new Map(state.results.map((result) => [result.id, result]));
  return NUMERICAL_VALIDATION_IDS
    .map((id) => byId.get(id))
    .filter(Boolean)
    .map((result) => `  ${result.id}: ${result.status} — ${result.detail}`)
    .join("\n");
}

async function main(signal) {
  const profile = await mkdtemp(join(tmpdir(), "fsrcnnx-browser-validation-"));
  let launched = null;
  let primaryError = null;
  try {
    const [browser, manifestSource, extensionRoot] = await Promise.all([
      findBrowser(),
      readFile(join(PROJECT_ROOT, "manifest.json"), "utf8"),
      realpath(PROJECT_ROOT),
    ]);
    const manifest = JSON.parse(manifestSource);
    if (manifest.manifest_version !== 3 || manifest.background?.service_worker !== "background.js") {
      throw new Error("browser validator requires the project's MV3 background.js service worker");
    }
    launched = await startBrowser(browser, profile, extensionRoot, signal);
    const browserWebSocket = await launched.endpoint;
    const httpBase = httpBaseFromWebSocket(browserWebSocket);
    const discovery = await discoverExtension(httpBase, manifest.name, extensionRoot, manifest.key, signal);
    await runPopupSmoke(httpBase, discovery.extensionId, manifest.name, signal);
    const { state } = await runValidation(httpBase, discovery.extensionId, manifest.name, signal);
    const webGpu = state.results.find((result) => result.id === "webgpu");
    console.log(
      `Browser validation passed: ${state.pass}/${state.total} checks ` +
      `(${basename(browser)}, ID from ${discovery.source}).`,
    );
    if (webGpu?.detail) console.log(`WebGPU: ${webGpu.detail}`);
    const numericalOutput = numericalDiagnostics(state);
    if (numericalOutput) console.log(`Numerical references:\n${numericalOutput}`);
    console.log("Popup browser smoke passed: module, unavailable state, controls, and accessibility.");
  } catch (error) {
    primaryError = error;
    if (launched?.output()) error.browserOutput = launched.output();
    throw error;
  } finally {
    try {
      await terminateBrowser(launched?.child);
    } catch (error) {
      if (!primaryError) throw error;
      console.error(`Browser cleanup warning: ${error.message}`);
    } finally {
      try {
        await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
      } catch (error) {
        if (!primaryError) throw error;
        console.error(`Profile cleanup warning: ${error.message}`);
      }
    }
  }
}

const abortController = new AbortController();
let receivedSignal = null;
const signalHandlers = new Map();
for (const name of ["SIGINT", "SIGTERM"]) {
  const handler = () => {
    receivedSignal = name;
    abortController.abort(new Error(`received ${name}`));
  };
  signalHandlers.set(name, handler);
  process.once(name, handler);
}

try {
  await main(abortController.signal);
} catch (error) {
  console.error(error.message || String(error));
  const numericalOutput = numericalDiagnostics(error.validationState);
  if (numericalOutput) console.error(`Numerical references:\n${numericalOutput}`);
  const eventOutput = diagnostics(error.browserEvents);
  if (eventOutput) console.error(`Validation page diagnostics:\n${eventOutput}`);
  if (error.browserOutput) console.error(`Browser output:\n${error.browserOutput.trim()}`);
  process.exitCode = receivedSignal === "SIGINT" ? 130 : receivedSignal === "SIGTERM" ? 143 : 1;
} finally {
  for (const [name, handler] of signalHandlers) process.removeListener(name, handler);
}

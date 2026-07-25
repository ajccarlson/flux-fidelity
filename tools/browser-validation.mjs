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

import { GENERATED_MODEL_CATALOG } from "../src/core/fsrcnnx-model-catalog.js";
import { createValidationPlan, REFERENCE_VALIDATION_CHECKS } from "../validation/fsrcnnx-validation.js";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE_ROOT = join(PROJECT_ROOT, "tests", "fixtures", "browser");
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
const INTEGRATION_TIMEOUT_MS = 120_000;
const CDP_TIMEOUT_MS = 15_000;
const HTTP_TIMEOUT_MS = 5_000;
const SHUTDOWN_TIMEOUT_MS = 5_000;
const MAX_PROTOCOL_MESSAGE_BYTES = 16 * 1024 * 1024;
const WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const FIXTURE_CONTENT_SECURITY_POLICY =
  "default-src 'none'; script-src 'self'; style-src 'self'; media-src 'self'; " +
  "base-uri 'none'; frame-ancestors 'none'";

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

const FIXTURE_ROUTES = Object.freeze(new Map([
  ["/", ["video.html", "text/html; charset=utf-8"]],
  ["/video.html", ["video.html", "text/html; charset=utf-8"]],
  ["/video-fixture.js", ["video-fixture.js", "text/javascript; charset=utf-8"]],
  ["/fixture-manifest.json", ["fixture-manifest.json", "application/json; charset=utf-8"]],
  ["/media/bt709-a.webm", ["media/bt709-a.webm", "video/webm"]],
  ["/media/bt709-b.webm", ["media/bt709-b.webm", "video/webm"]],
  ["/media/bt2020-pq.webm", ["media/bt2020-pq.webm", "video/webm"]],
  ["/media/bt2020-sdr.webm", ["media/bt2020-sdr.webm", "video/webm"]],
]));

function parseArguments(argv) {
  let extensionRoot = PROJECT_ROOT;
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === "--extension-root") {
      const value = argv[++index];
      if (!value || value.startsWith("--")) throw new Error("--extension-root requires a path");
      extensionRoot = resolve(PROJECT_ROOT, value);
      continue;
    }
    throw new Error(`unknown browser-validation argument: ${argument}`);
  }
  return Object.freeze({ extensionRoot });
}

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

async function browserVersion(browser, signal) {
  if (signal?.aborted) throw abortReason(signal);
  return new Promise((resolveVersion, rejectVersion) => {
    const child = spawn(browser, ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    const append = (chunk) => { output = (output + chunk.toString("utf8")).slice(-8 * 1024); };
    child.stdout.on("data", append);
    child.stderr.on("data", append);
    const timer = setTimeout(() => child.kill("SIGKILL"), HTTP_TIMEOUT_MS);
    const onAbort = () => child.kill("SIGKILL");
    signal?.addEventListener("abort", onAbort, { once: true });
    child.once("error", (error) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      rejectVersion(error);
    });
    child.once("exit", (code, exitSignal) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      if (signal?.aborted) rejectVersion(abortReason(signal));
      else if (code === 0 && output.trim()) resolveVersion(output.trim());
      else rejectVersion(new Error(
        `browser version probe failed (code ${code ?? "none"}, signal ${exitSignal || "none"})`,
      ));
    });
  });
}

function rangeForRequest(header, length) {
  if (!header) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || (!match[1] && !match[2])) return false;
  let start;
  let end;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return false;
    start = Math.max(0, length - suffix);
    end = length - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : length - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
      start < 0 || end < start || start >= length) return false;
  return { start, end: Math.min(end, length - 1) };
}

async function startFixtureServer(signal) {
  const files = new Map();
  for (const [route, [relativePath, type]] of FIXTURE_ROUTES) {
    files.set(route, { bytes: await readFile(join(FIXTURE_ROOT, relativePath)), type });
  }

  const server = http.createServer((request, response) => {
    try {
      const method = request.method || "GET";
      const url = new URL(request.url || "/", "http://127.0.0.1");
      const fixture = url.search || url.hash || (method !== "GET" && method !== "HEAD")
        ? null
        : files.get(url.pathname);
      if (!fixture) {
        response.writeHead(404, {
          "Cache-Control": "no-store",
          "Content-Type": "text/plain; charset=utf-8",
          "X-Content-Type-Options": "nosniff",
        });
        response.end(method === "HEAD" ? undefined : "Not found\n");
        return;
      }

      const requestedRange = rangeForRequest(request.headers.range, fixture.bytes.length);
      if (requestedRange === false) {
        response.writeHead(416, {
          "Accept-Ranges": "bytes",
          "Cache-Control": "no-store",
          "Content-Range": `bytes */${fixture.bytes.length}`,
          "X-Content-Type-Options": "nosniff",
        });
        response.end();
        return;
      }
      const range = requestedRange || { start: 0, end: fixture.bytes.length - 1 };
      const body = fixture.bytes.subarray(range.start, range.end + 1);
      const partial = !!requestedRange;
      response.writeHead(partial ? 206 : 200, {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Content-Length": body.length,
        ...(partial ? { "Content-Range": `bytes ${range.start}-${range.end}/${fixture.bytes.length}` } : {}),
        "Content-Security-Policy": FIXTURE_CONTENT_SECURITY_POLICY,
        "Content-Type": fixture.type,
        "Cross-Origin-Resource-Policy": "same-origin",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
      });
      response.end(method === "HEAD" ? undefined : body);
    } catch {
      if (!response.headersSent) response.writeHead(500, { "Cache-Control": "no-store" });
      response.end();
    }
  });
  server.on("clientError", (_error, socket) => socket.destroy());
  const onAbort = () => {
    server.close();
    server.closeAllConnections?.();
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    await new Promise((resolveListen, rejectListen) => {
      const fail = (error) => {
        server.removeListener("listening", ready);
        rejectListen(error);
      };
      const ready = () => {
        server.removeListener("error", fail);
        resolveListen();
      };
      server.once("error", fail);
      server.once("listening", ready);
      server.listen(0, "127.0.0.1");
    });
  } catch (error) {
    signal?.removeEventListener("abort", onAbort);
    throw error;
  }
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("fixture server did not expose a TCP port");
  return {
    baseUrl: new URL(`http://127.0.0.1:${address.port}/`),
    async close() {
      signal?.removeEventListener("abort", onAbort);
      server.closeAllConnections?.();
      await new Promise((resolveClose) => server.close(() => resolveClose()));
    },
  };
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

async function waitFor(label, probe, accept, {
  timeoutMs = INTEGRATION_TIMEOUT_MS,
  intervalMs = 100,
  signal = null,
} = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  let lastError = null;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw abortReason(signal);
    try {
      lastValue = await probe();
      lastError = null;
      if (accept(lastValue)) return lastValue;
    } catch (error) {
      lastError = error;
    }
    await delay(intervalMs, signal);
  }
  let detail = "";
  if (lastError) detail = `; last error: ${lastError.message}`;
  else if (lastValue !== undefined) {
    try { detail = `; last value: ${JSON.stringify(lastValue).slice(0, 2_000)}`; } catch {}
  }
  throw new Error(`${label} timed out after ${timeoutMs} ms${detail}`);
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function requestPageTarget(httpBase, url, signal) {
  if (new URL(url).protocol === "chrome-extension:") {
    throw new Error("private extension pages must be opened by the extension service worker");
  }
  const targetUrl = new URL(`/json/new?${encodeURIComponent(url)}`, httpBase);
  const target = await requestJson(targetUrl, { method: "PUT", signal });
  if (!target.id || !target.webSocketDebuggerUrl) {
    throw new Error("new page target is missing its ID or DevTools WebSocket URL");
  }
  return target;
}

async function requestExtensionPageTarget(httpBase, controlClient, url, signal) {
  if (new URL(url).protocol !== "chrome-extension:") {
    throw new Error("trusted extension target creation requires a chrome-extension:// URL");
  }
  const priorTargetIds = new Set((await listTargets(httpBase, signal)).map((target) => target.id));
  const openedTab = await evaluate(controlClient, `chrome.tabs.create({
    url: ${JSON.stringify(url)},
    active: true,
  }).then((tab) => ({ id: tab.id, url: tab.pendingUrl || tab.url || null }))`);
  if (!Number.isInteger(openedTab?.id)) throw new Error("extension page opener returned no tab ID");
  try {
    return await waitFor("trusted extension page target", async () =>
      (await listTargets(httpBase, signal)).find((target) =>
        !priorTargetIds.has(target.id) &&
        target.type === "page" &&
        target.url === url &&
        target.webSocketDebuggerUrl
      ) || null, Boolean, {
        timeoutMs: CDP_TIMEOUT_MS,
        intervalMs: 50,
        signal,
      });
  } catch (error) {
    try {
      await evaluate(controlClient, `chrome.tabs.remove(${openedTab.id})`);
    } catch {}
    throw error;
  }
}

async function createPageTarget(httpBase, url, signal, onEvent = null) {
  const target = await requestPageTarget(httpBase, url, signal);
  const client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
  try {
    if (onEvent) client.onEvent(onEvent);
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    return { target, client };
  } catch (error) {
    client.close();
    throw error;
  }
}

async function createExtensionPageTarget(
  httpBase,
  controlClient,
  url,
  signal,
  onEvent = null,
) {
  const target = await requestExtensionPageTarget(httpBase, controlClient, url, signal);
  const client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
  try {
    if (onEvent) client.onEvent(onEvent);
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    return { target, client };
  } catch (error) {
    client.close();
    throw error;
  }
}

async function closePageTarget(httpBase, page, {
  required = false,
  signal = null,
} = {}) {
  page?.client?.close();
  if (!page?.target?.id) return true;
  try {
    await requestJson(new URL(`/json/close/${encodeURIComponent(page.target.id)}`, httpBase));
  } catch {}
  try {
    await waitFor(`DevTools target ${page.target.id} retirement`, () => listTargets(httpBase, signal),
      (targets) => !targets.some((target) => target.id === page.target.id), {
        timeoutMs: CDP_TIMEOUT_MS,
        intervalMs: 50,
        signal,
      });
    return true;
  } catch (error) {
    if (required) throw error;
    return false;
  }
}

async function waitForDocument(client, expectedUrl, signal) {
  return waitFor("page load", () => evaluate(client, `({
    href: location.href,
    readyState: document.readyState,
    title: document.title,
  })`), (state) => state?.href === expectedUrl && state.readyState === "complete", {
    timeoutMs: CDP_TIMEOUT_MS,
    signal,
  });
}

function unexpectedRuntimeEvents(events) {
  return events.filter((event) => event.kind === "exception" ||
    (event.kind === "console" && ["error", "warning", "assert"].includes(event.type)));
}

function assertRuntimeClean(events, context) {
  const failures = unexpectedRuntimeEvents(events);
  if (!failures.length) return;
  const error = new Error(`${context} emitted ${failures.length} warning/error/exception event(s)`);
  error.browserEvents = events;
  throw error;
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
    "--autoplay-policy=no-user-gesture-required",
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

async function waitForDevToolsHttp(httpBase, signal) {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  let lastError = null;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw abortReason(signal);
    const remaining = Math.max(1, deadline - Date.now());
    try {
      const version = await requestJson(new URL("json/version", httpBase), {
        signal,
        timeoutMs: Math.min(HTTP_TIMEOUT_MS, remaining),
      });
      if (typeof version?.webSocketDebuggerUrl !== "string") {
        throw new Error("DevTools HTTP version response omitted webSocketDebuggerUrl");
      }
      return;
    } catch (error) {
      lastError = error;
    }
    const retryDelay = Math.min(200, deadline - Date.now());
    if (retryDelay > 0) await delay(retryDelay, signal);
  }
  const detail = lastError instanceof Error ? `: ${lastError.message}` : "";
  throw new Error(`DevTools HTTP endpoint did not become ready after ${STARTUP_TIMEOUT_MS} ms${detail}`);
}

async function listTargets(httpBase, signal) {
  return requestJson(new URL("json/list", httpBase), { signal });
}

async function discoverExtension(httpBase, expectedName, serviceWorkerPath, signal) {
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
    const candidates = targets.filter((target) => {
      if (target.type !== "service_worker" || !target.webSocketDebuggerUrl) return false;
      try {
        const url = new URL(target.url);
        return url.protocol === "chrome-extension:" &&
          /^[a-p]{32}$/.test(url.hostname) &&
          url.pathname === `/${serviceWorkerPath}`;
      } catch {
        return false;
      }
    });
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
        const controlClient = client;
        client = null;
        return { controlClient, extensionId, source: "service worker" };
      } catch (error) {
        lastError = error;
      } finally {
        client?.close();
      }
    }
    await delay(200, signal);
  }
  const suffix = lastError ? `: ${lastError.message}` : "";
  throw new Error(`could not discover the verified ${serviceWorkerPath} service worker${suffix}`);
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

async function runValidation(httpBase, controlClient, extensionId, expectedName, signal) {
  const extensionUrl = `chrome-extension://${extensionId}/validate.html?autorun=1`;
  const target = await requestExtensionPageTarget(httpBase, controlClient, extensionUrl, signal);
  const events = [];
  let client = null;
  try {
    client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
    client.onEvent((message) => collectRuntimeEvent(events, message));
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    await client.send("Page.bringToFront");
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
    await closePageTarget(httpBase, { target });
  }
}

async function runPopupSmoke(httpBase, controlClient, extensionId, expectedName, signal) {
  const popupUrl = `chrome-extension://${extensionId}/popup.html`;
  const target = await requestExtensionPageTarget(httpBase, controlClient, popupUrl, signal);
  const events = [];
  let client = null;
  try {
    client = await CdpClient.connect(target.webSocketDebuggerUrl, { signal });
    client.onEvent((message) => collectRuntimeEvent(events, message));
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    await client.send("Page.bringToFront");

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
            scriptType: document.querySelector('script[src="src/popup.js"]')?.type || null,
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
    if (!new Set(["unavailable", "disconnected"]).has(state?.webgpuText)) {
      problems.push("popup non-content-page state was not rendered");
    }
    if (!/(?:open an http, https, or permitted local file page|reload this page)/i.test(
      state?.operationText || "",
    )) {
      problems.push("popup non-content-page guidance is missing");
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
    await closePageTarget(httpBase, { target });
  }
}

async function fixtureSnapshot(client) {
  return evaluate(client, `(() => {
    const api = window.__FSRCNNX_VIDEO_FIXTURE__;
    const video = document.querySelector("video");
    let decodedColor = null;
    let decodedColorError = null;
    let frame = null;
    try {
      frame = video && typeof VideoFrame === "function" ? new VideoFrame(video) : null;
      if (frame) {
        decodedColor = {
          primaries: frame.colorSpace?.primaries ?? null,
          transfer: frame.colorSpace?.transfer ?? null,
          matrix: frame.colorSpace?.matrix ?? null,
          fullRange: frame.colorSpace?.fullRange ?? null,
        };
      }
    } catch (error) {
      decodedColorError = error?.message || String(error);
    } finally {
      try { frame?.close(); } catch {}
    }
    const computed = video ? getComputedStyle(video) : null;
    return {
      fixture: api?.snapshot?.() ?? null,
      decodedColor,
      decodedColorError,
      video: video ? {
        currentSrc: video.currentSrc,
        width: video.videoWidth,
        height: video.videoHeight,
        readyState: video.readyState,
        paused: video.paused,
        ended: video.ended,
        muted: video.muted,
        display: computed?.display || null,
        visibility: computed?.visibility || null,
        opacity: computed?.opacity || null,
        rect: (() => {
          const rect = video.getBoundingClientRect();
          return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
        })(),
        parentInlinePosition: video.parentElement?.style?.position || "",
      } : null,
      overlays: [...document.querySelectorAll("canvas[data-fsrcnnx-overlay]")].map((overlay) => {
        const style = getComputedStyle(overlay);
        const rect = overlay.getBoundingClientRect();
        return {
          role: overlay.dataset.fsrcnnxOverlay || null,
          connected: overlay.isConnected,
          display: style.display,
          visibility: style.visibility,
          opacity: style.opacity,
          width: overlay.width,
          height: overlay.height,
          rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        };
      }),
    };
  })()`);
}

async function loadFixtureSource(client, name) {
  const sourceName = JSON.stringify(name);
  return evaluate(client, `(async () => {
    const api = window.__FSRCNNX_VIDEO_FIXTURE__;
    if (!api?.loadSource) throw new Error("video fixture API is unavailable");
    return api.loadSource(${sourceName});
  })()`, INTEGRATION_TIMEOUT_MS);
}

async function setFixtureDisplayScale(client, scale) {
  requireCondition(Number.isFinite(scale) && scale > 0, "fixture display scale must be positive");
  return evaluate(client, `(() => {
    const video = document.querySelector("video");
    if (!video?.videoWidth || !video.videoHeight) throw new Error("decoded fixture video is unavailable");
    const width = Math.max(96, Math.round(video.videoWidth * ${Number(scale)}));
    const height = Math.max(72, Math.round(width * video.videoHeight / video.videoWidth));
    video.style.width = width + "px";
    video.style.height = height + "px";
    window.dispatchEvent(new Event("resize"));
    return { width, height };
  })()`);
}

async function activateExtensionTab(client, url) {
  const encodedUrl = JSON.stringify(url);
  return evaluate(client, `(async () => {
    const tabs = await chrome.tabs.query({});
    let tab = tabs.find((candidate) => candidate.url === ${encodedUrl}) || null;
    if (!tab) {
      for (const candidate of tabs) {
        if (!Number.isInteger(candidate.id)) continue;
        try {
          const response = await chrome.tabs.sendMessage(candidate.id, { type: "FSRCNNX_STATUS" });
          if (response && typeof response === "object") { tab = candidate; break; }
        } catch {}
      }
    }
    if (!tab || !Number.isInteger(tab.id)) throw new Error("fixture Chrome tab was not found");
    await chrome.tabs.update(tab.id, { active: true });
    return { id: tab.id, windowId: tab.windowId, url: tab.url || null };
  })()`);
}

async function extensionSnapshot(client, message = null, tabId = null) {
  const encodedMessage = message == null ? "null" : JSON.stringify(message);
  const encodedTabId = Number.isInteger(tabId) ? String(tabId) : "null";
  return evaluate(client, `(async () => {
    const requestedTabId = ${encodedTabId};
    const tab = requestedTabId == null
      ? (await chrome.tabs.query({ active: true, currentWindow: true }))[0] || null
      : await chrome.tabs.get(requestedTabId).catch(() => null);
    if (!tab || !Number.isInteger(tab.id)) return { tab: null, error: "no active tab" };
    let response = null;
    let messageError = null;
    const message = ${encodedMessage};
    if (message) {
      try { response = await chrome.tabs.sendMessage(tab.id, message); }
      catch (error) { messageError = error?.message || String(error); }
    }
    let badge = null;
    let title = null;
    try { badge = await chrome.action.getBadgeText({ tabId: tab.id }); } catch {}
    try { title = await chrome.action.getTitle({ tabId: tab.id }); } catch {}
    return {
      tab: { id: tab.id, url: tab.url || null, active: !!tab.active },
      response,
      messageError,
      badge,
      title,
    };
  })()`, INTEGRATION_TIMEOUT_MS);
}

async function contentStatus(client, tabId = null) {
  const snapshot = await extensionSnapshot(client, { type: "FSRCNNX_STATUS" }, tabId);
  if (snapshot.messageError || !snapshot.response) {
    throw new Error(
      `${snapshot.messageError || "content script returned no status"} ` +
      `(tab ${snapshot.tab?.id ?? "none"}, ${snapshot.tab?.url || "unknown URL"})`,
    );
  }
  return { ...snapshot, status: snapshot.response };
}

async function sendContentCommand(client, type, payload = {}, tabId = null) {
  const snapshot = await extensionSnapshot(client, { type, ...payload }, tabId);
  if (snapshot.messageError) throw new Error(`${type}: ${snapshot.messageError}`);
  if (!snapshot.response || snapshot.response.ok !== true) {
    throw new Error(`${type} failed: ${snapshot.response?.reason || snapshot.response?.error || "invalid response"}`);
  }
  return snapshot.response;
}

async function clickPopupMode(client, mode) {
  const encodedMode = JSON.stringify(mode);
  await waitFor(`popup mode control ${mode}`, () => evaluate(client, `(() => {
    const button = document.querySelector('.modes button[data-mode=' + JSON.stringify(${encodedMode}) + ']');
    return !!button && !button.disabled;
  })()`), Boolean, { timeoutMs: CDP_TIMEOUT_MS });
  const clicked = await evaluate(client, `(() => {
    const button = document.querySelector('.modes button[data-mode=' + JSON.stringify(${encodedMode}) + ']');
    if (!button || button.disabled) return false;
    button.click();
    return true;
  })()`);
  requireCondition(clicked === true, `popup mode button ${mode} is unavailable`);
}

async function changePopupControl(client, id, value) {
  const encodedId = JSON.stringify(id);
  const encodedValue = JSON.stringify(value);
  await waitFor(`popup control ${id}`, () => evaluate(client, `(() => {
    const control = document.getElementById(${encodedId});
    return !!control && !control.disabled;
  })()`), Boolean, { timeoutMs: CDP_TIMEOUT_MS });
  const changed = await evaluate(client, `(() => {
    const control = document.getElementById(${encodedId});
    if (!control || control.disabled) return false;
    const value = ${encodedValue};
    if (control instanceof HTMLInputElement && control.type === "checkbox") control.checked = !!value;
    else control.value = String(value);
    control.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  requireCondition(changed === true, `popup control ${id} is unavailable`);
}

async function popupSnapshot(client) {
  return evaluate(client, `(() => {
    const controls = [...document.querySelectorAll("button, input, select")];
    const accessibleName = (control) => {
      const direct = control.getAttribute("aria-label") || control.getAttribute("title");
      if (direct?.trim()) return direct.trim();
      const labelledBy = control.getAttribute("aria-labelledby");
      if (labelledBy) {
        const text = labelledBy.split(/\\s+/).map((id) => document.getElementById(id)?.textContent || "").join(" ").trim();
        if (text) return text;
      }
      const labelText = [...(control.labels || [])].map((label) => label.textContent || "").join(" ").trim();
      if (labelText) return labelText;
      return control.tagName === "BUTTON" ? (control.textContent || "").trim() : "";
    };
    const ids = [...document.querySelectorAll("[id]")].map((element) => element.id);
    const seen = new Set();
    const duplicateIds = ids.filter((id) => seen.has(id) || !seen.add(id));
    const banner = document.getElementById("drm-banner");
    return {
      href: location.href,
      readyState: document.readyState,
      modeStates: [...document.querySelectorAll(".modes button")].map((button) => ({
        mode: button.dataset.mode,
        pressed: button.getAttribute("aria-pressed"),
        disabled: button.disabled,
      })),
      unnamedControls: controls.filter((control) => !accessibleName(control)).map((control) => control.id || control.tagName),
      duplicateIds,
      banner: {
        hidden: banner?.hidden ?? true,
        display: banner ? getComputedStyle(banner).display : "none",
        text: banner?.textContent?.trim() || "",
        role: banner?.getAttribute("role") || null,
      },
      operation: document.getElementById("operation-status")?.textContent?.trim() || "",
      operationRole: document.getElementById("operation-status")?.getAttribute("role") || null,
      runtime: document.getElementById("runtime-status")?.textContent?.trim() || "",
      runtimeLive: document.getElementById("runtime-status")?.getAttribute("aria-live") || null,
      webgpu: document.getElementById("s-webgpu")?.textContent?.trim() || "",
      video: document.getElementById("s-video")?.textContent?.trim() || "",
      model: document.getElementById("s-model")?.textContent?.trim() || "",
      frames: document.getElementById("s-frames")?.textContent?.trim() || "",
      offDisabled: document.querySelector('.modes button[data-mode="off"]')?.disabled ?? true,
    };
  })()`);
}

function overlayByRole(snapshot, role) {
  return snapshot?.overlays?.filter((overlay) => overlay.role === role) || [];
}

function visibleOverlay(snapshot, role) {
  return overlayByRole(snapshot, role).find((overlay) => overlay.connected &&
    overlay.display !== "none" && overlay.visibility !== "hidden" && Number(overlay.opacity) !== 0 &&
    overlay.rect.width > 0 && overlay.rect.height > 0) || null;
}

async function captureOverlayPixels(client, snapshot, role) {
  const overlay = visibleOverlay(snapshot, role);
  requireCondition(!!overlay, `${role} pixel capture has no visible overlay`);
  const insetX = Math.max(2, Math.floor(overlay.rect.width * 0.06));
  const insetY = Math.max(2, Math.floor(overlay.rect.height * 0.06));
  const clip = {
    x: Math.max(0, Math.floor(overlay.rect.x + insetX)),
    y: Math.max(0, Math.floor(overlay.rect.y + insetY)),
    width: Math.floor(overlay.rect.width - insetX * 2),
    height: Math.floor(overlay.rect.height - insetY * 2),
    scale: 1,
  };
  requireCondition(clip.width >= 64 && clip.height >= 36,
    `${role} pixel capture region is too small: ${JSON.stringify(clip)}`);
  const encodedRole = JSON.stringify(role);
  const encodedRect = JSON.stringify(overlay.rect);
  const isolated = await evaluate(client, `(async () => {
    const restoreKey = "__FSRCNNX_PIXEL_ISOLATION_RESTORE__";
    globalThis[restoreKey]?.();
    const expected = ${encodedRect};
    const candidates = [...document.querySelectorAll(
      "canvas[data-fsrcnnx-overlay=" + JSON.stringify(${encodedRole}) + "]"
    )].filter((canvas) => {
      const style = getComputedStyle(canvas);
      const rect = canvas.getBoundingClientRect();
      return canvas.isConnected && style.display !== "none" && style.visibility !== "hidden" &&
        Number(style.opacity) !== 0 && rect.width > 0 && rect.height > 0;
    });
    candidates.sort((left, right) => {
      const a = left.getBoundingClientRect();
      const b = right.getBoundingClientRect();
      const distance = (rect) => Math.abs(rect.x - expected.x) + Math.abs(rect.y - expected.y) +
        Math.abs(rect.width - expected.width) + Math.abs(rect.height - expected.height);
      return distance(a) - distance(b);
    });
    const target = candidates[0] || null;
    if (!target) return false;
    const underlays = [
      document.querySelector("video"),
      ...document.querySelectorAll("canvas[data-fsrcnnx-overlay]"),
    ].filter((element) => element && element !== target);
    const records = underlays.map((element) => ({
      element,
      value: element.style.getPropertyValue("filter"),
      priority: element.style.getPropertyPriority("filter"),
    }));
    globalThis[restoreKey] = () => {
      for (const { element, value, priority } of records) {
        if (value) element.style.setProperty("filter", value, priority);
        else element.style.removeProperty("filter");
      }
      delete globalThis[restoreKey];
    };
    for (const { element } of records) {
      element.style.setProperty("filter", "brightness(0)", "important");
    }
    await new Promise((resolve) => requestAnimationFrame(() =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))));
    if (!records.length || records.some(({ element }) => getComputedStyle(element).filter === "none")) {
      globalThis[restoreKey]();
      return false;
    }
    return true;
  })()`);
  requireCondition(isolated === true, `${role} pixel capture could not isolate its underlay`);
  let screenshot;
  try {
    screenshot = await client.send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: false,
      clip,
    }, CDP_TIMEOUT_MS);
  } finally {
    const restored = await evaluate(client, `(async () => {
      const restore = globalThis.__FSRCNNX_PIXEL_ISOLATION_RESTORE__;
      if (typeof restore !== "function") return false;
      restore();
      await new Promise((resolve) => requestAnimationFrame(resolve));
      return true;
    })()`);
    requireCondition(restored === true, `${role} pixel capture did not restore its underlay`);
  }
  requireCondition(typeof screenshot?.data === "string" && screenshot.data.length > 0,
    `${role} pixel capture returned no PNG data`);
  const encodedPng = JSON.stringify(screenshot.data);
  const metrics = await evaluate(client, `(async () => {
    const binary = atob(${encodedPng});
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
    try {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("2D screenshot readback is unavailable");
      context.drawImage(bitmap, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const gridWidth = 16;
      const gridHeight = 9;
      const cellCount = gridWidth * gridHeight;
      const gridSums = new Float64Array(cellCount * 3);
      const gridCounts = new Uint32Array(cellCount);
      let lumaSum = 0;
      let lumaSquareSum = 0;
      let minLuma = 255;
      let maxLuma = 0;
      let nonBlack = 0;
      let nonWhite = 0;
      const lumaHistogram = new Uint32Array(256);
      let hash = 2166136261;
      for (let y = 0; y < canvas.height; y++) {
        const gridY = Math.min(gridHeight - 1, Math.floor(y * gridHeight / canvas.height));
        for (let x = 0; x < canvas.width; x++) {
          const offset = (y * canvas.width + x) * 4;
          const red = pixels[offset];
          const green = pixels[offset + 1];
          const blue = pixels[offset + 2];
          const luma = (77 * red + 150 * green + 29 * blue) / 256;
          lumaSum += luma;
          lumaSquareSum += luma * luma;
          minLuma = Math.min(minLuma, luma);
          maxLuma = Math.max(maxLuma, luma);
          if (luma > 8) nonBlack++;
          if (luma < 247) nonWhite++;
          lumaHistogram[Math.max(0, Math.min(255, Math.round(luma)))]++;
          const gridX = Math.min(gridWidth - 1, Math.floor(x * gridWidth / canvas.width));
          const cell = gridY * gridWidth + gridX;
          const base = cell * 3;
          gridSums[base] += red;
          gridSums[base + 1] += green;
          gridSums[base + 2] += blue;
          gridCounts[cell]++;
          hash = Math.imul(hash ^ red, 16777619);
          hash = Math.imul(hash ^ green, 16777619);
          hash = Math.imul(hash ^ blue, 16777619);
        }
      }
      const count = canvas.width * canvas.height;
      const meanLuma = lumaSum / count;
      const variance = Math.max(0, lumaSquareSum / count - meanLuma * meanLuma);
      const percentile = (fraction) => {
        const target = Math.max(1, Math.ceil(count * fraction));
        let cumulative = 0;
        for (let value = 0; value < lumaHistogram.length; value++) {
          cumulative += lumaHistogram[value];
          if (cumulative >= target) return value;
        }
        return 255;
      };
      const lumaSpread = percentile(0.9) - percentile(0.1);
      const grid = [];
      for (let cell = 0; cell < cellCount; cell++) {
        const base = cell * 3;
        const divisor = gridCounts[cell] || 1;
        grid.push(
          +(gridSums[base] / divisor).toFixed(3),
          +(gridSums[base + 1] / divisor).toFixed(3),
          +(gridSums[base + 2] / divisor).toFixed(3),
        );
      }
      return {
        width: canvas.width,
        height: canvas.height,
        meanLuma: +meanLuma.toFixed(3),
        lumaDeviation: +Math.sqrt(variance).toFixed(3),
        lumaRange: +(maxLuma - minLuma).toFixed(3),
        lumaSpread,
        nonBlackRatio: +(nonBlack / count).toFixed(6),
        nonWhiteRatio: +(nonWhite / count).toFixed(6),
        hash: (hash >>> 0).toString(16).padStart(8, "0"),
        grid,
      };
    } finally {
      bitmap.close();
    }
  })()`);
  requireCondition(metrics && Array.isArray(metrics.grid), `${role} PNG could not be sampled`);
  return metrics;
}

function pixelGridDistance(left, right) {
  if (!Array.isArray(left?.grid) || left.grid.length === 0 ||
      left.grid.length !== right?.grid?.length) return Number.POSITIVE_INFINITY;
  let difference = 0;
  for (let index = 0; index < left.grid.length; index++) {
    difference += Math.abs(left.grid[index] - right.grid[index]);
  }
  return difference / left.grid.length;
}

function pixelGridChangedCellRatio(left, right, threshold = 4) {
  if (!Array.isArray(left?.grid) || left.grid.length === 0 ||
      left.grid.length !== right?.grid?.length || left.grid.length % 3 !== 0) return 0;
  let changed = 0;
  const cells = left.grid.length / 3;
  for (let cell = 0; cell < cells; cell++) {
    const base = cell * 3;
    const maxDifference = Math.max(
      Math.abs(left.grid[base] - right.grid[base]),
      Math.abs(left.grid[base + 1] - right.grid[base + 1]),
      Math.abs(left.grid[base + 2] - right.grid[base + 2]),
    );
    if (maxDifference >= threshold) changed++;
  }
  return changed / cells;
}

function renderedPixelsValid(sample) {
  return !!sample && sample.meanLuma > 20 && sample.meanLuma < 235 &&
    sample.lumaDeviation >= 18 && sample.lumaRange >= 50 && sample.lumaSpread >= 50 &&
    sample.nonBlackRatio >= 0.35 && sample.nonWhiteRatio >= 0.35;
}

function requireRenderedPixels(sample, label) {
  requireCondition(renderedPixelsValid(sample),
  `${label} screenshot is black, blank, or uniform: ${JSON.stringify({
    meanLuma: sample?.meanLuma,
    lumaDeviation: sample?.lumaDeviation,
    lumaRange: sample?.lumaRange,
    lumaSpread: sample?.lumaSpread,
    nonBlackRatio: sample?.nonBlackRatio,
    nonWhiteRatio: sample?.nonWhiteRatio,
    hash: sample?.hash,
  })}`);
}

async function waitForRenderedOverlayPixels(client, role, label, signal) {
  const sample = await waitFor(label, async () => {
    const snapshot = await fixtureSnapshot(client);
    if (!visibleOverlay(snapshot, role)) throw new Error(`${role} overlay is not visible`);
    return captureOverlayPixels(client, snapshot, role);
  }, renderedPixelsValid, {
    timeoutMs: CDP_TIMEOUT_MS,
    intervalMs: 100,
    signal,
  });
  requireRenderedPixels(sample, label);
  return sample;
}

function requirePixelProgression(first, second, label) {
  requireRenderedPixels(first, `${label} first frame`);
  requireRenderedPixels(second, `${label} later frame`);
  const distance = pixelGridDistance(first, second);
  const changedCellRatio = pixelGridChangedCellRatio(first, second);
  requireCondition(first.hash !== second.hash && distance >= 3 && changedCellRatio >= 0.15,
    `${label} screenshot remained stale (distance ${distance.toFixed(3)}, ` +
    `changed cells ${(changedCellRatio * 100).toFixed(1)}%, hash ${first.hash})`);
  console.log(`Rendered pixels (${label}): ${first.hash} -> ${second.hash}, ` +
    `distance ${distance.toFixed(3)}, changed cells ${(changedCellRatio * 100).toFixed(1)}%`);
}

function fixtureColorMatchesExpected(snapshot) {
  const expected = snapshot?.fixture?.expectedColorSpace || snapshot?.fixture?.expected?.colorSpace || null;
  const decoded = snapshot?.fixture?.colorSpace?.available === true
    ? snapshot.fixture.colorSpace
    : snapshot?.decodedColor;
  if (!expected || !decoded) return false;
  return ["primaries", "transfer", "matrix", "fullRange"].every((field) => expected[field] === decoded[field]);
}

async function waitForStatus(client, tabId, label, accept, signal, timeoutMs = INTEGRATION_TIMEOUT_MS) {
  return waitFor(label, () => contentStatus(client, tabId), (snapshot) => accept(snapshot.status, snapshot), {
    timeoutMs,
    signal,
  });
}

async function waitForBadge(client, tabId, expected, signal) {
  return waitFor(`badge ${JSON.stringify(expected)}`, () => extensionSnapshot(client, null, tabId),
    (snapshot) => snapshot.badge === expected, { timeoutMs: CDP_TIMEOUT_MS, signal });
}

async function runRealVideoIntegration(
  httpBase,
  controlClient,
  fixtureBase,
  extensionId,
  expectedName,
  signal,
) {
  const fixtureUrl = new URL("video.html", fixtureBase).href;
  const popupUrl = `chrome-extension://${extensionId}/popup.html`;
  let fixturePage = null;
  let popupPage = null;
  const fixtureEvents = [];
  const popupEvents = [];
  const checkpoints = [];
  const checkpoint = (label) => {
    checkpoints.push(label);
    console.log(`Real-video checkpoint ${checkpoints.length}: ${label}`);
  };
  try {
    fixturePage = await createPageTarget(
      httpBase,
      fixtureUrl,
      signal,
      (message) => collectRuntimeEvent(fixtureEvents, message),
    );
    await waitForDocument(fixturePage.client, fixtureUrl, signal);
    await waitFor("video fixture API", () => evaluate(
      fixturePage.client,
      "typeof window.__FSRCNNX_VIDEO_FIXTURE__?.loadSource === 'function'",
    ), Boolean, { timeoutMs: CDP_TIMEOUT_MS, signal });
    await evaluate(fixturePage.client, "window.__FSRCNNX_VIDEO_FIXTURE__.ready()", CDP_TIMEOUT_MS);
    await loadFixtureSource(fixturePage.client, "bt709-a");

    popupPage = await createExtensionPageTarget(
      httpBase,
      controlClient,
      popupUrl,
      signal,
      (message) => collectRuntimeEvent(popupEvents, message),
    );
    await waitForDocument(popupPage.client, popupUrl, signal);
    const popupIdentity = await evaluate(popupPage.client, `({
      runtimeId: chrome.runtime.id,
      manifestName: chrome.runtime.getManifest().name,
    })`);
    requireCondition(popupIdentity?.runtimeId === extensionId && popupIdentity?.manifestName === expectedName,
      "integration popup extension identity is incorrect");
    let fixtureTab = await activateExtensionTab(popupPage.client, fixtureUrl);
    await fixturePage.client.send("Page.bringToFront");

    const initial = await waitForStatus(popupPage.client, fixtureTab.id, "content-script startup", (status) =>
      status.loading !== true && status.failed !== true && status.hasVideo === true, signal);
    const initialPage = await fixtureSnapshot(fixturePage.client);
    requireCondition(initial.tab?.id === fixtureTab.id,
      `popup transport selected tab ${initial.tab?.id ?? "none"}; expected ${fixtureTab.id}`);
    requireCondition(initial.status.mode === "off" && initial.status.activeMode === "off",
      "fresh fixture did not start with renderer off");
    requireCondition(initial.status.renderer?.phase === "off", "fresh fixture renderer phase is not off");
    requireCondition(initial.status.webgpu === true, "real fixture page does not expose WebGPU");
    requireCondition(initial.status.colorSupport?.code === "color-not-checked",
      "mode-off startup unexpectedly probed decoded video color");
    requireCondition(fixtureColorMatchesExpected(initialPage),
      `fixture manifest and decoded color metadata disagree: ${JSON.stringify(initialPage)}`);
    requireCondition(initialPage.overlays.length === 0, "renderer created an overlay while mode was off");
    await waitForBadge(popupPage.client, fixtureTab.id, "", signal);
    await waitFor("popup fixture status", () => popupSnapshot(popupPage.client), (snapshot) =>
      snapshot.offDisabled === false && snapshot.webgpu !== "unavailable" && snapshot.video !== "none",
    { timeoutMs: CDP_TIMEOUT_MS, signal });
    checkpoint("content injection/off baseline");

    await clickPopupMode(popupPage.client, "passthrough");
    const passthrough = await waitForStatus(popupPage.client, fixtureTab.id, "passthrough activation", (status) =>
      status.mode === "passthrough" && status.activeMode === "passthrough" &&
      status.renderer?.phase === "active" && status.frameCount > 0 &&
      status.colorSupport?.code === "color-supported", signal);
    requireCondition(passthrough.status.colorSupport?.colorSpace?.primaries === "bt709" &&
      ["bt709", "iec61966-2-1"].includes(passthrough.status.colorSupport?.colorSpace?.transfer) &&
      ["bt709", "rgb"].includes(passthrough.status.colorSupport?.colorSpace?.matrix),
    "real BT.709 fixture reported an unexpected decoded color tuple");
    const passthroughFirstPage = await fixtureSnapshot(fixturePage.client);
    const passthroughFirstOverlay = visibleOverlay(passthroughFirstPage, "primary");
    requireCondition(!!passthroughFirstOverlay, "passthrough did not present a visible primary overlay");
    requireCondition(passthroughFirstOverlay.width === passthroughFirstPage.video.width &&
      passthroughFirstOverlay.height === passthroughFirstPage.video.height,
    "passthrough canvas does not match decoded source dimensions");
    const passthroughFirstPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "primary", "passthrough first composited frame", signal,
    );
    const passthroughFrames = (await contentStatus(popupPage.client, fixtureTab.id)).status.frameCount;
    await waitForStatus(popupPage.client, fixtureTab.id, "passthrough frame progression",
      (status) => status.frameCount >= passthroughFrames + 3, signal);
    const passthroughPage = await fixtureSnapshot(fixturePage.client);
    const passthroughOverlay = visibleOverlay(passthroughPage, "primary");
    requireCondition(!!passthroughOverlay, "passthrough did not present a visible primary overlay");
    requireCondition(passthroughOverlay.width === passthroughPage.video.width &&
      passthroughOverlay.height === passthroughPage.video.height,
    "passthrough canvas does not match decoded source dimensions");
    const passthroughPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "primary", "passthrough later composited frame", signal,
    );
    requirePixelProgression(passthroughFirstPixels, passthroughPixels, "passthrough compositor");
    await waitForBadge(popupPage.client, fixtureTab.id, "··", signal);
    checkpoint("passthrough presentation");

    await changePopupControl(popupPage.client, "engine", "fsrcnnx");
    await waitForStatus(popupPage.client, fixtureTab.id, "FSRCNNX engine selection", (status) => status.engine === "fsrcnnx", signal);
    await changePopupControl(popupPage.client, "policy", "force2");
    await waitForStatus(popupPage.client, fixtureTab.id, "force2 policy selection", (status) => status.policy === "force2", signal);
    await changePopupControl(popupPage.client, "ssimds", false);
    await waitForStatus(popupPage.client, fixtureTab.id, "SSimDownscaler disable", (status) => status.ssimds === false, signal);
    await clickPopupMode(popupPage.client, "upscale");
    const upscaled = await waitForStatus(popupPage.client, fixtureTab.id, "FSRCNNX force2 presentation", (status) =>
      status.mode === "upscale" && status.activeMode === "upscale" &&
      status.renderer?.phase === "active" && status.renderer?.activeEngine === "fsrcnnx" &&
      status.scale === 2 && status.presentation?.committed === true &&
      status.presentation?.output?.width === status.presentation?.source?.width * 2 &&
      status.presentation?.output?.height === status.presentation?.source?.height * 2, signal);
    const upscaledFirstPage = await fixtureSnapshot(fixturePage.client);
    const upscaledFirstOverlay = visibleOverlay(upscaledFirstPage, "primary");
    requireCondition(upscaledFirstOverlay?.width === upscaled.status.presentation.source.width * 2 &&
      upscaledFirstOverlay?.height === upscaled.status.presentation.source.height * 2,
    "FSRCNNX force2 canvas backing dimensions do not differ from passthrough");
    const upscaledFirstPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "primary", "FSRCNNX first composited frame", signal,
    );
    const upscaleGeneration = (await contentStatus(
      popupPage.client, fixtureTab.id,
    )).status.presentation.generation;
    await waitForStatus(popupPage.client, fixtureTab.id, "FSRCNNX frame progression", (status) =>
      status.presentation?.generation >= upscaleGeneration + 3, signal);
    const upscaledPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "primary", "FSRCNNX later composited frame", signal,
    );
    requirePixelProgression(upscaledFirstPixels, upscaledPixels, "FSRCNNX compositor");
    await waitForBadge(popupPage.client, fixtureTab.id, "ON", signal);
    checkpoint("FSRCNNX force2 presentation");

    await setFixtureDisplayScale(fixturePage.client, 1.35);
    await changePopupControl(popupPage.client, "ssimds", true);
    const downscaled = await waitForStatus(popupPage.client, fixtureTab.id, "SSimDownscaler presentation", (status) =>
      status.ssimds === true && status.presentation?.ssimds != null &&
      status.presentation.output.width < status.presentation.source.width * 2, signal);
    requireCondition(downscaled.status.presentation.ssimds.output.width === downscaled.status.presentation.output.width,
      "SSimDownscaler diagnostics do not match committed output");
    await changePopupControl(popupPage.client, "sharpen-str", 1.2);
    await waitForStatus(popupPage.client, fixtureTab.id, "sharpen strength", (status) =>
      Math.abs(status.sharpenStrength - 1.2) < 1e-9, signal);
    await changePopupControl(popupPage.client, "sharpen", true);
    const sharpened = await waitForStatus(popupPage.client, fixtureTab.id, "sharpened presentation", (status) =>
      status.sharpen === true && status.presentation?.sharpen?.strength === 1.2, signal);
    requireCondition(sharpened.status.presentation.sharpen.output.width === sharpened.status.presentation.output.width,
      "sharpen diagnostics do not match committed output");
    checkpoint("SSimDownscaler and sharpening");

    const firstSource = await fixtureSnapshot(fixturePage.client);
    await loadFixtureSource(fixturePage.client, "bt709-b");
    const replacement = await waitForStatus(popupPage.client, fixtureTab.id, "same-element source replacement", (status) =>
      status.mode === "upscale" && status.renderer?.phase === "active" &&
      status.presentation?.committed === true &&
      status.presentation.source.width !== firstSource.video.width, signal);
    const replacementPage = await fixtureSnapshot(fixturePage.client);
    requireCondition(replacementPage.video.currentSrc !== firstSource.video.currentSrc,
      "fixture did not replace the current media resource");
    requireCondition(replacementPage.video.width !== firstSource.video.width ||
      replacementPage.video.height !== firstSource.video.height,
    "replacement fixture did not change decoded dimensions");
    requireCondition(overlayByRole(replacementPage, "primary").length === 1,
      "source replacement left duplicate primary overlays");
    requireCondition(replacement.status.presentation.source.width === replacementPage.video.width &&
      replacement.status.presentation.source.height === replacementPage.video.height,
    "source replacement resumed with stale presentation dimensions");
    checkpoint("same-element source replacement");

    await loadFixtureSource(fixturePage.client, "bt2020-pq");
    const hdr = await waitForStatus(popupPage.client, fixtureTab.id, "HDR block", (status) =>
      status.mode === "upscale" && status.activeMode === "off" &&
      status.renderer?.phase === "blocked" && status.renderer?.blockedReason === "color-hdr-unsupported",
    signal);
    const hdrPage = await fixtureSnapshot(fixturePage.client);
    requireCondition(hdr.status.protectedReason === "color-hdr-unsupported", "HDR reason is not specific");
    requireCondition(!visibleOverlay(hdrPage, "primary") && hdrPage.video.display !== "none" &&
      hdrPage.video.visibility !== "hidden" && Number(hdrPage.video.opacity) !== 0,
    "HDR fallback did not leave the native video visible");
    await waitForBadge(popupPage.client, fixtureTab.id, "✕", signal);
    const hdrPopup = await waitFor("HDR popup banner", () => popupSnapshot(popupPage.client), (snapshot) =>
      !snapshot.banner.hidden && snapshot.banner.display !== "none" && /HDR video is unsupported/i.test(snapshot.banner.text),
    { timeoutMs: CDP_TIMEOUT_MS, signal });
    requireCondition(hdrPopup.offDisabled === false, "HDR block incorrectly disabled the Off control");

    await loadFixtureSource(fixturePage.client, "bt2020-sdr");
    await waitForStatus(popupPage.client, fixtureTab.id, "wide-gamut SDR block", (status) =>
      status.renderer?.phase === "blocked" && status.renderer?.blockedReason === "color-wide-gamut-unsupported",
    signal);
    await waitFor("wide-gamut popup banner", () => popupSnapshot(popupPage.client), (snapshot) =>
      /Wide-gamut video is unsupported/i.test(snapshot.banner.text), { timeoutMs: CDP_TIMEOUT_MS, signal });
    checkpoint("HDR and wide-gamut fail-closed fallback");

    await loadFixtureSource(fixturePage.client, "bt709-a");
    await waitForStatus(popupPage.client, fixtureTab.id, "BT.709 intent recovery", (status) =>
      status.mode === "upscale" && status.activeMode === "upscale" &&
      status.renderer?.phase === "active" && status.presentation?.committed === true, signal);
    await waitForBadge(popupPage.client, fixtureTab.id, "ON", signal);

    await fixturePage.client.send("Page.setLifecycleEventsEnabled", { enabled: true });
    await fixturePage.client.send("Page.setWebLifecycleState", { state: "frozen" });
    await waitForBadge(popupPage.client, fixtureTab.id, "", signal);
    await fixturePage.client.send("Page.setWebLifecycleState", { state: "active" });
    // A synthetic frozen→active transition can leave an already-frontmost CDP
    // target visibility-hidden even though Chromium emits `resume`. Move focus
    // away and back so this checkpoint represents a genuinely visible active
    // document, matching the state in which rendering is allowed to restart.
    await popupPage.client.send("Page.bringToFront");
    await fixturePage.client.send("Page.bringToFront");
    await waitFor("visible document after lifecycle activation", () => evaluate(
      fixturePage.client,
      "document.visibilityState",
    ), (state) => state === "visible", { timeoutMs: CDP_TIMEOUT_MS, signal });
    await waitForStatus(popupPage.client, fixtureTab.id, "document reactivation intent", (status) =>
      status.mode === "upscale" && status.documentSuspended === false,
    signal, CDP_TIMEOUT_MS);
    await waitFor("fixture execution context after resume", () => evaluate(
      fixturePage.client,
      "typeof window.__FSRCNNX_VIDEO_FIXTURE__?.loadSource === 'function'",
    ), Boolean, { timeoutMs: CDP_TIMEOUT_MS, signal });
    const lifecyclePage = await fixtureSnapshot(fixturePage.client);
    const lifecycleEvents = lifecyclePage.fixture?.lifecycleEvents || lifecyclePage.fixture?.lifecycle || [];
    const lifecycleTypes = Array.isArray(lifecycleEvents)
      ? lifecycleEvents.map((event) => typeof event === "string" ? event : event?.type)
      : [];
    requireCondition(lifecycleTypes.includes("freeze") && lifecycleTypes.includes("resume"),
      `fixture did not observe real freeze/resume events: ${JSON.stringify(lifecycleEvents)}`);

    // Chromium's synthetic CDP freeze can permanently starve this target's
    // decoder even after active/play/load/reload. Retire that proven lifecycle
    // owner and open the same origin in a fresh target, then verify stored
    // renderer intent reattaches and presents without another mode command.
    await closePageTarget(httpBase, fixturePage, { required: true, signal });
    fixturePage = await createPageTarget(
      httpBase,
      fixtureUrl,
      signal,
      (message) => collectRuntimeEvent(fixtureEvents, message),
    );
    await waitForDocument(fixturePage.client, fixtureUrl, signal);
    await waitFor("replacement video fixture API", () => evaluate(
      fixturePage.client,
      "typeof window.__FSRCNNX_VIDEO_FIXTURE__?.ready === 'function'",
    ), Boolean, { timeoutMs: CDP_TIMEOUT_MS, signal });
    await evaluate(fixturePage.client, "window.__FSRCNNX_VIDEO_FIXTURE__.ready()", CDP_TIMEOUT_MS);
    fixtureTab = await activateExtensionTab(popupPage.client, fixtureUrl);
    await fixturePage.client.send("Page.bringToFront");
    await waitForStatus(popupPage.client, fixtureTab.id, "post-freeze persisted renderer recovery", (status) =>
      status.mode === "upscale" && status.renderer?.phase === "active" &&
      status.presentation?.committed === true && status.presentation.generation > 0, signal);
    await waitForBadge(popupPage.client, fixtureTab.id, "ON", signal);
    checkpoint("CDP freeze/resume ownership");

    // Interpolation-dependent popup controls are intentionally disabled while
    // interpolation is off. Configure the pending engine/order through the same
    // validated content bridge, then use the real popup switch for activation.
    await sendContentCommand(popupPage.client, "FSRCNNX_SETINTERPMODEL", { key: "blend" }, fixtureTab.id);
    await waitForStatus(popupPage.client, fixtureTab.id, "blend model selection", (status) => status.interpModel === "blend", signal);
    await sendContentCommand(popupPage.client, "FSRCNNX_SETINTERPTARGETFPS", { value: 60 }, fixtureTab.id);
    await waitForStatus(popupPage.client, fixtureTab.id, "blend target selection", (status) => status.interpTargetFps === 60, signal);
    await sendContentCommand(popupPage.client, "FSRCNNX_SETINVERT", { on: false }, fixtureTab.id);
    await waitForStatus(popupPage.client, fixtureTab.id, "normal interpolation order", (status) => status.interpInvert === false, signal);
    await changePopupControl(popupPage.client, "interpolate", true);
    await waitForStatus(popupPage.client, fixtureTab.id, "blend interpolation activation", (status) =>
      status.interpolate === true && status.interpolationRuntime?.phase === "active" &&
      status.interpStats?.forceBlend === true && status.interpStats?.gpuPath === true &&
      status.interpStats?.framesIn > 1 && status.interpStats?.framesOut > 1 &&
      status.interpolationRuntime?.takeoverActive === true &&
      status.interpolationRuntime?.presentation?.committed === true, signal);
    const interpolationFirstPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "interpolation", "blend interpolation first composited frame", signal,
    );
    const postCaptureInterpolation = await waitForStatus(
      popupPage.client,
      fixtureTab.id,
      "blend interpolation post-capture ownership",
      (status) => status.interpolationRuntime?.phase === "active" &&
        status.interpolationRuntime?.takeoverActive === true &&
        status.interpolationRuntime?.presentation?.committed === true,
      signal,
    );
    const interpolationGeneration =
      postCaptureInterpolation.status.interpolationRuntime.presentation.generation;
    await waitForStatus(popupPage.client, fixtureTab.id, "blend interpolation frame progression", (status) =>
      status.interpolationRuntime?.phase === "active" &&
      status.interpolationRuntime?.takeoverActive === true &&
      status.interpolationRuntime?.presentation?.committed === true &&
      status.interpolationRuntime.presentation.generation >= interpolationGeneration + 3 &&
      status.interpStats?.framesPresented >= interpolationGeneration + 3, signal);
    const interpolationPage = await fixtureSnapshot(fixturePage.client);
    requireCondition(!!visibleOverlay(interpolationPage, "interpolation"),
      "blend interpolation did not commit a visible interpolation overlay");
    const interpolationPixels = await waitForRenderedOverlayPixels(
      fixturePage.client, "interpolation", "blend interpolation later composited frame", signal,
    );
    requirePixelProgression(
      interpolationFirstPixels, interpolationPixels, "blend interpolation compositor",
    );
    checkpoint("blend interpolation presentation");

    await clickPopupMode(popupPage.client, "off");
    await waitForStatus(popupPage.client, fixtureTab.id, "standalone interpolation", (status) =>
      status.mode === "off" && status.renderer?.phase === "off" &&
      status.interpolate === true && status.interpolationRuntime?.phase === "active" &&
      status.interpolationRuntime?.takeoverActive === true &&
      status.interpolationRuntime?.presentation?.committed === true, signal);
    const standalonePage = await fixtureSnapshot(fixturePage.client);
    requireCondition(!!visibleOverlay(standalonePage, "interpolation"),
      "standalone interpolation did not retain its direct overlay");
    await waitForRenderedOverlayPixels(
      fixturePage.client, "interpolation", "standalone interpolation compositor", signal,
    );

    await changePopupControl(popupPage.client, "interpolate", false);
    const stopped = await waitForStatus(popupPage.client, fixtureTab.id, "complete renderer teardown", (status) =>
      status.mode === "off" && status.interpolate === false &&
      status.renderer?.phase === "off" && status.interpolationRuntime?.phase === "off", signal);
    const stoppedFrames = stopped.status.frameCount;
    await delay(500, signal);
    const finalStatus = await contentStatus(popupPage.client, fixtureTab.id);
    const finalPage = await fixtureSnapshot(fixturePage.client);
    requireCondition(finalStatus.status.frameCount === stoppedFrames,
      "renderer frame counter advanced after complete teardown");
    requireCondition(finalPage.overlays.length === 0 || finalPage.overlays.every((overlay) =>
      !overlay.connected || overlay.display === "none" || overlay.rect.width === 0 || overlay.rect.height === 0),
    "extension overlay remained visible after complete teardown");
    requireCondition(finalPage.video.parentInlinePosition === "",
      `temporary positioned parent was not restored (${finalPage.video.parentInlinePosition})`);
    requireCondition(finalPage.video.display !== "none" && finalPage.video.visibility !== "hidden" &&
      Number(finalPage.video.opacity) !== 0 && finalPage.video.paused === false,
    "native video was not usable after teardown");
    await waitForBadge(popupPage.client, fixtureTab.id, "", signal);
    checkpoint("standalone interpolation and teardown");

    const finalPopup = await popupSnapshot(popupPage.client);
    requireCondition(finalPopup.modeStates.length === 3 &&
      finalPopup.modeStates.filter(({ pressed }) => pressed === "true").length === 1 &&
      finalPopup.modeStates.find(({ pressed }) => pressed === "true")?.mode === "off",
    "popup mode aria-pressed state is invalid after teardown");
    requireCondition(finalPopup.unnamedControls.length === 0,
      `popup controls lack accessible names: ${finalPopup.unnamedControls.join(", ")}`);
    requireCondition(finalPopup.duplicateIds.length === 0,
      `popup contains duplicate IDs: ${finalPopup.duplicateIds.join(", ")}`);
    requireCondition(finalPopup.banner.role === "alert" && finalPopup.operationRole === "status" &&
      finalPopup.runtimeLive === "polite", "popup live-region roles are incomplete");
    checkpoint("popup state and accessibility");

    assertRuntimeClean(fixtureEvents, "real-video fixture");
    assertRuntimeClean(popupEvents, "real popup");
    return { checkpoints, fixtureEvents, popupEvents };
  } catch (error) {
    error.browserEvents = [...fixtureEvents, ...popupEvents];
    throw error;
  } finally {
    await closePageTarget(httpBase, popupPage);
    await closePageTarget(httpBase, fixturePage);
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

async function main(signal, options) {
  const profile = await mkdtemp(join(tmpdir(), "fsrcnnx-browser-validation-"));
  let launched = null;
  let fixtureServer = null;
  let extensionControl = null;
  let primaryError = null;
  try {
    const [browser, extensionRoot] = await Promise.all([
      findBrowser(),
      realpath(options.extensionRoot),
    ]);
    const [manifestSource, version] = await Promise.all([
      readFile(join(extensionRoot, "manifest.json"), "utf8"),
      browserVersion(browser, signal),
    ]);
    const manifest = JSON.parse(manifestSource);
    if (manifest.manifest_version !== 3 ||
        manifest.background?.service_worker !== "src/background.js") {
      throw new Error("browser validator requires the project's MV3 src/background.js service worker");
    }
    fixtureServer = await startFixtureServer(signal);
    launched = await startBrowser(browser, profile, extensionRoot, signal);
    const browserWebSocket = await launched.endpoint;
    const httpBase = httpBaseFromWebSocket(browserWebSocket);
    await waitForDevToolsHttp(httpBase, signal);
    // Loading an MV3 extension does not guarantee that its event worker is
    // already running. A local fixture page injects the packaged content script;
    // its normal startup ownership message provides a deterministic,
    // extension-authored service-worker bootstrap.
    const bootstrapUrl = new URL("video.html", fixtureServer.baseUrl).href;
    const bootstrapPage = await createPageTarget(httpBase, bootstrapUrl, signal);
    await waitForDocument(bootstrapPage.client, bootstrapUrl, signal);
    let discovery;
    try {
      discovery = await discoverExtension(
        httpBase,
        manifest.name,
        manifest.background.service_worker,
        signal,
      );
    } finally {
      await closePageTarget(httpBase, bootstrapPage);
    }
    extensionControl = discovery.controlClient;
    await runPopupSmoke(
      httpBase,
      extensionControl,
      discovery.extensionId,
      manifest.name,
      signal,
    );
    const { state } = await runValidation(
      httpBase,
      extensionControl,
      discovery.extensionId,
      manifest.name,
      signal,
    );
    const integration = await runRealVideoIntegration(
      httpBase,
      extensionControl,
      fixtureServer.baseUrl,
      discovery.extensionId,
      manifest.name,
      signal,
    );
    const webGpu = state.results.find((result) => result.id === "webgpu");
    console.log(
      `Browser validation passed: ${state.pass}/${state.total} checks ` +
      `(${basename(browser)}, ID from ${discovery.source}).`,
    );
    console.log(`Browser: ${version}`);
    console.log(`Extension root: ${extensionRoot}`);
    if (webGpu?.detail) console.log(`WebGPU: ${webGpu.detail}`);
    const numericalOutput = numericalDiagnostics(state);
    if (numericalOutput) console.log(`Numerical references:\n${numericalOutput}`);
    console.log("Popup browser smoke passed: module, unavailable state, controls, and accessibility.");
    console.log(
      `Real-video integration passed: ${integration.checkpoints.length}/${integration.checkpoints.length} checkpoints ` +
      `(${integration.checkpoints.join("; ")}).`,
    );
  } catch (error) {
    primaryError = error;
    if (launched?.output()) error.browserOutput = launched.output();
    throw error;
  } finally {
    extensionControl?.close();
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
      } finally {
        try {
          await fixtureServer?.close();
        } catch (error) {
          if (!primaryError) throw error;
          console.error(`Fixture-server cleanup warning: ${error.message}`);
        }
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
  await main(abortController.signal, parseArguments(process.argv.slice(2)));
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

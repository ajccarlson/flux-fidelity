import test from "node:test";
import assert from "node:assert/strict";
import { ImageUpscaler } from "../fsrcnnx-images.js";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

class FakeImageElement {
  constructor(src, { width = 100, height = 100, display = 220 } = {}) {
    this.nodeType = 1;
    this.tagName = "IMG";
    this.dataset = {};
    this.naturalWidth = width;
    this.naturalHeight = height;
    this.isConnected = true;
    this._display = display;
    this._attrs = new Map([["src", src]]);
    this.src = src;
    this.currentSrc = src;
    this.srcset = "";
    this._listeners = new Map();
  }
  getBoundingClientRect() { return { width: this._display, height: this._display }; }
  getAttribute(name) { return this._attrs.has(name) ? this._attrs.get(name) : null; }
  hasAttribute(name) { return this._attrs.has(name); }
  setAttribute(name, value) {
    const text = String(value);
    this._attrs.set(name, text);
    if (name === "src") { this.src = text; this.currentSrc = text; }
    if (name === "srcset") this.srcset = text;
  }
  removeAttribute(name) {
    this._attrs.delete(name);
    if (name === "srcset") this.srcset = "";
  }
  addEventListener(type, handler) { this._listeners.set(type, handler); }
  removeEventListener(type, handler) { if (this._listeners.get(type) === handler) this._listeners.delete(type); }
}

class FakeIntersectionObserver {
  constructor(callback) { this.callback = callback; this.observed = new Set(); }
  observe(node) { this.observed.add(node); }
  unobserve(node) { this.observed.delete(node); }
  disconnect() { this.observed.clear(); }
}

class FakeMutationObserver {
  constructor(callback) { this.callback = callback; }
  observe() {}
  disconnect() {}
}

function installDom(images) {
  const previous = {
    document: globalThis.document,
    window: globalThis.window,
    IntersectionObserver: globalThis.IntersectionObserver,
    MutationObserver: globalThis.MutationObserver,
  };
  const root = {
    nodeType: 1,
    tagName: "BODY",
    querySelectorAll(selector) { return selector === "img" ? images : []; },
  };
  globalThis.document = { body: root, documentElement: root, querySelectorAll: root.querySelectorAll.bind(root) };
  globalThis.window = { devicePixelRatio: 1 };
  globalThis.IntersectionObserver = FakeIntersectionObserver;
  globalThis.MutationObserver = FakeMutationObserver;
  return () => Object.assign(globalThis, previous);
}

function makeUpscaler(images, { warnings = [], errors = [], counts = [] } = {}) {
  const pipe = { getBindGroupLayout: () => ({}) };
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    createShaderModule: () => ({}),
    createComputePipeline: () => pipe,
    createRenderPipeline: () => pipe,
  };
  class Model { constructor() { this.scale = 2; } destroy() {} }
  class Downscaler { destroy() {} }
  return new ImageUpscaler({
    device, format: "bgra8unorm", sampler: {}, fsrcnnxSource: { manifest: {}, wgsl: "" },
    FsrcnnxModel: Model, SsimDownscaler: Downscaler,
    onCount: (count) => counts.push(count), onError: (error) => errors.push(error),
    warn: (message) => warnings.push(message),
  });
}

function bitmap() {
  return { width: 100, height: 100, closed: false, close() { this.closed = true; } };
}

test("disable and re-enable resets one-shot processed state", async (t) => {
  const img = new FakeImageElement("https://example.test/a.png");
  const cleanup = installDom([img]); t.after(cleanup);
  const counts = [];
  const up = makeUpscaler([img], { counts });
  let calls = 0;
  up.loadReadable = async () => bitmap();
  up.upscaleAndReplace = async () => { calls++; return true; };

  up.start();
  assert.equal(await up.tryProcess(img), true);
  assert.equal(await up.tryProcess(img), false);
  assert.equal(calls, 1);
  up.stop();
  await tick();
  up.start();
  assert.equal(await up.tryProcess(img), true);
  assert.equal(calls, 2);
  assert.deepEqual(counts.filter((value) => value === 0), [0]);
  up.stop();
});

test("shared model work is serialized and a stopped run cannot increment results", async (t) => {
  const one = new FakeImageElement("https://example.test/one.png");
  const two = new FakeImageElement("https://example.test/two.png");
  const three = new FakeImageElement("https://example.test/three.png");
  const cleanup = installDom([one, two, three]); t.after(cleanup);
  const up = makeUpscaler([one, two, three]);
  const releases = [];
  let active = 0, maxActive = 0;
  up.loadReadable = async () => bitmap();
  up.upscaleAndReplace = async () => {
    active++; maxActive = Math.max(maxActive, active);
    await new Promise((resolve) => releases.push(resolve));
    active--;
    return true;
  };

  up.start();
  const first = up.tryProcess(one);
  const second = up.tryProcess(two);
  await tick();
  assert.equal(releases.length, 1);
  releases.shift()();
  assert.equal(await first, true);
  await tick();
  assert.equal(releases.length, 1);
  up.stop();
  up.start();
  const third = up.tryProcess(three);
  await tick();
  assert.equal(releases.length, 1, "new generation waits for old active work");
  releases.shift()();
  assert.equal(await second, false);
  await tick();
  assert.equal(releases.length, 1);
  releases.shift()();
  assert.equal(await third, true);
  assert.equal(maxActive, 1);
  assert.equal(up.count, 1);
  up.stop();
});

test("processing failures close bitmaps and are rate-limited by category", async (t) => {
  const one = new FakeImageElement("https://example.test/one.png");
  const two = new FakeImageElement("https://example.test/two.png");
  const cleanup = installDom([one, two]); t.after(cleanup);
  const warnings = [], errors = [], opened = [];
  const up = makeUpscaler([one, two], { warnings, errors });
  up.loadReadable = async () => { const value = bitmap(); opened.push(value); return value; };
  up.upscaleAndReplace = async () => { throw new Error("mock GPU failure"); };

  up.start();
  await Promise.all([up.tryProcess(one), up.tryProcess(two)]);
  assert.equal(warnings.length, 1);
  assert.equal(errors.length, 2);
  assert.ok(opened.every((value) => value.closed));
  assert.deepEqual(up.getStats().failures, { processing: 2 });
  up.stop();
});

test("external source changes release owned URLs without overwriting the new src", (t) => {
  const img = new FakeImageElement("old.png");
  img.setAttribute("srcset", "old@2x.png 2x");
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  const oldRevoke = URL.revokeObjectURL;
  const revoked = [];
  URL.revokeObjectURL = (url) => revoked.push(url);
  t.after(() => { URL.revokeObjectURL = oldRevoke; });
  up.start();
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  img.setAttribute("src", "blob:owned");
  img.setAttribute("srcset", "");
  img.dataset.fsrcnnxDone = "1";
  up.replaced.add(img); up.count = 1;

  img.setAttribute("src", "new.png");
  up._handleMutations([{ type: "attributes", target: img, attributeName: "src" }]);
  assert.equal(img.getAttribute("src"), "new.png");
  assert.equal(img.getAttribute("srcset"), "old@2x.png 2x");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(up.count, 0);
  assert.equal(img.dataset.fsrcnnxDone, undefined);
  up.stop();
});

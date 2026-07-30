import test from "node:test";
import assert from "node:assert/strict";
import { ImageUpscaler } from "../src/core/fsrcnnx-images.js";

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
  static instances = [];
  constructor(callback) {
    this.callback = callback;
    this.records = [];
    this.disconnected = false;
    FakeMutationObserver.instances.push(this);
  }
  observe(root) { this.root = root; }
  disconnect() { this.disconnected = true; this.records = []; }
  takeRecords() { return this.records.splice(0); }
  enqueue(...records) { this.records.push(...records); }
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
  FakeMutationObserver.instances.length = 0;
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

test("image dimensions have no fixed pixel-area ceiling below the adapter limit", async () => {
  const warnings = [], errors = [];
  const up = makeUpscaler([], { warnings, errors });

  assert.equal(up._dimensionsAllowed(4096, 2500, 8192, 5000, true), true,
    "an x2 output above the former 8K-area budget remains adapter-valid");
  assert.equal(up._dimensionsAllowed(4097, 2500, 8194, 5000, true), false,
    "the adapter's per-axis texture limit still applies");
  assert.equal(errors.length, 1);
  assert.equal(errors[0].code, "limits");
  assert.match(errors[0].message, /8192px per edge/);
  await up.destroy();
});

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

test("images selected by picture sources fail closed instead of counting an unused blob", async (t) => {
  const img = new FakeImageElement("fallback.png");
  img.currentSrc = "selected-by-source.webp";
  const source = {
    tagName: "SOURCE",
    getAttribute(name) { return name === "srcset" ? "selected-by-source.webp 1x" : null; },
  };
  const picture = {
    tagName: "PICTURE",
    querySelectorAll(selector) { return selector === "source" ? [source] : []; },
  };
  img.closest = (selector) => selector === "picture" ? picture : null;
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  let reads = 0;
  up.loadReadable = async () => { reads++; return bitmap(); };

  up.start();
  assert.equal(await up.tryProcess(img), false);
  assert.equal(reads, 0, "the expensive GPU/readback path must not start");
  assert.equal(up.count, 0);
  assert.equal(img.getAttribute("src"), "fallback.png");
  assert.equal(img.currentSrc, "selected-by-source.webp");
  assert.equal(img._fsrcnnxURL, undefined);
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

test("destroy waits for active work before releasing shared GPU caches", async (t) => {
  const img = new FakeImageElement("https://example.test/active.png");
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  let release;
  let modelDestroyed = false;
  let downscalerDestroyed = false;
  up.model.destroy = () => { modelDestroyed = true; };
  up.ssimds.destroy = () => { downscalerDestroyed = true; };
  up.loadReadable = async () => bitmap();
  up.upscaleAndReplace = async () => {
    await new Promise((resolve) => { release = resolve; });
    return true;
  };

  up.start();
  const work = up.tryProcess(img);
  await tick();
  const retirement = up.destroy();
  assert.equal(modelDestroyed, false);
  assert.equal(downscalerDestroyed, false);
  assert.equal(up.running, false);

  release();
  assert.equal(await work, false, "the stopped generation cannot publish its result");
  await retirement;
  assert.equal(modelDestroyed, true);
  assert.equal(downscalerDestroyed, true);
  assert.equal(up.model, null);
  assert.equal(up.ssimds, null);
  assert.equal(up.extractPipe, null);
  assert.equal(up.recombinePipe, null);
  assert.equal(up.blitPipe, null);
  assert.equal(up.device, null);
  assert.equal(up.sampler, null);
  assert.strictEqual(up.destroy(), retirement, "destroy is idempotent while and after retirement");
});

test("destroy publishes its promise before a count callback can re-enter teardown", async (t) => {
  const cleanup = installDom([]); t.after(cleanup);
  const up = makeUpscaler([]);
  let modelReleases = 0;
  let callbackCalls = 0;
  let reentrantDestroy = null;
  up.model.destroy = () => { modelReleases++; };
  up.onCount = () => {
    callbackCalls++;
    reentrantDestroy = up.destroy();
  };

  const retirement = up.destroy();
  assert.strictEqual(reentrantDestroy, retirement,
    "synchronous teardown re-entry must observe the published single-flight promise");
  await retirement;
  assert.equal(callbackCalls, 1);
  assert.equal(modelReleases, 1);
  assert.strictEqual(up.destroy(), retirement);
});

test("constructor failure releases every previously created owned GPU resource", () => {
  const destroyed = [];
  const resource = (name) => ({ destroy() { destroyed.push(name); } });
  let renderPipelines = 0;
  const device = {
    createShaderModule: () => ({}),
    createComputePipeline: () => resource("extract pipeline"),
    createRenderPipeline: () => {
      renderPipelines++;
      if (renderPipelines === 2) throw new Error("blit pipeline failed");
      return resource("recombine pipeline");
    },
  };
  class Model {
    constructor() { this.scale = 2; }
    destroy() { destroyed.push("model"); }
  }
  class Downscaler { destroy() { destroyed.push("downscaler"); } }

  assert.throws(() => new ImageUpscaler({
    device,
    format: "bgra8unorm",
    sampler: {},
    fsrcnnxSource: { manifest: {}, wgsl: "" },
    FsrcnnxModel: Model,
    SsimDownscaler: Downscaler,
  }), /blit pipeline failed/);
  assert.deepEqual(new Set(destroyed), new Set([
    "model", "downscaler", "extract pipeline", "recombine pipeline",
  ]));
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

test("batched external src and srcset writes both survive replacement retirement", (t) => {
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
  img.setAttribute("srcset", "");
  img.setAttribute("src", "blob:owned");
  img.dataset.fsrcnnxDone = "1";
  up.replaced.add(img); up.count = 1;

  img.setAttribute("src", "new.png");
  img.setAttribute("srcset", "new-small.png 1x, new-large.png 2x");
  up._handleMutations([
    { type: "attributes", target: img, attributeName: "src", oldValue: "blob:owned" },
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "" },
  ]);

  assert.equal(img.getAttribute("src"), "new.png");
  assert.equal(img.getAttribute("srcset"), "new-small.png 1x, new-large.png 2x");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(up.count, 0);
  assert.equal(img.dataset.fsrcnnxDone, undefined);
  up.stop();
});

test("stop drains queued page writes before restoring owned image attributes", (t) => {
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
  up._ownWrite(img, () => {
    img.setAttribute("srcset", "");
    img.setAttribute("src", "blob:owned");
    img.dataset.fsrcnnxDone = "1";
  });
  up.replaced.add(img); up.count = 1;

  img.setAttribute("src", "page.png");
  // This same-value write is observable only through its queued mutation. It
  // intentionally keeps srcset empty instead of restoring the old candidate.
  img.setAttribute("srcset", "");
  FakeMutationObserver.instances[0].enqueue(
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "old@2x.png 2x" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "old.png" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "blob:owned" },
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "" },
  );

  up.stop();

  assert.equal(img.getAttribute("src"), "page.png");
  assert.equal(img.getAttribute("srcset"), "");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(img.dataset.fsrcnnxDone, undefined);
});

test("stop preserves page attributes when pending observer records are unavailable", (t) => {
  const img = new FakeImageElement("old.png");
  img.setAttribute("srcset", "old@2x.png 2x");
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  up.start();
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  img.setAttribute("src", "blob:owned");
  img.setAttribute("srcset", "");
  img.dataset.fsrcnnxDone = "1";
  up.replaced.add(img); up.count = 1;
  FakeMutationObserver.instances[0].takeRecords = undefined;

  img.setAttribute("src", "page.png");
  img.setAttribute("srcset", "page@2x.png 2x");
  up.stop();

  assert.equal(img.getAttribute("src"), "page.png");
  assert.equal(img.getAttribute("srcset"), "page@2x.png 2x");
  assert.equal(img.dataset.fsrcnnxDone, undefined);
});

test("removing a shadow host disconnects and forgets its root observer", (t) => {
  const cleanup = installDom([]); t.after(cleanup);
  const up = makeUpscaler([]);
  const shadowRoot = {
    nodeType: 11,
    querySelectorAll() { return []; },
  };
  const host = {
    nodeType: 1,
    tagName: "DIV",
    shadowRoot,
    querySelectorAll() { return []; },
  };

  up.start();
  up._scanRoot(host);
  const shadowObserver = up._mutationObservers.get(shadowRoot);
  assert.ok(shadowObserver);
  assert.equal(up._mutationObservers.size, 2);

  up._handleMutations([{
    type: "childList",
    target: document.body,
    removedNodes: [host],
    addedNodes: [],
  }]);

  assert.equal(shadowObserver.disconnected, true);
  assert.equal(up._mutationObservers.has(shadowRoot), false);
  assert.equal(up._mutationObservers.size, 1);
  up.stop();
});

test("shadow-host removal drains pending page writes before restoring the image", (t) => {
  const img = new FakeImageElement("old.png");
  img.setAttribute("srcset", "old@2x.png 2x");
  const cleanup = installDom([]); t.after(cleanup);
  const up = makeUpscaler([]);
  const oldRevoke = URL.revokeObjectURL;
  const revoked = [];
  URL.revokeObjectURL = (url) => revoked.push(url);
  t.after(() => { URL.revokeObjectURL = oldRevoke; });
  const shadowRoot = {
    nodeType: 11,
    querySelectorAll(selector) { return selector === "img" ? [img] : []; },
  };
  const host = {
    nodeType: 1,
    tagName: "DIV",
    shadowRoot,
    querySelectorAll() { return []; },
  };

  up.start();
  up._scanRoot(host);
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  up._ownWrite(img, () => {
    img.setAttribute("srcset", "");
    img.setAttribute("src", "blob:owned");
    img.dataset.fsrcnnxDone = "1";
  });
  up.replaced.add(img); up.count = 1;

  // The page intentionally keeps the owned empty srcset, then removes the host
  // in the same task. The document observer wins delivery order, so these shadow
  // records are available only through takeRecords() during disconnection.
  img.setAttribute("srcset", "");
  const shadowObserver = up._mutationObservers.get(shadowRoot);
  shadowObserver.enqueue(
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "old@2x.png 2x" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "old.png" },
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "" },
  );
  up._handleMutations([{
    type: "childList",
    target: document.body,
    removedNodes: [host],
    addedNodes: [],
  }]);

  assert.equal(img.getAttribute("src"), "old.png");
  assert.equal(img.getAttribute("srcset"), "",
    "the page's same-value write must survive shadow observer disconnection");
  assert.equal(img.dataset.fsrcnnxDone, undefined);
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(shadowObserver.disconnected, true);
  assert.equal(up._mutationObservers.has(shadowRoot), false);
  up.stop();
});

test("same-task page writes are distinguished from the upscaler's own mutations", (t) => {
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
  up._ownWrite(img, () => {
    img.setAttribute("srcset", "");
    img.setAttribute("src", "blob:owned");
    img.dataset.fsrcnnxDone = "1";
  });
  up.replaced.add(img); up.count = 1;

  // MutationObserver delivers all four records together at the end of the
  // task. The first two belong to us; the latter two belong to the page.
  img.setAttribute("src", "page.png");
  img.setAttribute("srcset", "page@1x.png 1x, page@2x.png 2x");
  up._handleMutations([
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "old@2x.png 2x" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "old.png" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "blob:owned" },
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "" },
  ]);

  assert.equal(img.getAttribute("src"), "page.png");
  assert.equal(img.getAttribute("srcset"), "page@1x.png 1x, page@2x.png 2x");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(up.count, 0);
  up.stop();
});

test("the upscaler's own delivered attribute records retain replacement ownership", (t) => {
  const img = new FakeImageElement("old.png");
  img.setAttribute("srcset", "old@2x.png 2x");
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  up.start();
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  up._ownWrite(img, () => {
    img.setAttribute("srcset", "");
    img.setAttribute("src", "blob:owned");
    img.dataset.fsrcnnxDone = "1";
  });
  up.replaced.add(img); up.count = 1;

  up._handleMutations([
    { type: "attributes", target: img, attributeName: "srcset", oldValue: "old@2x.png 2x" },
    { type: "attributes", target: img, attributeName: "src", oldValue: "old.png" },
  ]);

  assert.equal(img.getAttribute("src"), "blob:owned");
  assert.equal(img.getAttribute("srcset"), "");
  assert.equal(img.dataset.fsrcnnxDone, "1");
  assert.equal(up.count, 1);
  up.stop();
});

test("picture-driven currentSrc changes retire a no-longer-effective replacement", (t) => {
  const img = new FakeImageElement("fallback.png");
  img.setAttribute("srcset", "fallback@2x.png 2x");
  const picture = {
    nodeType: 1,
    tagName: "PICTURE",
    querySelectorAll(selector) { return selector === "img" ? [img] : []; },
  };
  const source = { nodeType: 1, tagName: "SOURCE", parentElement: picture };
  img.parentElement = picture;
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  const oldRevoke = URL.revokeObjectURL;
  const revoked = [];
  URL.revokeObjectURL = (url) => revoked.push(url);
  t.after(() => { URL.revokeObjectURL = oldRevoke; });
  up.start();
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  img.setAttribute("srcset", "");
  img.setAttribute("src", "blob:owned");
  img.dataset.fsrcnnxDone = "1";
  up.replaced.add(img); up.count = 1;

  const selectedByPicture = "https://example.test/wide.png";
  Object.defineProperty(img, "currentSrc", {
    configurable: true,
    get: () => selectedByPicture,
    set: () => {},
  });
  up._handleMutations([
    { type: "attributes", target: source, attributeName: "media", oldValue: "(min-width: 1200px)" },
  ]);

  assert.equal(img.currentSrc, selectedByPicture);
  assert.equal(img.getAttribute("src"), "fallback.png");
  assert.equal(img.getAttribute("srcset"), "fallback@2x.png 2x");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(up.count, 0);
  assert.equal(img.dataset.fsrcnnxDone, undefined);
  up.stop();
});

test("responsive load events validate effective currentSrc instead of the src attribute", (t) => {
  const img = new FakeImageElement("fallback.png");
  const cleanup = installDom([img]); t.after(cleanup);
  const up = makeUpscaler([img]);
  const oldRevoke = URL.revokeObjectURL;
  const revoked = [];
  URL.revokeObjectURL = (url) => revoked.push(url);
  t.after(() => { URL.revokeObjectURL = oldRevoke; });
  up.start();
  up._captureOriginal(img);
  img._fsrcnnxURL = "blob:owned";
  img.setAttribute("srcset", "");
  img.setAttribute("src", "blob:owned");
  img.dataset.fsrcnnxDone = "1";
  up.replaced.add(img); up.count = 1;

  const selectedByPicture = "https://example.test/narrow.png";
  Object.defineProperty(img, "currentSrc", {
    configurable: true,
    get: () => selectedByPicture,
    set: () => {},
  });
  up._observedImages.get(img)();

  assert.equal(img.currentSrc, selectedByPicture);
  assert.equal(img.getAttribute("src"), "fallback.png");
  assert.deepEqual(revoked, ["blob:owned"]);
  assert.equal(up.count, 0);
  up.stop();
});

test("a wide-gamut display warns once and still processes the image", async (t) => {
  const one = new FakeImageElement("https://example.test/wide-1.png");
  const two = new FakeImageElement("https://example.test/wide-2.png");
  const cleanup = installDom([one, two]); t.after(cleanup);
  const previous = Object.getOwnPropertyDescriptor(globalThis, "matchMedia");
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "matchMedia", previous);
    else delete globalThis.matchMedia;
  });
  const queries = [];
  Object.defineProperty(globalThis, "matchMedia", {
    configurable: true, writable: true,
    value: (query) => { queries.push(query); return { matches: true }; },
  });

  const warnings = [];
  const up = makeUpscaler([one, two], { warnings });
  let loads = 0;
  up.loadReadable = async () => { loads++; return bitmap(); };
  up.upscaleAndReplace = async () => true;

  up.start();
  await up.tryProcess(one);
  await up.tryProcess(two);

  // Source gamut is not inspectable from the web platform, and the round trip only
  // loses anything when the source itself is wide-gamut, so refusing outright would
  // disable the feature on most modern displays.
  assert.equal(loads, 2, "a wide-gamut display must not block processing");
  const notices = warnings.filter((message) => /wide-gamut/.test(String(message)));
  assert.equal(notices.length, 1, "the notice is emitted once, not per image");
  assert.deepEqual(up.getStats().failures, {}, "a warning is not a failure");
  assert.deepEqual(queries, ["(color-gamut: p3)"], "the gamut probe is cached after one read");
  up.stop();
});

test("an sRGB display still processes images", async (t) => {
  const img = new FakeImageElement("https://example.test/srgb.png");
  const cleanup = installDom([img]); t.after(cleanup);
  const previous = Object.getOwnPropertyDescriptor(globalThis, "matchMedia");
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "matchMedia", previous);
    else delete globalThis.matchMedia;
  });
  Object.defineProperty(globalThis, "matchMedia", {
    configurable: true, writable: true, value: () => ({ matches: false }),
  });

  const up = makeUpscaler([img]);
  let loads = 0;
  up.loadReadable = async () => { loads++; return bitmap(); };
  up.upscaleAndReplace = async () => true;

  up.start();
  await up.tryProcess(img);

  assert.equal(loads, 1);
  up.stop();
});

test("the image presentation blit reads alpha from the original source", () => {
  const bindings = [];
  const pipe = { getBindGroupLayout: () => ({}) };
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    createShaderModule: ({ code }) => ({ code }),
    createComputePipeline: () => pipe,
    createRenderPipeline: ({ fragment }) => { bindings.push(fragment.module.code); return pipe; },
  };
  class Model { constructor() { this.scale = 2; } destroy() {} }
  class Downscaler { destroy() {} }
  new ImageUpscaler({
    device, format: "bgra8unorm", sampler: {}, fsrcnnxSource: { manifest: {}, wgsl: "" },
    FsrcnnxModel: Model, SsimDownscaler: Downscaler,
    onCount: () => {}, onError: () => {}, warn: () => {},
  });

  // The recombine and SSimDownscaler tails both emit a hardcoded 1.0, so the blit
  // must take coverage from the source or every transparent image ships opaque.
  const blit = bindings.find((code) => code.includes("srcAlpha"));
  assert.ok(blit, "the blit shader must sample a source-alpha binding");
  assert.match(blit, /@group\(0\) @binding\(2\) var srcAlpha/);
  assert.match(blit, /vec4f\(rgb \* a, a\)/, "output must be premultiplied to match the canvas");
});

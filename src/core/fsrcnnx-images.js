// fsrcnnx-images.js — upscales qualifying <img> elements with a strong FSRCNNX (x2) +
// SSimDownscaler, as an advanced opt-in. Self-contained: owns its source/extract/
// recombine pipelines (the video path's luma extract reads texture_external, which
// is video-only), reuses the shared device + a dedicated FSRCNNX model + an
// SSimDownscaler passed in from the main module.
//
// Smart filter: only images the browser is visibly stretching UP (displayed size
// meaningfully larger than natural size) and that are a sensible size to process.
// Lazy: IntersectionObserver processes images as they near the viewport; a
// MutationObserver picks up images added later (feeds). One-shot per image (no
// per-frame loop) — the result replaces the image source and is restored on stop.

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";

const LUMA_FROM_TEX = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var outLuma : texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let dim = textureDimensions(outLuma);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(dim);
  let rgb = textureSampleLevel(src, samp, uv, 0.0).rgb;
  let y = dot(rgb, vec3f(0.2126, 0.7152, 0.0722)); // BT.709 luma
  textureStore(outLuma, vec2i(i32(gid.x), i32(gid.y)), vec4f(y, 0.0, 0.0, 1.0));
}`;

// Recombine upscaled luma with chroma from the (bilinearly upscaled) source RGB.
const RECOMBINE_IMG = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var srcRGB : texture_2d<f32>;
@group(0) @binding(2) var hiLuma : texture_2d<f32>;
struct VsOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VsOut{
  var p=array<vec2f,3>(vec2f(-1.,-3.),vec2f(-1.,1.),vec2f(3.,1.));
  var u=array<vec2f,3>(vec2f(0.,2.),vec2f(0.,0.),vec2f(2.,0.));
  var o:VsOut; o.pos=vec4f(p[i],0.,1.); o.uv=u[i]; return o;
}
fn rgb2y(c:vec3f)->f32 { return dot(c, vec3f(0.2126,0.7152,0.0722)); }
@fragment fn fs(@location(0) uv:vec2f)->@location(0) vec4f {
  let rgb = textureSampleLevel(srcRGB, samp, uv, 0.0).rgb; // bilinear chroma source
  let yNew = textureSampleLevel(hiLuma, samp, uv, 0.0).r;  // sharpened luma
  let yOld = rgb2y(rgb);
  // scale chroma by luma ratio (preserve hue/sat, swap luminance).
  // select() evaluates both operands, so the divisor is floored rather than
  // relying on the condition — otherwise every flat-black pixel computed an
  // Inf in the branch that is discarded.
  let ratio = select(yNew / max(yOld, 0.0001), 1.0, yOld < 0.0001);
  let outc = clamp(rgb * ratio, vec3f(0.0), vec3f(1.0));
  return vec4f(outc, 1.0);
}`;

// Alpha is resampled from the ORIGINAL image rather than carried through the
// chain: the luma/chroma recombine and the SSimDownscaler tail both emit a
// hardcoded 1.0, so every transparent PNG or WebP came back fully opaque and then
// overwrote the page's <img>. Alpha is a coverage channel, not detail, so taking
// it bilinearly from the source is also the correct filter for it.
const BLIT = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var srcAlpha : texture_2d<f32>;
struct VsOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VsOut{
  var p=array<vec2f,3>(vec2f(-1.,-3.),vec2f(-1.,1.),vec2f(3.,1.));
  var u=array<vec2f,3>(vec2f(0.,2.),vec2f(0.,0.),vec2f(2.,0.));
  var o:VsOut; o.pos=vec4f(p[i],0.,1.); o.uv=u[i]; return o;
}
@fragment fn fs(@location(0) uv:vec2f)->@location(0) vec4f {
  let rgb = textureSampleLevel(src, samp, uv, 0.0).rgb;
  let a = textureSampleLevel(srcAlpha, samp, uv, 0.0).a;
  // The presentation canvas is configured alphaMode "premultiplied".
  return vec4f(rgb * a, a);
}`;

const MIN_SOURCE_EDGE = 64;
const MIN_SOURCE_PIXELS = 96 * 96;
const MAX_PENDING = 32;
const MAX_DEFERRED = 128;
const IMAGE_LOAD_TIMEOUT_MS = 15000;
const FAILURE_LOG_INTERVAL_MS = 30000;

class ImageSkipError extends Error {
  constructor(code, message) { super(message); this.name = "ImageSkipError"; this.code = code; }
}

export class ImageUpscaler {
  constructor({ device, format, sampler, fsrcnnxSource, FsrcnnxModel, SsimDownscaler, onCount, onError, warn }) {
    this.device = device;
    this.format = format;
    this.sampler = sampler;
    this.onCount = onCount || (() => {});
    this.onError = onError || (() => {});
    this.warn = warn || ((...args) => console.warn("[FSRCNNX images]", ...args));
    this.count = 0;
    this.running = false;
    this.processed = new WeakSet(); // imgs we've handled (success or skip)
    this.replaced = new Set();      // imgs whose src we replaced (for restore)
    this._writing = new WeakMap();  // img -> nested synchronous writes
    this._ownedMutations = new WeakMap(); // img -> expected src/srcset transitions
    this._revisions = new WeakMap();
    this._jobs = new WeakMap();
    this._queue = [];
    this._active = null;
    this._activePromise = null;
    this._deferred = new Set();
    this._runGeneration = 0;
    this._pendingLoads = new Set();
    this._loadCancelByImage = new WeakMap();
    this._observedImages = new Map();
    // Keep the observed root alongside its observer. A WeakSet cannot be
    // enumerated when a shadow host is detached, which left both the observer
    // and its shadow tree retained until the whole image feature stopped.
    this._mutationObservers = new Map();
    this._failures = new Map();
    this._destroyed = false;
    this._destroyPromise = null;

    this.model = null;
    this.ssimds = null;
    this.extractPipe = null;
    this.recombinePipe = null;
    this.blitPipe = null;

    // Construction is transactional. Model/downscaler constructors allocate
    // caches, so a later shader or pipeline failure must not strand those
    // allocations behind a constructor that never returned to its caller.
    try {
      this.model = fsrcnnxSource
        ? new FsrcnnxModel(device, fsrcnnxSource.manifest, fsrcnnxSource.wgsl,
          { expectedName: fsrcnnxSource.name })
        : null;
      this.ssimds = new SsimDownscaler(device);

      this.extractPipe = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: LUMA_FROM_TEX }), entryPoint: "main" },
      });
      const rmod = device.createShaderModule({ code: RECOMBINE_IMG });
      this.recombinePipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: rmod, entryPoint: "vs" },
        fragment: { module: rmod, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
        primitive: { topology: "triangle-list" },
      });
      const bmod = device.createShaderModule({ code: BLIT });
      this.blitPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: bmod, entryPoint: "vs" },
        fragment: { module: bmod, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
    } catch (error) {
      this._releaseOwnedGpuResources();
      this.device = null;
      this.sampler = null;
      throw error;
    }
  }

  // --- smart filter ---------------------------------------------------------
  isCandidate(img) {
    if (this.processed.has(img)) return false;
    if (img.dataset && img.dataset.fsrcnnxDone) return false; // already replaced by us
    if (!this.model || ("isConnected" in img && !img.isConnected)) return false;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh) return false;                 // not loaded / not raster
    const src = img.currentSrc || img.src || "";
    if (src.startsWith("data:")) return false;     // inline data URI
    if (src.startsWith("blob:")) return false;      // our own (or page's) blob
    if (/\.svg(\?|$)/i.test(src)) return false;     // vector, no benefit
    // A matching <picture><source> wins over attributes on the fallback <img>.
    // Replacing only img.src/srcset would create and count an unused blob while
    // currentSrc remains page-owned. Until source ownership can be transactional,
    // fail closed for pictures which expose any responsive source candidates.
    if (this._pictureHasSourceCandidates(img)) return false;
    if (nw < MIN_SOURCE_EDGE || nh < MIN_SOURCE_EDGE || nw * nh < MIN_SOURCE_PIXELS) return false;
    const target = this._targetDimensions(img, nw, nh);
    if (!target) return false;
    // Only process pixels the page is actually stretching. This avoids spending
    // the shared GPU queue on thumbnails, avatars, and already-downscaled photos.
    if (target.w <= nw * 1.05 && target.h <= nh * 1.05) return false;
    if (!this._dimensionsAllowed(nw, nh, target.w, target.h, true)) return false;
    return true;
  }

  _pictureHasSourceCandidates(img) {
    let picture = null;
    try { picture = img?.closest?.("picture") || null; } catch {}
    if (!picture) {
      for (let parent = img?.parentElement; parent; parent = parent.parentElement) {
        if (parent.tagName === "PICTURE") { picture = parent; break; }
      }
    }
    if (!picture) return false;
    try {
      return [...(picture.querySelectorAll?.("source") || [])].some((source) =>
        String(source.getAttribute?.("srcset") || source.srcset || "").trim().length > 0);
    } catch {
      // An inaccessible/hostile picture subtree is not safe to rewrite.
      return true;
    }
  }

  _targetDimensions(img, nw, nh) {
    let rect;
    try { rect = img.getBoundingClientRect(); } catch { return null; }
    const dpr = window.devicePixelRatio || 1;
    const boxW = Math.max(0, rect.width * dpr);
    const boxH = Math.max(0, rect.height * dpr);
    if (!boxW || !boxH || !this.model) return null;
    const scale = Number(this.model.scale);
    if (!Number.isFinite(scale) || scale < 1) return null;
    const aspect = nw / nh;
    let w, h;
    if (boxW / boxH > aspect) {
      h = Math.round(Math.min(boxH, nh * scale));
      w = Math.round(h * aspect);
    } else {
      w = Math.round(Math.min(boxW, nw * scale));
      h = Math.round(w / aspect);
    }
    return { w: Math.max(1, w), h: Math.max(1, h) };
  }

  _dimensionsAllowed(nw, nh, finalW, finalH, report) {
    const scale = Number(this.model?.scale);
    const outW = nw * scale, outH = nh * scale;
    const limit = Math.max(1, Number(this.device?.limits?.maxTextureDimension2D) || 8192);
    const values = [nw, nh, outW, outH, finalW, finalH];
    const valid = values.every((v) => Number.isSafeInteger(v) && v > 0 && v <= limit);
    let modelValid = valid;
    if (modelValid && typeof this.model?.preflight === "function") {
      try { this.model.preflight(nw, nh); }
      catch (error) {
        if (!/^MODEL_/.test(error?.code || "")) throw error;
        modelValid = false;
        if (report) this._reportFailure("limits", `Skipped ${nw}x${nh}: ${error.message}`);
      }
    }
    if (!modelValid && valid && report) return false;
    if (!valid && report) {
      this._reportFailure("limits", `Skipped dimensions ${nw}x${nh} -> ${outW}x${outH}; adapter texture limit is ${limit}px per edge.`);
    }
    return modelValid;
  }

  // --- lifecycle ------------------------------------------------------------
  start() {
    if (this.running) return;
    if (this._destroyed) { this._reportFailure("destroyed", "Cannot restart a destroyed image upscaler."); return; }
    this.running = true;
    this._runGeneration++;
    this.processed = new WeakSet();
    this._revisions = new WeakMap();
    this._jobs = new WeakMap();
    this._queue = [];
    this._deferred.clear();
    // Do not clear `_active`: a job from the stopped generation may still be
    // unwinding submitted GPU work. The new generation deliberately waits for
    // that continuation before reusing the model and SSimDownscaler caches.
    if (typeof IntersectionObserver === "function") {
      this.io = new IntersectionObserver((entries) => {
        for (const e of entries) if (e.isIntersecting) this.tryProcess(e.target);
      }, { rootMargin: "200px" });
    } else {
      this.io = null;
    }
    this._observeMutationRoot(document.body || document.documentElement);
    this.scan();
  }

  stop() {
    // Consume mutations already queued for delivery before deciding which
    // attributes still belong to us. In particular, a page can replace src and
    // srcset and immediately disable the extension in the same task.
    const pendingMutations = [];
    for (const mo of [...this._mutationObservers.values()]) {
      try {
        const records = mo.takeRecords?.();
        if (records?.length) pendingMutations.push(...records);
      } catch {}
    }
    // Process the complete snapshot before a removed shadow host can disconnect
    // one of the other observers and discard its pending attribute records.
    if (pendingMutations.length) this._handleMutations(pendingMutations);
    this.running = false;
    this._runGeneration++;
    this.io?.disconnect(); this.io = null;
    for (const mo of this._mutationObservers.values()) mo.disconnect();
    this._mutationObservers.clear(); this.mo = null;
    for (const cancel of [...this._pendingLoads]) cancel();
    this._clearQueuedJobs();
    this._deferred.clear();
    for (const [img, handler] of this._observedImages) {
      try { img.removeEventListener("load", handler); } catch {}
    }
    this._observedImages.clear();
    // restore every image we replaced
    for (const img of [...this.replaced]) this.restore(img, false);
    this.replaced.clear();
    this.processed = new WeakSet();
    this._revisions = new WeakMap();
    this._jobs = new WeakMap();
    this._setCount(0);
  }

  // Re-run eligible images after a setting/source-policy change. Passing an
  // image limits invalidation to that element; no argument resets the full run.
  invalidate(img = null) {
    if (img) {
      this._cancelImageWork(img);
      if (this.replaced.has(img)) this.restore(img);
      this.processed.delete(img);
      if (this.running) this.observe(img, true);
      return;
    }
    this._runGeneration++;
    for (const cancel of [...this._pendingLoads]) cancel();
    this._clearQueuedJobs();
    this._deferred.clear();
    for (const node of [...this.replaced]) this.restore(node, false);
    this.replaced.clear();
    this.processed = new WeakSet();
    this._revisions = new WeakMap();
    this._jobs = new WeakMap();
    this._setCount(0);
    if (this.running) {
      for (const imgNode of this._observedImages.keys()) this.observe(imgNode, true);
      this.scan();
    }
  }

  destroy() {
    if (this._destroyPromise) return this._destroyPromise;
    // Publish both terminal state and the single-flight promise before stopping
    // observers. stop() reports count=0 through an injected callback, and that
    // callback is allowed to synchronously re-enter destroy().
    this._destroyed = true;
    let resolveDestroy, rejectDestroy;
    const promise = new Promise((resolve, reject) => {
      resolveDestroy = resolve;
      rejectDestroy = reject;
    });
    this._destroyPromise = promise;
    let stopError = null;
    try { this.stop(); } catch (error) { stopError = error; }
    (async () => {
      // A job can be awaiting submitted GPU work or PNG encoding after stop().
      // Keep its model/cache generation alive until that continuation is done.
      while (this._activePromise) {
        const active = this._activePromise;
        try { await active; } catch {}
        if (this._activePromise === active) this._activePromise = null;
      }
      try { await this.device?.queue?.onSubmittedWorkDone?.(); } catch {}
      this._releaseOwnedGpuResources();
      // The device and sampler are borrowed from the main runtime, so only
      // release our references after all submitted image work has drained.
      this.device = null;
      this.sampler = null;
      if (stopError) throw stopError;
    })().then(resolveDestroy, rejectDestroy);
    return promise;
  }

  // Put the original src/srcset back.
  restore(img, adjustCount = true) {
    if (!img || !img.dataset) return;
    if (img.dataset.fsrcnnxOrig != null) {
      const url = img._fsrcnnxURL;
      this._ownWrite(img, () => {
        // Restore only exact values published by this instance. Attribute
        // writes from the page may still be waiting in MutationObserver's
        // queue (or the observer may not implement takeRecords), and must win.
        if (url && img.getAttribute("src") === url) {
          this._restoreAttribute(img, "src", "fsrcnnxOrig", "fsrcnnxOrigHadSrc");
        }
        if (img.hasAttribute("srcset") && img.getAttribute("srcset") === "") {
          this._restoreAttribute(img, "srcset", "fsrcnnxOrigSrcset", "fsrcnnxOrigHadSrcset");
        }
        this._clearReplacementMetadata(img);
      });
    }
    if (img._fsrcnnxURL) { URL.revokeObjectURL(img._fsrcnnxURL); img._fsrcnnxURL = null; }
    const removed = this.replaced.delete(img);
    if (removed && adjustCount) this._setCount(Math.max(0, this.count - 1));
  }

  scan(root = document) {
    if (!this.running || !root) return;
    this._scanRoot(root);
  }

  _scanRoot(root) {
    if (!root) return;
    if (root.shadowRoot) {
      this._observeMutationRoot(root.shadowRoot);
      this._scanRoot(root.shadowRoot);
    }
    if (root.nodeType === 1 && root.tagName === "IMG") this.observe(root);
    try { root.querySelectorAll?.("img").forEach((img) => this.observe(img)); } catch {}
    try {
      root.querySelectorAll?.("*").forEach((el) => {
        if (el.shadowRoot) {
          this._observeMutationRoot(el.shadowRoot);
          this._scanRoot(el.shadowRoot);
        }
      });
    } catch {}
  }

  observe(img, refresh = false) {
    if (!this.running || !img || img.tagName !== "IMG") return;
    if (!this._observedImages.has(img)) {
      const onLoad = () => {
        if (!this.running || this._isWriting(img)) return;
        // A responsive-image selection can change currentSrc without changing
        // an attribute on the <img> itself (for example, after a <picture>
        // media query starts matching). Only suppress the load for the blob we
        // actually own; an effective-source change retires that replacement.
        if (this._isOwnReplacement(img)) return;
        if (this.replaced.has(img) || img.dataset?.fsrcnnxDone) {
          this._discardReplacementForExternalChange(img, new Set());
        }
        this._cancelImageWork(img);
        this.processed.delete(img);
        this.observe(img, true);
      };
      this._observedImages.set(img, onLoad);
      try { img.addEventListener("load", onLoad, { passive: true }); } catch {}
    }
    if (this.io) {
      try { if (refresh) this.io.unobserve(img); this.io.observe(img); } catch {}
    } else {
      this.tryProcess(img);
    }
  }

  _observeMutationRoot(root) {
    if (!this.running || !root || this._mutationObservers.has(root) || typeof MutationObserver !== "function") return;
    const mo = new MutationObserver((records) => this._handleMutations(records));
    try {
      mo.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeOldValue: true,
        attributeFilter: ["src", "srcset", "sizes", "media", "type"],
      });
      this._mutationObservers.set(root, mo);
      if (!this.mo) this.mo = mo;
    } catch { mo.disconnect(); }
  }

  _handleMutations(records) {
    if (!this.running) return;
    const imageAttributeRecords = new Map();
    const effectiveSourceChecks = new Set();

    for (const record of records) {
      if (record.type === "attributes" && record.target?.tagName === "IMG") {
        const img = record.target;
        let pending = imageAttributeRecords.get(img);
        if (!pending) imageAttributeRecords.set(img, pending = []);
        pending.push(record);
        continue;
      }
      if (record.type === "attributes" && record.target?.tagName === "SOURCE") {
        this._collectPictureImages(record.target, effectiveSourceChecks);
      } else if (record.type === "childList") {
        // Adding, removing, or reordering <source> nodes can replace currentSrc
        // while leaving the descendant <img>'s own attributes untouched.
        this._collectPictureImages(record.target, effectiveSourceChecks);
      }
    }

    // MutationObserver exposes the final attribute value for every record. Use
    // successive oldValue entries to reconstruct each transition, then handle
    // all externally changed attributes for an image as one ownership event.
    // This prevents a src record from restoring over a srcset change (or vice
    // versa) that occurred in the same observer delivery.
    for (const [img, pending] of imageAttributeRecords) {
      const externalAttributes = new Set();
      for (let index = 0; index < pending.length; index++) {
        const record = pending[index];
        const attribute = record.attributeName;
        let nextValue = img.getAttribute?.(attribute) ?? null;
        for (let next = index + 1; next < pending.length; next++) {
          if (pending[next].attributeName === attribute && "oldValue" in pending[next]) {
            nextValue = pending[next].oldValue;
            break;
          }
        }
        const oldValue = "oldValue" in record ? record.oldValue : undefined;
        if (!this._isOwnMutation(img, attribute, oldValue, nextValue)) externalAttributes.add(attribute);
      }
      if (!externalAttributes.size) continue;
      this._discardReplacementForExternalChange(img, externalAttributes);
      this._cancelImageWork(img);
      this.processed.delete(img);
      this.observe(img, true);
    }

    for (const img of effectiveSourceChecks) {
      const hasReplacement = this.replaced.has(img) || !!img.dataset?.fsrcnnxDone;
      if (hasReplacement && this._isOwnReplacement(img)) continue;
      if (hasReplacement) this._discardReplacementForExternalChange(img, new Set());
      this._cancelImageWork(img);
      this.processed.delete(img);
      this.observe(img, true);
    }

    // Tree ownership changes run after attribute ownership changes so removing
    // an image cannot restore over page-authored attributes from the same batch.
    for (const record of records) {
      if (record.type !== "childList") continue;
      for (const node of record.removedNodes || []) this._unobserveTree(node);
      for (const node of record.addedNodes || []) this._scanRoot(node);
    }
  }

  _collectPictureImages(node, output) {
    let picture = null;
    if (node?.nodeType === 1 && node.tagName === "PICTURE") picture = node;
    if (!picture) {
      try { picture = node?.closest?.("picture") || null; } catch {}
    }
    if (!picture) {
      for (let parent = node?.parentElement; parent; parent = parent.parentElement) {
        if (parent.tagName === "PICTURE") { picture = parent; break; }
      }
    }
    if (!picture) return;
    try { picture.querySelectorAll?.("img").forEach((img) => output.add(img)); } catch {}
  }

  _unobserveTree(root) {
    this._disconnectMutationRoot(root);
    const images = [];
    if (root?.nodeType === 1 && root.tagName === "IMG") images.push(root);
    try { root?.querySelectorAll?.("img").forEach((img) => images.push(img)); } catch {}
    if (root?.shadowRoot) this._unobserveTree(root.shadowRoot);
    try { root?.querySelectorAll?.("*").forEach((el) => { if (el.shadowRoot) this._unobserveTree(el.shadowRoot); }); } catch {}
    for (const img of images) {
      this._cancelImageWork(img);
      this._deferred.delete(img);
      try { this.io?.unobserve(img); } catch {}
      const handler = this._observedImages.get(img);
      if (handler) { try { img.removeEventListener("load", handler); } catch {} this._observedImages.delete(img); }
      if (this.replaced.has(img)) this.restore(img);
      this.processed.delete(img);
    }
  }

  _disconnectMutationRoot(root) {
    const mo = this._mutationObservers.get(root);
    if (!mo) return;
    // A document-root observer can report removal of a shadow host before the
    // shadow observer receives attribute records queued in the same task.
    // Consume them before disconnect(), which otherwise discards the records and
    // lets restore() overwrite the page's final same-value src/srcset writes.
    try {
      const records = mo.takeRecords?.();
      if (records?.length) this._handleMutations(records);
    } catch {}
    try { mo.disconnect(); } catch {}
    this._mutationObservers.delete(root);
    if (this.mo === mo) this.mo = this._mutationObservers.values().next().value || null;
  }

  // --- processing -----------------------------------------------------------
  tryProcess(img) {
    if (!this.running || !img || this._isWriting(img) || this._isOwnReplacement(img)) return Promise.resolve(false);
    const revision = this._revision(img);
    const existing = this._jobs.get(img);
    if (existing && existing.generation === this._runGeneration && existing.revision === revision) return existing.promise;
    if (this.processed.has(img) || !this.isCandidate(img)) return Promise.resolve(false);
    if (this._queue.length >= MAX_PENDING) {
      if (this._deferred.size < MAX_DEFERRED) this._deferred.add(img);
      this._reportFailure("queue-full", `Image queue reached ${MAX_PENDING}; additional visible images are deferred.`);
      return Promise.resolve(false);
    }
    this.processed.add(img); // mark before enqueue so IO/scan callbacks cannot duplicate it
    try { this.io?.unobserve(img); } catch {}
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    const job = { img, source: this._source(img), generation: this._runGeneration, revision, promise, resolve, settled: false };
    this._jobs.set(img, job);
    this._queue.push(job);
    this._drainQueue();
    return promise;
  }

  // The video path refuses wide-gamut sources outright; the image path had no
  // colour gate at all. There is no web API that reports a decoded image's colour
  // space, so source gamut cannot be inspected — but the round trip is only
  // *lossy* on a wide-gamut display, because an sRGB display would clip the same
  // pixels anyway. Gating on the display therefore covers exactly the case where
  // processing makes the result worse, and fails closed like the video path.
  // Cached because matchMedia is a layout-adjacent read on a hot path.
  _displayGamutBlocked() {
    if (this._wideGamutDisplay === undefined) {
      let wide = false;
      try { wide = globalThis.matchMedia?.("(color-gamut: p3)")?.matches === true; }
      catch { wide = false; }
      this._wideGamutDisplay = wide;
    }
    return this._wideGamutDisplay;
  }

  async _processJob(job) {
    let bitmap = null;
    try {
      if (this._displayGamutBlocked()) {
        if (this._jobCurrent(job)) {
          this._reportFailure(
            "color-wide-gamut-display",
            "Image upscaling is skipped on a wide-gamut display because the result would be clipped to sRGB.",
          );
        }
        return false;
      }
      bitmap = await this.loadReadable(job.img, job);
      if (!bitmap) {
        if (this._jobCurrent(job)) this._reportFailure("unreadable", "Image could not be read; cross-origin sources must provide CORS access.");
        return false;
      }
      if (!this._jobCurrent(job)) return false;
      const replaced = await this.upscaleAndReplace(job.img, bitmap, job);
      if (replaced && this._jobRunCurrent(job)) this._setCount(this.count + 1);
      return !!(replaced && this._jobRunCurrent(job));
    } catch (e) {
      if (this._jobCurrent(job)) {
        const code = e instanceof ImageSkipError ? e.code : "processing";
        this._reportFailure(code, e?.message || "Image processing failed.");
      }
      return false;
    } finally {
      try { bitmap?.close?.(); } catch {}
    }
  }

  _drainQueue() {
    if (!this.running || this._active) return;
    let job;
    while ((job = this._queue.shift())) {
      if (this._jobCurrent(job)) break;
      this._finishJob(job, false);
      job = null;
    }
    if (!job) { this._drainDeferred(); return; }
    this._active = job;
    const activePromise = this._processJob(job).then((ok) => this._finishJob(job, ok)).finally(() => {
      if (this._active === job) this._active = null;
      if (this._activePromise === activePromise) this._activePromise = null;
      this._drainDeferred();
      this._drainQueue();
    });
    this._activePromise = activePromise;
  }

  _finishJob(job, result) {
    if (job.settled) return;
    job.settled = true;
    if (this._jobs.get(job.img) === job) this._jobs.delete(job.img);
    job.resolve(!!result);
  }

  _drainDeferred() {
    if (!this.running || this._queue.length >= MAX_PENDING || !this._deferred.size) return;
    const [img] = this._deferred;
    this._deferred.delete(img);
    this.tryProcess(img);
  }

  _clearQueuedJobs() {
    for (const job of this._queue.splice(0)) this._finishJob(job, false);
    this._jobs = new WeakMap();
  }

  _jobCurrent(job) {
    return this._jobRunCurrent(job) && this._source(job.img) === job.source;
  }

  _jobRunCurrent(job) {
    return !!job && this.running && job.generation === this._runGeneration &&
      job.revision === this._revision(job.img) &&
      (!("isConnected" in job.img) || job.img.isConnected);
  }

  _revision(img) { return this._revisions.get(img) || 0; }

  _cancelImageWork(img) {
    this._revisions.set(img, this._revision(img) + 1);
    const cancel = this._loadCancelByImage.get(img);
    if (cancel) cancel();
    this._queue = this._queue.filter((job) => {
      if (job.img !== img) return true;
      this._finishJob(job, false);
      return false;
    });
    const current = this._jobs.get(img);
    if (current && current !== this._active) this._jobs.delete(img);
  }

  // Obtain a non-tainted ImageBitmap for the image. Same-origin loads directly;
  // cross-origin requires CORS — we re-request with crossOrigin=anonymous, which
  // succeeds only if the server sends Access-Control-Allow-Origin.
  loadReadable(img, job = null) {
    return new Promise((resolve) => {
      const src = job?.source || this._source(img);
      if (!src) { resolve(null); return; }
      const probe = new Image();
      let settled = false, timer = null;
      const finish = (bitmap) => {
        if (settled) { try { bitmap?.close?.(); } catch {} return; }
        settled = true;
        if (timer != null) clearTimeout(timer);
        probe.onload = probe.onerror = null;
        this._pendingLoads.delete(cancel);
        if (this._loadCancelByImage.get(img) === cancel) this._loadCancelByImage.delete(img);
        resolve(bitmap);
      };
      const cancel = () => {
        if (settled) return;
        probe.onload = probe.onerror = null;
        try { probe.src = ""; } catch {}
        finish(null);
      };
      this._pendingLoads.add(cancel);
      this._loadCancelByImage.set(img, cancel);
      probe.crossOrigin = "anonymous";
      probe.onload = async () => {
        try { finish(await createImageBitmap(probe)); }
        catch { finish(null); }
      };
      probe.onerror = () => finish(null); // CORS denied or load error -> skip
      timer = setTimeout(cancel, IMAGE_LOAD_TIMEOUT_MS);
      probe.src = src;
    });
  }

  async upscaleAndReplace(img, bitmap, job = null) {
    const device = this.device;
    const nw = bitmap.width, nh = bitmap.height;
    const target = this._targetDimensions(img, nw, nh);
    if (!target || !this._dimensionsAllowed(nw, nh, target.w, target.h, false)) {
      throw new ImageSkipError("limits", "Image dimensions changed outside safe GPU limits before processing.");
    }
    if (job && !this._jobCurrent(job)) return false;
    const dispW = target.w, dispH = target.h;
    const outW = nw * this.model.scale, outH = nh * this.model.scale;
    let srcTex = null, lumaTex = null, hiRGB = null, ctx = null, url = null, adoptedURL = false;
    try {
      srcTex = device.createTexture({
        size: { width: nw, height: nh }, format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture: srcTex, colorSpace: SRGB_COLOR_SPACE, premultipliedAlpha: false },
        { width: nw, height: nh },
      );
      lumaTex = device.createTexture({
        size: { width: nw, height: nh }, format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      const enc = device.createCommandEncoder();
      {
        const bg = device.createBindGroup({ layout: this.extractPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: this.sampler }, { binding: 1, resource: srcTex.createView() },
          { binding: 2, resource: lumaTex.createView() },
        ] });
        const cp = enc.beginComputePass();
        cp.setPipeline(this.extractPipe); cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(Math.ceil(nw / 8), Math.ceil(nh / 8)); cp.end();
      }
      this.model.allocate(nw, nh, lumaTex);
      const hiLuma = this.model.run(enc, lumaTex);
      hiRGB = device.createTexture({
        size: { width: outW, height: outH }, format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      {
        const bg = device.createBindGroup({ layout: this.recombinePipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: this.sampler }, { binding: 1, resource: srcTex.createView() },
          { binding: 2, resource: hiLuma.createView() },
        ] });
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: hiRGB.createView(), loadOp: "clear", clearValue: {r:0,g:0,b:0,a:1}, storeOp: "store" }] });
        rp.setPipeline(this.recombinePipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
      }
      let finalTex = hiRGB, finalW = outW, finalH = outH;
      if (outW > dispW * 1.05) {
        this.ssimds.prepare(outW, outH, dispW, dispH, hiRGB);
        finalTex = this.ssimds.run(enc, hiRGB); finalW = dispW; finalH = dispH;
      }
      const off = new OffscreenCanvas(finalW, finalH);
      ctx = off.getContext("webgpu");
      if (!ctx) throw new Error("OffscreenCanvas WebGPU context unavailable");
      ctx.configure({
        device,
        format: this.format,
        colorSpace: SRGB_COLOR_SPACE,
        alphaMode: "premultiplied",
      });
      {
        const bg = device.createBindGroup({ layout: this.blitPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: this.sampler }, { binding: 1, resource: finalTex.createView() },
          { binding: 2, resource: srcTex.createView() },
        ] });
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", clearValue: {r:0,g:0,b:0,a:0}, storeOp: "store" }] });
        rp.setPipeline(this.blitPipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
      }
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (job && !this._jobCurrent(job)) return false;
      const blob = await off.convertToBlob({ type: "image/png" });
      if (job && !this._jobCurrent(job)) return false;
      url = URL.createObjectURL(blob);
      if (job && !this._jobCurrent(job)) return false;
      this._captureOriginal(img);
      if (img._fsrcnnxURL) URL.revokeObjectURL(img._fsrcnnxURL);
      img._fsrcnnxURL = url;
      this._ownWrite(img, () => {
        img.setAttribute("srcset", "");
        img.setAttribute("src", url);
        img.dataset.fsrcnnxDone = "1";
      });
      this.replaced.add(img);
      adoptedURL = true;
      return true;
    } finally {
      try { ctx?.unconfigure?.(); } catch {}
      for (const tex of [srcTex, lumaTex, hiRGB]) { try { tex?.destroy?.(); } catch {} }
      if (url && !adoptedURL) { try { URL.revokeObjectURL(url); } catch {} }
    }
  }

  _source(img) { return img.currentSrc || img.src || ""; }
  _isWriting(img) { return (this._writing.get(img) || 0) > 0; }
  _isOwnReplacement(img) {
    const url = img?._fsrcnnxURL;
    return !!(url && img.dataset?.fsrcnnxDone && this._source(img) === url);
  }

  _isOwnMutation(img, attribute, oldValue = undefined, newValue = undefined) {
    if (this._isWriting(img)) return true;
    const byAttribute = this._ownedMutations.get(img);
    const transitions = byAttribute?.get(attribute);
    if (transitions?.length) {
      const match = transitions.findIndex((transition) =>
        (oldValue === undefined || transition.oldValue === oldValue) &&
        (newValue === undefined || transition.newValue === newValue));
      if (match >= 0) {
        transitions.splice(match, 1);
        if (!transitions.length) byAttribute.delete(attribute);
        if (!byAttribute.size) this._ownedMutations.delete(img);
        return true;
      }
    }
    // With attributeOldValue enabled, an unmatched transition is page-owned,
    // even when the page deliberately wrote the same value we already had.
    if (oldValue !== undefined && newValue !== undefined) return false;
    const url = img?._fsrcnnxURL;
    if (!url || !img.dataset?.fsrcnnxDone) return false;
    if (attribute === "src") return img.getAttribute("src") === url;
    if (attribute === "srcset") return img.getAttribute("src") === url && (img.getAttribute("srcset") || "") === "";
    return false;
  }

  _ownWrite(img, fn) {
    const attributes = ["src", "srcset"];
    const before = new Map(attributes.map((attribute) => [attribute, img.getAttribute?.(attribute) ?? null]));
    this._writing.set(img, (this._writing.get(img) || 0) + 1);
    try { fn(); }
    finally {
      for (const attribute of attributes) {
        const oldValue = before.get(attribute);
        const newValue = img.getAttribute?.(attribute) ?? null;
        if (oldValue !== newValue) this._recordOwnMutation(img, attribute, oldValue, newValue);
      }
      const remaining = (this._writing.get(img) || 1) - 1;
      if (remaining > 0) this._writing.set(img, remaining); else this._writing.delete(img);
    }
  }

  _recordOwnMutation(img, attribute, oldValue, newValue) {
    let byAttribute = this._ownedMutations.get(img);
    if (!byAttribute) this._ownedMutations.set(img, byAttribute = new Map());
    let transitions = byAttribute.get(attribute);
    if (!transitions) byAttribute.set(attribute, transitions = []);
    const transition = { oldValue, newValue };
    transitions.push(transition);
    // MutationObserver callbacks run before timers. Drop an undelivered token
    // after the current task so it cannot misclassify a later page mutation.
    setTimeout(() => {
      const pending = this._ownedMutations.get(img)?.get(attribute);
      const index = pending?.indexOf(transition) ?? -1;
      if (index >= 0) pending.splice(index, 1);
      const current = this._ownedMutations.get(img);
      if (pending && !pending.length) current?.delete(attribute);
      if (current && !current.size) this._ownedMutations.delete(img);
    }, 0);
  }

  _captureOriginal(img) {
    if (img.dataset.fsrcnnxOrig != null) return;
    const hasSrc = img.hasAttribute("src"), hasSrcset = img.hasAttribute("srcset");
    img.dataset.fsrcnnxOrig = hasSrc ? (img.getAttribute("src") || "") : "";
    img.dataset.fsrcnnxOrigHadSrc = hasSrc ? "1" : "0";
    img.dataset.fsrcnnxOrigSrcset = hasSrcset ? (img.getAttribute("srcset") || "") : "";
    img.dataset.fsrcnnxOrigHadSrcset = hasSrcset ? "1" : "0";
  }

  _restoreAttribute(img, attr, valueKey, hadKey) {
    const had = img.dataset[hadKey];
    if (had === "0") img.removeAttribute(attr);
    else if (img.dataset[valueKey] != null) img.setAttribute(attr, img.dataset[valueKey]);
    else if (attr === "srcset") img.removeAttribute(attr); // compatibility with pre-lifecycle metadata
  }

  _clearReplacementMetadata(img) {
    for (const key of ["fsrcnnxOrig", "fsrcnnxOrigHadSrc", "fsrcnnxOrigSrcset", "fsrcnnxOrigHadSrcset", "fsrcnnxDone"]) delete img.dataset[key];
  }

  _discardReplacementForExternalChange(img, changedAttributes) {
    if (!this.replaced.has(img) && !img.dataset?.fsrcnnxDone) return;
    const changed = typeof changedAttributes === "string"
      ? new Set([changedAttributes])
      : new Set(changedAttributes || []);
    const url = img._fsrcnnxURL;
    this._ownWrite(img, () => {
      // Preserve every attribute the page changed in this delivery. Restore an
      // untouched attribute only while it still has our exact owned value.
      if (!changed.has("src") && img.getAttribute("src") === url) {
        this._restoreAttribute(img, "src", "fsrcnnxOrig", "fsrcnnxOrigHadSrc");
      }
      if (!changed.has("srcset") && img.hasAttribute("srcset") && img.getAttribute("srcset") === "") {
        this._restoreAttribute(img, "srcset", "fsrcnnxOrigSrcset", "fsrcnnxOrigHadSrcset");
      }
      this._clearReplacementMetadata(img);
    });
    if (url) { try { URL.revokeObjectURL(url); } catch {} img._fsrcnnxURL = null; }
    if (this.replaced.delete(img)) this._setCount(Math.max(0, this.count - 1));
  }

  _releaseOwnedGpuResources() {
    // GPU pipelines do not currently expose destroy(), but honoring it keeps
    // cleanup correct for compatible implementations and focused test doubles.
    for (const resource of [this.blitPipe, this.recombinePipe, this.extractPipe, this.ssimds, this.model]) {
      try { resource?.destroy?.(); } catch {}
    }
    this.blitPipe = null;
    this.recombinePipe = null;
    this.extractPipe = null;
    this.ssimds = null;
    this.model = null;
  }

  _setCount(value) {
    this.count = value;
    try { this.onCount(value); } catch (e) { this._reportFailure("callback", e?.message || "Image count callback failed."); }
  }

  _reportFailure(code, message) {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const state = this._failures.get(code) || { total: 0, lastLog: -Infinity, suppressed: 0 };
    state.total++;
    try { this.onError({ code, message, count: state.total }); } catch {}
    if (now - state.lastLog >= FAILURE_LOG_INTERVAL_MS) {
      const suffix = state.suppressed ? ` (${state.suppressed} similar events suppressed)` : "";
      try { this.warn(`${message}${suffix}`); } catch {}
      state.lastLog = now; state.suppressed = 0;
    } else {
      state.suppressed++;
    }
    this._failures.set(code, state);
  }

  getStats() {
    return {
      running: this.running, processed: this.count,
      queued: this._queue.length + (this._active ? 1 : 0), deferred: this._deferred.size,
      failures: Object.fromEntries([...this._failures].map(([code, state]) => [code, state.total])),
    };
  }
}

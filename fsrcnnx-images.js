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
// per-frame loop) — the result is drawn to an overlay canvas covering the img.

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
  // scale chroma by luma ratio (preserve hue/sat, swap luminance)
  let ratio = select(yNew / yOld, 1.0, yOld < 0.0001);
  let outc = clamp(rgb * ratio, vec3f(0.0), vec3f(1.0));
  return vec4f(outc, 1.0);
}`;

const BLIT = `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
struct VsOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VsOut{
  var p=array<vec2f,3>(vec2f(-1.,-3.),vec2f(-1.,1.),vec2f(3.,1.));
  var u=array<vec2f,3>(vec2f(0.,2.),vec2f(0.,0.),vec2f(2.,0.));
  var o:VsOut; o.pos=vec4f(p[i],0.,1.); o.uv=u[i]; return o;
}
@fragment fn fs(@location(0) uv:vec2f)->@location(0) vec4f {
  return textureSampleLevel(src, samp, uv, 0.0);
}`;

export class ImageUpscaler {
  constructor({ device, format, sampler, fsrcnnxSource, FsrcnnxModel, SsimDownscaler, onCount }) {
    this.device = device;
    this.format = format;
    this.sampler = sampler;
    this.onCount = onCount || (() => {});
    this.count = 0;
    this.running = false;
    this.processed = new WeakSet(); // imgs we've handled (success or skip)
    this.replaced = new Set();      // imgs whose src we replaced (for restore)
    this._writing = new WeakSet();  // imgs we're writing src to (ignore our own mutations)

    // dedicated FSRCNNX model instance (own texture cache)
    this.model = fsrcnnxSource ? new FsrcnnxModel(device, fsrcnnxSource.manifest, fsrcnnxSource.wgsl) : null;
    this.ssimds = new SsimDownscaler(device);

    // pipelines
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
  }

  // --- smart filter ---------------------------------------------------------
  isCandidate(img) {
    if (this.processed.has(img)) return false;
    if (img.dataset && img.dataset.fsrcnnxDone) return false; // already replaced by us
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh) return false;                 // not loaded / not raster
    const src = img.currentSrc || img.src || "";
    if (src.startsWith("data:")) return false;     // inline data URI
    if (src.startsWith("blob:")) return false;      // our own (or page's) blob
    if (/\.svg(\?|$)/i.test(src)) return false;     // vector, no benefit
    // Process images whose native resolution is below the display (screen)
    // resolution — i.e. there's headroom to add real detail up toward what the
    // monitor can show. screen.width/height are in CSS px; multiply by DPR for
    // the actual device-pixel resolution of the panel.
    const dpr = window.devicePixelRatio || 1;
    const screenW = (window.screen?.width || 1920) * dpr;
    const screenH = (window.screen?.height || 1080) * dpr;
    if (nw >= screenW || nh >= screenH) return false; // already >= display res
    // NOTE: small-image exclusions (icons/avatars, tiny natural sizes) are
    // intentionally NOT applied here yet — we're testing whether processing
    // them is worthwhile. The only upper bound is the screen-resolution check
    // above. Re-add size floors here if tiny images prove not worth it.
    return true;
  }

  // --- lifecycle ------------------------------------------------------------
  start() {
    if (this.running) return;
    this.running = true;
    this.io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) this.tryProcess(e.target);
      }
    }, { rootMargin: "200px" }); // start a bit before they enter view
    // observe existing imgs
    this.scan();
    // watch for new imgs (feeds, lazy DOM). Ignore our own src writes.
    this.mo = new MutationObserver((muts) => {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n.nodeType !== 1) continue;
          if (n.tagName === "IMG") this.observe(n);
          else n.querySelectorAll && n.querySelectorAll("img").forEach((im) => this.observe(im));
        }
      }
    });
    this.mo.observe(document.body, { childList: true, subtree: true });
  }

  stop() {
    this.running = false;
    this.io?.disconnect(); this.mo?.disconnect();
    // restore every image we replaced
    for (const img of this.replaced) this.restore(img);
    this.replaced.clear();
    this.count = 0; this.onCount(0);
  }

  // Put the original src/srcset back.
  restore(img) {
    if (!img || !img.dataset) return;
    if (img.dataset.fsrcnnxOrig != null) {
      this._writing.add(img);
      img.src = img.dataset.fsrcnnxOrig;
      if (img.dataset.fsrcnnxOrigSrcset) img.srcset = img.dataset.fsrcnnxOrigSrcset;
      delete img.dataset.fsrcnnxOrig;
      delete img.dataset.fsrcnnxOrigSrcset;
      delete img.dataset.fsrcnnxDone;
      queueMicrotask(() => this._writing.delete(img));
    }
    if (img._fsrcnnxURL) { URL.revokeObjectURL(img._fsrcnnxURL); img._fsrcnnxURL = null; }
  }

  scan() { document.querySelectorAll("img").forEach((im) => this.observe(im)); }
  observe(img) { try { this.io.observe(img); } catch {} }

  // --- processing -----------------------------------------------------------
  async tryProcess(img) {
    if (!this.running || this.processed.has(img)) return;
    if (!this.isCandidate(img)) return;
    this.processed.add(img); // mark up front so we don't double-process
    try {
      const bitmap = await this.loadReadable(img);
      if (!bitmap) return; // cross-origin without CORS, or load failed -> skip
      await this.upscaleAndReplace(img, bitmap);
      bitmap.close && bitmap.close();
      this.count++; this.onCount(this.count);
    } catch (e) {
      // leave the original image untouched on any failure
    }
  }

  // Obtain a non-tainted ImageBitmap for the image. Same-origin loads directly;
  // cross-origin requires CORS — we re-request with crossOrigin=anonymous, which
  // succeeds only if the server sends Access-Control-Allow-Origin.
  loadReadable(img) {
    return new Promise((resolve) => {
      const src = img.currentSrc || img.src;
      const probe = new Image();
      probe.crossOrigin = "anonymous";
      probe.onload = async () => {
        try { resolve(await createImageBitmap(probe)); }
        catch { resolve(null); }
      };
      probe.onerror = () => resolve(null); // CORS denied or load error -> skip
      probe.src = src;
    });
  }

  async upscaleAndReplace(img, bitmap) {
    const device = this.device;
    const nw = bitmap.width, nh = bitmap.height;
    const rect = img.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    // The replacement blob must keep the image's NATIVE aspect ratio, otherwise
    // object-fit (contain/cover/letterboxing) renders differently than the
    // original. So size the output to the native aspect ratio, scaled so its
    // larger dimension matches what the box would show at most (box size * DPR).
    // This way object-fit: contain still letterboxes our blob identically.
    const boxW = Math.max(1, rect.width * dpr);
    const boxH = Math.max(1, rect.height * dpr);
    const aspect = nw / nh;
    // fit native-aspect box inside the displayed box (contain), capped so we
    // never store more than ~native*scale pixels
    let dispW, dispH;
    if (boxW / boxH > aspect) {
      // box is wider than the image -> image height is the limiting dim
      dispH = Math.round(Math.min(boxH, nh * this.model.scale));
      dispW = Math.round(dispH * aspect);
    } else {
      dispW = Math.round(Math.min(boxW, nw * this.model.scale));
      dispH = Math.round(dispW / aspect);
    }
    dispW = Math.max(1, dispW); dispH = Math.max(1, dispH);

    // source RGB texture from the image
    const srcTex = device.createTexture({
      size: { width: nw, height: nh }, format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: srcTex }, { width: nw, height: nh });

    // luma texture (rgba16float) at native size
    const lumaTex = device.createTexture({
      size: { width: nw, height: nh }, format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    const enc = device.createCommandEncoder();
    // 1. extract luma
    {
      const bg = device.createBindGroup({
        layout: this.extractPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: srcTex.createView() },
          { binding: 2, resource: lumaTex.createView() },
        ],
      });
      const cp = enc.beginComputePass();
      cp.setPipeline(this.extractPipe); cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(Math.ceil(nw / 8), Math.ceil(nh / 8));
      cp.end();
    }
    // 2. FSRCNNX -> hi-res luma
    this.model.allocate(nw, nh, lumaTex);
    const hiLuma = this.model.run(enc, lumaTex);
    const outW = nw * this.model.scale, outH = nh * this.model.scale;

    // 3. recombine hi luma + bilinear chroma -> hi-res RGB (rgba16float)
    const hiRGB = device.createTexture({
      size: { width: outW, height: outH }, format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    {
      const bg = device.createBindGroup({
        layout: this.recombinePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: srcTex.createView() },
          { binding: 2, resource: hiLuma.createView() },
        ],
      });
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: hiRGB.createView(), loadOp: "clear", clearValue: {r:0,g:0,b:0,a:1}, storeOp: "store" }] });
      rp.setPipeline(this.recombinePipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
    }

    // 4. SSimDownscaler from hi-res down to the target display size (sharp downscale)
    let finalTex = hiRGB, finalW = outW, finalH = outH;
    if (outW > dispW * 1.05) {
      this.ssimds.prepare(outW, outH, dispW, dispH, hiRGB);
      finalTex = this.ssimds.run(enc, hiRGB);
      finalW = dispW; finalH = dispH;
    }

    // 5. render to an offscreen canvas and read back to a blob
    const off = new OffscreenCanvas(finalW, finalH);
    const ctx = off.getContext("webgpu");
    ctx.configure({ device, format: this.format, alphaMode: "premultiplied" });
    {
      const bg = device.createBindGroup({
        layout: this.blitPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: finalTex.createView() }],
      });
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", clearValue: {r:0,g:0,b:0,a:0}, storeOp: "store" }] });
      rp.setPipeline(this.blitPipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
    }
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    // 6. blob -> object URL, replace the <img>'s pixels IN PLACE. The element is
    //    unchanged, so it keeps the page's layout, object-fit, hover/lightbox
    //    behavior, stacking and clipping — no overlay to track or escape.
    const blob = await off.convertToBlob({ type: "image/png" });
    const url = URL.createObjectURL(blob);

    // save originals for restore; clear srcset so it can't override our src on resize
    if (img.dataset.fsrcnnxOrig == null) {
      img.dataset.fsrcnnxOrig = img.getAttribute("src") || (img.currentSrc || "");
      if (img.srcset) { img.dataset.fsrcnnxOrigSrcset = img.srcset; }
    }
    if (img._fsrcnnxURL) URL.revokeObjectURL(img._fsrcnnxURL);
    img._fsrcnnxURL = url;

    this._writing.add(img);
    img.srcset = "";          // prevent responsive selection from replacing our src
    img.src = url;
    img.dataset.fsrcnnxDone = "1";
    queueMicrotask(() => this._writing.delete(img));

    this.replaced.add(img);

    // cleanup transient textures (model/ssimds keep their own caches)
    srcTex.destroy(); lumaTex.destroy(); hiRGB.destroy();
  }
}

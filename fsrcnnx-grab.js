// fsrcnnx-grab.js — clean WebGPU frame grabber for interpolation.
//
// Why this exists: grabbing video pixels through a 2D canvas (either
// captureStream→VideoFrame→createImageBitmap OR drawImage(videoElement)) does a
// YUV→RGB / chroma reconstruction that produces "wave" ringing on bright,
// high-detail regions. The upscaler never sees this because it reads the video
// via importExternalTexture (GPU-side conversion). This module does the same:
// samples the <video> through importExternalTexture into an rgba8 texture with a
// passthrough shader, then reads that texture back to CPU as clean RGBA bytes —
// which the existing RIFE + scheduler pipeline consumes unchanged.
//
// STAGE 1 of the WebGPU rework: only the ACQUISITION changes (to isolate whether
// the artifact is in acquisition). RIFE and presentation stay as they are. If the
// waves clear, acquisition was the cause. Uses its own device so interpolation
// doesn't depend on the upscaler being active.

const PASSTHROUGH_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var frame: texture_external;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  // fullscreen triangle
  var p = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
  var uv = array<vec2<f32>, 3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
  var o: VSOut;
  o.pos = vec4(p[i], 0.0, 1.0);
  o.uv = uv[i];
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // sampling an external (video) texture does GPU-side YUV->RGB, clean chroma
  return textureSampleBaseClampToEdge(frame, samp, in.uv);
}
`;

export class WebGPUGrabber {
  constructor({ log, warn } = {}) {
    this.log = log || console.log;
    this.warn = warn || console.warn;
    this.device = null;
    this.ready = false;
    this._w = 0; this._h = 0;
  }

  async init() {
    if (this.ready) return true;
    try {
      if (!("gpu" in navigator)) { this.warn("grab: no WebGPU"); return false; }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) { this.warn("grab: no adapter"); return false; }
      this.device = await adapter.requestDevice();
      this.device.lost.then((i) => { this.warn("grab device lost:", i.message); this.ready = false; });
      this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear" });
      this.pipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module: this.device.createShaderModule({ code: PASSTHROUGH_WGSL }), entryPoint: "vs" },
        fragment: {
          module: this.device.createShaderModule({ code: PASSTHROUGH_WGSL }),
          entryPoint: "fs",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.ready = true;
      return true;
    } catch (e) {
      this.warn("grab init failed:", e.message);
      this.ready = false;
      return false;
    }
  }

  _alloc(w, h) {
    if (this._w === w && this._h === h && this.tex) return;
    this._w = w; this._h = h;
    this.tex = this.device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.texView = this.tex.createView();
    // readback buffer: rows must be 256-byte aligned
    this.bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    this.readBuf = this.device.createBuffer({
      size: this.bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // Grab the current video frame as clean RGBA bytes. Returns an ImageData-like
  // { data: Uint8ClampedArray, width, height } or null on failure.
  async grab(video) {
    if (!this.ready) return null;
    // Reentrancy guard: the pipelined grab loop can tick again while a previous
    // mapAsync is pending; a second map on the same staging buffer throws. Skip
    // the frame instead (the caller treats null as "no frame this tick").
    if (this._mapPending) return null;
    this._mapPending = true;
    try { return await this._grabInner(video); } finally { this._mapPending = false; }
  }
  async _grabInner(video) {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return null;
    try {
      this._alloc(w, h);
      let ext;
      try { ext = this.device.importExternalTexture({ source: video }); }
      catch (e) { this.warn("grab importExternalTexture failed:", e.message); return null; }

      const bind = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: ext },
        ],
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.texView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
      enc.copyTextureToBuffer(
        { texture: this.tex },
        { buffer: this.readBuf, bytesPerRow: this.bytesPerRow, rowsPerImage: h },
        { width: w, height: h }
      );
      this.device.queue.submit([enc.finish()]);

      await this.readBuf.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(this.readBuf.getMappedRange());
      // de-pad rows (bytesPerRow may exceed w*4)
      const out = new Uint8ClampedArray(w * h * 4);
      const rowBytes = w * 4;
      for (let y = 0; y < h; y++) {
        out.set(mapped.subarray(y * this.bytesPerRow, y * this.bytesPerRow + rowBytes), y * rowBytes);
      }
      this.readBuf.unmap();
      return { data: out, width: w, height: h };
    } catch (e) {
      this.warn("grab failed:", e.message);
      return null;
    }
  }

  destroy() {
    try { this.tex && this.tex.destroy(); } catch {}
    try { this.readBuf && this.readBuf.destroy(); } catch {}
    this.ready = false;
  }
}

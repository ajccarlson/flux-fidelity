// fsrcnnx-grab.js — clean WebGPU frame grabber for interpolation.

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";

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

export class GrabResourceLimitError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = "GrabResourceLimitError";
    this.code = "GPU_RESOURCE_LIMIT";
    this.details = details;
  }
}

function positiveDeviceLimit(device, name, fallback) {
  const value = Number(device?.limits?.[name]);
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

export class WebGPUGrabber {
  constructor({ log, warn, onDeviceLost } = {}) {
    this.log = log || console.log;
    this.warn = warn || console.warn;
    this.onDeviceLost = typeof onDeviceLost === "function" ? onDeviceLost : null;
    this.device = null;
    this.ready = false;
    this._w = 0; this._h = 0;
    this._ownsDevice = false;
    this._deviceLost = false;
    this._mapPending = false;
    this._grabPromise = null;
    this._initPromise = null;
    this._destroyRequested = false;
    this._destroyPromise = null;
  }

  init() {
    if (this.ready) return Promise.resolve(true);
    if (this._destroyRequested) return Promise.resolve(false);
    if (this._initPromise) return this._initPromise;
    const promise = this._initInternal().finally(() => {
      if (this._initPromise === promise) this._initPromise = null;
    });
    this._initPromise = promise;
    return promise;
  }

  async _initInternal() {
    let candidateDevice = null;
    try {
      if (!("gpu" in navigator)) { this.warn("grab: no WebGPU"); return false; }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) { this.warn("grab: no adapter"); return false; }
      if (this._destroyRequested) return false;
      candidateDevice = await adapter.requestDevice();
      if (this._destroyRequested) {
        try { candidateDevice.destroy?.(); } catch {}
        candidateDevice = null;
        return false;
      }
      const sampler = candidateDevice.createSampler({ magFilter: "linear", minFilter: "linear" });
      const pipeline = candidateDevice.createRenderPipeline({
        layout: "auto",
        vertex: { module: candidateDevice.createShaderModule({ code: PASSTHROUGH_WGSL }), entryPoint: "vs" },
        fragment: {
          module: candidateDevice.createShaderModule({ code: PASSTHROUGH_WGSL }),
          entryPoint: "fs",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-list" },
      });
      if (this._destroyRequested) {
        try { candidateDevice.destroy?.(); } catch {}
        candidateDevice = null;
        return false;
      }
      this.device = candidateDevice;
      this._ownsDevice = true;
      this._deviceLost = false;
      this.sampler = sampler;
      this.pipeline = pipeline;
      const watchedDevice = candidateDevice;
      candidateDevice = null;
      if (watchedDevice?.lost?.then) {
        watchedDevice.lost.then(
          (info) => this._handleDeviceLost(watchedDevice, info),
          (error) => this._handleDeviceLost(watchedDevice, { reason: "unknown", message: error?.message || String(error) }),
        );
      }
      this.ready = true;
      return true;
    } catch (e) {
      try { candidateDevice?.destroy?.(); } catch {}
      if (!this._destroyRequested) this.warn("grab init failed:", e.message);
      this.ready = false;
      return false;
    }
  }

  _handleDeviceLost(lostDevice, info = {}) {
    // destroy() resolves GPUDevice.lost too. Only a spontaneous loss of the
    // currently-published device is actionable; stale/intentional events must not
    // invalidate a replacement or notify the coordinator.
    if (this.device !== lostDevice || this._destroyRequested || this._deviceLost) return;
    this._deviceLost = true;
    this.ready = false;
    try { this.warn("grab device lost:", info.message || info.reason || "unknown reason"); } catch {}
    try { this.onDeviceLost?.(lostDevice, info); }
    catch (error) { try { this.warn("grab device-loss callback failed:", error?.message || String(error)); } catch {} }
    void this.destroy();
  }

  _allocationPlan(w, h) {
    if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || w <= 0 || h <= 0) {
      throw new GrabResourceLimitError(`grab dimensions must be positive safe integers, got ${w}x${h}`, { width: w, height: h });
    }
    const maxTextureDimension2D = positiveDeviceLimit(this.device, "maxTextureDimension2D", 8192);
    if (w > maxTextureDimension2D || h > maxTextureDimension2D) {
      throw new GrabResourceLimitError(
        `grab dimensions ${w}x${h} exceed maxTextureDimension2D ${maxTextureDimension2D}`,
        { width: w, height: h, limit: maxTextureDimension2D, resource: "texture" },
      );
    }
    const rowBytes = w * 4;
    const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
    const bufferBytes = bytesPerRow * h;
    const maxBufferSize = positiveDeviceLimit(this.device, "maxBufferSize", 256 * 1024 * 1024);
    if (!Number.isSafeInteger(rowBytes) || !Number.isSafeInteger(bytesPerRow) ||
        !Number.isSafeInteger(bufferBytes) || bytesPerRow > 0xffffffff || bufferBytes > maxBufferSize) {
      throw new GrabResourceLimitError(
        `grab readback for ${w}x${h} requires ${bufferBytes} bytes; maxBufferSize is ${maxBufferSize}`,
        { width: w, height: h, requested: bufferBytes, limit: maxBufferSize, resource: "buffer" },
      );
    }
    return { bytesPerRow, bufferBytes };
  }

  _alloc(w, h) {
    if (!this.device || this._destroyRequested || this._deviceLost) throw new Error("grabber device is unavailable");
    const { bytesPerRow, bufferBytes } = this._allocationPlan(w, h);
    if (this._w === w && this._h === h && this.tex && this.texView && this.readBuf && this.bytesPerRow === bytesPerRow) {
      return { tex: this.tex, texView: this.texView, readBuf: this.readBuf, bytesPerRow };
    }

    let nextTex = null, nextReadBuf = null;
    try {
      nextTex = this.device.createTexture({
        size: [w, h],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const nextTexView = nextTex.createView();
      nextReadBuf = this.device.createBuffer({
        size: bufferBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const oldTex = this.tex, oldReadBuf = this.readBuf;
      this._w = w; this._h = h;
      this.tex = nextTex; this.texView = nextTexView;
      this.readBuf = nextReadBuf; this.bytesPerRow = bytesPerRow;
      nextTex = null; nextReadBuf = null;
      try { oldTex?.destroy?.(); } catch {}
      try { oldReadBuf?.destroy?.(); } catch {}
      return { tex: this.tex, texView: this.texView, readBuf: this.readBuf, bytesPerRow };
    } catch (error) {
      try { nextTex?.destroy?.(); } catch {}
      try { nextReadBuf?.destroy?.(); } catch {}
      throw error;
    }
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
    const promise = this._grabInner(video).finally(() => {
      this._mapPending = false;
      if (this._grabPromise === promise) this._grabPromise = null;
    });
    this._grabPromise = promise;
    return promise;
  }
  async _grabInner(video) {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return null;
    let mapped = false;
    let allocation = null;
    try {
      allocation = this._alloc(w, h);
      const device = this.device;
      let ext;
      try { ext = device.importExternalTexture({ source: video, colorSpace: SRGB_COLOR_SPACE }); }
      catch (e) { this.warn("grab importExternalTexture failed:", e.message); return null; }

      const bind = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: ext },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: allocation.texView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
      enc.copyTextureToBuffer(
        { texture: allocation.tex },
        { buffer: allocation.readBuf, bytesPerRow: allocation.bytesPerRow, rowsPerImage: h },
        { width: w, height: h }
      );
      device.queue.submit([enc.finish()]);

      await allocation.readBuf.mapAsync(GPUMapMode.READ);
      mapped = true;
      if (this._destroyRequested || this._deviceLost || this.device !== device || this.readBuf !== allocation.readBuf) return null;
      const mappedBytes = new Uint8Array(allocation.readBuf.getMappedRange());
      // de-pad rows (bytesPerRow may exceed w*4)
      const out = new Uint8ClampedArray(w * h * 4);
      const rowBytes = w * 4;
      for (let y = 0; y < h; y++) {
        out.set(mappedBytes.subarray(y * allocation.bytesPerRow, y * allocation.bytesPerRow + rowBytes), y * rowBytes);
      }
      return { data: out, width: w, height: h };
    } catch (e) {
      if (!this._destroyRequested && !this._deviceLost) this.warn("grab failed:", e.message);
      return null;
    } finally {
      if (mapped) {
        try { allocation?.readBuf?.unmap?.(); } catch {}
      }
    }
  }

  destroy() {
    if (this._destroyPromise) return this._destroyPromise;
    this.ready = false;
    this._destroyRequested = true;
    const pendingInit = this._initPromise;
    const pendingGrab = this._grabPromise;
    this._destroyPromise = (async () => {
      const pending = [pendingInit, pendingGrab].filter(Boolean);
      if (pending.length) await Promise.allSettled(pending);

      const device = this.device;
      const ownsDevice = this._ownsDevice;
      const tex = this.tex, readBuf = this.readBuf;
      this.device = null; this._ownsDevice = false;
      this.sampler = null; this.pipeline = null;
      this.tex = null; this.texView = null; this.readBuf = null;
      this._w = 0; this._h = 0; this.bytesPerRow = 0;
      try { tex?.destroy?.(); } catch {}
      try { readBuf?.destroy?.(); } catch {}
      if (ownsDevice) { try { device?.destroy?.(); } catch {} }
    })();
    return this._destroyPromise;
  }
}

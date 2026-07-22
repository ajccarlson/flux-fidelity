// fsrcnnx-rife-gpu.js — GPU-resident interpolation (GpuInterp).

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";

// Pixels stay on the GPU for capture, packing, inference, static-passthrough
// compositing, queued scheduling, and WebGPU-canvas presentation. There is no
// per-frame readback on this path.
//
// Flow per frame (grab loop calls captureCurrent then interpolate, then advance):
//   captureCurrent(video): importExternalTexture(current) → blit into curTex
//     (persistent). prevTex holds last frame (external textures are transient, so
//     the previous frame MUST be a persistent texture — hence the ping-pong).
//   interpolate(): pack(prevTex,curTex)→NCHW GPU buffer → ORT GPU-tensor infer →
//     composite+passthrough → pooled rgba8 result texture.
//   advance(): ping-pong prev/cur for next tick.
//
// DEVICE SHARING: buffers must be on ORT's device (Tensor.fromGpuBuffer). We use
// ort.env.webgpu.device. If unavailable / GPU tensors unsupported / any step fails,
// methods return null and the caller uses the CPU interpolate() path.

const BLIT_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var frame: texture_external;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>,3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
  var uv = array<vec2<f32>,3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
  var o: VSOut; o.pos = vec4(p[i],0.0,1.0); o.uv = uv[i]; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return textureSampleBaseClampToEdge(frame, samp, in.uv);
}
`;

const PACK_WGSL = `
struct P { padW:u32, padH:u32, w:u32, h:u32, channels:u32, t:f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(4) var<uniform> u: P;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= u.padW || y >= u.padH) { return; }
  let plane = u.padW * u.padH; let idx = y * u.padW + x;
  var a = vec3<f32>(0.0); var b = vec3<f32>(0.0);
  if (x < u.w && y < u.h) {
    let uv = vec2<f32>((f32(x)+0.5)/f32(u.w), (f32(y)+0.5)/f32(u.h));
    a = textureSampleLevel(texA, samp, uv, 0.0).rgb;
    b = textureSampleLevel(texB, samp, uv, 0.0).rgb;
  }
  outBuf[0u*plane+idx]=a.r; outBuf[1u*plane+idx]=a.g; outBuf[2u*plane+idx]=a.b;
  outBuf[3u*plane+idx]=b.r; outBuf[4u*plane+idx]=b.g; outBuf[5u*plane+idx]=b.b;
  if (u.channels >= 7u) { outBuf[6u*plane+idx] = u.t; }
}
`;

// composite: tween NCHW buffer + prev/cur textures -> rgba8 storage texture, with
// the static-region passthrough (mirrors the CPU version numerically).
const COMPOSITE_WGSL = `
struct P { padW:u32, padH:u32, w:u32, h:u32, iw:u32, ih:u32, tLo:f32, tHi:f32, staticOn:u32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> inBuf: array<f32>;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<uniform> u: P;
fn rifeAt(c: u32, yy: i32, xx: i32, plane: u32, stride: u32) -> f32 {
  return inBuf[c * plane + u32(yy) * stride + u32(xx)];
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= u.w || y >= u.h) { return; }
  // Bilinear fetch of the (possibly scale-reduced) model output back to full
  // res. Degenerates to exact identity when iw==w, ih==h.
  let plane = u.padW * u.padH;
  let sx = (f32(x) + 0.5) * f32(u.iw) / f32(u.w) - 0.5;
  let sy = (f32(y) + 0.5) * f32(u.ih) / f32(u.h) - 0.5;
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  let x0c = clamp(x0, 0, i32(u.iw) - 1); let x1c = clamp(x0 + 1, 0, i32(u.iw) - 1);
  let y0c = clamp(y0, 0, i32(u.ih) - 1); let y1c = clamp(y0 + 1, 0, i32(u.ih) - 1);
  let w00 = (1.0 - fx) * (1.0 - fy); let w10 = fx * (1.0 - fy);
  let w01 = (1.0 - fx) * fy;         let w11 = fx * fy;
  var rife = vec3<f32>(
    rifeAt(0u,y0c,x0c,plane,u.padW)*w00 + rifeAt(0u,y0c,x1c,plane,u.padW)*w10 + rifeAt(0u,y1c,x0c,plane,u.padW)*w01 + rifeAt(0u,y1c,x1c,plane,u.padW)*w11,
    rifeAt(1u,y0c,x0c,plane,u.padW)*w00 + rifeAt(1u,y0c,x1c,plane,u.padW)*w10 + rifeAt(1u,y1c,x0c,plane,u.padW)*w01 + rifeAt(1u,y1c,x1c,plane,u.padW)*w11,
    rifeAt(2u,y0c,x0c,plane,u.padW)*w00 + rifeAt(2u,y0c,x1c,plane,u.padW)*w10 + rifeAt(2u,y1c,x0c,plane,u.padW)*w01 + rifeAt(2u,y1c,x1c,plane,u.padW)*w11);
  let uv = vec2<f32>((f32(x)+0.5)/f32(u.w), (f32(y)+0.5)/f32(u.h));
  let a = textureSampleLevel(texA, samp, uv, 0.0).rgb;
  let b = textureSampleLevel(texB, samp, uv, 0.0).rgb;
  var outc = rife;
  if (u.staticOn != 0u) {
    let d = (abs(a.r-b.r)+abs(a.g-b.g)+abs(a.b-b.b))/3.0;
    if (d < u.tHi) {
      let real = (a+b)*0.5;
      let wRife = select((d-u.tLo)/(u.tHi-u.tLo), 0.0, d <= u.tLo);
      outc = rife*wRife + real*(1.0-wRife);
    }
  }
  textureStore(outTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

// present: sample a pooled result texture to the WebGPU canvas (no readback).
// blend: cheap non-AI tween = lerp(prev, cur, t) into a pooled result texture.
// Sub-millisecond fallback for when RIFE inference can't keep up (e.g. 1080p), so we
// can still process every source frame and hit ~2x output instead of dropping frames.
// Static regions stay stable naturally (A≈B → blend≈A); motion ghosts (crossfade).
const BLEND_WGSL = `
struct P { w:u32, h:u32, t:f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> u: P;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= u.w || y >= u.h) { return; }
  let uv = vec2<f32>((f32(x)+0.5)/f32(u.w), (f32(y)+0.5)/f32(u.h));
  let a = textureSampleLevel(texA, samp, uv, 0.0).rgb;
  let b = textureSampleLevel(texB, samp, uv, 0.0).rgb;
  let c = mix(a, b, u.t);
  textureStore(outTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(c, 1.0));
}
`;

const PRESENT_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>,3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
  var uv = array<vec2<f32>,3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
  var o: VSOut; o.pos = vec4(p[i],0.0,1.0); o.uv = uv[i]; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, in.uv, 0.0);
}
`;

export class GpuResourceLimitError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = "GpuResourceLimitError";
    this.code = "GPU_RESOURCE_LIMIT";
    this.details = details;
  }
}

function positiveDeviceLimit(device, name, fallback) {
  const value = Number(device?.limits?.[name]);
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function checkedProduct(values, label) {
  let result = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0 || result > Number.MAX_SAFE_INTEGER / value) {
      throw new GpuResourceLimitError(`${label} exceeds the safe integer range`, { values, resource: label });
    }
    result *= value;
  }
  return result;
}

export const DEFAULT_GPU_POOL_BUDGET_BYTES = 512 * 1024 * 1024;
export const DEFAULT_GPU_FRAME_BUDGET_BYTES = 256 * 1024 * 1024;
export const DEFAULT_GPU_INPUT_BUDGET_BYTES = 256 * 1024 * 1024;

export class GpuInterp {
  constructor({
    log,
    warn,
    onDeviceLost,
    maxPoolBytes = DEFAULT_GPU_POOL_BUDGET_BYTES,
    maxFrameBytes = DEFAULT_GPU_FRAME_BUDGET_BYTES,
    maxInputBytes = DEFAULT_GPU_INPUT_BUDGET_BYTES,
  } = {}) {
    if (!Number.isSafeInteger(maxPoolBytes) || maxPoolBytes <= 0) {
      throw new RangeError("maxPoolBytes must be a positive safe integer");
    }
    if (!Number.isSafeInteger(maxFrameBytes) || maxFrameBytes <= 0) {
      throw new RangeError("maxFrameBytes must be a positive safe integer");
    }
    if (!Number.isSafeInteger(maxInputBytes) || maxInputBytes <= 0) {
      throw new RangeError("maxInputBytes must be a positive safe integer");
    }
    this.log = log || console.log; this.warn = warn || console.warn;
    this.onDeviceLost = typeof onDeviceLost === "function" ? onDeviceLost : null;
    this.maxPoolBytes = maxPoolBytes;
    this.maxFrameBytes = maxFrameBytes;
    this.maxInputBytes = maxInputBytes;
    this.ready = false; this.device = null; this.ort = null;
    this._w = 0; this._h = 0; this._padW = 0; this._padH = 0; this._ch = 0;
    this._frames = 0;
    this._pool = []; // recycled result textures for the presentation queue
    this._allPooledTextures = new Set(); // includes checked-out/in-flight textures
    // Retired textures remain physically live until every captured operation is
    // idle and the device queue fence resolves. Keep their bytes in the bound so
    // rapid resolution churn cannot allocate around deferred destruction.
    this._retiringPooledBytes = 0;
    this._activeFrameBytes = 0;
    this._retiringFrameBytes = 0;
    this._activeInputBytes = 0;
    this._retiringInputBytes = 0;
    this._activeOps = 0;
    this._idleResolvers = [];
    this._retirements = new Set();
    this._destroyRequested = false;
    this._destroyPromise = null;
    this._initPromise = null;
    this._initDevice = undefined;
    this._initOrt = undefined;
    this._ownsDevice = false;
    this._deviceLost = false;
    this._deviceLossObserver = null;
    this.lastCaptureError = null;
    this.canvasCtx = null;
  }

  _trackRetirement(promise) {
    const tracked = Promise.resolve(promise).catch(() => {});
    this._retirements.add(tracked);
    tracked.finally(() => this._retirements.delete(tracked));
    return tracked;
  }

  _revokePooledTexture(texture) {
    if (!texture) return;
    if (texture._gpuInterpOwner === this) texture._gpuInterpOwner = null;
    texture._gpuInterpDevice = null;
    texture._gpuInterpBytes = 0;
    texture._refs = 0;
  }

  _afterSubmittedWork(callback, device = this.device) {
    let fence;
    try { fence = device?.queue?.onSubmittedWorkDone?.() || Promise.resolve(); }
    catch { fence = Promise.resolve(); }
    return this._trackRetirement(Promise.resolve(fence).catch(() => {}).then(callback));
  }

  _retireTensors(inputTensor, outputTensor, device = this.device, keepOutput = !!this.skipOutputDispose) {
    if (!inputTensor && (!outputTensor || keepOutput)) return;
    this._afterSubmittedWork(() => {
      try { inputTensor?.dispose?.(); } catch {}
      try { if (!keepOutput) outputTensor?.dispose?.(); } catch {}
    }, device);
  }

  _retireResources(resources, device = this.device) {
    const live = resources.filter(Boolean);
    if (!live.length) return;
    // A resize can replace buffers/textures while an operation is awaiting
    // session.run(). That operation has captured the old generation but may not
    // have submitted its composite command yet. Wait for every such user to
    // unwind, then fence the captured device before destroying that generation.
    const retirement = (async () => {
      await this._whenIdle();
      try { await device?.queue?.onSubmittedWorkDone?.(); } catch {}
      for (const resource of live) {
        if (resource?._gpuInterpOwner === this) this._revokePooledTexture(resource);
        try { resource.destroy?.(); } catch {}
      }
    })();
    return this._trackRetirement(retirement);
  }

  _retirePooledTextures(textures, device = this.device) {
    const live = [...new Set(textures.filter(Boolean))];
    if (!live.length) return Promise.resolve();
    const retiring = new Set(live);
    let bytes = 0;
    for (const texture of live) {
      const textureBytes = Number(texture._gpuInterpBytes) || 0;
      if (!Number.isSafeInteger(textureBytes) || textureBytes < 0 ||
          bytes > Number.MAX_SAFE_INTEGER - textureBytes) {
        throw new GpuResourceLimitError("retiring pooled texture accounting exceeds the safe integer range", {
          resource: "texture-pool",
        });
      }
      bytes += textureBytes;
    }
    if (this._retiringPooledBytes > Number.MAX_SAFE_INTEGER - bytes) {
      throw new GpuResourceLimitError("retiring pooled texture accounting exceeds the safe integer range", {
        resource: "texture-pool",
      });
    }
    this._pool = this._pool.filter((texture) => !retiring.has(texture));
    for (const texture of live) {
      this._allPooledTextures.delete(texture);
      // Revoke logical ownership as soon as the texture leaves the live set.
      // A stale caller reference must not retain/recycle an object that is only
      // waiting for its queue fence and physical destroy.
      this._revokePooledTexture(texture);
    }
    this._retiringPooledBytes += bytes;
    const retirement = (async () => {
      await this._whenIdle();
      try { await device?.queue?.onSubmittedWorkDone?.(); } catch {}
      for (const resource of live) { try { resource.destroy?.(); } catch {} }
    })().finally(() => {
      this._retiringPooledBytes = Math.max(0, this._retiringPooledBytes - bytes);
    });
    return this._trackRetirement(retirement);
  }

  _retireBudgetedResources(resources, bytes, counter, device = this.device) {
    const live = resources.filter(Boolean);
    if (!live.length || !bytes) return Promise.resolve();
    this[counter] += bytes;
    const retirement = (async () => {
      await this._whenIdle();
      try { await device?.queue?.onSubmittedWorkDone?.(); } catch {}
      for (const resource of live) { try { resource.destroy?.(); } catch {} }
    })().finally(() => {
      this[counter] = Math.max(0, this[counter] - bytes);
    });
    return this._trackRetirement(retirement);
  }

  _pooledBytesInUse() {
    let bytes = this._retiringPooledBytes;
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new GpuResourceLimitError("pooled texture accounting exceeds the safe integer range", {
        resource: "texture-pool",
      });
    }
    for (const texture of this._allPooledTextures) {
      const textureBytes = Number(texture._gpuInterpBytes) || 0;
      if (!Number.isSafeInteger(textureBytes) || textureBytes < 0 ||
          bytes > Number.MAX_SAFE_INTEGER - textureBytes) {
        throw new GpuResourceLimitError("pooled texture accounting exceeds the safe integer range", {
          resource: "texture-pool",
        });
      }
      bytes += textureBytes;
    }
    return bytes;
  }

  _endOperation() {
    this._activeOps = Math.max(0, this._activeOps - 1);
    if (this._activeOps === 0 && this._idleResolvers.length) {
      for (const resolve of this._idleResolvers.splice(0)) resolve();
    }
  }

  _whenIdle() {
    if (this._activeOps === 0) return Promise.resolve();
    return new Promise((resolve) => this._idleResolvers.push(resolve));
  }

  _handleDeviceLost(lostDevice, info = {}) {
    // An owned device's intentional destroy resolves `lost`; an adopted device can
    // also outlive this helper. Only the currently-published, spontaneously-lost
    // device may change state or notify the eventual lifecycle coordinator.
    if (this.device !== lostDevice || this._destroyRequested || this._deviceLost) return;
    this._deviceLost = true;
    this.ready = false;
    try { this.warn("gpu device lost:", info.message || info.reason || "unknown reason"); } catch {}
    try { this.onDeviceLost?.(lostDevice, info); }
    catch (error) { try { this.warn("gpu device-loss callback failed:", error?.message || String(error)); } catch {} }
    void this.destroy();
  }

  // init(device, ort): shared-device RIFE mode (device = ORT's device, ort provided).
  // init(null, null): STANDALONE blend mode — create our own device, no ORT needed;
  // RIFE inference (interpolateToPooledTex) is unavailable but blend works fully.
  _matchesInit(device, ort) {
    if (!this.ready) return false;
    const requiredOrt = ort?.Tensor?.fromGpuBuffer ? ort : null;
    if (device) {
      if (this.device !== device) return false;
      return !requiredOrt || this.ort === requiredOrt;
    }
    return !requiredOrt && this._ownsDevice;
  }

  // A helper can be reached by both the standalone-blend and RIFE startup paths.
  // Keep initialization single-flight at the instance boundary as well as at the
  // module coordinator: two callers must never request two owned devices or build
  // pipelines concurrently into the same mutable fields.
  init(device, ort) {
    if (this.ready) return Promise.resolve(this._matchesInit(device, ort));
    if (this._destroyRequested) return Promise.resolve(false);
    if (this._initPromise) {
      if (this._initDevice === device && this._initOrt === ort) return this._initPromise;
      return this._initPromise.then(() => false, () => false);
    }

    this._initDevice = device;
    this._initOrt = ort;
    let operation;
    operation = this._initInternal(device, ort).finally(() => {
      if (this._initPromise === operation) {
        this._initPromise = null;
        this._initDevice = undefined;
        this._initOrt = undefined;
      }
    });
    this._initPromise = operation;
    return operation;
  }

  async _initInternal(device, ort) {
    try {
      if (!device) {
        // standalone: request our own device (blend-only, no model load)
        if (!navigator.gpu) { this.warn("gpu: WebGPU unavailable"); return false; }
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) { this.warn("gpu: no adapter"); return false; }
        if (this._destroyRequested) return false;
        device = await adapter.requestDevice();
        if (this._destroyRequested) {
          try { device.destroy?.(); } catch {}
          return false;
        }
        this._ownsDevice = true;
        ort = null;
      } else if (!ort || !ort.Tensor || !ort.Tensor.fromGpuBuffer) {
        // device given but no usable ORT tensors: allow blend, disable RIFE infer
        ort = null;
      }
      this.device = device; this.ort = ort;
      this._deviceLost = false;
      this._rifeCapable = !!ort; // interpolateToPooledTex requires ORT GPU tensors
      this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
      const mk = (c) => device.createShaderModule({ code: c });
      this.blitPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: mk(BLIT_WGSL), entryPoint: "vs" },
        fragment: { module: mk(BLIT_WGSL), entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
      });
      this.packPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_WGSL), entryPoint: "main" } });
      this.compPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(COMPOSITE_WGSL), entryPoint: "main" } });
      this.blendPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(BLEND_WGSL), entryPoint: "main" } });
      this.blendParams = device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.packParams = device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.compParams = device.createBuffer({ size: 36, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this._canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      this.presentPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: mk(PRESENT_WGSL), entryPoint: "vs" },
        fragment: { module: mk(PRESENT_WGSL), entryPoint: "fs", targets: [{ format: this._canvasFormat }] },
        primitive: { topology: "triangle-list" },
      });
      // 2D-source blit (chain capture): samples a texture_2d (e.g. the upscaler's
      // tap texture — any canvas format) into our rgba8unorm working textures.
      this.blit2dPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: mk(PRESENT_WGSL), entryPoint: "vs" },
        fragment: { module: mk(PRESENT_WGSL), entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
      });
      const watchedDevice = device;
      if (watchedDevice?.lost?.then) {
        // A GPUDevice.lost promise cannot be unsubscribed and a shared ORT device
        // may outlive many helpers. Capture a detachable holder rather than this
        // instance so normal teardown can sever the otherwise permanent root.
        const observer = { target: this, device: watchedDevice };
        this._deviceLossObserver = observer;
        watchedDevice.lost.then(
          (info) => observer.target?._handleDeviceLost(observer.device, info),
          (error) => observer.target?._handleDeviceLost(observer.device, {
            reason: "unknown", message: error?.message || String(error),
          }),
        );
      }
      this.ready = true;
      // A device can arrive with an already-settled loss promise. Let that
      // watcher publish the loss before reporting initialization success.
      await Promise.resolve();
      if (this._destroyRequested || this._deviceLost || this.device !== watchedDevice) return false;
      return true;
    } catch (e) {
      this.warn("gpu init failed:", e.message);
      this.ready = false;
      // If an external destroy already owns cleanup it is waiting for this init
      // operation to settle. Do not await that promise from inside the operation
      // itself. Otherwise perform the same cleanup without the self-wait.
      if (!this._destroyRequested) {
        try { await this._beginDestroy(false); } catch {}
      }
      return false;
    }
  }

  _validateFrameDimensions(w, h, device = this.device) {
    if (!device) throw new GpuResourceLimitError("GPU device is unavailable", { resource: "device" });
    if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || w <= 0 || h <= 0) {
      throw new GpuResourceLimitError(`frame dimensions must be positive safe integers, got ${w}x${h}`, { width: w, height: h, resource: "texture" });
    }
    const limit = positiveDeviceLimit(device, "maxTextureDimension2D", 8192);
    if (w > limit || h > limit) {
      throw new GpuResourceLimitError(
        `frame dimensions ${w}x${h} exceed maxTextureDimension2D ${limit}`,
        { width: w, height: h, requested: Math.max(w, h), limit, resource: "texture" },
      );
    }
  }

  _inferencePlan(w, h, padTo, ch, scale, device = this.device) {
    this._validateFrameDimensions(w, h, device);
    if (!Number.isSafeInteger(padTo) || padTo <= 0 || padTo > 0xffffffff ||
        !Number.isSafeInteger(ch) || ch <= 0 || ch > 0xffffffff ||
        !Number.isFinite(scale) || scale <= 0) {
      throw new GpuResourceLimitError(
        `invalid inference configuration padTo=${padTo}, channels=${ch}, scale=${scale}`,
        { padTo, channels: ch, scale, resource: "inference" },
      );
    }
    const scaledW = w * scale, scaledH = h * scale;
    if (!Number.isSafeInteger(Math.round(scaledW)) || !Number.isSafeInteger(Math.round(scaledH))) {
      throw new GpuResourceLimitError("scaled inference dimensions exceed the safe integer range", { width: scaledW, height: scaledH, resource: "inference" });
    }
    const iw = Math.max(64, Math.round(scaledW));
    const ih = Math.max(64, Math.round(scaledH));
    const padW = Math.ceil(iw / padTo) * padTo;
    const padH = Math.ceil(ih / padTo) * padTo;
    if (!Number.isSafeInteger(padW) || !Number.isSafeInteger(padH) || padW > 0xffffffff || padH > 0xffffffff) {
      throw new GpuResourceLimitError("padded inference dimensions exceed the WebGPU u32 range", { padW, padH, resource: "inference" });
    }
    const maxWorkgroups = positiveDeviceLimit(device, "maxComputeWorkgroupsPerDimension", 65535);
    const workgroupsX = Math.ceil(padW / 8), workgroupsY = Math.ceil(padH / 8);
    if (workgroupsX > maxWorkgroups || workgroupsY > maxWorkgroups) {
      throw new GpuResourceLimitError(
        `inference dispatch ${workgroupsX}x${workgroupsY} exceeds maxComputeWorkgroupsPerDimension ${maxWorkgroups}`,
        { workgroupsX, workgroupsY, limit: maxWorkgroups, resource: "dispatch" },
      );
    }
    const bufferBytes = checkedProduct([ch, padH, padW, 4], "inference input buffer");
    const maxBufferSize = positiveDeviceLimit(device, "maxBufferSize", 256 * 1024 * 1024);
    const maxBindingSize = positiveDeviceLimit(device, "maxStorageBufferBindingSize", 128 * 1024 * 1024);
    const bufferLimit = Math.min(maxBufferSize, maxBindingSize);
    if (bufferBytes > bufferLimit) {
      throw new GpuResourceLimitError(
        `inference input requires ${bufferBytes} bytes; storage-buffer limit is ${bufferLimit}`,
        { requested: bufferBytes, limit: bufferLimit, resource: "buffer", padW, padH, channels: ch },
      );
    }
    return { iw, ih, padW, padH, bufferBytes };
  }

  _resize(w, h, padTo, ch, scale, includeInference) {
    const device = this.device;
    this._validateFrameDimensions(w, h, device);
    if (!Number.isSafeInteger(padTo) || padTo <= 0 || padTo > 0xffffffff ||
        !Number.isSafeInteger(ch) || ch <= 0 || ch > 0xffffffff) {
      throw new GpuResourceLimitError(`invalid frame configuration padTo=${padTo}, channels=${ch}`, { padTo, channels: ch, resource: "inference" });
    }
    // Validate the complete transaction before creating its first resource. This is
    // important when a frame resize and an input-buffer resize happen together:
    // a buffer limit failure must leave the former frame generation untouched.
    const inference = includeInference ? this._inferencePlan(w, h, padTo, ch, scale, device) : null;
    const frameBytes = checkedProduct([w, h, 4, 2], "frame texture pair");
    const needsFrames = this._w !== w || this._h !== h || !this.prevTex || !this.curTex;
    const needsInput = !!inference && (
      this._padW !== inference.padW || this._padH !== inference.padH || this._chBuf !== ch || !this.inBuf
    );
    const projectedFrameBytes = needsFrames
      ? this._activeFrameBytes + this._retiringFrameBytes + frameBytes
      : this._activeFrameBytes + this._retiringFrameBytes;
    if (!Number.isSafeInteger(projectedFrameBytes) || projectedFrameBytes > this.maxFrameBytes) {
      const replaceableBytes = this._activeFrameBytes + frameBytes;
      throw new GpuResourceLimitError(
        `live frame textures would require ${projectedFrameBytes} bytes; frame budget is ${this.maxFrameBytes}`,
        {
          requested: projectedFrameBytes,
          limit: this.maxFrameBytes,
          resource: "frame-textures",
          transient: Number.isSafeInteger(replaceableBytes) && replaceableBytes <= this.maxFrameBytes && this._retiringFrameBytes > 0,
        },
      );
    }
    const projectedInputBytes = needsInput
      ? this._activeInputBytes + this._retiringInputBytes + inference.bufferBytes
      : this._activeInputBytes + this._retiringInputBytes;
    if (!Number.isSafeInteger(projectedInputBytes) || projectedInputBytes > this.maxInputBytes) {
      const replaceableBytes = this._activeInputBytes + (inference?.bufferBytes || 0);
      throw new GpuResourceLimitError(
        `live inference input buffers would require ${projectedInputBytes} bytes; input budget is ${this.maxInputBytes}`,
        {
          requested: projectedInputBytes,
          limit: this.maxInputBytes,
          resource: "inference-inputs",
          transient: Number.isSafeInteger(replaceableBytes) && replaceableBytes <= this.maxInputBytes && this._retiringInputBytes > 0,
        },
      );
    }
    let nextPrev = null, nextCur = null, nextInBuf = null;
    try {
      if (needsFrames) {
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
        nextPrev = device.createTexture({ label: `rife-prev-${w}x${h}`, size: [w, h], format: "rgba8unorm", usage });
        nextCur = device.createTexture({ label: `rife-cur-${w}x${h}`, size: [w, h], format: "rgba8unorm", usage });
      }
      if (needsInput) {
        nextInBuf = device.createBuffer({
          label: `rife-inBuf-${inference.padW}x${inference.padH}`,
          size: inference.bufferBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
      }
    } catch (error) {
      for (const resource of [nextPrev, nextCur, nextInBuf]) { try { resource?.destroy?.(); } catch {} }
      throw error;
    }

    const retiredFrames = [];
    const retiredInputs = [];
    if (needsFrames) {
      const oldFrameBytes = this._activeFrameBytes;
      retiredFrames.push(this.prevTex, this.curTex);
      this.prevTex = nextPrev; this.curTex = nextCur;
      this._activeFrameBytes = frameBytes;
      this._w = w; this._h = h;
      this._frames = 0; // stale prev after a resize; need 2 fresh captures again
      if (retiredFrames.some(Boolean)) {
        this._retireBudgetedResources(retiredFrames, oldFrameBytes, "_retiringFrameBytes", device);
      }
    }
    this._padTo = padTo;
    this._ch = ch;
    if (inference) {
      if (needsInput) {
        const oldInputBytes = this._activeInputBytes;
        retiredInputs.push(this.inBuf);
        this.inBuf = nextInBuf;
        this._activeInputBytes = inference.bufferBytes;
        if (retiredInputs.some(Boolean)) {
          this._retireBudgetedResources(retiredInputs, oldInputBytes, "_retiringInputBytes", device);
        }
      }
      this._iw = inference.iw; this._ih = inference.ih;
      this._padW = inference.padW; this._padH = inference.padH; this._chBuf = ch;
    }
    return inference ? { padW: inference.padW, padH: inference.padH } : { padW: this._padW || 0, padH: this._padH || 0 };
  }

  _ensureFrameSize(w, h, padTo, ch) {
    return this._resize(w, h, padTo, ch, 1, false);
  }

  _size(w, h, padTo, ch, scale = 1) {
    return this._resize(w, h, padTo, ch, scale, true);
  }

  hasPrev() { return this._frames >= 2; }
  advance() { const t = this.prevTex; this.prevTex = this.curTex; this.curTex = t; }
  resetFrames() {
    // Keep the allocated ping-pong textures, but forget their logical contents.
    // The next capture becomes a fresh first frame after seek/source boundaries.
    this._frames = 0;
  }


  getSplit() { return this.lastTiming || { pack: 0, run: 0, comp: 0, read: 0 }; }

  // ---- Full GPU presentation (no readback): pooled result textures + WebGPU canvas ----

  configureCanvas(canvas) {
    try {
      this.canvasCtx = canvas.getContext("webgpu");
      this.canvasCtx.configure({
        device: this.device,
        format: this._canvasFormat,
        colorSpace: SRGB_COLOR_SPACE,
        alphaMode: "opaque",
      });
      this._presentCanvas = canvas;
      return true;
    } catch (e) { this.warn("gpu canvas configure failed:", e.message); return false; }
  }

  _acquireTex(w = this._w, h = this._h, device = this.device) {
    this._validateFrameDimensions(w, h, device);
    // Recycle a pooled texture sized to this operation's captured frame. An
    // inference can finish after capture has resized the instance, so consulting
    // mutable this._w/_h after session.run() would produce the wrong result size.
    let matchingIndex = -1;
    for (let index = this._pool.length - 1; index >= 0; index--) {
      const texture = this._pool[index];
      if (texture._w === w && texture._h === h &&
          (!texture._gpuInterpDevice || texture._gpuInterpDevice === device)) {
        matchingIndex = index;
        break;
      }
    }
    if (matchingIndex >= 0) {
      const [matching] = this._pool.splice(matchingIndex, 1);
      matching._refs = 1;
      return matching;
    }
    const requestedBytes = checkedProduct([w, h, 4], "pooled texture");
    // No exact reusable texture exists. Every free entry is stale for this
    // request, so evict the whole zero-reference generation. Its bytes remain in
    // accounting until the retirement fence completes.
    if (this._pool.length) {
      const stale = [...this._pool];
      this._retirePooledTextures(stale, stale[0]?._gpuInterpDevice || device);
    }
    const currentBytes = this._pooledBytesInUse();
    const projectedBytes = currentBytes + requestedBytes;
    if (!Number.isSafeInteger(projectedBytes) || projectedBytes > this.maxPoolBytes) {
      throw new GpuResourceLimitError(
        `pooled textures would require ${projectedBytes} bytes; pool budget is ${this.maxPoolBytes}`,
        {
          requested: projectedBytes,
          limit: this.maxPoolBytes,
          resource: "texture-pool",
          transient: requestedBytes <= this.maxPoolBytes && currentBytes > 0,
        },
      );
    }
    let tex;
    try {
      tex = device.createTexture({
        label: `rife-pool-${w}x${h}`,
        size: [w, h], format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      });
    } catch (error) {
      throw error;
    }
    tex._w = w; tex._h = h; tex._refs = 1;
    tex._gpuInterpOwner = this;
    tex._gpuInterpDevice = device;
    tex._gpuInterpBytes = requestedBytes;
    this._allPooledTextures.add(tex);
    return tex;
  }
  // Refcounted release: an in-flight (pipelined) inference retains its pair while
  // presentation may release the same pooled textures; only recycle at zero refs.
  retainTex(tex) {
    if (!tex || tex._gpuInterpOwner !== this || this._destroyRequested) return;
    tex._refs = (Number.isFinite(tex._refs) && tex._refs > 0 ? tex._refs : 1) + 1;
  }
  releaseTex(tex) {
    if (!tex || tex._gpuInterpOwner !== this || !Number.isFinite(tex._refs) || tex._refs <= 0) return;
    tex._refs--;
    if (tex._refs > 0) return;
    if (this._destroyRequested) return;
    if (this._pool.length < 64) this._pool.push(tex);
    else {
      this._retirePooledTextures([tex], tex._gpuInterpDevice || this.device);
    }
  }

  // Capture current video → a POOLED texture (returned, for the queue) and mirror
  // it into curTex (for inference). Ping-pong prev/cur handled by advance().
  // Current padded model-input dims (what the pack shader will produce). Used by
  // the session re-pinner: graph capture requires exactly these shapes.
  padDims() { return { padW: this._padW || 0, padH: this._padH || 0 }; }

  captureToPooled(video, padTo, ch) {
    this.lastCaptureError = null;
    if (!this.ready) return null;
    let pooled = null;
    try {
      const w = video.videoWidth, h = video.videoHeight;
      this._ensureFrameSize(w, h, padTo || 8, ch || 7);
      pooled = this._acquireTex();
      const ext = this.device.importExternalTexture({ source: video, colorSpace: SRGB_COLOR_SPACE });
      const bind = this.device.createBindGroup({
        layout: this.blitPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: ext }],
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: pooled.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(this.blitPipe); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
      // mirror into curTex for inference use
      enc.copyTextureToTexture({ texture: pooled }, { texture: this.curTex }, { width: w, height: h });
      this.device.queue.submit([enc.finish()]);
      this._frames = (this._frames || 0) + 1;
      return pooled;
    } catch (e) {
      this.lastCaptureError = e;
      if (pooled) this.releaseTex(pooled);
      this.warn("gpu captureToPooled failed:", e.message);
      return null;
    }
  }

  // Chain capture: same as captureToPooled but the source is a SAME-DEVICE texture
  // (the upscaler's tap) instead of the raw <video>. Samples via blit2dPipe (handles
  // any canvas format → rgba8unorm) and mirrors into curTex for tweening.
  captureTexToPooled(srcTex, padTo, ch) {
    this.lastCaptureError = null;
    if (!this.ready || !srcTex) return null;
    let pooled = null;
    try {
      const w = srcTex.width, h = srcTex.height;
      this._ensureFrameSize(w, h, padTo || 8, ch || 7);
      pooled = this._acquireTex();
      const bind = this.device.createBindGroup({
        layout: this.blit2dPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: srcTex.createView() }],
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: pooled.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(this.blit2dPipe); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
      enc.copyTextureToTexture({ texture: pooled }, { texture: this.curTex }, { width: w, height: h });
      this.device.queue.submit([enc.finish()]);
      this._frames = (this._frames || 0) + 1;
      return pooled;
    } catch (e) {
      this.lastCaptureError = e;
      if (pooled) this.releaseTex(pooled);
      this.warn("gpu captureTexToPooled failed:", e.message);
      return null;
    }
  }

  // Run pack→infer→composite writing into a POOLED result texture (returned). No
  // readback. Returns null if not enough frames yet or on failure.
  // NOTE: the w/h params are IGNORED — the capture (_size) defines the pipeline
  // dimensions. In chain mode the caller's video dims differ from the tap dims;
  // using them here packed/composited a mis-sized region (zoomed-corner tweens
  // alternating with full-size real frames — Aaron's shrink/expand jitter).
  async interpolateToPooledTex(session, MODEL_IO, _wIgnored, _hIgnored, t, useStatic, pairPrev = null, pairCur = null, scale = 1) {
    if (!this.ready || this._destroyRequested || !this._rifeCapable) return null;
    this._activeOps++;
    let inputTensor = null, outT = null, resultTex = null, tensorsRetired = false, op = null;
    try {
      // Apply the requested inference scale (re-sizes the inference alloc only if
      // it changed; frame textures are untouched, so prev stays valid).
      this._scale = scale;
      if (this._w) this._size(this._w, this._h, this._padTo || 8, this._ch, scale);
      // Capture every object and dimension used by this operation before the first
      // await. Capture may resize the instance while session.run() is pending; its
      // replacement generation must not leak into this operation's composite pass.
      // Explicit-pair mode likewise remains bound to the caller's frame pair.
      op = {
        device: this.device,
        ort: this.ort,
        packPipe: this.packPipe,
        compPipe: this.compPipe,
        packParams: this.packParams,
        compParams: this.compParams,
        sampler: this.sampler,
        inBuf: this.inBuf,
        prevTex: pairPrev || this.prevTex,
        curTex: pairCur || this.curTex,
        w: this._w,
        h: this._h,
        iw: this._iw,
        ih: this._ih,
        padW: this._padW,
        padH: this._padH,
        ch: this._ch,
        inputName: MODEL_IO.inputName,
        outputName: MODEL_IO.outputName,
        keepOutput: !!this.skipOutputDispose,
      };
      const prevT = op.prevTex, curT = op.curTex;
      if (!prevT || !curT) return null;
      // Explicit scheduler pairs are pooled textures annotated with the capture
      // dimensions. A source resize can otherwise combine an old retained frame
      // with the new allocation generation and make the pack/composite passes
      // read incompatible extents. Reject incomplete or mismatched pairs at this
      // final boundary even if a caller misses its normal resize flush.
      if (pairPrev || pairCur) {
        const pairMatches = pairPrev && pairCur
          && pairPrev._w === pairCur._w && pairPrev._h === pairCur._h
          && pairPrev._w === op.w && pairPrev._h === op.h;
        if (!pairMatches) return null;
      }
      if (!pairPrev && !this.hasPrev()) return null;
      const { w, h, iw, ih, padW, padH, ch } = op;
      const T = { pack: 0, run: 0, comp: 0, read: 0 };
      const _tp = performance.now();
      // pack prev/cur → inBuf
      {
        const p = new ArrayBuffer(24); const dv = new DataView(p);
        dv.setUint32(0, padW, true); dv.setUint32(4, padH, true);
        // content dims = SCALED infer dims: the shader samples the full-res
        // frame textures at these dims — the sampler performs the downscale.
        dv.setUint32(8, iw, true); dv.setUint32(12, ih, true);
        dv.setUint32(16, ch, true); dv.setFloat32(20, t, true);
        op.device.queue.writeBuffer(op.packParams, 0, p);
        const bind = op.device.createBindGroup({
          layout: op.packPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: op.sampler },
            { binding: 1, resource: prevT.createView() },
            { binding: 2, resource: curT.createView() },
            { binding: 3, resource: { buffer: op.inBuf } },
            { binding: 4, resource: { buffer: op.packParams } },
          ],
        });
        const enc = op.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(op.packPipe); pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(padW / 8), Math.ceil(padH / 8), 1);
        pass.end(); op.device.queue.submit([enc.finish()]);
      }
      T.pack = performance.now() - _tp; const _tr = performance.now();
      inputTensor = op.ort.Tensor.fromGpuBuffer(op.inBuf, { dataType: "float32", dims: [1, ch, padH, padW] });
      const feeds = {}; feeds[op.inputName] = inputTensor;
      let outGpu = null;
      try {
        const r = await session.run(feeds);
        outT = r[op.outputName];
        outGpu = outT?.gpuBuffer;
      } catch (e) {
        this.warn("gpu infer failed:", e.message);
        return null;
      }
      if (!outGpu || this._destroyRequested) return null;
      // GRAPH CAPTURE: run() is submit-only (returns in ~0.2ms without awaiting the
      // GPU). That await was BOTH our inference measurement AND the chain's
      // backpressure — without it the chain submitted 7 unthrottled replays/gap,
      // flooded the queue, and starved video compositing (in fps collapsed to 9.6).
      // An explicit completion fence restores both. Note: includes co-tenant queue
      // work (upscaler/present) — exactly the "loaded latency" the level gates want.
      if (op.keepOutput) { try { await op.device.queue.onSubmittedWorkDone(); } catch {} }
      T.run = performance.now() - _tr; const _tc = performance.now();
      // composite+passthrough → POOLED result texture (no readback)
      resultTex = this._acquireTex(w, h, op.device);
      {
        const p = new ArrayBuffer(36); const dv = new DataView(p);
        dv.setUint32(0, padW, true); dv.setUint32(4, padH, true);
        dv.setUint32(8, w, true); dv.setUint32(12, h, true);
        dv.setUint32(16, iw, true); dv.setUint32(20, ih, true);
        dv.setFloat32(24, 0.012, true); dv.setFloat32(28, 0.05, true);
        dv.setUint32(32, useStatic ? 1 : 0, true);
        op.device.queue.writeBuffer(op.compParams, 0, p);
        const bind = op.device.createBindGroup({
          layout: op.compPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: op.sampler },
            { binding: 1, resource: prevT.createView() },
            { binding: 2, resource: curT.createView() },
            { binding: 3, resource: { buffer: outGpu } },
            { binding: 4, resource: resultTex.createView() },
            { binding: 5, resource: { buffer: op.compParams } },
          ],
        });
        const enc = op.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(op.compPipe); pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1);
        pass.end(); op.device.queue.submit([enc.finish()]);
      }
      // The composite command reads outGpu asynchronously. Retire tensor wrappers
      // only after all work submitted through that command has completed; immediate
      // disposal can invalidate or recycle the ORT output while WebGPU still reads it.
      // Graph-capture sessions own/reuse their output tensor, so only our input
      // wrapper is retired in that mode.
      this._retireTensors(inputTensor, outT, op.device, op.keepOutput);
      inputTensor = null; outT = null; tensorsRetired = true;
      T.comp = performance.now() - _tc; this.lastTiming = T;
      if (this._destroyRequested) return null;
      const returned = resultTex;
      resultTex = null;
      return returned;
    } catch (e) {
      this.warn("gpu interpolateToPooledTex failed:", e.message);
      return null;
    } finally {
      if (!tensorsRetired) {
        // `op` is initialized before any inference await; failures before then do
        // not produce tensors. Preserve the captured device/output ownership when
        // unwinding a post-run failure or destroy request.
        const device = op?.device || this.device;
        const keepOutput = op ? op.keepOutput : !!this.skipOutputDispose;
        this._retireTensors(inputTensor, outT, device, keepOutput);
      }
      if (resultTex) this.releaseTex(resultTex);
      this._endOperation();
    }
  }

  // Cheap non-AI blend tween: lerp(prev, cur, t) → pooled texture. No pack/infer, so
  // it's sub-ms and lets us keep up with the source frame rate when RIFE can't.
  blendToPooledTex(t = 0.5, texA = null, texB = null) {
    // Pair mode (ladder): blend two EXPLICIT textures (real or RIFE tween). The
    // ping-pong guard only applies in legacy prev/cur mode.
    if (!this.ready || (!texA && !this.hasPrev())) return null;
    const _t0 = performance.now();
    let resultTex = null;
    try {
      const w = this._w, h = this._h;
      resultTex = this._acquireTex();
      const p = new ArrayBuffer(12); const dv = new DataView(p);
      dv.setUint32(0, w, true); dv.setUint32(4, h, true); dv.setFloat32(8, t, true);
      this.device.queue.writeBuffer(this.blendParams, 0, p);
      const bind = this.device.createBindGroup({
        layout: this.blendPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: (texA || this.prevTex).createView() },
          { binding: 2, resource: (texB || this.curTex).createView() },
          { binding: 3, resource: resultTex.createView() },
          { binding: 4, resource: { buffer: this.blendParams } },
        ],
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.blendPipe); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1);
      pass.end(); this.device.queue.submit([enc.finish()]);
      this.lastTiming = { pack: 0, run: 0, comp: performance.now() - _t0, read: 0 };
      return resultTex;
    } catch (e) {
      if (resultTex) this.releaseTex(resultTex);
      this.warn("gpu blendToPooledTex failed:", e.message);
      return null;
    }
  }

  // Render a pooled result texture to the WebGPU canvas.
  presentTexture(tex) {
    if (!this.ready || !this.canvasCtx || !tex) return false;
    try {
      // match canvas size to texture
      if (this._presentCanvas && (this._presentCanvas.width !== tex._w || this._presentCanvas.height !== tex._h)) {
        this._presentCanvas.width = tex._w; this._presentCanvas.height = tex._h;
      }
      const view = this.canvasCtx.getCurrentTexture().createView();
      const bind = this.device.createBindGroup({
        layout: this.presentPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: tex.createView() }],
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(this.presentPipe); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
      this.device.queue.submit([enc.finish()]);
      return true;
    } catch (e) { this.warn("gpu present failed:", e.message); return false; }
  }

  destroy() {
    return this._beginDestroy(true);
  }

  _beginDestroy(waitForInit) {
    if (this._destroyPromise) return this._destroyPromise;
    this.ready = false;
    this._destroyRequested = true;
    if (this._deviceLossObserver) this._deviceLossObserver.target = null;
    this._deviceLossObserver = null;
    this.onDeviceLost = null;
    const pendingInit = waitForInit ? this._initPromise : null;
    this._destroyPromise = (async () => {
      // requestDevice() can still be pending when teardown begins. init() checks
      // _destroyRequested and destroys any late device before settling; waiting
      // here makes the public destroy promise a complete physical barrier.
      try { await pendingInit; } catch {}
      const ownedDevice = this._ownsDevice ? this.device : null;
      this._ownsDevice = false;
      // Let an awaited ORT run unwind, then fence every command it and the
      // presentation path submitted before destroying referenced resources.
      await this._whenIdle();
      try { await this.device?.queue?.onSubmittedWorkDone?.(); } catch {}
      if (this._retirements.size) {
        try { await Promise.allSettled([...this._retirements]); } catch {}
      }
      try { this.canvasCtx?.unconfigure?.(); } catch {}
      for (const texture of this._allPooledTextures) {
        this._revokePooledTexture(texture);
        try { texture.destroy(); } catch {}
      }
      this._allPooledTextures.clear();
      this._pool = [];
      this._retiringPooledBytes = 0;
      for (const texture of [this.prevTex, this.curTex]) { try { texture?.destroy?.(); } catch {} }
      for (const buffer of [this.inBuf, this.packParams, this.compParams, this.blendParams]) { try { buffer?.destroy?.(); } catch {} }
      this.prevTex = null; this.curTex = null; this.inBuf = null;
      this.packParams = null; this.compParams = null; this.blendParams = null;
      this.sampler = null;
      this.blitPipe = null; this.blit2dPipe = null;
      this.packPipe = null; this.compPipe = null; this.blendPipe = null;
      this.presentPipe = null;
      this.canvasCtx = null; this._presentCanvas = null;
      this.device = null; this.ort = null; this._rifeCapable = false;
      this._w = 0; this._h = 0; this._padW = 0; this._padH = 0; this._frames = 0;
      this._activeFrameBytes = 0; this._retiringFrameBytes = 0;
      this._activeInputBytes = 0; this._retiringInputBytes = 0;
      // Standalone blend mode requests its own GPUDevice. Shared ORT/chain
      // devices remain owned by their provider and must not be destroyed here.
      try { ownedDevice?.destroy?.(); } catch {}
    })();
    return this._destroyPromise;
  }
}

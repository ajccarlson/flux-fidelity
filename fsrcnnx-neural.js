// fsrcnnx-neural.js — ONNX neural upscaler engine (v0.49.0, Tier-B ladder).
//
// Runs community super-resolution models (SPAN / RealPLKSR / DAT2 / ATD, any
// spandrel-exportable arch) as a SECOND ORT WebGPU session alongside RIFE's,
// sharing the same ORT env/device. Unlike the WGSL mpv ports (luma-only), these
// models are RGB: chroma gets neural treatment and the recombine stage is
// bypassed — output composites straight into an rgba16float texture that
// main.js presents through the existing SSimDS/sharpen/deband tail.
//
// Frame flow (mirrors the RIFE GPU-resident pattern exactly):
//   rvfc → pack pass (external texture → padded NCHW fp32 storage buffer,
//   submitted before any await so the external texture is consumed in-task)
//   → await session.run (fromGpuBuffer in, preferredOutputLocation gpu-buffer
//   out — zero readback) → composite pass (output buffer → rgba16float tex,
//   crop replicate-pad) → caller presents.
//
// Models are dropped into model/neural/ with a manifest.json; the export kit
// in tools/neural-export/ produces both. Sessions use DYNAMIC dims (no
// freeDimensionOverrides, no graph capture — experiment #1's verdict stands).

import { ensureOrt, getOrtDevice } from "./fsrcnnx-rife.js";

const PACK_EXT_WGSL = `
struct P { padW:u32, padH:u32, w:u32, h:u32 }
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_external;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.padW || gid.y >= u.padH) { return; }
  let xx = min(gid.x, u.w - 1u);
  let yy = min(gid.y, u.h - 1u);
  let uv = (vec2<f32>(f32(xx), f32(yy)) + vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(u.w), f32(u.h));
  let c = textureSampleBaseClampToEdge(src, samp, uv).rgb;
  let plane = u.padW * u.padH;
  let idx = gid.y * u.padW + gid.x;
  dst[idx] = c.r;
  dst[plane + idx] = c.g;
  dst[2u * plane + idx] = c.b;
}`;

// texture_2d twin (same body, sampled with explicit LOD) — the source-is-a-
// parameter doctrine: frames from a decoder and frames synthesized upstream
// go through identical math.
const PACK_TEX_WGSL = PACK_EXT_WGSL
  .replace("texture_external", "texture_2d<f32>")
  .replace("textureSampleBaseClampToEdge(src, samp, uv)", "textureSampleLevel(src, samp, uv, 0.0)");

const COMPOSITE_WGSL = `
struct P { strideW:u32, plane:u32, outW:u32, outH:u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.outW || gid.y >= u.outH) { return; }
  let idx = gid.y * u.strideW + gid.x;
  let r = clamp(src[idx], 0.0, 1.0);
  let g = clamp(src[u.plane + idx], 0.0, 1.0);
  let b = clamp(src[2u * u.plane + idx], 0.0, 1.0);
  textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, 1.0));
}`;

export function createNeuralEngine({ log = console.log, warn = console.warn } = {}) {
  let ort = null;
  let session = null;
  let device = null;
  let manifest = null;          // [{key,label,file,scale,padMultiple?,input?,output?}]
  let active = null;            // manifest entry of the loaded model
  let inputName = "input", outputName = "output";
  let fp16Model = false;

  // GPU resources (allocated on ORT's device)
  let sampler = null;
  let packExtPipe = null, packTexPipe = null, compPipe = null;
  let inBuf = null, inBufSize = 0;
  let packU = null, compU = null;
  let outTex = null, outTexW = 0, outTexH = 0;

  // instrumentation (mirrors RIFE's readout vocabulary)
  const stats = { last: 0, mu: 0, n: 0, skip: 0, fails: 0 };

  async function loadManifest() {
    if (manifest) return manifest;
    try {
      const r = await fetch(chrome.runtime.getURL("model/neural/manifest.json"));
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      manifest = Array.isArray(j) ? j : (j.models || []);
    } catch (e) {
      manifest = [];
    }
    return manifest;
  }

  function ensurePipelines() {
    if (packExtPipe) return;
    const mk = (code) => device.createShaderModule({ code });
    packExtPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_EXT_WGSL), entryPoint: "main" } });
    packTexPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_TEX_WGSL), entryPoint: "main" } });
    compPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(COMPOSITE_WGSL), entryPoint: "main" } });
    sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    packU = device.createBuffer({ label: "neural-packU", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    compU = device.createBuffer({ label: "neural-compU", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  function ensureInBuf(padW, padH) {
    const need = padW * padH * 3 * 4;
    if (inBuf && inBufSize === need) return;
    try { inBuf && inBuf.destroy(); } catch {}
    inBufSize = need;
    inBuf = device.createBuffer({
      label: `neural-in-${padW}x${padH}`,
      size: need,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  function ensureOutTex(w, h) {
    if (outTex && outTexW === w && outTexH === h) return;
    try { outTex && outTex.destroy(); } catch {}
    outTexW = w; outTexH = h;
    outTex = device.createTexture({
      label: `neural-out-${w}x${h}`,
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  async function init(key) {
    const list = await loadManifest();
    if (!list.length) throw new Error("no neural models: model/neural/manifest.json missing or empty");
    const entry = list.find((m) => m.key === key) || list[0];

    ort = await ensureOrt();
    // Create the NEW session before releasing any old one: the shared ORT
    // device's lifetime follows its sessions, and letting the refcount touch
    // zero mid-swap would tear down the device under the upscaler (crash-1's
    // lesson, applied preemptively).
    const url = chrome.runtime.getURL("model/neural/" + entry.file);
    const opts = {
      executionProviders: [{ name: "webgpu" }],
      graphOptimizationLevel: "all",
      enableGraphCapture: false,
      preferredOutputLocation: "gpu-buffer",
    };
    try { ort.env.webgpu.enableFp16 = true; } catch {}
    let next;
    try {
      next = await ort.InferenceSession.create(url, opts);
    } catch (e) {
      throw new Error("neural session create failed (" + entry.file + "): " + e.message);
    }
    const old = session;
    session = next;
    active = entry;
    inputName = entry.input || (session.inputNames && session.inputNames[0]) || "input";
    outputName = entry.output || (session.outputNames && session.outputNames[0]) || "output";
    fp16Model = /fp16/i.test(entry.file) || entry.fp16 === true;
    if (old) { try { old.release(); } catch {} }

    const dev = getOrtDevice();
    if (!dev) throw new Error("ORT device unavailable after neural session create");
    if (device && device !== dev) {
      // device changed under us (interp restart etc.) — drop stale resources
      destroyGpuResources();
    }
    device = dev;
    ensurePipelines();
    log(`neural: session ready — ${entry.label || entry.key} (${entry.scale}x, ${fp16Model ? "fp16" : "fp32"} weights, dynamic dims)`);
    return entry;
  }

  function destroyGpuResources() {
    for (const b of [inBuf, packU, compU]) { try { b && b.destroy(); } catch {} }
    try { outTex && outTex.destroy(); } catch {}
    inBuf = null; inBufSize = 0; packU = null; compU = null;
    outTex = null; outTexW = 0; outTexH = 0;
    packExtPipe = null; packTexPipe = null; compPipe = null; sampler = null;
  }

  // src: external texture (default) or { tex } for the texture_2d twin.
  // Returns { tex, outW, outH } or throws.
  async function run(src, srcW, srcH) {
    if (!session || !device) throw new Error("neural engine not initialized");
    const mult = Math.max(1, active.padMultiple | 0 || 1);
    const padW = Math.ceil(srcW / mult) * mult;
    const padH = Math.ceil(srcH / mult) * mult;
    const scale = active.scale;
    const t0 = performance.now();

    ensurePipelines();
    ensureInBuf(padW, padH);
    device.queue.writeBuffer(packU, 0, new Uint32Array([padW, padH, srcW, srcH]));

    // pack pass — MUST be encoded and submitted before any await (external
    // textures expire at task end).
    {
      const isTex = src && src.tex;
      const pipe = isTex ? packTexPipe : packExtPipe;
      const enc = device.createCommandEncoder();
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: isTex ? src.tex.createView() : src },
          { binding: 2, resource: { buffer: inBuf } },
          { binding: 3, resource: { buffer: packU } },
        ],
      });
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(Math.ceil(padW / 8), Math.ceil(padH / 8));
      cp.end();
      device.queue.submit([enc.finish()]);
    }

    const inTensor = ort.Tensor.fromGpuBuffer(inBuf, { dataType: "float32", dims: [1, 3, padH, padW] });
    let outT = null;
    try {
      const r = await session.run({ [inputName]: inTensor });
      outT = r[outputName];
      if (!outT || !outT.gpuBuffer) throw new Error("output not on GPU (" + outputName + ")");

      const outW = srcW * scale, outH = srcH * scale;
      const strideW = padW * scale;
      const plane = strideW * (padH * scale);
      ensureOutTex(outW, outH);
      device.queue.writeBuffer(compU, 0, new Uint32Array([strideW, plane, outW, outH]));
      const enc = device.createCommandEncoder();
      const bg = device.createBindGroup({
        layout: compPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outT.gpuBuffer } },
          { binding: 1, resource: outTex.createView() },
          { binding: 2, resource: { buffer: compU } },
        ],
      });
      const cp = enc.beginComputePass();
      cp.setPipeline(compPipe); cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(Math.ceil(outW / 8), Math.ceil(outH / 8));
      cp.end();
      device.queue.submit([enc.finish()]);

      const dt = performance.now() - t0;
      stats.last = dt;
      stats.mu = stats.n === 0 ? dt : stats.mu * 0.9 + dt * 0.1;
      stats.n++;
      return { tex: outTex, outW, outH };
    } finally {
      try { outT && outT.dispose(); } catch {}
    }
  }

  function stop() {
    // Release GPU resources but keep the SESSION alive: if this is the only
    // ORT session, releasing it tears down the shared device the upscaler has
    // adopted. Sessions persist until page unload (documented v1 tradeoff:
    // idle model VRAM in exchange for never orphaning the device).
    destroyGpuResources();
  }

  return {
    models: loadManifest,
    init,
    run,
    stop,
    ready: () => !!(session && device),
    activeEntry: () => active,
    device: () => device,
    stats: () => ({ ...stats }),
    bumpSkip: () => { stats.skip++; },
  };
}

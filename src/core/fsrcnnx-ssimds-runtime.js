// fsrcnnx-ssimds-runtime.js — builds + runs the SSimDownscaler pass chain.
//
// Input:  hiTex (rgba16float, oversized RGB from FSRCNNX recombine)
// Output: a display-res rgba16float RGB texture (perceptual downscale)
//
// Pipelines depend on the downscale ratio (baked into the mean + L2 shaders), so
// they're rebuilt when hi-res or target size changes. Bind groups are cached.

import {
  buildMeanShader, buildL2Shader, SSIMDS_MR_WGSL, SSIMDS_FINAL_WGSL,
} from "./fsrcnnx-ssimds.js";

const FMT = "rgba16float";

// Direct Catmull-Rom integration grows with the source/output ratio. These limits
// keep a transient tiny display target from turning one frame into an enormous
// fragment-shader loop. Callers need no special fallback: run() returns the input
// texture when prepare() elects to bypass, so their normal final blit performs the
// downscale instead.
export const SSIMDS_WORK_BUDGET = Object.freeze({
  maxAxisTaps: 129,
  maxMeanTapsPerPixel: 4096,
  maxEstimatedTextureSamples: 512_000_000,
});

function requireDimension(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return value;
}

// Pure estimator exported for deterministic tests and diagnostics. Tap counts are
// conservative upper bounds for inclusive integer loops spanning +/-2*ratio.
export function estimateSsimWork(hiW, hiH, dW, dH) {
  hiW = requireDimension(hiW, "hiW");
  hiH = requireDimension(hiH, "hiH");
  dW = requireDimension(dW, "dW");
  dH = requireDimension(dH, "dH");
  if (dW > hiW || dH > hiH) {
    throw new RangeError("SSimDownscaler only accepts output dimensions no larger than its input");
  }

  const ratioX = hiW / dW, ratioY = hiH / dH;
  const tapsX = Math.ceil(4 * ratioX) + 1;
  const tapsY = Math.ceil(4 * ratioY) + 1;
  const meanTapsPerPixel = tapsX * tapsY;
  const outputPixels = dW * dH;
  const terms = [
    outputPixels * meanTapsPerPixel, // mean pass
    hiW * dH * tapsY,               // vertical L2 pass
    outputPixels * tapsX,           // horizontal L2 pass
    outputPixels * 28,              // MR (18 reads) + final (10 reads)
  ];
  const estimatedTextureSamples = terms.reduce((sum, term) => sum + term, 0);
  const reasons = [];
  if (![tapsX, tapsY, meanTapsPerPixel, outputPixels, ...terms, estimatedTextureSamples]
    .every(Number.isSafeInteger)) {
    reasons.push("work estimate exceeds safe integer arithmetic");
  }
  if (tapsX > SSIMDS_WORK_BUDGET.maxAxisTaps || tapsY > SSIMDS_WORK_BUDGET.maxAxisTaps) {
    reasons.push(`axis taps exceed ${SSIMDS_WORK_BUDGET.maxAxisTaps}`);
  }
  if (meanTapsPerPixel > SSIMDS_WORK_BUDGET.maxMeanTapsPerPixel) {
    reasons.push(`mean taps per pixel exceed ${SSIMDS_WORK_BUDGET.maxMeanTapsPerPixel}`);
  }
  if (estimatedTextureSamples > SSIMDS_WORK_BUDGET.maxEstimatedTextureSamples) {
    reasons.push(`estimated texture samples exceed ${SSIMDS_WORK_BUDGET.maxEstimatedTextureSamples}`);
  }
  return {
    hiW, hiH, dW, dH, ratioX, ratioY, tapsX, tapsY, meanTapsPerPixel,
    estimatedTextureSamples,
    allowed: reasons.length === 0,
    reason: reasons.join("; ") || null,
  };
}

export class SsimDownscaler {
  constructor(device) {
    this.device = device;
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.hiW = 0; this.hiH = 0; this.dW = 0; this.dH = 0;
    this.pipelines = null;
    this.textures = null;
    this.bindGroups = null;
    this.bypassed = false;
    this.lastPlan = null;
  }

  _renderPipeline(code) {
    const m = this.device.createShaderModule({ code });
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs" },
      fragment: { module: m, entryPoint: "fs", targets: [{ format: FMT }] },
      primitive: { topology: "triangle-list" },
    });
  }

  _tex(w, h, label) {
    return this.device.createTexture({
      label, size: { width: w, height: h }, format: FMT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  // (Re)build for a given hi-res input and display target.
  prepare(hiW, hiH, dW, dH, hiTex) {
    const plan = estimateSsimWork(hiW, hiH, dW, dH);
    if (!hiTex || typeof hiTex.createView !== "function") {
      throw new TypeError("hiTex must be a GPU texture");
    }
    const maxDim = Number(this.device?.limits?.maxTextureDimension2D);
    if (Number.isFinite(maxDim) && maxDim > 0 &&
        [hiW, hiH, dW, dH].some((dimension) => dimension > maxDim)) {
      throw new RangeError(`SSimDownscaler dimensions exceed device limit ${maxDim}`);
    }

    const sizesSame = hiW === this.hiW && hiH === this.hiH && dW === this.dW && dH === this.dH;
    if (sizesSame && this.hiTex === hiTex && this.bindGroups && !this.bypassed) return true;

    if (!plan.allowed) {
      const oldTextures = this.textures;
      this.lastPlan = plan;
      this.hiW = hiW; this.hiH = hiH; this.dW = dW; this.dH = dH; this.hiTex = hiTex;
      this.bypassed = true;
      this.pipelines = null;
      this.textures = null;
      this.bindGroups = null;
      destroyTextureSet(oldTextures);
      return false;
    }

    const { ratioX, ratioY } = plan;
    const replaceOwned = !sizesSame || !this.pipelines || !this.textures || this.bypassed;
    let candidatePipelines = this.pipelines;
    let candidateTextures = this.textures;
    const createdTextures = [];
    try {
      if (replaceOwned) {
        candidatePipelines = {
          mean: this._renderPipeline(buildMeanShader(ratioX, ratioY)),
          l2v: this._renderPipeline(buildL2Shader(1, ratioY)), // vertical, squares input
          l2h: this._renderPipeline(buildL2Shader(0, ratioX)), // horizontal
          mr: this._renderPipeline(SSIMDS_MR_WGSL),
          final: this._renderPipeline(SSIMDS_FINAL_WGSL),
        };
        const makeTexture = (w, h, label) => {
          const texture = this._tex(w, h, label);
          createdTextures.push(texture);
          return texture;
        };
        candidateTextures = {
          mean: makeTexture(dW, dH, "ssimds-mean"),
          l2v: makeTexture(hiW, dH, "ssimds-l2v"),
          l2: makeTexture(dW, dH, "ssimds-l2"),
          mr: makeTexture(dW, dH, "ssimds-mr"),
          out: makeTexture(dW, dH, "ssimds-out"),
        };
      }

      const s = this.sampler;
      const bg = (pipe, entries) =>
        this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
      const candidateBindGroups = {
        mean: bg(candidatePipelines.mean, [
          { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
        ]),
        l2v: bg(candidatePipelines.l2v, [
          { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
        ]),
        l2h: bg(candidatePipelines.l2h, [
          { binding: 0, resource: s }, { binding: 1, resource: candidateTextures.l2v.createView() },
        ]),
        mr: bg(candidatePipelines.mr, [
          { binding: 0, resource: s },
          { binding: 1, resource: candidateTextures.mean.createView() },
          { binding: 2, resource: candidateTextures.l2.createView() },
        ]),
        final: bg(candidatePipelines.final, [
          { binding: 0, resource: s },
          { binding: 1, resource: candidateTextures.mean.createView() },
          { binding: 2, resource: candidateTextures.mr.createView() },
        ]),
      };

      const oldTextures = this.textures;
      this.lastPlan = plan;
      this.hiW = hiW; this.hiH = hiH; this.dW = dW; this.dH = dH; this.hiTex = hiTex;
      this.bypassed = false;
      this.pipelines = candidatePipelines;
      this.textures = candidateTextures;
      this.bindGroups = candidateBindGroups;
      if (replaceOwned) destroyTextureSet(oldTextures);
      return true;
    } catch (error) {
      destroyTextureSet(Object.fromEntries(createdTextures.map((texture, index) => [index, texture])));
      throw error;
    }
  }

  _pass(enc, pipe, bind, target) {
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    rp.setPipeline(pipe);
    rp.setBindGroup(0, bind);
    rp.draw(3);
    rp.end();
  }

  // Records the chain into `enc`. Returns the output texture (display-res RGB).
  run(enc, hiTex) {
    // Budget bypass contract: return the original texture without recording work.
    // Existing callers subsequently sample it into their display-sized target.
    if (this.bypassed) return hiTex;
    if (!this.pipelines || !this.textures || !this.bindGroups || this.hiTex !== hiTex) {
      throw new Error("SSimDownscaler.run requires a successful prepare for the same input texture");
    }
    this._pass(enc, this.pipelines.mean, this.bindGroups.mean, this.textures.mean);
    this._pass(enc, this.pipelines.l2v, this.bindGroups.l2v, this.textures.l2v);
    this._pass(enc, this.pipelines.l2h, this.bindGroups.l2h, this.textures.l2);
    this._pass(enc, this.pipelines.mr, this.bindGroups.mr, this.textures.mr);
    this._pass(enc, this.pipelines.final, this.bindGroups.final, this.textures.out);
    return this.textures.out;
  }

  _destroyTextures() {
    destroyTextureSet(this.textures);
    this.textures = null;
    this.bindGroups = null;
  }
  destroy() {
    this._destroyTextures();
    this.pipelines = null;
    this.hiTex = null;
    this.hiW = 0; this.hiH = 0; this.dW = 0; this.dH = 0;
    this.bypassed = false;
    this.lastPlan = null;
  }
}

function destroyTextureSet(textures) {
  if (!textures) return;
  for (const texture of new Set(Object.values(textures))) {
    try { texture?.destroy?.(); } catch {}
  }
}

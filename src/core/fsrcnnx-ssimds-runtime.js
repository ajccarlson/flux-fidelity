// fsrcnnx-ssimds-runtime.js — builds + runs the SSimDownscaler pass chain.
//
// Input:  hiTex (rgba16float, oversized RGB from FSRCNNX recombine)
// Output: a display-res rgba16float RGB texture (perceptual downscale)
//
// Pipelines depend on the downscale ratio (baked into the mean + L2 shaders), so
// they're rebuilt when hi-res or target size changes. Bind groups are cached.

import {
  buildL2Shader, SSIMDS_MR_WGSL, SSIMDS_FINAL_WGSL,
} from "./fsrcnnx-ssimds.js";

const FMT = "rgba16float";

// Direct Catmull-Rom integration grows with the source/output ratio. These are
// per-pass bounds, not source-resolution or total-work ceilings: ratios that
// exceed them use a logarithmic sequence of bounded separable moment passes.
export const SSIMDS_WORK_BUDGET = Object.freeze({
  maxAxisTaps: 129,
  maxMeanTapsPerPixel: 4096,
});
const SSIMDS_MAX_STAGE_RATIO = (SSIMDS_WORK_BUDGET.maxAxisTaps - 1) / 4;

function requireDimension(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return value;
}

function checkedProduct(values, label) {
  let result = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0 ||
        result > Number.MAX_SAFE_INTEGER / value) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
    result *= value;
  }
  return result;
}

function checkedSum(values, label) {
  let result = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 ||
        result > Number.MAX_SAFE_INTEGER - value) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
    result += value;
  }
  return result;
}

function tapCount(ratio, label) {
  const scaled = ratio * 4;
  const taps = Math.ceil(scaled) + 1;
  if (!Number.isFinite(ratio) || ratio < 1 ||
      !Number.isSafeInteger(taps) || taps <= 0) {
    throw new RangeError(`${label} tap count exceeds safe integer arithmetic`);
  }
  return taps;
}

function buildMomentStages(hiW, hiH, dW, dH) {
  const stages = [];
  let inputWidth = hiW;
  let inputHeight = hiH;
  while (inputWidth !== dW || inputHeight !== dH) {
    const outputWidth = inputWidth / dW <= SSIMDS_MAX_STAGE_RATIO
      ? dW
      : Math.ceil(inputWidth / SSIMDS_MAX_STAGE_RATIO);
    const outputHeight = inputHeight / dH <= SSIMDS_MAX_STAGE_RATIO
      ? dH
      : Math.ceil(inputHeight / SSIMDS_MAX_STAGE_RATIO);
    if (outputWidth > inputWidth || outputHeight > inputHeight ||
        (outputWidth === inputWidth && outputHeight === inputHeight)) {
      throw new RangeError("SSimDownscaler could not construct a bounded moment plan");
    }
    const ratioX = inputWidth / outputWidth;
    const ratioY = inputHeight / outputHeight;
    const tapsX = tapCount(ratioX, "horizontal stage");
    const tapsY = tapCount(ratioY, "vertical stage");
    if (tapsX > SSIMDS_WORK_BUDGET.maxAxisTaps ||
        tapsY > SSIMDS_WORK_BUDGET.maxAxisTaps) {
      throw new RangeError("SSimDownscaler moment stage exceeds its per-pass tap bound");
    }
    stages.push(Object.freeze({
      inputWidth,
      inputHeight,
      outputWidth,
      outputHeight,
      ratioX,
      ratioY,
      tapsX,
      tapsY,
    }));
    inputWidth = outputWidth;
    inputHeight = outputHeight;
  }
  return Object.freeze(stages);
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
  const tapsX = tapCount(ratioX, "horizontal direct pass");
  const tapsY = tapCount(ratioY, "vertical direct pass");
  const meanTapsPerPixel = checkedSum([tapsX, tapsY], "separable mean taps per pixel");
  const outputPixels = checkedProduct([dW, dH], "SSimDownscaler output area");
  const oneMomentSamples = checkedSum([
    checkedProduct([hiW, dH, tapsY], "direct vertical moment samples"),
    checkedProduct([outputPixels, tapsX], "direct horizontal moment samples"),
  ], "direct separable moment samples");
  const directTerms = [
    checkedProduct([oneMomentSamples, 2], "direct first and second moment samples"),
    checkedProduct([outputPixels, 28], "SSimDownscaler reconstruction samples"),
  ];
  const directEstimatedTextureSamples = checkedSum(
    directTerms,
    "direct SSimDownscaler sample estimate",
  );
  const direct = tapsX <= SSIMDS_WORK_BUDGET.maxAxisTaps &&
    tapsY <= SSIMDS_WORK_BUDGET.maxAxisTaps &&
    meanTapsPerPixel <= SSIMDS_WORK_BUDGET.maxMeanTapsPerPixel;
  const stages = direct ? Object.freeze([]) : buildMomentStages(hiW, hiH, dW, dH);
  const progressiveTerms = stages.flatMap((stage) => {
    const verticalPixels = checkedProduct(
      [stage.inputWidth, stage.outputHeight],
      "moment-stage vertical area",
    );
    const horizontalPixels = checkedProduct(
      [stage.outputWidth, stage.outputHeight],
      "moment-stage horizontal area",
    );
    const oneMoment = checkedSum([
      checkedProduct([verticalPixels, stage.tapsY], "moment-stage vertical samples"),
      checkedProduct([horizontalPixels, stage.tapsX], "moment-stage horizontal samples"),
    ], "moment-stage samples");
    return [checkedProduct([oneMoment, 2], "two-moment stage samples")];
  });
  const estimatedTextureSamples = direct
    ? directEstimatedTextureSamples
    : checkedSum([
      ...progressiveTerms,
      checkedProduct([outputPixels, 28], "SSimDownscaler reconstruction samples"),
    ], "multistage SSimDownscaler sample estimate");
  return {
    hiW, hiH, dW, dH, ratioX, ratioY, tapsX, tapsY, meanTapsPerPixel,
    directEstimatedTextureSamples,
    estimatedTextureSamples,
    direct,
    path: direct ? "direct" : "multistage",
    stages,
    allowed: true,
    reason: null,
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
    this.path = null;
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
    if (!hiTex || typeof hiTex.createView !== "function") {
      throw new TypeError("hiTex must be a GPU texture");
    }
    for (const [value, label] of [
      [hiW, "hiW"], [hiH, "hiH"], [dW, "dW"], [dH, "dH"],
    ]) {
      requireDimension(value, label);
    }
    if (dW > hiW || dH > hiH) {
      throw new RangeError("SSimDownscaler only accepts output dimensions no larger than its input");
    }
    const maxDim = Number(this.device?.limits?.maxTextureDimension2D);
    if (Number.isFinite(maxDim) && maxDim > 0 &&
        [hiW, hiH, dW, dH].some((dimension) => dimension > maxDim)) {
      throw new RangeError(`SSimDownscaler dimensions exceed device limit ${maxDim}`);
    }
    const plan = estimateSsimWork(hiW, hiH, dW, dH);

    const sizesSame = hiW === this.hiW && hiH === this.hiH && dW === this.dW && dH === this.dH;
    if (sizesSame && this.hiTex === hiTex && this.bindGroups) return true;

    const { ratioX, ratioY } = plan;
    const replaceOwned = !sizesSame || !this.pipelines || !this.textures ||
      this.path !== plan.path;
    let candidatePipelines = this.pipelines;
    let candidateTextures = this.textures;
    const createdTextures = [];
    try {
      if (replaceOwned) {
        const makeTexture = (w, h, label) => {
          const texture = this._tex(w, h, label);
          createdTextures.push(texture);
          return texture;
        };
        if (plan.path === "direct") {
          const horizontal = this._renderPipeline(
            buildL2Shader(0, ratioX, false),
          );
          candidatePipelines = {
            verticalMean: this._renderPipeline(
              buildL2Shader(1, ratioY, false),
            ),
            verticalL2: this._renderPipeline(
              buildL2Shader(1, ratioY, true),
            ),
            horizontal,
            mr: this._renderPipeline(SSIMDS_MR_WGSL),
            final: this._renderPipeline(SSIMDS_FINAL_WGSL),
          };
          candidateTextures = {
            vertical: makeTexture(hiW, dH, "ssimds-moment-v"),
            mean: makeTexture(dW, dH, "ssimds-mean"),
            l2: makeTexture(dW, dH, "ssimds-l2"),
            mr: makeTexture(dW, dH, "ssimds-mr"),
            out: makeTexture(dW, dH, "ssimds-out"),
          };
        } else {
          candidatePipelines = {
            stages: plan.stages.map((stage, index) => {
              const verticalMean = this._renderPipeline(
                buildL2Shader(1, stage.ratioY, false),
              );
              const verticalL2 = index === 0
                ? this._renderPipeline(buildL2Shader(1, stage.ratioY, true))
                : verticalMean;
              return {
                verticalMean,
                verticalL2,
                horizontal: this._renderPipeline(
                  buildL2Shader(0, stage.ratioX, false),
                ),
              };
            }),
            mr: this._renderPipeline(SSIMDS_MR_WGSL),
            final: this._renderPipeline(SSIMDS_FINAL_WGSL),
          };
          candidateTextures = {
            stages: plan.stages.map((stage, index) => ({
              vertical: makeTexture(
                stage.inputWidth,
                stage.outputHeight,
                `ssimds-moment-v-${index}`,
              ),
              mean: makeTexture(
                stage.outputWidth,
                stage.outputHeight,
                `ssimds-mean-${index}`,
              ),
              l2: makeTexture(
                stage.outputWidth,
                stage.outputHeight,
                `ssimds-l2-${index}`,
              ),
            })),
            mr: makeTexture(dW, dH, "ssimds-mr"),
            out: makeTexture(dW, dH, "ssimds-out"),
          };
        }
      }

      const s = this.sampler;
      const bg = (pipe, entries) =>
        this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
      let candidateBindGroups;
      if (plan.path === "direct") {
        candidateBindGroups = {
          verticalMean: bg(candidatePipelines.verticalMean, [
            { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
          ]),
          horizontalMean: bg(candidatePipelines.horizontal, [
            { binding: 0, resource: s },
            { binding: 1, resource: candidateTextures.vertical.createView() },
          ]),
          verticalL2: bg(candidatePipelines.verticalL2, [
            { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
          ]),
          horizontalL2: bg(candidatePipelines.horizontal, [
            { binding: 0, resource: s },
            { binding: 1, resource: candidateTextures.vertical.createView() },
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
      } else {
        let meanSource = hiTex;
        let l2Source = hiTex;
        const stageBindGroups = plan.stages.map((stage, index) => {
          const pipes = candidatePipelines.stages[index];
          const textures = candidateTextures.stages[index];
          const groups = {
            verticalMean: bg(pipes.verticalMean, [
              { binding: 0, resource: s },
              { binding: 1, resource: meanSource.createView() },
            ]),
            horizontalMean: bg(pipes.horizontal, [
              { binding: 0, resource: s },
              { binding: 1, resource: textures.vertical.createView() },
            ]),
            verticalL2: bg(pipes.verticalL2, [
              { binding: 0, resource: s },
              { binding: 1, resource: l2Source.createView() },
            ]),
            horizontalL2: bg(pipes.horizontal, [
              { binding: 0, resource: s },
              { binding: 1, resource: textures.vertical.createView() },
            ]),
          };
          meanSource = textures.mean;
          l2Source = textures.l2;
          return groups;
        });
        candidateBindGroups = {
          stages: stageBindGroups,
          mr: bg(candidatePipelines.mr, [
            { binding: 0, resource: s },
            { binding: 1, resource: meanSource.createView() },
            { binding: 2, resource: l2Source.createView() },
          ]),
          final: bg(candidatePipelines.final, [
            { binding: 0, resource: s },
            { binding: 1, resource: meanSource.createView() },
            { binding: 2, resource: candidateTextures.mr.createView() },
          ]),
        };
      }

      const oldTextures = this.textures;
      this.lastPlan = plan;
      this.hiW = hiW; this.hiH = hiH; this.dW = dW; this.dH = dH; this.hiTex = hiTex;
      this.path = plan.path;
      this.pipelines = candidatePipelines;
      this.textures = candidateTextures;
      this.bindGroups = candidateBindGroups;
      if (replaceOwned) destroyTextureSet(oldTextures);
      return true;
    } catch (error) {
      destroyTextureSet(createdTextures);
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
    if (!this.pipelines || !this.textures || !this.bindGroups || this.hiTex !== hiTex) {
      throw new Error("SSimDownscaler.run requires a successful prepare for the same input texture");
    }
    let meanTexture;
    if (this.path === "direct") {
      this._pass(
        enc,
        this.pipelines.verticalMean,
        this.bindGroups.verticalMean,
        this.textures.vertical,
      );
      this._pass(
        enc,
        this.pipelines.horizontal,
        this.bindGroups.horizontalMean,
        this.textures.mean,
      );
      this._pass(
        enc,
        this.pipelines.verticalL2,
        this.bindGroups.verticalL2,
        this.textures.vertical,
      );
      this._pass(
        enc,
        this.pipelines.horizontal,
        this.bindGroups.horizontalL2,
        this.textures.l2,
      );
      meanTexture = this.textures.mean;
    } else if (this.path === "multistage") {
      for (let index = 0; index < this.lastPlan.stages.length; index++) {
        const pipes = this.pipelines.stages[index];
        const textures = this.textures.stages[index];
        const groups = this.bindGroups.stages[index];
        this._pass(enc, pipes.verticalMean, groups.verticalMean, textures.vertical);
        this._pass(enc, pipes.horizontal, groups.horizontalMean, textures.mean);
        this._pass(enc, pipes.verticalL2, groups.verticalL2, textures.vertical);
        this._pass(enc, pipes.horizontal, groups.horizontalL2, textures.l2);
      }
      meanTexture = this.textures.stages.at(-1)?.mean || null;
    } else {
      throw new Error("SSimDownscaler has no prepared execution path");
    }
    this._pass(enc, this.pipelines.mr, this.bindGroups.mr, this.textures.mr);
    this._pass(enc, this.pipelines.final, this.bindGroups.final, this.textures.out);
    if (!meanTexture) throw new Error("SSimDownscaler prepared no mean texture");
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
    this.path = null;
    this.lastPlan = null;
  }
}

function destroyTextureSet(textures) {
  if (!textures) return;
  const found = new Set();
  const visited = new Set();
  const visit = (value) => {
    if (!value || (typeof value !== "object" && typeof value !== "function") ||
        visited.has(value)) return;
    visited.add(value);
    if (typeof value.destroy === "function") {
      found.add(value);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    for (const item of Object.values(value)) visit(item);
  };
  visit(textures);
  for (const texture of found) {
    try { texture?.destroy?.(); } catch {}
  }
}

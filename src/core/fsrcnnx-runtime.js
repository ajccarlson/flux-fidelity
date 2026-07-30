// fsrcnnx-runtime.js
// Builds a validated WebGPU compute chain from an FSRCNNX manifest and WGSL.

import {
  DEFAULT_MODEL_WORKING_SET_BYTES,
  passSaves,
  preflightModelDimensions,
  validateModelBundle,
} from "./fsrcnnx-model-bundle.js";

export class FsrcnnxModel {
  constructor(device, manifest, wgslSource, options = {}) {
    if (!device) throw new Error("FSRCNNX requires a GPU device");
    const bundle = validateModelBundle("fsrcnnx", manifest, wgslSource, {
      expectedName: options.expectedName,
      deviceLimits: device.limits,
    });
    this.device = device;
    this.manifest = bundle.manifest;
    this.scale = bundle.scale;
    this.entries = bundle.entries;
    this.maxWorkingSetBytes = options.maxWorkingSetBytes ?? DEFAULT_MODEL_WORKING_SET_BYTES;
    this.pipelines = [];
    this.textures = new Map();
    this.lumaW = 0;
    this.lumaH = 0;
    this.lumaTexture = null;
    this.outputTexture = null;
    this.outputView = null;
    this.views = null;
    this.bindGroups = null;
    this.allocationPlan = null;
    this.destroyed = false;
    this._warmPromise = null;
  }

  _assertLive() {
    if (this.destroyed) throw new Error("FSRCNNX model has been destroyed");
  }

  buildPipelines() {
    this._assertLive();
    if (this.pipelines.length === this.manifest.passes.length) return this.pipelines;
    const candidates = new Array(this.manifest.passes.length);
    for (let index = 0; index < this.manifest.passes.length; index++) {
      const pass = this.manifest.passes[index];
      const module = this.device.createShaderModule({
        label: `fsrcnnx-${this.manifest.name}-p${index}`,
        code: this.entries.get(index),
      });
      candidates[index] = this.device.createComputePipeline({
        label: `fsrcnnx-p${index}-${sanitize(pass.desc)}`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
    // Publish only a complete pipeline generation.
    this.pipelines = candidates;
    return candidates;
  }

  // createComputePipeline blocks the caller while the driver compiles, and
  // allocate() runs inside the frame callback — so the first frame of a model
  // paid for every pass at once. FSRCNNX High is 54 passes of a ~380 KB shader.
  // Warming through the async form off the critical path leaves buildPipelines()
  // as a correctness fallback that then finds the work already done.
  async warmPipelines() {
    this._assertLive();
    if (this.pipelines.length === this.manifest.passes.length) return this.pipelines;
    if (!this._warmPromise) {
      this._warmPromise = this._compileAsync().finally(() => { this._warmPromise = null; });
    }
    return this._warmPromise;
  }

  async _compileAsync() {
    // Not every device implements the async form — notably the fakes used by the
    // unit suites — so fall back rather than making warming a hard requirement.
    if (typeof this.device.createComputePipelineAsync !== "function") {
      return this.buildPipelines();
    }
    const candidates = await Promise.all(this.manifest.passes.map((pass, index) => {
      const module = this.device.createShaderModule({
        label: `fsrcnnx-${this.manifest.name}-p${index}`,
        code: this.entries.get(index),
      });
      return this.device.createComputePipelineAsync({
        label: `fsrcnnx-p${index}-${sanitize(pass.desc)}`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }));
    // Destruction or a synchronous build may have overtaken this compile. Publish
    // only when nothing else has, so a generation is never half-replaced.
    if (this.destroyed) return [];
    if (this.pipelines.length !== this.manifest.passes.length) this.pipelines = candidates;
    return this.pipelines;
  }

  preflight(lumaW, lumaH, options = {}) {
    this._assertLive();
    return preflightModelDimensions("fsrcnnx", this.manifest, lumaW, lumaH, {
      deviceLimits: this.device.limits,
      maxWorkingSetBytes: options.maxWorkingSetBytes ?? this.maxWorkingSetBytes,
    });
  }

  allocate(lumaW, lumaH, lumaTexture) {
    this._assertLive();
    if (!lumaTexture || typeof lumaTexture.createView !== "function") {
      throw new Error("FSRCNNX requires a valid luma texture");
    }
    if (lumaTexture.width != null && lumaTexture.width !== lumaW) throw new Error("FSRCNNX luma texture width mismatch");
    if (lumaTexture.height != null && lumaTexture.height !== lumaH) throw new Error("FSRCNNX luma texture height mismatch");

    const plan = this.preflight(lumaW, lumaH);
    const pipelines = this.buildPipelines();
    const saveResources = plan.resources.filter((resource) => resource.name !== "OUTPUT");
    const completeOwnedSet = this.outputTexture && this.textures.size === saveResources.length &&
      saveResources.every((resource) => this.textures.has(resource.name));
    const sameSize = completeOwnedSet && lumaW === this.lumaW && lumaH === this.lumaH;
    if (sameSize && this.lumaTexture === lumaTexture &&
        this.bindGroups?.length === this.manifest.passes.length) return plan;

    let candidateTextures = this.textures;
    let candidateOutput = this.outputTexture;
    const createdTextures = [];
    const replacingOwnedSet = !sameSize;
    try {
      if (replacingOwnedSet) {
        candidateTextures = new Map();
        for (const resource of saveResources) {
          const texture = this.device.createTexture({
            label: `fsrcnnx-tex-${resource.name}`,
            size: { width: resource.width, height: resource.height },
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          });
          createdTextures.push(texture);
          candidateTextures.set(resource.name, texture);
        }
        candidateOutput = this.device.createTexture({
          label: "fsrcnnx-out-luma",
          size: { width: plan.outputWidth, height: plan.outputHeight },
          format: "rgba16float",
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });
        createdTextures.push(candidateOutput);
      }

      const candidateViews = new Map();
      const viewOf = (name) => {
        const texture = name === "LUMA" ? lumaTexture : candidateTextures.get(name);
        if (!texture || typeof texture.createView !== "function") {
          throw new Error(`FSRCNNX resource ${name} is unavailable`);
        }
        if (!candidateViews.has(texture)) candidateViews.set(texture, texture.createView());
        return candidateViews.get(texture);
      };
      const candidateOutputView = candidateOutput.createView();
      const candidateBindGroups = this.manifest.passes.map((pass, index) => {
        const entries = pass.binds.map((name, binding) => ({ binding, resource: viewOf(name) }));
        // Inputs occupy bindings 0..n-1 and the outputs follow in manifest order.
        // A fused pass writes several of them from one dispatch; the terminal
        // shuffle declares no logical save and writes the model output instead.
        const outputs = passSaves(pass);
        if (!outputs.length) {
          entries.push({ binding: pass.binds.length, resource: candidateOutputView });
        } else {
          outputs.forEach((name, offset) => {
            entries.push({ binding: pass.binds.length + offset, resource: viewOf(name) });
          });
        }
        return this.device.createBindGroup({
          layout: pipelines[index].getBindGroupLayout(0),
          entries,
        });
      });

      const oldTextures = this.textures;
      const oldOutput = this.outputTexture;
      this.lumaW = lumaW;
      this.lumaH = lumaH;
      this.lumaTexture = lumaTexture;
      this.textures = candidateTextures;
      this.outputTexture = candidateOutput;
      this.views = candidateViews;
      this.outputView = candidateOutputView;
      this.bindGroups = candidateBindGroups;
      this.allocationPlan = plan;
      if (replacingOwnedSet) destroyOwnedTextures(oldTextures, oldOutput);
      return plan;
    } catch (error) {
      for (const texture of new Set(createdTextures)) {
        try { texture?.destroy?.(); } catch {}
      }
      throw error;
    }
  }

  run(encoder, lumaTexture) {
    this._assertLive();
    if (!encoder || typeof encoder.beginComputePass !== "function") throw new Error("FSRCNNX requires a command encoder");
    if (!this.lumaW || !this.lumaH) throw new Error("FSRCNNX model must be allocated before run");
    if (this.lumaTexture !== lumaTexture || !this.bindGroups) {
      this.allocate(this.lumaW, this.lumaH, lumaTexture);
    }
    for (let index = 0; index < this.manifest.passes.length; index++) {
      const pass = this.manifest.passes[index];
      const width = pass.kind === "shuffle" ? this.allocationPlan.outputWidth : this.lumaW;
      const height = pass.kind === "shuffle" ? this.allocationPlan.outputHeight : this.lumaH;
      const computePass = encoder.beginComputePass({ label: `fsrcnnx-p${index}` });
      computePass.setPipeline(this.pipelines[index]);
      computePass.setBindGroup(0, this.bindGroups[index]);
      computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      computePass.end();
    }
    return this.outputTexture;
  }

  destroyTextures() {
    destroyOwnedTextures(this.textures, this.outputTexture);
    this.textures = new Map();
    this.outputTexture = null;
    this.outputView = null;
    this.views = null;
    this.bindGroups = null;
    this.lumaTexture = null;
    this.lumaW = 0;
    this.lumaH = 0;
    this.allocationPlan = null;
  }

  resetAllocation() {
    this._assertLive();
    this.destroyTextures();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyTextures();
    this.pipelines = [];
    this.entries = new Map();
    this._warmPromise = null;
    this.destroyed = true;
  }
}

function destroyOwnedTextures(textures, outputTexture) {
  const resources = new Set(textures?.values?.() || []);
  if (outputTexture) resources.add(outputTexture);
  for (const texture of resources) {
    try { texture?.destroy?.(); } catch {}
  }
}

// Pick the right model for a given upscale ratio using the mpv //!WHEN
// thresholds. Validated models always carry finite positive thresholds.
export function selectModel(models, targetW, srcW) {
  const ratio = targetW / srcW;
  const eligible = models
    .filter((model) => ratio > model.manifest.whenThreshold)
    .sort((left, right) => right.manifest.whenThreshold - left.manifest.whenThreshold);
  return eligible[0] || null;
}

function sanitize(value) {
  return value.replace(/[^A-Za-z0-9]+/g, "_");
}

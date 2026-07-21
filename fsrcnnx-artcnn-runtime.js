// fsrcnnx-artcnn-runtime.js — runs a validated transpiled ArtCNN model.

import {
  DEFAULT_MODEL_WORKING_SET_BYTES,
  preflightModelDimensions,
  validateModelBundle,
} from "./fsrcnnx-model-bundle.js";

const PACKED = "rgba16float";

export class ArtCnnModel {
  constructor(device, manifest, wgslSource, options = {}) {
    if (!device) throw new Error("ArtCNN requires a GPU device");
    const bundle = validateModelBundle("artcnn", manifest, wgslSource, {
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
    this.lumaTex = null;
    this.outputTexture = null;
    this.outputView = null;
    this.views = null;
    this.bindGroups = null;
    this.allocationPlan = null;
    this.destroyed = false;
  }

  _assertLive() {
    if (this.destroyed) throw new Error("ArtCNN model has been destroyed");
  }

  buildPipelines() {
    this._assertLive();
    if (this.pipelines.length === this.manifest.passes.length) return this.pipelines;
    const candidates = new Array(this.manifest.passes.length);
    for (let index = 0; index < this.manifest.passes.length; index++) {
      const pass = this.manifest.passes[index];
      const module = this.device.createShaderModule({
        label: `artcnn-p${index}`,
        code: this.entries.get(index),
      });
      candidates[index] = this.device.createComputePipeline({
        label: `artcnn-p${index}-${pass.kind}`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
    // Do not let a synchronous failure leave a partially published generation.
    this.pipelines = candidates;
    return candidates;
  }

  preflight(lumaW, lumaH, options = {}) {
    this._assertLive();
    return preflightModelDimensions("artcnn", this.manifest, lumaW, lumaH, {
      deviceLimits: this.device.limits,
      maxWorkingSetBytes: options.maxWorkingSetBytes ?? this.maxWorkingSetBytes,
    });
  }

  allocate(lumaW, lumaH, lumaTex) {
    this._assertLive();
    if (!lumaTex || typeof lumaTex.createView !== "function") throw new Error("ArtCNN requires a valid luma texture");
    if (lumaTex.width != null && lumaTex.width !== lumaW) throw new Error("ArtCNN luma texture width mismatch");
    if (lumaTex.height != null && lumaTex.height !== lumaH) throw new Error("ArtCNN luma texture height mismatch");

    const plan = this.preflight(lumaW, lumaH);
    const pipelines = this.buildPipelines();
    const saveResources = plan.resources.filter((resource) => resource.name !== "OUTPUT");
    const completeOwnedSet = this.outputTexture && this.textures.size === saveResources.length &&
      saveResources.every((resource) => this.textures.has(resource.name));
    const sameSize = completeOwnedSet && lumaW === this.lumaW && lumaH === this.lumaH;
    if (sameSize && this.lumaTex === lumaTex &&
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
            label: `artcnn-${resource.name}`,
            size: { width: resource.width, height: resource.height },
            format: PACKED,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          });
          createdTextures.push(texture);
          candidateTextures.set(resource.name, texture);
        }
        candidateOutput = this.device.createTexture({
          label: "artcnn-out-luma",
          size: { width: plan.outputWidth, height: plan.outputHeight },
          format: PACKED,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });
        createdTextures.push(candidateOutput);
      }

      const candidateViews = new Map();
      const viewOf = (name) => {
        const texture = name === "LUMA" ? lumaTex : candidateTextures.get(name);
        if (!texture || typeof texture.createView !== "function") {
          throw new Error(`ArtCNN resource ${name} is unavailable`);
        }
        if (!candidateViews.has(texture)) candidateViews.set(texture, texture.createView());
        return candidateViews.get(texture);
      };
      const candidateOutputView = candidateOutput.createView();
      const candidateBindGroups = this.manifest.passes.map((pass, index) => {
        const entries = pass.binds.map((name, binding) => ({ binding, resource: viewOf(name) }));
        entries.push({
          binding: pass.binds.length,
          resource: pass.kind === "d2s" ? candidateOutputView : viewOf(pass.save),
        });
        return this.device.createBindGroup({
          layout: pipelines[index].getBindGroupLayout(0),
          entries,
        });
      });

      const oldTextures = this.textures;
      const oldOutput = this.outputTexture;
      this.lumaW = lumaW;
      this.lumaH = lumaH;
      this.lumaTex = lumaTex;
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

  run(encoder, lumaTex) {
    this._assertLive();
    if (!encoder || typeof encoder.beginComputePass !== "function") throw new Error("ArtCNN requires a command encoder");
    if (!this.lumaW || !this.lumaH) throw new Error("ArtCNN model must be allocated before run");
    if (this.lumaTex !== lumaTex || !this.bindGroups) this.allocate(this.lumaW, this.lumaH, lumaTex);
    for (let index = 0; index < this.manifest.passes.length; index++) {
      const pass = this.manifest.passes[index];
      const dispW = pass.kind === "d2s" ? this.allocationPlan.outputWidth : this.lumaW;
      const dispH = pass.kind === "d2s" ? this.allocationPlan.outputHeight : this.lumaH;
      const computePass = encoder.beginComputePass({ label: `artcnn-p${index}` });
      computePass.setPipeline(this.pipelines[index]);
      computePass.setBindGroup(0, this.bindGroups[index]);
      computePass.dispatchWorkgroups(Math.ceil(dispW / 8), Math.ceil(dispH / 8));
      computePass.end();
    }
    return this.outputTexture;
  }

  _destroyTextures() {
    destroyOwnedTextures(this.textures, this.outputTexture);
    this.textures = new Map();
    this.outputTexture = null;
    this.outputView = null;
    this.views = null;
    this.bindGroups = null;
    this.lumaTex = null;
    this.lumaW = 0;
    this.lumaH = 0;
    this.allocationPlan = null;
  }

  resetAllocation() {
    this._assertLive();
    this._destroyTextures();
  }

  destroy() {
    if (this.destroyed) return;
    this._destroyTextures();
    this.pipelines = [];
    this.entries = new Map();
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

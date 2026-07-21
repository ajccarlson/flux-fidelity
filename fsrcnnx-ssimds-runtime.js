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

export class SsimDownscaler {
  constructor(device) {
    this.device = device;
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.hiW = 0; this.hiH = 0; this.dW = 0; this.dH = 0;
    this.pipelines = null;
    this.textures = null;
    this.bindGroups = null;
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
    const sizesSame = hiW === this.hiW && hiH === this.hiH && dW === this.dW && dH === this.dH;
    if (sizesSame && this.hiTex === hiTex && this.bindGroups) return;
    this.hiW = hiW; this.hiH = hiH; this.dW = dW; this.dH = dH; this.hiTex = hiTex;

    const ratioX = hiW / dW, ratioY = hiH / dH;

    if (!sizesSame || !this.pipelines) {
      this._destroyTextures();
      this.pipelines = {
        mean: this._renderPipeline(buildMeanShader(ratioX, ratioY)),
        l2v: this._renderPipeline(buildL2Shader(1, ratioY)), // vertical, squares input
        l2h: this._renderPipeline(buildL2Shader(0, ratioX)), // horizontal
        mr: this._renderPipeline(SSIMDS_MR_WGSL),
        final: this._renderPipeline(SSIMDS_FINAL_WGSL),
      };
      // intermediate textures
      this.textures = {
        mean: this._tex(dW, dH, "ssimds-mean"),
        l2v: this._tex(hiW, dH, "ssimds-l2v"),   // vertical pass: hi width, display height
        l2: this._tex(dW, dH, "ssimds-l2"),
        mr: this._tex(dW, dH, "ssimds-mr"),
        out: this._tex(dW, dH, "ssimds-out"),
      };
    }

    const s = this.sampler;
    const bg = (pipe, entries) =>
      this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });

    this.bindGroups = {
      mean: bg(this.pipelines.mean, [
        { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
      ]),
      l2v: bg(this.pipelines.l2v, [
        { binding: 0, resource: s }, { binding: 1, resource: hiTex.createView() },
      ]),
      l2h: bg(this.pipelines.l2h, [
        { binding: 0, resource: s }, { binding: 1, resource: this.textures.l2v.createView() },
      ]),
      mr: bg(this.pipelines.mr, [
        { binding: 0, resource: s },
        { binding: 1, resource: this.textures.mean.createView() },
        { binding: 2, resource: this.textures.l2.createView() },
      ]),
      final: bg(this.pipelines.final, [
        { binding: 0, resource: s },
        { binding: 1, resource: this.textures.mean.createView() },
        { binding: 2, resource: this.textures.mr.createView() },
      ]),
    };
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
    this._pass(enc, this.pipelines.mean, this.bindGroups.mean, this.textures.mean);
    this._pass(enc, this.pipelines.l2v, this.bindGroups.l2v, this.textures.l2v);
    this._pass(enc, this.pipelines.l2h, this.bindGroups.l2h, this.textures.l2);
    this._pass(enc, this.pipelines.mr, this.bindGroups.mr, this.textures.mr);
    this._pass(enc, this.pipelines.final, this.bindGroups.final, this.textures.out);
    return this.textures.out;
  }

  _destroyTextures() {
    if (!this.textures) return;
    for (const t of Object.values(this.textures)) t.destroy?.();
    this.textures = null;
  }
  destroy() { this._destroyTextures(); }
}

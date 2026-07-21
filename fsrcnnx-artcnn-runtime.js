// fsrcnnx-artcnn-runtime.js — runs a transpiled ArtCNN model.
//
// Architecture (differs from FSRCNNX): compute passes with PACKED feature
// textures. Conv layers 0-5 output 32 features as a 4x2 block of rgba16f texels
// per source pixel, so those textures are (lumaW*4) x (lumaH*2). conv2d_6 outputs
// one rgba16f texel per pixel at native res (the 4 sub-pixel values). The final
// depth-to-space unpacks to 2x luma (r16float-incompatible-as-storage, so
// rgba16float here too... actually output is the 2x luma we feed to recombine).
//
// Each conv invocation handles ONE source pixel (writes its packed block); the
// d2s invocation handles one OUTPUT pixel. Dispatch sizes differ per pass.

const PACKED = "rgba16float";

export class ArtCnnModel {
  constructor(device, manifest, wgslSource) {
    this.device = device;
    this.manifest = manifest;
    this.scale = manifest.scale || 2;
    this.entries = splitEntries(wgslSource);
    this.pipelines = [];
    this.textures = new Map(); // save-name -> texture
    this.lumaW = 0; this.lumaH = 0;
    this.lumaTex = null;
    this.bindGroups = null;
  }

  buildPipelines() {
    if (this.pipelines.length) return;
    for (const pass of this.manifest.passes) {
      const code = this.entries.get(pass.index);
      if (!code) throw new Error(`ArtCNN: missing WGSL for pass ${pass.index}`);
      const module = this.device.createShaderModule({ label: `artcnn-p${pass.index}`, code });
      this.pipelines[pass.index] = this.device.createComputePipeline({
        label: `artcnn-p${pass.index}-${pass.kind}`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
  }

  // dims for a pass's SAVE texture, given luma size.
  _saveDims(pass) {
    const w = Math.round(this.lumaW * pass.widthMul);
    const h = Math.round(this.lumaH * pass.heightMul);
    return [w, h];
  }

  allocate(lumaW, lumaH, lumaTex) {
    const sameSize = lumaW === this.lumaW && lumaH === this.lumaH && this.textures.size;
    if (sameSize && this.lumaTex === lumaTex && this.bindGroups) return;
    this.lumaW = lumaW; this.lumaH = lumaH; this.lumaTex = lumaTex;
    this.buildPipelines();

    if (!sameSize) {
      this._destroyTextures();
      for (const pass of this.manifest.passes) {
        if (!pass.save) continue;
        const [w, h] = this._saveDims(pass);
        this.textures.set(pass.save, this.device.createTexture({
          label: `artcnn-${pass.save}`,
          size: { width: w, height: h }, format: PACKED,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        }));
      }
      // final 2x luma output (depth-to-space target). The d2s pass has no SAVE
      // (it writes OUTPUT); we make our own output texture at 2x luma.
      this.outputTexture = this.device.createTexture({
        label: "artcnn-out-luma",
        size: { width: lumaW * this.scale, height: lumaH * this.scale }, format: PACKED,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
    }

    const viewCache = new Map();
    const viewOf = (name) => {
      const tex = name === "LUMA" ? lumaTex : this.textures.get(name);
      if (!viewCache.has(tex)) viewCache.set(tex, tex.createView());
      return viewCache.get(tex);
    };
    this.outputView = this.outputTexture.createView();

    this.bindGroups = this.manifest.passes.map((pass) => {
      const pipe = this.pipelines[pass.index];
      const entries = pass.binds.map((b, slot) => ({ binding: slot, resource: viewOf(b) }));
      const outView = pass.kind === "d2s" ? this.outputView : viewOf(pass.save);
      entries.push({ binding: pass.binds.length, resource: outView });
      return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    });
  }

  run(encoder, lumaTex) {
    if (this.lumaTex !== lumaTex || !this.bindGroups) this.allocate(this.lumaW, this.lumaH, lumaTex);
    for (const pass of this.manifest.passes) {
      // dispatch: conv passes iterate over SOURCE pixels (lumaW x lumaH);
      // d2s iterates over OUTPUT pixels (2x luma).
      let dispW, dispH;
      if (pass.kind === "d2s") { dispW = this.lumaW * this.scale; dispH = this.lumaH * this.scale; }
      else { dispW = this.lumaW; dispH = this.lumaH; }
      const cp = encoder.beginComputePass({ label: `artcnn-p${pass.index}` });
      cp.setPipeline(this.pipelines[pass.index]);
      cp.setBindGroup(0, this.bindGroups[pass.index]);
      cp.dispatchWorkgroups(Math.ceil(dispW / 8), Math.ceil(dispH / 8));
      cp.end();
    }
    return this.outputTexture; // 2x luma, luma in .r
  }

  _destroyTextures() {
    for (const t of this.textures.values()) t.destroy?.();
    this.textures.clear();
    this.outputTexture?.destroy?.();
    this.outputTexture = null;
  }
  destroy() { this._destroyTextures(); }
}

function splitEntries(wgslSource) {
  const map = new Map();
  const start = wgslSource.indexOf("//==== ENTRY");
  const prelude = wgslSource.slice(0, start);
  for (const chunk of wgslSource.slice(start).split(/(?=\/\/==== ENTRY)/)) {
    const m = chunk.match(/ENTRY pass(\d+)/);
    if (m) map.set(parseInt(m[1], 10), prelude + chunk);
  }
  return map;
}

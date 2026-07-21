// fsrcnnx-runtime.js
// Builds a WebGPU compute pipeline chain from a transpiled FSRCNNX model
// (.passes.json manifest + .wgsl entries) and runs it on a luma plane.
//
// The model upscales LUMA only. The caller is responsible for:
//   1. extracting luma from the source RGB frame (see lumaExtract shader),
//   2. running this chain,
//   3. recombining upscaled luma with chroma (see recombine shader).
//
// Texture conventions:
//   LUMA + intermediates: rgba32float (feature/model passes write 4 channels;
//   the luma input occupies .x of an rgba32float so all conv inputs share a type).
//   The aggregation output is r32float (single upscaled luma channel).

export class FsrcnnxModel {
  constructor(device, manifest, wgslSource) {
    this.device = device;
    this.manifest = manifest;
    this.scale = manifest.passes.find((p) => p.kind === "shuffle")?.widthMul || 2;
    this.entries = splitEntries(wgslSource); // {name -> wgsl source for that pass}
    this.pipelines = []; // one per pass, built lazily in prepare()
    this.textures = new Map(); // save-name -> GPUTexture (allocated per size)
    this.lumaW = 0;
    this.lumaH = 0;
  }

  // Build compute pipelines once (size-independent).
  buildPipelines() {
    if (this.pipelines.length) return;
    this.manifest.passes.forEach((pass, i) => {
      const entryName = sanitize(pass.desc || `pass${i}`);
      const src = this.entries.get(pass.index);
      if (!src) throw new Error(`missing WGSL for pass ${pass.index} (${pass.desc})`);
      const module = this.device.createShaderModule({
        label: `fsrcnnx-${this.manifest.name}-p${pass.index}`,
        code: src,
      });
      const pipeline = this.device.createComputePipeline({
        label: `fsrcnnx-p${pass.index}-${entryName}`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines[pass.index] = pipeline;
    });
  }

  // (Re)allocate intermediate textures + cached bind groups for a luma size.
  // lumaTexture identity is stable across frames (we overwrite its contents),
  // so every bind group can be built once here and reused each frame.
  allocate(lumaW, lumaH, lumaTexture) {
    const sameSize = lumaW === this.lumaW && lumaH === this.lumaH && this.textures.size;
    if (sameSize && this.lumaTexture === lumaTexture && this.bindGroups) return;
    this.lumaW = lumaW;
    this.lumaH = lumaH;
    this.lumaTexture = lumaTexture;
    if (!sameSize) this.destroyTextures();

    const dev = this.device;
    // mpv stores FSRCNNX intermediates at 16-bit half float. Matching that
    // halves bandwidth vs rgba32float across all ~30 passes — the main perf win.
    if (!sameSize) {
      const needNames = new Set();
      for (const p of this.manifest.passes) {
        if (p.save && p.kind !== "shuffle") needNames.add(p.save);
      }
      for (const name of needNames) {
        this.textures.set(
          name,
          dev.createTexture({
            label: `fsrcnnx-tex-${name}`,
            size: { width: lumaW, height: lumaH },
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          })
        );
      }
      this.outputTexture = dev.createTexture({
        label: "fsrcnnx-out-luma",
        size: { width: lumaW * this.scale, height: lumaH * this.scale },
        format: "rgba16float", // r16float is not a writable storage format; use rgba16float, luma in .r
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
    }

    // Pre-create all views once.
    this.views = new Map();
    const viewOf = (name) => {
      const tex = name === "LUMA" ? lumaTexture : this.textures.get(name);
      if (!this.views.has(tex)) this.views.set(tex, tex.createView());
      return this.views.get(tex);
    };
    this.outputView = this.outputTexture.createView();

    // Build + cache one bind group per pass (static for this luma size).
    this.buildPipelines();
    this.bindGroups = this.manifest.passes.map((pass) => {
      const pipeline = this.pipelines[pass.index];
      const entries = pass.binds.map((b, slot) => ({ binding: slot, resource: viewOf(b) }));
      const outView = pass.kind === "shuffle" ? this.outputView : viewOf(pass.save);
      entries.push({ binding: pass.binds.length, resource: outView });
      return this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    });
  }

  // Run the chain. Bind groups are cached; this only records dispatches.
  run(encoder, lumaTexture) {
    if (this.lumaTexture !== lumaTexture || !this.bindGroups) {
      this.allocate(this.lumaW, this.lumaH, lumaTexture);
    }
    for (const pass of this.manifest.passes) {
      const [w, h] =
        pass.kind === "shuffle"
          ? [this.lumaW * this.scale, this.lumaH * this.scale]
          : [this.lumaW, this.lumaH];
      const cpass = encoder.beginComputePass({ label: `fsrcnnx-p${pass.index}` });
      cpass.setPipeline(this.pipelines[pass.index]);
      cpass.setBindGroup(0, this.bindGroups[pass.index]);
      cpass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      cpass.end();
    }
    return this.outputTexture;
  }

  destroyTextures() {
    for (const t of this.textures.values()) t.destroy?.();
    this.textures.clear();
    this.outputTexture?.destroy?.();
    this.outputTexture = null;
  }

  destroy() {
    this.destroyTextures();
  }
}

// Pick the right model for a given upscale ratio using the mpv //!WHEN
// thresholds: a model fires when target/source > its threshold. With x2/x3/x4
// loaded, choose the largest scale whose threshold the ratio clears.
export function selectModel(models, targetW, srcW) {
  const ratio = targetW / srcW;
  // models sorted by threshold ascending; pick the highest threshold <= ratio
  const eligible = models
    .filter((m) => ratio > m.manifest.whenThreshold)
    .sort((a, b) => b.manifest.whenThreshold - a.manifest.whenThreshold);
  return eligible[0] || null; // null => source already large enough; skip upscaling
}

// ---- WGSL splitting -------------------------------------------------------
// The transpiler concatenates a shared prelude + per-pass entries marked with
// `//==== ENTRY passN ...`. Each runtime pipeline needs prelude + one entry.
function splitEntries(wgslSource) {
  const map = new Map();
  const marker = "//==== ENTRY";
  const start = wgslSource.indexOf(marker);
  const prelude = wgslSource.slice(0, start);
  const chunks = wgslSource.slice(start).split(/(?=\/\/==== ENTRY)/);
  for (const chunk of chunks) {
    const m = chunk.match(/ENTRY pass(\d+)/);
    if (!m) continue;
    map.set(parseInt(m[1], 10), prelude + chunk);
  }
  return map;
}

function sanitize(s) {
  return s.replace(/[^A-Za-z0-9]+/g, "_");
}

#!/usr/bin/env node
/*
 * transpile-artcnn.js — ArtCNN (C-series) mpv-GLSL → WGSL compute + JSON manifest.
 *
 * ArtCNN differs from FSRCNNX: it's a COMPUTE-shader CNN with packed feature
 * storage. Each conv layer outputs 32 features as a 4x2 block of rgba16f texels
 * per source pixel (//!WIDTH LUMA.w 4*, //!HEIGHT LUMA.h 2*). Middle layers read
 * those 8 vec4 feature maps via texelFetch at (4,2) strides, do mat4 (32->32)
 * accumulation per output channel-group, ReLU, and store another 4x2 block.
 *
 * This transpiler emits, per file:
 *   <out>/<name>.artcnn.json  — ordered pass descriptors
 *   <out>/<name>.artcnn.wgsl  — one @compute entry per pass
 *
 * Port decisions (see chat): no workgroup shared memory in v1 (each invocation
 * does its own textureLoads — simpler, definitely correct, fine on strong GPUs);
 * f32 accumulation + rgba16float storage (the GLSL's f16 is a speed choice, f32
 * accum is numerically >= and avoids the shader-f16 device feature).
 *
 * Pass shapes recognised:
 *   L0  : 1 input (LUMA scalar), 8 results, weights are `V4(...) * inp_0_x_y`
 *   Lk  : 8 inputs (packed feat), 8 results, weights are `M4(...) * inp_k_x_y`,
 *         ReLU iff stores use max(result, V4(0.0))
 *   skip: 2 binds summed into inp[] before matmul (conv2d_5 + conv2d)
 *   d2s : depth-to-space, fixed template (unpack 4x2 features -> 2x luma)
 */
"use strict";
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const outDir = outIdx >= 0 ? args[outIdx + 1] : "./model";
const inputs = args.filter((a, i) => !a.startsWith("--") && i !== outIdx + 1);
if (!inputs.length) { console.error("usage: node transpile-artcnn.js <file.glsl>... --out ./model"); process.exit(1); }
fs.mkdirSync(outDir, { recursive: true });

const num = (s) => { const v = parseFloat(s); if (!Number.isFinite(v)) throw new Error(`bad number: ${s}`); return v; };
const splitFloats = (inner) => inner.split(",").map((x) => x.trim()).filter(Boolean).map(num);

function splitPasses(src) {
  const lines = src.split(/\r?\n/);
  const passes = []; let cur = null;
  for (const line of lines) {
    if (line.startsWith("//!DESC")) { if (cur) passes.push(cur); cur = { header: [], body: [] }; }
    if (!cur) continue;
    if (line.startsWith("//!")) cur.header.push(line); else cur.body.push(line);
  }
  if (cur) passes.push(cur);
  return passes;
}

function parseHeader(h) {
  const o = { desc: null, binds: [], save: null, components: 4, widthMul: 1, heightMul: 1, compute: null, when: null };
  for (const l of h) {
    const m = l.match(/^\/\/!(\w+)\s*(.*)$/); if (!m) continue;
    const [, k, rest] = m;
    if (k === "DESC") o.desc = rest.trim();
    else if (k === "BIND") o.binds.push(rest.trim());
    else if (k === "SAVE") o.save = rest.trim();
    else if (k === "COMPONENTS") o.components = parseInt(rest) || 4;
    else if (k === "COMPUTE") o.compute = rest.trim().split(/\s+/).map(Number); // [bw,bh,wgx,wgy]
    else if (k === "WHEN") o.when = rest.trim();
    else if (k === "WIDTH") { const mm = rest.match(/(\d+(?:\.\d+)?)\s*\*/); if (mm) o.widthMul = num(mm[1]); }
    else if (k === "HEIGHT") { const mm = rest.match(/(\d+(?:\.\d+)?)\s*\*/); if (mm) o.heightMul = num(mm[1]); }
  }
  return o;
}

// Parse one conv pass body into structured data.
function parseConv(body) {
  const text = body.join("\n");
  const out = { results: [], inputStrides: [], reluOnStore: false, isDepthToSpace: false,
                numFeatIn: 0, biases: [], skipSum: false, terms: [], numResults: 0 };

  // depth-to-space detection
  if (/imageStore/.test(text) && /Depth-To-Space|gather|gl_GlobalInvocationID/.test(text) && !/result0/.test(text)) {
    out.isDepthToSpace = true; return out;
  }
  // Some d2s passes do use result vars; detect via DESC handled by caller. Fallback:
  if (!/result0/.test(text)) { out.isDepthToSpace = true; return out; }

  // input loads: inp[k][y][x] = V4(SRC_mul * texelFetch(SRC_raw, input_base + ivec2(a,b),0) [+ ...]);
  // determine how many feature inputs (max k) and whether two sources are summed.
  const loadRe = /inp\[(\d+)\]\[y\]\[x\]\s*=\s*[VF]4?\(([^;]*)\);/g;
  let lm; const seen = new Set();
  while ((lm = loadRe.exec(text))) {
    seen.add(parseInt(lm[1], 10));
    if (/\+/.test(lm[2]) && /_raw.*\+.*_raw/s.test(lm[2])) out.skipSum = true;
  }
  out.numFeatIn = seen.size || 1;

  // biases: `V4 resultN = V4(....);`
  const biasRe = /[VF]4?\s+result(\d+)\s*=\s*[VF]4?\(([^)]*)\)\s*;/g;
  let bm; const biasMap = {};
  while ((bm = biasRe.exec(text))) biasMap[parseInt(bm[1], 10)] = splitFloats(bm[2]);
  const numResults = Object.keys(biasMap).length;
  for (let r = 0; r < numResults; r++) out.biases[r] = biasMap[r] || [0, 0, 0, 0];

  // accumulation terms per result:
  //  L0 scalar:  resultR += V4(w0,w1,w2,w3) * inp_0_X_Y;        (F scalar input)
  //  Lk matrix:  resultR += M4(16 weights) * inp_K_X_Y;
  // inp var name: inp_<feat>_<dx>_<dy>
  const terms = []; // {r, feat, dx, dy, kind:'vec'|'mat', weights}
  const vecRe = /result(\d+)\s*\+=\s*[V]4?\(([^)]*)\)\s*\*\s*inp_(\d+)_(\d+)_(\d+)\s*;/g;
  let vm;
  while ((vm = vecRe.exec(text))) {
    terms.push({ r: +vm[1], weights: splitFloats(vm[2]), feat: +vm[3], dx: +vm[4], dy: +vm[5], kind: "vec" });
  }
  const matRe = /result(\d+)\s*\+=\s*M4?\(([^)]*)\)\s*\*\s*inp_(\d+)_(\d+)_(\d+)\s*;/g;
  let mm2;
  while ((mm2 = matRe.exec(text))) {
    terms.push({ r: +mm2[1], weights: splitFloats(mm2[2]), feat: +mm2[3], dx: +mm2[4], dy: +mm2[5], kind: "mat" });
  }
  out.terms = terms;
  out.numResults = numResults;

  // ReLU if stores wrap result in max(result, V4(0.0))
  out.reluOnStore = /,\s*max\(result\d+,\s*[VF]4?\(0\.0\)\)/.test(text);

  return out;
}

// ---- WGSL emission --------------------------------------------------------
function fmtF(x){ let s=Number(x).toPrecision(8); if(!/[.eE]/.test(s)) s+=".0"; return s; }
const vec4f = (a)=>`vec4f(${[0,1,2,3].map(i=>fmtF(a[i]||0)).join(", ")})`;
const mat4f = (a)=>`mat4x4f(${a.map(fmtF).join(", ")})`;

// kernel offset for tap index (dx,dy in 0..2) -> pixel offset (dx-1, dy-1)
function emitConvPass(pass, idx) {
  const h = pass.header, b = pass.body, parsed = pass.parsed;
  const binds = h.binds;
  let w = `// ---- ARTCNN PASS ${idx}: ${h.desc} ----\n`;
  w += `// binds: ${binds.join(", ")}  save=${h.save}  widthMul=${h.widthMul} heightMul=${h.heightMul}\n`;

  if (parsed.isDepthToSpace) {
    // Depth-to-space: conv2d_6 holds 4 sub-pixel values per native pixel in one
    // rgba texel. For 2x output pixel (ox,oy): source pixel = floor((ox,oy)/2),
    // sub-index i0 = (ox%2, oy%2), output luma = texel[i0.y*2 + i0.x].
    // (Matches GLSL: f0=fract(pos*size); i0=ivec2(f0*2); result.x=tex(...)[i0.y*2+i0.x].)
    const src = binds[0];
    w += `@group(0) @binding(0) var t_${src} : texture_2d<f32>;\n`;
    w += `@group(0) @binding(1) var outTex : texture_storage_2d<rgba16float, write>;\n`;
    w += `@compute @workgroup_size(8,8)\n`;
    w += `fn main(@builtin(global_invocation_id) gid : vec3u) {\n`;
    w += `  let dims = textureDimensions(outTex);\n`;
    w += `  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\n`;
    w += `  let sp = vec2i(i32(gid.x) / 2, i32(gid.y) / 2);\n`;
    w += `  let i0 = vec2i(i32(gid.x) % 2, i32(gid.y) % 2);\n`;
    w += `  let t = textureLoad(t_${src}, sp, 0);\n`;
    w += `  let ch = i0.y * 2 + i0.x;\n`;
    w += `  var v = t.x;\n`;
    w += `  if (ch == 1) { v = t.y; }\n`;
    w += `  if (ch == 2) { v = t.z; }\n`;
    w += `  if (ch == 3) { v = t.w; }\n`;
    w += `  textureStore(outTex, vec2i(i32(gid.x), i32(gid.y)), vec4f(clamp(v, 0.0, 1.0), 0.0, 0.0, 1.0));\n`;
    w += `}\n`;
    return { wgsl: w, kind: "d2s" };
  }

  // conv pass. Inputs bound 0..n-1, output storage at n.
  binds.forEach((bd, i) => { w += `@group(0) @binding(${i}) var t_${bd} : texture_2d<f32>;\n`; });
  w += `@group(0) @binding(${binds.length}) var outTex : texture_storage_2d<rgba16float, write>;\n`;
  w += `@compute @workgroup_size(8,8)\n`;
  w += `fn main(@builtin(global_invocation_id) gid : vec3u) {\n`;
  // each invocation handles ONE source pixel -> writes a 4x2 packed block of features
  w += `  let sdim = vec2i(i32(textureDimensions(t_${binds[0]}).x), i32(textureDimensions(t_${binds[0]}).y));\n`;
  // output (packed) dims:
  w += `  let odim = textureDimensions(outTex);\n`;
  w += `  let sp = vec2i(i32(gid.x), i32(gid.y));\n`;
  // guard: source pixel range. For packed output, source res = odim/(4,2);
  // for native output (widthMul 1), source res = odim.
  if (h.widthMul >= 4) {
    w += `  if (sp.x * 4 >= i32(odim.x) || sp.y * 2 >= i32(odim.y)) { return; }\n`;
  } else {
    w += `  if (sp.x >= i32(odim.x) || sp.y >= i32(odim.y)) { return; }\n`;
  }

  const isL0 = parsed.terms.some(t => t.kind === "vec");
  const featStride = isL0 ? null : "vec2i(4, 2)";

  // load taps. For L0: scalar luma at (sp + (dx-1,dy-1)). For Lk: feature k packed.
  // We load every (feat, dx, dy) referenced.
  const tapSet = new Set();
  for (const t of parsed.terms) tapSet.add(`${t.feat}_${t.dx}_${t.dy}`);

  if (isL0) {
    // scalar luma input, single bind
    const src = binds[0];
    for (const key of tapSet) {
      const [feat, dx, dy] = key.split("_").map(Number);
      const ox = dx - 1, oy = dy - 1;
      w += `  let inp_${feat}_${dx}_${dy} = textureLoad(t_${src}, clamp(sp + vec2i(${ox}, ${oy}), vec2i(0), sdim - vec2i(1)), 0).x;\n`;
    }
  } else {
    // packed features: feature f of source pixel q lives at texel q*(4,2) + (f%4, f/4)
    // skip-sum: sum the two bound sources.
    const srcs = binds;
    for (const key of tapSet) {
      const [feat, dx, dy] = key.split("_").map(Number);
      const ox = dx - 1, oy = dy - 1;
      const fx = feat % 4, fy = Math.floor(feat / 4);
      const coord = `clamp((sp + vec2i(${ox}, ${oy})) * vec2i(4,2) + vec2i(${fx}, ${fy}), vec2i(0), sdim - vec2i(1))`;
      if (parsed.skipSum && srcs.length === 2) {
        w += `  let inp_${feat}_${dx}_${dy} = textureLoad(t_${srcs[0]}, ${coord}, 0) + textureLoad(t_${srcs[1]}, ${coord}, 0);\n`;
      } else {
        w += `  let inp_${feat}_${dx}_${dy} = textureLoad(t_${srcs[0]}, ${coord}, 0);\n`;
      }
    }
  }

  // results
  const nR = parsed.numResults;
  for (let r = 0; r < nR; r++) w += `  var result${r} = ${vec4f(parsed.biases[r])};\n`;
  for (const t of parsed.terms) {
    if (t.kind === "vec") {
      w += `  result${t.r} += ${vec4f(t.weights)} * inp_${t.feat}_${t.dx}_${t.dy};\n`;
    } else {
      w += `  result${t.r} += ${mat4f(t.weights)} * inp_${t.feat}_${t.dx}_${t.dy};\n`;
    }
  }
  // store. Output layout depends on widthMul: packed 4x2 block (widthMul==4) or
  // a single texel at source pos (widthMul==1, e.g. conv2d_6 -> 4 sub-pixel vals).
  const act = parsed.reluOnStore;
  const packed = h.widthMul >= 4;
  for (let r = 0; r < nR; r++) {
    const val = act ? `max(result${r}, vec4f(0.0))` : `result${r}`;
    if (packed) {
      const fx = r % 4, fy = Math.floor(r / 4);
      w += `  textureStore(outTex, sp * vec2i(4,2) + vec2i(${fx}, ${fy}), ${val});\n`;
    } else {
      w += `  textureStore(outTex, sp, ${val});\n`;
    }
  }
  w += `}\n`;
  return { wgsl: w, kind: "conv" };
}

for (const input of inputs) {
  const src = fs.readFileSync(input, "utf8");
  const base = path.basename(input).replace(/\.glsl$/i, "");
  const raw = splitPasses(src);
  const passes = raw.map((p) => {
    const header = parseHeader(p.header);
    const parsed = parseConv(p.body);
    return { header, body: p.body, parsed };
  });

  // attach parsed into shape emitConvPass expects
  let wgsl = `// AUTO-GENERATED by transpile-artcnn.js — do not edit.\n`;
  const manifest = { name: base, scale: 2, whenThreshold: 1.3, passes: [] };
  passes.forEach((p, i) => {
    p.parsed = p.parsed; // already there
    // mark depth-to-space from DESC if needed
    if (/Depth-To-Space/i.test(p.header.desc || "")) p.parsed.isDepthToSpace = true;
    const emitted = emitConvPass({ header: p.header, body: p.body, parsed: p.parsed }, i);
    wgsl += `\n//==== ENTRY pass${i} : ${(p.header.desc||"").replace(/[^A-Za-z0-9]+/g,"_")} ====\n` + emitted.wgsl;
    manifest.passes.push({
      index: i, desc: p.header.desc, binds: p.header.binds, save: p.header.save,
      widthMul: p.header.widthMul, heightMul: p.header.heightMul,
      kind: emitted.kind, relu: !!p.parsed.reluOnStore,
      numResults: p.parsed.numResults || 0, skipSum: !!p.parsed.skipSum,
    });
  });

  fs.writeFileSync(path.join(outDir, `${base}.artcnn.json`), JSON.stringify(manifest, null, 2));
  fs.writeFileSync(path.join(outDir, `${base}.artcnn.wgsl`), wgsl);
  const convN = manifest.passes.filter(p=>p.kind==="conv").length;
  const totalTerms = passes.reduce((s,p)=>s+(p.parsed.terms?p.parsed.terms.length:0),0);
  console.log(`${base}: ${passes.length} passes (${convN} conv, ${manifest.passes.length-convN} d2s), ${totalTerms} weight terms`);
}
console.log(`\nWrote ArtCNN model to ${outDir}/`);

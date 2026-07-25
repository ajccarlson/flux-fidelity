#!/usr/bin/env node
/*
 * transpile.js — FSRCNNX mpv-GLSL → WGSL + JSON model descriptor.
 *
 * Parses the //!HOOK passes of a libplacebo FSRCNNX shader and emits, per file:
 *   <out>/<name>.passes.json   — ordered pass descriptors (bindings, save, scale)
 *   <out>/<name>.wgsl          — one WGSL compute shader per pass, concatenated
 *                                with markers, plus shared constants.
 *
 * Recognised pass shapes (all that appear in FSRCNNX_x{2,3,4}_16-0-4-1):
 *   A) feature map   : res = bias; res += vec4(...) * float(LUMA_texOff(vec2(dx,dy)));  PReLU
 *   B) mapping/resid : res = bias; res += mat4(...) * SRC_texOff(vec2(dx,dy));           PReLU
 *                      (residual passes add a bare `res += FEATURE_texOff(0);`)
 *   C) subconv       : same as B but may end with `return vec4(res);` (no activation)
 *   D) aggregation   : pixel-shuffle; handled by a fixed template, not weight parsing.
 *
 * The emitted WGSL is render-to-texture friendly (fragment style) but written as
 * compute for the real pipeline. This script's job is faithful extraction; the
 * runtime (separate) wires textures together using the JSON.
 *
 * Usage: node transpile.js <input.glsl> [<input2.glsl> ...] --out ./model
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const VERIFIED_STANDARD_SOURCE = Object.freeze({
  file: "FSRCNNX_x2_16-0-4-1.glsl",
  sourcePath: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  sourceSha256: "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  upstream: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl",
  license: "LGPL-3.0-or-later",
  modificationNotice:
    "Transpiled in 2026 from the mpv/libplacebo GLSL hook format to WGSL compute passes and a JSON pass manifest for FSRCNNX-EXT; model weights and pass order are preserved.",
});

// This legacy High model is intentionally pinned by bytes without assigning
// source or license metadata that has not been independently verified. Its
// unresolved distribution clearance is enforced by the release gate; the
// transpiler still rejects accidental or silent weight substitution.
const PINNED_LEGACY_HIGH_SOURCE = Object.freeze({
  file: "FSRCNNX_x2_56-16-4-1.glsl",
  sourceSha256: "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
});

const PINNED_SOURCE_BY_FILE = new Map([
  VERIFIED_STANDARD_SOURCE,
  PINNED_LEGACY_HIGH_SOURCE,
].map((entry) => [entry.file, entry]));

// ---- arg parsing ----------------------------------------------------------
const args = process.argv.slice(2);
const outIdx = args.indexOf("--out");
const outDir = outIdx >= 0 ? args[outIdx + 1] : "./model";
const inputs = args.filter((a, i) => !a.startsWith("--") && i !== outIdx + 1 && !(outIdx >= 0 && i === outIdx));
if (!inputs.length) {
  console.error("usage: node transpile.js <file.glsl>... --out ./model");
  process.exit(1);
}
fs.mkdirSync(outDir, { recursive: true });

// ---- helpers --------------------------------------------------------------
const num = (s) => {
  const v = parseFloat(s);
  if (!Number.isFinite(v)) throw new Error(`bad number: ${s}`);
  return v;
};
const splitFloats = (inner) =>
  inner
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x.length)
    .map(num);

// Split the file into pass blocks. Each block starts at a `//!HOOK` line and
// runs until the next `//!HOOK` or EOF.
function splitPasses(src) {
  const lines = src.split(/\r?\n/);
  const passes = [];
  let cur = null;
  for (const line of lines) {
    if (line.startsWith("//!HOOK")) {
      if (cur) passes.push(cur);
      cur = { headerLines: [], bodyLines: [] };
    }
    if (!cur) continue;
    if (line.startsWith("//!")) cur.headerLines.push(line);
    else cur.bodyLines.push(line);
  }
  if (cur) passes.push(cur);
  return passes;
}

function parseHeader(headerLines) {
  const h = {
    hook: null, when: null, desc: null,
    binds: [], save: null, components: 1,
    widthMul: 1, heightMul: 1,
  };
  for (const l of headerLines) {
    const m = l.match(/^\/\/!(\w+)\s*(.*)$/);
    if (!m) continue;
    const [, key, rest] = m;
    switch (key) {
      case "HOOK": h.hook = rest.trim(); break;
      case "WHEN": h.when = rest.trim(); break;
      case "DESC": h.desc = rest.trim(); break;
      case "BIND": h.binds.push(rest.trim()); break;
      case "SAVE": h.save = rest.trim(); break;
      case "COMPONENTS": h.components = parseInt(rest.trim(), 10) || 1; break;
      case "WIDTH": {
        const mm = rest.match(/(\d+(?:\.\d+)?)\s*\*/);
        if (mm) h.widthMul = num(mm[1]);
        break;
      }
      case "HEIGHT": {
        const mm = rest.match(/(\d+(?:\.\d+)?)\s*\*/);
        if (mm) h.heightMul = num(mm[1]);
        break;
      }
    }
  }
  return h;
}

// Parse the `//!WHEN OUTPUT.w LUMA.w / X.XXX > ...` threshold (RPN-ish).
// We only need X (the scale-selection threshold).
function parseWhenThreshold(when) {
  if (!when) return null;
  const m = when.match(/LUMA\.w\s*\/\s*(\d+(?:\.\d+)?)/);
  return m ? num(m[1]) : null;
}

// Parse a body into structured ops.
// Returns { bias:[...], terms:[{kind:'vec'|'mat'|'add', src, off:[dx,dy], weights}], activation }
function parseBody(body, components) {
  const text = body.join("\n");
  const out = { bias: null, terms: [], activation: null, isShuffle: false };

  // Detect aggregation / pixel-shuffle by signature, handled separately.
  if (/fract\(\s*\w+_pos\s*\*\s*\w+_size\s*\)/.test(text)) {
    out.isShuffle = true;
    return out;
  }

  // output width = number of components this pass writes (3 for vec3, etc.)
  out.outDim = components;

  // ---- Stage 1: local variable definitions (x3/x4 sub-band residuals) ----
  // Pattern:
  //   vecN resK = SRC_texOff(vec2(..));
  //   resK = max(resK, vec4(0.0)) + vec4(..) * min(resK, vec4(0.0));   // optional PReLU
  // We record {name -> {src, off, prelu?}} so mat*resK terms can resolve.
  const locals = {};
  const localDefRe =
    /vec\d\s+(res\d+)\s*=\s*(\w+)_texOff\(\s*(?:vec2\(\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(-?\d+(?:\.\d+)?)\s*)?\)|(-?\d+(?:\.\d+)?))\s*\)\s*;/g;
  let lm;
  const ri = (s) => (s === undefined ? 0 : Math.round(parseFloat(s)));
  while ((lm = localDefRe.exec(text))) {
    const dx = lm[3] !== undefined ? ri(lm[3]) : (lm[5] !== undefined ? ri(lm[5]) : 0);
    const dy = lm[4] !== undefined ? ri(lm[4]) : 0;
    locals[lm[1]] = { src: lm[2], off: [dx, dy], prelu: null };
  }
  // per-local PReLU
  const localActRe =
    /(res\d+)\s*=\s*max\(\s*\1\s*,\s*vec\d\(0\.0\)\s*\)\s*\+\s*vec\d\(([^)]*)\)\s*\*\s*min\(\s*\1\s*,\s*vec\d\(0\.0\)\s*\)\s*;/g;
  let am;
  while ((am = localActRe.exec(text))) {
    if (locals[am[1]]) locals[am[1]].prelu = splitFloats(am[2]);
  }

  // bias: the main accumulator `res = vecN(...);` — must be `res` exactly,
  // not res1/res2/etc, and the RHS is a plain vec literal (not a max/min expr).
  // Offset forms seen across FSRCNNX variants:
  //   vec2(-2,-2)   two ints       (16-0-4-1 feature/mapping passes)
  //   vec2(0.0)     single float   (56-16-4-1 shrinking/mapping passes)
  //   vec2(0,0)     two ints zero
  //   0  /  0.0     bare scalar
  // Offsets are always whole-pixel; we round to int. This sub-pattern captures
  // the inside of the *_texOff(...) call as up to two numeric tokens.
  // Helper to parse a captured numeric token (int or float) to a rounded int.
  const toInt = (s) => (s === undefined || s === null || s === "" ? 0 : Math.round(parseFloat(s)));
  // Numeric token (int or float, optional exponent/sign).
  const N = String.raw`-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?`;
  // Offset inside *_texOff(...): vec2(a) | vec2(a,b) | a
  const OFF = String.raw`(?:vec2\(\s*(${N})\s*(?:,\s*(${N})\s*)?\)|(${N}))`;

  const biasM = text.match(/\bres\s*=\s*vec\d\(([\d.,\seE+-]*)\)\s*;/);
  if (biasM) out.bias = splitFloats(biasM[1]);

  // vec-weight terms: `res += vec4(...) * float(SRC_texOff(<off>));`
  const vecRe = new RegExp(
    String.raw`res\s*\+=\s*vec\d\(([^)]*)\)\s*\*\s*float\(\s*(\w+)_texOff\(\s*${OFF}\s*\)\s*\)\s*;`,
    "g"
  );
  let m;
  while ((m = vecRe.exec(text))) {
    // groups: 1=weights, 2=src, 3=vecX, 4=vecY, 5=scalar
    const dx = m[3] !== undefined ? toInt(m[3]) : toInt(m[5]);
    const dy = m[4] !== undefined ? toInt(m[4]) : 0;
    out.terms.push({
      kind: "vec",
      src: m[2],
      off: [dx, dy],
      weights: splitFloats(m[1]),
    });
  }

  // mat-weight terms. The matrix type encodes input×output:
  //   mat4  / mat4x4  -> 4 in, 4 out (16 vals)
  //   mat4x3          -> 4 in, 3 out (12 vals)  [GLSL mat<cols>x<rows>, col-major]
  //   mat4x2          -> 4 in, 2 out (8 vals)
  const matRe = new RegExp(
    String.raw`res\s*\+=\s*mat4(?:x(\d))?\(([^)]*)\)\s*\*\s*(\w+)_texOff\(\s*${OFF}\s*\)\s*;`,
    "g"
  );
  while ((m = matRe.exec(text))) {
    // groups: 1=matRows, 2=weights, 3=src, 4=vecX, 5=vecY, 6=scalar
    const rows = m[1] !== undefined ? parseInt(m[1], 10) : 4; // output dim
    const dx = m[4] !== undefined ? toInt(m[4]) : toInt(m[6]);
    const dy = m[5] !== undefined ? toInt(m[5]) : 0;
    out.terms.push({
      kind: "mat",
      src: m[3],
      off: [dx, dy],
      rows, // output components
      weights: splitFloats(m[2]), // column-major: 4 columns × `rows` rows
    });
  }

  // mat * local-variable terms: `res += mat4(...) * resK;`
  // Resolve resK to its source texture + optional pre-activation PReLU.
  const matLocalRe =
    /res\s*\+=\s*mat4(?:x(\d))?\(([^)]*)\)\s*\*\s*(res\d+)\s*;/g;
  while ((m = matLocalRe.exec(text))) {
    const local = locals[m[3]];
    if (!local) continue;
    const rows = m[1] !== undefined ? parseInt(m[1], 10) : 4;
    out.terms.push({
      kind: "mat",
      src: local.src,
      off: local.off,
      rows,
      weights: splitFloats(m[2]),
      preActivation: local.prelu,
    });
  }

  // bare residual add: `res += SRC_texOff(<off=0>);` (no weight matrix)
  const addRe = new RegExp(String.raw`res\s*\+=\s*(\w+)_texOff\(\s*${OFF}\s*\)\s*;`, "g");
  while ((m = addRe.exec(text))) {
    const dx = m[2] !== undefined ? toInt(m[2]) : toInt(m[4]);
    const dy = m[3] !== undefined ? toInt(m[3]) : 0;
    out.terms.push({ kind: "add", src: m[1], off: [dx, dy], weights: null });
  }

  // activation: PReLU `res = max(res, vec4(0.0)) + vec4(...) * min(res, vec4(0.0));`
  const preluM = text.match(
    /res\s*=\s*max\(\s*res\s*,\s*vec\d\(0\.0\)\s*\)\s*\+\s*vec\d\(([^)]*)\)\s*\*\s*min\(\s*res\s*,\s*vec\d\(0\.0\)\s*\)\s*;/
  );
  if (preluM) {
    out.activation = { type: "prelu", slope: splitFloats(preluM[1]) };
  } else {
    out.activation = { type: "none" };
  }

  return out;
}

// ---- WGSL emission --------------------------------------------------------
// Each non-shuffle pass becomes a compute entry that, for one output texel,
// accumulates bias + Σ (weights · sample). vec-term: scalar luma * vec4 weight.
// mat-term: mat4 (col-major) * vec4 sample.

function fmtF32(x) {
  // Keep full precision; WGSL accepts standard float literals.
  let s = x.toPrecision(9);
  if (!/[.eE]/.test(s)) s += ".0";
  return s;
}
const fmtVec4 = (a) => `vec4f(${a.map(fmtF32).join(", ")})`;
// GLSL mat4(c0r0,c0r1,c0r2,c0r3, c1r0,...) is column-major.
// WGSL mat4x4f takes columns too, so the 16 values map directly.
const fmtMat4 = (a) => `mat4x4f(${a.map(fmtF32).join(", ")})`;

function emitPassWGSL(pass, index) {
  const { header, body } = pass;
  const name = (header.desc || `pass${index}`).replace(/[^A-Za-z0-9]+/g, "_");
  const bindList = header.binds;
  // textures bound 0..n-1, output storage texture at n, sampler params via uniforms.
  let wgsl = `// ---- PASS ${index}: ${header.desc} (save=${header.save}, comps=${header.components}) ----\n`;
  wgsl += `// binds: ${bindList.join(", ")}\n`;

  if (body.isShuffle) {
    // Aggregation / pixel-shuffle. scale = widthMul. Reads `scale` SUBCONV
    // textures (bound 0..scale-1); each holds `scale` valid components.
    // GLSL builds mat = (sub1, sub2, ..., subScale) as columns, then returns
    // mat[index.x][index.y] where index = floor(fract(pos*size)*scale).
    // => column = sub texture index.x, row = component index.y.
    const scale = header.widthMul;
    const nSub = header.binds.length; // x2 binds 1 subconv, x3 binds 3, x4 binds 4
    wgsl += `// pixel-shuffle aggregation, scale=${scale}, ${nSub} subconv texture(s)\n`;
    for (let i = 0; i < nSub; i++) {
      wgsl += `@group(0) @binding(${i}) var src${i} : texture_2d<f32>;\n`;
    }
    wgsl += `@group(0) @binding(${nSub}) var outTex : texture_storage_2d<rgba16float, write>;\n`;
    wgsl += `@compute @workgroup_size(8, 8)\n`;
    wgsl += `fn main(@builtin(global_invocation_id) gid : vec3u) {\n`;
    wgsl += `  let dims = textureDimensions(outTex);\n`;
    wgsl += `  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\n`;
    wgsl += `  let lo = vec2i(i32(gid.x) / ${scale}, i32(gid.y) / ${scale});\n`;
    wgsl += `  let ix = i32(gid.x) % ${scale};\n`;
    wgsl += `  let iy = i32(gid.y) % ${scale};\n`;
    if (nSub === 1) {
      // x2 form: single texture, flat channel index = ix*scale + iy.
      wgsl += `  let v = textureLoad(src0, lo, 0);\n`;
      wgsl += `  let ch = ix * ${scale} + iy;\n`;
      wgsl += `  var y = v.x;\n`;
      for (let c = 1; c < scale * scale; c++) {
        wgsl += `  if (ch == ${c}) { y = v[${c}]; }\n`;
      }
    } else {
      // x3/x4 form: column = subconv texture (ix), row = component (iy).
      wgsl += `  var cols : array<vec4f, ${nSub}>;\n`;
      for (let i = 0; i < nSub; i++) {
        wgsl += `  cols[${i}] = textureLoad(src${i}, lo, 0);\n`;
      }
      wgsl += `  let col = cols[ix];\n`;
      wgsl += `  var y = col.x;\n`;
      for (let r = 1; r < scale; r++) {
        wgsl += `  if (iy == ${r}) { y = col[${r}]; }\n`;
      }
    }
    wgsl += `  textureStore(outTex, vec2i(i32(gid.x), i32(gid.y)), vec4f(y, 0.0, 0.0, 1.0));\n`;
    wgsl += `}\n`;
    return { name, wgsl, shuffle: true, scale };
  }

  // Standard conv pass.
  // Inputs: one texture per bind (luma is r32float single-channel; feature/model are rgba32float).
  bindList.forEach((b, i) => {
    // LUMA source sampled as scalar; others as vec4.
    wgsl += `@group(0) @binding(${i}) var t_${b} : texture_2d<f32>;\n`;
  });
  const outBinding = bindList.length;
  wgsl += `@group(0) @binding(${outBinding}) var outTex : texture_storage_2d<rgba16float, write>;\n`;
  wgsl += `@compute @workgroup_size(8, 8)\n`;
  wgsl += `fn main(@builtin(global_invocation_id) gid : vec3u) {\n`;
  wgsl += `  let dims = textureDimensions(outTex);\n`;
  wgsl += `  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\n`;
  wgsl += `  let p = vec2i(i32(gid.x), i32(gid.y));\n`;
  wgsl += `  var res = ${body.bias ? padVec4(body.bias) : "vec4f(0.0)"};\n`;

  for (const term of body.terms) {
    const src = `t_${term.src}`;
    const off = `vec2i(${term.off[0]}, ${term.off[1]})`;
    const coord = `clampCoord(p + ${off}, textureDimensions(${src}))`;
    if (term.kind === "vec") {
      wgsl += `  res += ${fmtVec4(term.weights)} * textureLoad(${src}, ${coord}, 0).x;\n`;
    } else if (term.kind === "mat") {
      const rows = term.rows || 4;
      const sampleVar = `s_${term.src}_${term.off[0]}_${term.off[1]}`.replace(/-/g, "m");
      const hasPre = term.preActivation && term.preActivation.length;
      if (hasPre) {
        // sample, then PReLU it, then matmul — used by x3/x4 sub-band residuals.
        wgsl += `  { var ${sampleVar} = textureLoad(${src}, ${coord}, 0);\n`;
        wgsl += `    ${sampleVar} = max(${sampleVar}, vec4f(0.0)) + ${padVec4(term.preActivation)} * min(${sampleVar}, vec4f(0.0));\n`;
        if (rows === 4) {
          wgsl += `    res += ${fmtMat4(term.weights)} * ${sampleVar};\n`;
        } else {
          for (let r = 0; r < rows; r++) {
            const c = [0, 1, 2, 3].map((k) => fmtF32(term.weights[k * rows + r]));
            wgsl += `    res[${r}] += dot(vec4f(${c.join(", ")}), ${sampleVar});\n`;
          }
        }
        wgsl += `  }\n`;
      } else if (rows === 4) {
        // square: direct mat4x4 multiply (col-major maps 1:1 to WGSL)
        wgsl += `  res += ${fmtMat4(term.weights)} * textureLoad(${src}, ${coord}, 0);\n`;
      } else {
        // non-square (mat4x<rows>): expand to per-output dot products.
        // GLSL col-major: weights = [c0r0..c0r(rows-1), c1r0.., c2.., c3..]
        // output[r] = Σ_c weights[c*rows + r] * sample[c]
        wgsl += `  { let ${sampleVar} = textureLoad(${src}, ${coord}, 0);\n`;
        for (let r = 0; r < rows; r++) {
          const c0 = fmtF32(term.weights[0 * rows + r]);
          const c1 = fmtF32(term.weights[1 * rows + r]);
          const c2 = fmtF32(term.weights[2 * rows + r]);
          const c3 = fmtF32(term.weights[3 * rows + r]);
          wgsl += `    res[${r}] += dot(vec4f(${c0}, ${c1}, ${c2}, ${c3}), ${sampleVar});\n`;
        }
        wgsl += `  }\n`;
      }
    } else if (term.kind === "add") {
      wgsl += `  res += textureLoad(${src}, ${coord}, 0);\n`;
    }
  }

  if (body.activation && body.activation.type === "prelu") {
    wgsl += `  res = max(res, vec4f(0.0)) + ${padVec4(body.activation.slope)} * min(res, vec4f(0.0));\n`;
  }
  wgsl += `  textureStore(outTex, p, res);\n`;
  wgsl += `}\n`;
  return { name, wgsl, shuffle: false };
}

function padVec4(arr) {
  const a = arr.slice(0, 4);
  while (a.length < 4) a.push(0);
  return fmtVec4(a);
}

// helper WGSL prelude (clampCoord) shared by all passes
const PRELUDE = `// AUTO-GENERATED by transpile.js — do not edit by hand.
// Shared helper. Each pass below is an independent compute entry; the runtime
// instantiates one pipeline per pass and binds textures per the JSON manifest.
fn clampCoord(c : vec2i, dim : vec2u) -> vec2i {
  return clamp(c, vec2i(0, 0), vec2i(i32(dim.x) - 1, i32(dim.y) - 1));
}
`;

// ---- main per-file --------------------------------------------------------
for (const input of inputs) {
  const sourceBytes = fs.readFileSync(input);
  const src = sourceBytes.toString("utf8");
  const base = path.basename(input).replace(/\.glsl$/i, "");
  const sourceFile = path.basename(input);
  const pinnedSource = PINNED_SOURCE_BY_FILE.get(sourceFile) ?? null;
  let sourceMetadata = null;
  if (pinnedSource) {
    const sourceSha256 = createHash("sha256").update(sourceBytes).digest("hex");
    if (sourceSha256 !== pinnedSource.sourceSha256) {
      throw new Error(
        `${pinnedSource.file}: SHA-256 ${sourceSha256}, ` +
        `expected ${pinnedSource.sourceSha256}`,
      );
    }
    if (pinnedSource === VERIFIED_STANDARD_SOURCE) {
      sourceMetadata = {
        license: VERIFIED_STANDARD_SOURCE.license,
        sourcePath: VERIFIED_STANDARD_SOURCE.sourcePath,
        sourceSha256,
        modificationNotice: VERIFIED_STANDARD_SOURCE.modificationNotice,
      };
    }
  }
  const rawPasses = splitPasses(src);

  const parsed = [];
  for (let i = 0; i < rawPasses.length; i++) {
    const header = parseHeader(rawPasses[i].headerLines);
    const body = parseBody(rawPasses[i].bodyLines, header.components);
    parsed.push({ header, body, index: i });
  }

  // Manifest: ordered passes with binds/save/components/scale + the WHEN threshold.
  const whenThreshold = parsed.map((p) => parseWhenThreshold(p.header.when)).find((x) => x != null) ?? null;
  const manifest = {
    name: base,
    ...(sourceMetadata ?? {}),
    whenThreshold, // select this model when target/source ratio > this
    passes: parsed.map((p) => ({
      index: p.index,
      desc: p.header.desc,
      binds: p.header.binds,
      save: p.header.save,
      components: p.header.components,
      widthMul: p.header.widthMul,
      heightMul: p.header.heightMul,
      kind: p.body.isShuffle ? "shuffle" : "conv",
      activation: p.body.isShuffle ? null : p.body.activation?.type,
      termCount: p.body.terms.length,
    })),
  };

  const sourceHeader = sourceMetadata
    ? `// License: ${sourceMetadata.license}\n` +
      `// Upstream: ${VERIFIED_STANDARD_SOURCE.upstream}\n` +
      `// Source path: ${sourceMetadata.sourcePath}\n` +
      `// Source SHA-256: ${sourceMetadata.sourceSha256}\n` +
      `// Modification notice: ${sourceMetadata.modificationNotice}\n`
    : "";
  let wgsl = sourceHeader + PRELUDE + "\n";
  const passMeta = [];
  for (const p of parsed) {
    const emitted = emitPassWGSL(p, p.index);
    wgsl += `\n//==== ENTRY pass${p.index} : ${emitted.name} ====\n`;
    wgsl += emitted.wgsl;
    passMeta.push({ index: p.index, name: emitted.name, shuffle: !!emitted.shuffle });
  }

  fs.writeFileSync(path.join(outDir, `${base}.passes.json`), JSON.stringify(manifest, null, 2));
  fs.writeFileSync(path.join(outDir, `${base}.wgsl`), wgsl);

  // sanity stats
  const convCount = parsed.filter((p) => !p.body.isShuffle).length;
  const shuffleCount = parsed.filter((p) => p.body.isShuffle).length;
  const totalTerms = parsed.reduce((s, p) => s + p.body.terms.length, 0);
  console.log(
    `${base}: ${parsed.length} passes (${convCount} conv, ${shuffleCount} shuffle), ` +
      `${totalTerms} weight terms, whenThreshold=${whenThreshold}`
  );
}
console.log(`\nWrote model to ${outDir}/`);

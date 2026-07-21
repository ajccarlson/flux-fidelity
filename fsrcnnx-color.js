// fsrcnnx-color.js — luma extraction and recombination shaders.
//
// FSRCNNX upscales luma only. Pipeline:
//   RGB frame --(extract)--> luma (rgba16float, Y in .x)
//   luma --(FSRCNNX chain)--> upscaled luma (r16float)
//   upscaled luma + original RGB --(recombine)--> output RGB
//
// Chromium converts each decoded frame's source primaries, transfer, YUV matrix,
// and full/limited range into the explicitly requested sRGB target at the WebGPU
// external-texture boundary. These shaders therefore operate only on sRGB/BT.709
// target-space RGB. The local reversible matrix extracts/replaces target-space
// luma; it does not perform source YUV range or primary conversion.

// Extract luma from an imported external video texture into rgba16float (Y in .x).
export const LUMA_EXTRACT_WGSL = /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var frame : texture_external;
@group(0) @binding(2) var outTex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let dims = textureDimensions(outTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(dims.x), f32(dims.y));
  let rgb = textureSampleBaseClampToEdge(frame, samp, uv).rgb;
  // BT.709 luma. FSRCNNX was trained on this convention.
  let y = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  textureStore(outTex, vec2i(i32(gid.x), i32(gid.y)), vec4f(y, 0.0, 0.0, 1.0));
}
`;

// Recombine: sample upscaled luma (r16float) + original RGB (external), output RGB.
// Renders a fullscreen triangle into the canvas at output resolution.
export const RECOMBINE_WGSL = /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var srcFrame : texture_external;     // original RGB
@group(0) @binding(2) var hiLuma : texture_2d<f32>;        // upscaled Y

struct VsOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  var p = array<vec2f,3>(vec2f(-1.,-3.), vec2f(-1.,1.), vec2f(3.,1.));
  var uv = array<vec2f,3>(vec2f(0.,2.), vec2f(0.,0.), vec2f(2.,0.));
  var o : VsOut;
  o.pos = vec4f(p[i], 0., 1.);
  o.uv = uv[i];
  return o;
}

fn rgb2ycbcr(c : vec3f) -> vec3f {
  let y  = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let cb = (c.b - y) / 1.8556;
  let cr = (c.r - y) / 1.5748;
  return vec3f(y, cb, cr);
}
fn ycbcr2rgb(v : vec3f) -> vec3f {
  let y = v.x; let cb = v.y; let cr = v.z;
  let r = y + 1.5748 * cr;
  let b = y + 1.8556 * cb;
  let g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
  return vec3f(r, g, b);
}

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let origRgb = textureSampleBaseClampToEdge(srcFrame, samp, uv).rgb;
  let ycc = rgb2ycbcr(origRgb);              // chroma from bilinear original
  let dims = vec2f(textureDimensions(hiLuma));
  let coord = vec2i(uv * dims);
  let yHi = textureLoad(hiLuma, clamp(coord, vec2i(0), vec2i(dims) - 1), 0).x;
  let outRgb = ycbcr2rgb(vec3f(yHi, ycc.y, ycc.z));   // swap in network luma
  return vec4f(clamp(outRgb, vec3f(0.), vec3f(1.)), 1.0);
}
`;

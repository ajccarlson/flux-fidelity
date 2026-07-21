// fsrcnnx-ssimds.js — SSimDownscaler, hand-ported from Shiandow's mpv shader.
//
// Purpose: when our FSRCNNX output OVERSHOOTS the display resolution (e.g. force
// x4 on 1080p = 7680px wide on a 5160px screen), the browser would normally
// downscale it with plain bilinear, discarding sharpness FSRCNNX just created.
// SSimDownscaler replaces that final downscale with a structural-similarity
// preserving one, matching how it runs after FSRCNNX in an mpv pipeline.
//
// mpv -> our mapping:
//   PREKERNEL  = hi-res RGB (FSRCNNX recombined output)        [hiTex]
//   POSTKERNEL = mean: hi-res downscaled to display (E[x])     [we compute in P0]
//   target     = display size (canvas)
//
// Pass chain (all render-to-texture, separable):
//   P0 mean : hiTex --(Mitchell-Netravali downscale)--> meanTex   (display res, rgba16f)
//   P1 L2v  : hiTex --(vertical MN on tex*tex)--> l2vTex          (hiW x dispH, rgba16f)
//   P2 L2h  : l2vTex --(horizontal MN)--> l2Tex                   (display res, rgba16f) = E[x^2]
//   P3 MR   : meanTex + l2Tex --(locality kernel, variance ratio)--> mrTex (rgba16f)
//   P4 final: meanTex + mrTex --(locality kernel, reconstruct)--> output RGB
//
// Kernels:
//   MN(0,0.5)            : Mitchell-Netravali (used for moment downscales), taps=2
//   pow(1/locality,|x|)  : locality weighting for variance smoothing, taps=3, locality=2

function requireDownscaleRatio(value, label) {
  let ratio;
  try { ratio = Number(value); } catch { ratio = NaN; }
  if (!Number.isFinite(ratio) || ratio < 1.0) {
    throw new RangeError(`${label} must be a finite downscale ratio >= 1`);
  }
  return ratio;
}

// Shared WGSL helpers (kernels). Prepended to each pass.
const HELPERS = /* wgsl */ `
fn mn(x : f32) -> f32 {
  // Mitchell-Netravali with B=0, C=0.5 (Catmull-Rom). x = abs(distance in taps).
  let B = 0.0; let C = 0.5;
  if (x < 1.0) {
    return ((2.0 - 1.5*B - C)*x + (-3.0 + 2.0*B + C))*x*x + (1.0 - B/3.0);
  } else if (x < 2.0) {
    return (((-B/6.0 - C)*x + (B + 5.0*C))*x + (-2.0*B - 8.0*C))*x + ((4.0/3.0)*B + 4.0*C);
  }
  return 0.0;
}
fn loc(x : f32) -> f32 { return pow(0.5, abs(x)); } // locality=2 -> 1/2
fn luma(c : vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
const NUM_EPS : f32 = 1.0e-6;
`;

// Fullscreen-triangle vertex shader shared by all passes.
const VS = /* wgsl */ `
struct VsOut { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  var p = array<vec2f,3>(vec2f(-1.,-3.), vec2f(-1.,1.), vec2f(3.,1.));
  var uv = array<vec2f,3>(vec2f(0.,2.), vec2f(0.,0.), vec2f(2.,0.));
  var o : VsOut; o.pos = vec4f(p[i], 0., 1.); o.uv = uv[i]; return o;
}
`;

// P0 — mean: separable MN downscale of hi-res to display res. The downscale
// ratio is baked per-build via buildMeanShader() since the shader needs it as a
// constant (output dims aren't available without derivatives).
export function buildMeanShader(ratioX, ratioY) {
  // taps=2 in output space; gather hi-res samples within +/- taps*ratio.
  const rx = requireDownscaleRatio(ratioX, "ratioX").toFixed(6);
  const ry = requireDownscaleRatio(ratioY, "ratioY").toFixed(6);
  return HELPERS + VS + `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var hiTex : texture_2d<f32>;
@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let hiDim = vec2f(textureDimensions(hiTex));
  let hiPt = 1.0 / hiDim;                 // one hi-res texel in uv
  let ratio = vec2f(${rx}, ${ry});        // hi texels per output texel
  let center = uv * hiDim - 0.5;          // hi-res coordinate of output center
  let lowX = ceil(center.x - 2.0*ratio.x);
  let highX = floor(center.x + 2.0*ratio.x);
  let lowY = ceil(center.y - 2.0*ratio.y);
  let highY = floor(center.y + 2.0*ratio.y);
  var acc = vec4f(0.0); var W = 0.0;
  for (var ky = lowY; ky <= highY; ky += 1.0) {
    let wy = mn(abs((ky - center.y) / ratio.y));
    for (var kx = lowX; kx <= highX; kx += 1.0) {
      let wx = mn(abs((kx - center.x) / ratio.x));
      let w = wx * wy;
      let p = (vec2f(kx, ky) + 0.5) * hiPt;
      acc += w * textureSampleLevel(hiTex, samp, p, 0.0);
      W += w;
    }
  }
  if (abs(W) <= NUM_EPS) {
    return textureSampleLevel(hiTex, samp, uv, 0.0);
  }
  return acc / W;
}`;
}

// P1+P2 combined as one separable build: vertical then horizontal MN on tex*tex.
// We emit two shaders sharing the kernel. axis 1 = vertical (P1), axis 0 = horiz.
export function buildL2Shader(axis, ratio) {
  if (axis !== 0 && axis !== 1) throw new RangeError("axis must be 0 or 1");
  const r = requireDownscaleRatio(ratio, "ratio").toFixed(6);
  const isV = axis === 1;
  // P1 reads hiTex (rgb), squares it. P2 reads l2v (already squared), no square.
  const square = isV ? "s = s * s;" : "";
  const dimAxis = isV ? "dim.y" : "dim.x";
  const uvStep = isV ? "vec2f(0.0, step)" : "vec2f(step, 0.0)";
  const centerAxis = isV ? "center.y" : "center.x";
  return HELPERS + VS + `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var src : texture_2d<f32>;
@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(src));
  let srcPt = 1.0 / dim;
  let ratio = ${r};                          // src texels per output texel (this axis)
  let center = uv * dim - 0.5;
  let lo = ceil(${centerAxis} - 2.0*ratio);
  let hi = floor(${centerAxis} + 2.0*ratio);
  let step = srcPt.${isV ? "y" : "x"};
  var acc = vec4f(0.0); var W = 0.0;
  for (var k = lo; k <= hi; k += 1.0) {
    let w = mn(abs((k - ${centerAxis}) / ratio));
    var p = uv;
    p.${isV ? "y" : "x"} = (k + 0.5) * step;
    var s = textureSampleLevel(src, samp, p, 0.0);
    ${square}
    acc += w * s;
    W += w;
  }
  if (abs(W) <= NUM_EPS) {
    var fallback = textureSampleLevel(src, samp, uv, 0.0);
    ${isV ? "fallback = fallback * fallback;" : ""}
    return fallback;
  }
  return acc / W;
}`;
}

// P3 — mean & R. Reads mean (E[x]) and L2 (E[x^2]) at display res, computes the
// variance ratio R with a separable locality kernel (taps=3). Output vec4(mean.rgb, R).
export const SSIMDS_MR_WGSL = HELPERS + VS + /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var meanTex : texture_2d<f32>;  // POSTKERNEL / HOOKED
@group(0) @binding(2) var l2Tex   : texture_2d<f32>;  // E[x^2]
const sigma_nsq : f32 = 10.0 / (255.0*255.0);
const oversharp : f32 = 0.0;

@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(meanTex));
  let pt = 1.0 / dim;
  // separable locality kernel, taps=3 -> k in [-1,1] (ceil(-1.5)..floor(1.5))
  var avg0 = vec3f(0.0); // E[x]
  var avg1 = vec3f(0.0); // E[x]^2 accumulation of mean*mean
  var avg2 = vec3f(0.0); // E[x^2]
  var W = 0.0;
  for (var ky = -1.0; ky <= 1.0; ky += 1.0) {
    let wy = loc(ky);
    for (var kx = -1.0; kx <= 1.0; kx += 1.0) {
      let wx = loc(kx);
      let w = wx * wy;
      let p = uv + vec2f(kx, ky) * pt;
      let L = textureSampleLevel(meanTex, samp, p, 0.0).rgb;
      let e2 = textureSampleLevel(l2Tex, samp, p, 0.0).rgb;
      avg0 += w * L;
      avg1 += w * (L * L);
      avg2 += w * e2;
      W += w;
    }
  }
  avg0 /= W; avg1 /= W; avg2 /= W;
  let Sl = luma(max(avg1 - avg0 * avg0, vec3f(0.0)));
  let Sh = luma(max(avg2 - avg0 * avg0, vec3f(0.0)));
  // select evaluates both value operands. Keep the low-variance branch finite on
  // a uniform field (Sl=Sh=0), even though the other branch is selected there.
  let varianceRatio = clamp(Sh / max(Sl, NUM_EPS), 0.0, 1.0);
  let R = select(
    sqrt((Sh + sigma_nsq) / (Sl + sigma_nsq)) * (1.0 + oversharp),
    varianceRatio,
    Sl > Sh
  );
  // store the local mean (avg0) and R
  return vec4f(avg0, R);
}
`;

// P4 — final reconstruction. Reads mean (HOOKED) and MR, applies the linear
// correction: out = avg[1] + avg[2]*L - avg[0], where the mat3 rows are
// (R*mean, mean, R) smoothed with the locality kernel.
export const SSIMDS_FINAL_WGSL = HELPERS + VS + /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var meanTex : texture_2d<f32>;  // HOOKED (E[x] at display res)
@group(0) @binding(2) var mrTex   : texture_2d<f32>;  // vec4(mean.rgb, R)

@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(meanTex));
  let pt = 1.0 / dim;
  // avg is a mat3: row0 = R*mean, row1 = mean, row2 = R (broadcast)
  var row0 = vec3f(0.0); var row1 = vec3f(0.0); var row2 = vec3f(0.0);
  var W = 0.0;
  for (var ky = -1.0; ky <= 1.0; ky += 1.0) {
    let wy = loc(ky);
    for (var kx = -1.0; kx <= 1.0; kx += 1.0) {
      let wx = loc(kx);
      let w = wx * wy;
      let p = uv + vec2f(kx, ky) * pt;
      let MR = textureSampleLevel(mrTex, samp, p, 0.0);
      row0 += w * (MR.a * MR.rgb);
      row1 += w * MR.rgb;
      row2 += w * vec3f(MR.a);
      W += w;
    }
  }
  row0 /= W; row1 /= W; row2 /= W;
  let L = textureSampleLevel(meanTex, samp, uv, 0.0);
  // out = row1 + row2*L.rgb - row0
  let outRgb = row1 + row2 * L.rgb - row0;
  return vec4f(clamp(outRgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

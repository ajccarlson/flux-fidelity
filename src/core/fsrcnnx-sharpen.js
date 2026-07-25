// Copyright (c) 2015-2021, bacondither
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer
//    in this position and unchanged.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// fsrcnnx-sharpen.js — adaptive-sharpen, hand-ported from bacondither's shader
// (version 2021-10-17, "tuned for use post-resize").
// Local WebGPU/WGSL port and numerical-safety modifications: 2026.
//
// Hooks OUTPUT in mpv: runs last, on the final RGB. In our pipeline it's the
// final pass before the canvas — after SSimDownscaler if it ran, else after
// recombine. Edge-adaptive luma sharpening with anti-ringing (soft-limited).
//
// Translation notes vs GLSL:
//  - fwidth() is supported in WebGPU fragment shaders -> dxdy maps directly.
//  - get(x,y) neighbor fetch -> textureSampleLevel at uv + offset*pt.
//  - no separate LUMA bind, so CtL computes luma from RGB (the #else branch).
//  - curve_height is exposed as a pipeline-overridable constant (strength).

export function buildSharpenShader(curveHeight = 1.0, overshootCtrl = false) {
  let requested;
  try { requested = Number(curveHeight); } catch { requested = NaN; }
  // Keep generated WGSL finite even when this builder is called directly or a
  // corrupted persisted value bypasses the public-setting normalization.
  const normalized = Number.isFinite(requested)
    ? Math.max(0.1, Math.min(2.0, requested))
    : 1.0;
  const ch = normalized.toFixed(4);
  const oc = overshootCtrl ? "true" : "false";
  return /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var tex : texture_2d<f32>;

const curve_height : f32 = ${ch};
const curveslope   : f32 = 0.5;
const L_compr_low  : f32 = 0.167;
const L_compr_high : f32 = 0.334;
const D_compr_low  : f32 = 0.250;
const D_compr_high : f32 = 0.500;
const scale_lim    : f32 = 0.1;
const scale_cs     : f32 = 0.056;
const pm_p         : f32 = 1.0;
const NUM_EPS      : f32 = 1.0e-6;

fn sat1(x : f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x : vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }

fn ctl(rgb : vec3f) -> f32 {
  let s = sat3(rgb);
  return sqrt(dot(s * s, vec3f(0.2126, 0.7152, 0.0722)));
}
fn dxdy(v : vec3f) -> f32 { return length(fwidth(v)); }

fn soft_lim(v : f32, s : f32) -> f32 {
  // A perfectly flat neighborhood has no permitted overshoot distance. Returning
  // zero preserves that field and avoids both 0/0 and an unbounded v/s ratio.
  if (s <= NUM_EPS) { return 0.0; }
  let r = v / s;
  // r may be negative; r*r is finite and well-defined where pow(r, 2) is not
  // guaranteed to be for a negative base on every shader implementation.
  let r2 = r * r;
  return sat1(abs(r) * (27.0 + r2) / (27.0 + 9.0 * r2)) * s;
}
fn wpmean(a : f32, b : f32, w : f32) -> f32 {
  return pow(w * pow(abs(a), pm_p) + abs(1.0 - w) * pow(abs(b), pm_p), 1.0 / pm_p);
}

struct VsOut { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  var p = array<vec2f,3>(vec2f(-1.,-3.), vec2f(-1.,1.), vec2f(3.,1.));
  var u = array<vec2f,3>(vec2f(0.,2.), vec2f(0.,0.), vec2f(2.,0.));
  var o : VsOut; o.pos = vec4f(p[i], 0., 1.); o.uv = u[i]; return o;
}

@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let pt = 1.0 / vec2f(textureDimensions(tex));
  // neighbor fetch helper inlined: g(x,y)
  // layout indices match the GLSL c[25] ordering.
  var c : array<vec3f, 25>;
  let offs = array<vec2f, 25>(
    vec2f( 0, 0), vec2f(-1,-1), vec2f( 0,-1), vec2f( 1,-1), vec2f(-1, 0),
    vec2f( 1, 0), vec2f(-1, 1), vec2f( 0, 1), vec2f( 1, 1), vec2f( 0,-2),
    vec2f(-2, 0), vec2f( 2, 0), vec2f( 0, 2), vec2f( 0, 3), vec2f( 1, 2),
    vec2f(-1, 2), vec2f( 3, 0), vec2f( 2, 1), vec2f( 2,-1), vec2f(-3, 0),
    vec2f(-2, 1), vec2f(-2,-1), vec2f( 0,-3), vec2f( 1,-2), vec2f(-1,-2));
  for (var i = 0; i < 25; i++) {
    c[i] = textureSampleLevel(tex, samp, uv + offs[i] * pt, 0.0).rgb;
  }

  var e : array<f32, 13>;
  for (var i = 0; i < 13; i++) { e[i] = dxdy(c[i]); }
  // Extra edge magnitudes needed by the overshoot branch. fwidth() must be called
  // in uniform control flow, so compute these here unconditionally (cheap) rather
  // than inside the if. Indices match c[] (13..24).
  let ex13 = dxdy(c[13]); let ex14 = dxdy(c[14]); let ex15 = dxdy(c[15]);
  let ex16 = dxdy(c[16]); let ex17 = dxdy(c[17]); let ex18 = dxdy(c[18]);
  let ex19 = dxdy(c[19]); let ex20 = dxdy(c[20]); let ex21 = dxdy(c[21]);
  let ex22 = dxdy(c[22]); let ex23 = dxdy(c[23]); let ex24 = dxdy(c[24]);

  var luma : array<f32, 25>;
  for (var i = 0; i < 25; i++) { luma[i] = ctl(c[i]); }

  let c0_Y = luma[0];

  let blur = (2.0 * (luma[2]+luma[4]+luma[5]+luma[7]) + (luma[1]+luma[3]+luma[6]+luma[8]) + 4.0 * luma[0]) / 16.0;
  let c_comp = sat1(0.266666681 + 0.9 * exp2(blur * blur * -7.4));

  // b_diff(pix) = (blur - luma[pix])^2
  let bd0 = (blur - luma[0]) * (blur - luma[0]);
  let bd1 = (blur - luma[1]) * (blur - luma[1]);
  let bd2 = (blur - luma[2]) * (blur - luma[2]);
  let bd3 = (blur - luma[3]) * (blur - luma[3]);
  let bd4 = (blur - luma[4]) * (blur - luma[4]);
  let bd5 = (blur - luma[5]) * (blur - luma[5]);
  let bd6 = (blur - luma[6]) * (blur - luma[6]);
  let bd7 = (blur - luma[7]) * (blur - luma[7]);
  let bd8 = (blur - luma[8]) * (blur - luma[8]);
  let bd9 = (blur - luma[9]) * (blur - luma[9]);
  let bd10 = (blur - luma[10]) * (blur - luma[10]);
  let bd11 = (blur - luma[11]) * (blur - luma[11]);
  let bd12 = (blur - luma[12]) * (blur - luma[12]);

  let edge = ( 1.38*bd0
             + 1.15*(bd2 + bd4 + bd5 + bd7)
             + 0.92*(bd1 + bd3 + bd6 + bd8)
             + 0.23*(bd9 + bd10 + bd11 + bd12) ) * c_comp;

  var cs = vec2f(L_compr_low, D_compr_low);

  if (${oc}) {
    let maxedge = max(max(max(max(e[1],e[2]),max(e[3],e[4])), max(max(e[5],e[6]),max(e[7],e[8]))),
                      max(max(max(e[9],e[10]),max(e[11],e[12])), e[0]));
    // soft_if(a,b,c) = sat((a+b+c + 0.056/2.5)/(maxedge + 0.03/2.5) - 0.85)
    let k = 0.056/2.5; let m = maxedge + 0.03/2.5;
    let sbe =
        sat1((e[2]+e[9]+ex22 + k)/m - 0.85) * sat1((e[7]+e[12]+ex13 + k)/m - 0.85)
      + sat1((e[4]+e[10]+ex19 + k)/m - 0.85) * sat1((e[5]+e[11]+ex16 + k)/m - 0.85)
      + sat1((e[1]+ex24+ex21 + k)/m - 0.85) * sat1((e[8]+ex14+ex17 + k)/m - 0.85)
      + sat1((e[3]+ex23+ex18 + k)/m - 0.85) * sat1((e[6]+ex20+ex15 + k)/m - 0.85);
    cs = mix(cs, vec2f(L_compr_high, D_compr_high), sat1(2.4002*sbe - 2.282));
  }

  let w1 = vec3f(0.5, 1.0, 1.41421356237);
  let w2 = vec3f(0.86602540378, 1.0, 0.54772255751);
  let dW = pow(mix(w1, w2, vec3f(sat1(2.4*edge - 0.82))), vec3f(2.0));

  let modif_e0 = 3.0 * e[0] + 0.02/2.5;

  var weights : array<f32, 12>;
  weights[0]  = min(modif_e0/max(e[1], NUM_EPS),  dW.y);
  weights[1]  = dW.x;
  weights[2]  = min(modif_e0/max(e[3], NUM_EPS),  dW.y);
  weights[3]  = dW.x;
  weights[4]  = dW.x;
  weights[5]  = min(modif_e0/max(e[6], NUM_EPS),  dW.y);
  weights[6]  = dW.x;
  weights[7]  = min(modif_e0/max(e[8], NUM_EPS),  dW.y);
  weights[8]  = min(modif_e0/max(e[9], NUM_EPS),  dW.z);
  weights[9]  = min(modif_e0/max(e[10], NUM_EPS), dW.z);
  weights[10] = min(modif_e0/max(e[11], NUM_EPS), dW.z);
  weights[11] = min(modif_e0/max(e[12], NUM_EPS), dW.z);

  weights[0] = (max(max((weights[8]  + weights[9])/4.0,  weights[0]), 0.25) + weights[0])/2.0;
  weights[2] = (max(max((weights[8]  + weights[10])/4.0, weights[2]), 0.25) + weights[2])/2.0;
  weights[5] = (max(max((weights[9]  + weights[11])/4.0, weights[5]), 0.25) + weights[5])/2.0;
  weights[7] = (max(max((weights[10] + weights[11])/4.0, weights[7]), 0.25) + weights[7])/2.0;

  var lowthrsum = 0.0; var weightsum = 0.0; var neg_laplace = 0.0;
  for (var pix = 0; pix < 12; pix++) {
    let lowthr = sat1(20.0*4.5*c_comp*e[pix + 1] - 0.221);
    neg_laplace += luma[pix+1] * luma[pix+1] * weights[pix] * lowthr;
    weightsum   += weights[pix] * lowthr;
    lowthrsum   += lowthr / 12.0;
  }
  // Flat fields make every lowthr zero, hence both accumulators are zero. Their
  // mathematically neutral result is the center luma: sharpdiff then remains zero.
  if (weightsum > NUM_EPS) {
    neg_laplace = sqrt(max(neg_laplace, 0.0) / weightsum);
  } else {
    neg_laplace = c0_Y;
  }

  let sharpen_val = curve_height/(curve_height*curveslope*edge + 0.625);
  var sharpdiff = (c0_Y - neg_laplace)*(lowthrsum*sharpen_val + 0.01);

  // Partial sort of luma[] to find local near-min/max (bubble-ish, as in source).
  var temp : f32;
  for (var i1 = 0; i1 < 24; i1 += 2) {
    temp = luma[i1];
    luma[i1]   = min(luma[i1], luma[i1+1]);
    luma[i1+1] = max(temp, luma[i1+1]);
  }
  for (var i2 = 24; i2 > 0; i2 -= 2) {
    temp = luma[0];
    luma[0]  = min(luma[0], luma[i2]);
    luma[i2] = max(temp, luma[i2]);
    temp = luma[24];
    luma[24]   = max(luma[24], luma[i2-1]);
    luma[i2-1] = min(temp, luma[i2-1]);
  }

  var min_dist = min(abs(luma[24] - c0_Y), abs(c0_Y - luma[0]));
  min_dist = min(min_dist, scale_lim*(1.0 - scale_cs) + min_dist*scale_cs);

  sharpdiff = wpmean(max(sharpdiff, 0.0), soft_lim(max(sharpdiff, 0.0), min_dist), cs.x)
            - wpmean(min(sharpdiff, 0.0), soft_lim(min(sharpdiff, 0.0), min_dist), cs.y);

  let sharpdiff_lim = sat1(c0_Y + sharpdiff) - c0_Y;
  let a = textureSampleLevel(tex, samp, uv, 0.0).a;
  return vec4f(vec3f(sharpdiff_lim) + c[0], a);
}
`;
}

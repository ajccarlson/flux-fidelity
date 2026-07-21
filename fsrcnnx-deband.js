// fsrcnnx-deband.js — debanding pass, modeled on mpv's f_deband / the classic
// haasn deband algorithm. Compressed web video shows visible banding in smooth
// gradients (skies, shadows, fades); debanding detects near-flat regions and
// smooths the quantization steps, optionally adding a touch of grain to mask
// residual banding.
//
// Algorithm (per pixel, `iterations` rounds at increasing radius):
//   - sample 4 points at a pseudo-random rotated offset at the current radius
//   - if all 4 are within `threshold` of the center (i.e. we're in a flat
//     region, not on an edge), replace the center with their average
//   - edges (where neighbors differ by > threshold) are left untouched
//   - finally add signed grain of magnitude `grain`
//
// Runs as a final-stage fragment pass on the RGB result, like adaptive-sharpen.
// Parameters mirror mpv defaults: threshold ~ 0.004*range, range grows per
// iteration, grain ~ small. We expose strength as a single slider that scales
// threshold + grain together.

export function buildDebandShader(strength = 1.0) {
  // strength 0.5..2.0 scales the smoothing aggressiveness.
  const s = Math.max(0.1, Math.min(3.0, strength));
  const threshold = (0.004 * s).toFixed(6);   // higher => smooths bigger steps
  const range = (16.0 * s).toFixed(3);         // sampling radius in pixels (grows per iter)
  const grain = (0.006 * s).toFixed(6);        // dither grain magnitude
  return /* wgsl */ `
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var<uniform> uTime : f32;

const ITER : i32 = 3;
const THRESHOLD : f32 = ${threshold};
const RANGE : f32 = ${range};
const GRAIN : f32 = ${grain};

// cheap hash -> [0,1)
fn h12(p : vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
fn rand2(co : vec2f) -> vec2f { return vec2f(h12(co), h12(co + 17.13)); }

struct VsOut { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  var p = array<vec2f,3>(vec2f(-1.,-3.), vec2f(-1.,1.), vec2f(3.,1.));
  var u = array<vec2f,3>(vec2f(0.,2.), vec2f(0.,0.), vec2f(2.,0.));
  var o : VsOut; o.pos = vec4f(p[i], 0., 1.); o.uv = u[i]; return o;
}

fn sampleAt(uv : vec2f) -> vec3f { return textureSampleLevel(tex, samp, uv, 0.0).rgb; }

// one deband iteration at the given radius (in pixels)
fn debandIter(uv : vec2f, pt : vec2f, radius : f32, seed : vec2f, center : vec3f) -> vec3f {
  // random angle + radius for this pixel/iteration
  let r = rand2(uv * 1024.0 + seed);
  let angle = r.x * 6.2831853;
  let dist = (0.5 + 0.5 * r.y) * radius;
  let o = vec2f(cos(angle), sin(angle)) * dist * pt;
  let oo = vec2f(-o.y, o.x); // perpendicular, for 4-tap cross
  let a = sampleAt(uv + o);
  let b = sampleAt(uv - o);
  let c = sampleAt(uv + oo);
  let d = sampleAt(uv - oo);
  let avg = (a + b + c + d) * 0.25;
  // per-channel: only smooth where the spread is below threshold (flat region)
  let diff = max(max(abs(a - center), abs(b - center)), max(abs(c - center), abs(d - center)));
  let flat = step(diff, vec3f(THRESHOLD)); // 1 where flat, 0 where edge
  return mix(center, avg, flat);
}

@fragment fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let pt = 1.0 / vec2f(textureDimensions(tex));
  var col = sampleAt(uv);
  let seed = vec2f(uTime, uTime * 1.37);
  // iterate at growing radii (mpv increases the radius each pass)
  for (var i = 0; i < ITER; i++) {
    let radius = RANGE * f32(i + 1) / f32(ITER);
    col = debandIter(uv, pt, radius, seed + vec2f(f32(i) * 3.7), col);
  }
  // signed grain to mask residual banding (ordered by hash, time-varying)
  let g = (h12(uv * 2048.0 + seed) - 0.5) * 2.0 * GRAIN;
  col += vec3f(g);
  let alpha = textureSampleLevel(tex, samp, uv, 0.0).a;
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), alpha);
}
`;
}

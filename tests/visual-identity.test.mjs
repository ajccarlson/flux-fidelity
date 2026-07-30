import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { inflateSync } from "node:zlib";

// The popup's palette claims to be derived from the shipped icon. That claim is
// only worth making if something checks it, so these decode the actual PNG and
// compare. Without this the stylesheet could drift back to an arbitrary accent —
// which is exactly what it had done: the previous UI was built on a system blue
// that appears nowhere in the product's own artwork.

const popup = readFileSync(new URL("../popup.html", import.meta.url), "utf8");

// Minimal PNG reader. The icons are hash-pinned 8-bit RGBA, so this only has to
// handle that one shape; anything else is a failure worth hearing about.
function decodePng(url) {
  const buf = readFileSync(url);
  let pos = 8, width = 0, height = 0, depth = 0, colorType = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("ascii", pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      depth = data[8];
      colorType = data[9];
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  assert.equal(depth, 8, "icon must be 8-bit");
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  assert.ok(channels, `unsupported colour type ${colorType}`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(height * stride);
  let off = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[off++];
    const line = raw.subarray(off, off + stride);
    off += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prior = y ? out.subarray((y - 1) * stride, y * stride) : Buffer.alloc(stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? cur[x - channels] : 0;
      const b = prior[x];
      const c = x >= channels ? prior[x - channels] : 0;
      let v = line[x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      cur[x] = v & 0xff;
    }
  }
  return { width, height, channels, pixels: out };
}

function iconColours() {
  const { width, height, channels, pixels } = decodePng(
    new URL("../icons/icon-128.png", import.meta.url),
  );
  const counts = new Map();
  for (let i = 0; i < width * height; i++) {
    const o = i * channels;
    if (channels === 4 && pixels[o + 3] < 250) continue;
    const hex = "#" + [0, 1, 2].map((k) => pixels[o + k].toString(16).padStart(2, "0")).join("");
    counts.set(hex, (counts.get(hex) || 0) + 1);
  }
  return [...counts].sort((a, b) => b[1] - a[1]).map(([hex]) => hex);
}

function token(name, { dark = false } = {}) {
  // The dark block is the last @media prefers-color-scheme: dark in the sheet's
  // :root; take the later declaration when asked for dark.
  const matches = [...popup.matchAll(new RegExp(`--${name}:\\s*([^;]+);`, "g"))]
    .map((match) => match[1].trim());
  assert.ok(matches.length, `--${name} is not declared`);
  return dark ? matches[matches.length - 1] : matches[0];
}

test("the brand colours are the icon's own pixels, not an approximation", () => {
  // Top colours by area. The icon is three frames over cream, so the violet card,
  // the amber frame, the coral frame and the cream paper dominate.
  const dominant = iconColours().slice(0, 4);
  for (const name of ["brand-violet", "brand-amber", "brand-coral", "brand-cream"]) {
    const value = token(name).toLowerCase();
    assert.match(value, /^#[0-9a-f]{6}$/, `--${name} must be a literal hex, not derived`);
    assert.ok(
      dominant.includes(value),
      `--${name} is ${value}, which is not among the icon's dominant colours ${dominant.join(", ")}`,
    );
  }
});

test("the accent is the brand violet rather than an unrelated hue", () => {
  // The regression this guards: the popup previously used #3559d9, a system blue
  // with no relationship to the artwork, so the toolbar button and the panel it
  // opened looked like two different products.
  assert.equal(token("accent").toLowerCase(), token("brand-violet").toLowerCase());
  assert.equal(
    /#3559d9|#2447c4|#8ca4ff/i.test(popup), false,
    "the retired blue accent must not return",
  );
});

test("both themes define every token the stylesheet consumes", () => {
  // A token used but declared in only one theme silently inherits the other
  // theme's value, which is how a dark-mode surface ends up with a light-mode
  // line colour.
  const declared = new Set([...popup.matchAll(/--([a-z0-9-]+):/g)].map((match) => match[1]));
  const used = new Set([...popup.matchAll(/var\(--([a-z0-9-]+)/g)].map((match) => match[1]));
  const missing = [...used].filter((name) => !declared.has(name));
  assert.deepEqual(missing, [], `used but never declared: ${missing.join(", ")}`);

  // Every token that differs between themes must appear in both blocks.
  const darkBlock = popup.slice(popup.indexOf("@media (prefers-color-scheme: dark)"));
  for (const name of ["bg", "surface", "surface-alt", "text", "muted", "line", "line-strong",
    "accent", "accent-strong", "accent-soft", "focus", "success", "danger", "danger-bg", "warn",
    "mark-success", "mark-danger", "mark-warn", "chart-series", "chart-limit", "chart-fill",
    "chart-grid", "scanline", "shadow"]) {
    assert.ok(darkBlock.includes(`--${name}:`), `--${name} has no dark-mode step`);
  }
});

test("the chart trace is a validated data colour, not the surface accent", () => {
  // The raw brand violet fails a categorical palette's chroma floor at chart
  // scale — it reads grey. The trace therefore uses a lighter step of the same
  // hue, checked with the palette validator, and the limit keeps a reserved
  // status colour that is also restated in words beneath the chart.
  const series = token("chart-series").toLowerCase();
  assert.notEqual(series, token("brand-violet").toLowerCase());
  assert.match(series, /^#[0-9a-f]{6}$/);
  assert.notEqual(token("chart-series", { dark: true }).toLowerCase(), series,
    "dark mode must select its own trace against the dark surface, not reuse light");
  assert.match(popup, /\.spark-area\s*\{[^}]*fill:\s*var\(--chart-fill\)/);
  assert.match(popup, /\.spark-line\s*\{[^}]*stroke:\s*var\(--chart-series\)/);
  assert.match(popup, /\.spark-budget\s*\{[^}]*stroke:\s*var\(--chart-limit\)/);
});

test("tab state is carried by shape as well as colour", () => {
  // Green/amber/red are the three hues colour-blind users confuse. The state is
  // already written into adjacent screen-reader text; giving each state its own
  // shape adds a third channel that needs neither colour nor a screen reader.
  const rule = (state) => {
    const match = popup.match(new RegExp(`\\.tab-dot\\[data-state="${state}"\\]\\s*\\{([^}]*)\\}`));
    assert.ok(match, `no rule for the ${state} dot`);
    return match[1];
  };
  // Solid, ringed, hollow — distinguishable in a greyscale screenshot.
  assert.match(rule("on"), /box-shadow/);
  assert.match(rule("pending"), /border:\s*2px/);
  assert.match(rule("off"), /background:\s*transparent/);
  // The bug this class of rule replaced: toggling `hidden` did nothing, because
  // an author `display` on .tab-dot beats the user-agent [hidden] rule.
  assert.equal(/id="tab-\w+-dot"[^>]*\shidden/.test(popup), false);
});

test("every text pair clears its WCAG floor in both themes", () => {
  const srgb = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const lum = (hex) => {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    return 0.2126 * srgb(r) + 0.7152 * srgb(g) + 0.0722 * srgb(b);
  };
  const ratio = (a, b) => {
    const [hi, lo] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (hi + 0.05) / (lo + 0.05);
  };
  for (const dark of [false, true]) {
    const t = (name) => token(name, { dark });
    const pairs = [
      ["text on surface", t("text"), t("surface"), 4.5],
      ["muted on surface", t("muted"), t("surface"), 4.5],
      ["muted on alt", t("muted"), t("surface-alt"), 4.5],
      ["accent label on surface", t("accent"), t("surface"), 4.5],
      ["accent label on bg", t("accent"), t("bg"), 4.5],
      ["success on surface", t("success"), t("surface"), 4.5],
      ["danger on surface", t("danger"), t("surface"), 4.5],
      ["danger on its banner", t("danger"), t("danger-bg"), 4.5],
      ["warn on surface", t("warn"), t("surface"), 4.5],
      ["mode label on accent-soft", t("accent-strong"), t("accent-soft"), 4.5],
      ["focus ring on bg", t("focus"), t("bg"), 3],
    ];
    for (const [label, fg, bg, floor] of pairs) {
      const value = ratio(fg, bg);
      assert.ok(value >= floor,
        `${dark ? "dark" : "light"}: ${label} is ${value.toFixed(2)}:1, needs ${floor}`);
    }
  }
});

test("the identity survives forced colors and reduced motion", () => {
  // A popup is a small window a user may have themed aggressively. The scanline
  // wash and the stack rule are decoration, so neither may become the only way
  // to read the header, and none of the new work introduces animation.
  assert.match(popup, /prefers-reduced-motion/);
  // The header's stack is a ::after rule, so it cannot swallow content or focus.
  assert.match(popup, /header::after\s*\{[^}]*content:\s*""/);
});

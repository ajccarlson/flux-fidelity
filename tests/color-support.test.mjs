import assert from "node:assert/strict";
import test from "node:test";
import { readdir, readFile } from "node:fs/promises";

import {
  SRGB_COLOR_SPACE,
  classifyVideoColorSpace,
  probeVideoColorSupport,
} from "../fsrcnnx-color-support.js";

const projectUrl = new URL("../", import.meta.url);
const mainUrl = new URL("../fsrcnnx-main.js", import.meta.url);
let revision = 0;

function metadata(overrides = {}) {
  return {
    primaries: "bt709",
    transfer: "bt709",
    matrix: "bt709",
    fullRange: false,
    ...overrides,
  };
}

function support(code, supported = code === "color-supported") {
  return Object.freeze({ supported, code, detail: code, colorSpace: Object.freeze(metadata()) });
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

async function loadMainColorPolicy(deps) {
  const source = await readFile(mainUrl, "utf8");
  const sourceIdentity = section(
    source,
    "function captureVideoSource(target)",
    "function interpolationRuntimeConfigKey()",
  );
  const policy = section(source, "function probeVideoAccess(v)", "function safeImportExternal()");
  const harness = `
    const deps = globalThis.__colorPolicyDeps;
    const COLOR_REPROBE_INTERVAL_MS = 2000;
    const SRGB_COLOR_SPACE = "srgb";
    const Date = { now: () => deps.now };
    const document = {
      createElement() {
        return {
          width: 0,
          height: 0,
          getContext() {
            return {
              drawImage(target) {
                if (target?.tainted) throw Object.assign(new Error("tainted"), { name: "SecurityError" });
              },
              getImageData() { return new Uint8ClampedArray(64); },
            };
          },
        };
      },
    };
    let video = deps.video;
    let videoColorSupportCache = new WeakMap();
    const uncheckedColorSupport = (detail) => ({
      supported: false,
      code: "color-not-checked",
      detail,
      colorSpace: { primaries: null, transfer: null, matrix: null, fullRange: null },
    });
    let selectedColorSupport = uncheckedColorSupport("not checked");
    const probeVideoColorSupport = (target) => deps.probe(target);
    ${sourceIdentity}
    ${policy}
    export { invalidateVideoColorSupport, probeVideo };
    export function selected() { return selectedColorSupport; }
  `;
  globalThis.__colorPolicyDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

test("the processing boundary is explicit sRGB", () => {
  assert.equal(SRGB_COLOR_SPACE, "srgb");
});

for (const transfer of ["bt709", "iec61966-2-1"]) {
  for (const matrix of ["bt709", "rgb"]) {
    for (const fullRange of [false, true]) {
      test(`BT.709 SDR ${transfer}/${matrix}/${fullRange ? "full" : "limited"} is supported`, () => {
        const result = classifyVideoColorSpace(metadata({ transfer, matrix, fullRange }));
        assert.equal(result.supported, true);
        assert.equal(result.code, "color-supported");
        assert.equal(result.colorSpace.fullRange, fullRange);
        assert.equal(Object.isFrozen(result), true);
        assert.equal(Object.isFrozen(result.colorSpace), true);
      });
    }
  }
}

for (const transfer of ["pq", "hlg", "smpte2084", "arib-std-b67"]) {
  test(`${transfer} is rejected as HDR even when other metadata is incomplete`, () => {
    const result = classifyVideoColorSpace({ transfer });
    assert.equal(result.supported, false);
    assert.equal(result.code, "color-hdr-unsupported");
    assert.match(result.detail, /HDR/i);
  });
}

for (const primaries of ["bt2020", "smpte432"]) {
  test(`${primaries} is rejected as wide gamut even when other metadata is incomplete`, () => {
    const result = classifyVideoColorSpace({ primaries });
    assert.equal(result.supported, false);
    assert.equal(result.code, "color-wide-gamut-unsupported");
    assert.match(result.detail, /wide-gamut/i);
  });
}

for (const missing of ["primaries", "transfer", "matrix", "fullRange"]) {
  test(`missing ${missing} fails closed as unavailable metadata`, () => {
    const value = metadata();
    delete value[missing];
    const result = classifyVideoColorSpace(value);
    assert.equal(result.supported, false);
    assert.equal(result.code, "color-metadata-unavailable");
    assert.equal(result.colorSpace[missing], null);
  });
}

test("complete non-BT.709 SDR metadata is rejected generically", () => {
  for (const value of [
    metadata({ primaries: "smpte170m" }),
    metadata({ transfer: "linear" }),
    metadata({ matrix: "smpte170m" }),
  ]) {
    const result = classifyVideoColorSpace(value);
    assert.equal(result.supported, false);
    assert.equal(result.code, "color-space-unsupported");
  }
});

test("the frame probe fails closed when VideoFrame is unavailable or construction fails", () => {
  assert.equal(
    probeVideoColorSupport({}, { VideoFrame: undefined }).code,
    "color-metadata-unavailable",
  );
  function ThrowingVideoFrame() {
    const error = new Error("not ready");
    error.name = "InvalidStateError";
    throw error;
  }
  const failed = probeVideoColorSupport({}, { VideoFrame: ThrowingVideoFrame });
  assert.equal(failed.code, "color-metadata-unavailable");
  assert.equal(failed.errorName, "InvalidStateError");
});

test("every constructed VideoFrame closes on supported, unsupported, and exceptional paths", () => {
  let closes = 0;
  class SupportedFrame {
    constructor(video) { assert.equal(video.id, "video"); }
    colorSpace = metadata();
    close() { closes++; }
  }
  assert.equal(
    probeVideoColorSupport({ id: "video" }, { VideoFrame: SupportedFrame }).code,
    "color-supported",
  );

  class UnsupportedFrame {
    colorSpace = metadata({ transfer: "pq" });
    close() { closes++; }
  }
  assert.equal(
    probeVideoColorSupport({}, { VideoFrame: UnsupportedFrame }).code,
    "color-hdr-unsupported",
  );

  class ThrowingGetterFrame {
    get colorSpace() { throw Object.assign(new Error("metadata failure"), { name: "DataError" }); }
    close() { closes++; }
  }
  const exceptional = probeVideoColorSupport({}, { VideoFrame: ThrowingGetterFrame });
  assert.equal(exceptional.code, "color-metadata-unavailable");
  assert.equal(exceptional.errorName, "DataError");
  assert.equal(closes, 3);

  class ThrowingCloseFrame {
    colorSpace = metadata();
    close() { closes++; throw new Error("already closed"); }
  }
  assert.doesNotThrow(() => probeVideoColorSupport({}, { VideoFrame: ThrowingCloseFrame }));
  assert.equal(closes, 4);
});

test("main policy caches stable metadata, refreshes source changes, and honors invalidation", async (t) => {
  const previous = globalThis.__colorPolicyDeps;
  t.after(() => { globalThis.__colorPolicyDeps = previous; });
  const video = {
    currentSrc: "https://example.test/a.mp4",
    src: "https://example.test/a.mp4",
    videoWidth: 640,
    videoHeight: 360,
  };
  const deps = { video, now: 100, calls: 0, result: support("color-supported") };
  deps.probe = () => { deps.calls++; return deps.result; };
  const policy = await loadMainColorPolicy(deps);

  assert.equal(policy.probeVideo(video), "ok");
  assert.equal(policy.probeVideo(video), "ok");
  assert.equal(deps.calls, 1, "unchanged decoded metadata is cached");
  assert.equal(policy.selected().code, "color-supported");

  video.currentSrc = "https://example.test/b.mp4";
  video.src = video.currentSrc;
  deps.result = support("color-wide-gamut-unsupported", false);
  assert.equal(policy.probeVideo(video), "color-wide-gamut-unsupported");
  assert.equal(deps.calls, 2, "a reused element with a new resource is reclassified");

  policy.invalidateVideoColorSupport(video);
  assert.equal(policy.selected().code, "color-not-checked");
  deps.result = support("color-supported");
  assert.equal(policy.probeVideo(video), "ok");
  assert.equal(deps.calls, 3);

  assert.equal(policy.probeVideo(video, { forceColor: true }), "ok");
  assert.equal(deps.calls, 4, "periodic forced probes bypass a stable cache entry");
});

test("cached classifications expire on the bounded monitor cadence", async (t) => {
  const previous = globalThis.__colorPolicyDeps;
  t.after(() => { globalThis.__colorPolicyDeps = previous; });
  const video = { currentSrc: "a", src: "a", videoWidth: 640, videoHeight: 360 };
  const deps = {
    video,
    now: 0,
    calls: 0,
    result: support("color-metadata-unavailable", false),
  };
  deps.probe = () => { deps.calls++; return deps.result; };
  const policy = await loadMainColorPolicy(deps);

  assert.equal(policy.probeVideo(video), "color-metadata-unavailable");
  deps.now = 1999;
  assert.equal(policy.probeVideo(video), "color-metadata-unavailable");
  assert.equal(deps.calls, 1);

  deps.now = 2000;
  deps.result = support("color-supported");
  assert.equal(policy.probeVideo(video), "ok");
  assert.equal(deps.calls, 2);
});

test("a known unsupported stream can recover without changing element identity or URL", async (t) => {
  const previous = globalThis.__colorPolicyDeps;
  t.after(() => { globalThis.__colorPolicyDeps = previous; });
  const video = { currentSrc: "a", src: "a", videoWidth: 640, videoHeight: 360 };
  const deps = {
    video,
    now: 0,
    calls: 0,
    result: support("color-hdr-unsupported", false),
  };
  deps.probe = () => { deps.calls++; return deps.result; };
  const policy = await loadMainColorPolicy(deps);

  assert.equal(policy.probeVideo(video), "color-hdr-unsupported");
  deps.now = 1999;
  deps.result = support("color-supported");
  assert.equal(policy.probeVideo(video), "color-hdr-unsupported");
  assert.equal(deps.calls, 1);

  deps.now = 2000;
  assert.equal(policy.probeVideo(video), "ok");
  assert.equal(deps.calls, 2);
});

test("DRM and taint checks short-circuit frame metadata probing", async (t) => {
  const previous = globalThis.__colorPolicyDeps;
  t.after(() => { globalThis.__colorPolicyDeps = previous; });
  const video = { mediaKeys: {}, videoWidth: 640, videoHeight: 360 };
  const deps = { video, now: 0, calls: 0, probe: () => { deps.calls++; return support("color-supported"); } };
  const policy = await loadMainColorPolicy(deps);

  assert.equal(policy.probeVideo(video), "drm");
  assert.equal(deps.calls, 0);
  assert.equal(policy.selected().code, "color-not-checked");

  video.mediaKeys = null;
  video.tainted = true;
  assert.equal(policy.probeVideo(video), "tainted");
  assert.equal(deps.calls, 0);
});

test("every runtime WebGPU color boundary explicitly requests sRGB", async () => {
  const entries = await readdir(projectUrl, { withFileTypes: true });
  const sources = [];
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".js")) {
      sources.push({ name: entry.name, source: await readFile(new URL(entry.name, projectUrl), "utf8") });
    }
  }

  const externalImports = [];
  const canvasConfigurations = [];
  const externalCopies = [];
  const canvas2dContexts = [];
  for (const file of sources) {
    for (const match of file.source.matchAll(/importExternalTexture\s*\(\s*\{([^}]*)\}\s*\)/g)) {
      externalImports.push({ name: file.name, descriptor: match[1] });
    }
    for (const match of file.source.matchAll(/\.configure\s*\(\s*\{([^}]*)\}\s*\)/g)) {
      canvasConfigurations.push({ name: file.name, descriptor: match[1] });
    }
    for (const match of file.source.matchAll(/copyExternalImageToTexture\s*\(\s*\{[^}]*\}\s*,\s*\{([^}]*)\}/g)) {
      externalCopies.push({ name: file.name, descriptor: match[1] });
    }
    for (const match of file.source.matchAll(/getContext\s*\(\s*["']2d["']\s*(?:,\s*\{([^}]*)\})?\s*\)/g)) {
      canvas2dContexts.push({ name: file.name, descriptor: match[1] || "" });
    }
  }

  assert.equal(externalImports.length, 3, "all three video import paths are covered");
  assert.equal(canvasConfigurations.length, 4, "all four presentation canvases are covered");
  assert.equal(externalCopies.length, 1, "the image upload path is covered");
  assert.equal(canvas2dContexts.length, 11, "all 2D readback and presentation contexts are covered");
  for (const boundary of [
    ...externalImports,
    ...canvasConfigurations,
    ...externalCopies,
    ...canvas2dContexts,
  ]) {
    assert.match(
      boundary.descriptor,
      /colorSpace\s*:\s*SRGB_COLOR_SPACE/,
      `${boundary.name} leaves a WebGPU color boundary implicit`,
    );
  }
  assert.match(externalCopies[0].descriptor, /premultipliedAlpha\s*:\s*false/);

  const mainSource = sources.find(({ name }) => name === "fsrcnnx-main.js")?.source || "";
  assert.match(mainSource, /probeVideo\(video, \{ publish: !renderTargetOwner \}\)/,
    "a secondary import failure must not replace the primary color status");
  assert.match(
    mainSource,
    /function handlePrimarySourceBoundary[\s\S]*?invalidateVideoColorSupport\(owner\.video\)/,
    "primary media-resource boundaries must invalidate decoded color metadata",
  );
  assert.match(
    mainSource,
    /function handleSecondarySourceBoundary[\s\S]*?invalidateVideoColorSupport\(target\.video\)/,
    "secondary media-resource boundaries must invalidate decoded color metadata",
  );

  const colorShader = await readFile(new URL("../fsrcnnx-color.js", import.meta.url), "utf8");
  assert.doesNotMatch(colorShader, /limited\s*->\s*full handled implicitly/i);
  assert.match(colorShader, /source primaries, transfer, YUV matrix/);
  assert.match(colorShader, /does not perform source YUV range or primary conversion/);
});

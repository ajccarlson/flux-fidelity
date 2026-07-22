#!/usr/bin/env node

import { createHash } from "node:crypto";
import { lstatSync, readFileSync, readdirSync } from "node:fs";
import { isAbsolute, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const DEFAULT_FIXTURE_DIRECTORY = "tests/fixtures/browser";
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const EXPECTED_GENERATOR = Object.freeze({
  script: "tools/generate-browser-fixtures.mjs",
  ffmpegVersion: "n7.1.1",
  libvpxVersion: "1.15.0",
  source: "FFmpeg lavfi testsrc2",
  frames: 48,
  frameRate: 24,
});
const EXPECTED_CLIPS = Object.freeze({
  "bt709-a": Object.freeze({
    path: "media/bt709-a.webm", codec: "vp9", profile: 0, pixelFormat: "yuv420p",
    bytes: 30946, sha256: "64efbcbfaea35f0f4cc9079cad225a3a8966c1ffd77b5554fa07580a141b470b",
    width: 160, height: 90,
    encodedColor: Object.freeze({ primaries: "bt709", transfer: "bt709", matrix: "bt709", range: "tv" }),
    colorSpace: Object.freeze({ primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false }),
  }),
  "bt709-b": Object.freeze({
    path: "media/bt709-b.webm", codec: "vp9", profile: 0, pixelFormat: "yuv420p",
    bytes: 30613, sha256: "0e68236e2bc519d227e9f678b1c86b0d34ecab9c0bbfb4d7427f5c5a41cedfe7",
    width: 128, height: 72,
    encodedColor: Object.freeze({ primaries: "bt709", transfer: "bt709", matrix: "bt709", range: "tv" }),
    colorSpace: Object.freeze({ primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false }),
  }),
  "bt2020-pq": Object.freeze({
    path: "media/bt2020-pq.webm", codec: "vp9", profile: 2, pixelFormat: "yuv420p10le",
    bytes: 29352, sha256: "a760f871bd1b09eedbeefbc82d9b6eaf032dc7e2a37d18fa2b9ae4cd30539a47",
    width: 128, height: 72,
    encodedColor: Object.freeze({ primaries: "bt2020", transfer: "smpte2084", matrix: "bt2020nc", range: "tv" }),
    colorSpace: Object.freeze({ primaries: "bt2020", transfer: "pq", matrix: "bt2020-ncl", fullRange: false }),
  }),
  "bt2020-sdr": Object.freeze({
    path: "media/bt2020-sdr.webm", codec: "vp9", profile: 0, pixelFormat: "yuv420p",
    bytes: 29847, sha256: "bc7c0f10be3c25d2fcbbc0b4cbb27c251a85ef7ab76b29fc989d7663ee01c4c3",
    width: 128, height: 72,
    encodedColor: Object.freeze({ primaries: "bt2020", transfer: "bt709", matrix: "bt2020nc", range: "tv" }),
    colorSpace: Object.freeze({ primaries: "bt2020", transfer: "bt709", matrix: "bt2020-ncl", fullRange: false }),
  }),
});
const EXPECTED_IDS = Object.freeze(Object.keys(EXPECTED_CLIPS));

export const BROWSER_FIXTURE_CONTRACT = Object.freeze({
  schemaVersion: 1,
  apiGlobal: "__FSRCNNX_VIDEO_FIXTURE__",
  manifestFile: "fixture-manifest.json",
  ids: EXPECTED_IDS,
});

function objectRecord(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function safeRelativePath(path) {
  return typeof path === "string" && path.length > 0 && !isAbsolute(path) &&
    !path.includes("\\") && normalize(path) === path && !path.split("/").includes("..");
}

function regularFile(path, label, errors) {
  try {
    const metadata = lstatSync(path);
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      errors.push(`${label}: must be a regular, non-symlink file`);
      return null;
    }
    return metadata;
  } catch (error) {
    errors.push(`${label}: ${error.code || error.message}`);
    return null;
  }
}

function sameRecord(left, right) {
  return objectRecord(left) && Object.keys(right).every((key) => left[key] === right[key]) &&
    Object.keys(left).length === Object.keys(right).length;
}

export function inspectBrowserFixtures({
  rootDir = root,
  fixtureDirectory = DEFAULT_FIXTURE_DIRECTORY,
} = {}) {
  const errors = [];
  const fixtureRoot = resolve(rootDir, fixtureDirectory);
  const relativeFixtureRoot = relative(rootDir, fixtureRoot);
  if (isAbsolute(relativeFixtureRoot) || relativeFixtureRoot === ".." ||
      relativeFixtureRoot.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)) {
    return [`${fixtureDirectory}: fixture directory escapes the repository`];
  }

  const manifestPath = resolve(fixtureRoot, BROWSER_FIXTURE_CONTRACT.manifestFile);
  let manifest = null;
  if (regularFile(manifestPath, `${fixtureDirectory}/${BROWSER_FIXTURE_CONTRACT.manifestFile}`, errors)) {
    try { manifest = JSON.parse(readFileSync(manifestPath, "utf8")); }
    catch (error) { errors.push(`${fixtureDirectory}/${BROWSER_FIXTURE_CONTRACT.manifestFile}: ${error.message}`); }
  }
  if (!manifest) return errors;
  if (manifest.schemaVersion !== BROWSER_FIXTURE_CONTRACT.schemaVersion) {
    errors.push(`browser fixture manifest: schemaVersion must be ${BROWSER_FIXTURE_CONTRACT.schemaVersion}`);
  }
  if (manifest.initialSource !== EXPECTED_IDS[0]) {
    errors.push(`browser fixture manifest: initialSource must be ${EXPECTED_IDS[0]}`);
  }
  if (!sameRecord(manifest.generator, EXPECTED_GENERATOR)) {
    errors.push("browser fixture manifest: generator toolchain and source contract changed");
  }
  if (!Array.isArray(manifest.clips)) {
    errors.push("browser fixture manifest: clips must be an array");
    return errors;
  }
  const actualIds = manifest.clips.map((clip) => clip?.id);
  if (actualIds.join("\n") !== EXPECTED_IDS.join("\n")) {
    errors.push(`browser fixture manifest: expected ordered clips ${EXPECTED_IDS.join(", ")}`);
  }

  const expectedMediaNames = new Set();
  for (const clip of manifest.clips) {
    const label = `browser fixture ${clip?.id || "<missing-id>"}`;
    if (!EXPECTED_IDS.includes(clip?.id)) continue;
    const expectedClip = EXPECTED_CLIPS[clip.id];
    const expectedPath = expectedClip.path;
    if (clip.path !== expectedPath) errors.push(`${label}: path must be ${expectedPath}`);
    if (!safeRelativePath(clip.path) || !clip.path.startsWith("media/")) {
      errors.push(`${label}: path must be a normalized media-relative path`);
      continue;
    }
    expectedMediaNames.add(clip.path.slice("media/".length));
    if (!Number.isSafeInteger(clip.bytes) || clip.bytes <= 0 || clip.bytes > 512 * 1024) {
      errors.push(`${label}: bytes must be a positive integer no greater than 524288`);
    }
    if (!SHA256_PATTERN.test(clip.sha256 || "")) {
      errors.push(`${label}: sha256 must be 64 lowercase hexadecimal characters`);
    }
    if (clip.bytes !== expectedClip.bytes) {
      errors.push(`${label}: canonical byte length is ${expectedClip.bytes}, received ${clip.bytes ?? "missing"}`);
    }
    if (clip.sha256 !== expectedClip.sha256) {
      errors.push(`${label}: canonical SHA-256 is ${expectedClip.sha256}, received ${clip.sha256 || "missing"}`);
    }
    for (const field of ["codec", "profile", "pixelFormat", "width", "height"]) {
      if (clip[field] !== expectedClip[field]) {
        errors.push(`${label}: ${field} is ${clip[field] ?? "missing"}, expected ${expectedClip[field]}`);
      }
    }
    if (clip.frameRate !== 24 || clip.frames !== 48 || clip.durationMs !== 2000) {
      errors.push(`${label}: expected 48 frames at 24 fps over 2000 ms`);
    }
    if (!sameRecord(clip.encodedColor, expectedClip.encodedColor)) {
      errors.push(`${label}: encodedColor does not match the pinned stream metadata`);
    }
    if (!sameRecord(clip.colorSpace, expectedClip.colorSpace)) {
      errors.push(`${label}: decoded colorSpace does not match the pinned contract`);
    }
    const mediaPath = resolve(fixtureRoot, clip.path);
    const local = relative(fixtureRoot, mediaPath);
    if (isAbsolute(local) || local === ".." || local.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)) {
      errors.push(`${label}: path escapes the fixture directory`);
      continue;
    }
    const metadata = regularFile(mediaPath, `${fixtureDirectory}/${clip.path}`, errors);
    if (!metadata) continue;
    const data = readFileSync(mediaPath);
    if (metadata.size !== clip.bytes) {
      errors.push(`${label}: size is ${metadata.size}, expected ${clip.bytes}`);
    }
    const digest = createHash("sha256").update(data).digest("hex");
    if (digest !== clip.sha256) errors.push(`${label}: hash is ${digest}, expected ${clip.sha256}`);
    if (data.length < 4 || data.readUInt32BE(0) !== 0x1a45dfa3) {
      errors.push(`${label}: file does not begin with the EBML signature`);
    }
  }

  try {
    const actualMediaNames = readdirSync(resolve(fixtureRoot, "media")).sort();
    const expected = [...expectedMediaNames].sort();
    if (actualMediaNames.join("\n") !== expected.join("\n")) {
      errors.push(
        `browser fixture media: expected exactly ${expected.join(", ")}; ` +
        `found ${actualMediaNames.join(", ") || "none"}`,
      );
    }
  } catch (error) {
    errors.push(`browser fixture media: ${error.code || error.message}`);
  }

  const htmlPath = resolve(fixtureRoot, "video.html");
  const scriptPath = resolve(fixtureRoot, "video-fixture.js");
  const htmlOk = regularFile(htmlPath, `${fixtureDirectory}/video.html`, errors);
  const scriptOk = regularFile(scriptPath, `${fixtureDirectory}/video-fixture.js`, errors);
  if (htmlOk) {
    const html = readFileSync(htmlPath, "utf8");
    if (!/<script\s+type="module"\s+src="video-fixture\.js"><\/script>/i.test(html)) {
      errors.push("browser fixture HTML: expected the external module script");
    }
    if (!/<video\b[^>]*\bid="fixture-video"[^>]*>/is.test(html)) {
      errors.push("browser fixture HTML: missing #fixture-video");
    }
    if (/<script\b(?![^>]*\bsrc=)[^>]*>/i.test(html) || /\sstyle\s*=/i.test(html) ||
        /<(?:img|script|source|video)\b[^>]*(?:src|href)=["']https?:/i.test(html)) {
      errors.push("browser fixture HTML: inline or remote executable/presentation content is forbidden");
    }
  }
  if (scriptOk) {
    const script = readFileSync(scriptPath, "utf8");
    for (const method of ["ready", "loadSource", "pause", "play", "snapshot"]) {
      if (!new RegExp(`\\b${method}\\b`).test(script)) {
        errors.push(`browser fixture script: missing ${method} API member`);
      }
    }
    if (!script.includes(`"${BROWSER_FIXTURE_CONTRACT.apiGlobal}"`)) {
      errors.push(`browser fixture script: missing window.${BROWSER_FIXTURE_CONTRACT.apiGlobal}`);
    }
    for (const clip of manifest.clips) {
      if (typeof clip?.id === "string" && !script.includes(`"${clip.id}"`)) {
        errors.push(`browser fixture script: missing source key ${clip.id}`);
      }
      if (typeof clip?.path === "string" && !script.includes(`"${clip.path}"`)) {
        errors.push(`browser fixture script: missing source path ${clip.path}`);
      }
    }
  }
  return errors;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const errors = inspectBrowserFixtures();
  if (errors.length) {
    for (const error of errors) console.error(error);
    process.exitCode = 1;
  } else {
    console.log(`Browser video fixtures: ok (${EXPECTED_IDS.length} tagged WebM clips)`);
  }
}

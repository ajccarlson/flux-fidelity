import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import {
  BROWSER_FIXTURE_CONTRACT,
  inspectBrowserFixtures,
} from "../tools/check-browser-fixtures.mjs";

const root = resolve(import.meta.dirname, "..");
const fixtureDirectory = "tests/fixtures/browser";
const fixtureScript = readFileSync(resolve(root, fixtureDirectory, "video-fixture.js"), "utf8");

function temporaryFixture(run) {
  const temporaryRoot = mkdtempSync(join(tmpdir(), "fsrcnnx-browser-fixture-"));
  try {
    cpSync(resolve(root, fixtureDirectory), resolve(temporaryRoot, fixtureDirectory), {
      recursive: true,
    });
    return run(temporaryRoot);
  } finally {
    rmSync(temporaryRoot, { recursive: true, force: true });
  }
}

test("browser video fixtures match their pinned manifest and page contract", () => {
  assert.deepEqual(inspectBrowserFixtures({ rootDir: root }), []);
  assert.equal(BROWSER_FIXTURE_CONTRACT.apiGlobal, "__FSRCNNX_VIDEO_FIXTURE__");
  assert.equal(BROWSER_FIXTURE_CONTRACT.manifestFile, "fixture-manifest.json");
  assert.deepEqual([...BROWSER_FIXTURE_CONTRACT.ids], [
    "bt709-a", "bt709-b", "bt2020-pq", "bt2020-sdr",
  ]);
});

test("browser video fixture verification detects media byte drift", () => {
  temporaryFixture((temporaryRoot) => {
    const path = resolve(temporaryRoot, fixtureDirectory, "media/bt709-a.webm");
    const changed = readFileSync(path);
    changed[changed.length - 1] ^= 0xff;
    writeFileSync(path, changed);
    const errors = inspectBrowserFixtures({ rootDir: temporaryRoot });
    assert.ok(errors.some((error) => /bt709-a: hash is/.test(error)));
  });
});

test("browser video fixture verification keeps an independent canonical hash", () => {
  temporaryFixture((temporaryRoot) => {
    const mediaPath = resolve(temporaryRoot, fixtureDirectory, "media/bt709-a.webm");
    const changed = readFileSync(mediaPath);
    changed[changed.length - 1] ^= 0xff;
    writeFileSync(mediaPath, changed);
    const manifestPath = resolve(temporaryRoot, fixtureDirectory, "fixture-manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    manifest.clips[0].bytes = changed.length;
    manifest.clips[0].sha256 = createHash("sha256").update(changed).digest("hex");
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    const errors = inspectBrowserFixtures({ rootDir: temporaryRoot });
    assert.ok(errors.some((error) => /bt709-a: canonical SHA-256 is/.test(error)));
  });
});

test("browser video fixture verification rejects path traversal and inline script", () => {
  temporaryFixture((temporaryRoot) => {
    const manifestPath = resolve(temporaryRoot, fixtureDirectory, "fixture-manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    manifest.clips[0].path = "../bt709-a.webm";
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    const htmlPath = resolve(temporaryRoot, fixtureDirectory, "video.html");
    writeFileSync(htmlPath, `${readFileSync(htmlPath, "utf8")}\n<script>bad()</script>\n`);
    const errors = inspectBrowserFixtures({ rootDir: temporaryRoot });
    assert.ok(errors.some((error) => /path must be media\/bt709-a\.webm/.test(error)));
    assert.ok(errors.some((error) => /normalized media-relative path/.test(error)));
    assert.ok(errors.some((error) => /inline or remote/.test(error)));
  });
});

test("fixture source replacement listens before activation and ignores ambiguous abort events", () => {
  assert.match(fixtureScript,
    /await waitForLoadedData\(generation, expectedUrl, \(\) => \{\s*video\.src = clip\.path;\s*video\.load\(\);\s*\}\);/);
  const waitStart = fixtureScript.indexOf("function waitForLoadedData");
  const waitEnd = fixtureScript.indexOf("function waitForPresentedFrame", waitStart);
  const waitContract = fixtureScript.slice(waitStart, waitEnd);
  assert.ok(waitContract.indexOf("video.addEventListener") < waitContract.indexOf("activate();"));
  assert.match(waitContract, /video\.currentSrc !== expectedUrl/);
  assert.doesNotMatch(waitContract, /addEventListener\("abort"/,
    "an abort from the retired resource cannot reject the replacement generation");
});

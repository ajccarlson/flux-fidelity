import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { PACKAGE_FILES } from "../tools/package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const manifest = JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf8"));
const privacy = readFileSync(resolve(root, "PRIVACY.md"), "utf8");

test("manifest requests only the capabilities used by the local media pipeline", () => {
  assert.deepEqual(manifest.permissions, ["activeTab", "storage"]);
  assert.equal(Object.hasOwn(manifest, "host_permissions"), false);
  assert.equal(Object.hasOwn(manifest, "optional_permissions"), false);
  assert.equal(Object.hasOwn(manifest, "optional_host_permissions"), false);
  assert.equal(Object.hasOwn(manifest, "externally_connectable"), false);
  assert.deepEqual(manifest.content_scripts, [{
    matches: ["<all_urls>"],
    js: ["src/content.js"],
    run_at: "document_idle",
    all_frames: false,
  }]);
  assert.equal(
    manifest.content_security_policy?.extension_pages,
    "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
  );
});

test("privacy disclosure is packaged and describes local handling, retention, and permissions", () => {
  assert.equal(PACKAGE_FILES.includes("PRIVACY.md"), true);
  for (const required of [
    /processes readable video/i,
    /does not transmit page or media data/i,
    /executes no remotely hosted code/i,
    /chrome\.storage\.local/i,
    /until the extension's local data is cleared or the extension is removed/i,
    /\*\*Site access:\*\*/,
    /\*\*Active tab:\*\*/,
    /\*\*Storage:\*\*/,
  ]) assert.match(privacy, required);
});

test("project-authored packaged runtime has no external endpoint or hidden network API", () => {
  const runtimeFiles = PACKAGE_FILES.filter(
    (file) => file.endsWith(".js") && file !== "tools/transpile.js",
  );
  const fetchFiles = [];
  for (const file of runtimeFiles) {
    const source = readFileSync(resolve(root, file), "utf8");
    assert.doesNotMatch(source, /https?:\/\//i, `${file} contains an external URL`);
    assert.doesNotMatch(
      source,
      /\b(?:XMLHttpRequest|WebSocket|EventSource)\b|\bsendBeacon\s*\(/,
      `${file} adds an undeclared network capability`,
    );
    assert.doesNotMatch(source, /\beval\s*\(|\bnew\s+Function\s*\(/,
      `${file} adds dynamic code execution`);
    if (/\bfetch\s*\(/.test(source)) fetchFiles.push(file);
  }
  assert.deepEqual(fetchFiles, [
    "src/core/fsrcnnx-main.js",
    "src/core/fsrcnnx-neural.js",
    "src/core/fsrcnnx-rife.js",
    "validation/fsrcnnx-reference-validation.js",
    "validation/validate.js",
  ]);
  for (const file of fetchFiles) {
    const source = readFileSync(resolve(root, file), "utf8");
    assert.match(source, /chrome\.runtime\.getURL|validation\/reference-fixtures\.json/,
      `${file} fetches without a bundled-resource root`);
  }
});

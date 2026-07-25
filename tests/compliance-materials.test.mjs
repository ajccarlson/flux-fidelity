import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import test from "node:test";
import {
  PACKAGE_FILES,
  REQUIRED_COMPLIANCE_FILES,
} from "../tools/package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const standardSourceSha = "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965";
const highSourceSha = "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6";
const modificationNotice =
  "Transpiled in 2026 from the mpv/libplacebo GLSL hook format to WGSL compute passes and a JSON pass manifest for FSRCNNX-EXT; model weights and pass order are preserved.";
const fsrcnnxSources = Object.freeze([
  Object.freeze({
    name: "FSRCNNX_x2_16-0-4-1",
    upstream: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl",
    sourcePath: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    sourceSha256: standardSourceSha,
  }),
  Object.freeze({
    name: "FSRCNNX_x2_56-16-4-1",
    upstream: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_56-16-4-1.glsl",
    sourcePath: "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
    sourceSha256: highSourceSha,
  }),
]);

const complianceHashes = Object.freeze({
  "docs/compliance/LGPL_REBUILDING.md": "cb63e0a154fe1b9696f3de4771cde762f006ab06aa391434e8f4dc2fbf767c17",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl": standardSourceSha,
  "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl": highSourceSha,
  "shaders/upstream/SSimDownscaler.glsl": "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
  "shaders/upstream/adaptive-sharpen.glsl": "827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4",
  "tools/transpile.js": "6abd739bc5356ea9fc151c754f6c4d9e017c39283d5e5ba477a70aafe814003a",
  "vendor/ort/LICENSE": "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",
});

function sha256(file) {
  return createHash("sha256").update(readFileSync(resolve(root, file))).digest("hex");
}

test("the package retains every exact rebuilding and license artifact", () => {
  assert.deepEqual(Object.keys(complianceHashes), REQUIRED_COMPLIANCE_FILES);
  for (const [file, expected] of Object.entries(complianceHashes)) {
    assert.equal(PACKAGE_FILES.includes(file), true, `${file} is outside the package boundary`);
    assert.equal(sha256(file), expected, `${file} bytes drifted`);
  }
});

test("Apache-2.0 metadata remains scoped to project-authored material", () => {
  const packageMetadata = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
  const readme = readFileSync(resolve(root, "README.md"), "utf8");
  const notice = readFileSync(resolve(root, "NOTICE"), "utf8");

  assert.equal(packageMetadata.license, "Apache-2.0");
  assert.match(readme, /Project-authored source code and documentation are licensed under/);
  assert.match(readme, /This does not relicense third-party or derived\s+material/);
  assert.match(notice, /third-party and derived material under\s+separate terms/);
});

test("FSRCNN generated files retain exact LGPL source metadata", () => {
  for (const source of fsrcnnxSources) {
    const manifest = JSON.parse(readFileSync(
      resolve(root, `model/${source.name}.passes.json`),
      "utf8",
    ));
    assert.deepEqual({
      license: manifest.license,
      sourcePath: manifest.sourcePath,
      sourceSha256: manifest.sourceSha256,
      modificationNotice: manifest.modificationNotice,
    }, {
      license: "LGPL-3.0-or-later",
      sourcePath: source.sourcePath,
      sourceSha256: source.sourceSha256,
      modificationNotice,
    });

    const wgsl = readFileSync(resolve(root, `model/${source.name}.wgsl`), "utf8");
    assert.match(wgsl, /^\/\/ License: LGPL-3\.0-or-later\n/);
    for (const line of [
      `// Upstream: ${source.upstream}`,
      `// Source path: ${source.sourcePath}`,
      `// Source SHA-256: ${source.sourceSha256}`,
      `// Modification notice: ${modificationNotice}`,
    ]) {
      assert.ok(wgsl.includes(`${line}\n`), `missing generated metadata line: ${line}`);
    }
  }
});

test("the transpiler rejects altered bytes under every pinned FSRCNNX source name", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-lgpl-source-"));
  try {
    for (const { sourcePath: path, sourceSha256: expected } of fsrcnnxSources) {
      const source = resolve(root, path);
      const altered = join(fixture, basename(source));
      writeFileSync(altered, Buffer.concat([readFileSync(source), Buffer.from("\n// altered\n")]));
      const result = spawnSync(process.execPath, [
        resolve(root, "tools/transpile.js"),
        altered,
        "--out", join(fixture, "model"),
      ], { cwd: root, encoding: "utf8" });
      assert.notEqual(result.status, 0, path);
      assert.match(
        result.stderr,
        new RegExp(`SHA-256 [0-9a-f]{64}, expected ${expected.slice(0, 16)}`),
        path,
      );
    }
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

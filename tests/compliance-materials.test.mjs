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
const modificationNotice =
  "Transpiled in 2026 from the mpv/libplacebo GLSL hook format to WGSL compute passes and a JSON pass manifest for FSRCNNX-EXT; model weights and pass order are preserved.";

const complianceHashes = Object.freeze({
  "LGPL_REBUILDING.md": "b23f3a64a3db81248f1fe99dc1138d4b55aace39b162cfaefaa667836240effe",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl": standardSourceSha,
  "shaders/upstream/SSimDownscaler.glsl": "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
  "shaders/upstream/adaptive-sharpen.glsl": "827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4",
  "transpile.js": "2ad45126cd36d52ce1064e8da1e189e10b5d256d8edc28a9dec3737957f4f631",
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

test("standard FSRCNN generated files retain exact LGPL source metadata", () => {
  const manifest = JSON.parse(readFileSync(
    resolve(root, "model/FSRCNNX_x2_16-0-4-1.passes.json"),
    "utf8",
  ));
  assert.deepEqual({
    license: manifest.license,
    sourcePath: manifest.sourcePath,
    sourceSha256: manifest.sourceSha256,
    modificationNotice: manifest.modificationNotice,
  }, {
    license: "LGPL-3.0-or-later",
    sourcePath: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    sourceSha256: standardSourceSha,
    modificationNotice,
  });

  const wgsl = readFileSync(resolve(root, "model/FSRCNNX_x2_16-0-4-1.wgsl"), "utf8");
  assert.match(wgsl, /^\/\/ License: LGPL-3\.0-or-later\n/);
  for (const line of [
    "// Source path: shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    `// Source SHA-256: ${standardSourceSha}`,
    `// Modification notice: ${modificationNotice}`,
  ]) {
    assert.ok(wgsl.includes(`${line}\n`), `missing generated metadata line: ${line}`);
  }
});

test("the standard transpiler rejects altered bytes under the verified source name", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-lgpl-source-"));
  try {
    const source = resolve(root, "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl");
    const altered = join(fixture, basename(source));
    writeFileSync(altered, Buffer.concat([readFileSync(source), Buffer.from("\n// altered\n")]));
    const result = spawnSync(process.execPath, [
      resolve(root, "transpile.js"),
      altered,
      "--out", join(fixture, "model"),
    ], { cwd: root, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /SHA-256 [0-9a-f]{64}, expected d5a24a271e5d9a3f/);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

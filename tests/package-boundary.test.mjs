import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { buildPackage } from "../tools/build-package.mjs";
import { validatePackage } from "../tools/check-package.mjs";
import { EXPECTED_PACKAGE_FILE_COUNT, PACKAGE_FILES } from "../tools/package-files.mjs";

function writeFixture(root, file, contents = file) {
  const path = join(root, file);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, contents);
}

function walk(root, prefix = "") {
  const files = [];
  for (const entry of readdirSync(join(root, prefix), { withFileTypes: true })) {
    const file = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...walk(root, file));
    else files.push(file);
  }
  return files.sort();
}

test("the package boundary is an exact, sorted 54-file allowlist", () => {
  assert.equal(PACKAGE_FILES.length, EXPECTED_PACKAGE_FILE_COUNT);
  assert.equal(EXPECTED_PACKAGE_FILE_COUNT, 54);
  assert.equal(new Set(PACKAGE_FILES).size, PACKAGE_FILES.length);
  assert.deepEqual(PACKAGE_FILES, [...PACKAGE_FILES].sort());
  assert.equal(PACKAGE_FILES.includes("fsrcnnx-development-only.js"), false);
});

test("package creation never discovers an extra runtime-looking local file", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-package-build-"));
  try {
    for (const file of PACKAGE_FILES) writeFixture(fixture, file);
    writeFixture(fixture, "fsrcnnx-development-only.js", "sensitive local experiment");

    const result = buildPackage({ rootDir: fixture, distDir: join(fixture, "output") });

    assert.equal(result.fileCount, 54);
    assert.deepEqual(walk(result.stage), PACKAGE_FILES);
    assert.equal(existsSync(join(result.stage, "fsrcnnx-development-only.js")), false);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("repository-only HTML and JavaScript references are not treated as packaged", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-package-check-"));
  try {
    writeFixture(fixture, "package.json", JSON.stringify({ version: "1.0.0" }));
    writeFixture(fixture, "manifest.json", JSON.stringify({
      manifest_version: 3,
      version: "1.0.0",
      action: { default_popup: "popup.html" },
      background: { service_worker: "background.js" },
    }));
    writeFixture(fixture, "popup.html", '<script src="repository-only-html.js"></script>');
    writeFixture(
      fixture,
      "background.js",
      'import "./repository-only-import.js";\nchrome.runtime.getURL("repository-only-model.onnx");',
    );
    for (const file of [
      "repository-only-html.js",
      "repository-only-import.js",
      "repository-only-model.onnx",
    ]) {
      writeFixture(fixture, file, "repository-only fixture");
      assert.equal(existsSync(join(fixture, file)), true);
    }

    const errors = validatePackage({
      rootDir: fixture,
      packageFiles: ["background.js", "manifest.json", "popup.html"],
    });

    for (const expected of [
      "popup.html: missing repository-only-html.js from package",
      "background.js: missing repository-only-import.js from package",
      "background.js: missing repository-only-model.onnx from package",
    ]) {
      assert.ok(errors.includes(expected), errors.join("\n"));
    }
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

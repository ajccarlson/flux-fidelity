import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import test from "node:test";
import { buildPackage } from "../tools/build-package.mjs";
import { validatePackage } from "../tools/check-package.mjs";
import {
  EXPECTED_PACKAGE_FILE_COUNT,
  PACKAGE_FILES,
  REQUIRED_COMPLIANCE_FILES,
  REQUIRED_RUNTIME_MODEL_FILES,
} from "../tools/package-files.mjs";

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

function manifestGlob(pattern) {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replaceAll("*", "[^/]*");
  return new RegExp(`^${escaped}$`);
}

test("the package boundary is an exact, sorted 77-file allowlist", () => {
  assert.equal(PACKAGE_FILES.length, EXPECTED_PACKAGE_FILE_COUNT);
  assert.equal(EXPECTED_PACKAGE_FILE_COUNT, 77);
  assert.equal(new Set(PACKAGE_FILES).size, PACKAGE_FILES.length);
  assert.deepEqual(PACKAGE_FILES, [...PACKAGE_FILES].sort());
  assert.equal(PACKAGE_FILES.includes("fsrcnnx-development-only.js"), false);
  assert.equal(PACKAGE_FILES.includes("model/rife.onnx"), false);
  assert.equal(PACKAGE_FILES.includes("model/neural/span2x_smoke.fp16.onnx"), false);
});

test("the package retains every legal notice and the machine-readable release gate", () => {
  for (const file of [
    "GPL-3.0.txt",
    "LGPL-3.0.txt",
    "LGPL_REBUILDING.md",
    "LICENSE",
    "MODEL_PROVENANCE.md",
    "PRIVACY.md",
    "THIRD_PARTY_NOTICES.md",
    "release-clearance.json",
    "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    "shaders/upstream/SSimDownscaler.glsl",
    "shaders/upstream/adaptive-sharpen.glsl",
    "tools/transpile.js",
    "vendor/ort/LICENSE",
    "vendor/ort/ThirdPartyNotices.txt",
  ]) {
    assert.equal(PACKAGE_FILES.includes(file), true, `${file} must remain in the package`);
  }
});

test("every required rebuilding and substitution file is packaged exactly", () => {
  assert.deepEqual(REQUIRED_COMPLIANCE_FILES, [...REQUIRED_COMPLIANCE_FILES].sort());
  assert.equal(new Set(REQUIRED_COMPLIANCE_FILES).size, REQUIRED_COMPLIANCE_FILES.length);
  for (const file of REQUIRED_COMPLIANCE_FILES) {
    assert.equal(PACKAGE_FILES.includes(file), true, `${file} must remain in the package`);
  }
});

test("package creation never discovers an extra runtime-looking local file", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-package-build-"));
  try {
    for (const file of PACKAGE_FILES) writeFixture(fixture, file);
    writeFixture(fixture, "package.json", JSON.stringify({ version: "1.2.3" }));
    writeFixture(fixture, "manifest.json", JSON.stringify({ version: "1.2.3" }));
    writeFixture(fixture, "fsrcnnx-development-only.js", "sensitive local experiment");

    const result = buildPackage({ rootDir: fixture, distDir: join(fixture, "output") });

    assert.equal(result.fileCount, 77);
    assert.equal(basename(result.archive), "fsrcnnx-ext-1.2.3.zip");
    assert.equal(
      readFileSync(result.checksums, "utf8"),
      `${result.digest}  fsrcnnx-ext-1.2.3.zip\n`,
    );
    assert.deepEqual(walk(result.stage), PACKAGE_FILES);
    assert.equal(existsSync(join(result.stage, "fsrcnnx-development-only.js")), false);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("package creation rejects invalid or divergent release versions", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-package-version-"));
  try {
    for (const file of PACKAGE_FILES) writeFixture(fixture, file);
    for (const version of ["0.0.0", "01.2.3", "1.65536.0", "1.2.3.4.5"]) {
      writeFixture(fixture, "package.json", JSON.stringify({ version }));
      writeFixture(fixture, "manifest.json", JSON.stringify({ version }));
      assert.throws(
        () => buildPackage({ rootDir: fixture, distDir: join(fixture, "output") }),
        /not a valid Chromium extension version/,
        version,
      );
    }
    writeFixture(fixture, "package.json", JSON.stringify({ version: "1.2.3" }));
    writeFixture(fixture, "manifest.json", JSON.stringify({ version: "1.2.4" }));
    assert.throws(
      () => buildPackage({ rootDir: fixture, distDir: join(fixture, "output") }),
      /Manifest version 1\.2\.4 differs from package 1\.2\.3/,
    );
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("every runtime-selected FSRCNNX, ArtCNN, and RIFE asset is required exactly", () => {
  assert.deepEqual(REQUIRED_RUNTIME_MODEL_FILES, [...REQUIRED_RUNTIME_MODEL_FILES].sort());
  assert.equal(new Set(REQUIRED_RUNTIME_MODEL_FILES).size, REQUIRED_RUNTIME_MODEL_FILES.length);

  for (const omitted of REQUIRED_RUNTIME_MODEL_FILES) {
    const errors = validatePackage({
      packageFiles: PACKAGE_FILES.filter((file) => file !== omitted),
    });
    assert.ok(
      errors.includes(`runtime model assets: missing ${omitted} from package`),
      `${omitted} omission was not rejected:\n${errors.join("\n")}`,
    );
  }
});

test("internal validation files and off-state icons stay packaged but private", () => {
  const manifest = JSON.parse(readFileSync(new URL("../manifest.json", import.meta.url), "utf8"));
  const exposedPatterns = (manifest.web_accessible_resources || [])
    .flatMap((group) => group.resources || []);
  const internalFiles = [
    "icons/icon-off-16.png",
    "icons/icon-off-32.png",
    "icons/icon-off-48.png",
    "icons/icon-off-128.png",
    "validation/fsrcnnx-validation.js",
    "validate.html",
    "validation/validate.js",
  ];

  for (const file of internalFiles) {
    assert.equal(PACKAGE_FILES.includes(file), true, `${file} must remain in the package`);
    assert.equal(
      exposedPatterns.some((pattern) => manifestGlob(pattern).test(file)),
      false,
      `${file} must not be web-accessible`,
    );
  }
  assert.deepEqual(validatePackage(), []);
});

test("every transitive content-script import and runtime URL must be web-accessible", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-content-closure-"));
  try {
    writeFixture(fixture, "package.json", JSON.stringify({ version: "1.0.0" }));
    writeFixture(
      fixture,
      "content.js",
      [
        'import "./static.js";',
        'void import("./dynamic.js");',
        'chrome.runtime.getURL("asset.bin");',
      ].join("\n"),
    );
    writeFixture(fixture, "static.js", 'export { marker } from "./nested/transitive.js";');
    writeFixture(fixture, "dynamic.js", 'export const pending = import("./nested/dynamic-transitive.js");');
    writeFixture(
      fixture,
      "nested/transitive.js",
      'chrome.runtime.getURL("transitive.bin");\nexport const marker = true;',
    );
    writeFixture(
      fixture,
      "nested/dynamic-transitive.js",
      'chrome.runtime.getURL("runtime/" + selected);',
    );
    writeFixture(fixture, "background.js", "void 0;");
    writeFixture(fixture, "popup.html", "<!doctype html><title>Fixture</title>");
    for (const file of ["asset.bin", "transitive.bin", "runtime/a.bin", "runtime/b.bin"]) {
      writeFixture(fixture, file, "fixture asset");
    }
    for (const file of REQUIRED_RUNTIME_MODEL_FILES) writeFixture(fixture, file, "fixture model");

    const exposed = [
      "asset.bin",
      "dynamic.js",
      "nested/dynamic-transitive.js",
      "nested/transitive.js",
      "runtime/*",
      "static.js",
      "transitive.bin",
    ];
    const manifestFor = (resources) => ({
      manifest_version: 3,
      version: "1.0.0",
      action: { default_popup: "popup.html" },
      background: { service_worker: "background.js" },
      content_scripts: [{ matches: ["<all_urls>"], js: ["content.js"] }],
      web_accessible_resources: [{ resources, matches: ["<all_urls>"] }],
    });
    const packageFiles = [
      ...REQUIRED_RUNTIME_MODEL_FILES,
      "asset.bin",
      "background.js",
      "content.js",
      "dynamic.js",
      "manifest.json",
      "nested/dynamic-transitive.js",
      "nested/transitive.js",
      "popup.html",
      "runtime/a.bin",
      "runtime/b.bin",
      "static.js",
      "transitive.bin",
    ].sort();

    writeFixture(fixture, "manifest.json", JSON.stringify(manifestFor(exposed)));
    assert.deepEqual(validatePackage({ rootDir: fixture, packageFiles }), []);

    const omissions = new Map([
      ["static.js", "content.js: content-script dependency static.js"],
      ["dynamic.js", "content.js: content-script dependency dynamic.js"],
      ["nested/transitive.js", "static.js: content-script dependency nested/transitive.js"],
      ["nested/dynamic-transitive.js", "dynamic.js: content-script dependency nested/dynamic-transitive.js"],
      ["asset.bin", "content.js: content-script dependency asset.bin"],
      ["transitive.bin", "nested/transitive.js: content-script dependency transitive.bin"],
      ["runtime/*", "nested/dynamic-transitive.js: content-script dependency runtime/a.bin"],
    ]);
    for (const [omitted, expectedPrefix] of omissions) {
      writeFixture(
        fixture,
        "manifest.json",
        JSON.stringify(manifestFor(exposed.filter((resource) => resource !== omitted))),
      );
      const errors = validatePackage({ rootDir: fixture, packageFiles });
      assert.ok(
        errors.some((error) => error.startsWith(expectedPrefix) &&
          error.endsWith("is not declared in web_accessible_resources")),
        `${omitted} omission was not rejected:\n${errors.join("\n")}`,
      );
    }
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

test("popup DOM validation follows a nested script reference", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-popup-contract-"));
  try {
    writeFixture(fixture, "package.json", JSON.stringify({ version: "1.0.0" }));
    writeFixture(fixture, "manifest.json", JSON.stringify({
      manifest_version: 3,
      version: "1.0.0",
      action: { default_popup: "popup.html" },
      background: { service_worker: "background.js" },
    }));
    writeFixture(fixture, "background.js", "void 0;");
    writeFixture(fixture, "popup.html", '<div id="present"></div><script src="src/popup.js"></script>');
    writeFixture(fixture, "src/popup.js", '$("present"); $("missing");');
    for (const file of REQUIRED_RUNTIME_MODEL_FILES) writeFixture(fixture, file, "fixture model");

    const errors = validatePackage({
      rootDir: fixture,
      packageFiles: [
        ...REQUIRED_RUNTIME_MODEL_FILES,
        "background.js",
        "manifest.json",
        "popup.html",
        "src/popup.js",
      ].sort(),
    });

    assert.ok(errors.includes("popup.html: missing #missing"), errors.join("\n"));
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";
import {
  readTrackedIndex,
  selectTrackedJavaScriptFiles,
} from "../tools/check-syntax.mjs";

test("syntax discovery includes tracked regular sources and ignores local or linked content", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-syntax-boundary-"));
  try {
    mkdirSync(join(fixture, "nested"));
    mkdirSync(join(fixture, "ignored"));
    writeFileSync(join(fixture, "tracked.js"), "export const tracked = true;\n");
    writeFileSync(join(fixture, "nested", "tool.mjs"), "export const tool = true;\n");
    writeFileSync(join(fixture, "ignored", "experiment.js"), "incomplete (\n");
    writeFileSync(join(fixture, "untracked.js"), "incomplete (\n");
    symlinkSync(join(fixture, "ignored", "experiment.js"), join(fixture, "linked.js"));

    const entries = [
      { path: "tracked.js", mode: 0o100644 },
      { path: "nested/tool.mjs", mode: 0o100644 },
      { path: "linked.js", mode: 0o120000 },
      { path: "README.md", mode: 0o100644 },
    ];
    const selected = selectTrackedJavaScriptFiles(fixture, entries)
      .map((file) => relative(fixture, file).replaceAll("\\", "/"));
    assert.deepEqual(selected, ["nested/tool.mjs", "tracked.js"]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("syntax discovery rejects a missing tracked JavaScript source", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-syntax-missing-"));
  try {
    assert.throws(
      () => selectTrackedJavaScriptFiles(fixture, [{ path: "deleted.js", mode: 0o100644 }]),
      /Tracked JavaScript source is missing: deleted\.js/,
    );
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("the repository index selection covers every tracked regular JavaScript source", () => {
  const root = resolve(import.meta.dirname, "..");
  const entries = readTrackedIndex(root);
  const expected = entries
    .filter((entry) => /\.(?:js|mjs)$/.test(entry.path))
    .filter((entry) => (entry.mode & 0o170000) === 0o100000)
    .map((entry) => entry.path)
    .sort();
  const selected = selectTrackedJavaScriptFiles(root, entries)
    .map((file) => relative(root, file).replaceAll("\\", "/"));
  assert.deepEqual(selected, [...new Set(expected)]);
});

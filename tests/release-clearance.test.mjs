import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { inspectReleaseClearance } from "../tools/check-release-clearance.mjs";

const root = resolve(import.meta.dirname, "..");

test("current release-clearance record is structurally valid and explicitly blocked", () => {
  const result = inspectReleaseClearance({ rootDir: root });
  assert.deepEqual(result.errors, []);
  assert.ok(result.blocked.length > 0);
});

test("release-clearance validation detects artifact drift", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-"));
  try {
    writeFileSync(join(fixture, "artifact.bin"), "changed");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "fixture-gate",
        status: "blocked",
        artifacts: [{
          path: "artifact.bin",
          sha256: createHash("sha256").update("original").digest("hex"),
        }],
        resolution: "Replace the fixture.",
      }],
    }));

    const result = inspectReleaseClearance({ rootDir: fixture });
    assert.equal(result.errors.length, 1);
    assert.match(result.errors[0], /artifact\.bin hash is/);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

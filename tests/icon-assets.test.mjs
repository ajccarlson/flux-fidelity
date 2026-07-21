import assert from "node:assert/strict";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { ICON_ASSET_CONTRACT, inspectIconAssets } from "../tools/check-icons.mjs";

const root = resolve(import.meta.dirname, "..");

test("committed PNGs are the sole canonical icon source", () => {
  assert.deepEqual(inspectIconAssets({ rootDir: root }), []);
  assert.equal(Object.isFrozen(ICON_ASSET_CONTRACT), true);
  assert.ok(ICON_ASSET_CONTRACT.every(Object.isFrozen));
});

test("icon verification rejects byte drift and competing legacy sources", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-icons-"));
  try {
    mkdirSync(join(fixture, "icons"));
    for (const spec of ICON_ASSET_CONTRACT) {
      copyFileSync(resolve(root, spec.path), resolve(fixture, spec.path));
    }
    writeFileSync(join(fixture, "icon.svg"), "<svg/>");
    writeFileSync(join(fixture, ICON_ASSET_CONTRACT[0].path), "changed");

    const errors = inspectIconAssets({ rootDir: fixture });
    assert.ok(errors.some((error) => /icon\.svg: competing legacy icon source/.test(error)));
    assert.ok(errors.some((error) => /icons\/icon-16\.png: not a PNG file/.test(error)));
    assert.ok(errors.some((error) => /icons\/icon-16\.png: hash is/.test(error)));
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

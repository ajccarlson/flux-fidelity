import test from "node:test";
import assert from "node:assert/strict";

import { Interpolator } from "../fsrcnnx-interpolate.js";
import { listModels, setModel } from "../fsrcnnx-rife.js";

function makeInterpolator() {
  return new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: () => {},
  });
}

test("interpolator retains engine choices made before runtime import", () => {
  const interpolator = makeInterpolator();

  assert.equal(interpolator.setInterpEngine("rife_orig"), false);
  assert.equal(interpolator._rifeModelKey, "rife_orig");
  assert.equal(interpolator._forceBlend, false);

  assert.equal(interpolator.setInterpEngine("blend"), true);
  assert.equal(interpolator._forceBlend, true);
  assert.equal(interpolator._interpMode, "blend");
});

test("concurrent starts share one lifecycle and failed startup cleans up", async () => {
  const interpolator = makeInterpolator();
  const first = interpolator.start();
  const second = interpolator.start();

  assert.equal(first, second);
  const result = await first;
  assert.equal(result.ok, false);
  assert.equal(result.reason, "unsupported");
  assert.equal(interpolator.running, false);
  assert.equal(interpolator._state, "idle");
  assert.deepEqual(interpolator.stop(), { ok: true, stopped: false });
});

test("blend cadence is bounded and derives from source timestamps", () => {
  const interpolator = makeInterpolator();
  interpolator.setTargetFps(120);

  assert.equal(interpolator._tweensForGap(0, 1_000_000 / 30), 3);
  assert.equal(interpolator._tweensForGap(0, 1_000_000), 7);
  assert.equal(interpolator.setTargetFps(10), 24);
  assert.equal(interpolator.setTargetFps(1000), 480);
  assert.equal(interpolator.setTargetFps("auto"), "auto");
});

test("RIFE model selection rejects unknown keys and updates the public inventory", () => {
  const initial = listModels();
  const original = initial.find((model) => model.current)?.key;
  assert.ok(original);
  assert.equal(setModel("missing-model"), false);

  const alternate = initial.find((model) => model.key !== original)?.key;
  assert.ok(alternate);
  assert.equal(setModel(alternate), true);
  assert.equal(listModels().find((model) => model.current)?.key, alternate);
  assert.equal(setModel(original), true);
});

test("stale inverted resize callbacks always release the restart guard", async () => {
  const interpolator = makeInterpolator();
  interpolator._stopped = false;
  const generation = interpolator._lifecycleGen;

  assert.equal(interpolator._scheduleDimsRestart(generation, 1920, 1080), true);
  assert.equal(interpolator._dimsRestarting, true);
  interpolator._lifecycleGen++;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(interpolator._dimsRestarting, false);

  interpolator._dimsRestarting = true;
  assert.deepEqual(interpolator.stop(), { ok: true, stopped: false });
  assert.equal(interpolator._dimsRestarting, false);
});

test("stale CPU tween completion closes every owned bitmap without enqueueing", () => {
  const interpolator = makeInterpolator();
  interpolator._stopped = false;
  interpolator._lifecycleGen = 4;
  let enqueues = 0;
  interpolator._enqueue = () => { enqueues++; };
  const closed = { tween: 0, current: 0, lookahead: 0 };
  const cur = {
    bmp: { close() { closed.current++; } },
    prevBmp: { close() { closed.lookahead++; } },
  };
  const tween = { close() { closed.tween++; } };
  const stats = { framesOut: 7 };

  assert.equal(interpolator._commitCpuTweenBitmap(3, cur, tween, 123, stats), false);
  assert.deepEqual(closed, { tween: 1, current: 1, lookahead: 1 });
  assert.equal(enqueues, 0);
  assert.equal(stats.framesOut, 7);
});

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

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail(message);
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

test("GPU capture resource limits schedule one stop and a newer lifecycle wins", async () => {
  const warnings = [];
  const interpolator = new Interpolator({
    findVideo: () => null,
    log: () => {},
    warn: (message) => warnings.push(message),
  });
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const generation = interpolator._lifecycleGen;
  interpolator._rifeMod = {
    gpuLastCaptureError: () => Object.assign(new Error("frame pair too large"), { code: "GPU_RESOURCE_LIMIT" }),
  };
  let stops = 0;
  interpolator.stop = () => { stops++; interpolator._lifecycleGen++; interpolator._stopped = true; };

  assert.equal(interpolator._handleGpuCaptureFailure(generation), true);
  assert.equal(interpolator._handleGpuCaptureFailure(generation), true);
  await Promise.resolve();
  assert.equal(stops, 1);
  assert.equal(warnings.length, 1);

  interpolator._stopped = false;
  interpolator._gpuResourceStopQueued = false;
  const staleGeneration = interpolator._lifecycleGen;
  assert.equal(interpolator._handleGpuCaptureFailure(staleGeneration), true);
  interpolator._lifecycleGen++;
  await Promise.resolve();
  assert.equal(stops, 1, "a user-owned newer lifecycle must cancel the queued stop");
});

test("transient GPU budget pressure drops a capture without stopping", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._rifeMod = {
    gpuLastCaptureError: () => Object.assign(new Error("waiting for queue fence"), {
      code: "GPU_RESOURCE_LIMIT",
      details: { transient: true },
    }),
  };
  let stops = 0;
  interpolator.stop = () => { stops++; };

  assert.equal(interpolator._handleGpuCaptureFailure(interpolator._lifecycleGen), false);
  await Promise.resolve();
  assert.equal(stops, 0);
  assert.equal(interpolator._gpuResourceStopQueued, false);
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

test("device-loss restart is single-flight and a newer user stop wins", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  let starts = 0;
  let stops = 0;
  interpolator.start = async () => { starts++; return { ok: true }; };
  interpolator.stop = () => {
    stops++;
    interpolator.running = false;
    interpolator._state = "idle";
    interpolator._stopped = true;
    interpolator._lifecycleGen++;
    return { ok: true };
  };

  const lostDevice = {};
  assert.equal(interpolator._handleRifeDeviceLoss(lostDevice, { message: "reset" }), true);
  assert.equal(interpolator._handleRifeDeviceLoss(lostDevice, { message: "duplicate" }), false);
  // Simulate an explicit off request before the queued recovery microtask.
  interpolator.stop();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(stops, 1);
  assert.equal(starts, 0, "the stale loss callback must not resurrect interpolation");
  assert.equal(interpolator._deviceRestarting, false);
});

test("loss of a replacement device queues one subsequent interpolation recovery", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const firstStartEntered = deferred();
  const finishFirstStart = deferred();
  const productionStop = interpolator.stop.bind(interpolator);
  let starts = 0;
  let stops = 0;
  interpolator.stop = (options) => {
    stops++;
    return productionStop(options);
  };
  interpolator.start = async () => {
    starts++;
    const generation = ++interpolator._lifecycleGen;
    interpolator._stopped = false;
    interpolator.running = true;
    interpolator._state = "starting";
    if (starts === 1) {
      firstStartEntered.resolve();
      await finishFirstStart.promise;
    }
    const current = interpolator._isCurrent(generation);
    if (current) interpolator._state = "running";
    return { ok: current };
  };

  const firstDevice = {};
  const replacementDevice = {};
  assert.equal(interpolator._handleRifeDeviceLoss(firstDevice, { message: "first reset" }), true);
  await firstStartEntered.promise;
  assert.equal(interpolator._handleRifeDeviceLoss(replacementDevice, { message: "replacement reset" }), true);
  assert.equal(interpolator._handleRifeDeviceLoss(replacementDevice, { message: "duplicate" }), false);

  finishFirstStart.resolve();
  await waitFor(() => starts === 2 && !interpolator._deviceRestarting, "queued replacement recovery did not finish");
  assert.equal(starts, 2);
  assert.equal(stops, 2);
  assert.equal(interpolator.running, true);
  assert.equal(interpolator._pendingDeviceLoss, null);
});

test("user stop cancels a replacement-device loss queued during recovery", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  const firstStartEntered = deferred();
  const finishFirstStart = deferred();
  const productionStop = interpolator.stop.bind(interpolator);
  let starts = 0;
  let stops = 0;
  interpolator.stop = (options) => {
    stops++;
    return productionStop(options);
  };
  interpolator.start = async () => {
    starts++;
    const generation = ++interpolator._lifecycleGen;
    interpolator._stopped = false;
    interpolator.running = true;
    interpolator._state = "starting";
    firstStartEntered.resolve();
    await finishFirstStart.promise;
    const current = interpolator._isCurrent(generation);
    if (current) interpolator._state = "running";
    return { ok: current };
  };

  assert.equal(interpolator._handleRifeDeviceLoss({}, { message: "first reset" }), true);
  await firstStartEntered.promise;
  assert.equal(interpolator._handleRifeDeviceLoss({}, { message: "replacement reset" }), true);
  interpolator.stop();
  finishFirstStart.resolve();

  await waitFor(() => !interpolator._deviceRestarting, "cancelled recovery did not unwind");
  assert.equal(starts, 1);
  assert.equal(stops, 2);
  assert.equal(interpolator.running, false);
  assert.equal(interpolator._pendingDeviceLoss, null);
});

test("CPU grab device loss replaces only the current grabber in one recovery flight", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  const generation = interpolator._lifecycleGen;
  const current = { ready: true };
  const replacement = { ready: true };
  interpolator._gpuGrab = current;
  let attempts = 0;
  interpolator._ensureCpuGrabber = async (requestedGeneration) => {
    attempts++;
    assert.equal(requestedGeneration, generation);
    interpolator._gpuGrab = replacement;
    return true;
  };

  assert.equal(interpolator._handleCpuGrabberDeviceLoss({}, generation, { message: "stale" }), false);
  assert.equal(interpolator._gpuGrab, current);
  assert.equal(interpolator._handleCpuGrabberDeviceLoss(current, generation, { message: "reset" }), true);
  assert.equal(interpolator._handleCpuGrabberDeviceLoss(current, generation, { message: "duplicate" }), false);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(attempts, 1);
  assert.equal(interpolator._gpuGrab, replacement);
  assert.equal(interpolator._cpuGrabRecovery, null);
});

test("CPU grab recovery is bounded and a newer lifecycle cancels remaining attempts", async () => {
  const interpolator = makeInterpolator();
  interpolator.running = true;
  interpolator._state = "running";
  interpolator._stopped = false;
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  const generation = interpolator._lifecycleGen;
  let attempts = 0;
  interpolator._ensureCpuGrabber = async () => { attempts++; return false; };

  const first = interpolator._scheduleCpuGrabberRecovery(generation);
  assert.equal(interpolator._scheduleCpuGrabberRecovery(generation), first);
  assert.equal(await first, false);
  assert.equal(attempts, 3);

  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  interpolator._cpuGrabRecoveryDelays = [0, 0, 0];
  interpolator._ensureCpuGrabber = async () => { attempts++; await gate; return false; };
  const cancelled = interpolator._scheduleCpuGrabberRecovery(generation);
  interpolator._lifecycleGen++;
  interpolator._stopped = true;
  release();
  assert.equal(await cancelled, false);
  assert.equal(attempts, 4, "no retry may survive the lifecycle that scheduled it");
});

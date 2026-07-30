import assert from "node:assert/strict";
import test from "node:test";

import {
  GPU_TIMING_FEATURE,
  GpuFrameTimer,
  gpuTimingAvailable,
} from "../src/core/fsrcnnx-gpu-timing.js";

// Minimal stand-ins. The real objects are opaque WebGPU handles, so the fakes
// only need identity and a destroy hook.
function fakeDevice({ features = [GPU_TIMING_FEATURE], failQuerySet = false } = {}) {
  const destroyed = [];
  const device = {
    features: new Set(features),
    destroyed,
    createQuerySet(descriptor) {
      if (failQuerySet) throw new Error("query sets unavailable");
      return { kind: "querySet", descriptor, destroy: () => destroyed.push("querySet") };
    },
    createBuffer(descriptor) {
      return {
        kind: "buffer",
        descriptor,
        destroy: () => destroyed.push(descriptor.label),
        mapAsync: async () => {},
        getMappedRange: () => device.nextMappedRange,
        unmap: () => {},
      };
    },
    nextMappedRange: null,
  };
  return device;
}

function mappedRange(startNs, endNs) {
  const array = new BigUint64Array([BigInt(startNs), BigInt(endNs)]);
  return array.buffer;
}

function fakeEncoder() {
  return {
    resolved: [],
    copied: [],
    resolveQuerySet(...args) { this.resolved.push(args); },
    copyBufferToBuffer(...args) { this.copied.push(args); },
  };
}

globalThis.GPUBufferUsage ??= {
  QUERY_RESOLVE: 0x200, COPY_SRC: 0x004, COPY_DST: 0x008, MAP_READ: 0x001,
};
globalThis.GPUMapMode ??= { READ: 0x001 };

test("a device without the feature yields an inert timer rather than throwing", () => {
  const timer = new GpuFrameTimer(fakeDevice({ features: [] }));
  assert.equal(timer.supported, false);
  assert.equal(gpuTimingAvailable(fakeDevice({ features: [] })), false);
  // Every call site spreads these unconditionally, so they must be empty
  // objects, not undefined — `...undefined` is legal but `{...{}}` is the
  // contract the render path relies on.
  assert.deepEqual(timer.beginningWrites(), {});
  assert.deepEqual(timer.endWrites(), {});
  assert.equal(timer.beginFrame(), false);
  assert.equal(timer.stats().supported, false);
  assert.equal(timer.stats().avgMs, null);
});

test("a null device is inert, which is the pre-initialization state", () => {
  const timer = new GpuFrameTimer(null);
  assert.equal(timer.supported, false);
  assert.deepEqual(timer.beginningWrites(), {});
  timer.resolve(fakeEncoder());
  assert.equal(timer.readPending, false);
});

test("sampling arms only on the interval so the cost stays bounded", () => {
  const timer = new GpuFrameTimer(fakeDevice(), { sampleInterval: 4 });
  assert.equal(timer.supported, true);
  const armed = [];
  for (let frame = 0; frame < 12; frame++) armed.push(timer.beginFrame());
  // Frames 4, 8 and 12 by 1-based count, i.e. indices 3, 7 and 11.
  assert.deepEqual(armed, [
    false, false, false, true,
    false, false, false, true,
    false, false, false, true,
  ]);
});

test("an armed frame carries distinct begin and end indices", () => {
  const timer = new GpuFrameTimer(fakeDevice(), { sampleInterval: 1 });
  timer.beginFrame();
  const begin = timer.beginningWrites();
  const end = timer.endWrites();
  assert.equal(begin.timestampWrites.beginningOfPassWriteIndex, 0);
  assert.equal(end.timestampWrites.endOfPassWriteIndex, 1);
  assert.equal(begin.timestampWrites.querySet, end.timestampWrites.querySet);
  // Only one of the two indices per pass: supplying both on the same pass would
  // time that pass alone rather than the span across the chain.
  assert.equal("endOfPassWriteIndex" in begin.timestampWrites, false);
  assert.equal("beginningOfPassWriteIndex" in end.timestampWrites, false);
});

test("resolve disarms so a bailed frame cannot leak writes into the next one", () => {
  const timer = new GpuFrameTimer(fakeDevice(), { sampleInterval: 1 });
  timer.beginFrame();
  const encoder = fakeEncoder();
  timer.resolve(encoder);
  assert.equal(encoder.resolved.length, 1);
  assert.equal(encoder.copied.length, 1);
  assert.equal(timer.armed, false);
  // A second resolve without a new beginFrame must not double-record.
  timer.resolve(encoder);
  assert.equal(encoder.resolved.length, 1);
});

test("a collected sample converts nanoseconds to milliseconds", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  device.nextMappedRange = mappedRange(1_000_000, 4_500_000);
  const ms = await timer.collect();
  assert.equal(ms, 3.5);
  assert.deepEqual(timer.series(), [3.5]);
  assert.equal(timer.stats().avgMs, 3.5);
  assert.equal(timer.stats().lastMs, 3.5);
  assert.equal(timer.stats().samples, 1);
});

test("a disjoint or reset counter is discarded instead of dominating the max", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });

  // Backwards delta: the subtraction underflows in BigInt terms, so a naive
  // Number() conversion would produce an enormous positive span.
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  device.nextMappedRange = mappedRange(9_000_000, 1_000_000);
  assert.equal(await timer.collect(), null);

  // Implausibly long span.
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  device.nextMappedRange = mappedRange(0, 60_000_000_000);
  assert.equal(await timer.collect(), null);

  assert.deepEqual(timer.series(), []);
  assert.equal(timer.stats().maxMs, null);
});

test("only one readback is in flight at a time", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  assert.equal(timer.readPending, true);
  // While pending, the next frame must not arm — otherwise a slow map would
  // queue an unbounded number of resolves.
  assert.equal(timer.beginFrame(), false);
  device.nextMappedRange = mappedRange(0, 2_000_000);
  await timer.collect();
  assert.equal(timer.readPending, false);
  assert.equal(timer.beginFrame(), true);
});

test("collect without a pending readback is a no-op", async () => {
  const timer = new GpuFrameTimer(fakeDevice(), { sampleInterval: 1 });
  assert.equal(await timer.collect(), null);
});

test("a map failure records the reason and stops sampling rather than looping", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  timer.readBuffer.mapAsync = async () => { throw new Error("device lost"); };
  assert.equal(await timer.collect(), null);
  assert.equal(timer.readPending, false);
  assert.match(timer.lastError, /device lost/);
});

test("an adapter advertising the feature but refusing a query set degrades cleanly", () => {
  const timer = new GpuFrameTimer(fakeDevice({ failQuerySet: true }));
  assert.equal(timer.supported, false);
  assert.match(timer.lastError, /query sets unavailable/);
  assert.equal(timer.querySet, null);
  assert.deepEqual(timer.beginningWrites(), {});
});

test("history is capped so the published series cannot grow without bound", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1, history: 3 });
  for (let index = 1; index <= 5; index++) {
    timer.beginFrame();
    timer.resolve(fakeEncoder());
    device.nextMappedRange = mappedRange(0, index * 1_000_000);
    await timer.collect();
  }
  assert.deepEqual(timer.series(), [3, 4, 5]);
  assert.equal(timer.stats().samples, 3);
});

test("destroy releases every resource and silences later calls", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  timer.destroy();
  assert.deepEqual(
    device.destroyed.sort(),
    ["fsrcnnx-timestamp-read", "fsrcnnx-timestamp-resolve", "querySet"],
  );
  assert.equal(timer.beginFrame(), false);
  assert.deepEqual(timer.beginningWrites(), {});
  assert.equal(await timer.collect(), null);
  timer.destroy();
});

test("reset clears samples without disabling the timer", async () => {
  const device = fakeDevice();
  const timer = new GpuFrameTimer(device, { sampleInterval: 1 });
  timer.beginFrame();
  timer.resolve(fakeEncoder());
  device.nextMappedRange = mappedRange(0, 5_000_000);
  await timer.collect();
  assert.equal(timer.stats().samples, 1);
  timer.reset();
  assert.equal(timer.stats().samples, 0);
  assert.equal(timer.supported, true);
});

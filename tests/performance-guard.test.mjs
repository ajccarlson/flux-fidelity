import assert from "node:assert/strict";
import test from "node:test";
import { PlaybackPerformanceGuard } from "../fsrcnnx-performance.js";

function feedWindow(guard, { start, frames = 30, skippedEvery = 0, qualityDrops = 0 }) {
  let presentedFrames = guard.lastPresentedFrames ?? 0;
  let totalVideoFrames = guard.lastQualityFrames ?? 0;
  let droppedVideoFrames = guard.lastQualityDrops ?? 0;
  for (let index = 1; index <= frames; index++) {
    const skipped = skippedEvery > 0 && index % skippedEvery === 0 ? 1 : 0;
    presentedFrames += 1 + skipped;
    totalVideoFrames += 1 + skipped;
    droppedVideoFrames += skipped || (index <= qualityDrops ? 1 : 0);
    guard.observeFrame({
      now: start + index * 70,
      metadata: { presentedFrames, mediaTime: (start + index * 70) / 1000 },
      quality: { totalVideoFrames, droppedVideoFrames },
    });
  }
}

test("isolated or recovered frame loss does not lower the renderer", () => {
  const guard = new PlaybackPerformanceGuard({ windowMs: 1000, minFrames: 10 });
  feedWindow(guard, { start: 0, skippedEvery: 5 });
  assert.equal(guard.snapshot().consecutiveDegradedWindows, 1);
  feedWindow(guard, { start: 2200 });
  assert.equal(guard.snapshot().consecutiveDegradedWindows, 0);
  feedWindow(guard, { start: 4400, skippedEvery: 5 });
  assert.equal(guard.snapshot().triggered, null);
});

test("three sustained dropped-frame windows trigger one sticky fallback signal", () => {
  const guard = new PlaybackPerformanceGuard({ windowMs: 1000, minFrames: 10 });
  feedWindow(guard, { start: 0, skippedEvery: 5 });
  feedWindow(guard, { start: 2200, skippedEvery: 5 });
  feedWindow(guard, { start: 4400, skippedEvery: 5 });
  const state = guard.snapshot();
  assert.equal(state.triggered.code, "sustained-frame-drops");
  assert.equal(state.triggered.evidence.consecutiveDegradedWindows, 3);
  assert.match(state.triggered.detail, /persisted across 3 observation windows/);
});

test("GPU backlog requires consecutive slow samples and uses frame cadence", () => {
  const guard = new PlaybackPerformanceGuard({
    backlogThresholdMs: 100,
    backlogFrameIntervals: 4,
    consecutiveBacklogs: 3,
  });
  guard.frameIntervalMs = 40;
  assert.equal(guard.observeQueueBacklog(170, 1), null);
  assert.equal(guard.observeQueueBacklog(50, 2), null);
  assert.equal(guard.observeQueueBacklog(170, 3), null);
  assert.equal(guard.observeQueueBacklog(180, 4), null);
  assert.equal(guard.observeQueueBacklog(190, 5).code, "sustained-gpu-backlog");
  assert.equal(guard.snapshot().triggered.evidence.thresholdMs, 160);
  guard.reset();
  assert.equal(guard.snapshot().triggered, null);
});

import assert from "node:assert/strict";
import test from "node:test";

import { FrameSignature } from "../src/core/fsrcnnx-frame-signature.js";

// A fake video whose "content" is a single byte value, plus a canvas that turns
// that value into a uniform frame. Enough to exercise every decision path
// without a real decoder.
function harness({ readback = "ok", drawThrows = false, contextThrows = false } = {}) {
  const state = { value: 0, drawCalls: 0, readCalls: 0 };
  const video = { videoWidth: 1920, videoHeight: 1080, readyState: 2 };
  const createCanvas = (width, height) => ({
    width,
    height,
    getContext() {
      if (contextThrows) throw new Error("2d unavailable");
      return {
        drawImage() {
          state.drawCalls++;
          if (drawThrows) throw new Error("frame not decodable");
        },
        getImageData(x, y, w, h) {
          state.readCalls++;
          if (readback === "tainted") {
            const error = new Error("The canvas has been tainted by cross-origin data.");
            error.name = "SecurityError";
            throw error;
          }
          // Uniform frame carrying the current content value.
          return { data: new Uint8ClampedArray(w * h * 4).fill(state.value) };
        },
      };
    },
  });
  return { state, video, createCanvas };
}

test("an unchanged frame is skipped and a changed one is not", () => {
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });

  // First frame has nothing to compare against.
  state.value = 10;
  assert.equal(signature.shouldSkip(video), false);
  // Identical content: reuse the previous result.
  assert.equal(signature.shouldSkip(video), true);
  assert.equal(signature.shouldSkip(video), true);
  // New content must render.
  state.value = 11;
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.stats().skipped, 2);
  assert.equal(signature.stats().duplicates, 2);
});

test("a run of skips is capped so a missed change cannot persist", () => {
  // The signature is a downscale, so it can only prove frames are
  // indistinguishable at that resolution. The cap bounds how long a change too
  // small to register could stay on screen.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8, maxRun: 3 });
  state.value = 5;
  signature.shouldSkip(video);
  const decisions = [];
  for (let frame = 0; frame < 8; frame++) decisions.push(signature.shouldSkip(video));
  // Three skips, then a forced render, then the pattern repeats.
  assert.deepEqual(decisions, [true, true, true, false, true, true, true, false]);
});

test("reset forgets the previous frame without losing statistics", () => {
  // Anything that invalidates the cached output — new video, resize, model change
  // — must reset, or the next frame is judged against a result no longer shown.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  state.value = 7;
  signature.shouldSkip(video);
  assert.equal(signature.shouldSkip(video), true);
  signature.reset();
  // Identical content, but there is no trusted previous result to reuse.
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.stats().skipped, 1);
  assert.equal(signature.shouldSkip(video), true);
});

test("a tainted canvas disables detection permanently rather than throwing", () => {
  // Cross-origin video without CORS makes getImageData throw SecurityError. That
  // is permanent for the source, so it must stop being paid for.
  const { video, createCanvas } = harness({ readback: "tainted" });
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.stats().supported, false);
  assert.match(signature.stats().disabledReason, /readback-blocked/);
  // And it never probes again.
  const before = signature.probes;
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.probes, before);
});

test("an undecodable frame is transient and does not disable detection", () => {
  const { video, createCanvas } = harness({ drawThrows: true });
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.stats().supported, true, "a transient draw failure is not permanent");
  assert.equal(signature.stats().disabledReason, null);
});

test("a missing 2d context disables detection cleanly", () => {
  const { video, createCanvas } = harness({ contextThrows: true });
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(signature.stats().supported, false);
  assert.match(signature.stats().disabledReason, /canvas-unavailable/);
});

test("a video that is not ready is never probed", () => {
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  video.readyState = 0;
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(state.drawCalls, 0);
  video.readyState = 2;
  video.videoWidth = 0;
  assert.equal(signature.shouldSkip(video), false);
  assert.equal(state.drawCalls, 0);
});

test("detection backs off when the content never repeats", () => {
  // Live action buys nothing from this, and the downscale plus readback is not
  // free, so it must stop paying for itself.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({
    createCanvas, width: 8, height: 8, reviewInterval: 10, minDuplicateRate: 0.2,
  });
  for (let frame = 0; frame < 11; frame++) {
    state.value = frame + 1;                 // every frame differs
    signature.shouldSkip(video);
  }
  assert.equal(signature.probing, false, "should have backed off");
  const probesAtBackoff = signature.probes;

  // While backed off it probes only occasionally, so most frames cost nothing.
  for (let frame = 0; frame < 20; frame++) {
    state.value = 100 + frame;
    signature.shouldSkip(video);
  }
  assert.ok(signature.probes - probesAtBackoff < 5,
    `backed-off probing must be rare; did ${signature.probes - probesAtBackoff}`);
});

test("backed-off detection resumes when the content becomes duplicate-heavy", () => {
  // A live-action opening followed by animation must not stay backed off forever.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({
    createCanvas, width: 8, height: 8, reviewInterval: 10, minDuplicateRate: 0.2,
    idleProbeInterval: 5,
  });
  for (let frame = 0; frame < 11; frame++) {
    state.value = frame + 1;
    signature.shouldSkip(video);
  }
  assert.equal(signature.probing, false);

  // Now hold the content still. The occasional idle probe sees a duplicate.
  state.value = 42;
  for (let frame = 0; frame < 20; frame++) signature.shouldSkip(video);
  assert.equal(signature.probing, true, "should have resumed on repeated content");
});

test("statistics describe what was actually avoided", () => {
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  state.value = 3;
  signature.shouldSkip(video);
  signature.shouldSkip(video);
  state.value = 4;
  signature.shouldSkip(video);
  const stats = signature.stats();
  assert.equal(stats.probes, 3);
  assert.equal(stats.duplicates, 1);
  assert.equal(stats.skipped, 1);
  assert.equal(stats.duplicateRate, 0.333);
  assert.equal(stats.supported, true);
});

// These pin the integration rules that the render path depends on. Each one
// corresponds to a hazard found while wiring this in, not a hypothetical.
test("a skipped frame must never be reported as a cheap render", () => {
  // encodeMs is a 120-sample ring of per-frame CPU time. Skipped frames cost
  // almost nothing, so including them would drag the reported figure toward zero
  // and hide the cost of the frames that actually rendered. The renderer excludes
  // them and counts them separately; this pins the shape that reporting needs.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  state.value = 1;
  signature.shouldSkip(video);
  signature.shouldSkip(video);
  const stats = signature.stats();
  assert.equal(stats.skipped, 1);
  assert.notEqual(stats.probes, stats.skipped, "probes and skips are distinct counts");
});

test("statistics survive a reset so the popup total does not jump backwards", () => {
  // reset() happens on every layout change and every scroll back into view. If it
  // cleared the counters, the reported total would fall and read as a bug.
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  state.value = 2;
  signature.shouldSkip(video);
  signature.shouldSkip(video);
  const before = signature.stats();
  signature.reset();
  const after = signature.stats();
  assert.equal(after.skipped, before.skipped);
  assert.equal(after.probes, before.probes);
  assert.equal(after.duplicates, before.duplicates);
});

test("disabling clears the retained frame so nothing stale can be matched later", () => {
  const { state, video, createCanvas } = harness();
  const signature = new FrameSignature({ createCanvas, width: 8, height: 8 });
  state.value = 9;
  signature.shouldSkip(video);
  signature.disable("test");
  assert.equal(signature.previous, null);
  assert.equal(signature.run, 0);
  assert.equal(signature.shouldSkip(video), false);
});

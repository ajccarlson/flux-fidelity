// fsrcnnx-frame-signature.js
// Detects decoded frames whose content is unchanged, so the upscaler can reuse
// the previous result instead of re-running the network.
//
// Why this is worth doing: FSRCNNX standard costs 10,448 multiply-accumulates
// per source pixel and High costs 14,680, and the renderer previously ran the
// full chain on every decoded frame. Animation is routinely drawn "on twos" or
// "on threes", so at 24 fps only 8-12 frames per second carry new pixels — and
// animation is exactly what these models target. Live action benefits far less,
// which is why detection backs itself off when it stops paying (see below).
//
// Why the signature is taken on the CPU rather than the GPU: the decision has to
// apply to the frame being rendered. A GPU reduce would need its result read
// back, and an asynchronous readback only answers a frame or two later, which
// turns the skip into a prediction — and a wrong prediction shows a stale frame.
// Drawing the video into a small 2D canvas and reading that back is a sync on a
// canvas context we own, costs a fraction of a millisecond, and answers now.
//
// The signature is a downscale, so it cannot prove two frames are identical, only
// that they are indistinguishable at that resolution. Two guards bound the
// consequence: comparison is exact (any differing byte is a change), and a run of
// skips is capped so a missed change cannot persist.

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";

const DEFAULT_WIDTH = 96;
const DEFAULT_HEIGHT = 54;
// A missed change can survive at most this many frames — ~130 ms at 60 fps. Real
// duplicate runs are 1-2 frames on twos and 2 on threes, so this costs nothing in
// the cases the feature exists for.
const DEFAULT_MAX_RUN = 8;
// Evaluate whether detection is earning its keep over this many probes.
const DEFAULT_REVIEW_INTERVAL = 240;
const DEFAULT_MIN_DUPLICATE_RATE = 0.02;
// While backed off, probe this rarely — just often enough to notice the content
// becoming animation partway through.
const DEFAULT_IDLE_PROBE_INTERVAL = 60;

export class FrameSignature {
  constructor(options = {}) {
    this.width = positiveInt(options.width, DEFAULT_WIDTH);
    this.height = positiveInt(options.height, DEFAULT_HEIGHT);
    this.maxRun = positiveInt(options.maxRun, DEFAULT_MAX_RUN);
    this.reviewInterval = positiveInt(options.reviewInterval, DEFAULT_REVIEW_INTERVAL);
    this.minDuplicateRate = Number.isFinite(options.minDuplicateRate)
      ? options.minDuplicateRate : DEFAULT_MIN_DUPLICATE_RATE;
    this.idleProbeInterval = positiveInt(options.idleProbeInterval, DEFAULT_IDLE_PROBE_INTERVAL);
    this.createCanvas = options.createCanvas || defaultCanvasFactory;

    this.canvas = null;
    this.context = null;
    this.previous = null;
    this.disabledReason = null;
    this.probing = true;
    this.frameIndex = 0;
    this.run = 0;
    this.probes = 0;
    this.duplicates = 0;
    this.skipped = 0;
    this.windowProbes = 0;
    this.windowDuplicates = 0;
  }

  // Forget the previous frame without losing the earned statistics. Any change
  // that invalidates the cached output — a new video, a resize, a different model
  // — must call this, or the next frame could be judged against a signature whose
  // result is no longer on screen.
  reset() {
    this.previous = null;
    this.run = 0;
  }

  _ensureContext() {
    if (this.context) return this.context;
    if (this.disabledReason) return null;
    try {
      this.canvas = this.createCanvas(this.width, this.height);
      // willReadFrequently keeps the canvas on a readback-friendly backing; without
      // it some engines round-trip the surface on every getImageData. The color
      // space is declared for the same reason every other boundary here declares
      // it: an implicit one can differ between engines, and comparisons must be
      // taken in a space that is stable frame to frame.
      this.context = this.canvas.getContext("2d", {
        colorSpace: SRGB_COLOR_SPACE,
        willReadFrequently: true,
      });
      if (!this.context) throw new Error("2d context unavailable");
    } catch (error) {
      this.disable(`canvas-unavailable: ${short(error)}`);
      return null;
    }
    return this.context;
  }

  disable(reason) {
    this.disabledReason = reason;
    this.context = null;
    this.canvas = null;
    this.previous = null;
    this.run = 0;
  }

  // Downscale the frame and return its bytes, or null when unavailable.
  _sample(video) {
    const context = this._ensureContext();
    if (!context) return null;
    try {
      context.drawImage(video, 0, 0, this.width, this.height);
    } catch (error) {
      // A frame that is not yet decodable is transient; do not disable for it.
      return null;
    }
    try {
      return context.getImageData(0, 0, this.width, this.height).data;
    } catch (error) {
      // Cross-origin video without CORS taints the canvas and getImageData throws
      // SecurityError. That is permanent for this source, so stop paying for it.
      this.disable(`readback-blocked: ${short(error)}`);
      return null;
    }
  }

  // True when this frame may reuse the previous result. Callers must treat a
  // false return as "render normally"; there is no third state.
  shouldSkip(video) {
    if (this.disabledReason || !video) return false;
    if (!(video.videoWidth > 0) || !(video.readyState >= 2)) return false;
    this.frameIndex++;

    // Backed off: probe rarely, and only to notice the content changing character.
    if (!this.probing && this.frameIndex % this.idleProbeInterval !== 0) return false;

    const sample = this._sample(video);
    if (!sample) return false;
    this.probes++;
    this.windowProbes++;

    const previous = this.previous;
    // Copy: getImageData may hand back a view onto reused storage.
    this.previous = sample.slice();

    let duplicate = false;
    if (previous && previous.length === sample.length) {
      duplicate = equalBytes(previous, sample);
    }
    if (duplicate) {
      this.duplicates++;
      this.windowDuplicates++;
      if (!this.probing) {
        // The content became duplicate-heavy after we backed off. Resume.
        this.probing = true;
        this.windowProbes = 0;
        this.windowDuplicates = 0;
      }
    }
    this._review();

    if (!duplicate) {
      this.run = 0;
      return false;
    }
    // Cap the run so a change too small for the signature cannot persist.
    if (this.run >= this.maxRun) {
      this.run = 0;
      return false;
    }
    this.run++;
    this.skipped++;
    return true;
  }

  // Detection costs a downscale and a small readback per frame. On live action
  // that buys nothing, so stop paying for it and re-probe occasionally instead.
  _review() {
    if (!this.probing || this.windowProbes < this.reviewInterval) return;
    const rate = this.windowDuplicates / this.windowProbes;
    if (rate < this.minDuplicateRate) {
      this.probing = false;
      this.previous = null;
      this.run = 0;
    }
    this.windowProbes = 0;
    this.windowDuplicates = 0;
  }

  stats() {
    return {
      supported: !this.disabledReason,
      disabledReason: this.disabledReason,
      probing: this.probing,
      probes: this.probes,
      duplicates: this.duplicates,
      skipped: this.skipped,
      duplicateRate: this.probes ? Math.round((this.duplicates / this.probes) * 1000) / 1000 : 0,
    };
  }
}

function equalBytes(left, right) {
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function defaultCanvasFactory(width, height) {
  if (typeof OffscreenCanvas === "function") return new OffscreenCanvas(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function positiveInt(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function short(error) {
  return String(error?.message || error).slice(0, 120);
}

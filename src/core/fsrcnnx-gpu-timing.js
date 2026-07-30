// fsrcnnx-gpu-timing.js
// Optional GPU-side frame timing built on the WebGPU timestamp-query feature.
//
// The renderer already reports `encodeMs`, but that is wall time spent in the
// JavaScript frame callback and it ends at queue.submit(). It says nothing about
// how long the GPU takes to execute the chain, so it cannot tell you whether the
// model, the downscaler, or presentation is the bottleneck. This module supplies
// the missing half.
//
// timestamp-query is optional and is frequently absent: the feature is gated
// behind a Chromium flag on several platforms and is unavailable under
// SwiftShader. Every entry point therefore degrades to a no-op rather than
// throwing, and callers can pass the writes through unconditionally.

// Two timestamps per sample: the beginning of the first pass and the end of the
// last one. Anything finer would need a query set per pass and a readback large
// enough to stall the very frame it is measuring.
const QUERY_COUNT = 2;
const BYTES_PER_TIMESTAMP = 8;
const RESOLVE_BYTES = QUERY_COUNT * BYTES_PER_TIMESTAMP;
// One sample every N frames. The readback is asynchronous and never blocks the
// frame that produced it, but each sample still costs a resolve plus a copy, and
// a 60 Hz stream converges on a stable average long before every frame is read.
const DEFAULT_SAMPLE_INTERVAL = 15;
const DEFAULT_HISTORY = 120;
// Discard implausible deltas. A disjoint or reset timestamp counter can report a
// negative or absurd span, and one bad value would otherwise dominate the max.
const MAX_PLAUSIBLE_MS = 10_000;

export const GPU_TIMING_FEATURE = "timestamp-query";

export function gpuTimingAvailable(adapterOrDevice) {
  return !!adapterOrDevice?.features?.has?.(GPU_TIMING_FEATURE);
}

export class GpuFrameTimer {
  constructor(device, options = {}) {
    this.device = device || null;
    this.sampleInterval = positiveInt(options.sampleInterval, DEFAULT_SAMPLE_INTERVAL);
    this.historyLimit = positiveInt(options.history, DEFAULT_HISTORY);
    this.supported = gpuTimingAvailable(device);
    this.samples = [];
    this.frameIndex = 0;
    this.destroyed = false;
    this.querySet = null;
    this.resolveBuffer = null;
    this.readBuffer = null;
    // A single in-flight readback. Sampling is skipped while one is pending, so
    // a slow map can never queue an unbounded number of buffers.
    this.readPending = false;
    this.armed = false;
    this.lastError = null;
    if (this.supported) this._allocate();
  }

  _allocate() {
    try {
      this.querySet = this.device.createQuerySet({
        label: "fsrcnnx-frame-timestamps",
        type: "timestamp",
        count: QUERY_COUNT,
      });
      this.resolveBuffer = this.device.createBuffer({
        label: "fsrcnnx-timestamp-resolve",
        size: RESOLVE_BYTES,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.readBuffer = this.device.createBuffer({
        label: "fsrcnnx-timestamp-read",
        size: RESOLVE_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    } catch (error) {
      // An adapter can advertise the feature and still refuse the query set.
      // Timing is a diagnostic, so record why and carry on without it.
      this.lastError = String(error?.message || error);
      this.supported = false;
      this._releaseResources();
    }
  }

  // Call once per frame before encoding. Returns true when this frame should
  // carry timestamp writes.
  beginFrame() {
    if (!this.supported || this.destroyed) return false;
    this.frameIndex++;
    this.armed = !this.readPending && this.frameIndex % this.sampleInterval === 0;
    return this.armed;
  }

  // Descriptor fragment for the first pass of the chain. Spreading the result is
  // safe when unarmed because it is then empty.
  beginningWrites() {
    if (!this.armed) return {};
    return { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0 } };
  }

  // Descriptor fragment for the final pass of the chain.
  endWrites() {
    if (!this.armed) return {};
    return { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: 1 } };
  }

  // Call after the last pass is encoded and before finishing the encoder.
  resolve(encoder) {
    if (!this.armed || this.destroyed) return;
    try {
      encoder.resolveQuerySet(this.querySet, 0, QUERY_COUNT, this.resolveBuffer, 0);
      encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, RESOLVE_BYTES);
      this.readPending = true;
    } catch (error) {
      this.lastError = String(error?.message || error);
      this.readPending = false;
    } finally {
      this.armed = false;
    }
  }

  // Call after queue.submit(). Never awaited by the frame loop: the sample lands
  // whenever the map resolves, which is deliberately a later frame.
  async collect() {
    if (!this.readPending || this.destroyed) return null;
    try {
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      if (this.destroyed) return null;
      const view = new BigUint64Array(this.readBuffer.getMappedRange().slice(0));
      this.readBuffer.unmap();
      const ms = Number(view[1] - view[0]) / 1e6;
      if (Number.isFinite(ms) && ms >= 0 && ms <= MAX_PLAUSIBLE_MS) {
        this.samples.push(ms);
        if (this.samples.length > this.historyLimit) this.samples.shift();
        return ms;
      }
      return null;
    } catch (error) {
      // Device loss races the map. Stop sampling rather than retry-looping.
      if (!this.destroyed) this.lastError = String(error?.message || error);
      return null;
    } finally {
      this.readPending = false;
    }
  }

  stats() {
    if (!this.samples.length) {
      return { supported: this.supported, samples: 0, avgMs: null, maxMs: null, lastMs: null };
    }
    let total = 0;
    let max = 0;
    for (const value of this.samples) {
      total += value;
      if (value > max) max = value;
    }
    return {
      supported: this.supported,
      samples: this.samples.length,
      avgMs: round1(total / this.samples.length),
      maxMs: round1(max),
      lastMs: round1(this.samples[this.samples.length - 1]),
    };
  }

  series() {
    return this.samples.map(round1);
  }

  reset() {
    this.samples = [];
    this.frameIndex = 0;
  }

  _releaseResources() {
    for (const resource of [this.querySet, this.resolveBuffer, this.readBuffer]) {
      try { resource?.destroy?.(); } catch {}
    }
    this.querySet = null;
    this.resolveBuffer = null;
    this.readBuffer = null;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.armed = false;
    this.readPending = false;
    this._releaseResources();
    this.samples = [];
  }
}

function positiveInt(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

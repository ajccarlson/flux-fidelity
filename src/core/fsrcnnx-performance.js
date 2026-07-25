// Sustained playback-pressure detection for adaptive renderer fallback.
// This module is deliberately independent of DOM and WebGPU globals so its
// hysteresis can be exercised deterministically in CI.

function finiteNonNegative(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export class PlaybackPerformanceGuard {
  constructor({
    windowMs = 2000,
    minFrames = 24,
    dropRatio = 0.08,
    consecutiveWindows = 3,
    queueSampleEvery = 30,
    backlogThresholdMs = 120,
    backlogFrameIntervals = 4,
    consecutiveBacklogs = 3,
  } = {}) {
    this.options = {
      windowMs,
      minFrames,
      dropRatio,
      consecutiveWindows,
      queueSampleEvery,
      backlogThresholdMs,
      backlogFrameIntervals,
      consecutiveBacklogs,
    };
    this.reset();
  }

  reset() {
    this.windowStartedAt = null;
    this.callbackFrames = 0;
    this.callbackSkips = 0;
    this.qualityFrames = 0;
    this.qualityDrops = 0;
    this.lastPresentedFrames = null;
    this.lastQualityFrames = null;
    this.lastQualityDrops = null;
    this.lastMediaTime = null;
    this.frameIntervalMs = null;
    this.degradedWindows = 0;
    this.backlogSamples = 0;
    this.submittedFrames = 0;
    this.triggered = null;
    this.lastWindow = null;
  }

  observeFrame({ now, metadata = null, quality = null } = {}) {
    const timestamp = finiteNonNegative(now);
    if (timestamp == null) return null;
    if (this.windowStartedAt == null) this.windowStartedAt = timestamp;

    const presented = finiteNonNegative(metadata?.presentedFrames);
    if (presented != null) {
      if (this.lastPresentedFrames != null && presented >= this.lastPresentedFrames) {
        const delta = presented - this.lastPresentedFrames;
        if (delta > 0) {
          this.callbackFrames += delta;
          this.callbackSkips += Math.max(0, delta - 1);
        }
      }
      this.lastPresentedFrames = presented;
    }

    const mediaTime = finiteNonNegative(metadata?.mediaTime);
    if (mediaTime != null) {
      if (this.lastMediaTime != null && mediaTime > this.lastMediaTime) {
        const interval = (mediaTime - this.lastMediaTime) * 1000;
        if (interval >= 4 && interval <= 250) {
          this.frameIntervalMs = this.frameIntervalMs == null
            ? interval
            : this.frameIntervalMs * 0.8 + interval * 0.2;
        }
      }
      this.lastMediaTime = mediaTime;
    }

    const totalFrames = finiteNonNegative(quality?.totalVideoFrames);
    const droppedFrames = finiteNonNegative(quality?.droppedVideoFrames);
    if (totalFrames != null && droppedFrames != null) {
      if (this.lastQualityFrames != null && this.lastQualityDrops != null &&
          totalFrames >= this.lastQualityFrames && droppedFrames >= this.lastQualityDrops) {
        this.qualityFrames += totalFrames - this.lastQualityFrames;
        this.qualityDrops += droppedFrames - this.lastQualityDrops;
      }
      this.lastQualityFrames = totalFrames;
      this.lastQualityDrops = droppedFrames;
    }

    if (timestamp - this.windowStartedAt < this.options.windowMs) return null;
    const callbackRatio = this.callbackFrames > 0 ? this.callbackSkips / this.callbackFrames : 0;
    const qualityRatio = this.qualityFrames > 0 ? this.qualityDrops / this.qualityFrames : 0;
    const observedFrames = Math.max(this.callbackFrames, this.qualityFrames);
    const ratio = Math.max(callbackRatio, qualityRatio);
    const sufficientlyObserved = observedFrames >= this.options.minFrames;
    const degraded = sufficientlyObserved && ratio >= this.options.dropRatio;
    this.degradedWindows = degraded ? this.degradedWindows + 1 : 0;
    this.lastWindow = {
      at: timestamp,
      durationMs: timestamp - this.windowStartedAt,
      observedFrames,
      callbackSkips: this.callbackSkips,
      qualityDrops: this.qualityDrops,
      dropRatio: ratio,
      degraded,
      consecutiveDegradedWindows: this.degradedWindows,
    };
    this.windowStartedAt = timestamp;
    this.callbackFrames = 0;
    this.callbackSkips = 0;
    this.qualityFrames = 0;
    this.qualityDrops = 0;

    if (!this.triggered && this.degradedWindows >= this.options.consecutiveWindows) {
      this.triggered = {
        code: "sustained-frame-drops",
        detail: `Dropped-frame ratio ${(ratio * 100).toFixed(1)}% persisted across ` +
          `${this.degradedWindows} observation windows.`,
        at: timestamp,
        evidence: { ...this.lastWindow },
      };
      return this.triggered;
    }
    return null;
  }

  shouldSampleQueue() {
    this.submittedFrames++;
    return this.submittedFrames % this.options.queueSampleEvery === 0;
  }

  observeQueueBacklog(durationMs, now = durationMs) {
    const duration = finiteNonNegative(durationMs);
    const timestamp = finiteNonNegative(now);
    if (duration == null || timestamp == null) return null;
    const threshold = Math.max(
      this.options.backlogThresholdMs,
      (this.frameIntervalMs || 0) * this.options.backlogFrameIntervals,
    );
    this.backlogSamples = duration >= threshold ? this.backlogSamples + 1 : 0;
    if (!this.triggered && this.backlogSamples >= this.options.consecutiveBacklogs) {
      this.triggered = {
        code: "sustained-gpu-backlog",
        detail: `GPU queue completion remained above ${Math.round(threshold)} ms for ` +
          `${this.backlogSamples} consecutive samples.`,
        at: timestamp,
        evidence: {
          durationMs: duration,
          thresholdMs: threshold,
          consecutiveBacklogs: this.backlogSamples,
        },
      };
      return this.triggered;
    }
    return null;
  }

  snapshot() {
    return {
      triggered: this.triggered ? { ...this.triggered } : null,
      lastWindow: this.lastWindow ? { ...this.lastWindow } : null,
      frameIntervalMs: this.frameIntervalMs,
      consecutiveDegradedWindows: this.degradedWindows,
      consecutiveBacklogs: this.backlogSamples,
    };
  }
}

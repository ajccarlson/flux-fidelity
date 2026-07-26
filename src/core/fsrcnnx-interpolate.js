// fsrcnnx-interpolate.js — real-time RIFE frame interpolation.
//
// Frame source: requestVideoFrameCallback on the <video> element + drawImage of
// the element to a canvas (clean YUV→RGB, the same kind of video-element read the
// upscaler does via importExternalTexture). This REPLACED an earlier captureStream
// → MediaStreamTrackProcessor → VideoFrame → createImageBitmap source, which
// introduced chroma-reconstruction "wave" artifacts on bright/high-detail regions
// (the upscaler, using importExternalTexture, never showed them — proving the
// artifact was in the VideoFrame conversion path, not the model or presentation).
//
// Pipeline: grab current frame → RIFE(prev, cur, 0.5) tween → enqueue tween+real
// → buffered strict-cadence scheduler draws to our canvas → audio delayed to match.

import { SRGB_COLOR_SPACE } from "./fsrcnnx-color-support.js";
import { videoPresentationState } from "./fsrcnnx-video-controller.js";

function interpolationDimensions(width, height) {
  return Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0
    ? Object.freeze({ width, height })
    : null;
}

// Chromium installs native loadedmetadata/ended listeners for the lifetime of an
// element every time captureStream() is called. Cache the first successful stream
// per element so toggles, model restarts, and source replacements cannot accumulate
// those listeners. Weak keys let a removed element and its native listener/stream
// cycle be collected with the document.
const elementCaptureRecords = new WeakMap();

function mediaStreamProvider(video) {
  let provider = null;
  try { provider = video?.srcObject || null; } catch { return null; }
  return provider && typeof provider.getTracks === "function" &&
    typeof provider.getAudioTracks === "function" ? provider : null;
}

function configurePersistentCaptureTrack(record, track) {
  if (!record || !track) return;
  const provider = mediaStreamProvider(record.video);
  // A cached URL capture follows later loadedmetadata events. Chromium can add
  // components from a page-owned srcObject to that old stream, so mark and leave
  // them untouched forever; active srcObject routing uses the provider directly.
  if (provider) {
    record.borrowedTracks.add(track);
    return;
  }
  if (record.borrowedTracks.has(track)) return;
  if (track.kind === "video") {
    if (track.readyState !== "ended") {
      try { track.stop?.(); } catch {}
    }
    return;
  }
  if (track.kind === "audio" && track.readyState !== "ended") {
    const current = record.trackGenerations.get(track) === record.sourceGeneration;
    try { track.enabled = current && !!record.owner; } catch {}
  }
}

function updatePersistentCaptureRecord(record, event = null) {
  if (!record) return;
  if (event?.type === "ended" && event?.target === record.video) record.exhausted = true;
  const metadataBoundary = event?.type === "loadedmetadata" && event?.target === record.video;
  if (metadataBoundary) {
    record.exhausted = false;
    record.hadAudio = false;
    ++record.sourceGeneration;
  }
  let tracks = [];
  try { tracks = Array.from(record.stream.getTracks()); } catch {}
  if (event?.type === "addtrack" && event.track && !tracks.includes(event.track)) {
    // Chromium dispatches addtrack asynchronously. A faster subsequent source
    // replacement can remove the component before this queued event arrives;
    // never promote that stale (possibly provider-backed) wrapper into the new
    // generation or mutate its enabled/stop state.
    record.borrowedTracks.add(event.track);
  }
  for (const track of tracks) {
    // The native Chromium listener runs before ours and synchronously updates
    // stream membership. Treat every live member at that boundary as current;
    // this also tolerates implementations that dispatch addtrack eagerly.
    if ((metadataBoundary && track?.readyState !== "ended") ||
        !record.trackGenerations.has(track)) {
      record.trackGenerations.set(track, record.sourceGeneration);
    }
    configurePersistentCaptureTrack(record, track);
  }
  if (tracks.some((track) => track?.kind === "audio" && track?.readyState !== "ended" &&
      record.trackGenerations.get(track) === record.sourceGeneration &&
      !record.borrowedTracks.has(track))) record.hadAudio = true;
  const session = record.owner;
  if (session && !session.disposed) {
    session.interpolator?._handleAudioCaptureChange?.(session, event);
  }
}

export class Interpolator {
  constructor(options = {}) {
    const { findVideo, log, warn, chain, onTerminalFailure } = options;
    this.findVideo = typeof findVideo === "function" ? findVideo : () => null;
    this.log = log || console.log;
    this.warn = warn || console.warn;
    this.onTerminalFailure = typeof onTerminalFailure === "function" ? onTerminalFailure : null;
    this.chain = chain || null; // upscaler chain accessors { tap, info, available, device }
    // An explicitly supplied source (element or getter) is authoritative, including
    // an explicit null.
    // This prevents the interpolator and renderer from independently scoring the
    // page and silently operating on different video elements.  start(video) is
    // also supported so existing constructor call sites can opt in incrementally.
    this._sourceProvided = Object.prototype.hasOwnProperty.call(options, "sourceVideo")
      || Object.prototype.hasOwnProperty.call(options, "video");
    this._sourceVideo = Object.prototype.hasOwnProperty.call(options, "sourceVideo")
      ? options.sourceVideo
      : options.video;
    this.running = false;
    this._state = "idle";       // "idle" | "starting" | "running"
    this._lifecycleGen = 0;      // invalidates asynchronous work after stop/restart
    this._startPromise = null;
    this._dimsRestarting = false;
    this._deviceRestarting = false;
    this._deviceRecoveryDevice = null;
    this._pendingDeviceLoss = null;
    this._gpuResourceStopQueued = false;
    this._pipelineFailureStopQueued = false;
    this._pipelineFailureStreaks = Object.create(null);
    this._pipelineFailureLimit = 5;
    this._deviceLossUnsubscribe = null;
    this._cpuGrabInit = null;
    this._cpuGrabRecovery = null;
    this._cpuGrabRecoveryDelays = [0, 100, 500];
    // WebGPUGrabber.destroy() is asynchronous because it drains any pending
    // readback before destroying its owned GPUDevice.  Logical ownership is
    // revoked synchronously, while this map keeps physical teardown visible to
    // every restart path so two grabber devices can never overlap.
    this._cpuGrabberTeardowns = new Map();
    this._resourceRetirementPromise = null;
    this._modelRetirementPromise = null;
    this._activeCpuFrameTasks = new Set();
    this._grab = null;
    this._grabCtx = null;
    this._blend = null;
    this._blendCtx = null;
    this._tw = null;
    this._twctx = null;
    this.video = null;
    this.overlay = null;
    this._takeoverActive = false;
    this._committedPresentation = null;
    this._presentationGeneration = 0;
    this._chainPresentationSuspended = false;
    this._mediaBoundaryVideo = null;
    this._mediaBoundaryHandlers = null;
    this._audioBoundaryPending = false;
    this._audioAttemptGeneration = 0;
    this._audioPreparation = null;
    this._audioCaptureSession = null;
    this._audioRoute = null;
    this._audioTimer = null;
    this._audioBlocks = new Set();
    this._productionWasEligible = true;
    this.processor = null;
    this.abort = null;
    // resolution control: "full" | "half" | "quarter" | "auto"
    // Now that static-region passthrough stabilizes still detail INDEPENDENTLY of
    // inference resolution, resolution no longer has to be maxed to avoid jitter —
    // so Auto can pick the resolution that holds cadence (much cheaper). Motion is
    // forgiving of lower res; static stability comes from passthrough, not res.
    this.resMode = "full";
    this._autoScale = 0.625; // explicit Auto starts moderate, then adapts toward the cadence budget
    // A/V fine-trim (ms) ON TOP of the measured buffer delay. The measured delay
    // (frame dwell time in the buffer) now matches the real gap, so this defaults
    // to 0 and is just a small manual nudge for hardware that needs it.
    this._avOffsetMs = 0;
    // Adaptive quality fallback: RIFE tweens are best but can be too slow (e.g.
    // 1080p), where the grab loop drops source frames and output rises <50% over
    // source — worse than not interpolating. When the measured output/source ratio
    // stays under this threshold, fall back to cheap blend tweens (which keep up and
    // hit ~2x). Sticky (won't oscillate); reset to RIFE on seek/pause/resolution
    // change to retry. 1.5 = "at least a 50% increase over source to be worth RIFE".
    this._interpMode = "rife";           // "rife" | "blend"
    this._fallbackRatio = 1.5;
    this._fallbackArmed = true;          // false once we've fallen back (until reset)
    this._autoFallback = false;           // performance downgrade is opt-in
    // Blend can multi-tween up to a target framerate (blend is ~free). Target is the
    // display refresh rate (auto-detected from rAF), with a manual override. RIFE
    // stays 2x (one tween) — N inferences per gap would be far too costly.
    this._forceBlend = false;            // true when user picks "Blend" as the model
    this._rifeModelKey = null;            // null = fsrcnnx-rife.js default
    this._targetFpsMode = "auto";        // "auto" (refresh rate) | a number
    this._detectedHz = null;             // measured display refresh (rAF)
    this._maxTweensPerGap = 7;           // cap N-1 (so up to 8x) to bound cost/memory
    this.stats = { framesIn: 0, framesOut: 0, started: 0, lastReport: 0, maxDriftMs: 0, maxGapMs: 0, lastGapMs: 0, stutters: 0 };
  }

  _resolveSourceVideo() {
    if (this._sourceProvided) {
      try {
        return typeof this._sourceVideo === "function" ? this._sourceVideo() : this._sourceVideo;
      } catch (error) {
        this.warn("interpolation: source accessor failed:", error.message);
        return null;
      }
    }
    // A chain source accessor is also authoritative.  In particular, null means
    // that the renderer currently owns no video; falling through to findVideo()
    // here would recreate the cross-video ownership bug this accessor prevents.
    if (typeof this.chain?.source === "function") {
      try { return this.chain.source(); }
      catch (error) {
        this.warn("interpolation: source accessor failed:", error.message);
        return null;
      }
    }
    return this.findVideo();
  }

  _chainOwnsVideo(video) {
    if (!this.chain) return false;
    if (typeof this.chain.source !== "function") return true;
    try { return this.chain.source() === video; }
    catch { return false; }
  }

  _chainCanInvert(video) {
    if (!this._chainOwnsVideo(video) || typeof this.chain?.canInvert !== "function") return false;
    try { return this.chain.canInvert(video) === true; }
    catch { return false; }
  }

  setResMode(mode) {
    if (["full", "half", "quarter", "auto"].includes(mode)) this.resMode = mode;
    return this.resMode;
  }

  // Interpolation "model": a RIFE model key, or "blend" for the non-AI blend mode.
  // Returns true if handled here (blend); false lets the caller set a RIFE model.
  setInterpEngine(key) {
    if (key === "blend") {
      this._forceBlend = true; this._interpMode = "blend";
      return true;
    }
    // Keep the choice even when the RIFE module has not been imported yet. This is
    // what makes a preference restored before start() reach initRife(). The return
    // value remains backward-compatible: false still identifies a RIFE engine.
    if (typeof key === "string" && key) this._rifeModelKey = key;
    this._forceBlend = false; this._interpMode = "rife"; this._fallbackArmed = true;
    if (!this.running && this._rifeMod && this._rifeMod.setModel && this._rifeModelKey) {
      this._rifeMod.setModel(this._rifeModelKey);
    }
    return false;
  }

  // Blend target framerate: "auto" (display refresh) or a number (manual override).
  setTargetFps(v) {
    this._targetFpsMode = (v === "auto" || v == null) ? "auto" : Math.max(24, Math.min(480, Number(v) || 0));
    return this._targetFpsMode;
  }

  // Effective blend target: auto → detected refresh (fallback 60 if undetected).
  _effectiveTargetFps() {
    if (this._targetFpsMode === "auto") return this._detectedHz || 60;
    return this._targetFpsMode;
  }

  // How many tweens to insert in a source-frame gap to approach the target fps.
  // Derived per-gap from the actual timestamps (µs), so it self-corrects for uneven
  // gaps. Returns N-1 tweens (the real frame is the Nth). RIFE ignores this (2x).
  _tweensForGap(prevTsUs, tsUs) {
    const gapS = Math.max(1e-4, (tsUs - prevTsUs) / 1e6);
    const framesNeeded = Math.round(this._effectiveTargetFps() * gapS);
    return Math.max(1, Math.min(this._maxTweensPerGap, framesNeeded - 1));
  }

  // Self-clocking RIFE chain for one pair. Bisection levels {1,3,7} — each a
  // uniform midpoint-anchored sampling, so stopping at any level boundary keeps
  // even cadence. No cross-gap prediction: level gates use THIS chain's own
  // measured per-inference time vs the actual gap (saturation latency measured
  // under the real load), and the NEXT pair preempts between inferences via the
  // pending slot instead of being skipped. Owns one ref on prev.tex and cur.tex;
  // releases both at the end, then immediately chains to a pending pair if one
  // parked while we ran.
  async _rifeChain(prev, cur, lifecycleGeneration = this._lifecycleGen) {
    const stats = this.stats;
    const gen = this._flushGen || 0;
    const lifecycleCurrent = () => this._isCurrent(lifecycleGeneration);
    const p0 = prev.ts, p1 = cur.ts;
    const gapMs = Math.max(1, (p1 - p0) / 1000);
    const desired = this._tweensForGap(p0, p1);
    const tAware = !this._rifeMod.timestepAware || this._rifeMod.timestepAware();
    const infScale = this._resolveScale(); // REAL now (v0.48.5): resolved once per chain; auto adapts between chains
    const levels = [[0.5]];
    if (tAware && desired >= 3) levels.push([0.25, 0.75]);
    if (tAware && desired >= 7) levels.push([0.125, 0.375, 0.625, 0.875]);
    const t0 = performance.now();
    let dtLast = 0;
    const produced = []; // {frac, tex} RIFE tweens retained for the blend ladder
    try {
      outer: for (let li = 0; li < levels.length; li++) {
        // gate: does this whole level fit in the remaining gap at measured cost?
        if (li > 0 && (performance.now() - t0) + levels[li].length * dtLast > gapMs * 0.95) break;
        for (const frac of levels[li]) {
          if (!lifecycleCurrent() || (this._flushGen || 0) !== gen || this._pendingPair) break outer;
          const tInf0 = performance.now();
          const tweenTex = await this._rifeMod.gpuTweenPair(prev.tex, cur.tex, frac, this._staticOn !== false, infScale);
          dtLast = Math.max(0.5, performance.now() - tInf0); // floor: a timing anomaly must never read as "inference is free"
        // rolling window: single-inference snapshots vary with queue position
        // (first-in-chain eats co-tenant wait, later ones don't) — the MEAN is the
        // comparable number across builds/experiments.
        (this._infWindow || (this._infWindow = [])).push(dtLast);
        if (this._infWindow.length > 60) this._infWindow.shift();
          stats.lastInferMs = dtLast;
          if (dtLast > (stats.maxInferMs || 0)) stats.maxInferMs = dtLast;
          this._adaptScale(dtLast, p0, p1);
          if (tweenTex) {
            this._tweenFailStreak = 0;
            if ((this._flushGen || 0) !== gen) { this._rifeMod.gpuRelease(tweenTex); break outer; }
            this._rifeMod.gpuRetain(tweenTex); // ladder ref (released in finally)
            produced.push({ frac, tex: tweenTex });
            if (this._enqueueTexOrdered(tweenTex, Math.round(p0 + (p1 - p0) * frac))) stats.framesOut++;
          } else if ((this._flushGen || 0) === gen && lifecycleCurrent()) {
            // CIRCUIT BREAKER: repeated inference failure (any cause — EP error,
            // shape mismatch, device trouble) must NEVER become an error storm
            // (the v0.45.0 freeze). Five consecutive failures → blend fallback.
            this._tweenFailStreak = (this._tweenFailStreak || 0) + 1;
            if (this._tweenFailStreak >= 5 && this._interpMode === "rife") {
              this._interpMode = "blend";
              this.warn("interp: RIFE failing repeatedly — blend fallback (circuit breaker)");
              break outer;
            }
          }
        }
      }
      // HYBRID BLEND LADDER (#3): when RIFE stopped short of the desired tween
      // count (gates, preemption, 6ch model), subdivide BETWEEN ADJACENT produced
      // frames with pair-blends. Each blend spans only a fraction of the source
      // gap, so ghosting shrinks proportionally (vs full-gap blend mode). Exactly
      // ONE blend level — blends only between real/RIFE frames, never
      // blend-of-blend. Sub-ms cost, so no gating; runs even when preempted.
      if (this._ladderOn === true && lifecycleCurrent() && (this._flushGen || 0) === gen && this._interpMode === "rife"
          && this._rifeMod.gpuBlendPair && produced.length < desired) {
        const frames = [{ frac: 0, tex: prev.tex }, ...produced, { frac: 1, tex: cur.tex }]
          .sort((a, b) => a.frac - b.frac);
        let count = produced.length;
        // CADENCE: fill only if the WHOLE level fits under `desired` — a partial
        // left-to-right fill subdivides the gap's first half but not its second,
        // producing uneven frame spacing (judder). Even cadence beats raw count.
        if (desired - count < frames.length - 1) return; // (finally still runs)
        for (let i = 0; i < frames.length - 1 && count < desired; i++) {
          const f = (frames[i].frac + frames[i + 1].frac) / 2;
          const btex = this._rifeMod.gpuBlendPair(frames[i].tex, frames[i + 1].tex, 0.5);
          if (btex && this._enqueueTexOrdered(btex, Math.round(p0 + (p1 - p0) * f))) {
            stats.framesOut++; count++;
            stats.ladderBlends = (stats.ladderBlends || 0) + 1;
          }
        }
      }
    } catch (e) {
      this.warn("rife chain failed:", e.message);
    } finally {
      for (const it of produced) { try { this._rifeMod.gpuRelease(it.tex); } catch {} }
      this._rifeMod.gpuRelease(prev.tex);
      this._rifeMod.gpuRelease(cur.tex);
      // A newer lifecycle owns _pendingPair/_inferBusy now. Never consume or reset
      // its scheduler state from an inference that belonged to the stopped run.
      if (!lifecycleCurrent()) return;
      // pop pending unconditionally: flush clears stale pairs, so anything
      // parked here is current-generation and valid (the new chain re-reads gen).
      const next = this._pendingPair;
      this._pendingPair = null;
      if (next) this._rifeChain(next.prev, next.cur, lifecycleGeneration); // refs transfer
      else this._inferBusy = false;
    }
  }


  // Toggle the hybrid blend ladder (experiment #3 A/B). Live: chains read the
  // flag per pair, no restart needed.
  setLadder(on) { this._ladderOn = !!on; return this._ladderOn; }

  // Toggle the PERFORMANCE fallback evaluator (RIFE→blend when output ratio
  // sags). Persisted per site; default OFF. The error CIRCUIT BREAKER (5
  // consecutive inference failures → blend) is deliberately NOT gated by this —
  // it guards against error storms, not slowness, and always announces itself.
  setAutoFallback(on) {
    this._autoFallback = !!on;
    if (this._autoFallback) {
      this._fallbackArmed = true; this._srcFrameBase = null; // fresh window
    } else if (this._interpMode === "blend" && !this._forceBlend
               && this._rifeMod && this._rifeMod.gpuRifeCapable && this._rifeMod.gpuRifeCapable()) {
      // disabling doubles as a "give me RIFE back NOW" lever mid-test
      this._interpMode = "rife"; this._srcFrameBase = null;
      this.log("interp: auto-fallback disabled — restoring RIFE");
    }
    return this._autoFallback;
  }

  // adjust A/V calibration offset (ms). Positive = delay audio more (use if audio
  // is ahead of video); negative = less (if audio lags video). Clamped sanely.
  setAvOffset(ms) {
    const v = Number(ms);
    if (isFinite(v)) this._avOffsetMs = Math.max(-100, Math.min(300, v));
    return this._avOffsetMs;
  }

  // Resolve the inference resolution scale for this frame.
  _resolveScale() {
    switch (this.resMode) {
      case "full": return 1.0;
      case "half": return 0.5;
      case "quarter": return 0.25;
      case "auto":
      default: return this._autoScale;
    }
  }

  // Auto mode: nudge the inference scale toward the frame budget. If inference is
  // over ~80% of the frame interval, step down (faster); if comfortably under
  // ~45%, step up (sharper). Hysteresis avoids oscillation. Clamped to [0.25,1].
  _adaptScale(inferMs, tsA, tsB) {
    if (this.resMode !== "auto") return;
    // estimate the source frame interval from recent timestamps (microseconds)
    const intervalMs = (tsB - tsA) / 1000 || 33;
    const ratio = inferMs / intervalMs;
    const steps = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0];
    let idx = steps.indexOf(this._autoScale);
    if (idx < 0) { idx = 3; } // default 0.625 if off-grid
    // Target keeping inference comfortably within the per-frame budget. Step down
    // when we're using too much of it (>0.7), step up only with real headroom
    // (<0.4). Static stability no longer depends on resolution (passthrough), so
    // prefer the cheapest resolution that holds smooth cadence.
    if (ratio > 0.7 && idx > 0) idx--;
    else if (ratio < 0.4 && idx < steps.length - 1) idx++;
    this._autoScale = steps[idx];
  }

  // push a display-size bitmap with its source timestamp (microseconds). Frames
  // are produced in order (real, tween, real, tween, ...); we present them on a
  // strict cadence rather than by raw timestamp, so RIFE latency can't clump them.
  _enqueue(bmp, ts) {
    if (!this.queue) { bmp.close && bmp.close(); return; }
    this.queue.push({ bmp, ts, enq: performance.now() }); // enq = wall time queued
    // learn the target presentation interval from consecutive frame timestamps
    // (these already include the tween midpoints, so this is the OUTPUT interval).
    if (this._lastEnqTs != null) {
      const dtMs = (ts - this._lastEnqTs) / 1000;
      if (dtMs > 1 && dtMs < 200) {
        // smooth estimate of the output frame interval
        this._targetInterval = this._targetInterval
          ? this._targetInterval * 0.9 + dtMs * 0.1
          : dtMs;
      }
    }
    this._lastEnqTs = ts;
    // hard cap so a stall can't grow memory unbounded. Drop the NEWEST (pop), never
    // the head: dropping the oldest under sustained overload pushed the about-to-be-
    // due frame off the queue every tick — a livelock where the head was forever
    // ~350ms in the future and NOTHING ever presented (the freeze Aaron isolated).
    while (this.queue.length > 48) { const o = this.queue.pop(); o.bmp && o.bmp.close && o.bmp.close(); if (o.tex && this._rifeMod) this._rifeMod.gpuRelease(o.tex); }
  }

  // GPU-present variant: queue a pooled result TEXTURE (no bitmap). Same timestamp/
  // Pipelined tweens finish AFTER their surrounding real frames were enqueued, so
  // they must be inserted by timestamp (the queue must stay ascending for the
  // scheduler). Returns false (and releases) if the slot already presented or a
  // flush invalidated the pair.
  _enqueueTexOrdered(tex, ts) {
    if (!this.queue || (this._lastPresentedTs != null && ts <= this._lastPresentedTs)) {
      this._rifeMod && this._rifeMod.gpuRelease(tex);
      return false;
    }
    let i = this.queue.length;
    while (i > 0 && this.queue[i - 1].ts > ts) i--;
    this.queue.splice(i, 0, { tex, ts, enq: performance.now() });
    // same overload policy as _enqueueTex: drop NEWEST
    while (this.queue.length > 48) { const o = this.queue.pop(); if (o.tex && this._rifeMod) this._rifeMod.gpuRelease(o.tex); o.bmp && o.bmp.close && o.bmp.close(); }
    return true;
  }

  // interval learning; the present loop renders it via WebGPU and recycles it.
  _enqueueTex(tex, ts) {
    if (!this.queue) { this._rifeMod && this._rifeMod.gpuRelease(tex); return; }
    this.queue.push({ tex, ts, enq: performance.now() });
    if (this._lastEnqTs != null) {
      const dtMs = (ts - this._lastEnqTs) / 1000;
      if (dtMs > 1 && dtMs < 200) this._targetInterval = this._targetInterval ? this._targetInterval * 0.9 + dtMs * 0.1 : dtMs;
    }
    this._lastEnqTs = ts;
    // drop NEWEST under overload (see bitmap enqueue comment: oldest-drop livelocks)
    while (this.queue.length > 48) { const o = this.queue.pop(); if (o.tex && this._rifeMod) this._rifeMod.gpuRelease(o.tex); o.bmp && o.bmp.close && o.bmp.close(); }
  }

  // Timestamp-accurate scheduler. Each frame carries its true source timestamp
  // (real frames at their mediaTime, tweens at the exact midpoint). We anchor a
  // mapping between source-time and wall-clock once the buffer has filled, then
  // present each frame when ITS OWN timestamp is due. Presenting on a blind
  // constant cadence (the old approach) placed the spatial-t=0.5 tween at a
  // non-0.5 time, which made static high-detail regions appear to jitter. Mapping
  // each frame to its own timestamp keeps spatial and temporal position matched.
  _present(generation = this._lifecycleGen) {
    const FILL_MS = 100;
    let started = false;
    let anchorWall = 0;   // wall-clock (ms) mapped to anchorSrc
    let anchorSrc = 0;    // source timestamp (ms) of the presentation origin
    // Display refresh-rate detection: sample rAF intervals; the median over a window
    // ≈ the frame period → Hz. Used as the blend target when target mode is "auto".
    let hzSamples = [];
    let hzLast = 0;
    const loop = () => {
      if (!this._isCurrent(generation) || !this.overlay) return;
      const q = this.queue;
      const now = performance.now();
      const interval = this._targetInterval || 16.7;

      // refresh-rate estimate from rAF cadence (median of recent intervals)
      if (hzLast) {
        const d = now - hzLast;
        if (d > 1 && d < 100) {
          hzSamples.push(d);
          if (hzSamples.length >= 60) {
            const s = hzSamples.slice().sort((a, b) => a - b);
            const med = s[s.length >> 1];
            if (med > 0) this._detectedHz = Math.round(1000 / med);
            hzSamples = [];
          }
        }
      }
      hzLast = now;

      if (q && q.length) {
        // discontinuity (seek/pause/flush): the wall↔source mapping is stale — rebuild
        if (this._reanchor) { this._reanchor = false; started = false; }
        if (!started) {
          const buffered = q.length * interval;
          if (buffered >= FILL_MS || q.length >= 4) {
            started = true;
            anchorWall = now;
            anchorSrc = q[0].ts / 1000; // µs → ms
          }
        }
        // SELF-HEAL: if the head frame is due implausibly far in the future OR the
        // past, the anchor was poisoned by a source-time jump (forward: resume-seek;
        // backward: e.g. YouTube restarting at 0 after a stream reset) — re-anchor.
        if (started && q.length) {
          const headDue = anchorWall + (q[0].ts / 1000 - anchorSrc);
          if (headDue - now > 1500 || headDue - now < -3000) {
            this.log && this.log(`interp: re-anchoring presentation (source time jumped ${((headDue - now)/1000).toFixed(1)}s)`);
            anchorWall = now;
            anchorSrc = q[0].ts / 1000;
          }
        }
        // WATCHDOG + RECOVERY: frames queued but none presented for >1s means the
        // scheduler is stalled — log its state and RE-ANCHOR to the head so playback
        // resumes immediately (no pause/play needed).
        if (started && q.length && this._lastPresentAt && now - this._lastPresentAt > 1000) {
          if (!this._stallLogAt || now - this._stallLogAt > 5000) {
            this._stallLogAt = now;
            const hd = anchorWall + (q[0].ts / 1000 - anchorSrc) - now;
            this.warn(`interp WATCHDOG: present stalled ${(now - this._lastPresentAt).toFixed(0)}ms (q=${q.length}, headDue=${hd.toFixed(0)}ms, headTs=${(q[0].ts/1e6).toFixed(2)}s) — re-anchoring to recover`);
          }
          anchorWall = now;
          anchorSrc = q[0].ts / 1000;
          this._lastPresentAt = now; // presenting resumes this tick
        }
        if (started) {
          // present every frame whose source time is due relative to the anchor.
          // (usually one per rAF tick; the loop handles catch-up if behind.)
          let presented = false;
          while (q.length) {
            const item = q[0];
            const dueWall = anchorWall + (item.ts / 1000 - anchorSrc);
            if (now >= dueWall) {
              q.shift();
              const presentationCandidate = {
                gpu: !!item.tex,
                width: item.tex?._w ?? item.bmp?.width,
                height: item.tex?._h ?? item.bmp?.height,
              };
              let itemReleased = false;
              const releaseItem = () => {
                if (itemReleased) return;
                itemReleased = true;
                if (item.tex) { this._rifeMod && this._rifeMod.gpuRelease(item.tex); }
                else { item.bmp.close && item.bmp.close(); }
              };
              // Audio routing is prepared behind a closed gain gate before either
              // visual sink runs. Native audio is muted only after the output and
              // overlay commit, so preparation failure always remains reversible.
              let takeover;
              try {
                takeover = this._stageTakeover(generation);
              } catch (error) {
                releaseItem();
                this._handlePipelineFailure(generation, error, "display takeover", { terminal: true });
                continue;
              }
              if (!takeover) {
                releaseItem();
                continue;
              }
              let outputReady = false;
              let outputError = null;
              let terminalOutputFailure = false;
              try {
                if (item.tex) {
                  // GPU present: render the pooled texture via WebGPU, then recycle.
                  // Inverted chain is legal only while the renderer continues to
                  // advertise an upscale path for this exact source video.
                  if (this._chainInverted && this.chain && this.chain.upscaleTex) {
                    if (!this._chainCanInvert(this.video)) {
                      terminalOutputFailure = true;
                      throw new Error("inverted upscale capability is no longer available");
                    }
                    outputReady = this.chain.upscaleTex(item.tex, item.tex._w, item.tex._h) === true;
                    if (!outputReady && !this._invUpWarned) {
                      this._invUpWarned = true;
                      this.warn("interp: INVERTED upscale REJECTED (upscaler off/deviceless?) — frames are being dropped");
                    }
                  } else {
                    outputReady = this._rifeMod?.gpuPresent?.(item.tex) === true;
                  }
                } else {
                  // bitmap present: lazily grab a 2D context if WebGPU didn't claim it
                  if (!this._octx && !this._gpuPresent) {
                    try { this._octx = this.overlay.getContext("2d", { colorSpace: SRGB_COLOR_SPACE }); }
                    catch {}
                  }
                  if (this._octx) {
                    if (this.overlay.width !== item.bmp.width || this.overlay.height !== item.bmp.height) {
                      this.overlay.width = item.bmp.width; this.overlay.height = item.bmp.height;
                    }
                    this._octx.drawImage(item.bmp, 0, 0);
                    outputReady = true;
                  }
                }
              } catch (error) {
                outputError = error;
              } finally {
                releaseItem();
              }
              if (!outputReady) {
                takeover.audioTransaction?.rollback?.();
                this._handlePipelineFailure(
                  generation,
                  outputError || new Error("presentation returned no output"),
                  "presentation",
                  { terminal: terminalOutputFailure },
                );
                continue;
              }
              this._recordPipelineSuccess("presentation");
              if (!this._activateTakeover(generation, takeover)) continue;
              this._recordCommittedPresentation(presentationCandidate);
              this._lastPresentAt = now;
              this._lastPresentedTs = item.ts;
              const dwell = now - item.enq;
              if (dwell >= 0 && dwell < 1000) {
                if (this._discontinuity) { this._videoLatencyMs = dwell; this._discontinuity = false; this._snapAudio = true; }
                else this._videoLatencyMs = this._videoLatencyMs == null ? dwell : this._videoLatencyMs * 0.9 + dwell * 0.1;
              }
              presented = true;
            } else break;
          }
          // if we've drifted far from the anchor (buffer underran / stall), re-anchor
          if (presented && q.length) {
            const nextDue = anchorWall + (q[0].ts / 1000 - anchorSrc);
            if (now - nextDue > interval * 3) { anchorWall = now; anchorSrc = q[0].ts / 1000; }
          }
          if (q.length === 0) started = false;
        }
      } else if (started) {
        started = false;
      }
      if (this._isCurrent(generation)) this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  // Audio takeover is intentionally reversible. HTMLMediaElement audio source
  // nodes permanently redirect an element into one AudioContext, which can strand
  // a later CORS-opaque resource in silence. Instead we capture temporary audio
  // tracks, stage their graph behind a zero-gain gate, and mute the element only
  // after the replacement frame and overlay have both succeeded.
  _setAudioBlocked(reason, blocked) {
    if (!reason) return;
    if (blocked) this._audioBlocks.add(reason);
    else this._audioBlocks.delete(reason);
    this._audioBoundaryPending = this._audioBlocks.size > 0;
  }

  _clearAudioBlocks() {
    this._audioBlocks.clear();
    this._audioBoundaryPending = false;
  }

  _readMediaSinkId(video) {
    try { return typeof video?.sinkId === "string" ? video.sinkId : ""; }
    catch { return ""; }
  }

  _audioSinkMatches(preparation, video = preparation?.video) {
    if (!preparation || this._readMediaSinkId(video) !== preparation.sinkId) return false;
    if (!preparation.sinkId) return true;
    try {
      return preparation.context?.sinkId === preparation.sinkId;
    } catch {
      return false;
    }
  }

  _clearAudioPreparationWait(preparation) {
    if (!preparation) return;
    if (preparation.timeout != null) {
      clearTimeout(preparation.timeout);
      preparation.timeout = null;
    }
    if (preparation.visibilityListener) {
      try { globalThis.document?.removeEventListener?.("visibilitychange", preparation.visibilityListener); } catch {}
      preparation.visibilityListener = null;
    }
    this._setAudioBlocked(preparation.blockReason, false);
  }

  _closeAudioContext(preparation) {
    if (!preparation) return;
    this._clearAudioPreparationWait(preparation);
    preparation.setupPromise = null;
    if (!preparation.context || preparation.contextClosed) return;
    preparation.contextClosed = true;
    try {
      const closing = preparation.context.close?.();
      if (closing && typeof closing.catch === "function") closing.catch(() => {});
    } catch {}
  }

  _failAudioPreparation(preparation, error) {
    if (!preparation) return;
    const current = this._audioPreparation === preparation &&
      preparation.token === this._audioAttemptGeneration;
    preparation.status = "blocked";
    preparation.error = error;
    this._closeAudioContext(preparation);
    if (!current || preparation.failureReported || !this._isCurrent(preparation.generation)) return;
    preparation.failureReported = true;
    this._setAudioBlocked("audio-terminal", true);
    this._handlePipelineFailure(
      preparation.generation,
      error || new Error("audio delay unavailable"),
      "audio takeover",
      { terminal: true },
    );
  }

  _resumeAudioPreparation(preparation) {
    if (!preparation || preparation.status === "blocked" || preparation.contextClosed) return preparation;
    if (preparation.setupPromise) return preparation;
    const { context } = preparation;
    if (!context || context.state === "closed") {
      this._failAudioPreparation(preparation, new Error("AudioContext is closed"));
      return preparation;
    }
    if (globalThis.document?.hidden) {
      preparation.status = "waiting-visible";
      this._setAudioBlocked(preparation.blockReason, true);
      if (!preparation.visibilityListener) {
        preparation.visibilityListener = () => {
          if (globalThis.document?.hidden || this._audioPreparation !== preparation) return;
          try { globalThis.document.removeEventListener("visibilitychange", preparation.visibilityListener); } catch {}
          preparation.visibilityListener = null;
          preparation.status = "idle";
          this._resumeAudioPreparation(preparation);
        };
        try { globalThis.document.addEventListener("visibilitychange", preparation.visibilityListener); } catch {}
      }
      return preparation;
    }

    preparation.status = "pending";
    this._setAudioBlocked(preparation.blockReason, true);
    const setup = (async () => {
      if (preparation.sinkId) {
        if (typeof context.setSinkId !== "function") {
          throw new Error("AudioContext cannot mirror the media output device");
        }
        await context.setSinkId(preparation.sinkId);
        if (this._readMediaSinkId(preparation.video) !== preparation.sinkId) {
          const error = new Error("media output device changed during audio preparation");
          error.code = "AUDIO_SINK_STALE";
          throw error;
        }
        if (!this._audioSinkMatches(preparation)) {
          throw new Error("AudioContext output device did not match the media element");
        }
      }
      if (context.state !== "running") await context.resume?.();
      if (context.state !== "running") throw new Error("AudioContext remained suspended");
    })();
    preparation.setupPromise = setup;
    const armTimeout = () => {
      if (preparation.timeout != null) clearTimeout(preparation.timeout);
      preparation.timeout = null;
      if (globalThis.document?.hidden) return;
      preparation.timeout = setTimeout(() => {
        if (this._audioPreparation === preparation && preparation.setupPromise === setup &&
            !globalThis.document?.hidden) {
          this._failAudioPreparation(preparation, new Error("AudioContext preparation timed out"));
        }
      }, 3000);
    };
    preparation.visibilityListener = () => {
      if (this._audioPreparation !== preparation || preparation.setupPromise !== setup) return;
      if (globalThis.document?.hidden) {
        if (preparation.timeout != null) clearTimeout(preparation.timeout);
        preparation.timeout = null;
        preparation.status = "waiting-visible";
      } else {
        preparation.status = "pending";
        armTimeout();
      }
    };
    try { globalThis.document?.addEventListener?.("visibilitychange", preparation.visibilityListener); } catch {}
    armTimeout();
    setup.then(() => {
      const current = this._audioPreparation === preparation &&
        preparation.token === this._audioAttemptGeneration &&
        this._isCurrent(preparation.generation) && this.video === preparation.video;
      if (!current) {
        this._closeAudioContext(preparation);
        return;
      }
      this._clearAudioPreparationWait(preparation);
      preparation.setupPromise = null;
      preparation.status = "ready";
    }).catch((error) => {
      if (preparation.setupPromise === setup) preparation.setupPromise = null;
      const current = this._audioPreparation === preparation &&
        preparation.token === this._audioAttemptGeneration &&
        this._isCurrent(preparation.generation) && this.video === preparation.video;
      const staleSink = error?.code === "AUDIO_SINK_STALE" ||
        this._readMediaSinkId(preparation.video) !== preparation.sinkId;
      if (staleSink) {
        preparation.stagedTransaction?.rollback?.();
        preparation.status = "stale";
        if (current) {
          this._audioPreparation = null;
          ++this._audioAttemptGeneration;
        }
        this._closeAudioContext(preparation);
        return;
      }
      if (current && globalThis.document?.hidden) {
        this._clearAudioPreparationWait(preparation);
        preparation.status = "idle";
        this._resumeAudioPreparation(preparation);
        return;
      }
      this._failAudioPreparation(preparation, error);
    });
    return preparation;
  }

  _prepareAudioDelay(video, generation) {
    const sinkId = this._readMediaSinkId(video);
    const existing = this._audioPreparation;
    if (existing && existing.video === video && existing.generation === generation &&
        existing.sinkId === sinkId) {
      if (existing.status === "ready" && existing.context?.state !== "running") {
        existing.status = "idle";
        this._resumeAudioPreparation(existing);
      }
      return existing;
    }
    if (existing) {
      existing.stagedTransaction?.rollback?.();
      this._closeAudioContext(existing);
    }
    const token = ++this._audioAttemptGeneration;
    const AC = globalThis.window?.AudioContext || globalThis.window?.webkitAudioContext;
    const preparation = {
      token,
      generation,
      video,
      sinkId,
      status: "idle",
      context: null,
      contextClosed: false,
      failureReported: false,
      setupPromise: null,
      timeout: null,
      visibilityListener: null,
      stagedTransaction: null,
      blockReason: `audio-context-${token}`,
    };
    this._audioPreparation = preparation;
    if (!AC) {
      this._failAudioPreparation(preparation, new Error("Web Audio is unavailable"));
      return preparation;
    }
    try { preparation.context = new AC(); }
    catch (error) {
      this._failAudioPreparation(preparation, error);
      return preparation;
    }
    return this._resumeAudioPreparation(preparation);
  }

  _audioSourceEligible(video) {
    if (!video) return false;
    const hasProvider = video.srcObject != null;
    if (!hasProvider) {
      const raw = video.currentSrc || video.src || "";
      if (!raw) return false;
      let sourceUrl;
      let pageUrl;
      try {
        const base = globalThis.document?.baseURI || globalThis.location?.href;
        sourceUrl = new URL(raw, base);
        pageUrl = new URL(base);
      } catch {
        return false;
      }
      // Data URLs have opaque resource origins. Blob/MSE URLs are safe only when
      // they belong to this document; ordinary cross-origin URLs require the media
      // element to have requested CORS explicitly. A playable CORS-mode element has
      // already passed response validation, while a no-CORS resource would be
      // required by Web Audio to emit silence without throwing.
      if (sourceUrl.protocol === "data:") return false;
      const sameOrigin = sourceUrl.origin === pageUrl.origin;
      const corsRequested = typeof video.crossOrigin === "string" ||
        video.hasAttribute?.("crossorigin") === true;
      if (sourceUrl.protocol === "blob:") {
        if (!sameOrigin) return false;
      } else if (!sameOrigin && !corsRequested) {
        return false;
      }
    }

    // URL checks cannot see a same-origin URL that redirected to a no-CORS
    // cross-origin response. Verify that the selected frame is origin-clean before
    // accepting its capture tracks as the replacement audio path.
    const canvas = globalThis.document?.createElement?.("canvas");
    if (canvas) {
      try {
        canvas.width = 1;
        canvas.height = 1;
        const context = canvas.getContext("2d", {
          colorSpace: SRGB_COLOR_SPACE,
          willReadFrequently: true,
        });
        if (!context) return false;
        context.drawImage(video, 0, 0, 1, 1);
        context.getImageData(0, 0, 1, 1);
      } catch {
        return false;
      }
    }
    return true;
  }

  _audioSourceSnapshot(video) {
    return {
      video,
      currentSrc: video?.currentSrc || "",
      src: video?.src || "",
      srcObject: video?.srcObject || null,
    };
  }

  _audioSourceMatches(snapshot, video) {
    return !!snapshot && snapshot.video === video &&
      snapshot.currentSrc === (video?.currentSrc || "") &&
      snapshot.src === (video?.src || "") &&
      snapshot.srcObject === (video?.srcObject || null);
  }

  _capturedAudioTracksHealthy(owner) {
    return !!owner && owner.audioTracks.length > 0 && owner.audioTracks.some((track) =>
      track?.readyState !== "ended" && track?.muted !== true);
  }

  _syncOwnedAudioTracks(route) {
    if (!route || route.disposed) return false;
    for (const pair of route.trackPairs || []) {
      if (!pair.source || !pair.output) return false;
      try {
        pair.output.enabled = pair.source.readyState !== "ended" && pair.output.readyState !== "ended" &&
          pair.source.enabled !== false && pair.source.muted !== true;
      }
      catch { return false; }
    }
    return true;
  }

  _refreshAudioCaptureTracks(session) {
    if (!session || session.disposed) return;
    for (const [track, type, listener] of session.trackListeners) {
      try { track.removeEventListener?.(type, listener); } catch {}
    }
    session.trackListeners = [];
    let tracks = [];
    try {
      tracks = Array.from(session.stream.getAudioTracks())
        .filter((track) => track?.readyState !== "ended" &&
          !session.captureRecord?.borrowedTracks?.has(track) &&
          (!session.captureRecord || session.captureRecord.trackGenerations.get(track) ===
            session.captureRecord.sourceGeneration));
    } catch {}
    session.audioTracks = tracks;
    const changed = (event) => this._handleAudioCaptureChange(session, event);
    for (const track of tracks) {
      for (const type of ["mute", "unmute", "ended"]) {
        try {
          track.addEventListener?.(type, changed);
          session.trackListeners.push([track, type, changed]);
        } catch {}
      }
    }
  }

  _handleAudioCaptureChange(session, event = null) {
    if (!session || session !== this._audioCaptureSession || session.disposed ||
        !this._isCurrent(session.generation) || this.video !== session.video) return;
    if (!this._audioSourceMatches(session.snapshot, session.video)) {
      this._relinquishPresentation({ preserveAudioContext: true, retireCapture: true });
      return;
    }
    if (session.captureRecord?.exhausted && session.captureRecord.hadAudio) {
      this._setAudioBlocked("audio-terminal", true);
      this._handlePipelineFailure(
        session.generation,
        new Error("the media element's cached audio capture ended; a new source is required"),
        "audio takeover",
        { terminal: true },
      );
      return;
    }
    this._refreshAudioCaptureTracks(session);
    const healthy = this._capturedAudioTracksHealthy(session);
    const noAudio = session.audioTracks.length === 0;
    this._setAudioBlocked(session.blockReason, !healthy && !noAudio);
    const staged = this._audioPreparation?.stagedTransaction;
    if (this._audioRoute || staged) {
      this._relinquishPresentation({ preserveAudioContext: true });
    }
  }

  _disposeAudioCaptureSession(session = this._audioCaptureSession) {
    if (!session || session.disposed) return;
    session.disposed = true;
    this._setAudioBlocked(session.blockReason, false);
    if (!session.captureRecord) {
      try { session.stream.removeEventListener?.("addtrack", session.onTrackChange); } catch {}
      try { session.stream.removeEventListener?.("removetrack", session.onTrackChange); } catch {}
    }
    for (const [track, type, listener] of session.trackListeners || []) {
      try { track.removeEventListener?.(type, listener); } catch {}
    }
    session.trackListeners = [];
    if (session.captureRecord?.owner === session) {
      session.captureRecord.owner = null;
      updatePersistentCaptureRecord(session.captureRecord);
    }
    if (this._audioCaptureSession === session) this._audioCaptureSession = null;
    session.audioTracks = [];
    session.captureRecord = null;
    session.stream = null;
  }

  _ensureAudioCaptureSession(video, generation) {
    const snapshot = this._audioSourceSnapshot(video);
    const existing = this._audioCaptureSession;
    if (existing && !existing.disposed && existing.generation === generation &&
        this._audioSourceMatches(existing.snapshot, video)) return existing;
    if (existing) this._disposeAudioCaptureSession(existing);
    if (!this._audioSourceEligible(video)) {
      this._setAudioBlocked("audio-terminal", true);
      this._handlePipelineFailure(
        generation,
        new Error("media is not safely audio-capturable"),
        "audio takeover",
        { terminal: true },
      );
      return null;
    }

    const provider = mediaStreamProvider(video);
    const providerIsStream = !!provider;
    let stream;
    let rawTracksOwned = false;
    let captureRecord = null;
    try {
      if (providerIsStream) {
        stream = provider;
      } else {
        captureRecord = elementCaptureRecords.get(video) || null;
        if (!captureRecord) {
          if (typeof video?.captureStream !== "function") {
            throw new Error("HTMLMediaElement.captureStream is unavailable");
          }
          stream = video.captureStream();
          if (!stream || typeof stream.getTracks !== "function" ||
              typeof stream.getAudioTracks !== "function") {
            throw new Error("audio capture returned an invalid MediaStream");
          }
          captureRecord = {
            video,
            stream,
            owner: null,
            borrowedTracks: new WeakSet(),
            exhausted: false,
            hadAudio: false,
            sourceGeneration: 0,
            trackGenerations: new WeakMap(),
            onTrackChange: null,
            onMediaLifecycle: null,
          };
          captureRecord.onTrackChange = (event) => updatePersistentCaptureRecord(captureRecord, event);
          captureRecord.onMediaLifecycle = (event) => updatePersistentCaptureRecord(captureRecord, event);
          try { stream.addEventListener?.("addtrack", captureRecord.onTrackChange); } catch {}
          try { stream.addEventListener?.("removetrack", captureRecord.onTrackChange); } catch {}
          try { video.addEventListener?.("loadedmetadata", captureRecord.onMediaLifecycle); } catch {}
          try { video.addEventListener?.("ended", captureRecord.onMediaLifecycle); } catch {}
          elementCaptureRecords.set(video, captureRecord);
          updatePersistentCaptureRecord(captureRecord);
        } else {
          stream = captureRecord.stream;
        }
        if (captureRecord.owner?.disposed) captureRecord.owner = null;
        if (captureRecord.owner) {
          throw new Error("media capture is already owned by another interpolation lifecycle");
        }
        rawTracksOwned = true;
      }
      if (!stream || typeof stream.getTracks !== "function" ||
          typeof stream.getAudioTracks !== "function") {
        throw new Error("audio capture returned an invalid MediaStream");
      }
    } catch (error) {
      this._setAudioBlocked("audio-terminal", true);
      this._handlePipelineFailure(generation, error, "audio takeover", { terminal: true });
      return null;
    }

    const session = {
      generation,
      video,
      snapshot,
      stream,
      rawTracksOwned,
      captureRecord,
      interpolator: this,
      audioTracks: [],
      trackListeners: [],
      onTrackChange: null,
      blockReason: `audio-tracks-${generation}-${this._audioAttemptGeneration}`,
      disposed: false,
    };
    session.onTrackChange = (event) => this._handleAudioCaptureChange(session, event);
    if (!captureRecord) {
      try { stream.addEventListener?.("addtrack", session.onTrackChange); } catch {}
      try { stream.addEventListener?.("removetrack", session.onTrackChange); } catch {}
    }
    this._audioCaptureSession = session;
    if (captureRecord) {
      captureRecord.owner = session;
      updatePersistentCaptureRecord(captureRecord);
    }
    this._refreshAudioCaptureTracks(session);
    this._setAudioBlocked(
      session.blockReason,
      session.audioTracks.length > 0 && !this._capturedAudioTracksHealthy(session),
    );
    return session;
  }

  _listenAudioRoute(route, target, type, listener) {
    if (!target?.addEventListener) return;
    target.addEventListener(type, listener);
    route.listeners.push([target, type, listener]);
  }

  _setAudioGain(route, value) {
    const parameter = route?.gain?.gain;
    if (!parameter) return;
    const next = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 1));
    try {
      if (typeof parameter.setValueAtTime === "function") {
        parameter.setValueAtTime(next, route.context.currentTime || 0);
      } else {
        parameter.value = next;
      }
    } catch {
      try { parameter.value = next; } catch {}
    }
  }

  _mediaMuteAccess(video) {
    const NativeMedia = globalThis.HTMLMediaElement;
    let descriptor = null;
    if (NativeMedia?.prototype) {
      try {
        // A page-owned shadow can report a successful write while leaving the
        // native sink untouched. Refuse takeover instead of risking double audio.
        if (Object.prototype.hasOwnProperty.call(video, "muted")) return null;
        descriptor = Object.getOwnPropertyDescriptor(NativeMedia.prototype, "muted") || null;
      } catch { return null; }
    }
    if (descriptor?.get && descriptor?.set) {
      return {
        read: () => !!descriptor.get.call(video),
        write(value) {
          descriptor.set.call(video, !!value);
          return !!descriptor.get.call(video) === !!value;
        },
      };
    }
    if (NativeMedia?.prototype) return null;
    // Plain-object fallback is for deterministic tests/non-DOM embedders only.
    return {
      read: () => !!video.muted,
      write(value) {
        video.muted = !!value;
        return !!video.muted === !!value;
      },
    };
  }

  _readMediaVolume(video) {
    const prototype = globalThis.HTMLMediaElement?.prototype;
    try {
      const descriptor = prototype && Object.getOwnPropertyDescriptor(prototype, "volume");
      const value = descriptor?.get ? Number(descriptor.get.call(video)) : Number(video?.volume);
      return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 1;
    } catch {
      return 1;
    }
  }

  _disposeAudioRoute(route, { restoreMute = true } = {}) {
    if (!route || route.disposed) return;
    route.disposed = true;
    // Close the processed route before restoring native audio so the two sinks can
    // never overlap, even if teardown was caused by context or source failure.
    this._setAudioGain(route, 0);
    if (route.timer != null) clearInterval(route.timer);
    for (const [target, type, listener] of route.listeners || []) {
      try { target.removeEventListener?.(type, listener); } catch {}
    }
    route.listeners = [];
    let stillOwnsMute = false;
    try { stillOwnsMute = route.muteAccess?.read?.() === true; } catch {}
    if (restoreMute && route.muteOwned && stillOwnsMute) {
      let restored = false;
      try { restored = route.muteAccess.write(route.mutedBefore); } catch {}
      if (!restored) {
        this.warn("interpolation: native audio mute state could not be restored");
      }
    }
    for (const source of route.sources || []) {
      try { source.disconnect?.(); } catch {}
    }
    try { route.delay?.disconnect?.(); } catch {}
    try { route.gain?.disconnect?.(); } catch {}
    for (const track of route.ownedTracks || []) {
      try { track.stop?.(); } catch {}
    }
    route.audioTracks = [];
    route.ownedTracks = [];
    route.trackPairs = [];
    route.sources = [];
    if (this._audioRoute === route) this._audioRoute = null;
    if (this._audioTimer === route.timer) this._audioTimer = null;
    if (this._delayNode === route.delay) this._delayNode = null;
    if (this._audioCtx === route.context) {
      this._audioCtx = null;
      this._audioSrc = null;
    }
  }

  _stageAudioDelay(video, generation) {
    const muteAccess = this._mediaMuteAccess(video);
    if (!muteAccess) {
      this._setAudioBlocked("audio-terminal", true);
      this._handlePipelineFailure(
        generation,
        new Error("media mute ownership is shadowed by the page"),
        "audio takeover",
        { terminal: true },
      );
      return null;
    }
    let muted = false;
    let volume = 1;
    try {
      muted = muteAccess.read();
      volume = this._readMediaVolume(video);
    } catch {}
    const explicitlySilent = muted || (Number.isFinite(volume) && volume <= 0);
    if (explicitlySilent) {
      this._audioPreparation?.stagedTransaction?.rollback?.();
      return this._stageSilentAudio(video, generation, "element-silent", null, muteAccess);
    }

    const session = this._ensureAudioCaptureSession(video, generation);
    if (!session) return null;
    if (session.captureRecord?.exhausted && session.captureRecord.hadAudio) {
      this._setAudioBlocked("audio-terminal", true);
      this._handlePipelineFailure(
        generation,
        new Error("the media element's cached audio capture ended; a new source is required"),
        "audio takeover",
        { terminal: true },
      );
      return null;
    }
    if (session.audioTracks.length === 0) {
      this._audioPreparation?.stagedTransaction?.rollback?.();
      return this._stageSilentAudio(video, generation, "no-audio-track", session, muteAccess);
    }
    if (!this._capturedAudioTracksHealthy(session)) {
      this._audioPreparation?.stagedTransaction?.rollback?.();
      this._setAudioBlocked(session.blockReason, true);
      return null;
    }
    this._setAudioBlocked(session.blockReason, false);

    const preparation = this._prepareAudioDelay(video, generation);
    if (!preparation || preparation.status !== "ready") return null;
    const { context } = preparation;
    if (!context || context.state !== "running" || !this._audioSinkMatches(preparation, video)) {
      preparation.status = "idle";
      this._resumeAudioPreparation(preparation);
      return null;
    }

    let transaction = preparation.stagedTransaction;
    if (transaction) {
      const route = transaction.route;
      const valid = transaction.status === "staged" && !route.disposed &&
        route.session === session && this._audioSourceMatches(route.snapshot, video) &&
        this._capturedAudioTracksHealthy(session) && context.state === "running" &&
        this._audioSinkMatches(preparation, video);
      if (!valid) {
        transaction.rollback();
        transaction = null;
      } else if ((context.currentTime || 0) >= route.primeReadyAt) {
        return transaction;
      } else {
        return null;
      }
    }

    const route = {
      generation,
      preparation,
      session,
      context,
      video,
      snapshot: this._audioSourceSnapshot(video),
      audioTracks: [],
      ownedTracks: [],
      trackPairs: [],
      sources: [],
      delay: null,
      gain: null,
      listeners: [],
      timer: null,
      muteAccess,
      muteOwned: false,
      mutedBefore: false,
      disposed: false,
      silent: false,
      primeReadyAt: 0,
    };
    try {
      for (const track of session.audioTracks) {
        if (!session.rawTracksOwned) {
          // A page-owned srcObject track remains the authority for enabled/muted
          // state. Feeding it directly preserves later page-side track toggles;
          // route disposal only disconnects the node and never stops the track.
          route.audioTracks.push(track);
          continue;
        }
        if (typeof track?.clone !== "function") {
          throw new Error("captured audio track cannot be cloned safely");
        }
        const ownedTrack = track.clone();
        if (!ownedTrack || ownedTrack === track) {
          throw new Error("captured audio track clone is not independently owned");
        }
        try { ownedTrack.enabled = track.enabled !== false; } catch {}
        route.ownedTracks.push(ownedTrack);
        route.trackPairs.push({ source: track, output: ownedTrack });
        route.audioTracks.push(ownedTrack);
      }
      if (!this._capturedAudioTracksHealthy(route) || !this._syncOwnedAudioTracks(route)) {
        throw new Error("captured audio became unavailable while staging");
      }
      const initialDelay = Math.max(0, Math.min(
        0.95,
        ((this._videoLatencyMs == null ? 100 : this._videoLatencyMs) + this._avOffsetMs) / 1000,
      ));
      route.delay = context.createDelay(1.0);
      route.delay.delayTime.value = initialDelay;
      route.gain = context.createGain();
      this._setAudioGain(route, 0);
      for (const track of route.audioTracks) {
        let source;
        if (typeof context.createMediaStreamTrackSource === "function") {
          source = context.createMediaStreamTrackSource(track);
        } else {
          const SingleTrackStream = globalThis.MediaStream;
          if (typeof SingleTrackStream !== "function" || typeof context.createMediaStreamSource !== "function") {
            throw new Error("per-track Web Audio capture is unavailable");
          }
          source = context.createMediaStreamSource(new SingleTrackStream([track]));
        }
        route.sources.push(source);
        source.connect(route.delay);
      }
      route.delay.connect(route.gain);
      route.gain.connect(context.destination);
      route.primeReadyAt = (context.currentTime || 0) + initialDelay + 0.025;
    } catch (error) {
      this._disposeAudioRoute(route, { restoreMute: false });
      this._failAudioPreparation(preparation, error);
      return null;
    }

    const owner = this;
    transaction = {
      bypass: false,
      silent: false,
      status: "staged",
      context,
      route,
      rollback() {
        if (this.status !== "staged") return;
        this.status = "rolled-back";
        if (preparation.stagedTransaction === this) preparation.stagedTransaction = null;
        owner._disposeAudioRoute(route, { restoreMute: false });
      },
    };
    preparation.stagedTransaction = transaction;
    return null;
  }

  _stageSilentAudio(video, generation, reason, session, muteAccess) {
    const route = {
      generation,
      preparation: null,
      session,
      context: null,
      video,
      snapshot: this._audioSourceSnapshot(video),
      audioTracks: [],
      ownedTracks: [],
      trackPairs: [],
      sources: [],
      delay: null,
      gain: null,
      listeners: [],
      timer: null,
      muteAccess,
      muteOwned: false,
      mutedBefore: false,
      disposed: false,
      silent: true,
      silentReason: reason,
    };
    const owner = this;
    return {
      bypass: false,
      silent: true,
      status: "staged",
      context: null,
      route,
      rollback() {
        if (this.status !== "staged") return;
        this.status = "rolled-back";
        owner._disposeAudioRoute(route, { restoreMute: false });
      },
    };
  }

  _silentAudioRouteValid(route) {
    if (!route?.silent || !this._audioSourceMatches(route.snapshot, route.video)) return false;
    if (route.silentReason === "no-audio-track") {
      return route.session === this._audioCaptureSession && route.session?.audioTracks.length === 0;
    }
    try {
      return route.muteAccess.read() || this._readMediaVolume(route.video) <= 0;
    } catch {
      return false;
    }
  }

  _commitAudioDelay(transaction, generation) {
    if (!transaction || transaction.status !== "staged") return false;
    if (transaction.bypass) {
      transaction.status = "committed";
      return true;
    }
    const route = transaction.route;
    if (transaction.silent) {
      if (!route || !this._isCurrent(generation) || this.video !== route.video ||
          !this._silentAudioRouteValid(route)) {
        transaction.rollback();
        return false;
      }
      route.onVolumeChange = () => {
        if (this._audioRoute === route && !this._silentAudioRouteValid(route)) {
          this._relinquishPresentation();
        }
      };
      this._listenAudioRoute(route, route.video, "volumechange", route.onVolumeChange);
      this._audioRoute = route;
      transaction.status = "committed";
      return true;
    }
    const preparation = route?.preparation;
    if (!route || !this._isCurrent(generation) || this.video !== preparation?.video ||
        route.context.state !== "running" || !this._audioSourceMatches(route.snapshot, this.video) ||
        route.session !== this._audioCaptureSession || !this._capturedAudioTracksHealthy(route) ||
        !this._capturedAudioTracksHealthy(route.session) ||
        !this._syncOwnedAudioTracks(route) ||
        !this._audioSinkMatches(preparation, this.video) ||
        (route.context.currentTime || 0) < route.primeReadyAt) {
      transaction.rollback();
      return false;
    }

    const relinquish = () => {
      if (this._audioRoute !== route) return;
      this._relinquishPresentation();
    };
    route.onStateChange = () => {
      if (route.context.state === "running" || this._audioRoute !== route) return;
      const state = route.context.state;
      if (state === "closed") {
        this._failAudioPreparation(preparation, new Error("AudioContext closed during takeover"));
        this._relinquishPresentation({ preserveAudioContext: false });
        return;
      }
      this._relinquishPresentation({ preserveAudioContext: true });
      preparation.status = "idle";
      preparation.setupPromise = null;
      this._resumeAudioPreparation(preparation);
    };
    route.onTrackBoundary = (event) => this._handleAudioCaptureChange(route.session, event);
    route.onVolumeChange = () => {
      if (this._audioRoute !== route) return;
      let ownsMute = false;
      try { ownsMute = route.muteOwned && route.muteAccess.read() === true; } catch {}
      if (!ownsMute) {
        this._setAudioGain(route, 0);
        this._relinquishPresentation({ preserveAudioContext: false });
        this._handlePipelineFailure(
          generation,
          new Error("page reclaimed media mute ownership"),
          "audio takeover",
          { terminal: true },
        );
        return;
      }
      this._setAudioGain(route, this._readMediaVolume(route.video));
    };
    this._listenAudioRoute(route, route.context, "statechange", route.onStateChange);
    for (const track of route.audioTracks) {
      this._listenAudioRoute(route, track, "mute", route.onTrackBoundary);
      this._listenAudioRoute(route, track, "ended", route.onTrackBoundary);
    }
    this._listenAudioRoute(route, route.video, "volumechange", route.onVolumeChange);

    this._audioRoute = route;
    this._audioCtx = route.context;
    this._audioSrc = route.sources[0] || null;
    this._delayNode = route.delay;
    try {
      if (!this._audioSourceMatches(route.snapshot, this.video) ||
          !this._capturedAudioTracksHealthy(route) || !this._syncOwnedAudioTracks(route) ||
          route.context.state !== "running" || route.session !== this._audioCaptureSession ||
          !this._capturedAudioTracksHealthy(route.session) ||
          !this._audioSinkMatches(preparation, this.video)) {
        throw new Error("media source changed during audio takeover");
      }
      route.mutedBefore = route.muteAccess.read();
      route.muteOwned = !route.mutedBefore;
      if (!route.muteOwned) {
        throw new Error("media became muted during audio takeover");
      }
      if (!route.muteAccess.write(true)) {
        throw new Error("native audio could not be muted");
      }
      if (this._audioRoute !== route || route.disposed || route.muteAccess.read() !== true) {
        throw new Error("native audio mute ownership was rejected");
      }
      this._setAudioGain(route, this._readMediaVolume(route.video));
    } catch (error) {
      this._disposeAudioRoute(route);
      transaction.status = "rolled-back";
      this._handlePipelineFailure(
        generation,
        error || new Error("native audio takeover failed"),
        "audio takeover",
        { terminal: true },
      );
      return false;
    }

    transaction.status = "committed";
    if (preparation.stagedTransaction === transaction) preparation.stagedTransaction = null;
    route.timer = setInterval(() => {
      if (this._audioRoute !== route) return;
      if (!this._audioSourceMatches(route.snapshot, route.video)) {
        this._relinquishPresentation({ preserveAudioContext: true, retireCapture: true });
        return;
      }
      if (!this._audioSinkMatches(preparation, route.video)) {
        this._relinquishPresentation({ preserveAudioContext: false });
        return;
      }
      if (!this._syncOwnedAudioTracks(route)) {
        this._relinquishPresentation({ preserveAudioContext: true });
        return;
      }
      if (this._videoLatencyMs == null) return;
      const target = Math.max(0, Math.min(0.95, (this._videoLatencyMs + this._avOffsetMs) / 1000));
      if (this._snapAudio) {
        route.delay.delayTime.cancelScheduledValues(route.context.currentTime);
        route.delay.delayTime.setValueAtTime(target, route.context.currentTime);
        this._snapAudio = false;
        return;
      }
      const cur = route.delay.delayTime.value;
      const next = cur * 0.85 + target * 0.15;
      route.delay.delayTime.setTargetAtTime(next, route.context.currentTime, 0.05);
    }, 250);
    this._audioTimer = route.timer;
    return true;
  }

  _teardownAudioDelay({
    preserveContext = false,
    preserveCapture = false,
    retireCapture = false,
  } = {}) {
    const route = this._audioRoute;
    if (route) this._disposeAudioRoute(route);
    const preparation = this._audioPreparation;
    const staged = preparation?.stagedTransaction;
    if (staged?.status === "staged") staged.rollback();
    const shouldRetireCapture = retireCapture || (!preserveContext && !preserveCapture);
    if (shouldRetireCapture) this._disposeAudioCaptureSession();
    if (preserveContext) {
      if (!preparation || (preparation.context?.state !== "closed" && !preparation.contextClosed)) {
        return;
      }
    }
    ++this._audioAttemptGeneration;
    this._audioPreparation = null;
    this._closeAudioContext(preparation);
    this._audioTimer = null;
    this._delayNode = null;
    this._audioSrc = null;
    this._audioCtx = null;
  }

  supported() {
    return (
      typeof createImageBitmap !== "undefined" &&
      typeof OffscreenCanvas !== "undefined"
    );
  }

  _isCurrent(generation) {
    return generation === this._lifecycleGen && !this._stopped;
  }

  _recordPipelineSuccess(stage) {
    if (this._pipelineFailureStreaks) this._pipelineFailureStreaks[stage] = 0;
  }

  _notifyTerminalFailure(generation, stage, error) {
    if (!this._isCurrent(generation) || !this.onTerminalFailure) return;
    try {
      this.onTerminalFailure({
        generation,
        stage,
        error,
        detail: error?.message || String(error || "unknown failure"),
        video: this.video,
        source: this._audioSourceSnapshot(this.video),
      });
    } catch {}
  }

  _handlePipelineFailure(generation, error, stage = "capture", { terminal = false } = {}) {
    if (!this._isCurrent(generation)) return false;
    const streaks = this._pipelineFailureStreaks || (this._pipelineFailureStreaks = Object.create(null));
    const count = (streaks[stage] || 0) + 1;
    streaks[stage] = count;
    if (!terminal && count < this._pipelineFailureLimit) return false;
    if (this._pipelineFailureStopQueued) return true;
    this._pipelineFailureStopQueued = true;
    const detail = error?.message || String(error || "unknown failure");
    // Restore the native presentation synchronously. The terminal callback can
    // immediately expose diagnostics to the popup, while the full asynchronous
    // stop below retires the remaining capture and GPU resources.
    try {
      this._relinquishPresentation({ preserveAudioContext: true });
    } catch (cleanupError) {
      this._takeoverActive = false;
      this._committedPresentation = null;
      try { this.overlay?.remove?.(); } catch {}
      this.warn("interp: immediate presentation cleanup failed:", cleanupError.message);
    }
    this._notifyTerminalFailure(generation, stage, error);
    this.warn(`interp: ${stage} failed ${terminal ? "terminally" : `${count} consecutive times`}; restoring original video (${detail})`);
    Promise.resolve().then(() => {
      if (this._isCurrent(generation)) this.stop();
    }).catch((stopError) => {
      this.warn("interp: failure cleanup failed:", stopError.message);
    }).finally(() => {
      this._pipelineFailureStopQueued = false;
    });
    return true;
  }

  // DOM visibility and audio routing are a transaction committed only after an
  // output frame has actually rendered. Until then the original element remains
  // the page's display/audio source, so unsupported or tainted capture paths fail
  // as passthrough instead of exposing a black canvas or silent audio graph.
  _videoWithin(root) {
    if (!root || !this.video) return false;
    let current = this.video;
    while (current) {
      if (current === root) return true;
      if (current.parentElement) current = current.parentElement;
      else {
        const tree = current.getRootNode?.();
        current = tree?.host || null;
      }
    }
    return false;
  }

  _overlayMountTarget() {
    if (videoPresentationState(this.video, globalThis.document).nativeRequired) return null;
    const fullscreen = globalThis.document?.fullscreenElement || null;
    if (!fullscreen) return globalThis.document?.body || null;
    const sourceRoot = this.video?.getRootNode?.();
    const innerFullscreen = sourceRoot?.fullscreenElement || null;
    if (innerFullscreen) {
      if (innerFullscreen === this.video || !this._videoWithin(innerFullscreen)) return null;
      return innerFullscreen;
    }
    // A canvas cannot be rendered as a child of a fullscreen <video> replaced
    // element. Keep the original source visible for that interval. Player-
    // container fullscreen is eligible because descendants join the top layer.
    if (fullscreen === this.video || !this._videoWithin(fullscreen)) return null;
    // If the source lives in a shadow tree, light-DOM children appended to its
    // fullscreen host may be unslotted and therefore never rendered. Mount next
    // to the video in its actual shadow root; that root is within the eligible
    // fullscreen subtree and the fixed overlay remains visible in the top layer.
    if (sourceRoot?.host && typeof sourceRoot.appendChild === "function") return sourceRoot;
    return fullscreen;
  }

  _sourceCanPresent() {
    const source = this.video;
    if (!source) return false;
    try {
      if (typeof source.checkVisibility === "function" &&
          !source.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false;
    } catch {}
    try {
      const style = globalThis.getComputedStyle?.(source);
      if (style && (style.display === "none" || style.visibility === "hidden" ||
          style.visibility === "collapse" || Number(style.opacity) === 0)) return false;
    } catch {}
    return true;
  }

  _removeTakeoverListeners() {
    if (this._onScroll) {
      try { window.removeEventListener("scroll", this._onScroll, { capture: true }); } catch {}
      try { window.removeEventListener("resize", this._onScroll); } catch {}
    }
    if (this._onPresentationBoundary) {
      try { document.removeEventListener("fullscreenchange", this._onPresentationBoundary); } catch {}
      try { this.video?.removeEventListener?.("enterpictureinpicture", this._onPresentationBoundary); }
      catch {}
      try { this.video?.removeEventListener?.("leavepictureinpicture", this._onPresentationBoundary); }
      catch {}
    }
    this._onScroll = null;
    this._onPresentationBoundary = null;
  }

  _installMediaBoundaryListeners(video, generation) {
    if (!video || (this._mediaBoundaryVideo === video && this._mediaBoundaryHandlers)) return;
    this._removeMediaBoundaryListeners();
    const current = () => this._isCurrent(generation) && this.video === video;
    const suspendAudio = ({ retireCapture = false } = {}) => {
      if (!current()) return;
      this._setAudioBlocked("media-boundary", true);
      this._flush?.();
      if (this._takeoverActive || this._audioRoute || this._audioPreparation?.stagedTransaction ||
          (this._chainInverted && !this._chainPresentationSuspended) || retireCapture) {
        this._relinquishPresentation({ preserveAudioContext: true, retireCapture });
      }
    };
    const resumeAudio = () => {
      if (!current()) return;
      this._setAudioBlocked("media-boundary", false);
      if (this._audioCaptureSession) this._handleAudioCaptureChange(this._audioCaptureSession);
      this._flush?.();
    };
    const ordinarySuspend = () => suspendAudio();
    const sourceSuspend = () => suspendAudio({ retireCapture: true });
    const handlers = new Map([
      ["seeking", ordinarySuspend],
      ["seeked", resumeAudio],
      ["loadstart", sourceSuspend],
      ["emptied", sourceSuspend],
      ["loadedmetadata", resumeAudio],
      ["pause", ordinarySuspend],
      ["ended", ordinarySuspend],
      ["waiting", ordinarySuspend],
      ["play", resumeAudio],
      ["playing", resumeAudio],
      ["canplay", resumeAudio],
      ["enterpictureinpicture", ordinarySuspend],
      ["leavepictureinpicture", resumeAudio],
    ]);
    this._onSeeking = ordinarySuspend;
    this._onPlay = resumeAudio;
    this._mediaBoundaryVideo = video;
    this._mediaBoundaryHandlers = handlers;
    for (const [type, listener] of handlers) video.addEventListener(type, listener);
  }

  _removeMediaBoundaryListeners() {
    const video = this._mediaBoundaryVideo;
    if (video && this._mediaBoundaryHandlers) {
      for (const [type, listener] of this._mediaBoundaryHandlers) {
        try { video.removeEventListener(type, listener); } catch {}
      }
    }
    this._mediaBoundaryVideo = null;
    this._mediaBoundaryHandlers = null;
    this._setAudioBlocked("media-boundary", false);
    this._onSeeking = null;
    this._onPlay = null;
  }

  _relinquishPresentation({ preserveAudioContext = true, retireCapture = false } = {}) {
    this._removeTakeoverListeners();
    this._teardownAudioDelay({
      preserveContext: preserveAudioContext,
      preserveCapture: !retireCapture,
      retireCapture,
    });
    // In inverted mode the normal renderer deliberately pauses while the
    // interpolator drives its presentation surface. Temporarily hand that surface
    // back whenever interpolation cannot present, otherwise fullscreen, PiP,
    // buffering, or audio recovery would leave the last frame frozen indefinitely.
    if (this._chainInverted && !this._chainPresentationSuspended) {
      try { this.chain?.setInverted?.(false); } catch {}
      this._chainPresentationSuspended = true;
    }
    try { this.overlay?.remove?.(); } catch {}
    this._takeoverActive = false;
    this._committedPresentation = null;
  }

  _productionEligible() {
    if (this._audioBlocks.size > 0) {
      if (this._takeoverActive || this._audioRoute || this._audioPreparation?.stagedTransaction ||
          (this._chainInverted && !this._chainPresentationSuspended)) {
        this._relinquishPresentation();
      }
      return false;
    }
    const presentable = this._sourceCanPresent() && !!this._overlayMountTarget();
    if (!presentable && (this._takeoverActive || this._audioRoute ||
        this._audioPreparation?.stagedTransaction ||
        (this._chainInverted && !this._chainPresentationSuspended))) {
      this._relinquishPresentation();
    }
    return presentable;
  }

  _stageTakeover(generation) {
    if (!this._isCurrent(generation) || !this.video || !this.overlay) return null;
    if (this._audioBoundaryPending) return null;
    if (!this._sourceCanPresent()) {
      this._relinquishPresentation();
      return null;
    }
    const mount = this._overlayMountTarget();
    if (!mount) {
      this._relinquishPresentation();
      return null;
    }
    if (this._takeoverActive) {
      if (this._audioRoute) {
        if (this._audioRoute.silent) {
          if (!this._silentAudioRouteValid(this._audioRoute)) {
            this._relinquishPresentation();
            return null;
          }
        } else {
          const route = this._audioRoute;
          let ownsMute = false;
          try { ownsMute = route.muteOwned && route.muteAccess.read() === true; } catch {}
          if (!this._audioSourceMatches(route.snapshot, this.video)) {
            this._relinquishPresentation({ preserveAudioContext: true, retireCapture: true });
            return null;
          }
          const sinkMatches = this._audioSinkMatches(route.preparation, this.video);
          if (!sinkMatches) {
            this._relinquishPresentation({ preserveAudioContext: false });
            return null;
          }
          if (route.context.state !== "running") {
            if (route.context.state === "closed") {
              this._failAudioPreparation(route.preparation, new Error("AudioContext closed during takeover"));
            }
            this._relinquishPresentation({ preserveAudioContext: route.context.state !== "closed" });
            return null;
          }
          if (route.session !== this._audioCaptureSession ||
              !this._capturedAudioTracksHealthy(route.session) ||
              !this._capturedAudioTracksHealthy(route) || !this._syncOwnedAudioTracks(route)) {
            if (route.session === this._audioCaptureSession) {
              this._handleAudioCaptureChange(route.session);
            } else {
              this._relinquishPresentation({ preserveAudioContext: true });
            }
            return null;
          }
          if (!ownsMute) {
            this._setAudioGain(route, 0);
            this._relinquishPresentation({ preserveAudioContext: false });
            this._handlePipelineFailure(
              generation,
              new Error("page reclaimed media mute ownership"),
              "audio takeover",
              { terminal: true },
            );
            return null;
          }
        }
      }
      return { mount, audioTransaction: null };
    }
    const audioTransaction = this._stageAudioDelay(this.video, generation);
    if (!audioTransaction) return null;
    if (!audioTransaction.bypass && !audioTransaction.silent &&
        audioTransaction.context.state !== "running") {
      audioTransaction.rollback();
      return null;
    }
    if (this._chainInverted && this._chainPresentationSuspended) {
      try {
        if (this.chain?.setInverted?.(true) === false) {
          audioTransaction.rollback?.();
          return null;
        }
        this._chainPresentationSuspended = false;
      } catch {
        audioTransaction.rollback?.();
        return null;
      }
    }
    return { mount, audioTransaction };
  }

  _activateTakeover(generation, staged = null) {
    let takeover = staged;
    if (!takeover) {
      try { takeover = this._stageTakeover(generation); }
      catch (error) {
        this._handlePipelineFailure(generation, error, "display takeover", { terminal: true });
        return false;
      }
    }
    const audioTransaction = takeover?.audioTransaction;
    if (!takeover || !this._isCurrent(generation) || !this.video || !this.overlay) {
      audioTransaction?.rollback?.();
      return false;
    }
    try {
      const mount = takeover.mount;
      if (!this._chainInverted) {
        if (!this.overlay.isConnected || this.overlay.parentNode !== mount) mount.appendChild(this.overlay);
        if (!this.overlay.isConnected) throw new Error("interpolation overlay did not connect to its mount");
      }
      this.position();
      if (!this._takeoverActive) {
        // The source remains visible beneath the opaque output canvas. If the
        // overlay is removed, becomes fullscreen-ineligible, or fails to mount,
        // presentation therefore degrades to the site's original video rather
        // than a black surface. Only audio is transactionally rerouted.
        if (!this._commitAudioDelay(audioTransaction, generation)) {
          const preserveAudioContext = audioTransaction?.context?.state === "running" &&
            this._audioSinkMatches(audioTransaction?.route?.preparation, this.video);
          this._relinquishPresentation({ preserveAudioContext });
          return false;
        }
        this._onScroll = () => this.position();
        this._onPresentationBoundary = () => {
          if (!this._isCurrent(generation)) return;
          const mount = this._overlayMountTarget();
          if (!mount) {
            this._relinquishPresentation();
            return;
          }
          if (this._chainInverted) return;
          try {
            if (!this.overlay?.isConnected || this.overlay.parentNode !== mount) mount.appendChild(this.overlay);
            this.position();
          } catch { this._relinquishPresentation(); }
        };
        window.addEventListener("scroll", this._onScroll, { passive: true, capture: true });
        window.addEventListener("resize", this._onScroll, { passive: true });
        document.addEventListener("fullscreenchange", this._onPresentationBoundary);
        this.video.addEventListener?.("enterpictureinpicture", this._onPresentationBoundary);
        this.video.addEventListener?.("leavepictureinpicture", this._onPresentationBoundary);
        this._takeoverActive = true;
      }
      return true;
    } catch (error) {
      audioTransaction?.rollback?.();
      this._removeTakeoverListeners();
      this._teardownAudioDelay({ retireCapture: true });
      this._takeoverActive = false;
      this._committedPresentation = null;
      try { this.overlay.remove(); } catch {}
      this._handlePipelineFailure(generation, error, "display takeover", { terminal: true });
      return false;
    }
  }

  _recordCommittedPresentation(candidate) {
    // Called only after the output sink submitted successfully and the takeover
    // transaction committed. Keep this distinct from queued/generated frames.
    const source = interpolationDimensions(candidate?.width, candidate?.height) ||
      interpolationDimensions(this.video?.videoWidth, this.video?.videoHeight);
    let output = null;
    const sink = this._chainInverted ? "renderer" : "overlay";
    if (this._chainInverted) {
      try {
        const dimensions = this.chain?.targetDims?.();
        output = interpolationDimensions(dimensions?.w, dimensions?.h);
      } catch {}
    } else {
      output = interpolationDimensions(this.overlay?.width, this.overlay?.height);
    }
    const generation = ++this._presentationGeneration;
    this.stats.framesPresented = generation;
    this._committedPresentation = Object.freeze({
      committed: true,
      generation,
      gpu: candidate?.gpu === true,
      sink,
      source,
      output,
      framesIn: this.stats.framesIn || 0,
      framesOut: this.stats.framesOut || 0,
    });
    return this._committedPresentation;
  }

  refreshLayout() {
    if (!this._takeoverActive) return false;
    return this._activateTakeover(this._lifecycleGen);
  }

  _scheduleDimsRestart(generation, width, height) {
    if (this._dimsRestarting) return false;
    this._dimsRestarting = true;
    this.log(`interp: source ${width}x${height} under inverted chain — clean restart`);
    setTimeout(async () => {
      try {
        // Keep the guard inside try so even a stale queued callback reaches finally.
        if (!this._isCurrent(generation)) return;
        this.stop();
        await this.start();
      } catch (e) {
        this.warn("dims-change restart failed:", e.message);
      } finally {
        this._dimsRestarting = false;
      }
    }, 0);
    return true;
  }

  _handleGpuCaptureFailure(generation) {
    const error = this._rifeMod?.gpuLastCaptureError?.();
    if (error?.code !== "GPU_RESOURCE_LIMIT") return false;
    // A prior generation may still be behind a queue fence, or checked-out
    // presentation textures may be released shortly. Drop this source tick and
    // retry without advancing curTex; only a request that cannot ever fit is
    // terminal for the current interpolation lifecycle.
    if (error.details?.transient) return true;
    if (this._gpuResourceStopQueued) return true;
    this._gpuResourceStopQueued = true;
    this._notifyTerminalFailure(generation, "GPU resource limit", error);
    this.warn(`interp: GPU resource limit reached; restoring original video (${error.message})`);
    Promise.resolve().then(() => {
      if (this._isCurrent(generation)) this.stop();
    }).finally(() => {
      this._gpuResourceStopQueued = false;
    });
    return true;
  }

  _handleRifeDeviceLoss(lostDevice, info) {
    if (!this.running) return false;
    if (this._deviceRestarting) {
      // RIFE identity-guards duplicate notifications for one device.  A different
      // device here is the replacement created by the in-flight restart; dropping
      // it would leave that new dead generation published with no later recovery.
      if (lostDevice === this._deviceRecoveryDevice ||
          lostDevice === this._pendingDeviceLoss?.lostDevice) return false;
      this._pendingDeviceLoss = { lostDevice, info, generation: this._lifecycleGen };
      return true;
    }
    const generation = this._lifecycleGen;
    this._deviceRestarting = true;
    this._deviceRecoveryDevice = lostDevice;
    const detail = info?.message || info?.reason || "unknown reason";
    this.warn(`interp: GPU device lost (${detail}); rebuilding interpolation`);
    Promise.resolve().then(async () => {
      // A user stop/model change after the loss owns the newer lifecycle and must
      // not be undone by this queued automatic restart.
      if (!this._isCurrent(generation)) return;
      this.stop({ preservePendingDeviceLoss: true });
      const result = await this.start();
      if (!result?.ok && this.running) {
        this.warn(`interp: device-loss restart failed: ${result?.reason || "unknown"}`);
      }
    }).catch((error) => {
      this.warn("interp: device-loss restart failed:", error.message);
    }).finally(() => {
      const pending = this._pendingDeviceLoss;
      this._pendingDeviceLoss = null;
      this._deviceRestarting = false;
      this._deviceRecoveryDevice = null;
      if (pending && this.running && this._isCurrent(pending.generation)) {
        this._handleRifeDeviceLoss(pending.lostDevice, pending.info);
      }
    });
    return true;
  }

  _commitCpuTweenBitmap(generation, cur, tweenBitmap, timestamp, stats, flushGeneration = null) {
    if (!this._isCurrent(generation) ||
        (flushGeneration != null && (this._flushGen || 0) !== flushGeneration)) {
      // The tween bitmap and both copies of the current frame still belong to the
      // stale async continuation. Close all three before the caller can enqueue a
      // frame or replace its prevFrame lookahead reference.
      for (const bitmap of [tweenBitmap, cur?.bmp, cur?.prevBmp]) {
        try { bitmap?.close?.(); } catch {}
      }
      return false;
    }
    if (tweenBitmap) {
      this._enqueue(tweenBitmap, timestamp);
      stats.framesOut++;
    }
    return true;
  }

  // Idempotent public lifecycle entry point. Concurrent callers share one start;
  // a start requested after stop waits for the cancelled start to unwind before
  // creating another pipeline, preventing two module initializers from racing.
  start(sourceVideo) {
    // Explicit retirement is a barrier, not a terminal state. A later enable may
    // reuse this Interpolator, but it must not publish a new capture lifecycle
    // while the previous model/session and its source-sized scratch state are
    // still being released.
    const retirement = this._modelRetirementPromise || this._resourceRetirementPromise;
    if (retirement) {
      const hasSource = arguments.length > 0;
      return retirement.catch(() => {}).then(() =>
        hasSource ? this.start(sourceVideo) : this.start());
    }
    if (arguments.length > 0) {
      let requestedVideo = sourceVideo;
      try { if (typeof sourceVideo === "function") requestedVideo = sourceVideo(); }
      catch (error) {
        this.warn("interpolation: source accessor failed:", error.message);
        return Promise.resolve({ ok: false, reason: "source-failed" });
      }
      // A caller must stop before transferring a live instance to another video.
      // Refusing the ambiguous request keeps the active lifecycle and its source
      // identity aligned; the requested source is not persisted on failure.
      if (this._state !== "idle" && this.video && requestedVideo !== this.video) {
        return Promise.resolve({ ok: false, reason: "source-active" });
      }
      this._sourceProvided = true;
      this._sourceVideo = sourceVideo;
    }
    if (this._state === "running" && this.running) return Promise.resolve({ ok: true });
    if (this._state === "starting" && this._startPromise) return this._startPromise;
    if (this._startPromise) {
      const pending = this._startPromise;
      return pending.catch(() => null).then(() => this.start());
    }

    const generation = ++this._lifecycleGen;
    this._stopped = false;
    this._state = "starting";
    // Treat initialization as active for callers deciding whether an engine change
    // requires stop/restart. Concurrent start() calls still share _startPromise.
    this.running = true;
    const beginStart = () => {
      if (!this._isCurrent(generation)) return { ok: false, reason: "cancelled" };
      return this._startInternal(generation);
    };
    // stop() deliberately returns before physical GPU teardown completes so DOM
    // and playback state are restored synchronously.  Keep this start published as
    // the single flight, but do not request any replacement devices until every
    // retired WebGPUGrabber has actually finished destroying its device.
    const pendingGrabberTeardown = this._cpuGrabberTeardowns.size
      ? this._waitForCpuGrabberTeardown().then(beginStart)
      : beginStart();
    const promise = Promise.resolve(pendingGrabberTeardown)
      .then((result) => {
        if (this._isCurrent(generation)) {
          if (result.ok) {
            this.running = true;
            this._state = "running";
          } else {
            this.stop();
          }
        }
        return result;
      })
      .catch((error) => {
        if (this._isCurrent(generation)) this.stop();
        this.warn("interpolation start failed:", error.message);
        return { ok: false, reason: error.message || "start-failed" };
      })
      .finally(() => {
        if (this._startPromise === promise) this._startPromise = null;
      });
    this._startPromise = promise;
    return promise;
  }

  _destroyCpuGrabber(grabber) {
    if (!grabber || (typeof grabber !== "object" && typeof grabber !== "function")) {
      return Promise.resolve();
    }
    const existing = this._cpuGrabberTeardowns.get(grabber);
    if (existing) return existing;

    let destruction;
    try { destruction = grabber.destroy?.(); }
    catch { destruction = null; }
    let tracked;
    tracked = Promise.resolve(destruction)
      // Teardown is best-effort at this layer, matching the previous lifecycle
      // contract.  Its completion, rather than its success, is the restart gate.
      .catch(() => {})
      .finally(() => {
        if (this._cpuGrabberTeardowns.get(grabber) === tracked) {
          this._cpuGrabberTeardowns.delete(grabber);
        }
      });
    this._cpuGrabberTeardowns.set(grabber, tracked);
    return tracked;
  }

  async _waitForCpuGrabberTeardown() {
    // A stale initializer can discover that it lost ownership while an earlier
    // destroy is draining and add its candidate to the map.  Re-snapshot until
    // stable so callers cannot slip between two physical teardown generations.
    while (this._cpuGrabberTeardowns.size) {
      await Promise.all([...this._cpuGrabberTeardowns.values()]);
    }
  }

  _ensureCpuGrabber(generation) {
    if (!this._isCurrent(generation)) return Promise.resolve(false);
    if (this._gpuGrab?.ready) return Promise.resolve(true);
    if (this._gpuGrab) {
      const unavailable = this._gpuGrab;
      this._gpuGrab = null;
      this._destroyCpuGrabber(unavailable);
    }
    if (this._cpuGrabInit?.generation === generation) return this._cpuGrabInit.promise;

    const attempt = { generation, grabber: null, promise: null };
    const promise = (async () => {
      let grabber = null;
      try {
        await this._waitForCpuGrabberTeardown();
        if (!this._isCurrent(generation)) return false;
        const gm = await import(chrome.runtime.getURL("src/core/fsrcnnx-grab.js"));
        if (!this._isCurrent(generation)) return false;
        grabber = new gm.WebGPUGrabber({
          log: this.log,
          warn: this.warn,
          onDeviceLost: (_lostDevice, info) =>
            this._handleCpuGrabberDeviceLoss(grabber, generation, info),
        });
        attempt.grabber = grabber;
        const ok = await grabber.init();
        // Device.lost may settle between init() publishing its device and this
        // continuation. Never publish that already-invalid grabber as ready.
        if (!this._isCurrent(generation) || !ok || !grabber.ready) {
          await this._destroyCpuGrabber(grabber);
          if (this._isCurrent(generation) && !ok) {
            this.warn("interp: WebGPU grab unavailable, using 2D fallback (may show waves)");
          }
          return false;
        }
        this._gpuGrab = grabber;
        this.log("interp: WebGPU clean CPU-path grab active");
        return true;
      } catch (e) {
        if (grabber) await this._destroyCpuGrabber(grabber);
        if (this._isCurrent(generation)) this.warn("interp: grab module load failed:", e.message);
        return false;
      }
    })().finally(() => {
      if (this._cpuGrabInit === attempt) this._cpuGrabInit = null;
    });
    attempt.promise = promise;
    this._cpuGrabInit = attempt;
    return promise;
  }

  _handleCpuGrabberDeviceLoss(grabber, generation, info) {
    const pending = this._cpuGrabInit;
    const ownsPublished = this._gpuGrab === grabber;
    const ownsPending = pending?.generation === generation && pending.grabber === grabber;
    if ((!ownsPublished && !ownsPending) || !this._isCurrent(generation)) return false;
    if (ownsPublished) this._gpuGrab = null;
    // WebGPUGrabber also self-destroys after invoking this callback. Calling its
    // idempotent destroy here lets the coordinator observe and gate on that same
    // physical teardown promise before recovery requests a replacement device.
    this._destroyCpuGrabber(grabber);
    const detail = info?.message || info?.reason || "unknown reason";
    this.warn(`interp: CPU grab device lost (${detail}); rebuilding clean grab path`);
    if (!this._gpuPresent) void this._scheduleCpuGrabberRecovery(generation);
    return true;
  }

  _scheduleCpuGrabberRecovery(generation) {
    if (!this._isCurrent(generation) || this._gpuPresent) return Promise.resolve(false);
    if (this._cpuGrabRecovery?.generation === generation) return this._cpuGrabRecovery.promise;
    const recovery = { generation, promise: null };
    const promise = (async () => {
      for (const delay of this._cpuGrabRecoveryDelays) {
        if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
        if (!this._isCurrent(generation) || this._gpuPresent) return false;
        if (await this._ensureCpuGrabber(generation)) return true;
      }
      if (this._isCurrent(generation) && !this._gpuPresent) {
        this.warn("interp: clean GPU grab recovery exhausted; continuing with 2D fallback");
      }
      return false;
    })().finally(() => {
      if (this._cpuGrabRecovery === recovery) this._cpuGrabRecovery = null;
    });
    recovery.promise = promise;
    this._cpuGrabRecovery = recovery;
    return promise;
  }

  async _startInternal(generation) {
    if (!this.supported()) {
      this.warn("interpolation: ImageBitmap/OffscreenCanvas not available in this browser");
      return { ok: false, reason: "unsupported" };
    }
    const video = this._resolveSourceVideo();
    if (!video) return { ok: false, reason: "no video" };
    this.video = video;
    this._takeoverActive = false;
    this._committedPresentation = null;
    this._presentationGeneration = 0;
    this._chainPresentationSuspended = false;
    this._clearAudioBlocks();
    this._productionWasEligible = true;
    this._pipelineFailureStopQueued = false;
    this._pipelineFailureStreaks = Object.create(null);
    this._interpMode = this._forceBlend ? "blend" : "rife";
    this._fallbackArmed = true; this._srcFrameBase = null;
    this._tweenFailStreak = 0; this._prevTs = null;
    this._lastVW = null; this._lastVH = null;
    this._tapStaleSince = 0; this._tapStaleLogAt = 0; this._invUpWarned = false;

    // Frame source: grab from the <video> ELEMENT directly via drawImage, driven
    // by requestVideoFrameCallback — the SAME kind of clean video-element read the
    // upscaler uses (importExternalTexture), instead of captureStream→VideoFrame→
    // createImageBitmap, which introduced chroma-reconstruction ringing (the
    // "waves") on bright/high-detail regions. The browser's video-element→canvas
    // conversion does proper YUV→RGB, matching the artifact-free upscaler path.
    if (typeof video.requestVideoFrameCallback !== "function") {
      this.warn("interpolation: requestVideoFrameCallback unavailable");
      return { ok: false, reason: "no-rvfc" };
    }
    // scratch canvas to turn clean RGBA readback into bitmaps for the pipeline
    this._grab = new OffscreenCanvas(video.videoWidth || 2, video.videoHeight || 2);
    this._grabCtx = this._grab.getContext("2d", {
      colorSpace: SRGB_COLOR_SPACE,
      willReadFrequently: false,
    });
    const grabCanvas = this._grab;
    const grabCtx = this._grabCtx;
    let cpuTickBusy = false;
    // The readback grabber is initialized only if GPU-resident presentation cannot
    // be established. Avoiding it on the normal path prevents a redundant adapter,
    // device, and staging allocation at every interpolation start.
    const retainedGrabber = this._gpuGrab;
    this._gpuGrab = null;
    if (retainedGrabber) {
      await this._destroyCpuGrabber(retainedGrabber);
      if (!this._isCurrent(generation)) return { ok: false, reason: "cancelled" };
    }
    this.stats = { framesIn: 0, framesOut: 0, started: performance.now(), lastReport: 0, maxDriftMs: 0, maxGapMs: 0, lastGapMs: 0, stutters: 0 };
    const stats = this.stats;
    const log = this.log;
    let lastFrameWall = 0;
    // CPU fallback keeps one lookahead frame. The GPU path retains explicit pooled
    // frame pairs so capture can continue while inference is in flight.
    let prevFrame = null;   // { bmp: ImageBitmap, ts: microseconds }
    this._closeCpuPrev = () => {
      try { prevFrame?.bmp?.close?.(); } catch {}
      prevFrame = null;
    };
    let rife = null;
    // Load the interp module. If the user picked the Blend engine, bring up the
    // STANDALONE blend GPU path (own device, NO RIFE/ONNX model load). Otherwise
    // load RIFE (ORT + model) as usual; if that fails, tweens fall back to blend.
    const pipelinePromise = import(chrome.runtime.getURL("src/core/fsrcnnx-rife.js")).then(async (m) => {
      if (!this._isCurrent(generation)) return false;
      rife = m;
      this._rifeMod = m; // expose for getStats (timing breakdown)
      if (!this._deviceLossUnsubscribe && m.addDeviceLossListener) {
        this._deviceLossUnsubscribe = m.addDeviceLossListener((lostDevice, info) =>
          this._handleRifeDeviceLoss(lostDevice, info));
      }
      if (this._rifeModelKey && m.setModel && !m.setModel(this._rifeModelKey)) {
        this.warn(`interp: unknown RIFE model '${this._rifeModelKey}', using module default`);
        this._rifeModelKey = null;
      }
      // CHAIN RULES: blend can join the upscaler's device directly. RIFE owns the
      // ORT device, so the upscaler adopts that device before textures are shared.
      const chainDev = (this._chainOwnsVideo(video) && this.chain.available && this.chain.available())
        ? (this.chain.device ? this.chain.device() : null) : null;
      if (this._forceBlend) {
        // standalone blend (no model download, no ORT session), chained if upscaling
        stats.rife = true; stats.rifeError = null; this._interpMode = "blend";
        const gpuOk = await m.initGpuBlendStandalone({ log: this.log, warn: this.warn, device: chainDev || undefined });
        if (!this._isCurrent(generation)) {
          try { await m.destroyGpuInterp?.(); } catch {}
          return false;
        }
        this._gpuInterpActive = gpuOk; stats.gpuPath = gpuOk;
        this._chainActive = !!(gpuOk && chainDev);
        if (this._chainActive && this.chain.tap) { this.chain.tap(true); log("interp: CHAIN active — interpolating UPSCALED frames (blend engine)"); }
        if (gpuOk && this.overlay && !this._octx && m.gpuConfigureCanvas) {
          try { this._gpuPresent = m.gpuConfigureCanvas(this.overlay); } catch { this._gpuPresent = false; }
          if (!this._gpuPresent) {
            this._octx = this.overlay.getContext("2d", { colorSpace: SRGB_COLOR_SPACE });
          }
          log(`interp: standalone BLEND ${this._gpuPresent ? "ACTIVE (no readback)" : "present unavailable"}`);
        }
        if (gpuOk && !this._gpuPresent) {
          if (this._chainActive && this.chain?.tap) { try { this.chain.tap(false); } catch {} }
          this._chainActive = false;
          await m.destroyGpuInterp?.();
          this._gpuInterpActive = false; stats.gpuPath = false;
        }
        if (!this._gpuPresent) await this._ensureCpuGrabber(generation);
        if (!this._isCurrent(generation)) return false;
        this._pipelineReady = true;
        return true;
      }
      // Pre-compute padded dimensions for the optional shape-specific session path.
      // Dynamic sessions ignore them and are reused across restarts.
      let pinW = 0, pinH = 0;
      try {
        const td = (this.chain && this.chain.available && this.chain.available() && this.chain.targetDims)
          ? this.chain.targetDims() : null;
        const bw = td ? td.w : video.videoWidth, bh = td ? td.h : video.videoHeight;
        if (bw && bh) { pinW = Math.ceil(bw / 8) * 8; pinH = Math.ceil(bh / 8) * 8; }
      } catch {}
      const ok = await m.initRife(pinW, pinH);
      if (!this._isCurrent(generation)) return false;
      log(`interp: RIFE ${ok ? "ready (WebGPU)" : "unavailable — blend fallback"}`);
      stats.rife = ok;
      stats.rifeError = ok ? null : (m.getLastError ? m.getLastError() : "unknown");
      if (ok) {
        try {
          const gpuOk = await m.initGpuInterp({ log: this.log, warn: this.warn });
          if (!this._isCurrent(generation)) {
            try { await m.destroyGpuInterp?.(); } catch {}
            return false;
          }
          this._gpuInterpActive = gpuOk;
          stats.gpuPath = gpuOk;
          // If the upscaler is active, unify devices: it rebuilds on ORT's device
          // so its tap texture is consumable by RIFE. On adoption failure, run
          // unchained on the raw video.
          if (gpuOk && chainDev && this.chain && this.chain.adopt) {
            const ortDev = m.getOrtDevice ? m.getOrtDevice() : null;
            let adopted = false;
            if (ortDev) {
              try { adopted = await this.chain.adopt(ortDev, () => this._isCurrent(generation)); }
              catch (e) { this.warn("interp: device adopt failed:", e.message); }
            }
            if (!this._isCurrent(generation)) return false;
            if (adopted && m.confirmOrtDeviceAdopted) {
              try { await m.confirmOrtDeviceAdopted(ortDev); }
              catch (e) { this.warn("interp: old ORT device guard release failed:", e.message); }
              if (!this._isCurrent(generation)) return false;
            }
            // Inversion is an explicit capability, not merely a saved preference.
            // The renderer must affirm that it owns this exact video and has an
            // active upscale path; otherwise hiding both surfaces can black out a
            // passthrough/standalone run when upscaleTex inevitably rejects.
            const wantInvert = adopted && this._chainCanInvert(video)
              && this.chain.invert && this.chain.invert()
              && this.chain.setInverted && this.chain.upscaleTex;
            if (wantInvert) {
              // #4 INVERTED CHAIN: RIFE runs on RAW video frames (source res —
              // far fewer pixels per inference); the upscaler runs once per
              // PRESENTED frame via chain.upscaleTex. The tap stays OFF, so the
              // capture path falls to raw gpuCapture(video) automatically. The
              // adopt above still matters: the upscaler must live on ORT's
              // device to consume our pooled textures.
              this._chainActive = false;
              this._chainInverted = true;
              this.chain.setInverted(true);
              if (this.overlay) this.overlay.style.display = "none"; // upscaler canvas is the display surface
              log("interp: INVERTED CHAIN active — RIFE on raw video, upscale per presented frame");
            } else {
              this._chainActive = !!adopted;
              if (adopted && this.chain.tap) { this.chain.tap(true); log("interp: CHAIN active — RIFE interpolating UPSCALED frames (unified device)"); }
              else log("interp: chain unavailable (adopt failed) — RIFE on raw video");
            }
          } else if (gpuOk && !chainDev && m.confirmOrtDeviceAdopted) {
            // No renderer is consuming the former adopted device, so the new
            // RIFE pipeline itself is sufficient confirmation to retire guards.
            try { await m.confirmOrtDeviceAdopted(m.getOrtDevice?.()); }
            catch (e) { this.warn("interp: old ORT device guard release failed:", e.message); }
            if (!this._isCurrent(generation)) return false;
          }
          if (gpuOk && this.overlay && !this._octx && m.gpuConfigureCanvas) {
            try { this._gpuPresent = m.gpuConfigureCanvas(this.overlay); } catch (e) { this._gpuPresent = false; }
            if (!this._gpuPresent) {
              this._octx = this.overlay.getContext("2d", { colorSpace: SRGB_COLOR_SPACE });
            }
            log(`interp: GPU present ${this._gpuPresent ? "ACTIVE (no readback)" : "unavailable — readback path"}`);
          }
          if (gpuOk && !this._gpuPresent) {
            if (this._chainActive && this.chain?.tap) { try { this.chain.tap(false); } catch {} }
            if (this._chainInverted) {
              try { this.chain?.setInverted?.(false); } catch {}
              this._chainInverted = false;
              if (this.overlay) this.overlay.style.display = "";
            }
            this._chainActive = false;
            await m.destroyGpuInterp?.();
            if (!this._isCurrent(generation)) return false;
            this._gpuInterpActive = false; stats.gpuPath = false;
          }
          if (this._gpuInterpActive) log("interp: GPU-resident path ACTIVE (no readback stall)");
          else log("interp: GPU-resident path unavailable — using CPU grab path");
        } catch (e) { this.warn("interp: GPU path init error:", e.message); this._gpuInterpActive = false; }
      }
      if (!this._gpuPresent) await this._ensureCpuGrabber(generation);
      if (!this._isCurrent(generation)) return false;
      this._pipelineReady = true;
      return true;
    }).catch(async (e) => {
      if (!this._isCurrent(generation)) return false;
      this.warn("RIFE load failed:", e.message);
      stats.rifeError = e.message;
      await this._ensureCpuGrabber(generation);
      if (!this._isCurrent(generation)) return false;
      this._pipelineReady = true;
      return true;
    });
    this._pipelineInitPromise = pipelinePromise;
    // reusable canvas for the blend fallback tween
    this._blend = new OffscreenCanvas(2, 2);
    this._blendCtx = this._blend.getContext("2d", { colorSpace: SRGB_COLOR_SPACE });
    const blendCanvas = this._blend;
    const bctx = this._blendCtx;
    const makeBlend = (a, b, w, h) => {
      if (blendCanvas.width !== w || blendCanvas.height !== h) { blendCanvas.width = w; blendCanvas.height = h; }
      bctx.globalAlpha = 1.0; bctx.drawImage(a, 0, 0, w, h);
      bctx.globalAlpha = 0.5; bctx.drawImage(b, 0, 0, w, h);
      bctx.globalAlpha = 1.0;
      return blendCanvas;
    };

    // Process one grabbed real frame: { bmp, ts, w, h }. Produces a RIFE tween
    // between the previous and current frame, then queues both.
    const processFrame = async (cur, frameFlushGeneration = this._flushGen || 0) => {
      const frameCurrent = () => this._isCurrent(generation) &&
        (this._flushGen || 0) === frameFlushGeneration;
      const closeCurrent = () => {
        for (const key of ["bmp", "prevBmp"]) {
          try { cur[key]?.close?.(); } catch {}
          cur[key] = null;
        }
      };
      let prior = null;
      try {
        if (!frameCurrent()) {
          closeCurrent();
          return;
        }
        stats.framesIn++;
        const now = performance.now();
        if (lastFrameWall) {
          const gap = now - lastFrameWall;
          stats.lastGapMs = gap;
          if (gap > stats.maxGapMs) stats.maxGapMs = gap;
          if (gap > 50 && stats.framesOut > 4) {
            stats.stutters = (stats.stutters || 0) + 1;
            stats.lastStutterAt = stats.framesOut;
          }
        }
        lastFrameWall = now;

        // Claim the lookahead locally before any inference await. A concurrent
        // seek/source flush can then clear the shared slot without closing a
        // bitmap still in use by this frame; this continuation owns `prior`.
        prior = prevFrame;
        prevFrame = null;
        if (prior) {
          const w = cur.w, h = cur.h;
          const tMid = Math.round((prior.ts + cur.ts) / 2);
          let tweenCanvas = null;
          if (rife && rife.isReady()) {
            const t0 = performance.now();
            const scale = this._resolveScale();
            const out = await rife.interpolate(prior.bmp, cur.bmp, w, h, 0.5, scale);
            if (!frameCurrent()) {
              try { prior.bmp?.close?.(); } catch {}
              prior = null;
              closeCurrent();
              return;
            }
            stats.lastInferMs = performance.now() - t0;
            if (stats.lastInferMs > (stats.maxInferMs || 0)) stats.maxInferMs = stats.lastInferMs;
            this._adaptScale(stats.lastInferMs, prior.ts, cur.ts);
            if (out) {
              if (!this._tw || this._tw.width !== w || this._tw.height !== h) {
                this._tw = new OffscreenCanvas(w, h);
                this._twctx = this._tw.getContext("2d", { colorSpace: SRGB_COLOR_SPACE });
              }
              this._twctx.drawImage(out, 0, 0, out._cropW, out._cropH, 0, 0, w, h);
              tweenCanvas = this._tw;
            }
          }
          if (!tweenCanvas) tweenCanvas = makeBlend(prior.bmp, cur.bmp, w, h);
          let tweenBitmap = null;
          try {
            tweenBitmap = await createImageBitmap(tweenCanvas, 0, 0, w, h);
          } catch {}
          if (!this._commitCpuTweenBitmap(
            generation, cur, tweenBitmap, tMid, stats, frameFlushGeneration,
          )) {
            cur.bmp = null;
            cur.prevBmp = null;
            try { prior.bmp?.close?.(); } catch {}
            prior = null;
            return;
          }
          // done with the previous frame's bitmap
          prior.bmp.close && prior.bmp.close();
          prior = null;
        }
        if (!frameCurrent()) {
          closeCurrent();
          return;
        }
        // queue the current real frame (already a clean RGB bitmap). Keep it as
        // prev for the next tween; the scheduler owns/closes the queued copy, so
        // enqueue a clone-equivalent: we enqueue cur.bmp and keep a separate grab
        // for prev (handled by the grab loop passing fresh bitmaps each time).
        this._enqueue(cur.bmp, cur.ts);
        cur.bmp = null;
        stats.framesOut++;
        prevFrame = { bmp: cur.prevBmp, ts: cur.ts }; // prevBmp = second copy for lookahead
        cur.prevBmp = null;

        if (now - stats.lastReport > 2000) {
          const elapsed = (now - stats.started) / 1000;
          const fpsIn = stats.framesIn / elapsed;
          const fpsOut = stats.framesOut / elapsed;
          const mode = (rife && rife.isReady()) ? `RIFE infer=${(stats.lastInferMs||0).toFixed(1)}ms max=${(stats.maxInferMs||0).toFixed(1)}ms` : "blend";
          log(`interp: in=${fpsIn.toFixed(1)}fps out=${fpsOut.toFixed(1)}fps ${mode} maxGap=${stats.maxGapMs.toFixed(0)}ms stutters=${stats.stutters || 0}`);
          stats.lastReport = now;
        }
      } catch (error) {
        try { prior?.bmp?.close?.(); } catch {}
        closeCurrent();
        throw error;
      }
    };

    // 5. Grab loop: on each presented video frame, draw the <video> element to a
    //    canvas (clean YUV→RGB) and hand two bitmap copies to processFrame — one
    //    to enqueue/present (scheduler owns/closes it), one to hold as prev for
    //    the next tween's lookahead.
    this.abort = new AbortController();
    this._stopped = false;
    this._pipelineReady = false; // gate: no frame processing until GPU/CPU decided
    const grabLoop = async (now, meta) => {
      if (!this._isCurrent(generation)) return;
      // PIPELINE: re-register for the NEXT frame FIRST. Previously this happened at
      // the end — after `await` on RIFE inference — so source frames arriving during
      // a 30-45ms inference were never observed (dropped frames while the GPU sat
      // mostly idle: latency-serialization, not compute limits).
      try {
        this._rvfcId = video.requestVideoFrameCallback(grabLoop);
      } catch (error) {
        this._handlePipelineFailure(generation, error, "frame callback", { terminal: true });
        return;
      }
      // Wait until the pipeline decision is finalized (GPU present vs CPU) before
      // processing ANY frame — otherwise early CPU frames grab a 2D context and block
      // the WebGPU-present config (the canvas can't have both).
      if (!this._pipelineReady) return;
      const productionEligible = this._productionEligible();
      if (!productionEligible) {
        if (this._productionWasEligible) this._flush?.();
        this._productionWasEligible = false;
        return;
      }
      if (!this._productionWasEligible) this._flush?.();
      this._productionWasEligible = true;
      try {
        const vw = video.videoWidth, vh = video.videoHeight;
        if (vw && vh) {
          // Decoder geometry is a source boundary for both GPU and CPU paths.
          // Flush before capturing the first new-size frame so no retained old
          // lookahead can blend across adaptive-stream replacements.
          if (this._lastVW !== vw || this._lastVH !== vh) {
            const firstSight = this._lastVW == null;
            this._lastVW = vw; this._lastVH = vh;
            if (!firstSight) {
              if (this._chainInverted) {
                this._scheduleDimsRestart(generation, vw, vh);
                return;
              }
              log(`interp: source resized to ${vw}x${vh} — flushing old-size scheduler state`);
              this._flush && this._flush();
            }
            if (!this._forceBlend && this._rifeMod?.gpuRifeCapable?.()) {
              this._interpMode = "rife"; this._fallbackArmed = true;
              this._srcFrameBase = null;
            }
          }
          const frameFlushGeneration = this._flushGen || 0;
          // GPU-RESIDENT PATH: capture, pack, inference, composite, queue, and
          // presentation all stay on the shared device. It is selected only when
          // the WebGPU canvas was configured successfully; otherwise the CPU path
          // below performs one clean grab and bitmap presentation.
          if (this._gpuPresent && this._rifeMod && this._rifeMod.gpuActive()) {
            const ts = Math.round((meta && meta.mediaTime != null ? meta.mediaTime : video.currentTime) * 1e6);
            // Adaptive fallback evaluation (~every 2s): compare output frames to the
            // SOURCE frames the browser actually presented (meta.presentedFrames
            // captures the ones we drop under load). If RIFE isn't buying ≥50% over
            // source, switch to cheap blend so we keep up and hit the target.
            if (!this._forceBlend && this._rifeMod.gpuRifeCapable && this._rifeMod.gpuRifeCapable()) {
              const pf = meta && meta.presentedFrames;
              if (this._srcFrameBase == null) {
                this._srcFrameBase = pf || 0; this._outFrameBase = stats.framesOut; this._fbWindowStart = now;
              } else if (now - this._fbWindowStart > 2000) {
                const srcDelta = (pf || 0) - this._srcFrameBase;
                const outDelta = stats.framesOut - this._outFrameBase;
                if (this._autoFallback !== false && this._interpMode === "rife" && this._fallbackArmed && srcDelta > 5) {
                  const ratio = outDelta / srcDelta;
                  if (ratio < this._fallbackRatio) {
                    this._interpMode = "blend"; this._fallbackArmed = false;
                    log(`interp: RIFE can't keep up (${ratio.toFixed(2)}x ≤ ${this._fallbackRatio}x source) → blend fallback`);
                  }
                }
                this._srcFrameBase = pf || 0; this._outFrameBase = stats.framesOut; this._fbWindowStart = now;
              }
            }
            {
              // capture → pooled real texture; tween(s) → pooled texture(s); queue;
              // present via WebGPU; recycle on release. Blend can insert MULTIPLE
              // tweens per gap to reach the target framerate; RIFE inserts one (2x).
              // CHAIN mode captures the UPSCALER's finished frame (same device)
              // instead of the raw video, so tweens are between upscaled frames.
              let realTex = null;
              if (this._chainActive && this.chain && this.chain.info) {
                const tap = this.chain.info();
                if (tap && tap.tex && tap.frame !== this._lastTapFrame) {
                  this._lastTapFrame = tap.frame;
                  this._recordPipelineSuccess("chain tap");
                  // tap size changed (e.g. upscale scale switch): flush the queue so
                  // old-size frames never present interleaved with new-size ones
                  if (this._lastTapW && (this._lastTapW !== tap.w || this._lastTapH !== tap.h)) {
                    log(`interp: chain tap resized ${this._lastTapW}x${this._lastTapH} → ${tap.w}x${tap.h} — flushing queue`);
                    this._flush && this._flush();
                  }
                  this._lastTapW = tap.w; this._lastTapH = tap.h;
                  realTex = this._rifeMod.gpuCaptureTex(tap.tex);
                } else {
                  // no (new) upscaled frame yet this tick — wait for the tap rather
                  // than mixing raw-video frames into an upscaled stream
                  if (!this._tapStaleSince) this._tapStaleSince = now;
                  else if (now - this._tapStaleSince > 2000 && !video.paused && (!this._tapStaleLogAt || now - this._tapStaleLogAt > 5000)) {
                    this._tapStaleLogAt = now;
                    this.warn(`interp WATCHDOG: chain tap stale ${(now - this._tapStaleSince).toFixed(0)}ms while video playing (upscaler not producing? tap=${tap ? tap.frame : "null"})`);
                  }
                  if (now - this._tapStaleSince > 2000 && !video.paused) {
                    this._handlePipelineFailure(
                      generation,
                      new Error("upscaler stopped producing source frames"),
                      "chain tap",
                    );
                  }
                  return; // already re-registered at top
                }
                this._tapStaleSince = 0;
              } else {
                realTex = this._rifeMod.gpuCapture(video);
              }
              // A failed capture did not update curTex. Advancing or counting it
              // would pair a stale/blank texture with the next frame. Temporary
              // pool/fence pressure drops only this tick; a request that can never
              // fit is terminal so presentation returns fully to the source video.
              if (!realTex) {
                if (!this._handleGpuCaptureFailure(generation)) {
                  const captureError = this._rifeMod?.gpuLastCaptureError?.()
                    || new Error("GPU capture returned no texture");
                  this._handlePipelineFailure(generation, captureError, "capture");
                }
                return;
              }
              if (!this._rifeMod.gpuHasPrev()) { this._rifeMod.gpuAdvance(); this._enqueueTex(realTex, ts); this._prevTs = ts; stats.framesIn++; }
              else {
                const t0 = performance.now();
                const p0 = this._prevTs != null ? this._prevTs : ts - 33000;
                // If we started standalone (blend-only, no ORT), RIFE inference isn't
                // available — use blend even in "rife" mode until a stop/start reloads.
                const canRife = this._interpMode === "rife" && (!this._rifeMod.gpuRifeCapable || this._rifeMod.gpuRifeCapable());
                if (this._interpMode === "blend" || !canRife) {
                  const nT = this._tweensForGap(p0, ts); // number of tweens to insert
                  for (let k = 1; k <= nT; k++) {
                    const frac = k / (nT + 1);
                    const tex = this._rifeMod.gpuBlend(frac);
                    if (tex) { this._enqueueTex(tex, Math.round(p0 + (ts - p0) * frac)); stats.framesOut++; }
                  }
                  stats.lastInferMs = performance.now() - t0;
                } else {
                  // PIPELINED MULTI-TWEEN RIFE (self-clocking): real frames always
                  // enqueue on time; _rifeChain runs bisection-level tweens for the
                  // pair, gated by its own measured inference time. If a chain is
                  // running, the new pair PARKS in the pending slot (preempting the
                  // chain between inferences) rather than being skipped — a gap
                  // only counts as skipped if its pending slot is replaced before
                  // it ever starts (zero tweens).
                  const pairPrev = this._pipePrev; // {tex, ts} retained pooled real
                  if (pairPrev && realTex) {
                    this._rifeMod.gpuRetain(realTex); // cur side of the pair
                    const pair = { prev: pairPrev, cur: { tex: realTex, ts } };
                    if (!this._inferBusy) {
                      this._inferBusy = true;
                      this._rifeChain(pair.prev, pair.cur, generation); // fire-and-forget; refs transfer
                    } else {
                      if (this._pendingPair) {
                        // replacing an unstarted pair: that gap gets zero tweens
                        stats.skippedTweens = (stats.skippedTweens || 0) + 1;
                        this._rifeMod.gpuRelease(this._pendingPair.prev.tex);
                        this._rifeMod.gpuRelease(this._pendingPair.cur.tex);
                      }
                      this._pendingPair = pair; // refs transfer to the slot
                    }
                  } else if (pairPrev) {
                    this._rifeMod.gpuRelease(pairPrev.tex); // no cur frame this tick
                  }
                  // hold cur as the prev of the NEXT pair (extra ref beyond the queue's)
                  if (realTex) { this._rifeMod.gpuRetain(realTex); this._pipePrev = { tex: realTex, ts }; }
                  else this._pipePrev = null;
                }
                if (realTex) { this._enqueueTex(realTex, ts); stats.framesOut++; }
                this._rifeMod.gpuAdvance();
                this._prevTs = ts; stats.framesIn++;
              }
              this._recordPipelineSuccess("capture");
            }
          } else {
          // CPU path awaits (readback, createImageBitmap): with rvfc re-registered at
          // the top of the loop, ticks can overlap — serialize them (skip overlapped
          // frames) so shared staging buffers/canvases aren't used concurrently.
          if (cpuTickBusy) return;
          cpuTickBusy = true;
          let finishCpuFrameTask;
          const cpuFrameTask = new Promise((resolve) => { finishCpuFrameTask = resolve; });
          this._activeCpuFrameTasks.add(cpuFrameTask);
          try {
          if (grabCanvas.width !== vw || grabCanvas.height !== vh) {
            grabCanvas.width = vw; grabCanvas.height = vh;
          }
          // CLEAN CPU PATH: WebGPU importExternalTexture → readback → putImageData.
          let got = false;
          const gpuGrab = this._gpuGrab;
          if (gpuGrab && gpuGrab.ready) {
            const rgba = await gpuGrab.grab(video);
            if (!this._isCurrent(generation) || (this._flushGen || 0) !== frameFlushGeneration) return;
            if (rgba) {
              grabCtx.putImageData(new ImageData(rgba.data, rgba.width, rgba.height), 0, 0);
              got = true;
            }
          }
          if (!got) grabCtx.drawImage(video, 0, 0, vw, vh); // fallback (may ring)
          const [bmp, prevBmp] = await Promise.all([
            createImageBitmap(grabCanvas),
            createImageBitmap(grabCanvas),
          ]);
          if (!this._isCurrent(generation) || (this._flushGen || 0) !== frameFlushGeneration) {
            bmp.close?.(); prevBmp.close?.(); return;
          }
          const ts = Math.round((meta && meta.mediaTime != null ? meta.mediaTime : video.currentTime) * 1e6);
          await processFrame({ bmp, prevBmp, ts, w: vw, h: vh }, frameFlushGeneration);
          if (this._isCurrent(generation) && (this._flushGen || 0) === frameFlushGeneration) {
            this._recordPipelineSuccess("capture");
          }
          } finally {
            cpuTickBusy = false;
            this._activeCpuFrameTasks.delete(cpuFrameTask);
            finishCpuFrameTask();
          }
          }
        }
      } catch (e) {
        if (!this._stopped) {
          this.warn("interp grab error:", e.message);
          this._handlePipelineFailure(generation, e, "capture");
        }
      }
    };

    // 6. present on a CANVAS we fully control. Every queued bitmap (real or tween)
    //    is drawn with the identical drawImage call, so there is exactly one pixel
    //    path and no color/brightness asymmetry. A rAF loop draws the due frame
    //    paced by the source timestamps. This replaces the MediaStreamTrackGenerator
    //    + overlay <video>, whose compositor color-managed our two frame types
    //    differently (the flicker).
    this.queue = [];           // { bmp, ts } in microseconds, ascending
    this._lastEnqTs = null;    // for learning the output interval
    this._targetInterval = 0;  // smoothed ms per output frame
    this._discontinuity = true; // snap latency on the first measurement
    this._videoLatencyMs = null;
    this.overlay = document.createElement("canvas");
    this.overlay.setAttribute?.("data-fsrcnnx-overlay", "interpolation");
    Object.assign(this.overlay.style, {
      position: "fixed", pointerEvents: "none", zIndex: "2147483646", background: "#000",
    });
    // Context selection is deferred: pipeline initialization claims WebGPU when
    // possible. Keep this canvas detached, and leave source visibility/audio alone,
    // until _present() has produced a real output and commits the takeover.

    // Seeking, or playing after a pause, makes the source clock jump relative to
    // our buffered frames — a legitimate latency discontinuity. Flush the stale
    // buffer and flag a snap so the latency estimate + audio delay re-zero hard
    // instead of crawling (which would leave audio desynced for seconds).
    this._flush = () => {
      if (this.queue) { for (const it of this.queue) { it.bmp && it.bmp.close && it.bmp.close(); if (it.tex && this._rifeMod) this._rifeMod.gpuRelease(it.tex); } this.queue.length = 0; }
      // a discontinuity is a fresh start — retry RIFE and re-measure whether it keeps
      // up (unless the user forced the blend engine)
      this._interpMode = this._forceBlend ? "blend" : "rife";
      this._fallbackArmed = true; this._srcFrameBase = null;
      this._discontinuity = true;
      this._reanchor = true;       // presentation anchor is stale after a discontinuity
      this._flushGen = (this._flushGen || 0) + 1; // invalidate in-flight pipelined tweens
      try { this._closeCpuPrev?.(); } catch {}
      try { this._rifeMod?.gpuResetFrames?.(); } catch {}
      if (this._pipePrev) { try { this._rifeMod && this._rifeMod.gpuRelease(this._pipePrev.tex); } catch {} this._pipePrev = null; }
      if (this._pendingPair) { try { this._rifeMod.gpuRelease(this._pendingPair.prev.tex); this._rifeMod.gpuRelease(this._pendingPair.cur.tex); } catch {} this._pendingPair = null; }
      this._lastPresentedTs = null;
      this._prevTs = null;
      this._lastEnqTs = null;
      this._targetInterval = 0;
      lastFrameWall = 0;
      this._videoLatencyMs = null; // force a fresh measurement
    };
    // Media discontinuities belong to the capture lifecycle, not the optional
    // canvas/audio takeover. They must flush retained history while startup is
    // pending and while fullscreen/visibility temporarily restores presentation
    // to the source element.
    this._installMediaBoundaryListeners(video, generation);
    // Re-assert persisted prefs on EVERY start: whatever restart path brought us
    // here (watchdog, video swap, toggle restart), the source of truth is main's
    // per-site prefs, not this instance's last life.
    if (this.chain && this.chain.ladder) this._ladderOn = !!this.chain.ladder();

    const pipelineOk = await pipelinePromise;
    if (this._pipelineInitPromise === pipelinePromise) this._pipelineInitPromise = null;
    if (!this._isCurrent(generation)) return { ok: false, reason: "cancelled" };
    if (!pipelineOk) return { ok: false, reason: "pipeline-unavailable" };
    try {
      this._rvfcId = video.requestVideoFrameCallback(grabLoop);
    } catch (error) {
      this.warn("interpolation: requestVideoFrameCallback scheduling failed:", error.message);
      return { ok: false, reason: "rvfc-schedule-failed" };
    }
    this._present(generation);
    this.log(`interpolation started (${this._forceBlend ? "blend" : this._rifeModelKey || "default RIFE"})`);
    return { ok: true };
  }

  position() {
    if (!this.overlay || !this.video) return;
    const r = this.video.getBoundingClientRect();
    // fixed positioning uses viewport coordinates directly (no scroll offset)
    Object.assign(this.overlay.style, {
      left: `${r.left}px`,
      top: `${r.top}px`,
      width: `${r.width}px`,
      height: `${r.height}px`,
    });
  }

  stop({ preservePendingDeviceLoss = false } = {}) {
    // A queued inverted-dimension callback belongs to the lifecycle being stopped.
    // Reset even for an already-idle instance so it can never suppress a later run.
    this._dimsRestarting = false;
    this._gpuResourceStopQueued = false;
    this._pipelineFailureStopQueued = false;
    const publishedGrabber = this._gpuGrab;
    const pendingGrabber = this._cpuGrabInit?.grabber;
    // Unpublish synchronously; physical device teardown remains observable through
    // _cpuGrabberTeardowns and is awaited by start(), recovery, and retirement.
    this._gpuGrab = null;
    this._destroyCpuGrabber(publishedGrabber);
    if (pendingGrabber !== publishedGrabber) this._destroyCpuGrabber(pendingGrabber);
    const active = this._state !== "idle" || this.running || this.video || this.overlay || this.queue ||
      publishedGrabber || pendingGrabber;
    if (!active) {
      this._takeoverActive = false;
      this._committedPresentation = null;
      this._presentationGeneration = 0;
      this.stats.framesPresented = 0;
      return { ok: true, stopped: false };
    }
    ++this._lifecycleGen; // invalidate module imports, inference continuations, and rvfc
    // An explicit stop/model change owns the newer lifecycle.  It must also cancel
    // a replacement-device loss queued behind an older automatic restart.
    if (!preservePendingDeviceLoss) this._pendingDeviceLoss = null;
    this.running = false;
    this._state = "idle";
    this._stopped = true; // ends the requestVideoFrameCallback grab loop
    try { this.abort?.abort(); } catch {}
    if (this.video && this._rvfcId != null && typeof this.video.cancelVideoFrameCallback === "function") {
      try { this.video.cancelVideoFrameCallback(this._rvfcId); } catch {}
    }
    this._rvfcId = null;
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
    this._removeTakeoverListeners();
    this._removeMediaBoundaryListeners();
    // Restore native audio before potentially expensive GPU draining/destruction.
    // The source stays a valid fallback throughout the remainder of teardown.
    this._teardownAudioDelay({ retireCapture: true });
    this._clearAudioBlocks();
    this._takeoverActive = false;
    this._committedPresentation = null;
    this._presentationGeneration = 0;
    this.stats.framesPresented = 0;
    // drain queued frames (release textures / close bitmaps) BEFORE freeing the GPU
    // pipeline, so nothing references pool textures after they're destroyed.
    if (this.queue) {
      for (const it of this.queue) { it.bmp && it.bmp.close && it.bmp.close(); if (it.tex && this._rifeMod) this._rifeMod.gpuRelease(it.tex); }
      this.queue = null;
    }
    if (this._pipePrev) { try { this._rifeMod && this._rifeMod.gpuRelease(this._pipePrev.tex); } catch {} this._pipePrev = null; }
    if (this._pendingPair) { try { this._rifeMod && this._rifeMod.gpuRelease(this._pendingPair.prev.tex); this._rifeMod.gpuRelease(this._pendingPair.cur.tex); } catch {} this._pendingPair = null; }
    try { this._closeCpuPrev?.(); } catch {}
    this._closeCpuPrev = null;
    // All externally-held pooled textures have been returned. GpuInterp defers
    // physical destruction until submitted work and any in-flight tween finish.
    try { this._rifeMod && this._rifeMod.destroyGpuInterp && this._rifeMod.destroyGpuInterp(); } catch {}
    if (this._chainActive && this.chain && this.chain.tap) { try { this.chain.tap(false); } catch {} }
    if (this._chainInverted) {
      if (!this._chainPresentationSuspended) {
        try { this.chain && this.chain.setInverted && this.chain.setInverted(false); } catch {}
      }
      this._chainInverted = false;
      this._chainPresentationSuspended = false;
      if (this.overlay) this.overlay.style.display = "";
    }
    this._chainActive = false; this._lastTapFrame = null; this._lastTapW = 0; this._lastTapH = 0;
    this._tapStaleSince = 0; this._tapStaleLogAt = 0; this._invUpWarned = false;
    this._inferBusy = false; this._flushGen = (this._flushGen || 0) + 1;
    this._gpuInterpActive = false; this._gpuPresent = false;
    this._octx = null;            // stale 2D ref would block WebGPU-present on restart
    this._pipelineReady = false;
    this._pipelineFailureStreaks = Object.create(null);
    this._lastEnqTs = null; this._targetInterval = 0;
    if (this.overlay) { this.overlay.remove(); this.overlay = null; }
    this.processor = null;
    this.abort = null;
    this.video = null;
    this._onScroll = null; this._onPresentationBoundary = null;
    this._onSeeking = null; this._onPlay = null;
    this.log("interpolation stopped");
    return { ok: true, stopped: true };
  }

  retireGpuResources() {
    if (this._resourceRetirementPromise) return this._resourceRetirementPromise;
    let resolveRetirement;
    const completion = new Promise((resolve) => { resolveRetirement = resolve; });
    let exposed;
    exposed = completion.finally(() => {
      if (this._resourceRetirementPromise === exposed) {
        this._resourceRetirementPromise = null;
      }
    });
    // Publish before stop(): chain.tap(false), injected logging, and DOM cleanup
    // are synchronous user-adjacent callbacks which may re-enter start(). Such a
    // request must queue behind this exact physical retirement generation.
    this._resourceRetirementPromise = exposed;

    const pendingStart = this._startPromise;
    const pendingPipeline = this._pipelineInitPromise;
    const pendingGrabInit = this._cpuGrabInit?.promise;
    const pendingGrabRecovery = this._cpuGrabRecovery?.promise;
    const pendingCpuFrameTasks = [...this._activeCpuFrameTasks];
    const grabber = this._gpuGrab || this._cpuGrabInit?.grabber || null;
    try { this.stop(); }
    catch (error) {
      try { this.warn("interpolation stop during GPU retirement failed:", error.message); } catch {}
    }
    this._destroyCpuGrabber(grabber);
    const operation = (async () => {
      await Promise.allSettled([
        Promise.resolve(pendingStart),
        Promise.resolve(pendingPipeline),
        Promise.resolve(pendingGrabInit),
        Promise.resolve(pendingGrabRecovery),
        ...pendingCpuFrameTasks,
      ]);
      await this._waitForCpuGrabberTeardown();
      try { await this._rifeMod?.destroyGpuInterp?.(); } catch {}
      // stop() deliberately leaves reusable per-run canvases and closures in
      // place for an ordinary restart. Explicit GPU retirement is stronger: by
      // this point every producer captured above has drained, so source-sized
      // backing stores can be discarded without racing an in-flight CPU frame.
      this._releaseRetainedLifecycleResources();
    });
    operation().then(
      () => resolveRetirement(),
      (error) => {
        try { this.warn("interpolation GPU retirement failed:", error.message); } catch {}
        resolveRetirement();
      },
    );
    return exposed;
  }

  _releaseRifeDeviceLossListener() {
    const unsubscribe = this._deviceLossUnsubscribe;
    this._deviceLossUnsubscribe = null;
    try { unsubscribe?.(); } catch {}
  }

  _releaseRetainedLifecycleResources() {
    // Resizing clears backing stores immediately; nulling both the canvas and its
    // context then removes the instance's last strong references. The blend canvas
    // used to live only in a per-run closure, so make its ownership explicit too.
    for (const canvas of [this._grab, this._blend, this._tw]) {
      if (!canvas) continue;
      try { canvas.width = 0; canvas.height = 0; } catch {}
    }
    this._grabCtx = null;
    this._grab = null;
    this._blendCtx = null;
    this._blend = null;
    this._twctx = null;
    this._tw = null;

    // _flush is created inside _startInternal and otherwise keeps that run's full
    // lexical environment (video, canvases, frame closures) reachable after stop.
    // Ordinary stop intentionally keeps reusable state; explicit GPU/model
    // retirement does not, and a subsequent start constructs a fresh lifecycle.
    this._flush = null;
    this._closeCpuPrev = null;
    this._activeCpuFrameTasks.clear();
    this._lastTapFrame = null;
    this._lastTapW = 0;
    this._lastTapH = 0;
    this._lastVW = null;
    this._lastVH = null;
    this._infWindow = null;
    this._srcFrameBase = null;
    this._outFrameBase = null;
    this._fbWindowStart = null;
    this._prevTs = null;
    this._lastPresentedTs = null;
    this._lastEnqTs = null;
    this._videoLatencyMs = null;
  }

  releaseModelResources() {
    if (this._modelRetirementPromise) return this._modelRetirementPromise;
    let resolveRetirement;
    let rejectRetirement;
    const completion = new Promise((resolve, reject) => {
      resolveRetirement = resolve;
      rejectRetirement = reject;
    });
    let exposed;
    exposed = completion.finally(() => {
      if (this._modelRetirementPromise === exposed) this._modelRetirementPromise = null;
    });
    // Publish before invoking the external unsubscribe callback for the same
    // re-entrancy reason as resource retirement above.
    this._modelRetirementPromise = exposed;
    // RIFE owns a module-global listener set. Unsubscribe synchronously so an idle
    // discarded Interpolator is no longer rooted while queue/model draining runs;
    // repeat after the drain to cover a listener installed by a losing start.
    this._releaseRifeDeviceLossListener();
    const operation = (async () => {
      try {
        await this.retireGpuResources();
        await this._rifeMod?.disposeRife?.();
      } finally {
        this._releaseRifeDeviceLossListener();
        this._releaseRetainedLifecycleResources();
        this._rifeMod = null;
      }
    });
    operation().then(resolveRetirement, rejectRetirement);
    return exposed;
  }

  getStats() {
    const s = this.stats;
    const elapsed = s.started ? (performance.now() - s.started) / 1000 : 0;
    return {
      running: this.running,
      phase: this._state,
      framesIn: s.framesIn,
      framesOut: s.framesOut,
      framesPresented: s.framesPresented || 0,
      fpsIn: elapsed ? +(s.framesIn / elapsed).toFixed(1) : 0,
      fpsOut: elapsed ? +(s.framesOut / elapsed).toFixed(1) : 0,
      maxGapMs: Math.round(s.maxGapMs),
      lastGapMs: Math.round(s.lastGapMs),
      stutters: s.stutters || 0,
      rife: !!s.rife,
      rifeError: s.rifeError || null,
      resMode: this.resMode,
      scale: this._resolveScale(),
      inferMs: s.lastInferMs ? +s.lastInferMs.toFixed(1) : 0,
      inferMeanMs: (this._infWindow && this._infWindow.length >= 10)
        ? +(this._infWindow.reduce((a, b) => a + b, 0) / this._infWindow.length).toFixed(1) : 0,
      maxInferMs: s.maxInferMs ? +s.maxInferMs.toFixed(1) : 0,
      timing: (this._rifeMod && this._rifeMod.getTiming) ? this._rifeMod.getTiming() : null,
      fp16: (this._rifeMod && this._rifeMod.isFp16) ? this._rifeMod.isFp16() : false,
      capture: (this._rifeMod && this._rifeMod.graphCaptureActive) ? this._rifeMod.graphCaptureActive() : false,
      jspi: (this._rifeMod && this._rifeMod.usingJspi) ? this._rifeMod.usingJspi() : false,
      inverted: !!this._chainInverted,
      takeoverActive: !!this._takeoverActive,
      presentation: this._takeoverActive ? this._committedPresentation : null,
      gpuPath: !!this._gpuInterpActive,
      chain: !!this._chainActive,
      skippedTweens: s.skippedTweens || 0,
      ladderBlends: s.ladderBlends || 0,
      skipRate: elapsed ? +((s.skippedTweens || 0) / elapsed).toFixed(2) : 0,
      interpMode: this._interpMode || "rife",
      forceBlend: !!this._forceBlend,
      targetFps: this._targetFpsMode,
      detectedHz: this._detectedHz || null,
      effectiveTargetFps: this._effectiveTargetFps(),
      models: (() => {
        const base = (this._rifeMod && this._rifeMod.listModels) ? this._rifeMod.listModels() : [];
        // mark RIFE models not-current when blend is forced; append the blend engine
        const list = base.map(m => ({ ...m, current: this._forceBlend ? false : m.current }));
        list.push({ key: "blend", label: "Blend (no AI, high fps)", current: !!this._forceBlend });
        return list;
      })(),
      videoLatencyMs: this._videoLatencyMs != null ? Math.round(this._videoLatencyMs) : null,
      audioDelayMs: this._delayNode ? Math.round(this._delayNode.delayTime.value * 1000) : null,
      avOffsetMs: this._avOffsetMs,
    };
  }
}

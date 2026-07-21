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

export class Interpolator {
  constructor({ findVideo, log, warn, chain }) {
    this.findVideo = findVideo;
    this.log = log || console.log;
    this.warn = warn || console.warn;
    this.chain = chain || null; // upscaler chain accessors { tap, info, available, device }
    this.running = false;
    this._state = "idle";       // "idle" | "starting" | "running"
    this._lifecycleGen = 0;      // invalidates asynchronous work after stop/restart
    this._startPromise = null;
    this._dimsRestarting = false;
    this.video = null;
    this.overlay = null;
    this.processor = null;
    this.abort = null;
    // resolution control: "full" | "half" | "quarter" | "auto"
    // Now that static-region passthrough stabilizes still detail INDEPENDENTLY of
    // inference resolution, resolution no longer has to be maxed to avoid jitter —
    // so Auto can pick the resolution that holds cadence (much cheaper). Motion is
    // forgiving of lower res; static stability comes from passthrough, not res.
    this.resMode = "auto";
    this._autoScale = 0.625; // start moderate; adapt toward the cadence budget
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
  // sags). Persisted per site; default ON. The error CIRCUIT BREAKER (5
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
  _present() {
    const FILL_MS = 100;
    let started = false;
    let anchorWall = 0;   // wall-clock (ms) mapped to anchorSrc
    let anchorSrc = 0;    // source timestamp (ms) of the presentation origin
    // Display refresh-rate detection: sample rAF intervals; the median over a window
    // ≈ the frame period → Hz. Used as the blend target when target mode is "auto".
    let hzSamples = [];
    let hzLast = 0;
    const loop = () => {
      if (!this.overlay) return;
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
              this._lastPresentAt = now;
              this._lastPresentedTs = item.ts;
              if (item.tex) {
                // GPU present: render the pooled texture via WebGPU, then recycle.
                // Inverted chain: the pooled tex is SOURCE-RES — hand it to the
                // upscaler (full pass chain to its canvas) instead of blitting.
                if (this._chainInverted && this.chain && this.chain.upscaleTex) {
                  if (!this.chain.upscaleTex(item.tex, item.tex._w, item.tex._h) && !this._invUpWarned) {
                    this._invUpWarned = true;
                    this.warn("interp: INVERTED upscale REJECTED (upscaler off/deviceless?) — frames are being dropped");
                  }
                } else {
                  this._rifeMod && this._rifeMod.gpuPresent(item.tex);
                }
              } else {
                // bitmap present: lazily grab a 2D context if WebGPU didn't claim it
                if (!this._octx && !this._gpuPresent) { try { this._octx = this.overlay.getContext("2d"); } catch {} }
                if (this._octx) {
                  if (this.overlay.width !== item.bmp.width || this.overlay.height !== item.bmp.height) {
                    this.overlay.width = item.bmp.width; this.overlay.height = item.bmp.height;
                  }
                  this._octx.drawImage(item.bmp, 0, 0);
                }
              }
              const dwell = now - item.enq;
              if (dwell >= 0 && dwell < 1000) {
                if (this._discontinuity) { this._videoLatencyMs = dwell; this._discontinuity = false; this._snapAudio = true; }
                else this._videoLatencyMs = this._videoLatencyMs == null ? dwell : this._videoLatencyMs * 0.9 + dwell * 0.1;
              }
              if (item.tex) { this._rifeMod && this._rifeMod.gpuRelease(item.tex); }
              else { item.bmp.close && item.bmp.close(); }
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
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  // Route the original element's audio through a Web Audio DelayNode so we can
  // delay it to match the video's presentation latency (the buffer we added for
  // smoothness put video ~100ms behind audio). MediaElementAudioSourceNode
  // redirects the element's audio into the graph, so the element no longer outputs
  // directly — no double audio. Only one source node may exist per element, ever.
  _setupAudioDelay(video) {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      // A MediaElementAudioSourceNode permanently belongs to the AudioContext that
      // created it. Cache the pair, not just the source: reconnecting a cached node
      // to a newly-created context is invalid and made every restart lose audio.
      let state = video._fsrcnnxAudioState;
      if (!state && video._fsrcnnxAudioSrc) {
        state = { context: video._fsrcnnxAudioSrc.context, source: video._fsrcnnxAudioSrc };
        video._fsrcnnxAudioState = state;
      }
      if (!state) {
        const context = new AC();
        const source = context.createMediaElementSource(video);
        state = { context, source };
        video._fsrcnnxAudioState = state;
        video._fsrcnnxAudioSrc = source; // compatibility with pre-v0.50 sessions
      }
      if (!state.context || state.context.state === "closed") {
        throw new Error("cached media audio context is closed");
      }
      this._audioState = state;
      this._audioCtx = state.context;
      this._audioSrc = state.source;
      // Always rebuild from a known graph. disconnect() prevents duplicate direct
      // and delayed routes when interpolation is toggled repeatedly.
      try { this._audioSrc.disconnect(); } catch {}
      this._delayNode = this._audioCtx.createDelay(1.0);
      this._delayNode.delayTime.value = 0.1;
      this._audioSrc.connect(this._delayNode);
      this._delayNode.connect(this._audioCtx.destination);
      this._audioCtx.resume?.();
      this._audioTimer = setInterval(() => {
        if (!this._delayNode || this._videoLatencyMs == null) return;
        // _videoLatencyMs is now the measured buffer dwell (the real gap); audio
        // delay = that + an optional manual fine-trim (default 0).
        const target = Math.max(0, Math.min(0.95, (this._videoLatencyMs + this._avOffsetMs) / 1000));
        if (this._snapAudio) {
          this._delayNode.delayTime.cancelScheduledValues(this._audioCtx.currentTime);
          this._delayNode.delayTime.setValueAtTime(target, this._audioCtx.currentTime);
          this._snapAudio = false;
          return;
        }
        const cur = this._delayNode.delayTime.value;
        const next = cur * 0.85 + target * 0.15;
        this._delayNode.delayTime.setTargetAtTime(next, this._audioCtx.currentTime, 0.05);
      }, 250);
    } catch (e) {
      try { this._audioSrc?.disconnect(); } catch {}
      try { this._delayNode?.disconnect(); } catch {}
      try { if (this._audioSrc && this._audioCtx) this._audioSrc.connect(this._audioCtx.destination); } catch {}
      this._delayNode = null;
      this.warn("audio delay setup failed (video will lead audio):", e.message);
    }
  }

  _teardownAudioDelay() {
    if (this._audioTimer) { clearInterval(this._audioTimer); this._audioTimer = null; }
    try { this._audioSrc?.disconnect(); } catch {}
    try { this._delayNode?.disconnect(); } catch {}
    // Keep the context/source pair alive because an element can only be wrapped
    // once. Reconnect exactly once to direct output until interpolation restarts.
    try {
      if (this._audioSrc && this._audioCtx) this._audioSrc.connect(this._audioCtx.destination);
    } catch {}
    this._delayNode = null;
    this._audioSrc = null;
    this._audioCtx = null;
    this._audioState = null;
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

  _commitCpuTweenBitmap(generation, cur, tweenBitmap, timestamp, stats) {
    if (!this._isCurrent(generation)) {
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
  start() {
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
    const promise = this._startInternal(generation)
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

  async _ensureCpuGrabber(generation) {
    if (!this._isCurrent(generation) || this._gpuGrab) return !!this._gpuGrab;
    try {
      const gm = await import(chrome.runtime.getURL("fsrcnnx-grab.js"));
      if (!this._isCurrent(generation)) return false;
      const grabber = new gm.WebGPUGrabber({ log: this.log, warn: this.warn });
      const ok = await grabber.init();
      if (!this._isCurrent(generation)) {
        try { grabber.destroy(); } catch {}
        return false;
      }
      if (ok) {
        this._gpuGrab = grabber;
        this.log("interp: WebGPU clean CPU-path grab active");
        return true;
      }
      try { grabber.destroy(); } catch {}
      this.warn("interp: WebGPU grab unavailable, using 2D fallback (may show waves)");
    } catch (e) {
      if (this._isCurrent(generation)) this.warn("interp: grab module load failed:", e.message);
    }
    return false;
  }

  async _startInternal(generation) {
    if (!this.supported()) {
      this.warn("interpolation: ImageBitmap/OffscreenCanvas not available in this browser");
      return { ok: false, reason: "unsupported" };
    }
    const video = this.findVideo();
    if (!video) return { ok: false, reason: "no video" };
    this.video = video;
    this._interpMode = this._forceBlend ? "blend" : "rife";
    this._fallbackArmed = true; this._srcFrameBase = null;
    this._tweenFailStreak = 0; this._prevTs = null;
    this._lastVW = null; this._lastVH = null;

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
    this._grabCtx = this._grab.getContext("2d", { willReadFrequently: false });
    const grabCanvas = this._grab;
    const grabCtx = this._grabCtx;
    let cpuTickBusy = false;
    // The readback grabber is initialized only if GPU-resident presentation cannot
    // be established. Avoiding it on the normal path prevents a redundant adapter,
    // device, and staging allocation at every interpolation start.
    this._gpuGrab = null;
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
    const pipelinePromise = import(chrome.runtime.getURL("fsrcnnx-rife.js")).then(async (m) => {
      if (!this._isCurrent(generation)) return false;
      rife = m;
      this._rifeMod = m; // expose for getStats (timing breakdown)
      if (this._rifeModelKey && m.setModel && !m.setModel(this._rifeModelKey)) {
        this.warn(`interp: unknown RIFE model '${this._rifeModelKey}', using module default`);
        this._rifeModelKey = null;
      }
      // CHAIN RULES: blend can join the upscaler's device directly. RIFE owns the
      // ORT device, so the upscaler adopts that device before textures are shared.
      const chainDev = (this.chain && this.chain.available && this.chain.available())
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
          if (!this._gpuPresent) this._octx = this.overlay.getContext("2d");
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
            const wantInvert = adopted && this.chain.invert && this.chain.invert()
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
            if (!this._gpuPresent) this._octx = this.overlay.getContext("2d");
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
    const blendCanvas = new OffscreenCanvas(2, 2);
    const bctx = blendCanvas.getContext("2d");
    const makeBlend = (a, b, w, h) => {
      if (blendCanvas.width !== w || blendCanvas.height !== h) { blendCanvas.width = w; blendCanvas.height = h; }
      bctx.globalAlpha = 1.0; bctx.drawImage(a, 0, 0, w, h);
      bctx.globalAlpha = 0.5; bctx.drawImage(b, 0, 0, w, h);
      bctx.globalAlpha = 1.0;
      return blendCanvas;
    };

    // Process one grabbed real frame: { bmp, ts, w, h }. Produces a RIFE tween
    // between the previous and current frame, then queues both.
    const processFrame = async (cur) => {
        if (!this._isCurrent(generation)) {
          try { cur.bmp?.close?.(); cur.prevBmp?.close?.(); } catch {}
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

        if (prevFrame) {
          const w = cur.w, h = cur.h;
          const tMid = Math.round((prevFrame.ts + cur.ts) / 2);
          let tweenCanvas = null;
          if (rife && rife.isReady()) {
            const t0 = performance.now();
            const scale = this._resolveScale();
            const out = await rife.interpolate(prevFrame.bmp, cur.bmp, w, h, 0.5, scale);
            if (!this._isCurrent(generation)) {
              try { cur.bmp?.close?.(); cur.prevBmp?.close?.(); } catch {}
              return;
            }
            stats.lastInferMs = performance.now() - t0;
            if (stats.lastInferMs > (stats.maxInferMs || 0)) stats.maxInferMs = stats.lastInferMs;
            this._adaptScale(stats.lastInferMs, prevFrame.ts, cur.ts);
            if (out) {
              if (!this._tw || this._tw.width !== w || this._tw.height !== h) {
                this._tw = new OffscreenCanvas(w, h); this._twctx = this._tw.getContext("2d");
              }
              this._twctx.drawImage(out, 0, 0, out._cropW, out._cropH, 0, 0, w, h);
              tweenCanvas = this._tw;
            }
          }
          if (!tweenCanvas) tweenCanvas = makeBlend(prevFrame.bmp, cur.bmp, w, h);
          let tweenBitmap = null;
          try {
            tweenBitmap = await createImageBitmap(tweenCanvas, 0, 0, w, h);
          } catch {}
          if (!this._commitCpuTweenBitmap(generation, cur, tweenBitmap, tMid, stats)) return;
          // done with the previous frame's bitmap
          prevFrame.bmp.close && prevFrame.bmp.close();
        }
        // queue the current real frame (already a clean RGB bitmap). Keep it as
        // prev for the next tween; the scheduler owns/closes the queued copy, so
        // enqueue a clone-equivalent: we enqueue cur.bmp and keep a separate grab
        // for prev (handled by the grab loop passing fresh bitmaps each time).
        this._enqueue(cur.bmp, cur.ts);
        stats.framesOut++;
        prevFrame = { bmp: cur.prevBmp, ts: cur.ts }; // prevBmp = second copy for lookahead

        if (now - stats.lastReport > 2000) {
          const elapsed = (now - stats.started) / 1000;
          const fpsIn = stats.framesIn / elapsed;
          const fpsOut = stats.framesOut / elapsed;
          const mode = (rife && rife.isReady()) ? `RIFE infer=${(stats.lastInferMs||0).toFixed(1)}ms max=${(stats.maxInferMs||0).toFixed(1)}ms` : "blend";
          log(`interp: in=${fpsIn.toFixed(1)}fps out=${fpsOut.toFixed(1)}fps ${mode} maxGap=${stats.maxGapMs.toFixed(0)}ms stutters=${stats.stutters || 0}`);
          stats.lastReport = now;
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
      this._rvfcId = video.requestVideoFrameCallback(grabLoop);
      // Wait until the pipeline decision is finalized (GPU present vs CPU) before
      // processing ANY frame — otherwise early CPU frames grab a 2D context and block
      // the WebGPU-present config (the canvas can't have both).
      if (!this._pipelineReady) return;
      try {
        const vw = video.videoWidth, vh = video.videoHeight;
        if (vw && vh) {
          // GPU-RESIDENT PATH: capture, pack, inference, composite, queue, and
          // presentation all stay on the shared device. It is selected only when
          // the WebGPU canvas was configured successfully; otherwise the CPU path
          // below performs one clean grab and bitmap presentation.
          if (this._gpuPresent && this._rifeMod && this._rifeMod.gpuActive()) {
            const ts = Math.round((meta && meta.mediaTime != null ? meta.mediaTime : video.currentTime) * 1e6);
            // A decoder resolution change invalidates every queued/retained pair,
            // regardless of the selected interpolation engine. Flush before the
            // first new-size capture so an old _pipePrev/_pendingPair can never be
            // paired with the new GpuInterp allocation generation.
            if (this._lastVW !== vw || this._lastVH !== vh) {
              const firstSight = this._lastVW == null;
              this._lastVW = vw; this._lastVH = vh;
              if (!firstSight) {
                // INVERTED: a mid-play source-resolution switch (YouTube
                // reattachOnConstraint) must NOT be ridden hot — queued old-dims
                // textures would interleave with new-dims frames through shared
                // pool/luma/model allocations (the 1080p storm). Clean restart via
                // the proven stop→start path; everything re-derives coherently.
                if (this._chainInverted) {
                  this._scheduleDimsRestart(generation, vw, vh);
                  return;
                }
                log(`interp: source resized to ${vw}x${vh} — flushing old-size scheduler state`);
                this._flush && this._flush();
              }
              // A new cost profile should retry RIFE and re-measure unless the
              // user explicitly selected blend.
              if (!this._forceBlend && this._rifeMod.gpuRifeCapable && this._rifeMod.gpuRifeCapable()) {
                this._interpMode = "rife"; this._fallbackArmed = true;
                this._srcFrameBase = null;
              }
            }
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
                  return; // already re-registered at top
                }
                this._tapStaleSince = 0;
              } else {
                realTex = this._rifeMod.gpuCapture(video);
              }
              if (!this._rifeMod.gpuHasPrev()) { this._rifeMod.gpuAdvance(); if (realTex) this._enqueueTex(realTex, ts); this._prevTs = ts; stats.framesIn++; }
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
            }
          } else {
          // CPU path awaits (readback, createImageBitmap): with rvfc re-registered at
          // the top of the loop, ticks can overlap — serialize them (skip overlapped
          // frames) so shared staging buffers/canvases aren't used concurrently.
          if (cpuTickBusy) return;
          cpuTickBusy = true;
          try {
          if (grabCanvas.width !== vw || grabCanvas.height !== vh) {
            grabCanvas.width = vw; grabCanvas.height = vh;
          }
          // CLEAN CPU PATH: WebGPU importExternalTexture → readback → putImageData.
          let got = false;
          const gpuGrab = this._gpuGrab;
          if (gpuGrab && gpuGrab.ready) {
            const rgba = await gpuGrab.grab(video);
            if (!this._isCurrent(generation)) return;
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
          if (!this._isCurrent(generation)) { bmp.close?.(); prevBmp.close?.(); return; }
          const ts = Math.round((meta && meta.mediaTime != null ? meta.mediaTime : video.currentTime) * 1e6);
          await processFrame({ bmp, prevBmp, ts, w: vw, h: vh });
          } finally { cpuTickBusy = false; }
          }
        }
      } catch (e) {
        if (!this._stopped) this.warn("interp grab error:", e.message);
      }
    };
    this._rvfcId = video.requestVideoFrameCallback(grabLoop);

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
    Object.assign(this.overlay.style, {
      position: "fixed", pointerEvents: "none", zIndex: "2147483646", background: "#000",
    });
    document.body.appendChild(this.overlay);
    // Context selection is deferred: the .then() above claims a WebGPU context for
    // present if the GPU path activates. A canvas can't have both context types, so
    // we must NOT grab 2D synchronously (it would block WebGPU). The present loop
    // lazily grabs a 2D context if, once RIFE has settled, GPU-present isn't active.
    this.position();
    this._present(); // start the rAF loop (no-ops until a context is ready)

    // hide the original (visibility keeps it decoding + audio playing)
    this._origVisibility = video.style.visibility;
    video.style.visibility = "hidden";

    // route audio through a DelayNode so it can be held back to match the video
    // buffer latency (keeps A/V in sync; adaptive to measured video lag).
    this._setupAudioDelay(video);

    this._onScroll = () => this.position();
    window.addEventListener("scroll", this._onScroll, { passive: true, capture: true });
    window.addEventListener("resize", this._onScroll, { passive: true });

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
      if (this._pipePrev) { try { this._rifeMod && this._rifeMod.gpuRelease(this._pipePrev.tex); } catch {} this._pipePrev = null; }
      if (this._pendingPair) { try { this._rifeMod.gpuRelease(this._pendingPair.prev.tex); this._rifeMod.gpuRelease(this._pendingPair.cur.tex); } catch {} this._pendingPair = null; }
      this._lastPresentedTs = null;
      this._videoLatencyMs = null; // force a fresh measurement
    };
    // Re-assert persisted prefs on EVERY start: whatever restart path brought us
    // here (watchdog, video swap, toggle restart), the source of truth is main's
    // per-site prefs, not this instance's last life.
    if (this.chain && this.chain.ladder) this._ladderOn = !!this.chain.ladder();
    this._onSeeking = () => this._flush();
    this._onPlay = () => this._flush();
    video.addEventListener("seeking", this._onSeeking);
    video.addEventListener("seeked", this._onSeeking);
    video.addEventListener("play", this._onPlay);
    video.addEventListener("playing", this._onPlay);

    const pipelineOk = await pipelinePromise;
    if (this._pipelineInitPromise === pipelinePromise) this._pipelineInitPromise = null;
    if (!this._isCurrent(generation)) return { ok: false, reason: "cancelled" };
    if (!pipelineOk) return { ok: false, reason: "pipeline-unavailable" };
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

  stop() {
    // A queued inverted-dimension callback belongs to the lifecycle being stopped.
    // Reset even for an already-idle instance so it can never suppress a later run.
    this._dimsRestarting = false;
    const active = this._state !== "idle" || this.running || this.video || this.overlay || this.queue;
    if (!active) return { ok: true, stopped: false };
    ++this._lifecycleGen; // invalidate module imports, inference continuations, and rvfc
    this.running = false;
    this._state = "idle";
    this._stopped = true; // ends the requestVideoFrameCallback grab loop
    try { this.abort?.abort(); } catch {}
    if (this.video && this._rvfcId != null && typeof this.video.cancelVideoFrameCallback === "function") {
      try { this.video.cancelVideoFrameCallback(this._rvfcId); } catch {}
    }
    this._rvfcId = null;
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
    if (this._onScroll) {
      window.removeEventListener("scroll", this._onScroll, { capture: true });
      window.removeEventListener("resize", this._onScroll);
    }
    if (this.video && this._onSeeking) {
      this.video.removeEventListener("seeking", this._onSeeking);
      this.video.removeEventListener("seeked", this._onSeeking);
      this.video.removeEventListener("play", this._onPlay);
      this.video.removeEventListener("playing", this._onPlay);
    }
    if (this.video) this.video.style.visibility = this._origVisibility || "";
    try { this._gpuGrab && this._gpuGrab.destroy(); } catch {}
    this._gpuGrab = null;
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
      try { this.chain && this.chain.setInverted && this.chain.setInverted(false); } catch {}
      this._chainInverted = false;
      if (this.overlay) this.overlay.style.display = "";
    }
    this._chainActive = false; this._lastTapFrame = null; this._lastTapW = 0; this._lastTapH = 0;
    this._inferBusy = false; this._flushGen = (this._flushGen || 0) + 1;
    this._gpuInterpActive = false; this._gpuPresent = false;
    this._octx = null;            // stale 2D ref would block WebGPU-present on restart
    this._pipelineReady = false;
    this._teardownAudioDelay();
    this._lastEnqTs = null; this._targetInterval = 0;
    if (this.overlay) { this.overlay.remove(); this.overlay = null; }
    this.processor = null;
    this.abort = null;
    this.video = null;
    this._onScroll = null; this._onSeeking = null; this._onPlay = null;
    this.log("interpolation stopped");
    return { ok: true, stopped: true };
  }

  getStats() {
    const s = this.stats;
    const elapsed = s.started ? (performance.now() - s.started) / 1000 : 0;
    return {
      running: this.running,
      framesIn: s.framesIn,
      framesOut: s.framesOut,
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

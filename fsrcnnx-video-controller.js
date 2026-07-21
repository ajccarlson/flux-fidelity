// Explicit ownership for one HTMLVideoElement's callbacks and DOM listeners.
// Keeping the owner object identity in every callback prevents a stale callback
// from an SPA-replaced video from clearing or driving a newer video's lifecycle.

function callable(value, fallback = null) {
  return typeof value === "function" ? value : fallback;
}

// Web-platform methods such as setTimeout() and getComputedStyle() are branded
// in extension isolated worlds: invoking one as a property of a controller
// supplies the controller as `this` and Chromium rejects it with an "Illegal
// invocation" TypeError. Bind only host-provided defaults to their Window.
// Explicit adapters stay untouched so deterministic tests and embedders retain
// their existing callback semantics.
function windowCallable(value, ownerWindow, name, fallback = null) {
  if (value !== undefined) return callable(value, fallback);
  let host = ownerWindow;
  let method = host?.[name];
  if (typeof method !== "function") {
    host = globalThis;
    method = host?.[name];
  }
  return typeof method === "function" ? method.bind(host) : fallback;
}

function boundedDelay(value, fallback, maximum) {
  let numeric;
  try { numeric = Number(value); } catch { return fallback; }
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(maximum, Math.max(0, numeric));
}

// Multiple video owners can legitimately render into the same player
// container. Keep the temporary positioning change until the final owner lets
// go, and only restore it if the page has not replaced our value meanwhile.
const positionedParentLeases = new WeakMap();

function acquirePositionedParent(element, computedStyle) {
  const existing = positionedParentLeases.get(element);
  if (existing) {
    existing.count++;
    return existing;
  }
  let position = "static";
  try { position = computedStyle(element)?.position || "static"; } catch {}
  if (position !== "static" || !element?.style) return null;
  const lease = { count: 1, element, original: element.style.position || "" };
  try { element.style.position = "relative"; }
  catch { return null; }
  positionedParentLeases.set(element, lease);
  return lease;
}

function releasePositionedParent(lease) {
  if (!lease?.element) return;
  const current = positionedParentLeases.get(lease.element);
  if (current !== lease) return;
  current.count--;
  if (current.count > 0) return;
  positionedParentLeases.delete(current.element);
  if (!current.element.style || current.element.style.position !== "relative") return;
  try {
    if (current.original) current.element.style.position = current.original;
    else if (typeof current.element.style.removeProperty === "function") {
      current.element.style.removeProperty("position");
    } else {
      current.element.style.position = "";
    }
  } catch {}
}

export class VideoController {
  constructor(video, {
    onFrame,
    onLayout,
    onHoverChange,
    onSourceChange,
    resolveHoverRegion,
    window: ownerWindow = globalThis.window,
    document: ownerDocument = globalThis.document,
    ResizeObserver: ResizeObserverClass = globalThis.ResizeObserver,
    MutationObserver: MutationObserverClass = globalThis.MutationObserver,
    requestAnimationFrame: requestFrame,
    cancelAnimationFrame: cancelFrame,
    setTimeout: setTimer,
    clearTimeout: clearTimer,
    getComputedStyle: computedStyle,
  } = {}) {
    if (!video) throw new TypeError("VideoController requires a video element");
    this.video = video;
    this.onFrame = callable(onFrame, () => {});
    this.onLayout = callable(onLayout, () => {});
    this.onHoverChange = callable(onHoverChange, () => {});
    this.onSourceChange = callable(onSourceChange, () => {});
    this.resolveHoverRegion = callable(resolveHoverRegion, (element) => element.parentElement || element);
    this.window = ownerWindow || null;
    this.document = ownerDocument || null;
    this.ResizeObserverClass = callable(ResizeObserverClass);
    this.MutationObserverClass = callable(MutationObserverClass);
    this.requestFrame = windowCallable(requestFrame, ownerWindow, "requestAnimationFrame");
    this.cancelFrame = windowCallable(cancelFrame, ownerWindow, "cancelAnimationFrame", () => {});
    this.setTimer = windowCallable(setTimer, ownerWindow, "setTimeout",
      (callback) => { callback(); return null; });
    this.clearTimer = windowCallable(clearTimer, ownerWindow, "clearTimeout", () => {});
    this.computedStyle = windowCallable(computedStyle, ownerWindow, "getComputedStyle",
      () => ({ position: "static" }));
    this.active = false;
    this._frame = null;
    this._resizeObserver = null;
    this._mutationObserver = null;
    this._fullscreenTimer = null;
    this._hoverRegion = null;
    this._hoverEnter = null;
    this._hoverLeave = null;
    this._positionPatch = null;
    this._positioningRequested = false;
    this._positionTarget = null;
    this._observedParent = undefined;
    this._layoutCallback = () => {
      if (!this.active) return;
      this._rebindParent();
      this._refreshHoverRegion();
      this.onLayout(this);
    };
    this._fullscreenCallback = () => {
      if (!this.active) return;
      if (this._fullscreenTimer != null) {
        try { this.clearTimer(this._fullscreenTimer.id); } catch {}
      }
      const pending = { id: null };
      this._fullscreenTimer = pending;
      try {
        pending.id = this.setTimer(() => {
          if (this._fullscreenTimer !== pending) return;
          this._fullscreenTimer = null;
          this._layoutCallback();
        }, 50);
      } catch {
        if (this._fullscreenTimer !== pending) return;
        this._fullscreenTimer = null;
        this._layoutCallback();
      }
    };
    this._sourceCallback = (event) => {
      if (!this.active) return;
      try { this.onSourceChange(this, event); } catch {}
    };
  }

  start() {
    if (this.active) return this;
    this.active = true;
    this._rebindParent();
    this.window?.addEventListener?.("resize", this._layoutCallback, { passive: true });
    this.window?.addEventListener?.("scroll", this._layoutCallback, { passive: true, capture: true });
    this.document?.addEventListener?.("fullscreenchange", this._fullscreenCallback);
    for (const type of ["loadstart", "emptied", "loadedmetadata"]) {
      this.video.addEventListener?.(type, this._sourceCallback);
    }
    this._refreshHoverRegion();
    this._layoutCallback();
    return this;
  }

  scheduleFrame() {
    if (!this.active || this._frame) return false;
    const pending = { kind: null, id: null };
    this._frame = pending;
    const callback = (...args) => {
      if (!this.active || this._frame !== pending) return;
      this._frame = null;
      this.onFrame(this, ...args);
    };
    try {
      if (typeof this.video.requestVideoFrameCallback === "function") {
        pending.kind = "video";
        pending.id = this.video.requestVideoFrameCallback(callback);
      } else {
        if (!this.requestFrame) throw new Error("requestAnimationFrame is unavailable");
        pending.kind = "animation";
        pending.id = this.requestFrame(callback);
      }
      return true;
    } catch (error) {
      if (this._frame === pending) this._frame = null;
      throw error;
    }
  }

  cancelScheduledFrame() {
    const pending = this._frame;
    if (!pending) return false;
    this._frame = null;
    try {
      if (pending.kind === "video" && typeof this.video.cancelVideoFrameCallback === "function") {
        this.video.cancelVideoFrameCallback(pending.id);
      } else if (pending.kind === "animation") {
        this.cancelFrame(pending.id);
      }
    } catch {}
    return true;
  }

  ensurePositionedParent(parent) {
    if (!parent) return false;
    this._positioningRequested = true;
    this._positionTarget = parent;
    if (this._positionPatch?.element === parent) return true;
    this._releasePositionedParent();
    this._positionPatch = acquirePositionedParent(parent, this.computedStyle);
    return !!this._positionPatch;
  }

  _releasePositionedParent() {
    const lease = this._positionPatch;
    this._positionPatch = null;
    releasePositionedParent(lease);
  }

  _resetResizeObserver(parent) {
    try { this._resizeObserver?.disconnect?.(); } catch {}
    this._resizeObserver = null;
    if (!this.active || !this.ResizeObserverClass) return;
    try {
      const observer = new this.ResizeObserverClass(this._layoutCallback);
      this._resizeObserver = observer;
      try { observer.observe(this.video); } catch {}
      try { if (parent) observer.observe(parent); } catch {}
    } catch {
      this._resizeObserver = null;
    }
  }

  _resetMutationObserver(parent) {
    try { this._mutationObserver?.disconnect?.(); } catch {}
    this._mutationObserver = null;
    if (!this.active || !this.MutationObserverClass || !parent) return;
    try {
      const observer = new this.MutationObserverClass(this._layoutCallback);
      this._mutationObserver = observer;
      observer.observe(this.video, {
        attributes: true,
        attributeFilter: ["class", "style", "src"],
      });
      observer.observe(parent, {
        attributes: true,
        childList: true,
        attributeFilter: ["class", "style"],
      });
    } catch {
      try { this._mutationObserver?.disconnect?.(); } catch {}
      this._mutationObserver = null;
    }
  }

  _rebindParent() {
    const next = this.video.parentElement || null;
    if (this._observedParent === next) return false;
    const previous = this._observedParent === undefined ? null : this._observedParent;
    this._observedParent = next;
    this._resetResizeObserver(next);
    this._resetMutationObserver(next);
    if (this._positioningRequested && this._positionTarget === previous) {
      this._releasePositionedParent();
      this._positionTarget = next;
      if (next) this._positionPatch = acquirePositionedParent(next, this.computedStyle);
    }
    return true;
  }

  _refreshHoverRegion() {
    let next = null;
    try { next = this.resolveHoverRegion(this.video) || null; } catch {}
    if (next === this._hoverRegion) return;
    this._detachHoverRegion();
    if (!next?.addEventListener) return;
    this._hoverRegion = next;
    this._hoverEnter = () => {
      if (this.active) {
        try { this.onHoverChange(true, this); } catch {}
      }
    };
    this._hoverLeave = () => {
      if (this.active) {
        try { this.onHoverChange(false, this); } catch {}
      }
    };
    next.addEventListener("pointerenter", this._hoverEnter, { passive: true });
    next.addEventListener("pointermove", this._hoverEnter, { passive: true });
    next.addEventListener("pointerleave", this._hoverLeave, { passive: true });
  }

  _detachHoverRegion() {
    const region = this._hoverRegion;
    if (region?.removeEventListener) {
      if (this._hoverEnter) {
        region.removeEventListener("pointerenter", this._hoverEnter);
        region.removeEventListener("pointermove", this._hoverEnter);
      }
      if (this._hoverLeave) region.removeEventListener("pointerleave", this._hoverLeave);
    }
    this._hoverRegion = null;
    this._hoverEnter = null;
    this._hoverLeave = null;
    try { this.onHoverChange(false, this); } catch {}
  }

  destroy() {
    if (!this.active && !this._frame && !this._resizeObserver && !this._mutationObserver &&
        !this._hoverRegion && !this._positionPatch && !this._positioningRequested &&
        this._fullscreenTimer == null) return false;
    this.active = false;
    this.cancelScheduledFrame();
    try { this._resizeObserver?.disconnect?.(); } catch {}
    try { this._mutationObserver?.disconnect?.(); } catch {}
    this._resizeObserver = null;
    this._mutationObserver = null;
    this.window?.removeEventListener?.("resize", this._layoutCallback);
    this.window?.removeEventListener?.("scroll", this._layoutCallback, { capture: true });
    this.document?.removeEventListener?.("fullscreenchange", this._fullscreenCallback);
    for (const type of ["loadstart", "emptied", "loadedmetadata"]) {
      this.video.removeEventListener?.(type, this._sourceCallback);
    }
    if (this._fullscreenTimer != null) {
      try { this.clearTimer(this._fullscreenTimer.id); } catch {}
      this._fullscreenTimer = null;
    }
    this._detachHoverRegion();
    this._releasePositionedParent();
    this._positioningRequested = false;
    this._positionTarget = null;
    this._observedParent = undefined;
    return true;
  }
}

// Watches the document for SPA media replacement and candidate-ranking changes.
// Selection itself remains application-owned, while this object owns every
// observer/listener/timer used to request reconciliation.
export class VideoSelectionMonitor {
  constructor({
    select,
    onSelection,
    onError,
    delayMs = 50,
    reconcileMs = 2000,
    window: ownerWindow = globalThis.window,
    document: ownerDocument = globalThis.document,
    MutationObserver: MutationObserverClass = globalThis.MutationObserver,
    setTimeout: setTimer,
    clearTimeout: clearTimer,
    setInterval: setRepeatingTimer,
    clearInterval: clearRepeatingTimer,
  } = {}) {
    if (typeof select !== "function" || typeof onSelection !== "function") {
      throw new TypeError("VideoSelectionMonitor requires select and onSelection callbacks");
    }
    this.select = select;
    this.onSelection = onSelection;
    this.onError = callable(onError, () => {});
    this.delayMs = boundedDelay(delayMs, 50, 1000);
    const requestedReconcileMs = boundedDelay(reconcileMs, 2000, 60000);
    this.reconcileMs = requestedReconcileMs === 0 ? 0 : Math.max(1000, requestedReconcileMs);
    this.window = ownerWindow || null;
    this.document = ownerDocument || null;
    this.MutationObserverClass = callable(MutationObserverClass);
    this.setTimer = windowCallable(setTimer, ownerWindow, "setTimeout",
      (callback) => { callback(); return null; });
    this.clearTimer = windowCallable(clearTimer, ownerWindow, "clearTimeout", () => {});
    this.setRepeatingTimer = windowCallable(setRepeatingTimer, ownerWindow, "setInterval");
    this.clearRepeatingTimer = windowCallable(clearRepeatingTimer, ownerWindow, "clearInterval", () => {});
    this.active = false;
    this.current = null;
    this._observer = null;
    this._timer = null;
    this._reconcileTimer = null;
    this._navigation = null;
    this._selectionAttempt = 0;
    this._eventCallback = () => this.request();
  }

  start(current = this.current) {
    this.setCurrent(current);
    if (this.active) {
      this.request();
      return this;
    }
    this.active = true;
    const root = this.document?.documentElement || this.document?.body;
    if (this.MutationObserverClass && root) {
      try {
        const observer = new this.MutationObserverClass(this._eventCallback);
        this._observer = observer;
        observer.observe(root, {
          attributes: true,
          childList: true,
          subtree: true,
          attributeFilter: ["class", "style", "src"],
        });
      } catch (error) {
        try { this._observer?.disconnect?.(); } catch {}
        this._observer = null;
        this._reportError(error, { phase: "observe", monitor: this });
      }
    }
    for (const type of ["play", "pause", "ended", "loadstart", "loadedmetadata", "emptied"]) {
      this.document?.addEventListener?.(type, this._eventCallback, { capture: true });
    }
    this.document?.addEventListener?.("visibilitychange", this._eventCallback);
    this.window?.addEventListener?.("resize", this._eventCallback, { passive: true });
    this.window?.addEventListener?.("scroll", this._eventCallback, { passive: true, capture: true });
    for (const type of ["popstate", "hashchange", "pageshow"]) {
      this.window?.addEventListener?.(type, this._eventCallback);
    }
    const navigation = this.window?.navigation;
    if (navigation?.addEventListener) {
      this._navigation = navigation;
      navigation.addEventListener("currententrychange", this._eventCallback);
      navigation.addEventListener("navigate", this._eventCallback);
    }
    if (this.reconcileMs > 0 && this.setRepeatingTimer) {
      // A document observer cannot see an open shadow root attached to an
      // already-present host. This bounded fallback lets the application-level
      // selector rediscover those videos without patching page history APIs.
      const pending = { id: null };
      this._reconcileTimer = pending;
      try {
        pending.id = this.setRepeatingTimer(this._eventCallback, this.reconcileMs);
      } catch (error) {
        if (this._reconcileTimer === pending) this._reconcileTimer = null;
        this._reportError(error, { phase: "reconcile-schedule", monitor: this });
      }
    }
    this.request();
    return this;
  }

  request() {
    if (!this.active || this._timer != null) return false;
    const pending = { id: null };
    this._timer = pending;
    try {
      pending.id = this.setTimer(() => {
        if (this._timer !== pending) return;
        this._timer = null;
        this._scan();
      }, this.delayMs);
    } catch (error) {
      this._timer = null;
      this._reportError(error, { phase: "scan-schedule", monitor: this });
      return false;
    }
    return true;
  }

  _scan() {
    if (!this.active) return;
    let candidate;
    try {
      candidate = this.select() || null;
    } catch (error) {
      this._reportError(error, { phase: "select", monitor: this });
      return;
    }
    const previous = this.current;
    const changed = candidate !== previous;
    // Record the observed identity immediately so repeated route signals
    // coalesce. A rejected latest handoff rolls it back; an obsolete rejection
    // cannot overwrite a newer successful observation.
    this.current = candidate;
    const attempt = ++this._selectionAttempt;
    let result;
    try {
      result = this.onSelection(candidate, previous, changed, this);
    } catch (error) {
      this._selectionFailed(error, { attempt, candidate, previous });
      return;
    }
    Promise.resolve(result).catch((error) => {
      this._selectionFailed(error, { attempt, candidate, previous });
    });
  }

  _selectionFailed(error, { attempt, candidate, previous }) {
    if (!this.active || attempt !== this._selectionAttempt) return;
    if (this.current === candidate) this.current = previous;
    this._reportError(error, {
      phase: "selection",
      candidate,
      previous,
      monitor: this,
    });
  }

  _reportError(error, context) {
    try {
      const result = this.onError(error, context);
      Promise.resolve(result).catch(() => {});
    } catch {}
  }

  setCurrent(current) {
    this.current = current || null;
    this._selectionAttempt++;
  }

  stop() {
    if (!this.active && !this._observer && this._timer == null && this._reconcileTimer == null) return false;
    this.active = false;
    this._selectionAttempt++;
    try { this._observer?.disconnect?.(); } catch {}
    this._observer = null;
    if (this._timer != null) {
      try { this.clearTimer(this._timer.id); } catch {}
      this._timer = null;
    }
    if (this._reconcileTimer != null) {
      try { this.clearRepeatingTimer(this._reconcileTimer.id); } catch {}
      this._reconcileTimer = null;
    }
    for (const type of ["play", "pause", "ended", "loadstart", "loadedmetadata", "emptied"]) {
      this.document?.removeEventListener?.(type, this._eventCallback, { capture: true });
    }
    this.document?.removeEventListener?.("visibilitychange", this._eventCallback);
    this.window?.removeEventListener?.("resize", this._eventCallback);
    this.window?.removeEventListener?.("scroll", this._eventCallback, { capture: true });
    for (const type of ["popstate", "hashchange", "pageshow"]) {
      this.window?.removeEventListener?.(type, this._eventCallback);
    }
    this._navigation?.removeEventListener?.("currententrychange", this._eventCallback);
    this._navigation?.removeEventListener?.("navigate", this._eventCallback);
    this._navigation = null;
    this.current = null;
    return true;
  }
}

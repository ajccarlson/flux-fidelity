import test from "node:test";
import assert from "node:assert/strict";

import { VideoController, VideoSelectionMonitor } from "../fsrcnnx-video-controller.js";

function eventTarget() {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, callback) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(callback);
    },
    removeEventListener(type, callback) { listeners.get(type)?.delete(callback); },
    emit(type, event = {}) { for (const callback of [...(listeners.get(type) || [])]) callback(event); },
  };
}

function videoElement(parent = null) {
  const target = eventTarget();
  let nextId = 0;
  const callbacks = new Map();
  return {
    ...target,
    parentElement: parent,
    callbacks,
    cancelled: [],
    requestVideoFrameCallback(callback) { const id = ++nextId; callbacks.set(id, callback); return id; },
    cancelVideoFrameCallback(id) { this.cancelled.push(id); callbacks.delete(id); },
    fire(id, ...args) { callbacks.get(id)?.(...args); },
  };
}

class FakeObserver {
  constructor(callback) { this.callback = callback; this.observed = []; this.disconnected = false; }
  observe(value) { this.observed.push(value); }
  disconnect() { this.disconnected = true; }
}

function styledParent(position = "") {
  return {
    ...eventTarget(),
    style: {
      position,
      removeProperty(name) { if (name === "position") this.position = ""; },
    },
  };
}

function fakeTimers() {
  let nextId = 0;
  const timers = new Map();
  const intervals = new Map();
  return {
    timers,
    intervals,
    setTimeout(callback) { const id = ++nextId; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    setInterval(callback) { const id = ++nextId; intervals.set(id, callback); return id; },
    clearInterval(id) { intervals.delete(id); },
    runTimer() {
      const entry = timers.entries().next().value;
      assert.ok(entry, "expected a pending one-shot timer");
      timers.delete(entry[0]);
      entry[1]();
    },
  };
}

test("VideoController gives each video exclusive ownership of its scheduled callback", () => {
  const parentA = { ...eventTarget(), style: { position: "" } };
  const parentB = { ...eventTarget(), style: { position: "" } };
  const videoA = videoElement(parentA);
  const videoB = videoElement(parentB);
  const frames = [];
  const common = {
    window: eventTarget(), document: eventTarget(),
    ResizeObserver: FakeObserver, MutationObserver: FakeObserver,
    getComputedStyle: () => ({ position: "static" }),
  };
  const first = new VideoController(videoA, { ...common, onFrame: (owner) => frames.push(owner.video) }).start();
  const second = new VideoController(videoB, { ...common, onFrame: (owner) => frames.push(owner.video) }).start();
  assert.equal(first.scheduleFrame(), true);
  const staleId = first._frame.id;
  assert.equal(second.scheduleFrame(), true);
  const currentId = second._frame.id;

  first.destroy();
  videoA.fire(staleId);
  assert.deepEqual(frames, []);
  assert.deepEqual(videoA.cancelled, [staleId]);

  videoB.fire(currentId);
  assert.deepEqual(frames, [videoB]);
  assert.equal(second._frame, null);
  second.destroy();
});

test("VideoController cleans observers, hover listeners, timers, and parent style patches", () => {
  const ownerWindow = eventTarget();
  const ownerDocument = eventTarget();
  const parent = { ...eventTarget(), style: {
    position: "",
    removeProperty(name) { if (name === "position") this.position = ""; },
  } };
  const video = videoElement(parent);
  const timers = new Map();
  let nextTimer = 0;
  const hover = [];
  const controller = new VideoController(video, {
    window: ownerWindow,
    document: ownerDocument,
    ResizeObserver: FakeObserver,
    MutationObserver: FakeObserver,
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    getComputedStyle: () => ({ position: "static" }),
    onHoverChange: (value) => hover.push(value),
  }).start();

  assert.equal(controller.ensurePositionedParent(parent), true);
  assert.equal(parent.style.position, "relative");
  parent.emit("pointerenter");
  assert.equal(hover.at(-1), true);
  ownerDocument.emit("fullscreenchange");
  assert.equal(timers.size, 1);
  const resizeObserver = controller._resizeObserver;
  const mutationObserver = controller._mutationObserver;

  controller.destroy();
  assert.equal(resizeObserver.disconnected, true);
  assert.equal(mutationObserver.disconnected, true);
  assert.equal(timers.size, 0);
  assert.equal(parent.style.position, "");
  assert.equal(hover.at(-1), false);
  assert.equal(ownerWindow.listeners.get("resize")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("fullscreenchange")?.size || 0, 0);
});

test("VideoController owns direct media-source boundary listeners", () => {
  const video = videoElement(styledParent());
  const boundaries = [];
  const controller = new VideoController(video, {
    window: eventTarget(),
    document: eventTarget(),
    ResizeObserver: null,
    MutationObserver: null,
    onSourceChange: (owner, event) => boundaries.push([owner.video, event.type]),
  }).start();

  for (const type of ["loadstart", "emptied", "loadedmetadata"]) video.emit(type, { type });
  assert.deepEqual(boundaries, [
    [video, "loadstart"], [video, "emptied"], [video, "loadedmetadata"],
  ]);
  controller.destroy();
  video.emit("loadstart", { type: "loadstart" });
  assert.equal(boundaries.length, 3, "destroy removes every source-boundary listener");
});

test("VideoController reference-counts a position lease shared by multiple owners", () => {
  const parent = styledParent();
  const common = {
    window: eventTarget(),
    document: eventTarget(),
    ResizeObserver: null,
    MutationObserver: null,
    getComputedStyle: () => ({ position: "static" }),
  };
  const first = new VideoController(videoElement(parent), common).start();
  const second = new VideoController(videoElement(parent), common).start();

  assert.equal(first.ensurePositionedParent(parent), true);
  assert.equal(second.ensurePositionedParent(parent), true);
  assert.equal(parent.style.position, "relative");
  assert.equal(first.destroy(), true);
  assert.equal(first.destroy(), false, "destroy should be idempotent");
  assert.equal(parent.style.position, "relative", "the remaining owner keeps the lease active");
  assert.equal(second.destroy(), true);
  assert.equal(parent.style.position, "", "the last owner restores the original inline value");

  const externallyChanged = styledParent();
  const third = new VideoController(videoElement(externallyChanged), common).start();
  third.ensurePositionedParent(externallyChanged);
  externallyChanged.style.position = "absolute";
  third.destroy();
  assert.equal(externallyChanged.style.position, "absolute", "page-owned style changes are preserved");
});

test("VideoController rebinds observers, hover ownership, and its position lease after reparenting", () => {
  const parentA = styledParent();
  const parentB = styledParent();
  const video = videoElement(parentA);
  const hover = [];
  let layouts = 0;
  const controller = new VideoController(video, {
    window: eventTarget(),
    document: eventTarget(),
    ResizeObserver: FakeObserver,
    MutationObserver: FakeObserver,
    getComputedStyle: () => ({ position: "static" }),
    onLayout: () => { layouts++; },
    onHoverChange: (value) => hover.push(value),
  }).start();
  controller.ensurePositionedParent(parentA);
  const resizeA = controller._resizeObserver;
  const mutationA = controller._mutationObserver;
  assert.deepEqual(resizeA.observed, [video, parentA]);
  assert.deepEqual(mutationA.observed, [video, parentA]);

  video.parentElement = parentB;
  mutationA.callback();
  assert.equal(resizeA.disconnected, true);
  assert.equal(mutationA.disconnected, true);
  assert.notEqual(controller._resizeObserver, resizeA);
  assert.notEqual(controller._mutationObserver, mutationA);
  assert.deepEqual(controller._resizeObserver.observed, [video, parentB]);
  assert.deepEqual(controller._mutationObserver.observed, [video, parentB]);
  assert.equal(parentA.style.position, "");
  assert.equal(parentB.style.position, "relative");
  assert.ok(layouts >= 2, "reparenting should trigger a fresh layout");

  const beforeOldHover = hover.length;
  parentA.emit("pointerenter");
  assert.equal(hover.length, beforeOldHover, "the old parent no longer owns hover events");
  parentB.emit("pointerenter");
  assert.equal(hover.at(-1), true);

  controller.destroy();
  assert.equal(parentB.style.position, "");
  assert.equal(parentB.listeners.get("pointerenter")?.size || 0, 0);
});

test("VideoController teardown survives faulty consumer callbacks and timer adapters", () => {
  const ownerWindow = eventTarget();
  const ownerDocument = eventTarget();
  const parent = styledParent();
  const controller = new VideoController(videoElement(parent), {
    window: ownerWindow,
    document: ownerDocument,
    ResizeObserver: FakeObserver,
    MutationObserver: FakeObserver,
    setTimeout: () => 7,
    clearTimeout: () => { throw new Error("host timer teardown failed"); },
    getComputedStyle: () => ({ position: "static" }),
    onHoverChange: () => { throw new Error("consumer hover callback failed"); },
  });

  assert.doesNotThrow(() => controller.start());
  controller.ensurePositionedParent(parent);
  parent.emit("pointerenter");
  ownerDocument.emit("fullscreenchange");
  assert.doesNotThrow(() => controller.destroy());
  assert.equal(parent.style.position, "");
  assert.equal(ownerWindow.listeners.get("resize")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("fullscreenchange")?.size || 0, 0);
});

test("VideoController falls back to rAF and cancels it through the same owner", () => {
  const video = videoElement();
  delete video.requestVideoFrameCallback;
  delete video.cancelVideoFrameCallback;
  const callbacks = new Map();
  const cancelled = [];
  let nextId = 0;
  const controller = new VideoController(video, {
    requestAnimationFrame(callback) { const id = ++nextId; callbacks.set(id, callback); return id; },
    cancelAnimationFrame(id) { cancelled.push(id); callbacks.delete(id); },
  }).start();
  controller.scheduleFrame();
  const id = controller._frame.id;
  controller.destroy();
  callbacks.get(id)?.();
  assert.deepEqual(cancelled, [id]);
});

test("VideoController binds default Window methods to their native receiver", () => {
  const ownerWindow = eventTarget();
  const ownerDocument = eventTarget();
  const parent = styledParent();
  const video = videoElement(parent);
  delete video.requestVideoFrameCallback;
  delete video.cancelVideoFrameCallback;
  const animationFrames = new Map();
  const timers = new Map();
  const cancelledFrames = [];
  const clearedTimers = [];
  let nextId = 0;
  let styleCalls = 0;
  const requireWindow = (implementation) => function (...args) {
    assert.equal(this, ownerWindow, "the native default must receive its owning Window");
    return implementation(...args);
  };
  Object.assign(ownerWindow, {
    requestAnimationFrame: requireWindow((callback) => {
      const id = ++nextId;
      animationFrames.set(id, callback);
      return id;
    }),
    cancelAnimationFrame: requireWindow((id) => {
      cancelledFrames.push(id);
      animationFrames.delete(id);
    }),
    setTimeout: requireWindow((callback) => {
      const id = ++nextId;
      timers.set(id, callback);
      return id;
    }),
    clearTimeout: requireWindow((id) => {
      clearedTimers.push(id);
      timers.delete(id);
    }),
    getComputedStyle: requireWindow(() => {
      styleCalls++;
      return { position: "static" };
    }),
  });

  const controller = new VideoController(video, {
    window: ownerWindow,
    document: ownerDocument,
    ResizeObserver: null,
    MutationObserver: null,
  }).start();
  assert.equal(controller.ensurePositionedParent(parent), true);
  assert.equal(styleCalls, 1);
  assert.equal(parent.style.position, "relative");
  assert.equal(controller.scheduleFrame(), true);
  const frameId = controller._frame.id;
  ownerDocument.emit("fullscreenchange");
  const timerId = controller._fullscreenTimer.id;
  assert.equal(animationFrames.has(frameId), true);
  assert.equal(timers.has(timerId), true);

  controller.destroy();
  assert.deepEqual(cancelledFrames, [frameId]);
  assert.deepEqual(clearedTimers, [timerId]);
  assert.equal(animationFrames.size, 0);
  assert.equal(timers.size, 0);
  assert.equal(parent.style.position, "");
});

test("VideoSelectionMonitor coalesces SPA mutations and reports identity changes", async () => {
  const ownerWindow = eventTarget();
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const timers = new Map();
  let nextTimer = 0;
  let selected = { id: "first" };
  const selections = [];
  const monitor = new VideoSelectionMonitor({
    select: () => selected,
    onSelection: (candidate, previous, changed) => selections.push({ candidate, previous, changed }),
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    reconcileMs: 0,
  });
  monitor.start(selected);
  assert.equal(timers.size, 1);
  monitor._observer.callback();
  ownerWindow.emit("scroll");
  assert.equal(timers.size, 1, "multiple invalidations should share one scan");
  const run = [...timers.values()][0]; timers.clear(); run();
  assert.equal(selections.at(-1).changed, false);

  const second = { id: "second" };
  selected = second;
  ownerDocument.emit("pause");
  const runSecond = [...timers.values()][0]; timers.clear(); runSecond();
  assert.deepEqual(selections.at(-1), { candidate: second, previous: selections[0].candidate, changed: true });

  ownerDocument.emit("ended");
  assert.equal(timers.size, 1, "ended should re-rank playback candidates");
  timers.clear();

  monitor.stop();
  assert.equal(monitor._observer, null);
  assert.equal(ownerWindow.listeners.get("scroll")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("play")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("pause")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("ended")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("loadstart")?.size || 0, 0);
});

test("VideoSelectionMonitor listens for route and lifecycle changes and periodically reconciles open-shadow candidates", () => {
  const navigation = eventTarget();
  const ownerWindow = Object.assign(eventTarget(), { navigation });
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const clock = fakeTimers();
  const first = { id: "first" };
  const shadowCandidate = { id: "open-shadow-video" };
  let selected = first;
  const selections = [];
  const monitor = new VideoSelectionMonitor({
    select: () => selected,
    onSelection: (candidate, previous, changed) => selections.push({ candidate, previous, changed }),
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
    reconcileMs: 2500,
  }).start(first);

  assert.equal(clock.intervals.size, 1, "only one low-frequency reconciliation loop is owned");
  clock.runTimer();
  for (const [target, type] of [
    [ownerWindow, "popstate"],
    [ownerWindow, "hashchange"],
    [ownerWindow, "pageshow"],
    [ownerDocument, "visibilitychange"],
    [navigation, "currententrychange"],
    [navigation, "navigate"],
  ]) {
    target.emit(type);
    assert.equal(clock.timers.size, 1, `${type} should request a scan`);
    clock.runTimer();
  }

  selected = shadowCandidate;
  [...clock.intervals.values()][0]();
  assert.equal(clock.timers.size, 1, "periodic reconciliation should request a bounded scan");
  monitor._observer.callback();
  assert.equal(clock.timers.size, 1, "periodic and mutation invalidations should coalesce");
  clock.runTimer();
  assert.deepEqual(selections.at(-1), { candidate: shadowCandidate, previous: first, changed: true });

  assert.equal(monitor.start(shadowCandidate), monitor, "start is idempotent while active");
  assert.equal(clock.intervals.size, 1);
  assert.equal(ownerWindow.listeners.get("hashchange")?.size || 0, 1);
  assert.equal(navigation.listeners.get("navigate")?.size || 0, 1);
  monitor.stop();
  assert.equal(clock.intervals.size, 0);
  assert.equal(clock.timers.size, 0);
  assert.equal(ownerWindow.listeners.get("hashchange")?.size || 0, 0);
  assert.equal(ownerWindow.listeners.get("pageshow")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("visibilitychange")?.size || 0, 0);
  assert.equal(navigation.listeners.get("currententrychange")?.size || 0, 0);
  assert.equal(navigation.listeners.get("navigate")?.size || 0, 0);
  assert.equal(monitor.stop(), false, "stop should be idempotent");
});

test("VideoSelectionMonitor binds default Window timers and their clear methods", () => {
  const ownerWindow = eventTarget();
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const timers = new Map();
  const intervals = new Map();
  const clearedTimers = [];
  const clearedIntervals = [];
  let nextId = 0;
  const requireWindow = (implementation) => function (...args) {
    assert.equal(this, ownerWindow, "the native default must receive its owning Window");
    return implementation(...args);
  };
  Object.assign(ownerWindow, {
    setTimeout: requireWindow((callback) => {
      const id = ++nextId;
      timers.set(id, callback);
      return id;
    }),
    clearTimeout: requireWindow((id) => {
      clearedTimers.push(id);
      timers.delete(id);
    }),
    setInterval: requireWindow((callback) => {
      const id = ++nextId;
      intervals.set(id, callback);
      return id;
    }),
    clearInterval: requireWindow((id) => {
      clearedIntervals.push(id);
      intervals.delete(id);
    }),
  });

  const monitor = new VideoSelectionMonitor({
    select: () => null,
    onSelection: () => {},
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    reconcileMs: 2000,
  }).start();
  const timerId = monitor._timer.id;
  const intervalId = monitor._reconcileTimer.id;
  assert.equal(timers.has(timerId), true);
  assert.equal(intervals.has(intervalId), true);

  monitor.stop();
  assert.deepEqual(clearedTimers, [timerId]);
  assert.deepEqual(clearedIntervals, [intervalId]);
  assert.equal(timers.size, 0);
  assert.equal(intervals.size, 0);
});

test("VideoSelectionMonitor reports selection failures, rolls back only the latest attempt, and recovers", async () => {
  const ownerWindow = eventTarget();
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const clock = fakeTimers();
  const previous = { id: "previous" };
  const candidate = { id: "candidate" };
  const finalCandidate = { id: "final" };
  let selected = candidate;
  let fail = true;
  let throwFromSelect = false;
  const attempts = [];
  const errors = [];
  const monitor = new VideoSelectionMonitor({
    select() {
      if (throwFromSelect) throw new Error("selection probe failed");
      return selected;
    },
    onSelection(next, prior, changed) {
      attempts.push({ next, prior, changed });
      return fail ? Promise.reject(new Error("handoff failed")) : undefined;
    },
    onError: (error, context) => errors.push({ message: error.message, phase: context.phase }),
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
    reconcileMs: 2000,
  }).start(previous);

  clock.runTimer();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(monitor.current, previous, "a rejected latest handoff restores the accepted selection");
  assert.deepEqual(errors, [{ message: "handoff failed", phase: "selection" }]);

  fail = false;
  [...clock.intervals.values()][0]();
  clock.runTimer();
  await Promise.resolve();
  assert.equal(monitor.current, candidate);
  assert.equal(attempts.at(-1).changed, true, "the failed identity is retried on reconciliation");

  throwFromSelect = true;
  ownerWindow.emit("pageshow");
  clock.runTimer();
  assert.equal(monitor.current, candidate, "a failed probe must not look like candidate removal");
  assert.deepEqual(errors.at(-1), { message: "selection probe failed", phase: "select" });

  throwFromSelect = false;
  selected = finalCandidate;
  ownerWindow.emit("hashchange");
  clock.runTimer();
  assert.equal(monitor.current, finalCandidate);
  monitor.stop();
});

test("VideoSelectionMonitor ignores a stale rejection after a newer selection succeeds", async () => {
  const ownerWindow = eventTarget();
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const clock = fakeTimers();
  const original = { id: "original" };
  const firstCandidate = { id: "first-candidate" };
  const latestCandidate = { id: "latest-candidate" };
  let selected = firstCandidate;
  let rejectFirst;
  const errors = [];
  const monitor = new VideoSelectionMonitor({
    select: () => selected,
    onSelection(candidate) {
      if (candidate === firstCandidate) {
        return new Promise((_resolve, reject) => { rejectFirst = reject; });
      }
      return undefined;
    },
    onError: (error) => errors.push(error.message),
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: clock.setInterval,
    clearInterval: clock.clearInterval,
    reconcileMs: 2000,
  }).start(original);

  clock.runTimer();
  assert.equal(monitor.current, firstCandidate);
  selected = latestCandidate;
  ownerWindow.emit("hashchange");
  clock.runTimer();
  assert.equal(monitor.current, latestCandidate);

  rejectFirst(new Error("obsolete handoff failed"));
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(monitor.current, latestCandidate);
  assert.deepEqual(errors, [], "an obsolete failure must not disturb or alarm the current owner");
  monitor.stop();
});

test("VideoSelectionMonitor stop releases observers and listeners even if timer adapters throw", () => {
  const navigation = eventTarget();
  const ownerWindow = Object.assign(eventTarget(), { navigation });
  const ownerDocument = { ...eventTarget(), documentElement: {} };
  const monitor = new VideoSelectionMonitor({
    select: () => null,
    onSelection: () => {},
    window: ownerWindow,
    document: ownerDocument,
    MutationObserver: FakeObserver,
    setTimeout: () => 11,
    clearTimeout: () => { throw new Error("one-shot teardown failed"); },
    setInterval: () => 12,
    clearInterval: () => { throw new Error("interval teardown failed"); },
  }).start();
  const observer = monitor._observer;

  assert.doesNotThrow(() => monitor.stop());
  assert.equal(observer.disconnected, true);
  assert.equal(ownerWindow.listeners.get("popstate")?.size || 0, 0);
  assert.equal(ownerDocument.listeners.get("play")?.size || 0, 0);
  assert.equal(navigation.listeners.get("navigate")?.size || 0, 0);
  assert.equal(monitor.stop(), false);
});

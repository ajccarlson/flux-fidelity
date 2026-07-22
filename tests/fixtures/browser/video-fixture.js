const API_VERSION = 1;
const LOAD_TIMEOUT_MS = 10_000;

const CLIPS = Object.freeze({
  "bt709-a": Object.freeze({
    path: "media/bt709-a.webm",
    dimensions: Object.freeze({ width: 160, height: 90 }),
    colorSpace: Object.freeze({
      primaries: "bt709",
      transfer: "bt709",
      matrix: "bt709",
      fullRange: false,
    }),
  }),
  "bt709-b": Object.freeze({
    path: "media/bt709-b.webm",
    dimensions: Object.freeze({ width: 128, height: 72 }),
    colorSpace: Object.freeze({
      primaries: "bt709",
      transfer: "bt709",
      matrix: "bt709",
      fullRange: false,
    }),
  }),
  "bt2020-pq": Object.freeze({
    path: "media/bt2020-pq.webm",
    dimensions: Object.freeze({ width: 128, height: 72 }),
    colorSpace: Object.freeze({
      primaries: "bt2020",
      transfer: "pq",
      matrix: "bt2020-ncl",
      fullRange: false,
    }),
  }),
  "bt2020-sdr": Object.freeze({
    path: "media/bt2020-sdr.webm",
    dimensions: Object.freeze({ width: 128, height: 72 }),
    colorSpace: Object.freeze({
      primaries: "bt2020",
      transfer: "bt709",
      matrix: "bt2020-ncl",
      fullRange: false,
    }),
  }),
});

const video = document.getElementById("fixture-video");
const status = document.getElementById("fixture-status");
const lifecycleEvents = [];
let currentKey = null;
let sourceGeneration = 0;
let initialReady = null;

function finite(value) {
  return Number.isFinite(value) ? value : null;
}

function sourceChangedError() {
  return new DOMException("A newer fixture source replaced this load", "AbortError");
}

function assertCurrent(generation) {
  if (generation !== sourceGeneration) throw sourceChangedError();
}

function mediaError() {
  const code = Number(video.error?.code) || 0;
  return new Error(code ? `Video fixture failed with MediaError code ${code}` : "Video fixture failed to load");
}

function waitForLoadedData(generation, expectedUrl, activate) {
  return new Promise((resolve, reject) => {
    let timer = null;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
    };
    const settle = (callback, value) => {
      cleanup();
      callback(value);
    };
    const onLoaded = () => {
      try {
        assertCurrent(generation);
        if (video.currentSrc !== expectedUrl) return;
        settle(resolve);
      } catch (error) {
        settle(reject, error);
      }
    };
    const onError = () => {
      try {
        assertCurrent(generation);
        if (video.currentSrc !== expectedUrl) return;
        settle(reject, mediaError());
      } catch (error) {
        settle(reject, error);
      }
    };
    video.addEventListener("loadeddata", onLoaded);
    video.addEventListener("error", onError);
    timer = setTimeout(() => settle(
      reject,
      new Error(`Video fixture did not reach loadeddata within ${LOAD_TIMEOUT_MS} ms`),
    ), LOAD_TIMEOUT_MS);
    // Install the listeners before replacing the resource. A generic `abort`
    // event is deliberately not authoritative here because it can belong to the
    // resource retired by this exact activation; generation checks on the next
    // loaded/error event and exact currentSrc distinguish superseded calls without
    // rejecting the new one.
    try { activate(); }
    catch (error) { settle(reject, error); }
  });
}

function waitForPresentedFrame(generation) {
  if (typeof video.requestVideoFrameCallback !== "function") {
    return Promise.reject(new Error("requestVideoFrameCallback is unavailable"));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    let callbackId = null;
    const timer = setTimeout(() => finish(
      reject,
      new Error(`Video fixture did not present a frame within ${LOAD_TIMEOUT_MS} ms`),
    ), LOAD_TIMEOUT_MS);
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (callback === reject && callbackId != null &&
          typeof video.cancelVideoFrameCallback === "function") {
        try { video.cancelVideoFrameCallback(callbackId); } catch {}
      }
      callback(value);
    };
    try {
      callbackId = video.requestVideoFrameCallback((_now, metadata) => {
        try {
          assertCurrent(generation);
          finish(resolve, {
            mediaTime: finite(metadata?.mediaTime),
            presentedFrames: finite(metadata?.presentedFrames),
          });
        } catch (error) {
          finish(reject, error);
        }
      });
    } catch (error) {
      finish(reject, error);
    }
  });
}

function decodedColorSpace() {
  if (typeof VideoFrame !== "function") {
    return { available: false, error: "VideoFrame is unavailable" };
  }
  let frame = null;
  try {
    frame = new VideoFrame(video);
    const value = frame.colorSpace?.toJSON?.() || frame.colorSpace || {};
    return {
      available: true,
      primaries: value.primaries ?? null,
      transfer: value.transfer ?? null,
      matrix: value.matrix ?? null,
      fullRange: typeof value.fullRange === "boolean" ? value.fullRange : null,
    };
  } catch (error) {
    return {
      available: false,
      error: error?.name || "VideoFrameError",
    };
  } finally {
    try { frame?.close?.(); } catch {}
  }
}

function snapshot() {
  const clip = currentKey ? CLIPS[currentKey] : null;
  const rect = video.getBoundingClientRect();
  const playback = video.getVideoPlaybackQuality?.();
  return {
    apiVersion: API_VERSION,
    key: currentKey,
    source: clip?.path || null,
    currentSrc: video.currentSrc || "",
    dimensions: {
      width: Number(video.videoWidth) || 0,
      height: Number(video.videoHeight) || 0,
    },
    expectedDimensions: clip ? { ...clip.dimensions } : null,
    colorSpace: decodedColorSpace(),
    expectedColorSpace: clip ? { ...clip.colorSpace } : null,
    readyState: Number(video.readyState),
    networkState: Number(video.networkState),
    paused: video.paused,
    ended: video.ended,
    muted: video.muted,
    currentTime: finite(video.currentTime),
    duration: finite(video.duration),
    rect: {
      left: finite(rect.left),
      top: finite(rect.top),
      width: finite(rect.width),
      height: finite(rect.height),
    },
    playback: playback ? {
      totalVideoFrames: finite(playback.totalVideoFrames),
      droppedVideoFrames: finite(playback.droppedVideoFrames),
    } : null,
    visibilityState: document.visibilityState,
    lifecycleEvents: lifecycleEvents.map((event) => ({ ...event })),
    sourceGeneration,
  };
}

async function play() {
  const generation = sourceGeneration;
  assertCurrent(generation);
  video.muted = true;
  await video.play();
  assertCurrent(generation);
  await waitForPresentedFrame(generation);
  assertCurrent(generation);
  return snapshot();
}

function pause() {
  video.pause();
  return Promise.resolve().then(snapshot);
}

async function loadSource(key) {
  const clip = CLIPS[key];
  if (!clip) throw new RangeError(`Unknown video fixture source: ${String(key)}`);
  const generation = ++sourceGeneration;
  currentKey = key;
  status.textContent = `Loading ${key}…`;
  video.pause();
  video.muted = true;
  const expectedUrl = new URL(clip.path, document.baseURI).href;
  await waitForLoadedData(generation, expectedUrl, () => {
    video.src = clip.path;
    video.load();
  });
  assertCurrent(generation);
  await video.play();
  assertCurrent(generation);
  await waitForPresentedFrame(generation);
  assertCurrent(generation);
  const state = snapshot();
  status.textContent = `Ready: ${key} (${state.dimensions.width}×${state.dimensions.height})`;
  return state;
}

function recordLifecycle(type, event) {
  lifecycleEvents.push({
    type,
    persisted: typeof event?.persisted === "boolean" ? event.persisted : null,
    visibilityState: document.visibilityState,
    at: Math.round(performance.now()),
  });
  if (lifecycleEvents.length > 32) lifecycleEvents.shift();
}

document.addEventListener("visibilitychange", (event) => recordLifecycle("visibilitychange", event));
document.addEventListener("freeze", (event) => recordLifecycle("freeze", event));
document.addEventListener("resume", (event) => recordLifecycle("resume", event));
window.addEventListener("pagehide", (event) => recordLifecycle("pagehide", event));
window.addEventListener("pageshow", (event) => recordLifecycle("pageshow", event));

const api = Object.freeze({
  ready: () => initialReady,
  loadSource,
  pause,
  play,
  snapshot,
});

Object.defineProperty(window, "__FSRCNNX_VIDEO_FIXTURE__", {
  configurable: false,
  enumerable: false,
  writable: false,
  value: api,
});

initialReady = loadSource("bt709-a");

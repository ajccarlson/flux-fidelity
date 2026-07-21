// Thin content-script shim. Content scripts can't be ES modules directly, but
// they can dynamic-import() a web-accessible module. We load the pipeline module
// and relay popup messages to its exported API.

let api = null;
const ready = (async () => {
  const url = chrome.runtime.getURL("fsrcnnx-main.js");
  api = await import(url);
  // restore saved per-site preferences (settings + saved mode)
  try { await api.restoreSitePrefs(); } catch {}
  console.log("[FSRCNNX] module imported into content script");
})().catch((e) => console.error("[FSRCNNX] module import failed:", e));

chrome.runtime.onMessage.addListener((msg, _sender, send) => {
  if (msg?.type === "FSRCNNX_SETMODE") {
    ready.then(() => api.setMode(msg.mode)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_RESTORE") {
    ready.then(() => api.restoreSitePrefs()).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETENGINE") {
    ready.then(() => api.setEngine(msg.engine)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETNEURALMODEL") {
    ready.then(() => api.setNeuralModel(msg.model)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETARTVARIANT") {
    ready.then(() => api.setArtVariant(msg.variant)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPOLATE") {
    ready.then(() => api.setInterpolate(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPRES") {
    ready.then(() => api.setInterpolateRes(msg.mode)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPAVOFFSET") {
    ready.then(() => api.setInterpolateAvOffset(msg.ms)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPMODEL") {
    ready.then(() => api.setInterpolateModel(msg.key)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPTARGETFPS") {
    ready.then(() => api.setInterpolateTargetFps(msg.value)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETLADDER") {
    ready.then(() => api.setInterpolateLadder(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETAUTOFALLBACK") {
    ready.then(() => api.setInterpolateAutoFallback(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINVERT") {
    ready.then(() => api.setInterpolateInvert(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETINTERPDIAG") {
    ready.then(() => api.setInterpolateDiag(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETIMAGES") {
    ready.then(() => api.setImages(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETDEBAND") {
    ready.then(() => api.setDeband(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETDEBANDSTR") {
    ready.then(() => api.setDebandStrength(msg.strength)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETHOVERREVEAL") {
    ready.then(() => api.setHoverReveal(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETALLVIDEOS") {
    ready.then(() => api.setAllVideos(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETSHARPEN") {
    ready.then(() => api.setSharpen(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETSHARPENSTR") {
    ready.then(() => api.setSharpenStrength(msg.strength)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETSSIMDS") {
    ready.then(() => api.setSSimDS(msg.on)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_SETPOLICY") {
    ready.then(() => api.setPolicy(msg.policy)).then(send);
    return true;
  }
  if (msg?.type === "FSRCNNX_STATUS") {
    if (!api) { send({ mode: "off", hasVideo: false, webgpu: "gpu" in navigator, frameCount: 0, loading: true }); return false; }
    send(api.getStatus());
    return false;
  }
});

const $ = (id) => document.getElementById(id);
function setBool(el, v, ok="yes", no="no"){ el.textContent = v?ok:no; el.className = "v "+(v?"ok":"no"); }

async function tab(){ const [t] = await chrome.tabs.query({active:true,currentWindow:true}); return t; }
async function send(type, extra={}){
  const t = await tab();
  if(!t || !/^https?:\/\//.test(t.url||"")) return {error:"unsupported-page"};
  try { return await chrome.tabs.sendMessage(t.id, {type, ...extra}); }
  catch { return {error:"no-cs"}; }
}

async function refresh(){
  const st = await send("FSRCNNX_STATUS");
  if(st?.error){ $("s-webgpu").textContent = st.error==="unsupported-page"?"open a video page":"reload tab"; $("s-webgpu").className="v no"; return; }
  setBool($("s-webgpu"), st.webgpu);
  setBool($("s-video"), st.hasVideo);
  // DRM/protected source: show a clear state and disable the mode buttons' effect.
  const banner = $("drm-banner");
  if(banner){
    if(st.protected){
      banner.style.display="block";
      banner.textContent = st.protectedReason === "tainted"
        ? "This video is served cross-origin without CORS headers, so the browser blocks reading its pixels — upscaling isn't possible here."
        : "This video appears DRM-protected — its frames can't be read, so upscaling is unavailable here.";
    }
    else banner.style.display="none";
  }
  $("s-model").textContent = st.model ? (st.engine==="artcnn" ? st.model.replace("ArtCNN_","ArtCNN ") : `${st.scale}× ${st.model.replace("FSRCNNX_","").replace("_16-0-4-1","").replace("_56-16-4-1"," high")}`) : "—";
  $("s-model").className = "v " + (st.model?"ok":"");
  $("s-frames").textContent = st.frameCount ?? 0;
  if(st.engine && $("engine").value !== st.engine){ $("engine").value = st.engine; const isArt=st.engine==="artcnn"; $("artvariant").style.display = isArt?"block":"none"; $("neuralrow").style.display = st.engine==="neural"?"block":"none"; buildPolicyOptions(st.engine, st.policy); }
  if (Array.isArray(st.neuralModels)) {
    const sel = $("neural-model");
    if (sel.options.length !== Math.max(1, st.neuralModels.length)) {
      sel.innerHTML = st.neuralModels.length
        ? st.neuralModels.map(m=>`<option value="${m.key}">${m.label} (${m.scale}x)</option>`).join("")
        : `<option value="">no models — see tools/neural-export</option>`;
    }
    if (st.neural && st.neural.model && sel.value !== st.neural.model) sel.value = st.neural.model;
    $("neural-note").textContent = st.engine==="neural" ? (st.neural && st.neural.ready ? `\u03bc${(st.neural.mu||0).toFixed(1)}ms  skip:${st.neural.skip||0}` : "initializing\u2026") : "";
  }
  if (st.engine === "neural" && st.neural) $("s-model").textContent = `${st.neural.scale||"?"}\u00d7 ${st.neural.label||st.neural.model||"neural"}`;
  if(st.artVariant && $("artvariant").value !== st.artVariant) $("artvariant").value = st.artVariant;
  if(st.policy && $("policy").value !== st.policy && [...$("policy").options].some(o=>o.value===st.policy)) $("policy").value = st.policy;
  for(const b of document.querySelectorAll(".modes button")) b.dataset.active = (b.dataset.mode === st.mode) ? "1" : "0";
  if(typeof st.ssimds === "boolean") $("ssimds").checked = st.ssimds;
  if(typeof st.sharpen === "boolean"){ $("sharpen").checked = st.sharpen; $("sharpen-row").style.display = st.sharpen ? "block" : "none"; }
  if(typeof st.deband === "boolean"){ $("deband").checked = st.deband; $("deband-row").style.display = st.deband ? "block" : "none"; }
  if(typeof st.debandStrength === "number"){ $("deband-str").value = st.debandStrength; $("deband-val").textContent = st.debandStrength.toFixed(1); }
  if(typeof st.hoverReveal === "boolean") $("hover-reveal").checked = st.hoverReveal;
  if(typeof st.interpAutoFallback === "boolean") $("interp-autofallback").checked = st.interpAutoFallback;
  if(typeof st.interpLadder === "boolean") $("interp-ladder").checked = st.interpLadder;
  if(typeof st.interpInvert === "boolean") $("interp-invert").checked = st.interpInvert;
  if(typeof st.allVideos === "boolean") $("all-videos").checked = st.allVideos;
  if($("multi-count")) $("multi-count").textContent = (st.allVideos && st.multiCount) ? `(${st.multiCount} active)` : "";
  if(typeof st.images === "boolean") $("images").checked = st.images;
  if($("image-count")) $("image-count").textContent = (st.images && st.imageCount) ? `(${st.imageCount} done)` : "";
  if(typeof st.interpolate === "boolean"){ $("interpolate").checked = st.interpolate; $("interp-res-row").style.display = st.interpolate ? "block" : "none"; }
  if(st.interpStats && Array.isArray(st.interpStats.models) && st.interpStats.models.length && document.activeElement !== $("interp-model")){
    const sel = $("interp-model");
    const want = st.interpStats.models.map(m=>m.key).join(",");
    if(sel._sig !== want){
      sel.innerHTML = st.interpStats.models.map(m=>`<option value="${m.key}">${m.label}</option>`).join("");
      sel._sig = want;
    }
    const cur = st.interpStats.models.find(m=>m.current);
    if(cur && sel.value !== cur.key) sel.value = cur.key;
  }
  if(st.interpStats && st.interpStats.resMode && $("interp-res").value !== st.interpStats.resMode) $("interp-res").value = st.interpStats.resMode;
  if(st.interpStats && st.interpStats.targetFps != null && document.activeElement !== $("interp-target")){
    const tv = String(st.interpStats.targetFps);
    if($("interp-target").value !== tv) $("interp-target").value = tv;
    const hz = st.interpStats.detectedHz;
    $("interp-target-hz").textContent = st.interpStats.targetFps === "auto"
      ? (hz ? `detected ${hz}Hz` : "detecting…")
      : (st.interpStats.effectiveTargetFps ? `→ ${st.interpStats.effectiveTargetFps} fps` : "");
  }
  if(st.interpStats && typeof st.interpStats.avOffsetMs === "number" && document.activeElement !== $("interp-avoff")){ $("interp-avoff").value = st.interpStats.avOffsetMs; $("interp-avoff-val").textContent = st.interpStats.avOffsetMs; }
  if($("interp-stats")) $("interp-stats").textContent = (st.interpolate && st.interpStats) ? `(in ${st.interpStats.fpsIn} → out ${st.interpStats.fpsOut} fps, ${st.interpStats.rife ? `${st.interpStats.interpMode === "blend" ? `BLEND @100%→${st.interpStats.effectiveTargetFps}fps${st.interpStats.chain ? " ⛓upscaled" : ""}` : `RIFE ${st.interpStats.inferMs}ms${st.interpStats.inferMeanMs ? ` (μ${st.interpStats.inferMeanMs})` : ""} @${Math.round((st.interpStats.scale||1)*100)}%`}${st.interpStats.gpuPath ? " gpu" : ""}${st.interpStats.fp16 ? " fp16" : ""}${st.interpStats.capture ? " gc" : ""}${st.interpStats.jspi ? " jspi" : ""}${st.interpStats.inverted ? " \u26d3inv" : ""}${(st.interpStats.interpMode !== "blend" && st.interpStats.timing) ? ` [pre ${st.interpStats.timing.pre.toFixed(1)}/inf ${st.interpStats.timing.infer.toFixed(1)}/post ${st.interpStats.timing.post.toFixed(1)}]` : ""}${st.interpStats.skippedTweens ? ` skip:${st.interpStats.skippedTweens} (${st.interpStats.skipRate}/s)` : ""}${st.interpStats.ladderBlends ? ` lad:${st.interpStats.ladderBlends}` : ""}` : `blend${st.interpStats.rifeError ? " — "+st.interpStats.rifeError : ""}`}${st.interpStats.audioDelayMs!=null ? `, a/v sync ${st.interpStats.audioDelayMs}ms` : ""}, gap ${st.interpStats.maxGapMs}ms)` : "";
  if(typeof st.sharpenStrength === "number"){ $("sharpen-str").value = st.sharpenStrength; $("sharpen-val").textContent = st.sharpenStrength.toFixed(1); }
}

const POLICY_OPTS = {
  fsrcnnx: [
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv WHEN thresholds)"],
    ["force2", "Always ×2"],
    ["force3", "Always ×3"],
    ["force4", "Always ×4"],
  ],
  "fsrcnnx-hi": [
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv WHEN thresholds)"],
    ["force2", "Always ×2"],
    ["force4", "Always ×4"],
    ["force8", "Always ×8"],
  ],
  artcnn: [
    ["display", "Source below display (recommended)"],
    ["auto", "Auto (mpv WHEN thresholds)"],
    ["force2", "Always ×2"],
    ["force4", "Always ×4"],
    ["force8", "Always ×8"],
  ],
};

function buildPolicyOptions(engine, selected) {
  const sel = $("policy");
  const opts = POLICY_OPTS[engine] || POLICY_OPTS.fsrcnnx;
  sel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if (selected && opts.some(([v]) => v === selected)) sel.value = selected;
}

$("engine").addEventListener("change", async () => {
  const e = $("engine").value;
  $("neuralrow").style.display = e==="neural"?"block":"none";
  const isArt = e === "artcnn";
  $("artvariant").style.display = isArt ? "block" : "none";
  await send("FSRCNNX_SETENGINE", { engine: e });
  // rebuild policy options for the new engine; if current policy isn't valid for
  // the new engine (e.g. force3 on a 2x-only engine), fall back to display.
  const cur = $("policy").value;
  buildPolicyOptions(e, cur);
  // ensure the module's policy matches what's now shown
  await send("FSRCNNX_SETPOLICY", { policy: $("policy").value });
  setTimeout(refresh, 150);
});
$("artvariant").addEventListener("change", async () => {
  await send("FSRCNNX_SETARTVARIANT", { variant: $("artvariant").value });
  setTimeout(refresh, 150);
});

$("policy").addEventListener("change", async () => {
  await send("FSRCNNX_SETPOLICY", { policy: $("policy").value });
  setTimeout(refresh, 150);
});

$("ssimds").addEventListener("change", async () => {
  await send("FSRCNNX_SETSSIMDS", { on: $("ssimds").checked });
  setTimeout(refresh, 150);
});

$("sharpen").addEventListener("change", async () => {
  $("sharpen-row").style.display = $("sharpen").checked ? "block" : "none";
  await send("FSRCNNX_SETSHARPEN", { on: $("sharpen").checked });
  setTimeout(refresh, 150);
});

$("sharpen-str").addEventListener("input", async () => {
  $("sharpen-val").textContent = parseFloat($("sharpen-str").value).toFixed(1);
});
$("sharpen-str").addEventListener("change", async () => {
  await send("FSRCNNX_SETSHARPENSTR", { strength: parseFloat($("sharpen-str").value) });
  setTimeout(refresh, 150);
});

$("deband").addEventListener("change", async () => {
  $("deband-row").style.display = $("deband").checked ? "block" : "none";
  await send("FSRCNNX_SETDEBAND", { on: $("deband").checked });
  setTimeout(refresh, 150);
});
$("deband-str").addEventListener("input", () => {
  $("deband-val").textContent = parseFloat($("deband-str").value).toFixed(1);
});
$("deband-str").addEventListener("change", async () => {
  await send("FSRCNNX_SETDEBANDSTR", { strength: parseFloat($("deband-str").value) });
  setTimeout(refresh, 150);
});
$("interpolate").addEventListener("change", async () => {
  $("interp-res-row").style.display = $("interpolate").checked ? "block" : "none";
  await send("FSRCNNX_SETINTERPOLATE", { on: $("interpolate").checked });
  setTimeout(refresh, 200);
});
$("interp-res").addEventListener("change", async () => {
  await send("FSRCNNX_SETINTERPRES", { mode: $("interp-res").value });
  setTimeout(refresh, 200);
});
$("interp-avoff").addEventListener("input", () => {
  $("interp-avoff-val").textContent = $("interp-avoff").value;
});
$("interp-avoff").addEventListener("change", async () => {
  await send("FSRCNNX_SETINTERPAVOFFSET", { ms: Number($("interp-avoff").value) });
});
$("interp-diag").addEventListener("change", async () => {
  await send("FSRCNNX_SETINTERPDIAG", { on: $("interp-diag").checked });
});
$("interp-ladder").addEventListener("change", async () => {
  await send("FSRCNNX_SETLADDER", { on: $("interp-ladder").checked });
});
$("interp-autofallback").addEventListener("change", async () => {
  await send("FSRCNNX_SETAUTOFALLBACK", { on: $("interp-autofallback").checked });
});
$("interp-invert").addEventListener("change", async () => {
  $("interp-invert").disabled = true; // flip restarts the interpolator
  await send("FSRCNNX_SETINVERT", { on: $("interp-invert").checked });
  setTimeout(() => { $("interp-invert").disabled = false; refresh(); }, 400);
});
$("interp-model").addEventListener("change", async () => {
  $("interp-model").disabled = true;
  await send("FSRCNNX_SETINTERPMODEL", { key: $("interp-model").value });
  setTimeout(() => { $("interp-model").disabled = false; refresh(); }, 400);
});

$("interp-target").addEventListener("change", async () => {
  const v = $("interp-target").value;
  await send("FSRCNNX_SETINTERPTARGETFPS", { value: v === "auto" ? "auto" : Number(v) });
  setTimeout(refresh, 150);
});

$("images").addEventListener("change", async () => {
  await send("FSRCNNX_SETIMAGES", { on: $("images").checked });
  setTimeout(refresh, 150);
});

$("hover-reveal").addEventListener("change", async () => {
  await send("FSRCNNX_SETHOVERREVEAL", { on: $("hover-reveal").checked });
  setTimeout(refresh, 150);
});
$("all-videos").addEventListener("change", async () => {
  await send("FSRCNNX_SETALLVIDEOS", { on: $("all-videos").checked });
  setTimeout(refresh, 150);
});

for(const b of document.querySelectorAll(".modes button")){
  b.addEventListener("click", async () => {
    const res = await send("FSRCNNX_SETMODE", {mode: b.dataset.mode});
    if(res && res.ok === false){
      const banner = $("drm-banner");
      if(banner){
        banner.style.display = "block";
        banner.textContent =
          res.reason === "drm" ? "This video appears DRM-protected — its frames can't be read, so upscaling is unavailable here." :
          res.reason === "tainted" ? "This video is served cross-origin without CORS headers, so the browser blocks reading its pixels — upscaling isn't possible here." :
          res.reason === "no video" ? "No playable video found on this page yet. Start the video, then try again." :
          res.reason === "WebGPU init failed" ? "WebGPU couldn't initialize on this page." :
          "Couldn't enable on this source.";
      }
    }
    setTimeout(refresh, 200);
  });
}
buildPolicyOptions($("engine").value);
refresh(); setInterval(refresh, 1000);

$("neural-model").addEventListener("change", async () => {
  const k = $("neural-model").value;
  if (k) await send("FSRCNNX_SETNEURALMODEL", { model: k });
});

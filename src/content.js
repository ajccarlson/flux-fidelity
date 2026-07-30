// Thin content-script shim. Content scripts cannot be ES modules directly, so
// this file loads the web-accessible pipeline module and exposes only the
// commands used by the extension popup.

let api = null;
let startupPhase = "loading";
let startupError = null;
let restorePromise = null;
let restoreResult = null;

function errorMessage(error) {
  if (error && typeof error.message === "string" && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "Unknown extension error";
}

function restoreOnce() {
  if (!api) return Promise.reject(new Error("FSRCNNX module is not loaded"));
  if (!restorePromise) {
    restorePromise = Promise.resolve()
      .then(() => api.restoreSitePrefs())
      .then((result) => {
        restoreResult = result;
        return result;
      })
      // A rejection used to be memoized for the life of the document, so one
      // transient storage failure during first restore permanently poisoned
      // FSRCNNX_RESTORE with no way back. Clear the memo so a later attempt can
      // succeed, while still rejecting this caller.
      .catch((error) => {
        restorePromise = null;
        throw error;
      });
  }
  return restorePromise;
}

const moduleLoaded = (async () => {
  const url = chrome.runtime.getURL("src/core/fsrcnnx-main.js");
  api = await import(url);
  return api;
})();

const startup = moduleLoaded.then(async () => {
  // Prerendered and already-hidden documents must enter the renderer's
  // suspended state before restoring durable preferences. Otherwise restore
  // can allocate GPU resources for a document that is not yet eligible to own
  // the tab and may activate without a fresh script injection.
  if (requestedDocumentState === "hidden") {
    startEarlyHiddenDrain();
    if (earlyHiddenDrain) await earlyHiddenDrain;
    if (lifecycleFailureState === "hidden") throw new Error("initial suspension failed");
  }
  await restoreOnce();
  startupPhase = "ready";
  console.log("[FSRCNNX] module imported into content script");
  return { ok: true };
}).catch((error) => {
  startupPhase = "failed";
  startupError = error;
  console.error("[FSRCNNX] module initialization failed:", error);
  // Resolve with an explicit failure record so a failed import/restore never
  // becomes an unhandled rejection while no popup is open.
  return { ok: false, error };
});

async function loadedApi() {
  const result = await startup;
  if (!result.ok || !api) throw result.error || new Error("FSRCNNX module failed to load");
  return api;
}

function invalidInput(reason, field) {
  return { ok: false, error: "invalid-input", reason, field };
}

function validatePayloadShape(msg, fields) {
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(msg, field)) {
      return invalidInput(`Missing required field: ${field}`, field);
    }
  }
  const allowed = new Set(["type", ...fields]);
  const extra = Object.keys(msg).filter((field) => !allowed.has(field)).sort()[0];
  return extra === undefined ? null : invalidInput(`Unexpected field: ${extra}`, extra);
}

function noPayload(run) {
  return Object.freeze({
    mutates: false,
    validate: (msg) => validatePayloadShape(msg, []),
    run,
  });
}

function fieldPayload(field, accepts, expectation, run) {
  return Object.freeze({
    mutates: true,
    validate(msg) {
      const shapeError = validatePayloadShape(msg, [field]);
      if (shapeError) return shapeError;
      return accepts(msg[field]) ? null : invalidInput(`${field} ${expectation}`, field);
    },
    run,
  });
}

function booleanPayload(run) {
  return fieldPayload("on", (value) => typeof value === "boolean", "must be a boolean", run);
}

// These arrays must stay identical to fsrcnnx-setting-contract.js. They cannot
// simply import it: content scripts are not ES modules, and this gate has to
// validate commands that arrive before the pipeline module finishes loading, so
// it cannot depend on the module being present. tests/setting-contract.test.mjs
// fails if any of them drifts — that drift is what made the neural "native"
// policy unreachable through the UI.
function enumPayload(field, values, run) {
  const accepted = new Set(values);
  return fieldPayload(
    field,
    (value) => typeof value === "string" && accepted.has(value),
    `must be one of: ${values.join(", ")}`,
    run,
  );
}

function boundedNumberPayload(field, min, max, run) {
  return fieldPayload(
    field,
    (value) => typeof value === "number" && Number.isFinite(value) && value >= min && value <= max,
    `must be a finite number from ${min} to ${max}`,
    run,
  );
}

const COMMANDS = Object.freeze({
  FSRCNNX_SETMODE: enumPayload("mode", ["off", "passthrough", "upscale"],
    (module, msg) => module.setMode(msg.mode)),
  FSRCNNX_RESTORE: noPayload(() => restoreOnce().then(() => restoreResult)),
  FSRCNNX_SETENGINE: enumPayload("engine", ["fsrcnnx", "fsrcnnx-hi", "artcnn", "neural"],
    (module, msg) => module.setEngine(msg.engine)),
  FSRCNNX_SETNEURALMODEL: fieldPayload(
    "model",
    (value) => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value),
    "must be a safe model key",
    (module, msg) => module.setNeuralModel(msg.model),
  ),
  FSRCNNX_SETARTVARIANT: enumPayload(
    "variant",
    ["ArtCNN_C4F32", "ArtCNN_C4F32_DS", "ArtCNN_C4F32_DN"],
    (module, msg) => module.setArtVariant(msg.variant),
  ),
  FSRCNNX_SETINTERPOLATE: booleanPayload((module, msg) => module.setInterpolate(msg.on)),
  FSRCNNX_SETINTERPRES: enumPayload("mode", ["auto", "full", "half", "quarter"],
    (module, msg) => module.setInterpolateRes(msg.mode)),
  FSRCNNX_SETINTERPAVOFFSET: boundedNumberPayload("ms", -100, 300,
    (module, msg) => module.setInterpolateAvOffset(msg.ms)),
  FSRCNNX_SETINTERPMODEL: enumPayload(
    "key",
    ["rife_v4.26_fp16", "rife_v4.26", "blend"],
    (module, msg) => module.setInterpolateModel(msg.key),
  ),
  FSRCNNX_SETINTERPTARGETFPS: fieldPayload(
    "value",
    (value) => value === "auto" ||
      (typeof value === "number" && Number.isFinite(value) && value >= 24 && value <= 480),
    'must be "auto" or a finite number from 24 to 480',
    (module, msg) => module.setInterpolateTargetFps(msg.value),
  ),
  FSRCNNX_SETLADDER: booleanPayload((module, msg) => module.setInterpolateLadder(msg.on)),
  FSRCNNX_SETAUTOFALLBACK: booleanPayload((module, msg) => module.setInterpolateAutoFallback(msg.on)),
  FSRCNNX_SETINVERT: booleanPayload((module, msg) => module.setInterpolateInvert(msg.on)),
  FSRCNNX_SETINTERPDIAG: booleanPayload((module, msg) => module.setInterpolateDiag(msg.on)),
  FSRCNNX_SETIMAGES: booleanPayload((module, msg) => module.setImages(msg.on)),
  FSRCNNX_SETHOVERREVEAL: booleanPayload((module, msg) => module.setHoverReveal(msg.on)),
  FSRCNNX_SETALLVIDEOS: booleanPayload((module, msg) => module.setAllVideos(msg.on)),
  FSRCNNX_SETIDLEPOWERSAVING: booleanPayload((module, msg) => module.setIdlePowerSaving(msg.on)),
  FSRCNNX_SETAUTOQUALITYFALLBACK: booleanPayload(
    (module, msg) => module.setAutoQualityFallback(msg.on),
  ),
  FSRCNNX_SETSHARPEN: booleanPayload((module, msg) => module.setSharpen(msg.on)),
  FSRCNNX_SETSHARPENSTR: boundedNumberPayload("strength", 0.1, 2,
    (module, msg) => module.setSharpenStrength(msg.strength)),
  FSRCNNX_SETSSIMDS: booleanPayload((module, msg) => module.setSSimDS(msg.on)),
  FSRCNNX_SETPOLICY: enumPayload(
    "policy",
    ["display", "auto", "force2", "force3", "force4", "force8", "native"],
    (module, msg) => module.setPolicy(msg.policy),
  ),
});

function normalizeCommandResponse(result) {
  if (!result || typeof result !== "object" || typeof result.ok !== "boolean") {
    return {
      ok: false,
      error: "invalid-response",
      reason: "Command response must be an object with a boolean ok field",
    };
  }
  return result;
}

function baseStatus(extra = {}) {
  const webgpu = typeof navigator !== "undefined" && "gpu" in navigator;
  const loading = startupPhase === "loading";
  const failed = startupPhase === "failed";
  const runtimePhase = loading ? "loading" : failed ? "failed" : "idle";
  return {
    statusVersion: 1,
    mode: "off",
    activeMode: "off",
    hasVideo: false,
    webgpu,
    gpuState: webgpu ? "idle" : "unavailable",
    frameCount: 0,
    runtime: {
      phase: runtimePhase,
      api: webgpu ? "available" : "unavailable",
    },
    renderer: {
      phase: loading ? "loading" : failed ? "failed" : "off",
      requestedMode: "off",
      activeMode: "off",
    },
    loading,
    failed,
    ...extra,
  };
}

function preferenceSyncStatus(status) {
  if (preferenceSyncPhase === "pending" || preferenceSyncPhase === "syncing") {
    return {
      ...status,
      runtime: {
        ...(status.runtime || {}),
        phase: "syncing",
      },
      renderer: {
        ...(status.renderer || {}),
        phase: "suspended",
      },
    };
  }
  if (preferenceSyncPhase !== "failed") return status;
  return {
    ...status,
    loading: false,
    failed: true,
    error: "preference-sync-failed",
    reason: errorMessage(preferenceSyncError),
    runtime: {
      ...(status.runtime || {}),
      phase: "failed",
    },
    renderer: {
      ...(status.renderer || {}),
      phase: "suspended",
    },
  };
}

function statusSnapshot() {
  if (startupPhase === "loading") return baseStatus();
  if (startupPhase === "failed" || !api) {
    return baseStatus({
      loading: false,
      failed: true,
      error: "startup-failed",
      reason: errorMessage(startupError),
    });
  }
  try {
    const reported = api.getStatus();
    const current = reported && typeof reported === "object" ? reported : {};
    const fallback = baseStatus({ loading: false, failed: false });
    const webgpu = typeof current.webgpu === "boolean" ? current.webgpu : fallback.webgpu;
    return preferenceSyncStatus({
      ...fallback,
      ...current,
      statusVersion: Number.isInteger(current.statusVersion) ? current.statusVersion : 1,
      webgpu,
      gpuState: typeof current.gpuState === "string"
        ? current.gpuState
        : webgpu ? "idle" : "unavailable",
      runtime: {
        ...fallback.runtime,
        ...(current.runtime && typeof current.runtime === "object" ? current.runtime : {}),
      },
      renderer: {
        ...fallback.renderer,
        requestedMode: current.mode || fallback.renderer.requestedMode,
        activeMode: current.activeMode || fallback.renderer.activeMode,
        ...(current.renderer && typeof current.renderer === "object" ? current.renderer : {}),
      },
      loading: false,
      failed: false,
    });
  } catch (error) {
    return baseStatus({
      loading: false,
      failed: true,
      error: "status-failed",
      reason: errorMessage(error),
    });
  }
}

async function dispatch(msg) {
  if (msg.type === "FSRCNNX_STATUS") {
    return validatePayloadShape(msg, []) || statusSnapshot();
  }
  const command = COMMANDS[msg.type];
  const validationError = command.validate(msg);
  if (validationError) return validationError;
  const module = await loadedApi();
  if (command.mutates) {
    await waitForActivePreferenceSync();
    // BFCache activation owns one lifecycle sync, but mutations can also race
    // a live storage.onChanged application while the page remains active. A
    // fresh main-world barrier drains both physical storage and the stable
    // external-application tail before the popup publishes newer intent.
    if (typeof module.syncSitePrefs === "function") {
      const syncResult = await module.syncSitePrefs();
      if (!transitionSucceeded(syncResult)) {
        throw new Error(errorMessage(
          syncResult?.reason || syncResult?.error || "Site preferences could not be synchronized",
        ));
      }
    }
  }

  let result;
  let commandError = null;
  try {
    result = await command.run(module, msg);
  } catch (error) {
    commandError = error;
  }

  let flushError = null;
  if (command.mutates && typeof module.flushPreferenceWrites === "function") {
    try {
      const flushResult = await module.flushPreferenceWrites();
      if (!transitionSucceeded(flushResult)) {
        throw new Error(errorMessage(
          flushResult?.reason || flushResult?.error || "Preference writes could not be flushed",
        ));
      }
    } catch (error) {
      flushError = error;
    }
  }
  if (commandError) throw commandError;
  if (flushError) throw new Error(`Preference write flush failed: ${errorMessage(flushError)}`);
  return normalizeCommandResponse(result);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Not exploitable today: onMessage only fires for same-extension senders and the
  // manifest declares no externally_connectable, so a page cannot reach this. But
  // background.js verifies sender.id on its capability handlers, and this listener
  // drives every mutating command — the check costs one line and removes the
  // dependency on nobody ever adding externally_connectable.
  if (sender && sender.id !== chrome.runtime.id) return false;
  const type = msg && typeof msg === "object" ? msg.type : null;
  if (type !== "FSRCNNX_STATUS" && !Object.prototype.hasOwnProperty.call(COMMANDS, type)) {
    return false;
  }

  // Lifecycle events can be lost while a document is frozen. Any extension
  // interaction is a safe reconciliation point because visibility and
  // prerendering state are authoritative at the time the message arrives.
  const observedDocumentState = currentDocumentState();
  if (observedDocumentState !== requestedDocumentState) {
    requestDocumentState(observedDocumentState);
  }

  let responded = false;
  const respondOnce = (response) => {
    if (responded) return;
    responded = true;
    try { sendResponse(response); } catch {}
  };

  Promise.resolve()
    .then(() => dispatch(msg))
    .then(
      (result) => respondOnce(result),
      (error) => respondOnce({
        ok: false,
        error: startupPhase === "failed" ? "startup-failed" : "command-failed",
        reason: errorMessage(error),
      }),
    );
  return true;
});

function sendDocumentState(state, generation = documentStateGeneration) {
  try {
    const pending = chrome.runtime.sendMessage({
      type: "FSRCNNX_DOCUMENT",
      state,
      generation,
    });
    if (pending && typeof pending.catch === "function") return pending.catch(() => {});
  } catch {}
  return Promise.resolve();
}

function publishCurrentDocumentState() {
  if (!api || startupPhase !== "ready") return Promise.resolve();
  try {
    const status = api.getStatus();
    // `host` is deliberately omitted: background.js never read it, so publishing it
    // moved the visited hostname into the service worker on every navigation for no
    // consumer's benefit.
    const message = status.protected
      ? { type: "FSRCNNX_PROTECTED" }
      : {
          type: "FSRCNNX_STATE",
          mode: status.activeMode || "off",
          requestedMode: status.mode,
        };
    const pending = chrome.runtime.sendMessage(message);
    if (pending && typeof pending.catch === "function") return pending.catch(() => {});
  } catch {}
  return Promise.resolve();
}

let requestedDocumentState = null;
let appliedDocumentState = null;
let documentStateGeneration = 0;
let lifecycleDrain = null;
let earlyHiddenDrain = null;
let activeClaimRetry = null;
let activeTransitionInFlight = false;
let lifecycleFailureState = null;
let activePreferenceSync = null;
let preferenceSyncPhase = "idle";
let preferenceSyncError = null;
let preferenceSyncRetryTimer = null;
let preferenceSyncRetryAttempt = 0;
const PREFERENCE_SYNC_RETRY_MAX_ATTEMPTS = 3;
const PREFERENCE_SYNC_RETRY_BASE_MS = 100;

function transitionSucceeded(result) {
  return result !== false && (!result || result.ok !== false);
}

function createActivePreferenceSync(generation, replace = false) {
  if (!replace && activePreferenceSync?.generation === generation) return activePreferenceSync;
  let resolve;
  const gate = {
    generation,
    started: false,
    settled: false,
    promise: new Promise((done) => { resolve = done; }),
    resolve,
  };
  activePreferenceSync = gate;
  preferenceSyncPhase = "pending";
  preferenceSyncError = null;
  return gate;
}

function settlePreferenceSync(gate, outcome) {
  if (!gate || gate.settled) return;
  gate.settled = true;
  gate.resolve(outcome);
}

function cancelPreferenceSyncRetry({ resetAttempt = true } = {}) {
  if (preferenceSyncRetryTimer != null) clearTimeout(preferenceSyncRetryTimer);
  preferenceSyncRetryTimer = null;
  if (resetAttempt) preferenceSyncRetryAttempt = 0;
}

function schedulePreferenceSyncRetry(generation) {
  if (preferenceSyncRetryTimer != null ||
      preferenceSyncRetryAttempt >= PREFERENCE_SYNC_RETRY_MAX_ATTEMPTS) return;
  const delay = PREFERENCE_SYNC_RETRY_BASE_MS * Math.pow(2, preferenceSyncRetryAttempt);
  preferenceSyncRetryAttempt++;
  preferenceSyncRetryTimer = setTimeout(() => {
    preferenceSyncRetryTimer = null;
    if (requestedDocumentState !== "active" || documentStateGeneration !== generation ||
        lifecycleFailureState !== "active") return;
    lifecycleFailureState = null;
    createActivePreferenceSync(generation, true);
    startLifecycleDrain();
  }, delay);
}

function supersedeActivePreferenceSync() {
  cancelPreferenceSyncRetry();
  const gate = activePreferenceSync;
  if (gate && !gate.settled) {
    settlePreferenceSync(gate, {
      ok: false,
      error: "preference-sync-superseded",
      reason: "Document state changed during preference synchronization",
      superseded: true,
    });
  }
  activePreferenceSync = null;
  preferenceSyncPhase = "idle";
  preferenceSyncError = null;
}

function activePreferenceSyncIsCurrent(gate) {
  return !!gate && activePreferenceSync === gate && requestedDocumentState === "active" &&
    documentStateGeneration === gate.generation;
}

function activePreferenceSyncNeedsDrain() {
  return activePreferenceSyncIsCurrent(activePreferenceSync) && !activePreferenceSync.settled;
}

async function syncPreferencesForActiveTransition(generation) {
  const gate = activePreferenceSync?.generation === generation ? activePreferenceSync : null;
  if (!gate) return { ok: true, synced: false };
  if (gate.started) return gate.promise;

  gate.started = true;
  preferenceSyncPhase = typeof api?.syncSitePrefs === "function" ? "syncing" : "ready";
  let syncResult;
  let syncError = null;
  try {
    syncResult = typeof api?.syncSitePrefs === "function"
      ? await api.syncSitePrefs()
      : { ok: true, synced: false };
  } catch (error) {
    syncError = error;
  }

  if (!activePreferenceSyncIsCurrent(gate)) {
    const outcome = {
      ok: false,
      error: "preference-sync-superseded",
      reason: "Document state changed during preference synchronization",
      superseded: true,
    };
    settlePreferenceSync(gate, outcome);
    return outcome;
  }

  if (syncError || !transitionSucceeded(syncResult)) {
    const reason = syncError
      ? errorMessage(syncError)
      : errorMessage(syncResult?.reason || syncResult?.error || "Site preferences could not be synchronized");
    preferenceSyncPhase = "failed";
    preferenceSyncError = syncError || new Error(reason);
    const outcome = { ok: false, error: "preference-sync-failed", reason };
    settlePreferenceSync(gate, outcome);
    schedulePreferenceSyncRetry(generation);
    console.error("[FSRCNNX] preference synchronization failed:", preferenceSyncError);
    return outcome;
  }

  preferenceSyncPhase = "ready";
  preferenceSyncError = null;
  cancelPreferenceSyncRetry();
  const outcome = { ok: true, result: syncResult };
  settlePreferenceSync(gate, outcome);
  return outcome;
}

async function waitForActivePreferenceSync() {
  const gate = activePreferenceSync;
  if (!activePreferenceSyncIsCurrent(gate)) return;
  const outcome = await gate.promise;
  if (!outcome.ok) throw new Error(outcome.reason || "Site preferences could not be synchronized");
}

function startEarlyHiddenDrain() {
  if (earlyHiddenDrain || requestedDocumentState !== "hidden" ||
      (appliedDocumentState === "hidden" && !activeTransitionInFlight)) return;
  earlyHiddenDrain = moduleLoaded.then(async () => {
    // The outer check may have observed an active resume in flight while the
    // last completed state was already hidden. Do not repeat that stale-state
    // check here: the resume can finish between these two callbacks and must be
    // fenced by a real suspend.
    if (requestedDocumentState !== "hidden" || !api) return;
    let succeeded = true;
    try {
      if (typeof api.suspendDocument === "function") {
        succeeded = transitionSucceeded(await api.suspendDocument());
      }
    } catch (error) {
      succeeded = false;
      console.error("[FSRCNNX] document hidden transition failed:", error);
    }
    if (!succeeded) {
      lifecycleFailureState = "hidden";
      return;
    }
    // Record what actually completed, even if a pageshow arrived while suspend
    // was in flight. The normal drain will then apply the newer active request.
    lifecycleFailureState = null;
    appliedDocumentState = "hidden";
  }).catch(() => {}).finally(() => {
    earlyHiddenDrain = null;
    startLifecycleDrain();
  });
}

async function drainDocumentLifecycle() {
  while (appliedDocumentState !== requestedDocumentState || activePreferenceSyncNeedsDrain()) {
    const state = requestedDocumentState;
    const stateGeneration = documentStateGeneration;
    // A suspend that started during restoration owns the transition until it
    // settles, even if pageshow has already requested active again. Letting the
    // active path overtake it can finish in the wrong (suspended) state.
    if (earlyHiddenDrain) {
      await earlyHiddenDrain;
      continue;
    }
    let result;
    try {
      // Hidden documents must quiesce as soon as the module exists, even while
      // preference restoration is still pending. Active/resumed documents wait
      // for restoration so their first reconciliation uses authoritative intent.
      result = state === "hidden" ? await moduleLoaded.then(() => ({ ok: true })) : await startup;
    } catch (error) {
      result = { ok: false, error };
    }
    // The page may have hidden again while import/restore was pending. Avoid
    // briefly resuming an already-obsolete state; the loop will apply the most
    // recent request instead.
    if (state !== requestedDocumentState || stateGeneration !== documentStateGeneration) continue;
    // earlyHiddenDrain can be created while the startup await above is pending.
    // Recheck it before invoking resume so suspend/resume stay serialized.
    if (earlyHiddenDrain) {
      await earlyHiddenDrain;
      continue;
    }
    if (!result.ok || !api) {
      if (state === "active") {
        const gate = activePreferenceSync?.generation === stateGeneration
          ? activePreferenceSync
          : null;
        settlePreferenceSync(gate, {
          ok: false,
          error: "startup-failed",
          reason: errorMessage(result.error || startupError),
        });
      }
      lifecycleFailureState = state;
      return;
    }
    if (result.ok && api) {
      const method = state === "hidden" ? "suspendDocument" : "resumeDocument";
      let succeeded = true;
      let transitionInvoked = false;
      try {
        // Cross-document messages can arrive out of order around BFCache. A
        // second active claim immediately before resume gives the background
        // worker a post-pagehide ownership signal before renderer state emits.
        if (state === "active") {
          activeTransitionInFlight = true;
          await sendDocumentState("active", stateGeneration);
          if (state === requestedDocumentState && stateGeneration === documentStateGeneration) {
            succeeded = transitionSucceeded(
              await syncPreferencesForActiveTransition(stateGeneration),
            );
          }
        }
        if (succeeded && state === requestedDocumentState &&
            stateGeneration === documentStateGeneration &&
            typeof api[method] === "function") {
          transitionInvoked = true;
          succeeded = transitionSucceeded(await api[method]());
        }
      } catch (error) {
        succeeded = false;
        console.error(`[FSRCNNX] document ${state} transition failed:`, error);
      } finally {
        if (state === "active") activeTransitionInFlight = false;
      }
      if (state !== requestedDocumentState || stateGeneration !== documentStateGeneration) {
        // A resume that settled after pagehide may have performed work before
        // observing main's generation fence. Force one final suspension rather
        // than trusting the previously-applied hidden marker.
        if (state === "active" && transitionInvoked && requestedDocumentState === "hidden") {
          appliedDocumentState = null;
          startEarlyHiddenDrain();
        }
        continue;
      }
      if (!succeeded) {
        lifecycleFailureState = state;
        return;
      }
    }
    if (state !== requestedDocumentState || stateGeneration !== documentStateGeneration) continue;
    lifecycleFailureState = null;
    appliedDocumentState = state;
  }
}

function requestDocumentState(state) {
  if (state !== "active" && state !== "hidden") return;
  if (requestedDocumentState === state) {
    // Repeated page lifecycle signals are useful recovery messages: service
    // workers can restart and a previous send can be rejected during a
    // navigation transition. Reannounce ownership even when no local state
    // change is needed, and retry a transition that explicitly failed.
    void sendDocumentState(state);
    if (lifecycleFailureState === state) {
      lifecycleFailureState = null;
      if (state === "active") {
        cancelPreferenceSyncRetry();
        createActivePreferenceSync(documentStateGeneration, true);
      }
      if (state === "hidden") startEarlyHiddenDrain();
      startLifecycleDrain();
    } else if (state === "hidden") {
      startEarlyHiddenDrain();
    }
    return;
  }
  const previousState = requestedDocumentState;
  requestedDocumentState = state;
  documentStateGeneration++;
  lifecycleFailureState = null;
  if (state === "active" && previousState === "hidden") {
    cancelPreferenceSyncRetry();
    createActivePreferenceSync(documentStateGeneration);
  } else if (state === "hidden") {
    supersedeActivePreferenceSync();
  }
  if (activeClaimRetry != null) {
    clearTimeout(activeClaimRetry);
    activeClaimRetry = null;
  }
  // Announce immediately. In particular, an active document must own the tab
  // before restore/resume can publish renderer state, and a hidden document must
  // stop owning its badge before any late async work completes.
  void sendDocumentState(state);
  if (state === "active") {
    // One bounded retry closes the remaining cross-context delivery window. It
    // republishes the current effective state only after the ownership message
    // has settled, so a recovered claim does not leave a falsely blank badge.
    activeClaimRetry = setTimeout(async () => {
      activeClaimRetry = null;
      if (requestedDocumentState !== "active") return;
      await sendDocumentState("active");
      if (requestedDocumentState === "active") await publishCurrentDocumentState();
    }, 250);
  }
  if (state === "hidden") startEarlyHiddenDrain();
  startLifecycleDrain();
}

function startLifecycleDrain() {
  if (requestedDocumentState === "hidden" && earlyHiddenDrain) return;
  if (lifecycleFailureState === requestedDocumentState) return;
  if (lifecycleDrain ||
      (appliedDocumentState === requestedDocumentState && !activePreferenceSyncNeedsDrain())) return;
  lifecycleDrain = drainDocumentLifecycle()
    .catch((error) => console.error("[FSRCNNX] document lifecycle failed:", error))
    .finally(() => {
      lifecycleDrain = null;
      startLifecycleDrain();
    });
}

function currentDocumentState() {
  return document.prerendering === true || document.visibilityState === "hidden"
    ? "hidden"
    : "active";
}

requestDocumentState(currentDocumentState());
window.addEventListener("pagehide", () => requestDocumentState("hidden"));
document.addEventListener("freeze", () => requestDocumentState("hidden"));
window.addEventListener("pageshow", () => requestDocumentState(currentDocumentState()));
document.addEventListener("resume", () => requestDocumentState(currentDocumentState()));
document.addEventListener("visibilitychange", () => requestDocumentState(currentDocumentState()));
document.addEventListener("prerenderingchange", () => {
  requestDocumentState(currentDocumentState());
});

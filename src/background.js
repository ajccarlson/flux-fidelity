// Service worker that reflects the active top-level document's renderer state
// on the toolbar icon. Document ownership prevents delayed messages from a
// replaced navigation or a cached BFCache document from changing the badge.

const COLORS = {
  upscale: "#1f9e8f",
  passthrough: "#2f80ed",
  protected: "#c0392b",
};

const ACTIVE_ICON = {
  16: "icons/icon-16.png", 32: "icons/icon-32.png",
  48: "icons/icon-48.png", 128: "icons/icon-128.png",
};
const OFF_ICON = {
  16: "icons/icon-off-16.png", 32: "icons/icon-off-32.png",
  48: "icons/icon-off-48.png", 128: "icons/icon-off-128.png",
};
const VALID_MODES = new Set(["off", "upscale", "passthrough", "protected"]);
const NEURAL_CAPABILITY_MINT = "FSRCNNX_NEURAL_FRAME_CAPABILITY_MINT";
const NEURAL_CAPABILITY_CONSUME = "FSRCNNX_NEURAL_FRAME_CAPABILITY_CONSUME";
const NEURAL_FRAME_PATH = "/src/frame/neural-frame.html";
const NONCE_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;
const CAPABILITY_PATTERN = /^[a-f0-9]{48}$/;
const CAPABILITY_TTL_MS = 15_000;
const MAX_NEURAL_CAPABILITIES = 128;
const tabDocuments = new Map();
const pendingDocuments = new Map();
const documentLifecycles = new Map();
const neuralCapabilities = new Map();

function titleFor(mode) {
  switch (mode) {
    case "upscale": return "FluFi — upscaling active";
    case "passthrough": return "FluFi — passthrough";
    case "protected": return "FluFi — source can't be processed";
    default: return "FluFi";
  }
}

function actionCall(method, details) {
  try {
    const pending = chrome.action[method]?.(details);
    if (pending && typeof pending.catch === "function") pending.catch(() => {});
  } catch {}
}

// Last mode published by each tab's active document. The keyboard shortcut needs
// it to decide what to toggle to, and every mode transition already funnels
// through setBadge, so this is the one place that cannot miss one.
const tabModes = new Map();

function setBadge(tabId, mode) {
  if (!Number.isInteger(tabId) || !VALID_MODES.has(mode)) return;
  tabModes.set(tabId, mode);
  let text = "", color = "#000000";
  if (mode === "upscale") { text = "ON"; color = COLORS.upscale; }
  else if (mode === "passthrough") { text = "··"; color = COLORS.passthrough; }
  else if (mode === "protected") { text = "✕"; color = COLORS.protected; }

  actionCall("setBadgeText", { tabId, text });
  if (text) {
    actionCall("setBadgeBackgroundColor", { tabId, color });
    actionCall("setBadgeTextColor", { tabId, color: "#ffffff" });
  }
  actionCall("setTitle", { tabId, title: titleFor(mode) });
  actionCall("setIcon", { tabId, path: mode === "upscale" ? ACTIVE_ICON : OFF_ICON });
}

function resetTab(tabId) {
  tabDocuments.delete(tabId);
  pendingDocuments.delete(tabId);
  documentLifecycles.delete(tabId);
  clearNeuralCapabilities(tabId);
  setBadge(tabId, "off");
}

function clearNeuralCapabilities(tabId) {
  for (const [capability, record] of neuralCapabilities) {
    if (record.tabId === tabId) neuralCapabilities.delete(capability);
  }
}

function pruneNeuralCapabilities(now = Date.now()) {
  for (const [capability, record] of neuralCapabilities) {
    if (record.expiresAt <= now) neuralCapabilities.delete(capability);
  }
  while (neuralCapabilities.size >= MAX_NEURAL_CAPABILITIES) {
    neuralCapabilities.delete(neuralCapabilities.keys().next().value);
  }
}

function randomNeuralCapability() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  let capability = "";
  for (const byte of bytes) capability += byte.toString(16).padStart(2, "0");
  return capability;
}

function hasOnlyKeys(value, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const actual = Object.keys(value);
  return actual.length === keys.size && actual.every((key) => keys.has(key));
}

function pageOriginFromSender(sender) {
  if (sender?.id !== chrome.runtime.id || typeof sender.url !== "string") return null;
  try {
    const parsed = new URL(sender.url);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return { parentOrigin: parsed.origin, opaqueParent: false };
    }
    if (parsed.protocol === "file:") {
      return { parentOrigin: "null", opaqueParent: true };
    }
  } catch {}
  return null;
}

function isPackagedNeuralFrameSender(sender) {
  if (sender?.id !== chrome.runtime.id ||
      !Number.isInteger(sender?.tab?.id) ||
      !Number.isInteger(sender.frameId) ||
      sender.frameId === 0 ||
      sender.documentLifecycle !== "active" ||
      typeof sender.url !== "string") return false;
  try {
    const parsed = new URL(sender.url);
    return parsed.protocol === "chrome-extension:" &&
      parsed.pathname === NEURAL_FRAME_PATH;
  } catch {
    return false;
  }
}

function mintNeuralCapability(msg, sender) {
  if (!hasOnlyKeys(msg, new Set(["type", "instanceNonce"])) ||
      !NONCE_PATTERN.test(msg.instanceNonce)) return { ok: false };
  const identity = senderDocument(sender);
  const parent = pageOriginFromSender(sender);
  if (!parent || !isCurrentActiveDocument(identity)) return { ok: false };

  try {
    pruneNeuralCapabilities();
    const capability = randomNeuralCapability();
    neuralCapabilities.set(capability, {
      tabId: identity.tabId,
      documentId: identity.documentId,
      instanceNonce: msg.instanceNonce,
      parentOrigin: parent.parentOrigin,
      opaqueParent: parent.opaqueParent,
      expiresAt: Date.now() + CAPABILITY_TTL_MS,
    });
    return { ok: true, capability };
  } catch {
    return { ok: false };
  }
}

function consumeNeuralCapability(msg, sender) {
  if (!hasOnlyKeys(
    msg,
    new Set(["type", "capability", "instanceNonce"]),
  ) ||
      !CAPABILITY_PATTERN.test(msg.capability) ||
      !NONCE_PATTERN.test(msg.instanceNonce) ||
      !isPackagedNeuralFrameSender(sender)) return { ok: false };

  const record = neuralCapabilities.get(msg.capability);
  // A capability is single-use even when the attempted consumption is invalid.
  if (record) neuralCapabilities.delete(msg.capability);
  if (!record || record.expiresAt <= Date.now() ||
      record.tabId !== sender.tab.id ||
      record.instanceNonce !== msg.instanceNonce) return { ok: false };
  const owner = tabDocuments.get(record.tabId);
  if (owner?.state !== "active" || owner.documentId !== record.documentId) {
    return { ok: false };
  }
  return {
    ok: true,
    parentOrigin: record.parentOrigin,
    opaqueParent: record.opaqueParent,
  };
}

function senderDocument(sender) {
  const tabId = sender?.tab?.id;
  if (!Number.isInteger(tabId)) return null;
  if (typeof sender.documentId !== "string" || !sender.documentId) return null;
  return {
    tabId,
    documentId: sender.documentId,
    frameId: sender.frameId,
    lifecycle: sender.documentLifecycle,
  };
}

function isActiveTopDocument(identity) {
  // The manifest injects this content script only into outermost documents
  // (`all_frames` is false). Prerender/BFCache outermost documents can retain a
  // nonzero frameId across activation, so lifecycle + document identity are the
  // stable authority; a numeric frame ID is not.
  return !!identity && identity.lifecycle === "active";
}

function lifecycleRecord(tabId, documentId) {
  return documentLifecycles.get(tabId)?.get(documentId) || null;
}

function observeDocumentLifecycle(msg, identity) {
  if (!Number.isSafeInteger(msg.generation) || msg.generation < 1) return null;
  let records = documentLifecycles.get(identity.tabId);
  if (!records) {
    records = new Map();
    documentLifecycles.set(identity.tabId, records);
  }
  let record = records.get(identity.documentId);
  if (!record) {
    record = {
      generation: msg.generation,
      state: msg.state,
      fallbackRetiredAt: null,
    };
    records.set(identity.documentId, record);
    return record;
  }
  if (msg.generation < record.generation) return null;
  if (msg.generation === record.generation) {
    return msg.state === record.state ? record : null;
  }
  // Content increments the generation exactly once per active/hidden change.
  // This parity check rejects impossible skipped-state claims while allowing
  // delivery to omit any number of complete lifecycle transitions.
  const stateChanged = msg.state !== record.state;
  if (stateChanged !== ((msg.generation - record.generation) % 2 === 1)) return null;
  record.generation = msg.generation;
  record.state = msg.state;
  return record;
}

function clearFallbackRetirement(tabId, documentId) {
  const record = lifecycleRecord(tabId, documentId);
  if (record) record.fallbackRetiredAt = null;
}

function retireActiveOwner(tabId, documentId) {
  let records = documentLifecycles.get(tabId);
  if (!records) {
    records = new Map();
    documentLifecycles.set(tabId, records);
  }
  let record = records.get(documentId);
  if (!record) {
    // The worker may have reconstructed ownership from a renderer-state
    // message without seeing its active handshake.
    record = { generation: 0, state: "hidden", fallbackRetiredAt: 0 };
    records.set(documentId, record);
  } else {
    // An active tab owner paired with a hidden lifecycle record means its
    // active transition was not delivered. Retire that inferred generation.
    record.fallbackRetiredAt = record.generation + (record.state === "active" ? 0 : 1);
  }
  return record;
}

function retainLoadingTombstone(tabId, blockedDocumentId, outgoingWasActive) {
  if (!blockedDocumentId) {
    documentLifecycles.delete(tabId);
    return;
  }
  let record = lifecycleRecord(tabId, blockedDocumentId);
  if (outgoingWasActive) {
    record = retireActiveOwner(tabId, blockedDocumentId);
  } else if (!record) {
    // Ownership can have been rebuilt from a state message after an MV3 worker
    // restart. A hidden owner needs only an ordering baseline; its next active
    // generation remains a legitimate BFCache return.
    record = { generation: 0, state: "hidden", fallbackRetiredAt: null };
  }
  // A full navigation makes every other document record irrelevant. Retaining
  // only the outgoing tombstone keeps this worker's state bounded per tab.
  documentLifecycles.set(tabId, new Map([[blockedDocumentId, record]]));
}

function claimDocument(identity, mode = "off") {
  const previous = tabDocuments.get(identity.tabId);
  if (previous?.documentId && previous.documentId !== identity.documentId) {
    clearNeuralCapabilities(identity.tabId);
  }
  tabDocuments.set(identity.tabId, { documentId: identity.documentId, state: "active" });
  pendingDocuments.delete(identity.tabId);
  clearFallbackRetirement(identity.tabId, identity.documentId);
  setBadge(identity.tabId, mode);
}

function stagePendingDocument(identity, mode = null, { confirmationGeneration = null } = {}) {
  if (!isActiveTopDocument(identity)) return false;
  const existing = pendingDocuments.get(identity.tabId);
  if (existing?.documentId === identity.documentId) {
    if (mode != null) existing.mode = mode;
    if (confirmationGeneration != null) {
      if (existing.confirmationGeneration === confirmationGeneration) {
        existing.confirmations = Math.min(2, existing.confirmations + 1);
      } else {
        existing.confirmationGeneration = confirmationGeneration;
        existing.confirmations = 1;
      }
    }
    return existing;
  }
  const pending = {
    documentId: identity.documentId,
    mode: mode == null ? "off" : mode,
    confirmationGeneration,
    confirmations: confirmationGeneration == null ? 0 : 1,
  };
  pendingDocuments.set(identity.tabId, pending);
  return pending;
}

function promotePendingDocument(tabId) {
  const pending = pendingDocuments.get(tabId);
  if (!pending) return false;
  const previous = tabDocuments.get(tabId);
  if (previous?.documentId && previous.documentId !== pending.documentId) {
    clearNeuralCapabilities(tabId);
  }
  pendingDocuments.delete(tabId);
  tabDocuments.set(tabId, { documentId: pending.documentId, state: "active" });
  clearFallbackRetirement(tabId, pending.documentId);
  setBadge(tabId, pending.mode);
  return true;
}

function isCurrentActiveDocument(identity) {
  if (!isActiveTopDocument(identity)) return false;
  const owner = tabDocuments.get(identity.tabId);
  // MV3 workers are disposable. If this worker has no record at all, active
  // browser metadata is sufficient to rebuild ownership without requiring the
  // already-running content script to be reinjected. A navigation tombstone,
  // by contrast, must wait for the new document's explicit handshake.
  if (!owner) {
    tabDocuments.set(identity.tabId, { documentId: identity.documentId, state: "active" });
    return true;
  }
  return owner?.documentId === identity.documentId && owner.state === "active";
}

function handleDocumentHandshake(msg, identity) {
  if (!identity || (msg.state !== "active" && msg.state !== "hidden")) return;
  if (msg.state === "active" && !isActiveTopDocument(identity)) return;
  const lifecycle = observeDocumentLifecycle(msg, identity);
  if (!lifecycle) return;
  const owner = tabDocuments.get(identity.tabId);

  if (msg.state === "active") {
    if (lifecycle.fallbackRetiredAt != null) {
      // Apply retirement before every claim path, including a hidden owner or
      // loading tombstone: neither may turn an old queued retry into authority.
      // Two newer generations prove an unobserved hidden -> active cycle.
      if (msg.generation < lifecycle.fallbackRetiredAt + 2) return;
      lifecycle.fallbackRetiredAt = null;
    }
    // Only a browser-confirmed active document may claim (or reclaim from
    // BFCache) a tab. Claiming with a new document resets every visible state.
    if (owner?.state === "loading") {
      // `tabs.onUpdated` can run before delayed callbacks from the outgoing
      // document. Keep its identity as a tombstone and accept only the new
      // navigation's document ID.
      if (owner.blockedDocumentId === identity.documentId) return;
      claimDocument(identity);
      return;
    }
    // A different document cannot displace a still-active owner by message
    // ordering alone. Full navigations install a loading tombstone, while
    // BFCache transitions make the outgoing owner hidden before the incoming
    // document resumes. If that hidden signal is lost while the MV3 worker is
    // unavailable, a repeated browser-confirmed active handshake recovers the
    // handoff without letting a single delayed message steal ownership.
    if (owner?.documentId && owner.documentId !== identity.documentId && owner.state === "active") {
      const pending = stagePendingDocument(identity, null, {
        confirmationGeneration: msg.generation,
      });
      if (pending.confirmations >= 2) {
        claimDocument(identity, pending.mode);
        retireActiveOwner(identity.tabId, owner.documentId);
      }
      return;
    }
    if (!owner || owner.documentId !== identity.documentId) claimDocument(identity);
    else {
      const pending = pendingDocuments.get(identity.tabId);
      // A reaffirmation from the current owner invalidates an interleaved
      // claimant's confirmations. Preserve its identity and buffered mode so
      // an explicit owner-hidden signal can still complete the normal BFCache
      // handoff; fallback requires two fresh, consecutive retries.
      if (pending && pending.documentId !== identity.documentId) {
        pending.confirmationGeneration = null;
        pending.confirmations = 0;
      } else if (pending) {
        pendingDocuments.delete(identity.tabId);
      }
      tabDocuments.set(identity.tabId, { documentId: identity.documentId, state: "active" });
    }
    return;
  }

  // A cached outermost document can report a nonzero frame ID. Relinquishing is
  // safe by exact document identity even when it is no longer the active top
  // frame. A pending document that hides before promotion simply withdraws.
  // Hidden evidence also retires every older queued active retry by generation
  // and permits this document to participate in a later BFCache return.
  // Only hidden evidence newer than the retired active generation may re-arm
  // this document. A duplicate hidden message from the older recorded epoch
  // can arrive after ownership was reconstructed from renderer state and must
  // not authorize delayed active retries from that inferred generation.
  if (lifecycle.fallbackRetiredAt == null ||
      msg.generation > lifecycle.fallbackRetiredAt) {
    lifecycle.fallbackRetiredAt = null;
  }
  const pending = pendingDocuments.get(identity.tabId);
  if (pending?.documentId === identity.documentId) {
    pendingDocuments.delete(identity.tabId);
    return;
  }
  // A hidden/cached document may relinquish only its own current claim. A late
  // pagehide from an older document cannot clear the replacement's badge.
  if (!owner || owner.documentId !== identity.documentId) return;
  tabDocuments.set(identity.tabId, { documentId: identity.documentId, state: "hidden" });
  if (!promotePendingDocument(identity.tabId)) setBadge(identity.tabId, "off");
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === NEURAL_CAPABILITY_MINT) {
    sendResponse?.(mintNeuralCapability(msg, sender));
    return;
  }
  if (msg?.type === NEURAL_CAPABILITY_CONSUME) {
    sendResponse?.(consumeNeuralCapability(msg, sender));
    return;
  }
  // The capability handlers above verify sender.id; the state handlers relied only
  // on senderDocument(). Same non-exploitability as the content script, same
  // one-line fix, and it keeps every branch of this listener consistent.
  if (sender?.id !== chrome.runtime.id) return;
  const identity = senderDocument(sender);
  if (msg?.type === "FSRCNNX_DOCUMENT") {
    handleDocumentHandshake(msg, identity);
    return;
  }
  if (msg?.type !== "FSRCNNX_STATE" && msg?.type !== "FSRCNNX_PROTECTED") return;
  if (msg.type === "FSRCNNX_STATE" && !VALID_MODES.has(msg.mode)) return;
  if (!isCurrentActiveDocument(identity)) {
    const owner = identity ? tabDocuments.get(identity.tabId) : null;
    const lifecycle = identity ? lifecycleRecord(identity.tabId, identity.documentId) : null;
    if (owner?.state === "active" && owner.documentId !== identity?.documentId &&
        lifecycle?.fallbackRetiredAt == null &&
        stagePendingDocument(identity, msg.type === "FSRCNNX_PROTECTED" ? "protected" : msg.mode)) {
      return;
    }
    const pending = identity ? pendingDocuments.get(identity.tabId) : null;
    if (isActiveTopDocument(identity) && pending?.documentId === identity.documentId) {
      pending.mode = msg.type === "FSRCNNX_PROTECTED" ? "protected" : msg.mode;
    }
    return;
  }

  if (msg?.type === "FSRCNNX_STATE") {
    setBadge(identity.tabId, msg.mode);
  } else if (msg?.type === "FSRCNNX_PROTECTED") {
    setBadge(identity.tabId, "protected");
  }
});

// A loading navigation invalidates the old document before its delayed async
// callbacks can report. The newly injected content script will claim ownership.
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === "loading" || typeof changeInfo.url === "string") {
    clearNeuralCapabilities(tabId);
  }
  if (changeInfo.status === "loading") {
    const current = tabDocuments.get(tabId);
    const blockedDocumentId = current?.state === "loading"
      ? current.blockedDocumentId
      : current?.documentId || null;
    const blockedWasActive = current?.state === "loading"
      ? current.blockedWasActive === true
      : current?.state === "active";
    pendingDocuments.delete(tabId);
    retainLoadingTombstone(tabId, blockedDocumentId, blockedWasActive);
    setBadge(tabId, "off");
    tabDocuments.set(tabId, {
      documentId: null,
      state: "loading",
      blockedDocumentId,
      blockedWasActive,
    });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  tabDocuments.delete(tabId);
  pendingDocuments.delete(tabId);
  documentLifecycles.delete(tabId);
  tabModes.delete(tabId);
  clearNeuralCapabilities(tabId);
});

// The popup cannot be opened while the page is fullscreen, so before this there
// was no way at all to toggle enhancement during the product's primary use case:
// watching a video full-screen. Off toggles to upscale; anything else toggles off,
// so the shortcut is a single predictable switch rather than a mode cycle.
// Not exported: this service worker is registered as a classic script, so it has
// no module scope. The suite reaches it through the vm context instead.
function nextToggleMode(currentMode) {
  return currentMode === "upscale" ? "off" : "upscale";
}

chrome.commands?.onCommand?.addListener(async (command) => {
  if (command !== "toggle-enhancement") return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!Number.isInteger(tab?.id)) return;
    const mode = nextToggleMode(tabModes.get(tab.id));
    // A protected or video-less page answers with ok:false, which the badge
    // already reflects; the shortcut deliberately stays silent because there is no
    // popup open to report into.
    await chrome.tabs.sendMessage(tab.id, { type: "FSRCNNX_SETMODE", mode });
  } catch {}
});

chrome.tabs.onReplaced.addListener((addedTabId, removedTabId) => {
  tabDocuments.delete(removedTabId);
  pendingDocuments.delete(removedTabId);
  documentLifecycles.delete(removedTabId);
  clearNeuralCapabilities(removedTabId);
  resetTab(addedTabId);
  tabDocuments.set(addedTabId, {
    documentId: null,
    state: "loading",
    blockedDocumentId: null,
    blockedWasActive: false,
  });
});

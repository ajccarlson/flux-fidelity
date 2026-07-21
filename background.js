// background.js — service worker that reflects the extension's per-tab state on
// the toolbar icon via a badge. The content script sends FSRCNNX_STATE messages
// (with the active mode) whenever it changes; we set a small badge on the sender's
// tab so the icon shows at a glance whether upscaling is active.
//
// Badge meanings:
//   upscale     -> "ON"  teal   (actively upscaling)
//   passthrough -> "··"  blue   (capturing/displaying, no upscale)
//   off / none  -> cleared
//   protected   -> "✕"   red    (source can't be processed)

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

// Color icon when upscaling is genuinely active; monochrome otherwise
// (off, passthrough, or protected — none of which are "enhancing the image").
function setIcon(tabId, mode) {
  if (tabId == null) return;
  const path = mode === "upscale" ? ACTIVE_ICON : OFF_ICON;
  try { chrome.action.setIcon({ tabId, path }); } catch {}
}

function setBadge(tabId, mode) {
  if (tabId == null) return;
  let text = "", color = "#000000";
  if (mode === "upscale") { text = "ON"; color = COLORS.upscale; }
  else if (mode === "passthrough") { text = "··"; color = COLORS.passthrough; }
  else if (mode === "protected") { text = "✕"; color = COLORS.protected; }
  // else off -> empty text clears the badge
  try {
    chrome.action.setBadgeText({ tabId, text });
    if (text) {
      chrome.action.setBadgeBackgroundColor({ tabId, color });
      if (chrome.action.setBadgeTextColor) chrome.action.setBadgeTextColor({ tabId, color: "#ffffff" });
    }
    chrome.action.setTitle({ tabId, title: titleFor(mode) });
    setIcon(tabId, mode);
  } catch {}
}

function titleFor(mode) {
  switch (mode) {
    case "upscale": return "Video Upscaler — upscaling active";
    case "passthrough": return "Video Upscaler — passthrough";
    case "protected": return "Video Upscaler — source can't be processed";
    default: return "Video Upscaler";
  }
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (!sender || !sender.tab) return;
  if (msg?.type === "FSRCNNX_STATE") {
    setBadge(sender.tab.id, msg.mode || "off");
  } else if (msg?.type === "FSRCNNX_PROTECTED") {
    setBadge(sender.tab.id, "protected");
  }
});

// Clear badge when a tab navigates away or reloads (content script will re-report).
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === "loading") {
    try { chrome.action.setBadgeText({ tabId, text: "" }); } catch {}
  }
});

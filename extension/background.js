// ===== background.js — LinkSus Service Worker =====
//
// This runs persistently in the background even when the popup is closed.
// Its only job is:
//   1. Listen for tab updates (page loads / URL changes)
//   2. When a page fully loads, check if auto-detect is ON
//   3. If ON, store the tab's URL so the popup can read it instantly on open
//
// WHY store instead of scan directly?
// Service workers cannot call the Flask API and update the popup UI at the
// same time — the popup isn't open yet. So the background just stores the
// latest URL. When the popup opens, popup.js reads it and runs the scan.
// This is the correct Chrome extension architecture for MV3.

const BACKEND_URL = "http://127.0.0.1:5000/api/check";

// ===== LISTEN FOR TAB UPDATES =====
// status === "complete" means the page fully loaded — not just started loading.
// We ignore "loading" status to avoid scanning incomplete pages.
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
    if (changeInfo.status !== "complete") return;
    if (!tab.url) return;

    // Skip Chrome internal pages — can't scan these
    if (tab.url.startsWith("chrome://") ||
        tab.url.startsWith("chrome-extension://") ||
        tab.url.startsWith("about:") ||
        tab.url.startsWith("edge://")) {
        return;
    }

    // Check if auto-detect is enabled before doing anything
    chrome.storage.local.get("autoDetect", function(data) {
        const autoDetectOn = data.autoDetect === true;
        if (!autoDetectOn) return;

        // Store the latest URL and clear any old result
        // The popup will read this when it opens and auto-scan
        chrome.storage.local.set({
            latestURL: tab.url,
            latestTabId: tabId,
            autoScanPending: true   // flag tells popup to auto-scan on open
        });
    });
});

// ===== LISTEN FOR TAB ACTIVATION =====
// When user switches to a different tab, update the stored URL
// so the popup shows the correct page when opened
chrome.tabs.onActivated.addListener(function(activeInfo) {
    chrome.tabs.get(activeInfo.tabId, function(tab) {
        if (!tab || !tab.url) return;
        if (tab.url.startsWith("chrome://") ||
            tab.url.startsWith("chrome-extension://") ||
            tab.url.startsWith("about:")) {
            return;
        }

        chrome.storage.local.get("autoDetect", function(data) {
            if (data.autoDetect !== true) return;
            chrome.storage.local.set({
                latestURL: tab.url,
                latestTabId: activeInfo.tabId,
                autoScanPending: true
            });
        });
    });
});

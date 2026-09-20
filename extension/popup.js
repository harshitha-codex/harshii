// ===== CONFIG =====
const BACKEND_URL = "http://127.0.0.1:5000/api/check";

let currentURL = "";

// ===== ON POPUP OPEN =====
document.addEventListener("DOMContentLoaded", function () {

    // 1. Load the saved toggle state from Chrome storage and apply it
    chrome.storage.local.get("autoDetect", function (data) {
        const isAuto = data.autoDetect === true;
        document.getElementById("auto-toggle").checked = isAuto;
        updateAutoModeBar(isAuto);
    });

    // 2. Attach scan button click listener (inline onclick blocked by MV3 CSP)
    document.getElementById("scan-btn").addEventListener("click", function () {
        scanURL();
    });

    // 3. Listen for toggle changes — save to storage immediately
    document.getElementById("auto-toggle").addEventListener("change", function () {
        const isAuto = this.checked;
        chrome.storage.local.set({ autoDetect: isAuto });
        updateAutoModeBar(isAuto);

        // If user just turned OFF auto-detect, clear any pending auto-scan flag
        if (!isAuto) {
            chrome.storage.local.remove("autoScanPending");
        }
    });

    // 4. Get the current tab URL and display it
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (!tabs || !tabs[0]) {
            document.getElementById("current-url").textContent = "Could not get URL";
            showError("Could not access the current tab.");
            return;
        }

        currentURL = tabs[0].url;
        const displayURL = currentURL.replace(/^https?:\/\/(www\.)?/, "");
        document.getElementById("current-url").textContent = displayURL || currentURL;

        // 5. Check if auto-detect is ON AND background flagged a pending scan
        //    If both are true → auto-scan immediately on popup open
        chrome.storage.local.get(["autoDetect", "autoScanPending", "latestURL"], function (data) {
            const isAuto = data.autoDetect === true;
            const isPending = data.autoScanPending === true;

            if (isAuto && isPending) {
                // Clear the pending flag so it doesn't re-scan on next open
                // unless background sets it again (i.e. a new page loads)
                chrome.storage.local.remove("autoScanPending");

                // Use the URL background stored (should match current tab)
                // Fall back to current tab URL if storage URL is missing
                const urlToScan = data.latestURL || currentURL;

                // Update the display URL if it differs from what we showed
                if (urlToScan !== currentURL) {
                    currentURL = urlToScan;
                    const display = urlToScan.replace(/^https?:\/\/(www\.)?/, "");
                    document.getElementById("current-url").textContent = display;
                }

                // Auto-scan
                performScan(currentURL);

            }
            // If auto is OFF or no pending scan → do nothing, wait for manual click
        });
    });
});

// ===== UPDATE AUTO MODE INDICATOR BAR =====
function updateAutoModeBar(isAuto) {
    const bar = document.getElementById("auto-mode-bar");
    if (isAuto) {
        bar.classList.add("visible");
    } else {
        bar.classList.remove("visible");
    }
}

// ===== MANUAL SCAN — triggered by button click =====
// This is the existing behavior — completely unchanged
function scanURL() {
    if (!currentURL) return;

    if (currentURL.startsWith("chrome://") ||
        currentURL.startsWith("chrome-extension://") ||
        currentURL.startsWith("about:")) {
        showError("This is a browser internal page — nothing to scan here.");
        return;
    }

    performScan(currentURL);
}

// ===== CORE SCAN FUNCTION =====
// Used by both manual and auto-detect modes.
// Separated so both paths call the same logic — no duplication.
function performScan(url) {
    if (!url) return;

    if (url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("about:")) {
        showError("This is a browser internal page — nothing to scan here.");
        return;
    }

    showLoading(true);
    hideAll();

    fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    })
    .then(function (res) {
        if (!res.ok) throw new Error("Server returned " + res.status);
        return res.json();
    })
    .then(function (data) {
        showLoading(false);
        showResult(data);
    })
    .catch(function (err) {
        showLoading(false);
        console.error("Scan error:", err);
        showError(
            "Could not connect to LinkSus backend.\n\n" +
            "Make sure your Flask app is running at localhost:5000.\n\n" +
            "Run: python app.py"
        );
    });
}

// ===== SHOW RESULT — unchanged from original =====
function showResult(data) {
    const box        = document.getElementById("result-box");
    const verdict    = document.getElementById("result-verdict");
    const aiSection  = document.getElementById("ai-explanation");
    const aiText     = document.getElementById("ai-text");
    const sigSection = document.getElementById("signals-section");
    const sigList    = document.getElementById("signals-list");

    const isPhishing = data.prediction === "bad";

    box.style.display = "block";
    box.className = "result-box " + (isPhishing ? "phishing" : "safe");
    verdict.className = "result-verdict " + (isPhishing ? "phishing" : "safe");
    verdict.textContent = isPhishing ? "⚠️ Phishing Website Detected" : "✅ Safe Website";

    if (data.gemini_text) {
        aiText.textContent = data.gemini_text;
        aiSection.style.display = "block";
    } else {
        aiSection.style.display = "none";
    }

    if (data.explanation && data.explanation.length > 0) {
        sigList.innerHTML = "";
        data.explanation.forEach(function (item) {
            const row   = document.createElement("div");
            row.className = "signal-row";

            const name  = document.createElement("span");
            name.className = "signal-name";
            name.textContent = item.name;

            const badge = document.createElement("span");
            badge.className = "signal-badge " + (item.direction === "phishing" ? "badge-risk" : "badge-safe");
            badge.textContent = item.direction === "phishing" ? "⚠ Risk" : "✓ Safe";

            row.appendChild(name);
            row.appendChild(badge);
            sigList.appendChild(row);
        });
        sigSection.style.display = "block";
    } else {
        sigSection.style.display = "none";
    }
}

// ===== HELPERS — unchanged =====
function showError(msg) {
    const box = document.getElementById("error-box");
    box.style.display = "block";
    box.textContent = "⚠️ " + msg;
}

function showLoading(show) {
    document.getElementById("loading").style.display = show ? "block" : "none";
    document.getElementById("scan-btn").disabled = show;
}

function hideAll() {
    document.getElementById("result-box").style.display = "none";
    document.getElementById("error-box").style.display = "none";
}

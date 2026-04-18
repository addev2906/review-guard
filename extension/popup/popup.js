/**
 * ReviewGuard — Popup Controller
 */

const $ = (sel) => document.querySelector(sel);

const DOM = {
  statusDot:    $("#statusDot .status-dot"),
  statusLabel:  $("#statusLabel"),
  btnScan:      $("#btnScan"),
  btnScanLabel: $("#btnScanLabel"),
  statsSection: $("#statsSection"),
  statScanned:  $("#statScanned"),
  statGenuine:  $("#statGenuine"),
  statFake:     $("#statFake"),
  statUncertain:$("#statUncertain"),
  scoreBarWrap: $("#scoreBarWrap"),
  trustPct:     $("#trustPct"),
  trustFill:    $("#trustFill"),
  message:      $("#message"),
  apiUrlInput:  $("#apiUrlInput"),
  nvidiaKeyInput: $("#nvidiaKeyInput"),
  btnSave:      $("#btnSave"),
  modelAccuracy:$("#modelAccuracy"),
  // Jump navigation
  jumpNav:      $("#jumpNav"),
  jumpCounter:  $("#jumpCounter"),
  btnPrev:      $("#btnPrev"),
  btnNext:      $("#btnNext"),
};

let fakeCount = 0;
let currentFakeIndex = -1;

// ── Helpers ─────────────────────────────────────────────────────────────────

function setStatus(connected) {
  DOM.statusDot.className = connected
    ? "status-dot status-dot--connected"
    : "status-dot status-dot--disconnected";
  DOM.statusLabel.textContent = connected ? "Connected" : "Disconnected";
  DOM.btnScan.disabled = !connected;
}

function showMessage(text, type = "info") {
  DOM.message.textContent = text;
  DOM.message.className = `message message--${type}`;
  DOM.message.style.display = "block";
  if (type !== "error") {
    setTimeout(() => { DOM.message.style.display = "none"; }, 5000);
  }
}

function updateStats(stats) {
  DOM.statsSection.style.display = "grid";
  DOM.statScanned.textContent = stats.scanned;
  DOM.statGenuine.textContent = stats.genuine;
  DOM.statFake.textContent = stats.fake;
  DOM.statUncertain.textContent = stats.uncertain;

  // Trust score = % genuine of total scanned
  if (stats.scanned > 0) {
    const trust = Math.round((stats.genuine / stats.scanned) * 100);
    DOM.scoreBarWrap.style.display = "block";
    DOM.trustPct.textContent = `${trust}%`;
    requestAnimationFrame(() => {
      DOM.trustFill.style.width = `${trust}%`;
    });
  }

  // Show jump nav if fakes found
  fakeCount = stats.fake || 0;
  if (fakeCount > 0) {
    currentFakeIndex = -1;
    DOM.jumpNav.style.display = "block";
    DOM.jumpCounter.textContent = `0 / ${fakeCount}`;
  } else {
    DOM.jumpNav.style.display = "none";
  }
}

function setScanning(active) {
  if (active) {
    DOM.btnScan.classList.add("btn-scan--scanning");
    DOM.btnScanLabel.textContent = "Scanning…";
    DOM.btnScan.disabled = true;
  } else {
    DOM.btnScan.classList.remove("btn-scan--scanning");
    DOM.btnScanLabel.textContent = "Scan Reviews";
    DOM.btnScan.disabled = false;
  }
}

// ── Jump Navigation ─────────────────────────────────────────────────────────

function jumpToFake(direction) {
  if (fakeCount === 0) return;

  if (direction === "next") {
    currentFakeIndex = (currentFakeIndex + 1) % fakeCount;
  } else {
    currentFakeIndex = (currentFakeIndex - 1 + fakeCount) % fakeCount;
  }

  DOM.jumpCounter.textContent = `${currentFakeIndex + 1} / ${fakeCount}`;

  // Send message to content script to scroll to the fake review
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    if (!tab?.id) return;
    chrome.tabs.sendMessage(tab.id, {
      action: "jumpToFake",
      index: currentFakeIndex,
    });
  });
}

// ── Init ────────────────────────────────────────────────────────────────────

async function init() {
  // Load saved API URL
  try {
    const data = await chrome.storage.local.get(["apiUrl", "nvidiaKey"]);
    if (data.apiUrl) DOM.apiUrlInput.value = data.apiUrl;
    if (data.nvidiaKey) DOM.nvidiaKeyInput.value = data.nvidiaKey;
  } catch { /* ignore */ }

  // Check health
  chrome.runtime.sendMessage({ action: "health" }, (resp) => {
    if (resp && resp.connected) {
      setStatus(true);
      if (resp.model_accuracy != null) {
        DOM.modelAccuracy.textContent = `${(resp.model_accuracy * 100).toFixed(1)}%`;
      }
    } else {
      setStatus(false);
      showMessage("Backend not reachable. Start the server or update the URL in settings.", "error");
    }
  });
}

// ── Event listeners ─────────────────────────────────────────────────────────

DOM.btnScan.addEventListener("click", () => {
  setScanning(true);
  DOM.message.style.display = "none";
  DOM.jumpNav.style.display = "none";

  chrome.runtime.sendMessage({ action: "scanPage" }, (resp) => {
    setScanning(false);

    if (!resp) {
      showMessage("No response — is the page a supported review site?", "error");
      return;
    }
    if (resp.error) {
      showMessage(resp.error, "error");
      return;
    }
    if (resp.scanned === 0) {
      showMessage("No reviews found on this page. Try scrolling to load more reviews, or this site may not be supported.", "info");
      return;
    }

    updateStats(resp);
    const msg = resp.fake > 0
      ? `⚠ Found ${resp.fake} potentially fake review${resp.fake > 1 ? "s" : ""}! Use the arrows below to jump to each one.`
      : `✓ All ${resp.scanned} reviews look genuine.`;
    showMessage(msg, resp.fake > 0 ? "error" : "success");
  });
});

DOM.btnPrev.addEventListener("click", () => jumpToFake("prev"));
DOM.btnNext.addEventListener("click", () => jumpToFake("next"));

DOM.btnSave.addEventListener("click", async () => {
  const url = DOM.apiUrlInput.value.trim().replace(/\/+$/, "");
  const token = DOM.nvidiaKeyInput.value.trim();
  if (!url) return;
  await chrome.storage.local.set({ apiUrl: url, nvidiaKey: token });
  showMessage("Settings saved. Rechecking connection…", "info");
  setTimeout(init, 500);
});

// ── Boot ────────────────────────────────────────────────────────────────────
init();

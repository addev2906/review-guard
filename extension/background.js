/**
 * ReviewGuard — Background Service Worker
 * Handles communication between popup/content scripts and the backend API.
 */

const DEFAULT_API_URL = "https://review-guard-backend.onrender.com";

async function getApiUrl() {
  try {
    const data = await chrome.storage.local.get(["apiUrl", "nvidiaKey"]);
    return {
      apiUrl: data.apiUrl || DEFAULT_API_URL,
      nvidiaKey: data.nvidiaKey || ""
    };
  } catch {
    return { apiUrl: DEFAULT_API_URL, nvidiaKey: "" };
  }
}

async function checkHealth() {
  const { apiUrl } = await getApiUrl();
  try {
    const res = await fetch(`${apiUrl}/api/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return { connected: false };
    const body = await res.json();
    return { connected: true, ...body };
  } catch {
    return { connected: false };
  }
}

async function predictBatch(reviews) {
  const { apiUrl } = await getApiUrl();
  const res = await fetch(`${apiUrl}/api/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

async function predictSingle(text) {
  const { apiUrl } = await getApiUrl();
  const res = await fetch(`${apiUrl}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

async function explainReview(text, verdict) {
  const { apiUrl, nvidiaKey } = await getApiUrl();
  const res = await fetch(`${apiUrl}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_text: text, verdict: verdict, nvidia_key: nvidiaKey }),
    signal: AbortSignal.timeout(20000), // NVIDIA API might be slow
  });
  if (!res.ok) {
    let err = await res.text();
    try { err = JSON.parse(err).detail || err; } catch {}
    throw new Error(`Explain API Error: ${err}`);
  }
  return res.json();
}

// ── Message router ──────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const handler = async () => {
    try {
      switch (message.action) {
        case "health":
          return await checkHealth();

        case "predict":
          return await predictSingle(message.text);

        case "explain":
          return await explainReview(message.text, message.verdict);

        case "predictBatch": {
          // Split into chunks of 50
          const all = message.reviews;
          const results = [];
          for (let i = 0; i < all.length; i += 50) {
            const chunk = all.slice(i, i + 50);
            const resp = await predictBatch(chunk);
            results.push(...resp.results);
          }
          return { results };
        }

        case "scanPage":
          // Forward to content script on the active tab
          const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
          if (!tab?.id) throw new Error("No active tab");
          // Inject content script if not already there
          try {
            await chrome.scripting.executeScript({
              target: { tabId: tab.id },
              files: ["content.js"],
            });
            await chrome.scripting.insertCSS({
              target: { tabId: tab.id },
              files: ["content.css"],
            });
          } catch { /* may already be injected */ }
          return new Promise((resolve) => {
            chrome.tabs.sendMessage(tab.id, { action: "extractAndScan" }, (resp) => {
              resolve(resp || { error: "No response from content script" });
            });
          });

        default:
          return { error: `Unknown action: ${message.action}` };
      }
    } catch (err) {
      return { error: err.message };
    }
  };

  handler().then(sendResponse);
  return true; // keep channel open for async
});

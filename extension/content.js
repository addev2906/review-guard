/**
 * ReviewGuard — Content Script
 * Extracts reviews from product pages and injects result badges.
 */

(() => {
  "use strict";

  // Prevent double-injection
  if (window.__reviewGuardInjected) return;
  window.__reviewGuardInjected = true;

  // ── Site-specific selectors ─────────────────────────────────────────────
  const SITE_CONFIGS = {
    "amazon": {
      reviewSelector: '[data-hook="review-body"] span, .review-text-content span',
      reviewContainer: '[data-hook="review"], .a-section.review',
    },
    "flipkart": {
      reviewSelector: '._6K-7Co, .t-ZTKy, div[class*="review"] > div > div > div:last-child, ._11pzQk',
      reviewContainer: '.col.EPCmJX, .col._2wzgFH, div[class*="review-card"], ._1AtVbE .col',
    },
    "yelp": {
      reviewSelector: '.comment__09f24__D0cxf p, [class*="comment"] p, .raw__09f24__T4Ezm',
      reviewContainer: '[class*="review__"], li[class*="margin-b"]',
    },
    "tripadvisor": {
      reviewSelector: '.fIrGe._T, [data-automation="reviewCard"] .biGQs, q.QewHA span',
      reviewContainer: '[data-automation="reviewCard"], .review-container',
    },
  };

  function detectSite() {
    const host = window.location.hostname;
    if (host.includes("amazon")) return "amazon";
    if (host.includes("flipkart")) return "flipkart";
    if (host.includes("yelp")) return "yelp";
    if (host.includes("tripadvisor")) return "tripadvisor";
    return null;
  }

  // ── Generic fallback extractor ──────────────────────────────────────────
  function extractGenericReviews() {
    const candidates = [];
    // Look for common review-like elements
    const selectors = [
      '[itemprop="reviewBody"]',
      '[class*="review-text"]',
      '[class*="review_text"]',
      '[class*="reviewText"]',
      '[class*="review-body"]',
      '[class*="review_body"]',
      '[class*="reviewBody"]',
      '[class*="comment-text"]',
      '[class*="comment-body"]',
      '[data-review]',
    ];
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach((el) => {
        const text = el.innerText?.trim();
        if (text && text.length > 20) {
          candidates.push({ element: el, text, container: el.closest('[class*="review"], [class*="comment"]') || el.parentElement });
        }
      });
    }
    return candidates;
  }

  // ── Extract reviews from current page ───────────────────────────────────
  function extractReviews() {
    const site = detectSite();
    const reviews = [];

    if (site && SITE_CONFIGS[site]) {
      const config = SITE_CONFIGS[site];
      const elements = document.querySelectorAll(config.reviewSelector);
      elements.forEach((el) => {
        const text = el.innerText?.trim();
        if (text && text.length > 20) {
          const container = el.closest(config.reviewContainer) || el.parentElement;
          reviews.push({ element: el, text, container });
        }
      });
    }

    // If site-specific found nothing, try generic
    if (reviews.length === 0) {
      reviews.push(...extractGenericReviews());
    }

    // Deduplicate by text
    const seen = new Set();
    return reviews.filter((r) => {
      if (seen.has(r.text)) return false;
      seen.add(r.text);
      return true;
    });
  }

  // ── Badge injection ─────────────────────────────────────────────────────
  function createBadge(result) {
    const badge = document.createElement("div");
    badge.className = "rg-badge";

    let icon, statusClass, label;
    if (result.fake_probability >= 0.6) {
      icon = "⚠";
      statusClass = "rg-badge--fake";
      label = "Likely Fake";
    } else if (result.fake_probability <= 0.4) {
      icon = "✓";
      statusClass = "rg-badge--genuine";
      label = "Genuine";
    } else {
      icon = "?";
      statusClass = "rg-badge--uncertain";
      label = "Uncertain";
    }

    badge.classList.add(statusClass);

    const pct = Math.round(result.fake_probability * 100);
    badge.innerHTML = `
      <span class="rg-badge__icon">${icon}</span>
      <span class="rg-badge__label">${label}</span>
      <span class="rg-badge__pct">${pct}% fake</span>
    `;

    // Tooltip with details
    const tooltip = document.createElement("div");
    tooltip.className = "rg-tooltip";
    tooltip.innerHTML = `
      <div class="rg-tooltip__title">ReviewGuard Analysis</div>
      <div class="rg-tooltip__row">
        <span>Verdict</span>
        <strong>${result.label}</strong>
      </div>
      <div class="rg-tooltip__row">
        <span>Fake Probability</span>
        <strong>${(result.fake_probability * 100).toFixed(1)}%</strong>
      </div>
      <div class="rg-tooltip__row">
        <span>Confidence</span>
        <strong>${result.confidence_band}</strong>
      </div>
      ${result.flags.length > 0 ? `
        <div class="rg-tooltip__flags">
          <span>Flags:</span>
          ${result.flags.map(f => `<span class="rg-tooltip__flag">${f}</span>`).join("")}
        </div>
      ` : ""}
      ${result.top_terms.length > 0 ? `
        <div class="rg-tooltip__terms">
          <span>Key terms:</span>
          <span class="rg-tooltip__term-list">${result.top_terms.slice(0, 5).join(", ")}</span>
        </div>
      ` : ""}
    `;

    badge.appendChild(tooltip);
    return badge;
  }

  function injectBadge(container, result) {
    // Remove any existing badge
    const existing = container.querySelector(".rg-badge");
    if (existing) existing.remove();

    // Make container position relative if needed
    const pos = window.getComputedStyle(container).position;
    if (pos === "static") {
      container.style.position = "relative";
    }

    const badge = createBadge(result);
    container.insertBefore(badge, container.firstChild);
  }

  // ── Scan orchestrator ───────────────────────────────────────────────────
  async function scanPage() {
    const reviews = extractReviews();
    if (reviews.length === 0) {
      return { scanned: 0, genuine: 0, fake: 0, uncertain: 0 };
    }

    const texts = reviews.map((r) => r.text);

    // Send batch to background for prediction
    const response = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { action: "predictBatch", reviews: texts },
        (resp) => resolve(resp)
      );
    });

    if (response.error) {
      console.error("[ReviewGuard]", response.error);
      return { scanned: 0, error: response.error };
    }

    const stats = { scanned: 0, genuine: 0, fake: 0, uncertain: 0 };

    response.results.forEach((result, idx) => {
      const review = reviews[idx];
      if (!review) return;

      injectBadge(review.container, result);
      stats.scanned++;

      if (result.fake_probability >= 0.6) stats.fake++;
      else if (result.fake_probability <= 0.4) stats.genuine++;
      else stats.uncertain++;
    });

    return stats;
  }

  // ── Listen for messages ─────────────────────────────────────────────────
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "extractAndScan") {
      scanPage().then(sendResponse);
      return true;
    }
    if (message.action === "getReviewCount") {
      const reviews = extractReviews();
      sendResponse({ count: reviews.length });
      return false;
    }
  });
})();

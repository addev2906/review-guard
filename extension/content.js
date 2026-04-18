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
      reviewSelector: '[data-hook*="review-body"], .review-text, .review-text-content',
      reviewContainer: '[data-hook*="review"], .review',
    },
    "flipkart": {
      // Flipkart obfuscates classes constantly — use broad attribute-contains patterns
      reviewSelector: '[class*="ZmyHeo"], [class*="_6K-7Co"], [class*="t-ZTKy"], [class*="qwjRop"], div[class] > div[class] > div[class] > div:not([class*="star"]):not([class*="rating"]) > p, div[class*="review"] p',
      reviewContainer: 'div[class*="review-card"], div[class*="col"] > div[class], div[class*="_27M-vq"], div[class*="_1AtVbE"], div[class*="EKFha"]',
    },
    "yelp": {
      // Yelp randomises classes — target structural patterns and comment paragraphs
      reviewSelector: '[class*="comment"] p, [class*="review__"] p, p[class*="raw__"], [data-testid*="review"] p, span[class*="raw__"], [lang] p',
      reviewContainer: 'li:has([class*="comment"]), li:has(p[class*="raw__"]), [class*="review__"], li[class*="margin-b"], [data-testid*="review"], ul > li:has(p)',
    },
    "tripadvisor": {
      reviewSelector: 'q.QewHA span, .yCeTE, .IRsPn',
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
    const selectors = [
      '[itemprop="reviewBody"]',
      '[class*="review-text"]',
      '[class*="review_text"]',
      '[class*="reviewText"]',
      '[class*="review-content"]',
      '[class*="review-body"]',
      '[class*="review_body"]',
      '[class*="reviewBody"]',
      '[class*="comment-text"]',
      '[class*="comment-body"]',
      // Flipkart & Yelp patterns
      '[data-testid*="review"] p',
      'p[class*="raw__"]',
      'span[class*="raw__"]',
      '[class*="comment__"] p',
    ];

    // Collect all elements that match any generic selector
    const allMatches = new Set();
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach(el => allMatches.add(el));
    }

    // Filter to innermost elements to prevent selecting a giant wrapper
    const innermostMatches = Array.from(allMatches).filter(el => {
      for (const other of allMatches) {
        if (el !== other && el.contains(other)) {
          return false; // el is a wrapper, discard it
        }
      }
      return true;
    });

    for (const el of innermostMatches) {
      const text = el.innerText?.trim();
      if (text && text.length >= 2) {
        let container = el.closest('[itemprop="review"], [class*="review-container"], [class*="reviewCard"], li, article') 
                        || el.closest('[class*="review"], [class*="comment"]') 
                        || el.parentElement;
        candidates.push({ element: el, text, container });
      }
    }
    return candidates;
  }

  // ── Extract reviews from current page ───────────────────────────────────
  function extractReviews() {
    const site = detectSite();
    const rawReviews = [];

    // 1. Site-specific extraction
    if (site && SITE_CONFIGS[site]) {
      const config = SITE_CONFIGS[site];
      const elements = document.querySelectorAll(config.reviewSelector);
      elements.forEach((el) => {
        const text = el.innerText?.trim();
        // Lowered threshold to 2 to catch extremely short reviews like "Good"
        if (text && text.length >= 2) {
          const container = el.closest(config.reviewContainer) || el.parentElement;
          rawReviews.push({ element: el, text, container });
        }
      });
    }

    // 2. Always run generic fallback to guarantee we miss nothing!
    rawReviews.push(...extractGenericReviews());

    // 3. Precise Deduplication
    const uniqueReviews = [];
    for (const r of rawReviews) {
      if (!r.container || !r.text) continue;
      
      let isDuplicate = false;
      for (const u of uniqueReviews) {
        // Since we blocked the colossal wrappers earlier, it's 100% safe to deduplicate
        // by checking if the containers equal or enclose each other on the page.
        // This stops short reviews from accidentally merging with long reviews that
        // happen to share the same substring of words.
        if (r.container === u.container || r.container.contains(u.container) || u.container.contains(r.container)) {
          isDuplicate = true;
          // Keep the variant that managed to grab more of the text naturally
          if (r.text.length > u.text.length) {
             u.text = r.text;
             u.container = r.container; 
          }
          break;
        }
      }
      
      if (!isDuplicate) {
        uniqueReviews.push(r);
      }
    }

    return uniqueReviews;
  }

  // ── Badge injection ─────────────────────────────────────────────────────
  function createBadge(result, reviewText) {
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
      <div class="rg-tooltip__ai">
        <button class="rg-tooltip__ai-btn">Ask AI to Explain</button>
        <div class="rg-tooltip__ai-result" style="display: none;"></div>
      </div>
    `;

    // Add event listener for AI Explanation
    const aiBtn = tooltip.querySelector(".rg-tooltip__ai-btn");
    const aiResult = tooltip.querySelector(".rg-tooltip__ai-result");
    
    aiBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();
      
      aiBtn.disabled = true;
      aiBtn.textContent = "Asking AI...";
      
      try {
        const response = await new Promise((resolve) => {
          chrome.runtime.sendMessage(
            { action: "explain", text: reviewText, verdict: result.label },
            (resp) => resolve(resp || { error: "No response from background." })
          );
        });
        
        aiBtn.style.display = "none";
        aiResult.style.display = "block";
        
        if (response.error) {
          aiResult.innerHTML = `<span style="color: #ef4444;">Error: ${response.error}</span>`;
        } else {
          aiResult.innerHTML = response.explanation;
        }
      } catch (err) {
        aiBtn.disabled = false;
        aiBtn.textContent = "Ask AI to Explain";
        alert("Failed to reach AI: " + err.message);
      }
    });

    // Tooltip click toggle instead of hover
    badge.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      
      const isOpen = tooltip.classList.contains("rg-tooltip--visible");
      
      // Close any other open tooltips
      document.querySelectorAll(".rg-tooltip--visible").forEach(t => {
        t.classList.remove("rg-tooltip--visible");
      });
      document.querySelectorAll(".rg-badge--open").forEach(b => {
        b.classList.remove("rg-badge--open");
      });
      
      if (!isOpen) {
        tooltip.classList.add("rg-tooltip--visible");
        badge.classList.add("rg-badge--open");
      }
    });

    tooltip.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent clicks inside tooltip from bubbling up and closing it
    });

    badge.appendChild(tooltip);
    return badge;
  }

  function injectBadge(container, result, text) {
    // Remove any existing badge
    const existing = container.querySelector(".rg-badge");
    if (existing) existing.remove();

    // Make container position relative if needed
    const pos = window.getComputedStyle(container).position;
    if (pos === "static") {
      container.style.position = "relative";
    }

    const badge = createBadge(result, text);
    container.insertBefore(badge, container.firstChild);
  }

  // ── Fake review tracking for jump navigation ─────────────────────────────
  let fakeReviewContainers = [];

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
    fakeReviewContainers = []; // reset

    response.results.forEach((result, idx) => {
      const review = reviews[idx];
      if (!review) return;

      injectBadge(review.container, result, review.text);
      stats.scanned++;

      if (result.fake_probability >= 0.6) {
        stats.fake++;
        fakeReviewContainers.push(review.container);
      } else if (result.fake_probability <= 0.4) {
        stats.genuine++;
      } else {
        stats.uncertain++;
      }
    });

    return stats;
  }

  // ── Jump to fake review ─────────────────────────────────────────────────
  function jumpToFakeReview(index) {
    if (index < 0 || index >= fakeReviewContainers.length) return;

    // Remove highlight from all
    fakeReviewContainers.forEach((el) => el.classList.remove("rg-highlight"));

    const target = fakeReviewContainers[index];
    target.scrollIntoView({ behavior: "smooth", block: "center" });

    // Add highlight pulse
    target.classList.add("rg-highlight");
    setTimeout(() => target.classList.remove("rg-highlight"), 2500);
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
    if (message.action === "jumpToFake") {
      jumpToFakeReview(message.index);
      sendResponse({ ok: true });
      return false;
    }
  });

  // Close tooltips when clicking outside
  document.addEventListener("click", () => {
    document.querySelectorAll(".rg-tooltip--visible").forEach(t => {
      t.classList.remove("rg-tooltip--visible");
    });
    document.querySelectorAll(".rg-badge--open").forEach(b => {
      b.classList.remove("rg-badge--open");
    });
  });
})();


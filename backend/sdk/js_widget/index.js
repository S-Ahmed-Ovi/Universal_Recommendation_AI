/**
 * sdk/js_widget/index.js — Proactive AI embed widget (v3)
 *
 * A single-script drop-in that renders a recommendation carousel on any
 * webpage.  Fetches recommendations from the Proactive AI /recommend
 * endpoint and injects a styled widget into a host element.
 *
 * ── Quick start ────────────────────────────────────────────────────────────
 *
 *   <div id="proactive-recommendations"></div>
 *
 *   <script>
 *     window.ProactiveAIConfig = {
 *       apiUrl:    "https://api.yourcompany.com",  // base URL (no trailing slash)
 *       apiKey:    "pk_live_...",                  // tenant API key
 *       userId:    "user_42",                      // current user
 *       containerId: "proactive-recommendations",  // mount element id
 *     };
 *   </script>
 *   <script src="https://cdn.yourcompany.com/proactive-ai/widget.js" async></script>
 *
 * ── Config reference ────────────────────────────────────────────────────────
 *
 *   apiUrl         {string}   Required. Base URL of the Proactive AI API.
 *   apiKey         {string}   Required. Bearer token / API key for the tenant.
 *   userId         {string}   Required. User ID to fetch recommendations for.
 *   containerId    {string}   Optional. Host element id. Default: "proactive-recommendations".
 *   topN           {number}   Optional. Number of items to show. Default: 10.
 *   strategy       {string}   Optional. Recommendation strategy (see /recommend docs).
 *   algorithm      {string}   Optional. Force a specific algorithm.
 *   title          {string}   Optional. Widget heading. Default: "Recommended for you".
 *   theme          {string}   Optional. "light" | "dark". Default: "light".
 *   onItemClick    {function} Optional. Called with the item object when a card is clicked.
 *   onError        {function} Optional. Called with an Error object on failure.
 *   renderItem     {function} Optional. Custom card renderer — receives an item object,
 *                             must return an HTMLElement.
 *
 * ── Item object shape ───────────────────────────────────────────────────────
 *
 *   { item_id: string, score: number, rank: number }
 *
 * ── Programmatic API ────────────────────────────────────────────────────────
 *
 *   const widget = window.ProactiveAI.init(config);
 *   widget.refresh();           // re-fetch and re-render
 *   widget.destroy();           // remove widget and clean up
 *   widget.setUser("user_99");  // switch user and re-fetch
 */

(function (global) {
  "use strict";

  /* ── Constants ─────────────────────────────────────────────────────────── */

  var WIDGET_ID      = "proactive-ai-widget";
  var DEFAULT_TOP_N  = 10;
  var DEFAULT_TITLE  = "Recommended for you";
  var DEFAULT_THEME  = "light";
  var API_TIMEOUT_MS = 8000;

  /* ── Minimal CSS injected once ─────────────────────────────────────────── */

  var CSS = [
    "#proactive-ai-widget{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;box-sizing:border-box;width:100%}",
    "#proactive-ai-widget *{box-sizing:border-box}",
    "#proactive-ai-widget .pai-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}",
    "#proactive-ai-widget .pai-title{font-size:16px;font-weight:600;margin:0}",
    "#proactive-ai-widget .pai-refresh{background:none;border:none;cursor:pointer;padding:4px;opacity:.6;font-size:14px}",
    "#proactive-ai-widget .pai-refresh:hover{opacity:1}",
    "#proactive-ai-widget .pai-track{display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;-webkit-overflow-scrolling:touch;scrollbar-width:thin}",
    "#proactive-ai-widget .pai-track::-webkit-scrollbar{height:4px}",
    "#proactive-ai-widget .pai-track::-webkit-scrollbar-thumb{border-radius:2px}",
    "#proactive-ai-widget .pai-card{flex:0 0 140px;border-radius:8px;padding:14px 12px;cursor:pointer;transition:transform .15s,box-shadow .15s;user-select:none}",
    "#proactive-ai-widget .pai-card:hover{transform:translateY(-2px)}",
    "#proactive-ai-widget .pai-card-id{font-size:13px;font-weight:600;word-break:break-all;margin-bottom:6px}",
    "#proactive-ai-widget .pai-card-score{font-size:11px;opacity:.65}",
    "#proactive-ai-widget .pai-card-rank{font-size:10px;opacity:.45;margin-top:2px}",
    "#proactive-ai-widget .pai-empty{font-size:13px;opacity:.55;padding:12px 0}",
    "#proactive-ai-widget .pai-error{font-size:12px;padding:10px 12px;border-radius:6px;margin-top:4px}",
    "#proactive-ai-widget .pai-loading{display:flex;gap:8px;padding:12px 0}",
    "#proactive-ai-widget .pai-skeleton{flex:0 0 140px;height:72px;border-radius:8px;animation:pai-pulse 1.4s ease-in-out infinite}",
    "@keyframes pai-pulse{0%,100%{opacity:.4}50%{opacity:.8}}",

    /* Light theme */
    "#proactive-ai-widget.pai-light .pai-title{color:#111}",
    "#proactive-ai-widget.pai-light .pai-refresh{color:#555}",
    "#proactive-ai-widget.pai-light .pai-card{background:#f5f5f5;border:1px solid #e8e8e8}",
    "#proactive-ai-widget.pai-light .pai-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.1)}",
    "#proactive-ai-widget.pai-light .pai-card-id{color:#111}",
    "#proactive-ai-widget.pai-light .pai-error{background:#fff2f0;color:#c0392b;border:1px solid #ffd7d4}",
    "#proactive-ai-widget.pai-light .pai-skeleton{background:#e0e0e0}",
    "#proactive-ai-widget.pai-light .pai-track::-webkit-scrollbar-thumb{background:#ccc}",

    /* Dark theme */
    "#proactive-ai-widget.pai-dark .pai-title{color:#f0f0f0}",
    "#proactive-ai-widget.pai-dark .pai-refresh{color:#aaa}",
    "#proactive-ai-widget.pai-dark .pai-card{background:#2a2a2a;border:1px solid #3a3a3a}",
    "#proactive-ai-widget.pai-dark .pai-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.4)}",
    "#proactive-ai-widget.pai-dark .pai-card-id{color:#f0f0f0}",
    "#proactive-ai-widget.pai-dark .pai-error{background:#3a1a1a;color:#e07070;border:1px solid #5a2a2a}",
    "#proactive-ai-widget.pai-dark .pai-skeleton{background:#3a3a3a}",
    "#proactive-ai-widget.pai-dark .pai-track::-webkit-scrollbar-thumb{background:#555}",
  ].join("\n");

  /* ── DOM utilities ─────────────────────────────────────────────────────── */

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "className") { node.className = attrs[k]; }
      else if (k === "textContent") { node.textContent = attrs[k]; }
      else if (k.startsWith("on")) { node.addEventListener(k.slice(2).toLowerCase(), attrs[k]); }
      else { node.setAttribute(k, attrs[k]); }
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function injectCSS() {
    if (document.getElementById("proactive-ai-css")) return;
    var style = document.createElement("style");
    style.id = "proactive-ai-css";
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  /* ── HTTP helper ───────────────────────────────────────────────────────── */

  function fetchJSON(url, options, timeoutMs) {
    return new Promise(function (resolve, reject) {
      var timedOut = false;
      var timer = setTimeout(function () {
        timedOut = true;
        reject(new Error("Request timed out after " + timeoutMs + "ms"));
      }, timeoutMs || API_TIMEOUT_MS);

      fetch(url, options)
        .then(function (res) {
          clearTimeout(timer);
          if (timedOut) return;
          if (!res.ok) {
            return res.json().catch(function () { return {}; }).then(function (body) {
              reject(new Error(body.detail || ("HTTP " + res.status)));
            });
          }
          return res.json().then(resolve);
        })
        .catch(function (err) {
          clearTimeout(timer);
          if (!timedOut) reject(err);
        });
    });
  }

  /* ── Auth ──────────────────────────────────────────────────────────────── */

  /**
   * Determine whether the supplied apiKey is a JWT (three dot-separated
   * segments) or a plain API key, then return the appropriate auth headers.
   *
   * - Plain API key  → { "X-API-Key": key }
   * - JWT            → { "Authorization": "Bearer <jwt>" }
   *
   * This replaces the old resolveToken() approach that tried to exchange
   * an API key for a JWT via /auth/login and then sent it as a Bearer
   * token — which always returned 401 because the server's JWT validator
   * would reject a raw API key string.
   */
  function resolveAuthHeaders(config) {
    var key = config.apiKey || "";

    // A JWT has exactly two dots (header.payload.signature)
    var isJwt = (key.match(/\./g) || []).length >= 2;

    if (isJwt) {
      return Promise.resolve({ "Authorization": "Bearer " + key });
    }

    // Plain API key — send directly; no token exchange needed
    return Promise.resolve({ "X-API-Key": key });
  }

  /* ── API calls ─────────────────────────────────────────────────────────── */

  function fetchRecommendations(config, authHeaders) {
    var body = {
      user_id: config.userId,
      top_n:   config.topN || DEFAULT_TOP_N,
    };
    if (config.strategy)  body.strategy  = config.strategy;
    if (config.algorithm) body.algorithm = config.algorithm;
    if (config.modelId)   body.model_id  = config.modelId;

    return fetchJSON(
      config.apiUrl + "/recommend",
      {
        method:  "POST",
        headers: Object.assign({ "Content-Type": "application/json" }, authHeaders),
        body: JSON.stringify(body),
      }
    );
  }

  /* ── Rendering ─────────────────────────────────────────────────────────── */

  function renderSkeletons(track, n) {
    track.innerHTML = "";
    var loader = el("div", { className: "pai-loading" });
    for (var i = 0; i < Math.min(n, 6); i++) {
      loader.appendChild(el("div", { className: "pai-skeleton" }));
    }
    track.appendChild(loader);
  }

  function defaultRenderItem(item, onItemClick) {
    var score = typeof item.score === "number"
      ? item.score.toFixed(4)
      : "–";

    var card = el("div", { className: "pai-card" }, [
      el("div", { className: "pai-card-id", textContent: item.item_id }),
      el("div", { className: "pai-card-score", textContent: "Score: " + score }),
      el("div", { className: "pai-card-rank", textContent: "#" + item.rank }),
    ]);

    card.addEventListener("click", function () {
      if (typeof onItemClick === "function") onItemClick(item);
    });

    return card;
  }

  function renderItems(track, items, config) {
    track.innerHTML = "";
    if (!items || !items.length) {
      track.appendChild(el("div", { className: "pai-empty", textContent: "No recommendations available." }));
      return;
    }
    items.forEach(function (item) {
      var card = typeof config.renderItem === "function"
        ? config.renderItem(item)
        : defaultRenderItem(item, config.onItemClick);
      track.appendChild(card);
    });
  }

  function renderError(track, message) {
    track.innerHTML = "";
    track.appendChild(el("div", { className: "pai-error", textContent: "⚠ " + message }));
  }

  /* ── Widget class ──────────────────────────────────────────────────────── */

  function Widget(config) {
    this._config  = Object.assign({}, config);
    this._root    = null;
    this._track   = null;
    this._mounted = false;
  }

  Widget.prototype._mount = function () {
    var containerId = this._config.containerId || "proactive-recommendations";
    var host = document.getElementById(containerId);
    if (!host) {
      console.warn("[ProactiveAI] Container #" + containerId + " not found.");
      return false;
    }

    injectCSS();

    var theme = this._config.theme === "dark" ? "pai-dark" : "pai-light";
    this._root = el("div", { id: WIDGET_ID, className: theme });

    var track = el("div", { className: "pai-track" });
    this._track = track;

    var self = this;
    var header = el("div", { className: "pai-header" }, [
      el("p", { className: "pai-title", textContent: this._config.title || DEFAULT_TITLE }),
      el("button", {
        className: "pai-refresh",
        title: "Refresh recommendations",
        textContent: "↻",
        onClick: function () { self.refresh(); },
      }),
    ]);

    this._root.appendChild(header);
    this._root.appendChild(track);
    host.appendChild(this._root);
    this._mounted = true;
    return true;
  };

  Widget.prototype._resolveAuthHeaders = function () {
    // Auth headers are stateless for API keys; no caching needed.
    return resolveAuthHeaders(this._config);
  };

  Widget.prototype.refresh = function () {
    if (!this._mounted) {
      if (!this._mount()) return;
    }

    var self = this;
    var topN = self._config.topN || DEFAULT_TOP_N;
    renderSkeletons(self._track, topN);

    self._resolveAuthHeaders()
      .then(function (authHeaders) {
        return fetchRecommendations(self._config, authHeaders);
      })
      .then(function (data) {
        // API returns { recommendations: [...] } or { items: [...] }
        var items = data.recommendations || data.items || [];
        // Ensure rank field
        items = items.map(function (item, idx) {
          return Object.assign({ rank: idx + 1 }, item);
        });
        renderItems(self._track, items, self._config);
      })
      .catch(function (err) {
        renderError(self._track, err.message || "Failed to load recommendations.");
        if (typeof self._config.onError === "function") {
          self._config.onError(err);
        }
      });
  };

  Widget.prototype.setUser = function (userId) {
    this._config.userId = userId;
    this.refresh();
  };

  Widget.prototype.destroy = function () {
    if (this._root && this._root.parentNode) {
      this._root.parentNode.removeChild(this._root);
    }
    this._root    = null;
    this._track   = null;
    this._mounted = false;
  };

  /* ── Public API ────────────────────────────────────────────────────────── */

  var ProactiveAI = {
    /**
     * Initialise and render the widget.
     *
     * @param  {object} config  Widget configuration (see top of file).
     * @returns {Widget}        Widget instance with refresh/destroy/setUser.
     */
    init: function (config) {
      if (!config || !config.apiUrl) {
        console.error("[ProactiveAI] apiUrl is required.");
        return null;
      }
      if (!config.userId) {
        console.error("[ProactiveAI] userId is required.");
        return null;
      }
      if (!config.apiKey) {
        console.error("[ProactiveAI] apiKey is required.");
        return null;
      }

      var widget = new Widget(config);
      widget._mount();
      widget.refresh();
      return widget;
    },
  };

  /* ── Auto-init from window.ProactiveAIConfig ───────────────────────────── */

  function autoInit() {
    var cfg = global.ProactiveAIConfig;
    if (cfg && cfg.apiUrl && cfg.userId && cfg.apiKey) {
      global.ProactiveAI = ProactiveAI;
      ProactiveAI.init(cfg);
    } else {
      // Still expose the API so callers can call ProactiveAI.init() manually
      global.ProactiveAI = ProactiveAI;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoInit);
  } else {
    autoInit();
  }

})(window);
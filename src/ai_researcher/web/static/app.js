/* Progressive enhancement only — every page works with JS disabled. */
(function () {
  "use strict";

  var toastEl = document.getElementById("toast");
  var toastTimer = null;

  function toast(msg, ms) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.remove("show");
    }, ms || 2600);
  }

  /* ── save / unsave ─────────────────────────────────────────── */
  document.addEventListener("click", function (ev) {
    var copyBtn = ev.target.closest("[data-copy-target]");
    if (copyBtn) {
      ev.preventDefault();
      var el = document.getElementById(copyBtn.getAttribute("data-copy-target"));
      if (!el) return;
      var text = el.value || el.textContent || "";
      var done = function () { toast("Copied handoff for agent"); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
          toast("Could not copy");
        });
      } else {
        el.focus();
        el.select();
        try { document.execCommand("copy"); done(); }
        catch (e) { toast("Could not copy"); }
      }
      return;
    }
    var btn = ev.target.closest(".save");
    if (!btn) return;
    ev.preventDefault();
    var id = btn.dataset.id;
    btn.disabled = true;
    fetch("/api/save/" + id, { method: "POST" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (d) {
        btn.classList.toggle("on", d.saved);
        btn.textContent = d.saved ? "★" : "☆";
        btn.title = d.saved ? "Saved" : "Save";
        toast(d.saved ? "Saved" : "Removed from saved");
      })
      .catch(function () { toast("Could not save — is the server still up?"); })
      .finally(function () { btn.disabled = false; });
  });

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".fb");
    if (!btn) return;
    ev.preventDefault();
    var id = btn.dataset.id;
    var kind = btn.dataset.kind;
    fetch("/api/feedback/" + id + "?kind=" + encodeURIComponent(kind), { method: "POST" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function () { toast("Noted: " + kind); })
      .catch(function () { toast("Could not record feedback"); });
  });

  var regen = document.getElementById("regen-brief");
  if (regen) {
    regen.addEventListener("click", function (ev) {
      ev.preventDefault();
      regen.disabled = true;
      regen.textContent = "Regenerating…";
      fetch("/api/brief/regenerate", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function () {
          toast("Brief regenerated — reloading");
          setTimeout(function () { location.reload(); }, 700);
        })
        .catch(function () {
          toast("Could not regenerate the brief");
          regen.disabled = false;
          regen.textContent = "Regenerate brief";
        });
    });
  }

  /* ── verbose ingest status ─────────────────────────────────── */
  var VERBOSE_KEY = "air.verboseIngest";
  var verboseToggle = document.getElementById("verbose-ingest");
  var ingestStatus = document.getElementById("ingest-status");
  var ingestStage = document.getElementById("ingest-stage");
  var ingestDetail = document.getElementById("ingest-detail");
  var ingestCurrent = document.getElementById("ingest-current");
  var ingestCounts = document.getElementById("ingest-counts");
  var verbose = false;

  try {
    verbose = localStorage.getItem(VERBOSE_KEY) === "1";
  } catch (e) { /* private mode */ }

  // Table headers also stick, and at the same offset the strip claims. Measure
  // rather than hardcode: the strip wraps to two lines on a narrow window, and
  // a fixed guess buries the column headers underneath it while scrolling.
  function syncStickTop() {
    var top = 54;
    if (ingestStatus && !ingestStatus.hasAttribute("hidden")) {
      top += ingestStatus.offsetHeight;
    }
    document.documentElement.style.setProperty("--stick-top", top + "px");
  }

  function setVerbose(on) {
    verbose = !!on;
    if (verboseToggle) verboseToggle.checked = verbose;
    if (ingestStatus) {
      if (verbose) ingestStatus.removeAttribute("hidden");
      else ingestStatus.setAttribute("hidden", "");
    }
    try {
      localStorage.setItem(VERBOSE_KEY, verbose ? "1" : "0");
    } catch (e) { /* ignore */ }
    syncStickTop();
    syncPolling();
  }

  function renderProgress(progress, running) {
    if (!ingestStatus || !verbose) return;
    progress = progress || {};
    var stage = progress.stage || (running ? "running" : "idle");
    var detail = progress.detail || (running ? "Working…" : "Idle — toggle stays on for the next refresh");
    var current = progress.current || "";
    var done = progress.done || 0;
    var total = progress.total || 0;
    var active = progress.active || [];

    if (ingestStage) ingestStage.textContent = stage;
    if (ingestDetail) ingestDetail.textContent = detail;
    if (ingestCurrent) {
      if (current) {
        ingestCurrent.textContent = current;
        ingestCurrent.hidden = false;
      } else if (active.length) {
        ingestCurrent.textContent = "active: " + active.slice(0, 4).join(", ")
          + (active.length > 4 ? " +" + (active.length - 4) : "");
        ingestCurrent.hidden = false;
      } else {
        ingestCurrent.textContent = "";
        ingestCurrent.hidden = true;
      }
    }
    if (ingestCounts) {
      if (total > 0) {
        ingestCounts.textContent = done + " / " + total;
        ingestCounts.hidden = false;
      } else {
        ingestCounts.textContent = "";
        ingestCounts.hidden = true;
      }
    }
    ingestStatus.classList.toggle("live", !!running);
    syncStickTop();
  }

  if (verboseToggle) {
    verboseToggle.addEventListener("change", function () {
      setVerbose(verboseToggle.checked);
      if (verbose) fetchStatus();
    });
  }
  setVerbose(verbose);

  /* ── manual refresh + progress polling ─────────────────────── */
  var refreshBtn = document.getElementById("refresh");
  var dot = document.getElementById("statusdot");
  var lastRun = document.getElementById("lastrun");
  var pollTimer = null;
  var wasRunning = !!(refreshBtn && refreshBtn.disabled);

  function applyStats(stats) {
    if (!stats) return;
    document.querySelectorAll("[data-stat]").forEach(function (el) {
      var key = el.getAttribute("data-stat");
      if (key && stats[key] != null) el.textContent = stats[key];
    });
    if (lastRun && stats.last_run) lastRun.textContent = stats.last_run;
  }

  function fetchStatus() {
    return fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var running = !!(d.run && d.run.running);
        var progress = (d.run && d.run.progress) || {};
        applyStats(d.stats);
        renderProgress(progress, running);
        if (dot) dot.classList.toggle("busy", running);
        if (refreshBtn) {
          refreshBtn.disabled = running;
          refreshBtn.textContent = running ? "Refreshing…" : "Refresh";
        }
        if (wasRunning && !running) {
          wasRunning = false;
          toast("Refresh complete — reloading", 1500);
          setTimeout(function () { location.reload(); }, 900);
          syncPolling();
          return d;
        }
        wasRunning = running;
        return d;
      })
      .catch(function () { return null; });
  }

  function syncPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    var needPoll = verbose || wasRunning || (refreshBtn && refreshBtn.disabled);
    if (!needPoll) return;
    var ms = verbose ? 1000 : 4000;
    pollTimer = setInterval(fetchStatus, ms);
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", function () {
      refreshBtn.disabled = true;
      refreshBtn.textContent = "Refreshing…";
      if (dot) dot.classList.add("busy");
      wasRunning = true;
      if (verbose) {
        renderProgress({
          stage: "starting",
          detail: "Starting ingest run",
          current: "",
          done: 0,
          total: 0,
          active: [],
        }, true);
      }
      fetch("/api/refresh", { method: "POST" })
        .then(function (r) {
          if (r.status === 409) { toast("A refresh is already running"); }
          else { toast("Ingest started — this can take a few minutes"); }
          syncPolling();
          fetchStatus();
        })
        .catch(function () {
          toast("Could not reach the server");
          refreshBtn.disabled = false;
          refreshBtn.textContent = "Refresh";
          wasRunning = false;
          syncPolling();
        });
    });
    if (refreshBtn.disabled) {
      if (dot) dot.classList.add("busy");
      wasRunning = true;
    }
  }

  syncPolling();
  syncStickTop();
  window.addEventListener("resize", syncStickTop);
  if (verbose || wasRunning) fetchStatus();

  /* ── keyboard shortcuts ────────────────────────────────────── */
  document.addEventListener("keydown", function (ev) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (typing || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    var go = { d: "/", f: "/feed", s: "/search", b: "/saved", a: "/adapt", h: "/sources", r: "/runs" };
    if (go[ev.key]) { location.href = go[ev.key]; return; }
    if (ev.key === "/") {
      ev.preventDefault();
      var box = document.querySelector('input[type=search]');
      if (box) box.focus(); else location.href = "/search";
    }
  });
})();

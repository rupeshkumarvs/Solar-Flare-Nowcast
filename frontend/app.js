/* Solar Flare Nowcast — frontend engine.
   Every value rendered here comes from a live FastAPI call or a real recorded
   replay. No mock data, no fallback numbers: missing/unavailable data renders an
   honest empty / "not loaded" / "pending" state instead. */

"use strict";

const API = ""; // same origin (served by FastAPI StaticFiles)
const GAUGE_CIRC = 2 * Math.PI * 50; // matches r=50 in the SVG

let replayActive = false;   // when true, the live WS is ignored and the chart shows a replay
let lastClass = null;
let lastAlertLevel = "green";
let lastSpaceWeather = null; // cached /api/spaceweather (used by the L1 transit-time calc)

/* ───────────── Chart.js: live light curve ───────────── */

const CLASS_LINES = [
  { y: 1e-8, label: "A" }, { y: 1e-7, label: "B" }, { y: 1e-6, label: "C" },
  { y: 1e-5, label: "M" }, { y: 1e-4, label: "X" },
];

// Dashed class-threshold lines + labels.
const classLinesPlugin = {
  id: "classLines",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!scales.y) return;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.font = "10px 'JetBrains Mono', monospace";
    for (const line of CLASS_LINES) {
      const y = scales.y.getPixelForValue(line.y);
      if (y < chartArea.top || y > chartArea.bottom) continue;
      ctx.strokeStyle = "rgba(138,150,184,0.22)";
      ctx.beginPath(); ctx.moveTo(chartArea.left, y); ctx.lineTo(chartArea.right, y); ctx.stroke();
      ctx.fillStyle = "rgba(138,150,184,0.7)";
      ctx.fillText(line.label, chartArea.right - 14, y - 3);
    }
    ctx.restore();
  },
};

// Soft glow on the flux lines (applied only during dataset draw).
const glowPlugin = {
  id: "glow",
  beforeDatasetsDraw(chart) {
    const ctx = chart.ctx;
    ctx.save();
    ctx.shadowColor = "rgba(45,212,255,0.45)";
    ctx.shadowBlur = 7;
  },
  afterDatasetsDraw(chart) { chart.ctx.restore(); },
};

// Glowing "now" dot at the right edge of the long-band trace.
const nowMarkerPlugin = {
  id: "nowMarker",
  afterDatasetsDraw(chart) {
    const ds = chart.data.datasets[0].data;
    if (!ds.length || !chart.scales.x) return;
    const last = ds[ds.length - 1];
    const x = chart.scales.x.getPixelForValue(last.x);
    const y = chart.scales.y.getPixelForValue(last.y);
    const ctx = chart.ctx;
    ctx.save();
    ctx.shadowColor = "rgba(45,212,255,0.9)";
    ctx.shadowBlur = 12;
    ctx.fillStyle = "#2dd4ff";
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  },
};

// Replay markers: vertical lines for "alert fired" and "flare peak" — the money shot.
const replayMarkerPlugin = {
  id: "replayMarkers",
  afterDatasetsDraw(chart) {
    const m = chart._markers;
    if (!m) return;
    const { ctx, chartArea, scales } = chart;
    const vline = (epoch, color, label, align) => {
      if (epoch == null) return;
      const x = scales.x.getPixelForValue(epoch);
      if (x < chartArea.left || x > chartArea.right) return;
      ctx.save();
      ctx.setLineDash([5, 4]); ctx.strokeStyle = color; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.font = "bold 10px 'JetBrains Mono', monospace";
      ctx.textAlign = align === "right" ? "right" : "left";
      ctx.fillText(label, align === "right" ? x - 6 : x + 6, chartArea.top + 12);
      ctx.restore();
    };
    vline(m.alert, "#ffb02e", "▲ ALERT FIRED", "left");
    vline(m.peak, "#ff4d5e", "◆ FLARE PEAK", "right");
  },
};

let chart;
function initChart() {
  const ctx = document.getElementById("flux-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        { label: "Long band (0.1–0.8nm)", data: [], borderColor: "#2dd4ff",
          backgroundColor: "rgba(45,212,255,0.08)", borderWidth: 2, pointRadius: 0, tension: 0.25, fill: true },
        { label: "Short band (0.05–0.4nm)", data: [], borderColor: "#ffb02e",
          backgroundColor: "transparent", borderWidth: 1.3, pointRadius: 0, tension: 0.25 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 10 }, maxRotation: 0,
            autoSkipPadding: 24, callback: (v) => new Date(v).toISOString().substr(11, 5) },
          grid: { color: "rgba(30,39,66,0.5)" },
        },
        y: {
          type: "logarithmic", min: 1e-9, max: 1e-3,
          ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 10 },
            callback: (v) => { const e = Math.log10(v); return Number.isInteger(e) ? "1e" + e : ""; } },
          grid: { color: "rgba(30,39,66,0.4)" },
        },
      },
      plugins: {
        legend: { labels: { color: "#8a96b8", font: { family: "'Inter'", size: 11 }, boxWidth: 12, boxHeight: 12 } },
        tooltip: {
          backgroundColor: "#0d1222", borderColor: "#1e2742", borderWidth: 1,
          titleFont: { family: "'JetBrains Mono'" }, bodyFont: { family: "'JetBrains Mono'" },
          callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toExponential(2)} W/m²` },
        },
      },
    },
    plugins: [glowPlugin, classLinesPlugin, replayMarkerPlugin, nowMarkerPlugin],
  });
}

function rebuildChartFromHistory(history) {
  const longData = [], shortData = [];
  for (const h of history) {
    const t = new Date(h.time).getTime();
    if (h.long != null) longData.push({ x: t, y: Math.max(h.long, 1e-9) });
    if (h.short != null) shortData.push({ x: t, y: Math.max(h.short, 1e-9) });
  }
  chart.data.datasets[0].data = longData;
  chart.data.datasets[1].data = shortData;
  chart.update();
}

function pushChartPoint(point) {
  const t = new Date(point.time).getTime();
  const last = chart.data.datasets[0].data;
  if (last.length && last[last.length - 1].x === t) return;
  if (point.long != null) chart.data.datasets[0].data.push({ x: t, y: Math.max(point.long, 1e-9) });
  if (point.short != null) chart.data.datasets[1].data.push({ x: t, y: Math.max(point.short, 1e-9) });
  const cutoff = t - 6 * 3600 * 1000;
  chart.data.datasets[0].data = chart.data.datasets[0].data.filter((p) => p.x >= cutoff);
  chart.data.datasets[1].data = chart.data.datasets[1].data.filter((p) => p.x >= cutoff);
  chart.update();
}

function clearChart() {
  chart.data.datasets[0].data = [];
  chart.data.datasets[1].data = [];
  chart._markers = null;
  chart.update();
}

/* ───────────── Class badge + hero status ───────────── */
function classLetter(cls) {
  if (!cls || cls === "Below A") return "na";
  const c = cls[0].toUpperCase();
  return ["A", "B", "C", "M", "X"].includes(c) ? c : "na";
}
function updateClassBadge(cls) {
  lastClass = cls;
  const el = document.getElementById("class-badge");
  el.textContent = cls && cls !== "Below A" ? cls : "—";
  el.className = "class-badge cls-" + classLetter(cls);
  updateHero();
}
function updateHero() {
  const word = document.getElementById("hero-word");
  const L = classLetter(lastClass);
  let state, label;
  if (L === "M" || L === "X") { state = "flare"; label = "FLARE IN PROGRESS"; }
  else if (L === "C" || lastAlertLevel === "red" || lastAlertLevel === "yellow") { state = "active"; label = "ACTIVE"; }
  else { state = "quiet"; label = "QUIET"; }
  word.textContent = label;
  word.className = "hero-word " + state;
}

/* ───────────── Forecast gauges ───────────── */
function levelClass(p) { return p >= 0.6 ? "lvl-red" : p >= 0.3 ? "lvl-amber" : "lvl-green"; }
function setGauge(el, prob) {
  const fill = el.querySelector(".g-fill");
  const val = el.querySelector(".g-val");
  el.classList.remove("lvl-green", "lvl-amber", "lvl-red", "lvl-off");
  if (prob == null || Number.isNaN(prob)) {
    el.classList.add("lvl-off");
    fill.style.strokeDashoffset = GAUGE_CIRC;
    val.textContent = "—";
    return;
  }
  el.classList.add(levelClass(prob));
  fill.style.strokeDashoffset = GAUGE_CIRC * (1 - prob);
  val.textContent = Math.round(prob * 100) + "%";
}

function renderForecast(fc) {
  const statusEl = document.getElementById("forecast-status");
  const trendEl = document.getElementById("trend-indicator");
  const gauges = {
    10: document.querySelector('.gauge[data-h="10"]'),
    30: document.querySelector('.gauge[data-h="30"]'),
    60: document.querySelector('.gauge[data-h="60"]'),
  };

  if (!fc || fc.error || fc.model_loaded === false) {
    setGauge(gauges[10], null); setGauge(gauges[30], null); setGauge(gauges[60], null);
    statusEl.textContent = (fc && (fc.error || fc.status)) || "Model not loaded";
    statusEl.className = "forecast-status err";
    trendEl.textContent = "—"; trendEl.className = "trend-indicator";
    lastAlertLevel = "unknown"; updateAlertBanner(null); updateHero();
    return;
  }

  setGauge(gauges[10], fc.prob_10min);
  setGauge(gauges[30], fc.prob_30min);
  setGauge(gauges[60], fc.prob_60min);

  const trend = fc.trend || "unknown";
  const arrow = { rising: "▲ RISING", falling: "▼ FALLING", flat: "▬ FLAT" }[trend] || "— " + trend.toUpperCase();
  trendEl.textContent = arrow;
  trendEl.className = "trend-indicator " + (["rising", "falling", "flat"].includes(trend) ? trend : "");

  if (fc.prob_10min == null) {
    statusEl.textContent = fc.status || "Warming up…";
    statusEl.className = "forecast-status warn";
  } else {
    const lt = fc.lead_time_estimate;
    statusEl.textContent = lt ? `Estimated lead time: ${lt} min` : "Live forecast active";
    statusEl.className = "forecast-status";
  }

  // Red-alert FX: fire once on transition into red (screen pulse + gauge flash).
  const level = fc.alert_level || "green";
  if (level === "red" && lastAlertLevel !== "red") triggerRedFx(gauges);
  lastAlertLevel = level;
  updateAlertBanner(fc);
  updateHero();
}

function triggerRedFx(gauges) {
  const sp = document.getElementById("screen-pulse");
  sp.classList.remove("flash"); void sp.offsetWidth; sp.classList.add("flash");
  for (const h of [10, 30, 60]) {
    const g = gauges[h];
    if (g && g.classList.contains("lvl-red")) {
      g.classList.remove("flash"); void g.offsetWidth; g.classList.add("flash");
    }
  }
}

/* ───────────── Alert banner ───────────── */
function updateAlertBanner(fc) {
  const banner = document.getElementById("alert-banner");
  const text = document.getElementById("alert-text");
  const level = fc && fc.alert_level;
  if (!fc || level === "green" || level === "unknown" || !level) {
    banner.className = "alert-banner hidden";
    return;
  }
  const p30 = fc.prob_30min != null ? Math.round(fc.prob_30min * 100) + "%" : "—";
  const trend = fc.trend || "—";
  if (level === "red") {
    banner.className = "alert-banner red";
    text.textContent = `HIGH FLARE RISK — Trend ${trend}, 30-min M+ probability ${p30}`;
  } else {
    banner.className = "alert-banner amber";
    text.textContent = `ELEVATED FLARE RISK — Trend ${trend}, 30-min M+ probability ${p30}`;
  }
}

/* ───────────── Status pill ───────────── */
function setStatus(state, timeStr, stale) {
  const pill = document.getElementById("status-pill");
  const txt = document.getElementById("status-text");
  const time = document.getElementById("status-time");
  pill.className = "status-pill " + state + (stale ? " stale" : "");
  txt.textContent = stale ? "STALE" : ({ live: "LIVE", connecting: "RECONNECTING…", down: "OFFLINE" }[state] || state);
  if (timeStr) {
    const d = new Date(timeStr);
    time.textContent = "· " + d.toISOString().substr(11, 8) + "Z";
  }
}

/* ───────────── Catalog table ───────────── */
function fmtTime(t) {
  if (!t) return "—";
  try { return new Date(t.includes("T") ? t : t.replace(" ", "T") + "Z").toISOString().substr(0, 19).replace("T", " "); }
  catch { return t; }
}
function duration(start, end) {
  if (!start || !end) return "—";
  try {
    const ms = new Date(end.replace(" ", "T")) - new Date(start.replace(" ", "T"));
    if (Number.isNaN(ms) || ms < 0) return "—";
    return Math.round(ms / 60000) + " min";
  } catch { return "—"; }
}
function renderCatalog(rows) {
  const body = document.getElementById("catalog-body");
  if (!Array.isArray(rows) || rows.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="table-empty">No flares detected yet in current session</td></tr>';
    return;
  }
  body.innerHTML = rows.map((r) => {
    const cl = classLetter(r.class);
    return `<tr>
      <td>${fmtTime(r.start_time)}</td><td>${fmtTime(r.peak_time)}</td><td>${fmtTime(r.end_time)}</td>
      <td class="cat-class cls-${cl}" style="border:none;background:none">${r.class || "—"}</td>
      <td>${r.peak_flux != null ? Number(r.peak_flux).toExponential(2) : "—"}</td>
      <td>${duration(r.start_time, r.end_time)}</td>
    </tr>`;
  }).join("");
}

/* ───────────── Data source chips + Aditya precursor ───────────── */
function setChip(id, state, label) {
  const el = document.getElementById(id);
  el.className = "chip " + state;
  el.querySelector("b").textContent = label;
}

async function refreshSourceChips() {
  try {
    const cur = await fetchJSON("/api/current");
    const ok = cur && !cur.__status && cur.long != null;
    const stale = ok && cur.stale;
    setChip("chip-goes", ok ? (stale ? "pending" : "live") : "down", ok ? (stale ? "STALE" : "LIVE") : "DOWN");
  } catch { setChip("chip-goes", "down", "DOWN"); }
  await refreshAditya();
}

// Aditya-L1 hard X-ray precursor. Honest pending state until real FITS arrive via PRADAN.
async function refreshAditya() {
  const panel = document.getElementById("precursor");
  const state = document.getElementById("precursor-state");
  const fill = document.getElementById("precursor-fill");
  const valEl = document.getElementById("precursor-val");
  try {
    const a = await fetchJSON("/api/aditya/status");
    if (a && a.fits_loaded) {
      setChip("chip-aditya", "live", "LIVE");
      let ratio = null;
      try {
        const series = await fetchJSON("/api/aditya/series");
        if (Array.isArray(series) && series.length) {
          const lastWithRatio = [...series].reverse().find((p) => p.hard_soft_ratio != null);
          if (lastWithRatio) ratio = lastWithRatio.hard_soft_ratio;
        }
      } catch {}
      panel.className = "precursor";
      state.className = "precursor-state live";
      state.textContent = `LIVE · ${a.solexs_points + a.hel1os_points} pts`;
      if (ratio != null) {
        valEl.textContent = ratio.toFixed(3);
        fill.style.width = Math.max(2, Math.min(100, ratio * 100)) + "%";
      } else {
        valEl.textContent = "—";
        fill.style.width = "0%";
      }
    } else {
      setChip("chip-aditya", "pending", "Pending PRADAN");
      panel.className = "precursor disabled";
      state.className = "precursor-state pending";
      state.textContent = "Awaiting Aditya-L1 via PRADAN";
      valEl.textContent = "—";
      fill.style.width = "0%";
    }
  } catch {
    setChip("chip-aditya", "pending", "Pending PRADAN");
    panel.className = "precursor disabled";
  }
}

async function refreshDonkiChip() {
  try {
    const r = await fetch(API + "/api/donki?days=1");
    setChip("chip-donki", r.ok ? "live" : "down", r.ok ? "LIVE" : "DOWN");
  } catch { setChip("chip-donki", "down", "DOWN"); }
}

/* ───────────── Live space-weather context (keyless NOAA feeds) ───────────── */
function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function swClass(el, cls) { if (!el) return; el.classList.remove("quiet", "elevated", "storm"); if (cls) el.classList.add(cls); }
function swSci(x) { if (x == null) return "—"; return (x >= 100 || x < 0.01) ? Number(x).toExponential(1) : Number(x).toFixed(2); }
const G_LABEL = { G0: "G0 · quiet", G1: "G1 · minor storm", G2: "G2 · moderate", G3: "G3 · strong", G4: "G4 · severe", G5: "G5 · extreme" };

async function refreshSpaceWeather() {
  const d = await fetchJSON("/api/spaceweather");
  if (!d || d.__status) return; // honest: keep placeholders until first real fetch
  lastSpaceWeather = d;

  // Proton flux → S-scale radiation storm
  const p = d.proton;
  if (p) {
    setText("sw-proton-val", p.storm || "—");
    setText("sw-proton-sub", p.flux_10mev != null ? `${swSci(p.flux_10mev)} pfu ≥10 MeV` : "—");
    swClass(document.getElementById("sw-proton"), p.storm === "S0" ? "quiet" : ["S1", "S2"].includes(p.storm) ? "elevated" : "storm");
  }
  // Solar wind → speed + Bz
  const w = d.solar_wind;
  if (w) {
    setText("sw-wind-val", w.speed != null ? `${Math.round(w.speed)} km/s` : "—");
    setText("sw-wind-sub", `Bz ${w.bz != null ? w.bz.toFixed(1) : "—"} · Bt ${w.bt != null ? w.bt.toFixed(1) : "—"} nT`);
    swClass(document.getElementById("sw-wind"), w.bz == null ? "" : w.bz <= -10 ? "storm" : w.bz <= -5 ? "elevated" : "quiet");
  }
  // Kp → G-scale geomagnetic
  const k = d.kp;
  if (k) {
    setText("sw-kp-val", k.value != null ? k.value.toFixed(2) : "—");
    setText("sw-kp-sub", G_LABEL[k.storm] || k.storm || "—");
    swClass(document.getElementById("sw-kp"), k.storm === "G0" ? "quiet" : ["G1", "G2"].includes(k.storm) ? "elevated" : "storm");
  }
  // F10.7 radio flux
  if (d.f107) setText("sw-f107-val", d.f107.flux != null ? Math.round(d.f107.flux) : "—");
  // Active regions → count + delta (flare-productive) flag
  const r = d.regions;
  if (r) {
    setText("sw-regions-val", r.count != null ? r.count : "—");
    const nd = (r.flare_productive || []).length;
    setText("sw-regions-sub", nd > 0 ? `${nd} δ flare-productive` : r.count ? "no delta regions" : "—");
    swClass(document.getElementById("sw-regions"), nd > 0 ? "storm" : r.count ? "quiet" : "");
  }
  // NOAA SWPC's OWN flare forecast (Radio Blackout R1-R2 = M, R3+ = X)
  const f = d.swpc_forecast;
  if (f) {
    const m = f.m_class && f.m_class.length ? f.m_class[0] : null;
    const x = f.x_class && f.x_class.length ? f.x_class[0] : null;
    setText("sw-noaa-val", `M ${m ?? "—"}% · X ${x ?? "—"}%`);
    setText("sw-noaa-sub", f.issued ? `issued ${f.issued}` : "official · next 24h");
    swClass(document.getElementById("sw-noaa"), m == null ? "" : (m >= 50 || x >= 10) ? "storm" : m >= 25 ? "elevated" : "quiet");
    setText("noaa-m", m != null ? m + "%" : "—");
    setText("noaa-x", x != null ? x + "%" : "—");
  }
}

/* ───────────── Fetch helpers ───────────── */
async function fetchJSON(path) {
  const r = await fetch(API + path);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    return { __status: r.status, ...err };
  }
  return r.json();
}
async function loadCatalog() {
  const rows = await fetchJSON("/api/catalog");
  renderCatalog(Array.isArray(rows) ? rows : []);
}

/* ───────────── Live WebSocket with backoff ───────────── */
let ws, backoff = 1000;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { backoff = 1000; };
  ws.onmessage = (ev) => {
    if (replayActive) return; // replay owns the chart/gauges while active
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg) return;
    setStatus("live", msg.updated || msg.time, msg.stale);
    updateClassBadge(msg.class);
    pushChartPoint(msg);
    if (msg.forecast) renderForecast(msg.forecast);
  };
  ws.onclose = () => { setStatus("connecting"); setTimeout(connectWS, backoff); backoff = Math.min(backoff * 2, 30000); };
  ws.onerror = () => { try { ws.close(); } catch {} };
}

/* ═════════════ REPLAY MODE ═════════════ */
let replayWS = null;

async function populateReplayDropdown() {
  const sel = document.getElementById("replay-select");
  try {
    const events = await fetchJSON("/api/replay/events");
    if (!Array.isArray(events) || !events.length) return;
    for (const e of events) {
      const opt = document.createElement("option");
      opt.value = e.id;
      const d = new Date(e.peak_time).toISOString().substr(0, 16).replace("T", " ");
      opt.textContent = `${e.class} — ${d}Z`;
      sel.appendChild(opt);
    }
  } catch (e) { console.warn("replay events load failed", e); }
}

function startReplay(eventId) {
  if (!eventId) return;
  exitReplay(true); // tear down any prior replay socket, keep flag handling below
  replayActive = true;

  document.getElementById("replay-exit").classList.remove("hidden");
  document.getElementById("replay-play").disabled = true;
  const banner = document.getElementById("replay-banner");
  banner.classList.remove("hidden");

  clearChart();
  setStatus("live"); // pill stays calm; replay banner carries the context

  const proto = location.protocol === "https:" ? "wss" : "ws";
  replayWS = new WebSocket(`${proto}://${location.host}/ws?mode=replay&event=${encodeURIComponent(eventId)}&speed=600`);

  replayWS.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === "replay_meta") {
      document.getElementById("replay-banner-label").textContent = m.label;
      const lead = m.lead_time_min;
      document.getElementById("replay-lead").textContent =
        lead != null ? `lead ${lead} min before peak` : "lead —";
      chart._markers = {
        alert: m.alert_fired_time ? new Date(m.alert_fired_time).getTime() : null,
        peak: m.peak_time ? new Date(m.peak_time).getTime() : null,
      };
    } else if (m.type === "replay_frame") {
      updateClassBadge(m.class);
      pushChartPoint(m);
      if (m.forecast) renderForecast(m.forecast);
      document.getElementById("replay-progress").textContent = `frame ${m.frame + 1} / ${m.total}`;
    } else if (m.type === "replay_done") {
      document.getElementById("replay-progress").textContent = "complete";
    } else if (m.type === "replay_error") {
      document.getElementById("replay-banner-label").textContent = "replay error: " + (m.error || "unknown");
    }
  };
  replayWS.onclose = () => { document.getElementById("replay-play").disabled = false; };
  replayWS.onerror = () => { try { replayWS.close(); } catch {} };
}

function exitReplay(silent) {
  if (replayWS) { try { replayWS.close(); } catch {} replayWS = null; }
  if (silent) return;
  replayActive = false;
  document.getElementById("replay-exit").classList.add("hidden");
  document.getElementById("replay-play").disabled = !document.getElementById("replay-select").value;
  document.getElementById("replay-banner").classList.add("hidden");
  chart._markers = null;
  // Restore the live view from the real history buffer.
  fetchJSON("/api/history").then((h) => { if (Array.isArray(h)) rebuildChartFromHistory(h); });
  fetchJSON("/api/current").then((c) => {
    if (c && !c.__status) { updateClassBadge(c.class); if (c.forecast) renderForecast(c.forecast); setStatus("live", c.updated, c.stale); }
  });
}

function wireReplayControls() {
  const sel = document.getElementById("replay-select");
  const play = document.getElementById("replay-play");
  sel.addEventListener("change", () => { play.disabled = !sel.value; });
  play.addEventListener("click", () => startReplay(sel.value));
  document.getElementById("replay-exit").addEventListener("click", () => exitReplay(false));
}

/* ═════════════ MODEL PERFORMANCE PANEL ═════════════ */
let perfData = null, perfHorizon = "10";
let leadHistChart = null, baselineChart = null, calibChart = null;

const fmtPct = (x) => (x == null || Number.isNaN(x) ? "—" : (x * 100).toFixed(1) + "%");
const fmt2 = (x) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(3));

async function loadPerformance() {
  const note = document.getElementById("perf-note");
  const data = await fetchJSON("/api/metrics");
  if (!data || data.__status) {
    note.textContent = "Model performance metrics not available yet — run `python -m backend.train` to generate them.";
    return;
  }
  perfData = data;
  const tr = data.data.test_range, rng = data.data.train_range;
  note.innerHTML = `Evaluated on <b>chronologically held-out</b> data — no shuffling, no leakage. ` +
    `Train: ${rng[0]} → ${rng[1]} · Test: ${tr[0]} → ${tr[1]} · ${data.data.rows.toLocaleString()} minutes of real GOES XRS. ` +
    `Alert threshold ${data.alert_threshold}.`;
  renderPerfCards();
  renderLeadHist();
  renderBaselineBar();
  renderCalibration();
  renderConfusion();
}

function renderPerfCards() {
  const wrap = document.getElementById("perf-cards");
  wrap.innerHTML = ["10", "30", "60"].map((h) => {
    const e = perfData.horizons[h].enhanced;
    const m = e.metrics, lt = e.lead_time;
    const tssCls = m.TSS >= 0.5 ? "good" : m.TSS >= 0.3 ? "warn" : "";
    const farCls = m.FAR <= 0.1 ? "good" : m.FAR <= 0.3 ? "warn" : "";
    return `<div class="perf-card">
      <h3>Horizon ${h} min</h3>
      <div class="pc-sub">≥M-class within next ${h} minutes</div>
      <div class="pc-grid">
        <div class="pc-metric"><span class="k">TSS</span><span class="v ${tssCls}">${fmt2(m.TSS)}</span></div>
        <div class="pc-metric"><span class="k">HSS</span><span class="v">${fmt2(m.HSS)}</span></div>
        <div class="pc-metric"><span class="k">Precision</span><span class="v">${fmtPct(m.Precision)}</span></div>
        <div class="pc-metric"><span class="k">Recall (POD)</span><span class="v">${fmtPct(m.Recall)}</span></div>
        <div class="pc-metric"><span class="k">False Alarm Rate</span><span class="v ${farCls}">${fmtPct(m.FAR)}</span></div>
        <div class="pc-metric"><span class="k">ROC-AUC</span><span class="v">${fmt2(m["ROC-AUC"])}</span></div>
      </div>
      <div class="pc-lead">
        <div class="lead-head">LEAD TIME BEFORE FLARE PEAK · ${lt.n_caught}/${lt.n_events} events caught</div>
        <div class="lead-row"><span>median</span><b>${lt.median_min} min</b></div>
        <div class="lead-row"><span>p25 / p75</span><span>${lt.p25_min} / ${lt.p75_min} min</span></div>
        <div class="lead-row"><span>max</span><span>${lt.max_min} min</span></div>
      </div>
    </div>`;
  }).join("");
}

function histogram(values, binWidth, maxVal) {
  const bins = [];
  const top = Math.min(maxVal, Math.ceil(Math.max(...values, 0) / binWidth) * binWidth);
  for (let lo = 0; lo < top; lo += binWidth) bins.push({ lo, hi: lo + binWidth, n: 0 });
  if (!bins.length) bins.push({ lo: 0, hi: binWidth, n: 0 });
  for (const v of values) {
    let idx = Math.floor(v / binWidth);
    if (idx >= bins.length) idx = bins.length - 1;
    if (idx < 0) idx = 0;
    bins[idx].n++;
  }
  return bins;
}

function renderLeadHist() {
  const lt = perfData.horizons[perfHorizon].enhanced.lead_time;
  const bins = histogram(lt.lead_times || [], 10, 120);
  const labels = bins.map((b) => `${b.lo}–${b.hi}`);
  const counts = bins.map((b) => b.n);
  const ctx = document.getElementById("lead-hist").getContext("2d");
  const cfg = {
    type: "bar",
    data: { labels, datasets: [{ label: `Events (n=${(lt.lead_times || []).length})`, data: counts,
      backgroundColor: "rgba(255,176,46,0.55)", borderColor: "#ffb02e", borderWidth: 1 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8a96b8", font: { family: "'Inter'", size: 11 } } },
        tooltip: { callbacks: { title: (i) => `Lead ${i[0].label} min` } } },
      scales: {
        x: { title: { display: true, text: "minutes before flare peak", color: "#5a6488", font: { size: 10 } },
          ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 9 } }, grid: { color: "rgba(30,39,66,0.4)" } },
        y: { ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 10 }, precision: 0 }, grid: { color: "rgba(30,39,66,0.4)" } },
      },
    },
  };
  if (leadHistChart) leadHistChart.destroy();
  leadHistChart = new Chart(ctx, cfg);
}

function renderBaselineBar() {
  const hs = ["10", "30", "60"];
  const modelTSS = hs.map((h) => perfData.horizons[h].enhanced.metrics.TSS);
  const persTSS = hs.map((h) => perfData.horizons[h].persistence.metrics.TSS);
  const modelHSS = hs.map((h) => perfData.horizons[h].enhanced.metrics.HSS);
  const persHSS = hs.map((h) => perfData.horizons[h].persistence.metrics.HSS);
  const ctx = document.getElementById("baseline-bar").getContext("2d");
  const cfg = {
    type: "bar",
    data: { labels: hs.map((h) => h + " min"),
      datasets: [
        { label: "Model TSS", data: modelTSS, backgroundColor: "rgba(45,212,255,0.75)" },
        { label: "Persistence TSS", data: persTSS, backgroundColor: "rgba(90,100,136,0.6)" },
        { label: "Model HSS", data: modelHSS, backgroundColor: "rgba(40,209,124,0.75)" },
        { label: "Persistence HSS", data: persHSS, backgroundColor: "rgba(90,100,136,0.35)" },
      ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8a96b8", font: { family: "'Inter'", size: 10 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 10 } }, grid: { display: false } },
        y: { min: 0, max: 1, ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 10 } }, grid: { color: "rgba(30,39,66,0.4)" } },
      },
    },
  };
  if (baselineChart) baselineChart.destroy();
  baselineChart = new Chart(ctx, cfg);
}

function renderCalibration() {
  const cal = perfData.horizons[perfHorizon].enhanced.calibration;
  const pts = (cal.mean_predicted || []).map((x, i) => ({ x, y: cal.fraction_positive[i] }));
  const ctx = document.getElementById("calib-curve").getContext("2d");
  const cfg = {
    type: "line",
    data: { datasets: [
      { label: "Model", data: pts, borderColor: "#2dd4ff", backgroundColor: "rgba(45,212,255,0.1)",
        borderWidth: 2, pointRadius: 3, pointBackgroundColor: "#2dd4ff", tension: 0.2, showLine: true },
      { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "rgba(138,150,184,0.5)",
        borderDash: [5, 4], borderWidth: 1, pointRadius: 0 },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8a96b8", font: { family: "'Inter'", size: 10 } } } },
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "predicted probability", color: "#5a6488", font: { size: 10 } },
          ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 9 } }, grid: { color: "rgba(30,39,66,0.4)" } },
        y: { min: 0, max: 1, title: { display: true, text: "observed frequency", color: "#5a6488", font: { size: 10 } },
          ticks: { color: "#5a6488", font: { family: "'JetBrains Mono'", size: 9 } }, grid: { color: "rgba(30,39,66,0.4)" } },
      },
    },
  };
  if (calibChart) calibChart.destroy();
  calibChart = new Chart(ctx, cfg);
}

function renderConfusion() {
  const m = perfData.horizons[perfHorizon].enhanced.metrics;
  const el = document.getElementById("confusion");
  el.innerHTML = `
    <div class="cm-cell cm-corner"></div>
    <div class="cm-cell cm-head">Predicted: Flare</div>
    <div class="cm-cell cm-head">Predicted: Quiet</div>
    <div class="cm-cell cm-head" style="writing-mode:vertical-rl;transform:rotate(180deg)">Actual: Flare</div>
    <div class="cm-cell cm-tp"><div class="cm-v">${m.TP.toLocaleString()}</div><div class="cm-k">True Positive</div></div>
    <div class="cm-cell cm-fn"><div class="cm-v">${m.FN.toLocaleString()}</div><div class="cm-k">False Negative (missed)</div></div>
    <div class="cm-cell cm-head" style="writing-mode:vertical-rl;transform:rotate(180deg)">Actual: Quiet</div>
    <div class="cm-cell cm-fp"><div class="cm-v">${m.FP.toLocaleString()}</div><div class="cm-k">False Positive (false alarm)</div></div>
    <div class="cm-cell cm-tn"><div class="cm-v">${m.TN.toLocaleString()}</div><div class="cm-k">True Negative</div></div>`;
}

function wirePerfControls() {
  document.querySelectorAll("#lead-toggle .htog").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#lead-toggle .htog").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      perfHorizon = btn.dataset.h;
      if (perfData) { renderLeadHist(); renderCalibration(); renderConfusion(); }
    });
  });
}

/* ═════════════ SATELLITE TRACKING (live, real positions) ═════════════ */
const SAT_COLORS = { xray: "#2dd4ff", solar: "#ffb02e", target: "#ff8a3d", wind: "#28d17c", leo: "#b388ff" };
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

let satState = null, landGeo = null, trackTimer = null, trackActive = false, landLoading = false;
let satrecs = null;  // parsed TLEs for client-side per-second SGP4 propagation
const satTrails = {}; // recent ground-track points per fast LEO satellite (motion trail)

// Parse the live TLEs into SGP4 records once; we then propagate them every second.
async function loadTLEs() {
  if (typeof satellite === "undefined") { satrecs = null; return; }
  const d = await fetchJSON("/api/satellites/tle");
  if (!d || d.__status || !Array.isArray(d.satellites)) { satrecs = null; return; }
  satrecs = [];
  for (const s of d.satellites) {
    try {
      const rec = satellite.twoline2satrec(s.line1, s.line2);
      satrecs.push({ name: s.name, kind: s.kind, role: s.role, rec });
    } catch (e) { /* skip a bad TLE */ }
  }
}

// Propagate every satellite to `date` in the browser → real live sub-satellite point.
function computeEarthSats(date) {
  if (!satrecs || typeof satellite === "undefined") {
    return satState ? satState.earth_satellites : [];
  }
  const gmst = satellite.gstime(date);
  return satrecs.map((s) => {
    try {
      const pv = satellite.propagate(s.rec, date);
      if (!pv || !pv.position) return { name: s.name, kind: s.kind, role: s.role, position: null, tracked: false };
      const geo = satellite.eciToGeodetic(pv.position, gmst);
      return {
        name: s.name, kind: s.kind, role: s.role, tracked: true,
        position: { lat: satellite.degreesLat(geo.latitude), lon: satellite.degreesLong(geo.longitude), alt_km: geo.height },
      };
    } catch (e) {
      return { name: s.name, kind: s.kind, role: s.role, position: null, tracked: false };
    }
  });
}

// Real sub-solar point from UTC (mirrors backend astronomy; updates live every second).
function subsolarPoint(date) {
  const jd = 2440587.5 + date.getTime() / 86400000;
  const n = jd - 2451545.0;
  const L = (280.460 + 0.9856474 * n) % 360;
  const g = rad((357.528 + 0.9856003 * n) % 360);
  const lam = rad(L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g));
  const eps = rad(23.439 - 0.0000004 * n);
  const decl = deg(Math.asin(Math.sin(eps) * Math.sin(lam)));
  const ra = (deg(Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam))) + 360) % 360;
  const eot = ((L - ra + 540) % 360) - 180;
  const utcH = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  let lon = -(15 * (utcH - 12) + eot);
  lon = ((lon + 540) % 360) - 180;
  return { lat: decl, lon };
}

function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(2, Math.round(r.width * dpr));
  cv.height = Math.max(2, Math.round(r.height * dpr));
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

async function loadLand() {
  if (landGeo || landLoading || typeof topojson === "undefined") return;
  landLoading = true;
  try {
    const r = await fetch("https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json");
    const topo = await r.json();
    landGeo = topojson.feature(topo, topo.objects.land);
  } catch (e) { landGeo = null; }   // graceful: map shows graticule only
  landLoading = false;
}

function drawWorldMap(state, ss) {
  const cv = document.getElementById("world-map");
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  const P = (lon, lat) => [((lon + 180) / 360) * w, ((90 - lat) / 180) * h];

  ctx.fillStyle = "#0a1024"; ctx.fillRect(0, 0, w, h);

  // continents (real world-atlas land). topojson.feature() yields a FeatureCollection.
  if (landGeo) {
    ctx.fillStyle = "#1b2a4e"; ctx.strokeStyle = "rgba(90,100,136,0.4)"; ctx.lineWidth = 0.6;
    const feats = landGeo.type === "FeatureCollection" ? landGeo.features : [landGeo];
    for (const f of feats) {
      const geom = f && f.geometry;
      if (!geom) continue;
      const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
      for (const poly of polys) {
        ctx.beginPath();
        for (const ring of poly) {
          ring.forEach((c, i) => { const [x, y] = P(c[0], c[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
          ctx.closePath();
        }
        ctx.fill(); ctx.stroke();
      }
    }
  }

  // graticule
  ctx.strokeStyle = "rgba(45,212,255,0.08)"; ctx.lineWidth = 1;
  for (let lon = -150; lon <= 150; lon += 30) { ctx.beginPath(); const [x] = P(lon, 0); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let lat = -60; lat <= 60; lat += 30) { ctx.beginPath(); const [, y] = P(0, lat); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  ctx.strokeStyle = "rgba(45,212,255,0.18)"; ctx.beginPath(); const [, eqy] = P(0, 0); ctx.moveTo(0, eqy); ctx.lineTo(w, eqy); ctx.stroke();

  // live day/night terminator
  drawTerminator(ctx, w, h, P, ss);

  // sub-solar point (the Sun overhead)
  const [sx, sy] = P(ss.lon, ss.lat);
  const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, 22);
  grd.addColorStop(0, "rgba(255,210,90,0.9)"); grd.addColorStop(1, "rgba(255,160,46,0)");
  ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(sx, sy, 22, 0, 7); ctx.fill();
  ctx.fillStyle = "#ffd45e"; ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 7); ctx.fill();

  // motion trails for the fast LEO movers (so live motion is visible)
  for (const name in satTrails) {
    const trail = satTrails[name];
    if (!trail || trail.length < 2) continue;
    ctx.strokeStyle = SAT_COLORS.leo; ctx.lineWidth = 1.4;
    let prevLon = null, pen = false;
    for (let i = 0; i < trail.length; i++) {
      const [x, y] = P(trail[i].lon, trail[i].lat);
      ctx.globalAlpha = 0.05 + 0.32 * (i / trail.length); // fade in toward the satellite
      if (prevLon != null && Math.abs(trail[i].lon - prevLon) > 180) pen = false; // dateline break
      if (!pen) { ctx.beginPath(); ctx.moveTo(x, y); pen = true; }
      else { ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); }
      prevLon = trail[i].lon;
    }
    ctx.globalAlpha = 1;
  }

  // satellites + GEO footprints
  for (const s of state.earth_satellites) {
    if (!s.position) continue;
    const color = SAT_COLORS[s.kind] || "#8a96b8";
    const { lon, lat, alt_km } = s.position;
    if (alt_km > 20000) drawFootprint(ctx, P, lat, lon, alt_km, color);
    const [x, y] = P(lon, lat);
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = 8;
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
    ctx.restore();
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    const rightSide = x > w - 70;
    ctx.textAlign = rightSide ? "right" : "left";
    ctx.fillStyle = "#e6ecff"; ctx.fillText(s.name, x + (rightSide ? -7 : 7), y + 3);
    ctx.textAlign = "left";
  }
}

function drawTerminator(ctx, w, h, P, ss) {
  const lat0 = rad(ss.lat || 1e-6);
  ctx.beginPath();
  let first = true;
  for (let lon = -180; lon <= 180; lon += 2) {
    const H = rad(lon - ss.lon);
    const latT = deg(Math.atan(-Math.cos(H) / Math.tan(lat0)));
    const [x, y] = P(lon, latT);
    first ? (ctx.moveTo(x, y), (first = false)) : ctx.lineTo(x, y);
  }
  if (ss.lat >= 0) { ctx.lineTo(w, h); ctx.lineTo(0, h); } else { ctx.lineTo(w, 0); ctx.lineTo(0, 0); }
  ctx.closePath();
  ctx.fillStyle = "rgba(3,6,20,0.62)"; ctx.fill();
}

function drawFootprint(ctx, P, lat, lon, alt, color) {
  const R = 6371, lam = Math.acos(R / (R + alt));
  const f1 = rad(lat), l1 = rad(lon);
  ctx.save(); ctx.beginPath();
  let prevLon = null, started = false;
  for (let b = 0; b <= 360; b += 5) {
    const t = rad(b);
    const f2 = Math.asin(Math.sin(f1) * Math.cos(lam) + Math.cos(f1) * Math.sin(lam) * Math.cos(t));
    const l2 = l1 + Math.atan2(Math.sin(t) * Math.sin(lam) * Math.cos(f1), Math.cos(lam) - Math.sin(f1) * Math.sin(f2));
    const lon2 = ((deg(l2) + 540) % 360) - 180, lat2 = deg(f2);
    const [x, y] = P(lon2, lat2);
    if (!started || (prevLon != null && Math.abs(lon2 - prevLon) > 180)) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    started = true; prevLon = lon2;
  }
  ctx.strokeStyle = color; ctx.globalAlpha = 0.32; ctx.lineWidth = 1.2; ctx.stroke();
  ctx.restore();
}

function drawL1(state, sw) {
  const cv = document.getElementById("l1-map");
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.fillStyle = "#0a1024"; ctx.fillRect(0, 0, w, h);
  const cy = h * 0.5, sunX = w * 0.12, earthX = w * 0.85, l1X = earthX - (earthX - sunX) * 0.22;

  // Sun–Earth line
  ctx.strokeStyle = "rgba(90,100,136,0.4)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(sunX, cy); ctx.lineTo(earthX, cy); ctx.stroke(); ctx.setLineDash([]);

  // live solar-wind arrow Sun → Earth, labelled with real transit time
  let transit = "—";
  const speed = sw && sw.solar_wind && sw.solar_wind.speed;
  if (speed) {
    const mins = (state.geometry.l1_distance_km / speed) / 60;
    transit = `≈ ${Math.round(mins)} min @ ${Math.round(speed)} km/s`;
  }
  ctx.strokeStyle = "rgba(45,212,255,0.5)"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(sunX + 26, cy - 26); ctx.lineTo(l1X, cy - 26); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(l1X, cy - 26); ctx.lineTo(l1X - 7, cy - 30); ctx.lineTo(l1X - 7, cy - 22); ctx.closePath();
  ctx.fillStyle = "rgba(45,212,255,0.6)"; ctx.fill();
  ctx.fillStyle = "#8a96b8"; ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
  ctx.fillText("solar wind " + transit, (sunX + l1X) / 2, cy - 32);

  // Sun
  const sg = ctx.createRadialGradient(sunX, cy, 0, sunX, cy, 34);
  sg.addColorStop(0, "rgba(255,200,80,0.95)"); sg.addColorStop(0.6, "rgba(255,140,20,0.5)"); sg.addColorStop(1, "rgba(255,120,0,0)");
  ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(sunX, cy, 34, 0, 7); ctx.fill();
  ctx.fillStyle = "#ffd45e"; ctx.beginPath(); ctx.arc(sunX, cy, 16, 0, 7); ctx.fill();
  ctx.fillStyle = "#ffb02e"; ctx.textAlign = "center"; ctx.font = "bold 11px 'JetBrains Mono'"; ctx.fillText("SUN", sunX, cy + 50);

  // Earth
  ctx.fillStyle = "#2c66c9"; ctx.beginPath(); ctx.arc(earthX, cy, 11, 0, 7); ctx.fill();
  ctx.strokeStyle = "rgba(45,212,255,0.5)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(earthX, cy, 11, 0, 7); ctx.stroke();
  ctx.fillStyle = "#9fc0ff"; ctx.font = "bold 11px 'JetBrains Mono'"; ctx.fillText("EARTH", earthX, cy + 50);

  // GOES + SDO ring around Earth
  const ring = 26;
  ctx.strokeStyle = "rgba(45,212,255,0.18)"; ctx.beginPath(); ctx.arc(earthX, cy, ring, 0, 7); ctx.stroke();
  const orb = state.earth_satellites.filter((s) => s.position);
  orb.forEach((s, i) => {
    const a = (i / Math.max(1, orb.length)) * Math.PI * 2 - Math.PI / 2;
    const x = earthX + Math.cos(a) * ring, y = cy + Math.sin(a) * ring;
    ctx.fillStyle = SAT_COLORS[s.kind] || "#8a96b8";
    ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
  });
  ctx.fillStyle = "#5a6488"; ctx.font = "9px 'JetBrains Mono'"; ctx.fillText("Earth-orbit fleet", earthX, cy + 66);

  // L1 marker + constellation
  ctx.strokeStyle = "rgba(255,138,61,0.7)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(l1X - 5, cy - 5); ctx.lineTo(l1X + 5, cy + 5); ctx.moveTo(l1X + 5, cy - 5); ctx.lineTo(l1X - 5, cy + 5); ctx.stroke();
  ctx.fillStyle = "#ff8a3d"; ctx.font = "bold 10px 'JetBrains Mono'"; ctx.textAlign = "center";
  ctx.fillText("L1", l1X, cy + 18);
  ctx.fillStyle = "#5a6488"; ctx.font = "9px 'JetBrains Mono'"; ctx.fillText("1.5M km", l1X, cy + 30);

  const sats = state.l1_satellites || [];
  ctx.textAlign = "left"; ctx.font = "10px 'JetBrains Mono'";
  sats.forEach((s, i) => {
    const y = cy - 64 + i * 15;
    ctx.fillStyle = SAT_COLORS[s.kind] || "#8a96b8";
    ctx.beginPath(); ctx.arc(l1X + 16, y - 3, 3.2, 0, 7); ctx.fill();
    ctx.fillStyle = s.kind === "target" ? "#ff8a3d" : "#cdd6f0";
    ctx.fillText(s.name, l1X + 24, y);
  });

  // honest "not to scale" note
  ctx.fillStyle = "#5a6488"; ctx.font = "9px 'JetBrains Mono'"; ctx.textAlign = "right";
  ctx.fillText("schematic · not to scale (L1 ≈ 1% of the Sun–Earth distance)", w - 8, h - 8);
  ctx.textAlign = "left";
}

function buildSatLegend(state) {
  const el = document.getElementById("sat-legend");
  if (!el) return;
  const items = [];
  for (const s of state.earth_satellites) {
    const p = s.position;
    const pos = p ? `${Math.abs(p.lat).toFixed(1)}°${p.lat >= 0 ? "N" : "S"}, ${Math.abs(p.lon).toFixed(1)}°${p.lon >= 0 ? "E" : "W"}` : "no fix";
    items.push(`<span class="legend-item"><span class="legend-dot" style="background:${SAT_COLORS[s.kind]}"></span><b>${s.name}</b> <span class="lg-role">${pos} · ${s.role}</span></span>`);
  }
  for (const s of state.l1_satellites) {
    items.push(`<span class="legend-item"><span class="legend-dot" style="background:${SAT_COLORS[s.kind]}"></span><b>${s.name}</b> <span class="lg-role">L1 · ${s.role}</span></span>`);
  }
  el.innerHTML = items.join("");
}

function updateTrackReadout(state, ss, now) {
  setText("ro-utc", now.toISOString().substr(11, 8) + "Z");
  setText("ro-subsolar", `${Math.abs(ss.lat).toFixed(1)}°${ss.lat >= 0 ? "N" : "S"}, ${Math.abs(ss.lon).toFixed(1)}°${ss.lon >= 0 ? "E" : "W"}`);
  setText("ro-tracked", state.earth_satellites.filter((s) => s.tracked).length + " Earth-orbit + " + (state.l1_satellites || []).length + " at L1");
  const speed = lastSpaceWeather && lastSpaceWeather.solar_wind && lastSpaceWeather.solar_wind.speed;
  setText("ro-transit", speed ? `≈ ${Math.round((state.geometry.l1_distance_km / speed) / 60)} min @ ${Math.round(speed)} km/s` : "awaiting solar wind");
}

async function refreshSatellites() {
  const d = await fetchJSON("/api/satellites");
  if (d && !d.__status) { satState = d; buildSatLegend(d); }
}

function renderTracking() {
  if (!satState) return;
  const now = new Date();
  const ss = subsolarPoint(now);
  // Live positions recomputed THIS instant (client-side SGP4) — fully dynamic.
  const live = computeEarthSats(now);
  // Accumulate motion trails for the fast LEO movers (visible real-time motion).
  for (const s of live) {
    if (s.kind === "leo" && s.position) {
      (satTrails[s.name] = satTrails[s.name] || []).push({ lon: s.position.lon, lat: s.position.lat });
      if (satTrails[s.name].length > 130) satTrails[s.name].shift();
    }
  }
  const merged = Object.assign({}, satState, { earth_satellites: live });
  // Independent try/catch so one map failing never blanks the other.
  try { drawWorldMap(merged, ss); } catch (e) { console.error("world map:", e); }
  try { drawL1(merged, lastSpaceWeather); } catch (e) { console.error("L1 map:", e); }
  updateTrackReadout(merged, ss, now);
  return merged;
}

function startTracking() {
  trackActive = true;
  loadLand().then(() => trackActive && renderTracking());
  // Pull L1 metadata/geometry once, and the TLEs for per-second propagation.
  Promise.all([refreshSatellites(), loadTLEs()]).then(() => {
    const m = renderTracking();
    if (m) buildSatLegend(m);
  });
  renderTracking();
  if (trackTimer) clearInterval(trackTimer);
  let tick = 0;
  trackTimer = setInterval(() => {
    if (!trackActive) return;
    const m = renderTracking();             // recompute + redraw EVERY second
    if (m && ++tick % 5 === 0) buildSatLegend(m);  // refresh legend positions
    // If client-side SGP4 is unavailable (CDN blocked), fall back to server positions.
    if (!satrecs && tick % 30 === 0) refreshSatellites();
    if (tick % 21600 === 0) loadTLEs();     // refresh TLEs every ~6h
  }, 1000);
}
function stopTracking() { trackActive = false; if (trackTimer) { clearInterval(trackTimer); trackTimer = null; } }

/* ───────────── Tabs ───────────── */
function wireTabs() {
  const views = {
    live: document.getElementById("view-live"),
    performance: document.getElementById("view-performance"),
    tracking: document.getElementById("view-tracking"),
  };
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const v = tab.dataset.view;
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      Object.entries(views).forEach(([k, el]) => el && el.classList.toggle("hidden", k !== v));
      if (v !== "tracking") stopTracking();
      if (v === "performance") {
        if (!perfData) loadPerformance(); else { renderLeadHist(); renderBaselineBar(); renderCalibration(); }
      } else if (v === "tracking") {
        startTracking();
      } else {
        chart.resize();
      }
    });
  });
}

/* ───────────── Boot ───────────── */
async function boot() {
  initChart();
  wireTabs();
  wireReplayControls();
  wirePerfControls();
  populateReplayDropdown();

  const [history, current, forecast] = await Promise.all([
    fetchJSON("/api/history"), fetchJSON("/api/current"), fetchJSON("/api/forecast"),
  ]);

  if (Array.isArray(history)) rebuildChartFromHistory(history);
  if (current && !current.__status) {
    updateClassBadge(current.class);
    setStatus("live", current.updated || current.time, current.stale);
  }
  renderForecast(forecast && forecast.__status === 503 ? { model_loaded: false, error: forecast.error } : forecast);

  await loadCatalog();
  await refreshSourceChips();
  refreshDonkiChip();
  refreshSpaceWeather();

  connectWS();

  setInterval(loadCatalog, 60000);
  setInterval(refreshSourceChips, 60000);
  setInterval(refreshDonkiChip, 600000);
  setInterval(refreshSpaceWeather, 120000);
  setInterval(async () => {
    if (replayActive) return;
    const fc = await fetchJSON("/api/forecast");
    renderForecast(fc && fc.__status === 503 ? { model_loaded: false, error: fc.error } : fc);
  }, 60000);
}

boot().catch((e) => { console.error("Boot failed:", e); setStatus("down"); });

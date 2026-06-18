# Solar Flare Nowcast & Forecast — Aditya-L1 / GOES

A live space-weather **nowcasting + forecasting** system built for ISRO Hackathon 2026.
Python 3.11+, FastAPI, Uvicorn, HTTPX, LightGBM. It ingests real-time GOES X-ray flux,
forecasts ≥M-class flares at 10/30/60-minute horizons with a calibrated LightGBM model,
auto-catalogs flares from the live feed, and can **replay real historical X-flares**
through the exact live pipeline for demos.

> **Zero mock data, always.** Every number on the dashboard traces to a live NOAA/NASA
> feed, a real cached GOES archive, or a held-out training run. Anything not yet available
> (e.g. Aditya-L1 via PRADAN) renders an honest *pending* / *stale* / *not-loaded* state —
> never a fabricated value.

---

## Quick start

```bash
# 1. Install (reproducible, exact versions)
pip install -r requirements.lock.txt      # or: pip install -r requirements.txt

# 2. (optional) NASA key for DONKI — DEMO_KEY works out of the box
copy .env.example .env

# 3. Run the server (serves API + dashboard on http://127.0.0.1:8000)
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** — the dashboard is served by the same process.

The trained models (`backend/model/flare_model_{10,30,60}min.pkl`) and the cached GOES
archive (`backend/data/goes_flux_*.parquet`) are already in the repo, so the dashboard,
the forecast, **and the replay feature all work immediately** without retraining.

---

## What makes it strong

- **Honest, multi-horizon forecast** — calibrated (isotonic) LightGBM per horizon, with a
  real *lead-time* evaluation (see below) instead of a single hand-wavy number.
- **Historical Flare Replay** — pick a real past X/M-flare and watch the gauges climb and
  the alert fire on **real recorded GOES data**, played through the same model. Essential
  because the Sun is usually quiet on demo day.
- **Model Performance panel** — TSS / HSS / precision / recall / **false-alarm rate** per
  horizon, lead-time distribution, skill-vs-persistence, a reliability curve and a
  confusion matrix — all loaded from a real `metrics.json`, nothing hardcoded.
- **Aditya-L1 fusion-ready** — drop SoLEXS + HEL1OS FITS into one folder and the hard
  X-ray precursor panel activates; until then it states *Awaiting Aditya-L1 via PRADAN*.
- **Live space-weather context (all keyless NOAA)** — proton flux (S-scale radiation
  storm), solar wind speed + Bz, Kp (G-scale), F10.7, active sunspot regions (flags
  flare-productive delta regions), and **NOAA SWPC's own flare forecast shown right next
  to ours** for an honest side-by-side. No API keys required — all `services.swpc.noaa.gov`.
- **Live satellite tracking (Satellite Tracking tab)** — real CelesTrak TLEs are propagated
  with **SGP4 in the browser every second**, so positions move continuously (not in polling
  steps). GOES-16/18/19 + SDO sit at their true sub-satellite points with GEO coverage
  footprints, while the **ISS and NOAA-20 visibly cross the map with motion trails** (the ISS
  matters here: flares/SEP events drive astronaut radiation risk). A live day/night terminator
  and sub-solar point update every second. The Sun–Earth L1 panel shows Aditya-L1 / DSCOVR /
  ACE / WIND / SOHO at the real L1 geometry, with the solar-wind transit time computed live
  from DSCOVR's measured speed. No mock positions — it correctly shows GOES-19 as operational
  GOES-East after the 2025 hand-over.

---

## Forecasting & lead time

### Target Definition
The target is: *will the GOES long band reach ≥M-class within the next N minutes?*

```
Φ_long(t) ≥ 1×10⁻⁵ W/m²   →   M-class event
    ↑           ↑
  long-band    threshold
  flux rate
```

Trained for **N ∈ {10, 30, 60}** minutes. Models are evaluated on a **chronologically held-out** tail of 2024 (no shuffling, no leakage).

### Lead Time — Per Real Flare Event

**Lead time is measured per real flare event**, not per alert sample. For every M+ flare in
the held-out set, we record:

```
t_lead = min[t : P(M+|t) > threshold] − t_peak    (within 120-min lookback)
```

We report the full distribution. **The headline result:**

```
Horizon    │  Median lead time  │  p25   │  p75  │  Events
───────────┼────────────────────┼────────┼───────┼────────
10-minute  │      8.0 min       │  2 min │ 49 min │  202/202
30-minute  │      9.5 min       │  3 min │ 70 min │  202/202
60-minute  │     15.0 min       │  5 min │ 93 min │  202/202
```

**Key interpretation:**
- **Median** = trustworthy "typical fresh warning" — half of real flares are caught 8–15 min before peak.
- **p75 tail** (49–93 min) reflects active-region clustering: probability legitimately stays elevated between flares.
- **100% catch rate** = zero missed events in held-out evaluation.

See the *Model Performance* tab for skill metrics (TSS, HSS, precision, recall, FAR) and reliability curves.

### Alert Threshold & Decision Boundary

```
Forecast Probability P(M+ within N min)
    ↑
 1.0 │                          ALERT FIRES ✓
     │                       (threshold = 0.5)
 0.5 │───────────────●─────────┤
     │        ↑       ↑          ↑
     │    Rising      │     LEAD TIME captured
     │    signal    Peak        here
     │               │          ↑
 0.0 │_______________█__________
     └─────────────────────────→ Time (minutes before flare peak)
              -120         0    +N
                   ↑       ↑    ↑
              Look-back  Peak  Horizon
```

**Metrics per held-out event:**
- **TSS** (True Skill Statistic) = 2×(hit rate − false alarm rate) − 1
- **HSS** (Heidke Skill Score) = (Po − Pe) / (1 − Pe)
- **FAR** (False Alarm Ratio) = false alarms / (hits + false alarms)
- **Lead = first threshold crossing − peak** (within 120-min window)

### Retrain & Update Metrics

```bash
python -m backend.train                 # trains on cached GOES years, writes metrics.json
python -m backend.train --smoke-test    # quick 1-year (2023) run
```

`train.py` prints a **before/after** table comparing the legacy 16-feature set against the
enhanced precursor feature set (acceleration / 2nd derivative of log-flux, short-band rise
rate, short/long spectral-hardness ratio + its rate, flare-clustering counts), and writes
all real metrics to `backend/model/metrics.json` (the single source of truth for the panel).

---

## Replay mode (real recorded data)

1. The backend scans the cached GOES parquet for real M5+/X-class events automatically.
2. In the top bar, choose an event from **REPLAY REAL FLARE** (e.g. `X8.9 — 2024-10-03`).
3. Press **PLAY** — the dashboard clears and streams that real flare minute-by-minute
   through the same `compute_features → model → alert` pipeline at accelerated speed.
4. A vertical **▲ ALERT FIRED** marker (where the model crossed threshold) and a
   **◆ FLARE PEAK** marker are overlaid on the light curve, visually proving the lead time.
5. **EXIT REPLAY** returns to the live feed.

Replays are labelled as replays end-to-end and can never be mistaken for live data.

---

## Aditya-L1 fusion — how to drop in FITS

The architecture is fusion-ready today. To activate it with real ISSDC PRADAN data:

1. Place SoLEXS and HEL1OS FITS files into **`backend/data/aditya_fits/`** (filenames
   containing `solexs` / `hel1os`, `.fits` or `.fits.gz`).
2. The Aditya source chip flips to **LIVE** and the *Hard X-ray Precursor Signal* panel
   begins showing the real hard/soft ratio from `/api/aditya/series` — no other changes
   needed.
3. For full *predictive* fusion benefit, retrain (`python -m backend.train`) once an
   overlapping GOES + Aditya window is archived, so the `hard_soft_ratio` feature carries
   real signal. Until then the GOES-only short/long ratio (`sl_ratio`) serves as the
   always-on precursor proxy. This is stated honestly in the UI rather than faked.

---

## API reference

| Endpoint | Description |
|---|---|
| `GET /api/current` | Latest flux, flare class, nowcast, forecast; carries `stale` + `data_age_seconds`. |
| `GET /api/history` | Last ~6h of long/short band flux for charting. |
| `GET /api/forecast` | Multi-horizon M+ probabilities, alert level, lead estimate. **503** (honest) if no model. |
| `GET /api/flares` | Latest GOES flare summary from NOAA SWPC. |
| `GET /api/donki?days=N` | NASA DONKI flare catalog proxy. **503** (honest) if NASA unreachable. |
| `GET /api/catalog` | Flares auto-detected from the live feed (SQLite). |
| `GET /api/spaceweather` | Live keyless NOAA context: proton flux (S-scale), solar wind, Kp (G-scale), F10.7, active regions, and **NOAA's own 3-day flare forecast**. **503** until first fetch. |
| `GET /api/satellites` | Live spacecraft positions: GOES + SDO + ISS + NOAA-20 from real CelesTrak TLEs (SGP4), plus the Sun–Earth L1 constellation. **503** until first fetch. |
| `GET /api/satellites/tle` | Raw live TLEs for browser-side per-second SGP4 propagation (smooth motion). |
| `GET /api/metrics` | Real held-out evaluation metrics (Model Performance panel). **404** if not trained. |
| `GET /api/replay/events` | Catalog of real M5+/X-class flares available to replay. |
| `GET /api/replay/{id}` | Full precomputed replay track (frames + lead-time markers). |
| `GET /api/aditya/status` | Aditya-L1 FITS load status (honest pending until PRADAN). |
| `GET /api/aditya/series` | Merged SoLEXS+HEL1OS series (**404** until FITS present). |
| `GET /health` | Liveness/readiness: `model_loaded`, `last_data_age_seconds`, `sources_live`, `stale`. |
| `WS  /ws` | Live state pushed every 60s. |
| `WS  /ws?mode=replay&event=<id>&speed=<x>` | Streams a real historical flare through the live pipeline. |

---

## Reliability for demo day

- The server **never crashes** if NOAA/NASA briefly fails: it keeps serving the last
  known-good state and flags it **STALE** (with the data age) until the feed recovers.
- `/health` answers even when degraded, so you can confirm status at a glance.
- The poller retries every 60s; a transient SSL/network error at startup will not stop boot.
- TLS uses the **OS trust store** (`truststore`) so HTTPS to NOAA/NASA works on networks
  with TLS interception — without ever disabling certificate verification.
  BUILT BY @vsrupeshkumar

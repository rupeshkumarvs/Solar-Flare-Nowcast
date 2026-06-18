import asyncio
import json
import math
import os
import logging

# Use the operating-system trust store for TLS. On networks that perform TLS
# interception (the corporate root CA lives in the Windows store but not in
# certifi's bundle) this is what lets us reach NOAA/NASA over HTTPS cleanly,
# without disabling certificate verification.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore is optional
    logging.getLogger(__name__).warning(
        "truststore unavailable; falling back to default certifi CA bundle."
    )

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import httpx
import pandas as pd
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Import our custom database, features, and Aditya reader helper logic
from backend.database import init_db, log_flare_event, get_latest_flares
from backend.features import compute_features
from backend.nowcast import detect_and_catalog_flares, classify_flare_class
from backend.aditya_reader import get_fits_files_status, read_solexs, read_hel1os, merge_aditya, FITS_DIR
from backend.replay import list_replay_events, build_replay_track
from backend import spaceweather
from backend import satellites

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SWPC_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
SWPC_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"

# Load .env so a personal NASA_API_KEY is actually picked up (without this the
# app silently falls back to the heavily-throttled DEMO_KEY).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    logger.warning("python-dotenv unavailable; NASA_API_KEY must be set in the environment.")

NASA_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
if NASA_KEY and NASA_KEY != "DEMO_KEY":
    logger.info(f"NASA DONKI using personal API key (…{NASA_KEY[-4:]}).")
else:
    logger.warning("NASA DONKI using DEMO_KEY (rate-limited); set NASA_API_KEY in .env for reliability.")

POLL_SECONDS = 60

app = FastAPI(
    title="Solar Flare Nowcast & Forecast API",
    description="Live space weather dashboard and forecast engine for GOES X-ray and Aditya-L1 data.",
    version="2.0.0"
)

# CORS open to all origins (required for separate frontend dashboard)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {
    "current": None,
    "history": deque(maxlen=720),
    "flares": [],
    "last_ok": None,        # datetime of last successful SWPC fetch
    "goes_live": False,     # is the live GOES feed currently reachable?
    "spaceweather": None,   # extra NOAA context (proton/solar wind/Kp/F10.7/regions/forecast)
    "satellites": None,     # live TLE-tracked positions + L1 constellation
}
SPACEWEATHER_POLL_SECONDS = 180  # these feeds update slowly; no need to hammer them
SATELLITES_POLL_SECONDS = 60     # TLEs are cached 6h; this just recomputes sub-points
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
STALE_AFTER_SECONDS = 300   # live data older than this is flagged "stale"
clients: set[WebSocket] = set()
models_cache = {}


def _data_age_seconds():
    """Seconds since the last successful live data poll, or None if never."""
    if not STATE["last_ok"]:
        return None
    return (datetime.now(timezone.utc) - STATE["last_ok"]).total_seconds()

def load_forecast_models():
    """Cache-loads the trained LightGBM horizon classifiers from the models directory."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    for horizon in [10, 30, 60]:
        model_path = os.path.join(model_dir, f"flare_model_{horizon}min.pkl")
        if os.path.exists(model_path) and horizon not in models_cache:
            try:
                models_cache[horizon] = joblib.load(model_path)
                logger.info(f"Loaded trained LGBM model for horizon {horizon}min successfully.")
            except Exception as e:
                logger.warning(f"Could not load horizon {horizon}min model: {e}")

def get_real_time_forecast() -> dict:
    """
    Evaluates the trained ML models on the live GOES history to return forecast
    probabilities. NEVER fabricates a probability: if the model is not loaded the
    result carries model_loaded=False so callers can return HTTP 503 honestly.
    If the model IS loaded but there is not yet enough live buffer, probs are None
    and status reflects 'warming up' — again, no fake numbers.
    """
    load_forecast_models()

    # Determine trend from the live long-band slope (shared with the nowcast).
    nc = nowcast_baseline(list(STATE["history"]))
    trend = nc.get("trend", "unknown")

    # Case 1: no trained model on disk -> caller must surface a 503.
    if not models_cache:
        return {
            "model_loaded": False,
            "prob_10min": None,
            "prob_30min": None,
            "prob_60min": None,
            "trend": trend,
            "lead_time_estimate": None,
            "alert_level": "unknown",
            "status": "model not trained yet",
        }

    history = list(STATE["history"])

    # Case 2: model loaded but live buffer too short to compute real features.
    if len(history) < 60:
        return {
            "model_loaded": True,
            "prob_10min": None,
            "prob_30min": None,
            "prob_60min": None,
            "trend": trend,
            "lead_time_estimate": None,
            "alert_level": "unknown",
            "status": f"warming up: need 60 min of live data, have {len(history)}",
        }

    try:
        df = pd.DataFrame(history)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()
        df = df.rename(columns={'long': 'long_band', 'short': 'short_band'})

        feat_df = compute_features(df)
        latest_features = feat_df.iloc[-1:]

        probs = {}
        for horizon in [10, 30, 60]:
            payload = models_cache[horizon]
            model = payload["model"]
            feature_names = payload["features"]
            X = latest_features[feature_names]
            prob = model.predict_proba(X)[0, 1]
            probs[horizon] = round(float(prob), 3)

        p10, p30, p60 = probs[10], probs[30], probs[60]
        max_prob = max(p10, p30, p60)

        if max_prob >= 0.6:
            alert_level = "red"
        elif max_prob >= 0.3:
            alert_level = "yellow"
        else:
            alert_level = "green"

        if p10 >= 0.5:
            lead_time_est = 10.0
        elif p30 >= 0.5:
            lead_time_est = 30.0
        elif p60 >= 0.5:
            lead_time_est = 60.0
        else:
            lead_time_est = 0.0

        return {
            "model_loaded": True,
            "prob_10min": p10,
            "prob_30min": p30,
            "prob_60min": p60,
            "trend": trend,
            "lead_time_estimate": lead_time_est,
            "alert_level": alert_level,
            "status": "Forecast computed successfully",
        }
    except Exception as e:
        logger.error(f"Error executing real-time ML forecast: {e}")
        # A runtime failure is NOT a clear probability — report it honestly.
        return {
            "model_loaded": True,
            "prob_10min": None,
            "prob_30min": None,
            "prob_60min": None,
            "trend": trend,
            "lead_time_estimate": None,
            "alert_level": "unknown",
            "status": f"forecast error: {e}",
        }

def nowcast_baseline(history):
    """
    REAL baseline nowcast on LIVE flux (combines current level with the
    short-term log-rise rate of the long band).
    """
    pts = [h for h in history if h["long"]][-30:]  # last ~30 min of real data
    if len(pts) < 5:
        return {"prob_M_6h": None, "prob_X_24h": None, "trend": "unknown"}
    cur, prev = pts[-1]["long"], pts[0]["long"]
    
    # Clip to avoid division by zero
    cur_val = max(cur, 1e-10)
    prev_val = max(prev, 1e-10)
    
    slope = (math.log10(cur_val) - math.log10(prev_val)) / len(pts)   # decades/min
    level_score = (math.log10(cur_val) + 6) / 2                    # ~0 at C, ~1 at M
    rise_score = max(0.0, slope * 40)
    
    try:
        p_m = 1 / (1 + math.exp(-(2.2 * level_score + rise_score - 2.0)))
    except OverflowError:
        p_m = 0.0
    p_x = min(p_m * 0.35, 1.0)
    trend = "rising" if slope > 0.002 else "falling" if slope < -0.002 else "flat"
    return {"prob_M_6h": round(p_m, 3), "prob_X_24h": round(p_x, 3), "trend": trend}

async def fetch_swpc():
    """Ingests NOAA SWPC GOES 6-hour X-ray flux, merges bands, and logs cataloged events."""
    async with httpx.AsyncClient(timeout=20) as c:
        rows = (await c.get(SWPC_XRAY_URL)).json()
        by_time = {}
        for r in rows:
            t = r["time_tag"]
            by_time.setdefault(t, {"time": t, "long": None, "short": None})
            if "0.1-0.8" in r["energy"]:        # robust to spacing in the field
                by_time[t]["long"] = r["flux"]
            else:
                by_time[t]["short"] = r["flux"]
                
        series = sorted(by_time.values(), key=lambda r: r["time"])
        STATE["history"] = deque(series, maxlen=720)
        latest = series[-1]
        
        # Pre-process current state
        STATE["current"] = {
            **latest,
            "class": classify_flare_class(latest["long"]),
            "nowcast": nowcast_baseline(list(STATE["history"])),
            "forecast": get_real_time_forecast(),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        
        # Real-time event cataloging running on the live feed
        try:
            # Map history items to keys matching detect_and_catalog_flares
            nowcast_history = []
            for item in list(STATE["history"]):
                nowcast_history.append({
                    "time_tag": item["time"],
                    "long_band": item["long"],
                    "short_band": item["short"]
                })
            detect_and_catalog_flares(nowcast_history)
        except Exception as e:
            logger.error(f"Error cataloging flare events: {e}")
            
        try:
            fl = (await c.get(SWPC_FLARES_URL)).json()
            STATE["flares"] = fl if isinstance(fl, list) else [fl]
        except Exception:
            pass

        # Mark the live feed healthy only after a successful primary ingest.
        STATE["last_ok"] = datetime.now(timezone.utc)
        STATE["goes_live"] = True

async def poller():
    """Background polling loop executing every 60 seconds."""
    while True:
        try:
            await fetch_swpc()
            dead = set()
            for ws in clients:
                try:
                    await ws.send_json(STATE["current"])
                except Exception:
                    dead.add(ws)
            clients.difference_update(dead)
        except Exception as e:
            # NOAA briefly unreachable: keep serving last-known-good, flag not-live.
            STATE["goes_live"] = False
            logger.warning(f"poll error (serving last-known-good): {e}")
        await asyncio.sleep(POLL_SECONDS)


async def spaceweather_poller():
    """Refreshes the extra NOAA context feeds on a slow cadence, last-known-good on failure."""
    while True:
        try:
            async with httpx.AsyncClient(timeout=25, follow_redirects=True) as c:
                sw = await spaceweather.fetch_all(c)
            STATE["spaceweather"] = sw
        except Exception as e:
            logger.warning(f"spaceweather poll error (serving last-known-good): {e}")
        await asyncio.sleep(SPACEWEATHER_POLL_SECONDS)


def _refresh_satellites():
    """Synchronous (sgp4 is CPU-bound) recompute of live satellite positions."""
    with httpx.Client(timeout=20, follow_redirects=True) as c:
        STATE["satellites"] = satellites.get_state(c)


async def satellites_poller():
    """Recomputes live satellite sub-points each minute (TLEs themselves cached 6h)."""
    while True:
        try:
            await asyncio.to_thread(_refresh_satellites)
        except Exception as e:
            logger.warning(f"satellites poll error (serving last-known-good): {e}")
        await asyncio.sleep(SATELLITES_POLL_SECONDS)

@app.on_event("startup")
async def startup():
    # Initialize SQLite flare catalog database
    init_db()
    # Cache ML models
    load_forecast_models()
    # Perform initial fetch — never let a transient network/SSL error stop the
    # server from booting; the poller will retry every 60s.
    try:
        await fetch_swpc()
    except Exception as e:
        logger.warning(f"Initial SWPC fetch failed (poller will retry): {e}")
    # Initial space-weather context fetch (non-fatal) + its slow poller.
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as c:
            STATE["spaceweather"] = await spaceweather.fetch_all(c)
    except Exception as e:
        logger.warning(f"Initial space-weather fetch failed (poller will retry): {e}")
    # Initial satellite fetch (non-fatal) — TLEs from CelesTrak.
    try:
        await asyncio.to_thread(_refresh_satellites)
    except Exception as e:
        logger.warning(f"Initial satellite fetch failed (poller will retry): {e}")
    # Spawn background poller tasks
    asyncio.create_task(poller())
    asyncio.create_task(spaceweather_poller())
    asyncio.create_task(satellites_poller())

# REST API Endpoints
@app.get("/api/current")
def current():
    """Returns the latest current solar state, including long/short flux, flare class, nowcasts, and forecasts.

    Carries an honest `stale` flag + `data_age_seconds` so the dashboard can show
    last-known-good data truthfully if the live NOAA feed briefly drops.
    """
    if STATE["current"]:
        STATE["current"]["forecast"] = get_real_time_forecast()
        age = _data_age_seconds()
        STATE["current"]["data_age_seconds"] = round(age, 1) if age is not None else None
        STATE["current"]["stale"] = bool(age is not None and age > STALE_AFTER_SECONDS)
    return STATE["current"]


@app.get("/health")
def health():
    """Liveness/readiness probe for demo day — never throws, always answers."""
    age = _data_age_seconds()
    return {
        "status": "ok" if (models_cache and STATE["goes_live"]) else "degraded",
        "model_loaded": bool(models_cache),
        "horizons_loaded": sorted(models_cache.keys()),
        "last_data_age_seconds": round(age, 1) if age is not None else None,
        "sources_live": {"goes_noaa": STATE["goes_live"]},
        "history_points": len(STATE["history"]),
        "stale": bool(age is not None and age > STALE_AFTER_SECONDS),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/metrics")
def metrics():
    """Real held-out evaluation metrics from the last training run (Model Performance panel).

    Every number here is written by backend/train.py from a chronologically
    held-out test set — nothing is hardcoded. Returns 404 (honestly) if training
    has not been run yet.
    """
    if not os.path.exists(METRICS_PATH):
        return JSONResponse(status_code=404, content={"error": "metrics not available — run `python -m backend.train`"})
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"could not read metrics: {e}"})


@app.get("/api/spaceweather")
def get_spaceweather():
    """Live NOAA context: proton flux (S-scale), solar wind, Kp (G-scale), F10.7,
    active regions, and NOAA SWPC's own 3-day flare forecast. All keyless & real.

    Returns 503 (honestly) until the first successful fetch completes."""
    if STATE["spaceweather"] is None:
        return JSONResponse(status_code=503, content={"error": "space-weather context not fetched yet"})
    return STATE["spaceweather"]


@app.get("/api/satellites")
def get_satellites():
    """Live spacecraft positions: GOES + SDO tracked from real CelesTrak TLEs via SGP4,
    plus the Sun–Earth L1 constellation (Aditya-L1/DSCOVR/ACE/WIND/SOHO) at real L1
    geometry. 503 (honestly) until the first fetch completes."""
    if STATE["satellites"] is None:
        return JSONResponse(status_code=503, content={"error": "satellite positions not fetched yet"})
    return STATE["satellites"]


@app.get("/api/satellites/tle")
def get_satellite_tles():
    """Raw live TLEs for the Earth-orbit satellites, so the frontend can propagate
    them every second (SGP4) for smooth, continuous motion. Same real CelesTrak data."""
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as c:
            return satellites.get_tles(c)
    except Exception as e:
        logger.warning(f"TLE endpoint error: {e}")
        return JSONResponse(status_code=503, content={"error": f"TLEs unavailable: {e}"})


@app.get("/api/replay/events")
def replay_events():
    """Catalog of real major (M5+/X-class) flares available for replay from cached GOES data."""
    try:
        return list_replay_events()
    except Exception as e:
        logger.error(f"replay event scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"replay scan error: {e}")


@app.get("/api/replay/{event_id}")
def replay_event(event_id: str):
    """Full precomputed replay track for one real historical flare (REST; the WS streams it live)."""
    load_forecast_models()
    track = build_replay_track(event_id, models_cache)
    if track is None:
        raise HTTPException(status_code=404, detail=f"replay event '{event_id}' not found in cached data")
    return track

@app.get("/api/history")
def history():
    """Returns the last 6h of long+short band flux (for charting)."""
    return list(STATE["history"])

@app.get("/api/flares")
def flares():
    """Returns the latest GOES flare event from NOAA SWPC."""
    return STATE["flares"]

@app.get("/api/donki")
async def donki(days: int = 30):
    """Proxies the NASA DONKI flare catalog for N days.

    Returns HTTP 503 with an honest error if NASA is unreachable/slow rather
    than a 500 — so the dashboard source chip can show DOWN truthfully.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"https://api.nasa.gov/DONKI/FLR?startDate={start}&endDate={end}&api_key={NASA_KEY}"
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"DONKI proxy failed: {type(e).__name__}: {e}")
        return JSONResponse(status_code=503, content={"error": "NASA DONKI unavailable", "detail": str(e)})

@app.get("/api/catalog")
async def get_catalog():
    """Returns the last 30 detected flares cataloged in SQLite database."""
    try:
        return get_latest_flares(limit=30)
    except Exception as e:
        logger.error(f"Error accessing catalog database: {e}")
        raise HTTPException(status_code=500, detail="Database error retrieving flare catalog")

@app.get("/api/forecast")
async def get_forecast():
    """Returns the multi-horizon forecasting predictions, alert level, and estimated lead times.

    If no trained model is available, returns HTTP 503 with an honest error —
    never a fabricated probability number.
    """
    res = get_real_time_forecast()
    if not res.get("model_loaded"):
        return JSONResponse(status_code=503, content={"error": "model not trained yet"})
    return res

@app.get("/api/aditya/status")
async def get_aditya_status():
    """Returns status of local Aditya-L1 FITS files."""
    status = get_fits_files_status()
    solexs_points = 0
    hel1os_points = 0
    
    if status["fits_loaded"]:
        try:
            files = os.listdir(FITS_DIR)
            solexs_f = [f for f in files if "solexs" in f.lower()][0]
            hel1os_f = [f for f in files if "hel1os" in f.lower()][0]
            
            solexs_df = read_solexs(os.path.join(FITS_DIR, solexs_f))
            hel1os_df = read_hel1os(os.path.join(FITS_DIR, hel1os_f))
            
            solexs_points = len(solexs_df)
            hel1os_points = len(hel1os_df)
        except Exception as e:
            logger.warning(f"Error reading counts from Aditya FITS: {e}")
            
    message = (
        f"Loaded {status['solexs_files']} SoLEXS + {status['hel1os_files']} HEL1OS FITS file(s)."
        if status["fits_loaded"]
        else "Awaiting PRADAN approval"
    )

    return {
        "fits_loaded": status["fits_loaded"],
        "message": message,
        "solexs_points": solexs_points,
        "hel1os_points": hel1os_points,
        "solexs_files_count": status["solexs_files"],
        "hel1os_files_count": status["hel1os_files"]
    }

@app.get("/api/aditya/series")
async def get_aditya_series():
    """Returns merged SoLEXS + HEL1OS time series data if FITS files are loaded, else 404."""
    status = get_fits_files_status()
    if not status["fits_loaded"]:
        raise HTTPException(
            status_code=404, 
            detail="Aditya-L1 FITS files are not loaded. Place SoLEXS and HEL1OS FITS files in backend/data/aditya_fits/ to enable the fusion layer."
        )
        
    try:
        files = os.listdir(FITS_DIR)
        solexs_f = [f for f in files if "solexs" in f.lower()][0]
        hel1os_f = [f for f in files if "hel1os" in f.lower()][0]
        
        solexs_df = read_solexs(os.path.join(FITS_DIR, solexs_f))
        hel1os_df = read_hel1os(os.path.join(FITS_DIR, hel1os_f))
        
        merged_df = merge_aditya(solexs_df, hel1os_df)
        
        return merged_df.head(1000).to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error processing Aditya FITS series: {e}")
        raise HTTPException(status_code=500, detail=f"Aditya FITS Parsing Error: {str(e)}")

async def stream_replay(ws: WebSocket, event_id: str, speed: float):
    """
    Streams a real historical flare minute-by-minute through the SAME pipeline at
    accelerated speed. Sends a `replay_meta` message first (with the actual peak
    time and the time the model fired its alert — the lead-time markers), then one
    `replay_frame` per simulated minute, then a `replay_done`.
    """
    load_forecast_models()
    track = build_replay_track(event_id, models_cache)
    if track is None:
        await ws.send_json({"type": "replay_error", "error": f"event '{event_id}' not found"})
        return

    await ws.send_json({
        "type": "replay_meta",
        "event": track["event"],
        "label": track["label"],
        "peak_time": track["peak_time"],
        "alert_fired_time": track["alert_fired_time"],
        "lead_time_min": track["lead_time_min"],
        "total": len(track["frames"]),
    })

    # `speed` is the ×real-time multiplier; clamp the per-frame delay so a replay is
    # always watchable (not instant) but never tediously slow.
    delay = max(0.05, min(1.0, 60.0 / max(speed, 1.0)))
    for i, frame in enumerate(track["frames"]):
        payload = {
            "type": "replay_frame",
            "is_replay": True,
            "replay_label": track["label"],
            "frame": i,
            "total": len(track["frames"]),
            "updated": frame["time"],
            **frame,
        }
        try:
            await ws.send_json(payload)
        except Exception:
            return
        await asyncio.sleep(delay)

    await ws.send_json({
        "type": "replay_done",
        "peak_time": track["peak_time"],
        "alert_fired_time": track["alert_fired_time"],
        "lead_time_min": track["lead_time_min"],
    })


# WebSocket Connection
@app.websocket("/ws")
async def ws(ws: WebSocket, mode: str = Query("live"), event: str = Query(None), speed: float = Query(60.0)):
    """Live feed by default. With ?mode=replay&event=<id> it plays a real historical
    flare through the same pipeline instead, then closes."""
    await ws.accept()

    if mode == "replay" and event:
        try:
            await stream_replay(ws, event, speed)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"replay stream error: {e}")
        finally:
            try:
                await ws.close()
            except Exception:
                pass
        return

    clients.add(ws)
    if STATE["current"]:
        await ws.send_json(STATE["current"])
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)

# Serve the static dashboard. Mounted LAST so it never shadows the /api/* routes
# or the /ws WebSocket. Visiting http://127.0.0.1:8000/ serves frontend/index.html.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

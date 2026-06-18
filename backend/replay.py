"""
Historical flare REPLAY — real recorded GOES data, never synthetic.

Scans the cached NOAA GOES parquet (the very same files the model was trained on)
for real M5+/X-class flares, and replays a chosen event minute-by-minute through
the EXACT live pipeline (compute_features -> the same horizon models -> the same
alert logic). This lets a judge watch the gauges climb and the alert fire on a
real past X-flare even when the Sun is quiet on demo day.

Everything here is real recorded data. The only thing "simulated" is the clock:
we play real minutes back faster than real time. Replays are labelled as replays
end-to-end so they can never be mistaken for the live feed.
"""

import os
import glob
import numpy as np
import pandas as pd

from backend.features import compute_features
from backend.nowcast import classify_flare_class

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
M_CLASS = 1e-5
X_CLASS = 1e-4
MAJOR_THRESHOLD = 5e-5      # M5 and above qualifies as a "major" replayable event
ALERT_THRESHOLD = 0.5       # operating point — matches training / lead-time eval

PRE_MIN = 120               # minutes of lead-in shown before the peak
POST_MIN = 60               # minutes shown after the peak
WARMUP_MIN = 180            # extra lead-in (not shown) so rolling features are warm

_flux_cache = {}            # year -> DataFrame
_events_cache = None        # list[dict]


def _load_all_flux() -> pd.DataFrame:
    """Concatenate every cached GOES parquet into one sorted, de-duplicated frame."""
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "goes_flux_*.parquet"))):
        year = os.path.basename(path).replace("goes_flux_", "").replace(".parquet", "")
        if year not in _flux_cache:
            _flux_cache[year] = pd.read_parquet(path)
        frames.append(_flux_cache[year])
    if not frames:
        return pd.DataFrame(columns=["long_band", "short_band"])
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _scan_events(df: pd.DataFrame, threshold: float):
    """Find distinct flare events (contiguous runs >= threshold) and their peaks."""
    vals = df["long_band"].values
    above = vals >= threshold
    events = []
    i, n = 0, len(vals)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            seg = vals[i:j]
            peak_pos = i + int(np.argmax(seg))
            peak_time = df.index[peak_pos]
            peak_flux = float(vals[peak_pos])
            events.append({
                "id": pd.Timestamp(peak_time).strftime("%Y%m%dT%H%M"),
                "class": classify_flare_class(peak_flux),
                "peak_time": pd.Timestamp(peak_time).isoformat(),
                "peak_flux": peak_flux,
                "start_time": pd.Timestamp(df.index[i]).isoformat(),
                "end_time": pd.Timestamp(df.index[j - 1]).isoformat(),
            })
            i = j
        else:
            i += 1
    return events


def list_replay_events(force: bool = False):
    """Return the catalog of replayable major real flares, newest first."""
    global _events_cache
    if _events_cache is not None and not force:
        return _events_cache
    df = _load_all_flux()
    if df.empty:
        _events_cache = []
        return _events_cache
    events = _scan_events(df, MAJOR_THRESHOLD)
    events.sort(key=lambda e: e["peak_flux"], reverse=True)  # most intense first
    # Cap to keep the dropdown meaningful; the strongest events are the demo-worthy ones.
    _events_cache = events[:40]
    # Re-sort the kept set chronologically (newest first) for display.
    _events_cache.sort(key=lambda e: e["peak_time"], reverse=True)
    return _events_cache


def _find_event(event_id: str):
    for e in list_replay_events():
        if e["id"] == event_id:
            return e
    return None


def build_replay_track(event_id: str, models: dict):
    """
    Precompute the full replay track for one event: real flux per minute plus the
    causally-correct model forecast at each minute (features at minute t depend
    only on data up to t, so indexing per-row after a one-shot compute is honest).

    Returns a dict with the per-minute frames, the actual peak time, and the time
    the model FIRST fired its alert — the two markers that visually prove lead time.
    """
    event = _find_event(event_id)
    if event is None:
        return None

    df = _load_all_flux()
    peak_time = pd.Timestamp(event["peak_time"])
    win_start = peak_time - pd.Timedelta(minutes=PRE_MIN)
    win_end = peak_time + pd.Timedelta(minutes=POST_MIN)
    calc_start = peak_time - pd.Timedelta(minutes=PRE_MIN + WARMUP_MIN)

    slab = df.loc[calc_start:win_end].copy()
    if slab.empty:
        return None

    # One-shot feature + forecast computation over the whole slab.
    feats = compute_features(slab)
    horizons = [10, 30, 60]
    prob_cols = {}
    if models:
        for h in horizons:
            payload = models.get(h)
            if not payload:
                continue
            model = payload["model"]
            names = payload["features"]
            try:
                prob_cols[h] = pd.Series(model.predict_proba(feats[names])[:, 1], index=feats.index)
            except Exception:
                prob_cols[h] = pd.Series(np.nan, index=feats.index)

    # Restrict to the visible window.
    show = slab.loc[win_start:win_end]
    frames = []
    alert_fired_time = None

    for ts, row in show.iterrows():
        long_v = float(row["long_band"]) if pd.notna(row["long_band"]) else None
        short_v = float(row["short_band"]) if pd.notna(row["short_band"]) else None

        p10 = _safe_prob(prob_cols.get(10), ts)
        p30 = _safe_prob(prob_cols.get(30), ts)
        p60 = _safe_prob(prob_cols.get(60), ts)
        probs = [p for p in (p10, p30, p60) if p is not None]
        max_prob = max(probs) if probs else None

        if max_prob is None:
            alert_level = "unknown"
        elif max_prob >= 0.6:
            alert_level = "red"
        elif max_prob >= 0.3:
            alert_level = "yellow"
        else:
            alert_level = "green"

        if alert_fired_time is None and max_prob is not None and max_prob >= ALERT_THRESHOLD and ts <= peak_time:
            alert_fired_time = ts

        frames.append({
            "time": pd.Timestamp(ts).isoformat(),
            "long": long_v,
            "short": short_v,
            "class": classify_flare_class(long_v) if long_v else "Below A",
            "forecast": {
                "model_loaded": bool(prob_cols),
                "prob_10min": p10, "prob_30min": p30, "prob_60min": p60,
                "trend": _trend(prob_cols.get(60), ts),
                "alert_level": alert_level,
                "status": "Replay forecast",
            },
        })

    lead_min = None
    if alert_fired_time is not None:
        lead_min = round((peak_time - alert_fired_time).total_seconds() / 60.0, 1)

    return {
        "event": event,
        "frames": frames,
        "peak_time": peak_time.isoformat(),
        "alert_fired_time": alert_fired_time.isoformat() if alert_fired_time is not None else None,
        "lead_time_min": lead_min,
        "label": f"REPLAY: {event['class']} — {peak_time.strftime('%Y-%m-%d %H:%M')}Z",
    }


def _safe_prob(series, ts):
    if series is None or ts not in series.index:
        return None
    v = series.loc[ts]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return round(float(v), 3) if pd.notna(v) else None


def _trend(series, ts):
    """Rising/flat/falling from the 60-min probability slope around ts."""
    if series is None or ts not in series.index:
        return "unknown"
    pos = series.index.get_loc(ts)
    if isinstance(pos, slice):
        pos = pos.start
    if pos < 5:
        return "unknown"
    cur = series.iloc[pos]
    prev = series.iloc[pos - 5]
    if pd.isna(cur) or pd.isna(prev):
        return "unknown"
    if cur - prev > 0.05:
        return "rising"
    if cur - prev < -0.05:
        return "falling"
    return "flat"

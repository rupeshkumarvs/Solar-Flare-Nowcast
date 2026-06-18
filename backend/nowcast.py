import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from backend.database import log_flare_event, get_latest_flares

def classify_flare_class(flux: float) -> str:
    """
    Classify flare class based on GOES long-band (0.1-0.8nm) flux in W/m^2.
    A >= 1e-8, B >= 1e-7, C >= 1e-6, M >= 1e-5, X >= 1e-4.
    Format like "M2.3", "C5.1".
    """
    if flux <= 0:
        return "Below A"
    
    if flux >= 1e-4:
        letter = "X"
        value = flux / 1e-4
    elif flux >= 1e-5:
        letter = "M"
        value = flux / 1e-5
    elif flux >= 1e-6:
        letter = "C"
        value = flux / 1e-6
    elif flux >= 1e-7:
        letter = "B"
        value = flux / 1e-7
    elif flux >= 1e-8:
        letter = "A"
        value = flux / 1e-8
    else:
        letter = "A"
        value = flux / 1e-8

    return f"{letter}{value:.1f}"

def parse_iso_time(time_str: str) -> datetime:
    time_str = time_str.replace(" ", "T")
    if not time_str.endswith("Z"):
        time_str += "Z"
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def detect_and_catalog_flares(history_list: List[Dict[str, Any]]):
    """
    Scans the history list for flare events according to NOAA criteria:
    - START: 4 consecutive 1-minute rises in long-band flux.
    - PEAK: maximum flux value reached during the active flare state.
    - END: flux drops below 1.5x pre-flare background.
    
    Logs newly completed events to the SQLite database.
    """
    if len(history_list) < 5:
        return
        
    times = [pt.get("time_tag") for pt in history_list]
    fluxes = [pt.get("long_band") for pt in history_list]
    
    active = False
    start_idx = None
    peak_idx = None
    peak_val = 0.0
    pre_flare_bg = 0.0
    
    # Retrieve already cataloged events to prevent duplicates
    try:
        logged_events = {e['start_time'] for e in get_latest_flares(100)}
    except Exception:
        logged_events = set()
    
    for i in range(4, len(fluxes)):
        # Skip elements that are None
        if any(f is None for f in fluxes[i-4:i+1]):
            continue
            
        if not active:
            # Check for 4 consecutive rises
            if fluxes[i] > fluxes[i-1] > fluxes[i-2] > fluxes[i-3] > fluxes[i-4]:
                active = True
                start_idx = i - 4
                pre_flare_bg = fluxes[start_idx]
                peak_idx = i
                peak_val = fluxes[i]
        else:
            # Flare is active, track peak
            if fluxes[i] > peak_val:
                peak_val = fluxes[i]
                peak_idx = i
                
            # Check for flare end: drops below 1.5x background level
            if fluxes[i] <= 1.5 * pre_flare_bg:
                start_time = times[start_idx]
                peak_time = times[peak_idx]
                end_time = times[i]
                
                if start_time not in logged_events:
                    flare_class = classify_flare_class(peak_val)
                    try:
                        log_flare_event(
                            start_time=start_time,
                            peak_time=peak_time,
                            end_time=end_time,
                            peak_flux=peak_val,
                            flare_class=flare_class,
                            source="GOES"
                        )
                        logged_events.add(start_time)
                    except Exception as e:
                        print(f"Error cataloging flare event: {e}")
                
                active = False

def nowcast(history_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    # SWAP for trained model.
    """
    Operate on the live flux history to output:
      - prob_M_6h: probability of M-class or higher flare in the next 6 hours (0.0 to 1.0)
      - prob_X_24h: probability of X-class or higher flare in the next 24 hours (0.0 to 1.0)
      - trend: "rising", "flat", or "falling"
    """
    if not history_list:
        return {
            "prob_M_6h": 0.0,
            "prob_X_24h": 0.0,
            "trend": "flat",
            "rate_of_change_log10_per_hr": 0.0
        }

    points: List[Tuple[datetime, float]] = []
    for pt in history_list:
        time_str = pt.get("time_tag")
        flux_val = pt.get("long_band")
        if time_str and flux_val is not None and flux_val > 0:
            dt = parse_iso_time(time_str)
            points.append((dt, flux_val))

    points.sort(key=lambda x: x[0])

    if not points:
        return {
            "prob_M_6h": 0.0,
            "prob_X_24h": 0.0,
            "trend": "flat",
            "rate_of_change_log10_per_hr": 0.0
        }

    t_curr, flux_curr = points[-1]
    log10_curr = math.log10(flux_curr)

    target_dt = 30 * 60
    t_past = None
    flux_past = None
    
    best_diff = float("inf")
    for t, f in reversed(points[:-1]):
        elapsed = (t_curr - t).total_seconds()
        diff = abs(elapsed - target_dt)
        if diff < best_diff and elapsed > 0:
            best_diff = diff
            t_past, flux_past = t, f
        if elapsed > 7200:
            break

    if t_past is None and len(points) > 1:
        t_past, flux_past = points[0]

    if t_past is not None and flux_past is not None and flux_past > 0:
        dt_hours = (t_curr - t_past).total_seconds() / 3600.0
        if dt_hours > 0.01:
            log10_past = math.log10(flux_past)
            rate_per_hr = (log10_curr - log10_past) / dt_hours
        else:
            rate_per_hr = 0.0
    else:
        rate_per_hr = 0.0

    if rate_per_hr > 0.15:
        trend = "rising"
    elif rate_per_hr < -0.15:
        trend = "falling"
    else:
        trend = "flat"

    expected_log10_6h = log10_curr + rate_per_hr * 6.0
    dist_to_M = -5.0 - expected_log10_6h
    base_M_prob = 1.0 / (1.0 + math.exp(-3.0 * (log10_curr - (-5.5))))
    trend_M_prob = 1.0 / (1.0 + math.exp(1.5 * dist_to_M))
    prob_M_6h = max(base_M_prob, trend_M_prob)

    expected_log10_24h = log10_curr + rate_per_hr * 24.0
    dist_to_X = -4.0 - expected_log10_24h
    base_X_prob = 1.0 / (1.0 + math.exp(-3.0 * (log10_curr - (-4.5))))
    trend_X_prob = 1.0 / (1.0 + math.exp(2.0 * dist_to_X))
    prob_X_24h = max(base_X_prob, trend_X_prob * 0.7)

    if flux_curr >= 1e-5:
        prob_M_6h = 0.99
    else:
        prob_M_6h = max(0.01, min(0.95, prob_M_6h))

    if flux_curr >= 1e-4:
        prob_X_24h = 0.99
    else:
        prob_X_24h = max(0.01, min(0.90, prob_X_24h))

    return {
        "prob_M_6h": round(prob_M_6h, 3),
        "prob_X_24h": round(prob_X_24h, 3),
        "trend": trend,
        "rate_of_change_log10_per_hr": round(rate_per_hr, 4)
    }

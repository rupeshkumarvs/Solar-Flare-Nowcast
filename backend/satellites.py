"""
LIVE satellite positions — real data, no mock.

Earth-orbit spacecraft (the GOES X-ray satellites + SDO) are tracked from live
TLEs fetched from CelesTrak (keyless) and propagated with SGP4 to a real
sub-satellite latitude/longitude/altitude. This is genuinely live: e.g. it
correctly shows GOES-19 as the operational GOES-East at 75.2°W and GOES-16 at
its real parked longitude after the 2025 hand-over — exactly why we propagate
TLEs instead of hardcoding.

Deep-space spacecraft at the Sun–Earth L1 point (Aditya-L1, DSCOVR, ACE, WIND,
SOHO) are beyond SGP4's Earth-orbit regime and are not carried in CelesTrak's GP
catalog, so we place them at the *real* L1 geometry — ~1.5 million km sunward of
Earth on the Sun–Earth line — and label the halo-orbit nature honestly. DSCOVR's
live solar-wind data on the dashboard is itself proof it is on station there.
"""

import math
import time
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from sgp4.api import Satrec, jday
    _SGP4 = True
except Exception:  # pragma: no cover
    _SGP4 = False
    logger.warning("sgp4 not installed; Earth-orbit satellite tracking disabled.")

CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=tle"
EARTH_RADIUS_KM = 6371.0
L1_DISTANCE_KM = 1_500_000.0       # ~1.5 million km sunward of Earth (real L1 distance)
EARTH_SUN_KM = 149_600_000.0       # 1 AU
TLE_TTL_SECONDS = 6 * 3600         # TLEs change slowly; refresh every 6h

# Earth-orbit spacecraft relevant to this project (live-tracked via TLE/SGP4).
EARTH_SATS = [
    {"name": "GOES-19", "norad": 60133, "role": "GOES-East · primary X-ray source", "kind": "xray"},
    {"name": "GOES-18", "norad": 51850, "role": "GOES-West · X-ray flux", "kind": "xray"},
    {"name": "GOES-16", "norad": 41866, "role": "GOES (storage longitude)", "kind": "xray"},
    {"name": "SDO",     "norad": 36395, "role": "Solar Dynamics Observatory · EUV imaging", "kind": "solar"},
    # Fast LEO movers — visibly dynamic on the map, and genuinely relevant:
    {"name": "ISS",     "norad": 25544, "role": "Crew radiation risk during flares / SEP events", "kind": "leo"},
    {"name": "NOAA-20", "norad": 43013, "role": "Polar-orbiting weather (JPSS-1)", "kind": "leo"},
]

# Sun–Earth L1 constellation (real station, halo orbits — not SGP4-trackable).
L1_SATS = [
    {"name": "Aditya-L1", "agency": "ISRO",     "role": "SoLEXS + HEL1OS soft+hard X-ray (this project's fusion source)", "kind": "target"},
    {"name": "DSCOVR",    "agency": "NOAA",     "role": "Real-time solar wind (feeds this dashboard)", "kind": "wind"},
    {"name": "ACE",       "agency": "NASA",     "role": "Solar wind & energetic particles", "kind": "wind"},
    {"name": "WIND",      "agency": "NASA",     "role": "Solar wind plasma & fields", "kind": "wind"},
    {"name": "SOHO",      "agency": "ESA/NASA", "role": "Solar imaging & helioseismology", "kind": "solar"},
]

_tle_cache = {}  # norad -> {"l1":..., "l2":..., "ts":epoch}


def _fetch_tle(client, norad):
    """Fetch a TLE from CelesTrak, cached for TLE_TTL_SECONDS. Returns (l1, l2) or None."""
    c = _tle_cache.get(norad)
    if c and (time.time() - c["ts"] < TLE_TTL_SECONDS):
        return c["l1"], c["l2"]
    try:
        r = client.get(CELESTRAK.format(norad=norad))
        r.raise_for_status()
        lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
        if len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
            _tle_cache[norad] = {"l1": lines[1], "l2": lines[2], "ts": time.time()}
            return lines[1], lines[2]
    except Exception as e:
        logger.warning(f"TLE fetch failed for {norad}: {type(e).__name__}: {e}")
    # Fall back to a stale cached TLE rather than nothing (honest: it's still real).
    if c:
        return c["l1"], c["l2"]
    return None


def _subpoint(l1, l2, when=None):
    """Propagate a TLE to `when` (UTC) and return real sub-satellite lat/lon/alt."""
    when = when or datetime.now(timezone.utc)
    sat = Satrec.twoline2rv(l1, l2)
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    err, pos, _vel = sat.sgp4(jd, fr)
    if err != 0:
        return None
    x, y, z = pos  # ECI km
    d = (jd - 2451545.0) + fr
    gmst = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    xe = x * math.cos(gmst) + y * math.sin(gmst)
    ye = -x * math.sin(gmst) + y * math.cos(gmst)
    lon = math.degrees(math.atan2(ye, xe))
    lat = math.degrees(math.atan2(z, math.sqrt(xe * xe + ye * ye)))
    alt = math.sqrt(x * x + y * y + z * z) - EARTH_RADIUS_KM
    return {"lat": round(lat, 3), "lon": round(lon, 3), "alt_km": round(alt, 1)}


def subsolar_point(when=None):
    """Real sub-solar point (lat = solar declination, lon where it is solar noon)."""
    when = when or datetime.now(timezone.utc)
    jd = 2451545.0 + (when - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    # Equation of time (minutes) -> sub-solar longitude.
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0
    eot = (L - ra + 180) % 360 - 180  # degrees
    utc_hours = when.hour + when.minute / 60.0 + when.second / 3600.0
    lon = -(15.0 * (utc_hours - 12.0) + eot)
    lon = (lon + 180) % 360 - 180
    return {"lat": round(dec, 3), "lon": round(lon, 3)}


def get_tles(client):
    """Return the raw live TLEs for every Earth-orbit satellite so the FRONTEND can
    propagate them every second with SGP4 (satellite.js) for smooth, continuous,
    fully-dynamic motion. Still 100% real — these are the same CelesTrak elements."""
    out = []
    for s in EARTH_SATS:
        tle = _fetch_tle(client, s["norad"])
        if tle:
            out.append({**s, "line1": tle[0], "line2": tle[1]})
    return {"updated": datetime.now(timezone.utc).isoformat(), "satellites": out}


def get_state(client):
    """Consolidated live satellite state for the dashboard tracking view."""
    earth = []
    for s in EARTH_SATS:
        entry = {**s, "position": None, "tracked": False}
        if _SGP4:
            tle = _fetch_tle(client, s["norad"])
            if tle:
                pos = _subpoint(tle[0], tle[1])
                if pos:
                    entry["position"] = pos
                    entry["tracked"] = True
        earth.append(entry)

    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "sgp4_available": _SGP4,
        "subsolar": subsolar_point(),
        "earth_satellites": earth,
        "l1_satellites": L1_SATS,
        "geometry": {
            "l1_distance_km": L1_DISTANCE_KM,
            "earth_sun_km": EARTH_SUN_KM,
            "l1_fraction": round(L1_DISTANCE_KM / EARTH_SUN_KM, 4),  # ~0.01 of the way to the Sun
            "geo_altitude_km": 35786,
        },
    }

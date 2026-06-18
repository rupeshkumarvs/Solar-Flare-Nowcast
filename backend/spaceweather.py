"""
Additional LIVE space-weather context from NOAA SWPC — all keyless, all real.

Each source is fetched independently with its own error handling, so one feed
being down never blanks the others (the dashboard shows a per-source honest
state). Nothing here is mocked: every value is parsed from the live NOAA JSON/text
the moment it is fetched.

Sources:
  - GOES integral proton flux (>=10 MeV)  -> NOAA S-scale radiation storm
  - DSCOVR solar wind plasma (speed, density) + mag (Bz, Bt)
  - Planetary Kp index                    -> NOAA G-scale geomagnetic storm
  - F10.7 cm solar radio flux             -> solar activity level
  - Solar region summary                  -> active regions, flare-productive deltas
  - SWPC 3-day text forecast              -> NOAA's OWN flare forecast (Radio Blackout
                                             R1-R2 = M-class, R3+ = X-class probabilities)
"""

import re
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE = "https://services.swpc.noaa.gov"
PROTON_URL = f"{BASE}/json/goes/primary/integral-protons-6-hour.json"
PLASMA_URL = f"{BASE}/products/solar-wind/plasma-5-minute.json"
MAG_URL = f"{BASE}/products/solar-wind/mag-5-minute.json"
KP_URL = f"{BASE}/products/noaa-planetary-k-index.json"
F107_URL = f"{BASE}/json/f107_cm_flux.json"
REGIONS_URL = f"{BASE}/json/solar_regions.json"   # note: underscore, not hyphen
FORECAST_URL = f"{BASE}/text/3-day-forecast.txt"


def s_scale(flux_pfu):
    """NOAA solar radiation storm scale from the >=10 MeV integral proton flux (pfu)."""
    if flux_pfu is None:
        return None
    for thr, lvl in [(1e5, "S5"), (1e4, "S4"), (1e3, "S3"), (1e2, "S2"), (1e1, "S1")]:
        if flux_pfu >= thr:
            return lvl
    return "S0"


def g_scale(kp):
    """NOAA geomagnetic storm scale from Kp."""
    if kp is None:
        return None
    if kp >= 9: return "G5"
    if kp >= 8: return "G4"
    if kp >= 7: return "G3"
    if kp >= 6: return "G2"
    if kp >= 5: return "G1"
    return "G0"


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_proton(rows):
    pts = [r for r in rows if r.get("energy") == ">=10 MeV" and r.get("flux") is not None]
    if not pts:
        return None
    latest = max(pts, key=lambda r: r["time_tag"])
    flux = _to_float(latest["flux"])
    return {"flux_10mev": flux, "storm": s_scale(flux), "time": latest["time_tag"]}


def parse_csvjson(rows):
    """SWPC 'products' feeds are [header, row, row, ...]; return (header, last_row)."""
    if not isinstance(rows, list) or len(rows) < 2:
        return None, None
    return rows[0], rows[-1]


def parse_solar_wind(plasma_rows, mag_rows):
    out = {"speed": None, "density": None, "bz": None, "bt": None, "time": None}
    hdr, last = parse_csvjson(plasma_rows)
    if hdr and last:
        idx = {name: i for i, name in enumerate(hdr)}
        out["density"] = _to_float(last[idx.get("density", 1)])
        out["speed"] = _to_float(last[idx.get("speed", 2)])
        out["time"] = last[idx.get("time_tag", 0)]
    hdr2, last2 = parse_csvjson(mag_rows)
    if hdr2 and last2:
        idx2 = {name: i for i, name in enumerate(hdr2)}
        out["bz"] = _to_float(last2[idx2.get("bz_gsm", 3)])
        out["bt"] = _to_float(last2[idx2.get("bt", 6)])
    return out if (out["speed"] or out["bz"] is not None) else None


def parse_kp(rows):
    pts = [r for r in rows if r.get("Kp") is not None]
    if not pts:
        return None
    latest = max(pts, key=lambda r: r["time_tag"])
    kp = _to_float(latest["Kp"])
    return {"value": kp, "storm": g_scale(kp), "time": latest["time_tag"]}


def parse_f107(rows):
    pts = [r for r in rows if r.get("flux") is not None]
    if not pts:
        return None
    latest = max(pts, key=lambda r: r["time_tag"])   # feed is NOT time-sorted
    return {"flux": _to_float(latest["flux"]), "time": latest["time_tag"]}


def parse_regions(rows):
    if not isinstance(rows, list) or not rows:
        return {"count": 0, "date": None, "regions": [], "flare_productive": []}
    latest_date = max(r.get("observed_date", "") for r in rows)
    today = [r for r in rows if r.get("observed_date") == latest_date and r.get("region")]
    regions = []
    for r in today:
        mag = (r.get("mag_class") or "").upper()
        regions.append({
            "region": r.get("region"),
            "location": r.get("location"),
            "spot_class": r.get("spot_class"),
            "mag_class": r.get("mag_class"),
            "area": r.get("area"),
            "number_spots": r.get("number_spots"),
            # A delta ('D') in the Mount Wilson class is the flare-productive signature.
            "delta": "D" in mag,
        })
    regions.sort(key=lambda x: (x["delta"], x["area"] or 0), reverse=True)
    deltas = [r for r in regions if r["delta"]]
    return {"count": len(regions), "date": latest_date, "regions": regions[:8], "flare_productive": deltas}


def parse_forecast(text):
    """Parse SWPC's 3-day text forecast. The Radio Blackout R-scale IS NOAA's own
    flare forecast: R1-R2 ≈ M-class, R3+ ≈ X-class probability over each UTC day."""
    def pcts(label):
        m = re.search(rf"^{re.escape(label)}\s+(.*)$", text, re.MULTILINE)
        if not m:
            return []
        return [int(x) for x in re.findall(r"(\d+)\s*%", m.group(1))]

    issued = None
    mi = re.search(r":Issued:\s*(.+)", text)
    if mi:
        issued = mi.group(1).strip()
    # Day labels from the Radio Blackout header (e.g. "Jun 18  Jun 19  Jun 20").
    days = []
    md = re.search(r"Radio Blackout Forecast for .*?\n\s*((?:[A-Z][a-z]{2}\s+\d{1,2}\s*)+)", text)
    if md:
        days = re.findall(r"[A-Z][a-z]{2}\s+\d{1,2}", md.group(1))
    kp_max = None
    mk = re.search(r"greatest expected 3 hr Kp for .*? is\s+([\d.]+)", text)
    if mk:
        kp_max = _to_float(mk.group(1))

    return {
        "issued": issued,
        "days": days,
        "m_class": pcts("R1-R2"),          # M-class flare (radio blackout) probability %
        "x_class": pcts("R3 or greater"),  # X-class flare probability %
        "radiation_s1": pcts("S1 or greater"),
        "kp_max_expected": kp_max,
    }


async def fetch_all(client):
    """Fetch every context feed concurrently-ish (sequential awaits, each isolated).
    Returns a consolidated dict with per-source honest None/error on failure."""
    async def get_json(url):
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

    async def get_text(url):
        r = await client.get(url)
        r.raise_for_status()
        return r.text

    out = {"updated": datetime.now(timezone.utc).isoformat()}

    async def safe(key, coro, parser):
        try:
            out[key] = parser(await coro)
        except Exception as e:
            logger.warning(f"spaceweather '{key}' failed: {type(e).__name__}: {e}")
            out[key] = None

    await safe("proton", get_json(PROTON_URL), parse_proton)
    # solar wind needs two feeds; handle as a small special case
    try:
        plasma = await get_json(PLASMA_URL)
        mag = await get_json(MAG_URL)
        out["solar_wind"] = parse_solar_wind(plasma, mag)
    except Exception as e:
        logger.warning(f"spaceweather 'solar_wind' failed: {type(e).__name__}: {e}")
        out["solar_wind"] = None
    await safe("kp", get_json(KP_URL), parse_kp)
    await safe("f107", get_json(F107_URL), parse_f107)
    await safe("regions", get_json(REGIONS_URL), parse_regions)
    await safe("swpc_forecast", get_text(FORECAST_URL), parse_forecast)
    return out

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Feature sets.
#
# LEGACY_FEATURES is the original 16-feature set we shipped first. We keep it so
# train.py can run an honest BEFORE/AFTER comparison (does the new precursor
# engineering actually buy lead time, or are we fooling ourselves?).
#
# ENHANCED_FEATURES adds precursor-oriented features aimed squarely at the
# weak spot: lead time. Flares show *curvature* (acceleration) in log-flux and a
# hardening of the spectrum (short-band rising faster than long-band) BEFORE the
# long band crosses M. Those are the GOES-only proxies for the soft/hard
# precursor effect until real Aditya-L1 SoLEXS+HEL1OS data arrives via PRADAN.
# ─────────────────────────────────────────────────────────────────────────────

LEGACY_FEATURES = [
    'long_band', 'short_band', 'log10_long', 'log10_short',
    'rise_rate_1min', 'rise_rate_5min', 'rise_rate_10min', 'rise_rate_30min',
    'max_10min', 'std_10min',
    'max_30min', 'std_30min',
    'max_60min', 'std_60min',
    'mins_since_last_C', 'hard_soft_ratio',
]

ENHANCED_FEATURES = [
    # raw + log levels
    'long_band', 'short_band', 'log10_long', 'log10_short',
    # multi-window rise rates of the long band (1st derivative of log flux)
    'rise_rate_1min', 'rise_rate_5min', 'rise_rate_10min', 'rise_rate_15min', 'rise_rate_30min',
    # acceleration — 2nd derivative of log flux (curvature precedes the spike)
    'accel_1min', 'accel_5min',
    # short-band rise rate (impulsive phase shows up in the soft channel first)
    'rise_rate_short_5min',
    # spectral hardness proxy: short/long ratio and how fast it is changing
    'sl_ratio', 'sl_ratio_rate_5min',
    # trailing statistics
    'max_10min', 'std_10min',
    'max_30min', 'std_30min',
    'max_60min', 'std_60min',
    # recency / clustering — active regions fire flares in bursts
    'mins_since_last_C', 'mins_since_last_M',
    'flares_6h', 'flares_24h',
    # Aditya-L1 fusion slot (0.0 until real FITS are present; see aditya_reader)
    'hard_soft_ratio',
]


def compute_features(df_flux: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full ENHANCED feature superset from a DataFrame of GOES X-ray
    flux. The DataFrame must have a DatetimeIndex and contain 'long_band' and
    'short_band'. Optionally it can contain 'hard_soft_ratio' (Aditya fusion);
    when absent it defaults cleanly to 0.0 so a GOES-only model is unaffected.

    Returns every column in ENHANCED_FEATURES. Callers that want the original
    16-feature behaviour simply select LEGACY_FEATURES from the result — that is
    how train.py runs its before/after comparison, and how each saved model
    selects exactly the columns it was trained on at inference time.
    """
    df = df_flux.copy()
    df = df.sort_index()

    # Fill small gaps
    df['long_band'] = df['long_band'].ffill()
    df['short_band'] = df['short_band'].ffill()

    # Log10 values (clip to avoid log of zero / negatives from bad samples)
    df['log10_long'] = np.log10(df['long_band'].clip(lower=1e-10))
    df['log10_short'] = np.log10(df['short_band'].clip(lower=1e-10))

    # 1. Long-band rise rates (log10 difference over 1, 5, 10, 15, 30 min)
    for w in (1, 5, 10, 15, 30):
        df[f'rise_rate_{w}min'] = (df['log10_long'] - df['log10_long'].shift(w)).fillna(0.0)

    # 2. Acceleration — the 2nd derivative of log flux. A flare's log-flux curve
    #    bends upward (positive curvature) before it actually crosses M, so this
    #    is the single most direct "early warning" feature we can build.
    df['accel_1min'] = (df['rise_rate_1min'] - df['rise_rate_1min'].shift(1)).fillna(0.0)
    df['accel_5min'] = (df['rise_rate_5min'] - df['rise_rate_5min'].shift(5)).fillna(0.0)

    # 3. Short-band rise rate — the soft/impulsive channel often leads the long band.
    df['rise_rate_short_5min'] = (df['log10_short'] - df['log10_short'].shift(5)).fillna(0.0)

    # 4. Spectral hardness proxy. log10(short) - log10(long) = log10(short/long).
    #    A rising ratio means the spectrum is hardening — the GOES-only stand-in
    #    for the hard X-ray precursor until Aditya HEL1OS data is fused in.
    df['sl_ratio'] = df['log10_short'] - df['log10_long']
    df['sl_ratio_rate_5min'] = (df['sl_ratio'] - df['sl_ratio'].shift(5)).fillna(0.0)

    # 5. Trailing max and std over 10/30/60-min windows (cadence is 1-minute).
    for w in (10, 30, 60):
        df[f'max_{w}min'] = df['long_band'].rolling(f'{w}min', min_periods=1).max()
        df[f'std_{w}min'] = df['long_band'].rolling(f'{w}min', min_periods=1).std().fillna(0.0)

    # 6. Minutes since the last C+ (>=1e-6) and M+ (>=1e-5) sample.
    df['mins_since_last_C'] = _mins_since_threshold(df, 1e-6)
    df['mins_since_last_M'] = _mins_since_threshold(df, 1e-5)

    # 7. Flare clustering: number of distinct M+ flare ONSETS in the trailing
    #    6h / 24h. An "onset" is a minute that crosses up through M from below,
    #    so a single long flare counts once, not once per minute.
    is_m = df['long_band'] >= 1e-5
    onset = (is_m & ~is_m.shift(1, fill_value=False)).astype(float)
    df['flares_6h'] = onset.rolling('360min', min_periods=1).sum()
    df['flares_24h'] = onset.rolling('1440min', min_periods=1).sum()

    # 8. Aditya-L1 hard/soft ratio fusion slot.
    if 'hard_soft_ratio' not in df.columns:
        df['hard_soft_ratio'] = 0.0
    else:
        df['hard_soft_ratio'] = df['hard_soft_ratio'].fillna(0.0)

    return df[ENHANCED_FEATURES]


def _mins_since_threshold(df: pd.DataFrame, threshold: float, default_min: float = 4320.0) -> np.ndarray:
    """Minutes since the long band was last >= threshold. Defaults to 3 days.

    Fully vectorised over numpy datetime64 (handles the non-uniform cadence left
    by gap-dropping correctly, since it differences real timestamps).
    """
    n = len(df)
    times = df.index.values  # datetime64[ns]
    event_pos = np.where((df['long_band'] >= threshold).values)[0]
    if len(event_pos) == 0:
        return np.full(n, default_min)
    # For each row, the most recent event position at or before it.
    k = np.searchsorted(event_pos, np.arange(n), side='right') - 1
    out = np.full(n, default_min, dtype=float)
    valid = k >= 0
    last_event_time = times[event_pos[k[valid]]]
    out[valid] = (times[valid] - last_event_time) / np.timedelta64(1, 'm')
    return out

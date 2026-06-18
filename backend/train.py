import os
import glob
import json
import argparse
import logging
from datetime import datetime

# Use the OS trust store for TLS so any GOES archive download succeeds on networks
# that perform TLS interception (same fix main.py uses) — without disabling verify.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover
    logging.getLogger(__name__).warning("truststore unavailable; using certifi bundle.")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, precision_score, recall_score
import joblib
import httpx
from bs4 import BeautifulSoup
import xarray as xr

# Import features function + the two feature sets we compare.
from backend.features import compute_features, LEGACY_FEATURES, ENHANCED_FEATURES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZONS = [10, 30, 60]
M_CLASS = 1e-5          # M-class long-band threshold (W/m^2)
ALERT_THRESHOLD = 0.5   # operating point for alerts / lead-time evaluation


def download_and_parse_goes_year_direct(year: int, smoke_test: bool = False) -> pd.DataFrame:
    """
    Downloads GOES-16 1-minute averaged XRS flux files directly from NOAA NCEI archive over HTTP.
    Parses them with xarray, and saves the cleaned and resampled year dataset as parquet.
    """
    parquet_path = os.path.join(DATA_DIR, f"goes_flux_{year}.parquet")
    if os.path.exists(parquet_path):
        print(f"Loading cached parquet data for {year} from {parquet_path}")
        return pd.read_parquet(parquet_path)

    base_url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/{year}/"
    print(f"Crawl NOAA NCEI folder: {base_url}")

    all_dfs = []
    with httpx.Client(timeout=30.0) as client:
        months = [1] if smoke_test else list(range(1, 13))
        for month in months:
            month_str = f"{month:02d}"
            month_url = f"{base_url}{month_str}/"
            print(f"Scanning month: {month_str}...")
            try:
                r = client.get(month_url)
                if r.status_code != 200:
                    print(f"Month {month_str} folder not found or unavailable. Skipping.")
                    continue
                soup = BeautifulSoup(r.text, 'html.parser')
                nc_links = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.nc')]
                if not nc_links:
                    print(f"No NetCDF files found in month {month_str}.")
                    continue
                print(f"Found {len(nc_links)} NetCDF files for month {month_str}. Downloading...")
                month_dir = os.path.join(DATA_DIR, f"temp_{year}_{month_str}")
                os.makedirs(month_dir, exist_ok=True)
                month_dfs = []
                for link in sorted(nc_links):
                    file_url = f"{month_url}{link}"
                    file_path = os.path.join(month_dir, link)
                    if not os.path.exists(file_path) or os.path.getsize(file_path) < 10000:
                        with client.stream("GET", file_url) as response:
                            if response.status_code == 200:
                                with open(file_path, "wb") as f:
                                    for chunk in response.iter_bytes():
                                        f.write(chunk)
                    try:
                        with xr.open_dataset(file_path, engine='netcdf4') as ds:
                            df_file = ds[['xrsb_flux', 'xrsa_flux']].to_dataframe()
                            df_file = df_file.rename(columns={'xrsb_flux': 'long_band', 'xrsa_flux': 'short_band'})
                            month_dfs.append(df_file)
                    except Exception as e:
                        print(f"Error parsing file {link}: {e}")
                import shutil
                shutil.rmtree(month_dir, ignore_errors=True)
                if month_dfs:
                    all_dfs.append(pd.concat(month_dfs))
                    print(f"Month {month_str} processed successfully.")
            except Exception as e:
                print(f"Error crawling month {month_str}: {e}")

    if not all_dfs:
        raise ValueError(f"Failed to retrieve any GOES NetCDF files for year {year}")

    print("Combining monthly datasets...")
    df_all = pd.concat(all_dfs).sort_index()
    print(f"Resampling year {year} to 1-minute cadence...")
    df_resampled = df_all.resample('1min').mean()
    df_resampled['long_band'] = df_resampled['long_band'].ffill(limit=60)
    df_resampled['short_band'] = df_resampled['short_band'].ffill(limit=60)
    df_resampled = df_resampled.dropna(subset=['long_band'])
    print(f"Saving combined parquet to: {parquet_path}")
    df_resampled.to_parquet(parquet_path)
    return df_resampled


def compute_metrics(y_true, y_pred, y_prob):
    """ML validation metrics including True Skill Statistic (TSS) and False Alarm Rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / probability of detection
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tss = tpr - fpr

    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / (tp + fp + tn + fn)
    obs_total = tp + fp + tn + fn
    hss = (tp + tn - expected_correct) / (obs_total - expected_correct) if (obs_total - expected_correct) > 0 else 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    # False Alarm Rate (operational definition): fraction of alerts that were wrong.
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
    brier = brier_score_loss(y_true, y_prob)

    return {
        "TSS": float(tss), "HSS": float(hss),
        "Precision": float(precision), "Recall": float(recall),
        "FAR": float(far), "ROC-AUC": float(roc_auc), "Brier": float(brier),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def find_m_events(flux: pd.Series):
    """
    Identify distinct M+ flare events as contiguous runs where long_band >= M-class.
    Returns a list of dicts with the integer peak position and the peak timestamp.
    """
    vals = flux.values
    is_m = vals >= M_CLASS
    events = []
    i, n = 0, len(vals)
    while i < n:
        if is_m[i]:
            j = i
            while j < n and is_m[j]:
                j += 1
            seg = vals[i:j]
            peak_pos = i + int(np.argmax(seg))
            events.append({
                "start_pos": i, "peak_pos": peak_pos, "end_pos": j - 1,
                "peak_time": flux.index[peak_pos], "peak_flux": float(vals[peak_pos]),
            })
            i = j
        else:
            i += 1
    return events


def evaluate_lead_time(df_flux: pd.DataFrame, prob: pd.Series, threshold=ALERT_THRESHOLD, max_lookback_min=120):
    """
    HONEST per-event lead time. For every real M+ flare event in the test set, find
    the EARLIEST minute within `max_lookback_min` before the peak at which the model's
    probability first crossed `threshold`, and record (peak_time - first_alert_time)
    in minutes. Events never alerted before their peak are counted as missed.

    Returns the full distribution (median / p25 / p75 / max / mean), the number of
    events caught vs missed, and the raw per-event lead times — no single cherry-picked
    number.
    """
    events = find_m_events(df_flux['long_band'])
    leads = []
    caught_at_or_before_peak = 0
    missed = 0

    for ev in events:
        peak_time = ev["peak_time"]
        lb_start = peak_time - pd.Timedelta(minutes=max_lookback_min)
        window = prob.loc[lb_start:peak_time]
        crossed = window[window >= threshold]
        if len(crossed) > 0:
            first_t = crossed.index[0]
            lead = (peak_time - first_t).total_seconds() / 60.0
            leads.append(max(0.0, lead))
            caught_at_or_before_peak += 1
        else:
            missed += 1

    leads_arr = np.array(leads, dtype=float)
    has = len(leads_arr) > 0
    return {
        "n_events": len(events),
        "n_caught": caught_at_or_before_peak,
        "n_missed": missed,
        "hit_rate": float(caught_at_or_before_peak / len(events)) if events else 0.0,
        "median_min": float(np.median(leads_arr)) if has else 0.0,
        "p25_min": float(np.percentile(leads_arr, 25)) if has else 0.0,
        "p75_min": float(np.percentile(leads_arr, 75)) if has else 0.0,
        "max_min": float(np.max(leads_arr)) if has else 0.0,
        "mean_min": float(np.mean(leads_arr)) if has else 0.0,
        "lead_times": [round(x, 1) for x in leads_arr.tolist()],
    }


def build_label(df_long: pd.Series, horizon: int) -> pd.Series:
    """Target: does long_band reach M-class within the NEXT `horizon` minutes?"""
    shifted = df_long.shift(-1)
    future_max = shifted.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    return (future_max >= M_CLASS).astype(int)


def train_one(X_train, y_train, X_test, y_test, df_flux_test, horizon, calib_cv):
    """Train + calibrate a LightGBM model for one horizon, return (model, metrics-bundle)."""
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    base_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, scale_pos_weight=scale_pos_weight, verbosity=-1,
    )
    model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=calib_cv)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= ALERT_THRESHOLD).astype(int)
    prob_series = pd.Series(y_prob, index=X_test.index)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    lead = evaluate_lead_time(df_flux_test, prob_series)

    # Reliability / calibration curve (predicted prob vs observed frequency).
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
        calib = {"mean_predicted": [round(float(x), 4) for x in mean_pred],
                 "fraction_positive": [round(float(x), 4) for x in frac_pos]}
    except Exception:
        calib = {"mean_predicted": [], "fraction_positive": []}

    return model, {"metrics": metrics, "lead_time": lead, "calibration": calib}


def persistence_bundle(df_flux_test, y_test, horizon):
    """Persistence baseline: predict M+ next N min iff the last N min already had M+."""
    past_max = df_flux_test['long_band'].rolling(f"{horizon}min", min_periods=1).max()
    pred = (past_max >= M_CLASS).astype(int)
    prob = pred.astype(float)
    metrics = compute_metrics(y_test, pred, prob)
    lead = evaluate_lead_time(df_flux_test, pd.Series(prob.values, index=df_flux_test.index))
    return {"metrics": metrics, "lead_time": lead}


def main():
    parser = argparse.ArgumentParser(description="Train multi-horizon GOES Solar Flare Forecasting models.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-year smoke test on 2023.")
    parser.add_argument("--calib-cv", type=int, default=5, help="CV folds for isotonic calibration.")
    args = parser.parse_args()

    cached_years = sorted(
        int(os.path.basename(p).replace("goes_flux_", "").replace(".parquet", ""))
        for p in glob.glob(os.path.join(DATA_DIR, "goes_flux_*.parquet"))
    )
    if args.smoke_test:
        years = [2023]
        print("--- RUNNING IN SMOKE TEST MODE (Year 2023 only) ---")
    elif cached_years:
        # Default: train on whatever real GOES years are already cached locally
        # (avoids re-crawling NOAA for years we don't have). Add years to the
        # cache by running the downloader, then they're picked up automatically.
        years = cached_years
        print(f"--- TRAINING on cached GOES years: {years} ---")
    else:
        years = list(range(2014, 2025))
        print(f"--- No cache found; attempting full download (Years {years[0]}-{years[-1]}) ---")

    yearly_dfs = []
    for y in years:
        try:
            yearly_dfs.append(download_and_parse_goes_year_direct(y, smoke_test=args.smoke_test))
        except Exception as e:
            print(f"Error processing year {y}: {e}")
    if not yearly_dfs:
        print("Error: No data available for training.")
        return

    print("Concatenating all processed years...")
    df_all = pd.concat(yearly_dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    print(f"Total shape of raw resampled data: {df_all.shape}")

    print("Computing ENHANCED features (legacy set is selected as a subset)...")
    df_features = compute_features(df_all)

    metrics_out = {
        "generated_at": datetime.now().isoformat(),
        "data": {
            "rows": int(df_all.shape[0]),
            "train_range": None, "test_range": None,
            "smoke_test": args.smoke_test,
            "note": "Evaluated on a chronologically held-out tail of the data — no shuffling, no leakage.",
        },
        "alert_threshold": ALERT_THRESHOLD,
        "m_class_threshold": M_CLASS,
        "horizons": {},
    }

    print("\n" + "#" * 78)
    print("# BEFORE / AFTER  —  legacy 16-feature set  vs  enhanced precursor feature set")
    print("#" * 78)

    for horizon in HORIZONS:
        print(f"\n>>> Horizon N={horizon} min")

        label = build_label(df_all['long_band'], horizon)
        df_dataset = df_features.join(label.rename('label')).dropna()
        if df_dataset.shape[0] < 100:
            print(f"Skipping horizon {horizon}min: dataset too small ({df_dataset.shape[0]}).")
            continue

        split_idx = int(len(df_dataset) * 0.8)
        y = df_dataset['label']
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_index = df_dataset.index[split_idx:]
        df_flux_test = df_all.loc[test_index]

        if metrics_out["data"]["train_range"] is None:
            metrics_out["data"]["train_range"] = [str(df_dataset.index[0]), str(df_dataset.index[split_idx - 1])]
            metrics_out["data"]["test_range"] = [str(test_index[0]), str(test_index[-1])]

        # --- LEGACY (before) ---
        Xl = df_dataset[LEGACY_FEATURES]
        Xl_train, Xl_test = Xl.iloc[:split_idx], Xl.iloc[split_idx:]
        _, legacy = train_one(Xl_train, y_train, Xl_test, y_test, df_flux_test, horizon, args.calib_cv)

        # --- ENHANCED (after) — this is the model we ship ---
        Xe = df_dataset[ENHANCED_FEATURES]
        Xe_train, Xe_test = Xe.iloc[:split_idx], Xe.iloc[split_idx:]
        model, enhanced = train_one(Xe_train, y_train, Xe_test, y_test, df_flux_test, horizon, args.calib_cv)

        # --- persistence baseline ---
        persistence = persistence_bundle(df_flux_test, y_test, horizon)

        # Print honest before/after table.
        lm, em, pm = legacy["metrics"], enhanced["metrics"], persistence["metrics"]
        ll, el, pl = legacy["lead_time"], enhanced["lead_time"], persistence["lead_time"]
        print("-" * 78)
        print(f"{'Metric':<22} | {'Legacy (before)':>16} | {'Enhanced (after)':>16} | {'Persistence':>12}")
        print("-" * 78)
        for m in ["TSS", "HSS", "Precision", "Recall", "FAR", "ROC-AUC"]:
            print(f"{m:<22} | {lm[m]:>16.4f} | {em[m]:>16.4f} | {pm[m]:>12.4f}")
        print(f"{'Lead median (min)':<22} | {ll['median_min']:>16.1f} | {el['median_min']:>16.1f} | {pl['median_min']:>12.1f}")
        print(f"{'Lead p75 (min)':<22} | {ll['p75_min']:>16.1f} | {el['p75_min']:>16.1f} | {pl['p75_min']:>12.1f}")
        print(f"{'Lead max (min)':<22} | {ll['max_min']:>16.1f} | {el['max_min']:>16.1f} | {pl['max_min']:>12.1f}")
        print(f"{'Events caught/total':<22} | {str(ll['n_caught'])+'/'+str(ll['n_events']):>16} | "
              f"{str(el['n_caught'])+'/'+str(el['n_events']):>16} | {str(pl['n_caught'])+'/'+str(pl['n_events']):>12}")
        print("-" * 78)

        # Save the ENHANCED model for this horizon (this is what serves live).
        model_path = os.path.join(MODEL_DIR, f"flare_model_{horizon}min.pkl")
        joblib.dump({
            "model": model,
            "features": ENHANCED_FEATURES,
            "horizon_minutes": horizon,
            "trained_at": datetime.now().isoformat(),
            "smoke_test": args.smoke_test,
        }, model_path)
        print(f"Saved enhanced {horizon}min model -> {model_path}")

        metrics_out["horizons"][str(horizon)] = {
            "enhanced": enhanced,
            "legacy": {"metrics": legacy["metrics"], "lead_time": legacy["lead_time"]},
            "persistence": persistence,
            "class_balance": {"train_pos": int((y_train == 1).sum()), "train_neg": int((y_train == 0).sum()),
                              "test_pos": int((y_test == 1).sum()), "test_neg": int((y_test == 0).sum())},
        }

    # Write the single source of truth for the Model Performance panel.
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nWrote real evaluation metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()

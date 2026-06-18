import os
import pandas as pd
from astropy.io import fits

FITS_DIR = os.path.join(os.path.dirname(__file__), "data", "aditya_fits")

def get_fits_files_status() -> dict:
    """Checks the existence of SoLEXS and HEL1OS FITS files in the data directory."""
    if not os.path.exists(FITS_DIR):
        return {"fits_loaded": False, "solexs_files": 0, "hel1os_files": 0}
    
    files = os.listdir(FITS_DIR)
    solexs_files = [f for f in files if "solexs" in f.lower() and f.endswith((".fits", ".fits.gz"))]
    hel1os_files = [f for f in files if "hel1os" in f.lower() and f.endswith((".fits", ".fits.gz"))]
    
    return {
        "fits_loaded": len(solexs_files) > 0 or len(hel1os_files) > 0,
        "solexs_files": len(solexs_files),
        "hel1os_files": len(hel1os_files)
    }

def read_solexs(fits_path: str) -> pd.DataFrame:
    """
    Parses a SoLEXS FITS file from ISSDC PRADAN.
    Expected output columns: [time, soft_flux]
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"SoLEXS FITS file not found: {fits_path}")
        
    print(f"Reading SoLEXS FITS: {fits_path}")
    with fits.open(fits_path) as hdul:
        hdul.info()  # Print the HDU list for logging/user review
        # Extract binary table data from second HDU (index 1)
        data = hdul[1].data
        df = pd.DataFrame(data)
        
        # Look for time and flux columns dynamically
        time_col = None
        flux_col = None
        for col in df.columns:
            if "time" in col.lower():
                time_col = col
            if "flux" in col.lower() or "count" in col.lower() or "rate" in col.lower():
                flux_col = col
                
        # Default fallback column names if not auto-detected
        if not time_col:
            time_col = df.columns[0]
        if not flux_col:
            flux_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        df = df[[time_col, flux_col]].rename(columns={time_col: 'time', flux_col: 'soft_flux'})
        
        # Convert time to datetime if it's float/int (relative seconds or JD/MJD)
        # Assuming ISSDC standard fits time or relative seconds since file start
        # Let's keep it as timestamp or convert:
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='ignore')
        return df

def read_hel1os(fits_path: str) -> pd.DataFrame:
    """
    Parses a HEL1OS FITS file from ISSDC PRADAN.
    Expected output columns: [time, hard_flux]
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"HEL1OS FITS file not found: {fits_path}")
        
    print(f"Reading HEL1OS FITS: {fits_path}")
    with fits.open(fits_path) as hdul:
        hdul.info()  # Print the HDU list for logging/user review
        data = hdul[1].data
        df = pd.DataFrame(data)
        
        time_col = None
        flux_col = None
        for col in df.columns:
            if "time" in col.lower():
                time_col = col
            if "flux" in col.lower() or "count" in col.lower() or "rate" in col.lower():
                flux_col = col
                
        if not time_col:
            time_col = df.columns[0]
        if not flux_col:
            flux_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        df = df[[time_col, flux_col]].rename(columns={time_col: 'time', flux_col: 'hard_flux'})
        df['time'] = pd.to_datetime(df['time'], unit='s', errors='ignore')
        return df

def merge_aditya(solexs_df: pd.DataFrame, hel1os_df: pd.DataFrame) -> pd.DataFrame:
    """Merges SoLEXS and HEL1OS dataframes on time at a 1-second cadence."""
    # Ensure time column is index for both
    solexs = solexs_df.set_index('time').sort_index()
    hel1os = hel1os_df.set_index('time').sort_index()
    
    # Merge and resample to 1-second cadence
    merged = pd.merge_asof(solexs, hel1os, left_index=True, right_index=True, direction='nearest')
    merged = merged.resample('1s').mean().ffill()
    
    # Compute hard/soft X-ray ratio for ML features
    # Clip denominator to prevent division by zero
    merged['hard_soft_ratio'] = merged['hard_flux'] / merged['soft_flux'].clip(lower=1e-8)
    
    return merged.reset_index()

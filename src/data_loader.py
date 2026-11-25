import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from src.config import (
    RAW_COLUMN_MAPPING,
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)

def clean_timestamp_string(val: Union[str, pd.Timestamp]) -> Optional[pd.Timestamp]:
    """
    Cleans and parses various timestamp formats, including typos like '7/1/2026 0:00 tt'.
    """
    if pd.isnull(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    
    val_str = str(val).strip()
    # Handle known anomaly 'tt' or stray characters
    val_str = re.sub(r'\s*tt\s*$', '', val_str, flags=re.IGNORECASE)
    
    try:
        return pd.to_datetime(val_str)
    except Exception:
        # Fallback with flexible parser
        return pd.to_datetime(val_str, errors='coerce')

def load_aws_dataset(
    file_path: Optional[Union[str, Path]] = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE
) -> pd.DataFrame:
    """
    Loads AWS and groundwater sensor data from an Excel or CSV file.
    Performs initial structure validation, timestamp cleaning, and coordinate embedding.
    """
    if file_path is None:
        file_path = DEFAULT_RAW_DATA_PATH
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"[*] Loading dataset from: {file_path}")
    if file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    print(f"[*] Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Rename known raw columns to standard clean names
    rename_dict = {}
    for col in df.columns:
        if col in RAW_COLUMN_MAPPING:
            rename_dict[col] = RAW_COLUMN_MAPPING[col]
        else:
            # Check lowercase match
            col_clean = col.strip()
            if "latitude" in col_clean.lower() or "lat" in col_clean.lower():
                rename_dict[col] = "latitude"
            elif "longitude" in col_clean.lower() or "lon" in col_clean.lower() or "long" in col_clean.lower():
                rename_dict[col] = "longitude"
    
    df = df.rename(columns=rename_dict)

    # Clean and parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(clean_timestamp_string)
        # Interpolate or fill NaT timestamps if any (time-series continuity)
        if df["timestamp"].isnull().any():
            null_count = df["timestamp"].isnull().sum()
            print(f"[!] Found {null_count} unparseable timestamps. Imputing based on time step...")
            df["timestamp"] = df["timestamp"].interpolate(method="linear")
        
        # Sort chronologically
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        # Remove exact duplicate timestamps if any
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Check or assign coordinates
    if "latitude" not in df.columns:
        df["latitude"] = latitude
        print(f"[*] Embedded default station Latitude: {latitude}")
    if "longitude" not in df.columns:
        df["longitude"] = longitude
        print(f"[*] Embedded default station Longitude: {longitude}")

    print(f"[+] Loaded and standardized {len(df)} records successfully.")
    return df

if __name__ == "__main__":
    df_sample = load_aws_dataset()
    print(df_sample.info())
    print(df_sample.head())

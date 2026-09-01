"""
PART 1 — Data Processing
=========================
Downloads the UCI "Individual Household Electric Power Consumption" dataset,
cleans it, engineers time-based features, remaps timestamps to current dates,
and saves cleaned_energy_data.csv.

Dataset columns:
  Date, Time, Global_active_power, Global_reactive_power, Voltage,
  Global_intensity, Sub_metering_1 (kitchen), Sub_metering_2 (laundry),
  Sub_metering_3 (water heater + AC)

Run FIRST before anything else:
    python data.py
"""

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_TXT = os.path.join(DATA_DIR, "household_power_consumption.txt")
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")

# UCI ML Repository download URL
UCI_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00235/household_power_consumption.zip"
)


def download_dataset():
    """Download the zipped dataset from UCI and extract the txt file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_TXT):
        print("[OK] Raw data file already exists — skipping download.")
        return

    zip_path = os.path.join(DATA_DIR, "dataset.zip")
    print("[..] Downloading dataset from UCI (~20 MB) ...")
    urllib.request.urlretrieve(UCI_URL, zip_path)

    print("[..] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("[OK] Downloaded and extracted.")


def load_and_clean() -> pd.DataFrame:
    """
    Load the semicolon-separated txt file.
    - Combines Date + Time into a single 'datetime' column.
    - Coerces numeric columns (the dataset marks NaN as '?').
    - Drops rows where the target (Global_active_power) is missing.
    - Forward-fills then back-fills any remaining NaN.
    """
    print("[..] Loading and cleaning ...")

    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values=["?"],  # dataset uses '?' for missing values
        low_memory=False,
    )

    # Combine Date and Time into a single datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df.drop(columns=["Date", "Time"], inplace=True)

    # Ensure every measurement column is truly numeric
    num_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where the target is completely missing
    df.dropna(subset=["Global_active_power"], inplace=True)

    # Forward-fill then back-fill remaining NaN in other columns
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print(f"[OK] Rows after cleaning: {len(df):,}")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level data to hourly averages.
    This reduces ~2M rows to ~34K rows — manageable for a Streamlit demo.
    """
    print("[..] Resampling to hourly frequency ...")
    df = df.set_index("datetime").sort_index()

    hourly = df.resample("h").agg(
        {
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "mean",
            "Sub_metering_2": "mean",
            "Sub_metering_3": "mean",
        }
    )
    hourly.dropna(inplace=True)
    hourly = hourly.reset_index()

    print(f"[OK] Hourly rows: {len(hourly):,}")
    return hourly


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features that the ML models and dashboard need:
      - hour          (0-23)
      - day_of_week   (0=Mon ... 6=Sun)
      - is_weekend    (1 if Sat/Sun, else 0)
      - day_of_month  (1-31)
      - week_of_year  (1-53)

    These are needed for:
      - Calendar view coloring (day_of_month, week_of_year)
      - Weekly/monthly aggregation (day_of_week, week_of_year)
      - ML model input (hour, day_of_week, is_weekend)
    """
    print("[..] Engineering features ...")
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    return df


def remap_to_current_dates(df: pd.DataFrame, last_n_days: int = 90) -> pd.DataFrame:
    """
    Re-map the dataset's timestamps so the most recent data points correspond
    to today's date range, while preserving ALL usage values and patterns.

    Keeps each row's Hour, Day_of_week, and relative position in the dataset
    exactly as they are (preserving real usage patterns). Only the calendar
    dates are shifted.

    How it works:
      1. Takes the last `last_n_days` worth of hourly data from the dataset.
      2. Sets the FIRST row's timestamp to "today minus N days" at midnight.
      3. Every subsequent row's timestamp increases at the same hourly interval
         as the original data.

    Result: the "latest" row has a timestamp of today or yesterday, and the
    calendar view shows the CURRENT month by default.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'datetime' column (already cleaned and resampled).
    last_n_days : int
        Number of trailing days to include in the remapped window (default 90).

    Returns
    -------
    pd.DataFrame with remapped datetime column and updated time features.
    """
    print(f"[..] Remapping timestamps to current dates (last {last_n_days} days) ...")
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep the last N days of hourly data
    last_ts = df["datetime"].max()
    cutoff = last_ts - pd.Timedelta(days=last_n_days)
    df = df[df["datetime"] >= cutoff].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Compute the original hourly interval from the data
    if len(df) > 1:
        median_interval = df["datetime"].diff().dropna().median()
    else:
        median_interval = pd.Timedelta(hours=1)

    # Map the first row to "today minus N days" at midnight
    # If data spans fewer days than last_n_days, anchor so latest data ends near today
    today = pd.Timestamp.now().normalize()
    actual_span = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 86400
    effective_days = min(last_n_days, max(actual_span, 1))
    anchor = today - pd.Timedelta(days=effective_days)

    # Create new datetime range starting from anchor at the original interval
    new_datetimes = pd.date_range(
        start=anchor, periods=len(df), freq=median_interval
    )
    df["datetime"] = new_datetimes

    # Re-extract time features from the new timestamps
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)

    print(f"[OK] Remapped {len(df):,} rows: {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def main():
    """Full pipeline: download -> clean -> resample -> feature engineer -> remap -> save."""
    download_dataset()
    df = load_and_clean()
    df = resample_hourly(df)
    df = engineer_features(df)
    df = remap_to_current_dates(df, last_n_days=90)

    # Save cleaned dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"  CLEANED DATASET SAVED -> {CLEAN_CSV}")
    print(f"{'='*55}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Range   : {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

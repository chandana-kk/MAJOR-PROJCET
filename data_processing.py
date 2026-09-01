"""
PART 1 — Data Processing
=========================
Downloads the UCI "Individual Household Electric Power Consumption" dataset,
cleans it, engineers time-based features, and saves cleaned_energy_data.csv.

Dataset columns:
  Date, Time, Global_active_power, Global_reactive_power, Voltage,
  Global_intensity, Sub_metering_1 (kitchen), Sub_metering_2 (laundry),
  Sub_metering_3 (water heater + AC)

Run FIRST before anything else:
    python data_processing.py
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
    Add time-based features that the ML models will use:
      - hour          (0-23)
      - day_of_week   (0=Mon … 6=Sun)
      - is_weekend    (1 if Sat/Sun, else 0)
    """
    print("[..] Engineering features ...")
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def main():
    """Full pipeline: download → clean → resample → feature engineer → save."""
    download_dataset()
    df = load_and_clean()
    df = resample_hourly(df)
    df = engineer_features(df)

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

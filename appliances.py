"""
PART 2 — Appliance-Level Usage Breakdown
=========================================
Maps the three sub-metering channels to labeled appliance categories
and computes an "Other" category for unmetered usage.

Sub-metering channels (from the UCI dataset):
  Sub_metering_1 -> Kitchen        (watt-hours per minute)
  Sub_metering_2 -> Laundry Room   (watt-hours per minute)
  Sub_metering_3 -> Water Heater & AC (watt-hours per minute)

Unit conversion note:
  - Sub_metering values are in WATT-HOURS per minute (energy consumed
    by that sub-meter during each minute-level reading).
  - Global_active_power is in KILOWATTS (instantaneous power draw).
  - To make them comparable we convert Global_active_power to watt-hours:
      energy (Wh) = power (kW) * 1000 * time_fraction
    For hourly resampled data, each row represents 1 hour, so:
      energy (Wh) = Global_active_power (kW) * 1000 * 1 hour = Wh
    The sub_metering values are already in Wh (they represent total Wh
    over the measurement interval), so we just compare directly after
    converting Global_active_power to Wh.

Usage:
    from appliances import get_appliance_breakdown
    breakdown = get_appliance_breakdown(df, "2007-01-01", "2007-01-07")
"""

import pandas as pd

# Named categories for each sub-metering channel
APPLIANCE_MAP = {
    "Sub_metering_1": "Kitchen",
    "Sub_metering_2": "Laundry Room",
    "Sub_metering_3": "Water Heater & AC",
}


def compute_appliance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four appliance energy columns (in Wh) to the dataframe.

    The sub_metering channels already report energy in Wh.
    Global_active_power is in kW; multiply by 1000 to get Wh per hour.
    The "Other" category = total Wh - sum of three sub-meters.
    """
    df = df.copy()

    # Convert global power from kW to Wh (each row = 1 hour)
    df["_total_wh"] = df["Global_active_power"] * 1000.0

    # Sub-meters are already in Wh
    for col, label in APPLIANCE_MAP.items():
        df[f"_{label}_wh"] = df[col]

    # Other = total minus the three known sub-meters (floor at 0)
    df["_Other_wh"] = (
        df["_total_wh"]
        - df["_Kitchen_wh"]
        - df["_Laundry Room_wh"]
        - df["_Water Heater & AC_wh"]
    ).clip(lower=0)

    return df


def get_appliance_breakdown(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """
    Return total energy usage (Wh) per appliance category for a date range.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with datetime column and sub_metering columns.
    start, end : str
        Date range strings parseable by pd.to_datetime (e.g. "2007-01-01").
        The range is inclusive on both ends.

    Returns
    -------
    pd.DataFrame with columns: Appliance, Wh, Pct
    """
    # Filter to the requested date range (inclusive)
    mask = (df["datetime"] >= pd.to_datetime(start)) & (
        df["datetime"] <= pd.to_datetime(end)
    )
    subset = df.loc[mask].copy()

    if subset.empty:
        return pd.DataFrame(columns=["Appliance", "Wh", "Pct"])

    subset = compute_appliance_columns(subset)

    # Sum each category over the range
    totals = {}
    for label in ["Kitchen", "Laundry Room", "Water Heater & AC", "Other"]:
        totals[label] = subset[f"_{label}_wh"].sum()

    result = pd.DataFrame(
        [{"Appliance": k, "Wh": v} for k, v in totals.items()]
    )
    total_wh = result["Wh"].sum()
    result["Pct"] = (result["Wh"] / total_wh * 100).round(1) if total_wh > 0 else 0.0

    return result.sort_values("Wh", ascending=False).reset_index(drop=True)


def get_appliance_for_hour(row: pd.Series) -> dict:
    """
    Given a single row (dict-like), return a dict with the energy
    consumed by each appliance in that hour (in Wh).

    Useful for the real-time dashboard to show a per-hour breakdown.
    """
    total_wh = row.get("Global_active_power", 0) * 1000.0
    kitchen = row.get("Sub_metering_1", 0)
    laundry = row.get("Sub_metering_2", 0)
    wh_ac = row.get("Sub_metering_3", 0)
    other = max(0, total_wh - kitchen - laundry - wh_ac)

    return {
        "Kitchen": kitchen,
        "Laundry Room": laundry,
        "Water Heater & AC": wh_ac,
        "Other": other,
        "Total": total_wh,
    }

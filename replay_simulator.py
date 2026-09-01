"""
PART 3 — Simulated Real-Time Replay
====================================
Since there is no physical smart meter, this module replays
cleaned_energy_data.csv row-by-row with a configurable delay
to simulate a live data stream.

The dashboard (app.py) creates a ReplaySimulator instance and stores it
in st.session_state so it persists across Streamlit reruns.  The background
thread keeps advancing through the data while the dashboard reads the
"latest row" to display live metrics.

Usage in Streamlit:
    from replay_simulator import ReplaySimulator
    sim = ReplaySimulator(delay=3.0)
    sim.start()
    row = sim.latest_row        # dict with the current row
    window = sim.get_window(50) # last 50 rows as a DataFrame

Standalone test:
    python replay_simulator.py
"""

import os
import time
import threading

import pandas as pd

# Path to the cleaned CSV produced by data_processing.py
CLEAN_CSV = os.path.join("data", "cleaned_energy_data.csv")


class ReplaySimulator:
    """
    Replays historical electricity data one row at a time in a background
    thread, mimicking a live smart-meter feed.
    """

    def __init__(self, delay: float = 3.0, csv_path: str = CLEAN_CSV):
        """
        Parameters
        ----------
        delay : float
            Seconds to wait between consecutive rows.
            Default is 3 seconds (one row = one simulated "hour").
        csv_path : str
            Path to the cleaned CSV file.
        """
        self.delay = delay
        self.csv_path = csv_path
        self._df: pd.DataFrame = pd.DataFrame()
        self._index: int = 0
        self._current_row: dict | None = None
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._load_data()

    # ── Internal ────────────────────────────────────────────────────

    def _load_data(self):
        """Load the cleaned CSV into memory."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"{self.csv_path} not found.\nRun:  python data_processing.py"
            )
        self._df = pd.read_csv(self.csv_path, parse_dates=["datetime"])

    def _replay_loop(self):
        """Background loop: advance one row every `self.delay` seconds."""
        while self._running and self._index < len(self._df):
            # Store the current row as a dict so readers can access it
            row = self._df.iloc[self._index]
            self._current_row = row.to_dict()
            self._index += 1
            time.sleep(self.delay)

    # ── Public API ──────────────────────────────────────────────────

    def start(self):
        """Start the background replay thread. Safe to call multiple times."""
        if self._running:
            return  # already running
        self._running = True
        self._index = 0
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Pause the replay (can be resumed with start())."""
        self._running = False

    def reset(self):
        """Stop and rewind to the beginning."""
        self.stop()
        self._index = 0
        self._current_row = None
        time.sleep(0.1)  # brief pause to let the thread notice it stopped

    @property
    def latest_row(self) -> dict | None:
        """
        Return the most recently replayed row as a dict.
        Returns None if replay hasn't started yet.

        Example keys: datetime, Global_active_power, Voltage,
        Sub_metering_1, hour, day_of_week, is_weekend, etc.
        """
        return self._current_row

    @property
    def progress(self) -> tuple:
        """Return (rows_played_so_far, total_rows_in_dataset)."""
        return self._index, len(self._df)

    @property
    def is_finished(self) -> bool:
        """True if all rows have been replayed."""
        return self._index >= len(self._df)

    @property
    def is_running(self) -> bool:
        """True if the background thread is active."""
        return self._running

    def get_window(self, n: int = 50) -> pd.DataFrame:
        """
        Return the last `n` rows that have been replayed so far.
        Useful for showing a recent trend chart or building an LSTM window.
        """
        if self._index == 0:
            return pd.DataFrame()
        start = max(0, self._index - n)
        return self._df.iloc[start : self._index].copy()

    def get_full_history(self) -> pd.DataFrame:
        """Return ALL rows replayed so far (from the beginning)."""
        if self._index == 0:
            return pd.DataFrame()
        return self._df.iloc[: self._index].copy()

    def get_row_at(self, idx: int) -> dict | None:
        """Return a specific row by index (for debugging)."""
        if 0 <= idx < len(self._df):
            return self._df.iloc[idx].to_dict()
        return None


# ── Standalone test ──────────────────────────────────────────────────
# Run this file directly to see the replay in action in the terminal.
if __name__ == "__main__":
    print("Replay Simulator — standalone test")
    print("Press Ctrl+C to stop.\n")

    sim = ReplaySimulator(delay=1.0)  # 1 second per row for quick test
    sim.start()

    try:
        while not sim.is_finished:
            row = sim.latest_row
            if row:
                ts = row.get("datetime", "?")
                gap = row.get("Global_active_power", 0)
                idx, total = sim.progress
                print(f"  [{idx}/{total}] {ts}  —  {gap:.3f} kW")
            time.sleep(0.5)
    except KeyboardInterrupt:
        sim.stop()
        print("\nStopped by user.")

    print("Replay finished.")

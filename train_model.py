"""
PART 2 — Hybrid AI Model Training
==================================
Trains an XGBoost regressor and an LSTM neural network on the cleaned
household electricity data, then combines them into a hybrid predictor.

Models are saved to the models/ directory:
  - xgboost_model.pkl   — trained XGBoost (scikit-learn compatible)
  - lstm_model.keras     — trained LSTM (TensorFlow/Keras)
  - lstm_scaler.pkl      — MinMaxScaler for LSTM input normalisation

Run AFTER data_processing.py:
    python train_model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")

# ── Model settings ─────────────────────────────────────────────────────
TARGET = "Global_active_power"  # the column we want to predict (in kW)

# Features used by both XGBoost and LSTM (time + raw measurements)
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

WINDOW_SIZE = 10  # LSTM sliding-window length (past 10 hours)
TEST_FRACTION = 0.2  # hold out last 20 % for evaluation
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the cleaned CSV that data_processing.py produced."""
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"{CLEAN_CSV} not found.\nRun:  python data_processing.py"
        )
    return pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])


def split_time(df, test_frac=TEST_FRACTION):
    """
    Time-ordered train/test split (no shuffling!).
    The first (1-test_frac) rows train, the rest evaluate.
    """
    idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


# ──────────────────────────────────────────────────────────────────────
# XGBoost training
# ──────────────────────────────────────────────────────────────────────

def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train an XGBoost regressor on time + measurement features.
    Returns (trained_model, predictions_on_test, MAE).
    """
    print("\n" + "=" * 55)
    print("  Training XGBoost Regressor")
    print("=" * 55)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=200,  # number of trees
        max_depth=6,  # max depth of each tree
        learning_rate=0.1,  # how much each tree contributes
        random_state=SEED,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  XGBoost MAE : {mae:.4f} kW")
    return model, preds, mae


# ──────────────────────────────────────────────────────────────────────
# LSTM training
# ──────────────────────────────────────────────────────────────────────

def make_sequences(features_2d: np.ndarray, target_1d: np.ndarray, window: int):
    """
    Build sliding-window (X, y) pairs for the LSTM.

    For a window size of 10, each training sample is:
      X = features[i-10 : i]   (10 consecutive hours of all features)
      y = target[i]             (the next hour's power usage)

    This lets the LSTM learn from temporal patterns.
    """
    X, y = [], []
    for i in range(window, len(features_2d)):
        X.append(features_2d[i - window : i])
        y.append(target_1d[i])
    return np.array(X), np.array(y)


def train_lstm(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train a 2-layer LSTM neural network using a sliding window.
    Returns (trained_model, scaler, predictions_on_test, MAE).
    """
    print("\n" + "=" * 55)
    print("  Training LSTM Neural Network")
    print("=" * 55)

    # Scale features to [0, 1] — LSTMs are sensitive to input scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[FEATURE_COLS + [TARGET]])
    test_scaled = scaler.transform(test[FEATURE_COLS + [TARGET]])

    y_train = train[TARGET].values
    y_test = test[TARGET].values

    # Create sliding window sequences
    X_train_seq, y_train_seq = make_sequences(train_scaled, y_train, WINDOW_SIZE)
    X_test_seq, y_test_seq = make_sequences(test_scaled, y_test, WINDOW_SIZE)

    n_features = X_train_seq.shape[2]  # number of input features per timestep

    # Build the LSTM architecture:
    #   Layer 1: LSTM with 64 units, returns sequences for stacking
    #   Dropout 20% to prevent overfitting
    #   Layer 2: LSTM with 32 units
    #   Dropout 20%
    #   Dense hidden layer (16 neurons, ReLU)
    #   Output: single neuron (predicted kW value)
    model = keras.Sequential(
        [
            keras.layers.LSTM(
                64,
                return_sequences=True,
                input_shape=(WINDOW_SIZE, n_features),
            ),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("  Training (this may take a few minutes) ...")
    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=20,
        batch_size=32,
        validation_split=0.1,  # use 10% of training data for validation
        verbose=1,
    )

    preds = model.predict(X_test_seq, verbose=0).flatten()
    mae = mean_absolute_error(y_test_seq, preds)
    print(f"  LSTM MAE    : {mae:.4f} kW")
    return model, scaler, preds, mae


# ──────────────────────────────────────────────────────────────────────
# Hybrid model
# ──────────────────────────────────────────────────────────────────────

def compute_hybrid(xgb_preds, lstm_preds, y_true):
    """
    Average the XGBoost and LSTM predictions to form the hybrid output.

    IMPORTANT: LSTM predictions start at index WINDOW_SIZE (because
    make_sequences discards the first WINDOW_SIZE rows).  XGBoost
    predictions cover every test row.  We must align them by slicing
    XGBoost preds to start at the same position as LSTM preds.

    Returns (hybrid_predictions, hybrid_MAE).
    """
    n = min(len(xgb_preds), len(lstm_preds))
    # Align: XGBoost preds from [WINDOW_SIZE:] match LSTM preds from [0:]
    hybrid = (xgb_preds[WINDOW_SIZE : WINDOW_SIZE + n] + lstm_preds[:n]) / 2.0
    # The true values that correspond to these aligned predictions
    aligned_y = y_true[WINDOW_SIZE : WINDOW_SIZE + n]
    mae = mean_absolute_error(aligned_y, hybrid)
    return hybrid, mae


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data and split
    df = load_data()
    train, test = split_time(df)
    y_test = test[TARGET].values

    # Train both models
    xgb_model, xgb_preds, xgb_mae = train_xgboost(train, test)
    lstm_model, scaler, lstm_preds, lstm_mae = train_lstm(train, test)

    # Combine into hybrid
    _, h_mae = compute_hybrid(xgb_preds, lstm_preds, y_test)

    # Print comparison table
    print(f"\n{'=' * 55}")
    print("  MODEL COMPARISON (lower MAE = better)")
    print("-" * 55)
    print(f"  {'Model':<18} {'MAE (kW)':>12}")
    print("-" * 55)
    print(f"  {'XGBoost':<18} {xgb_mae:>12.4f}")
    print(f"  {'LSTM':<18} {lstm_mae:>12.4f}")
    print(f"  {'Hybrid (avg)':<18} {h_mae:>12.4f}")
    print("=" * 55)

    # Save XGBoost model as pickle
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"\n[OK] XGBoost model  -> {xgb_path}")

    # Save LSTM model as Keras format
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    lstm_model.save(lstm_path)
    print(f"[OK] LSTM model     -> {lstm_path}")

    # Save the scaler (needed at inference time to preprocess LSTM inputs)
    scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] LSTM scaler    -> {scaler_path}")

    # Save the hybrid MAE value so the dashboard can use it
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    meta = {
        "xgb_mae": xgb_mae,
        "lstm_mae": lstm_mae,
        "hybrid_mae": h_mae,
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "window_size": WINDOW_SIZE,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[OK] Model metadata -> {meta_path}")


if __name__ == "__main__":
    main()

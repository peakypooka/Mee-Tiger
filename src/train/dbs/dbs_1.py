import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from pathlib import Path
from src.db.postgres import PostgresDB

CSV_PATH = Path(__file__).parent / "exported_data.csv"

class DBS_Data:
    
    def load_all_data():
        data = PostgresDB.fetch_all_data()
        return data
    
    def load_single_row():
        data = PostgresDB.fetch_one()
        return data

class MachineLearning:
    """
    Time-series prediction for CDU power.
    - Filters rows where variable == 'power' AND (model == 'cdu' OR device_id contains 'cdu')
    - Resamples to a fixed interval
    - Builds lag features
    - Trains a RandomForestRegressor
    - Evaluates on the most recent holdout slice
    - Produces a simple recursive forecast for the next horizon
    """

    @staticmethod
    def _col(name_guess: str, fallback_idx: int, frame: pd.DataFrame):
        """Resolve a column by name (case-insensitive) or fallback to positional index."""
        if isinstance(frame.columns, pd.Index):
            for c in frame.columns:
                if str(c).lower() == name_guess:
                    return c
        return frame.columns[fallback_idx] if frame.columns.size > fallback_idx else fallback_idx

    @staticmethod
    def _prepare_cdu_power(df: pd.DataFrame, resample_rule: str = "5min") -> pd.Series:
        """Return a single time series (sum across devices) of CDU power."""
        if df is None or df.empty:
            return pd.Series(dtype=float)

        ts_col = MachineLearning._col("timestamp", 0, df)
        device_col = MachineLearning._col("device_id", 1, df)
        variable_col = MachineLearning._col("variable", 2, df)
        value_col = MachineLearning._col("value", 3, df)
        model_col = MachineLearning._col("model", 4, df)

        # Ensure timestamp and numeric value
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        val = pd.to_numeric(df[value_col], errors="coerce")

        tmp = pd.DataFrame({
            "ts": ts,
            "device": df[device_col].astype(str),
            "variable": df[variable_col].astype(str).str.lower(),
            "model": df[model_col].astype(str).str.lower(),
            "value": val
        }).dropna(subset=["ts", "value"])

        # Filter to CDU power rows
        is_power = tmp["variable"].eq("power")
        is_cdu = tmp["model"].eq("cdu") | tmp["device"].str.contains("cdu", case=False, na=False)
        cdu = tmp.loc[is_power & is_cdu, ["ts", "value"]]
        if cdu.empty:
            return pd.Series(dtype=float)

        # Resample (sum) to regular interval
        ts_series = (
            cdu.set_index("ts")
                .sort_index()
                .resample(resample_rule)
                .sum()["value"]
                .astype(float)
        )
        # Drop leading/trailing NaNs introduced by resampling (keep internal NaNs as zeros)
        ts_series = ts_series.fillna(0.0)
        return ts_series

    @staticmethod
    def _make_supervised(ts: pd.Series, lags=(1, 2, 3, 6, 12, 24)):
        """Create a supervised learning table from a 1D series using lag features."""
        dfX = pd.DataFrame({"y": ts})
        for L in lags:
            dfX[f"lag_{L}"] = ts.shift(L)
        # Optional calendar features
        dfX["hour"] = dfX.index.hour
        dfX["dow"] = dfX.index.dayofweek
        dfX = dfX.dropna()

        y = dfX["y"].values
        X = dfX.drop(columns=["y"]).values
        idx = dfX.index
        return X, y, idx

    @staticmethod
    def train_and_forecast(df: pd.DataFrame,
                           resample_rule: str = "5min",
                           lags=(1, 2, 3, 6, 12, 24),
                           holdout_ratio: float = 0.2,
                           horizon: int = 12):
        """
        Train on historical CDU power and forecast the next `horizon` steps.
        Returns a dict with model, metrics, fitted arrays, and forecast series.
        """
        ts = MachineLearning._prepare_cdu_power(df, resample_rule=resample_rule)
        if ts.empty or ts.size < max(lags) + 10:
            return {
                "ok": False,
                "reason": "Not enough CDU power data after filtering/resampling.",
            }

        X, y, idx = MachineLearning._make_supervised(ts, lags=lags)
        n = len(y)
        split = max(int(n * (1 - holdout_ratio)), 1)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        idx_tr, idx_te = idx[:split], idx[split:]

        # Model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        # Evaluate
        y_hat_te = model.predict(X_te) if len(y_te) > 0 else np.array([])
        mae = float(mean_absolute_error(y_te, y_hat_te)) if len(y_te) > 0 else None
        r2 = float(r2_score(y_te, y_hat_te)) if len(y_te) > 0 else None

        # Recursive forecast using the last known window
        last_index = idx[-1]
        freq = pd.infer_freq(idx) or resample_rule  # fallback to provided rule
        # Build the latest feature row from the tail of ts
        tail = ts.copy()
        # Forecast horizon steps
        preds = []
        cur_ts = tail.copy()

        for step in range(horizon):
            # Recompute last features each step (lags shift as we append y_pred)
            feat = []
            for L in lags:
                feat.append(cur_ts.iloc[-L])
            # calendar feats for next timestamp
            next_time = last_index + pd.tseries.frequencies.to_offset(freq) * (step + 1)
            feat.extend([next_time.hour, next_time.dayofweek])
            feat = np.array(feat).reshape(1, -1)
            y_next = float(model.predict(feat)[0])
            preds.append((next_time, y_next))
            # append to series for subsequent lags
            cur_ts.loc[next_time] = y_next

        forecast_index = pd.DatetimeIndex([t for t, _ in preds], tz="UTC")
        forecast_values = np.array([v for _, v in preds], dtype=float)
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="forecast")

        return {
            "ok": True,
            "model": model,
            "mae": mae,
            "r2": r2,
            "y_test_index": idx_te,
            "y_test": y_te,
            "y_pred": y_hat_te,
            "forecast": forecast_series,
            "resample_rule": resample_rule,
            "lags": lags,
        }


# Test print single row
print("Testing DBS_Data.load_single_row()...", flush=True)
row = DBS_Data.load_single_row()
print(f"Fetched row: {row}", flush=True)
if row is None:
    print("No data fetched.", flush=True)
else:
    print(f"Fetched {len(row)} columns.", flush=True)

# Save to CSV and append only new data (query only the delta)
data = None
last_ts = None

# Determine columns from DB
cols = PostgresDB.get_columns()

# Find watermark from existing CSV
if CSV_PATH.exists():
    try:
        # If we know the timestamp column name, we can parse by that; otherwise parse first column
        if cols and len(cols) > 0:
            ts_col_name = cols[0]
            existing_df = pd.read_csv(CSV_PATH, parse_dates=[ts_col_name])
            last_ts = existing_df[ts_col_name].max() if not existing_df.empty else None
        else:
            existing_df = pd.read_csv(CSV_PATH, parse_dates=[0])
            last_ts = existing_df.iloc[:, 0].max() if not existing_df.empty else None
    except Exception as e:
        print(f"Warning: failed to read existing CSV ({e}), falling back to full fetch.", flush=True)
        last_ts = None

# Fetch only new rows if we have a watermark, else fetch all
if last_ts is not None and not pd.isna(last_ts):
    # Use first column name as timestamp column unless you want to hardcode it
    ts_col_name = cols[0] if cols else "timestamp"
    print(f"Fetching rows since {last_ts} using ts_col '{ts_col_name}' ...", flush=True)
    data = PostgresDB.fetch_since(last_ts, ts_col=ts_col_name)
else:
    print("No watermark found; fetching full dataset once.", flush=True)
    data = PostgresDB.fetch_all_data()

if data is None or len(data) == 0:
    print("No data fetched for CSV export.", flush=True)
else:
    # Build DataFrame with column names if available
    df = pd.DataFrame(data, columns=cols if cols else None)

    # Ensure timestamp column is parsed as datetime
    if cols and len(cols) > 0:
        ts_col_name = cols[0]
        try:
            df[ts_col_name] = pd.to_datetime(df[ts_col_name], errors='coerce', utc=True)
        except Exception:
            pass
    else:
        # if no cols available, try first column
        try:
            df[0] = pd.to_datetime(df[0], errors='coerce', utc=True)
        except Exception:
            pass

    # Ensure directory exists
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append-only write (header only if file doesn't exist or empty)
    write_header = not CSV_PATH.exists()
    if not write_header:
        try:
            if CSV_PATH.stat().st_size == 0:
                write_header = True
        except Exception:
            pass

    df.to_csv(CSV_PATH, mode="a" if CSV_PATH.exists() else "w", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {CSV_PATH}", flush=True)
    
    
# === Train & forecast CDU power ===
try:
    results = MachineLearning.train_and_forecast(df, resample_rule="5min", lags=(1,2,3,6,12,24), holdout_ratio=0.2, horizon=12)
    if not results.get("ok"):
        print(f"[CDU ML] Skipped training: {results.get('reason')}", flush=True)
    else:
        print(f"[CDU ML] Test MAE: {results['mae']:.4f} | R2: {results['r2']:.4f}", flush=True)
        # Save forecast to a CSV next to exported_data.csv
        forecast_path = CSV_PATH.parent / "cdu_power_forecast.csv"
        results["forecast"].to_csv(forecast_path, header=True)
        print(f"[CDU ML] Saved {len(results['forecast'])}‑step forecast to {forecast_path}", flush=True)
except Exception as e:
    print(f"[CDU ML] Error during training/forecast: {e}", flush=True)

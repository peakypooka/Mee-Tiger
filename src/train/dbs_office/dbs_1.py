import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pycaret.regression import setup, compare_models, tune_model, finalize_model, predict_model
from pathlib import Path
from src.db.postgres import PostgresDB
from sklearn.metrics import mean_absolute_error, r2_score


CSV_PATH = Path(__file__).parent / "exported_data.csv"
PLOTS_DIR = Path(__file__).parent / "plots"


class DBS_Data: #ref from the postgres.py
    
    @staticmethod
    def load_all_data():
        data = PostgresDB.fetch_all_data()
        return data
    @staticmethod
    def load_single_row():
        data = PostgresDB.fetch_one()
        return data

class MachineLearning:
    
    @staticmethod
    def CreateColumn(title: str, index: int, frame: pd.DataFrame):
        """Resolve a CreateColumn by name (case-insensitive) or fallback to positional index."""
        if isinstance(frame.columns, pd.Index):
            for content in frame.columns:
                if str(content).lower() == title:
                    return content
        return frame.columns[index] if frame.columns.size > index else index
    
    @staticmethod
    def _prepare_cdu_power(frame: pd.DataFrame, resample_rule: str = "5min") -> pd.Series:
        """Return a single time series (sum across devices) of CDU power."""
        if frame is None or frame.empty:
            return pd.Series(dtype=float)

        ts_col = MachineLearning.CreateColumn("timestamp", 0, frame)
        device_col = MachineLearning.CreateColumn("device_id", 1, frame)
        variable_col = MachineLearning.CreateColumn("variable", 2, frame)
        value_col = MachineLearning.CreateColumn("value", 3, frame)
        model_col = MachineLearning.CreateColumn("model", 4, frame)

        # Ensure timestamp and numeric value
        timestamp = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
        val = pd.to_numeric(frame[value_col], errors="coerce")

        tmp = pd.DataFrame({
            "timestamp": timestamp,
            "device": frame[device_col].astype(str),
            "variable": frame[variable_col].astype(str).str.lower(),
            "model": frame[model_col].astype(str).str.lower(),
            "value": val
        }).dropna(subset=["timestamp", "value"])

        # Filter to CDU power rows
        is_power = tmp["variable"].eq("power")
        is_cdu = tmp["model"].eq("cdu") | tmp["device"].str.contains("cdu", case=False, na=False)
        cdu = tmp.loc[is_power & is_cdu, ["timestamp", "value"]]
        if cdu.empty:
            return pd.Series(dtype=float)

        # Resample (sum) to regular interval
        ts_series = (
            cdu.set_index("timestamp")
                .sort_index()
                .resample(resample_rule)
                .sum()["value"]
                .astype(float)
        )
        
        # Drop leading/trailing NaNs introduced by resampling (keep internal NaNs as zeros)
        ts_series = ts_series.fillna(0.0)
        return ts_series
    
    @staticmethod
    def _make_supervised(timestamp: pd.Series, lags=(1, 2, 3, 6, 12, 24)):
        """Create a supervised learning table from a 1D series using lag features."""
        dfX = pd.DataFrame({"y": timestamp})
        for L in lags:
            dfX[f"lag_{L}"] = timestamp.shift(L)
        # Optional calendar features
        dfX["hour"] = dfX.index.hour
        dfX["dow"] = dfX.index.dayofweek
        dfX = dfX.dropna()

        y = dfX["y"].values
        X = dfX.drop(columns=["y"]).values
        idx = dfX.index
        return X, y, idx
    
    @staticmethod
    def _plot_feature_timeseries(df_feat: pd.DataFrame, out_dir: Path, max_points: int = 1000):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Plot each feature's recent time series
        tail = df_feat.tail(max_points)
        for col in tail.columns:
            try:
                plt.figure(figsize=(10, 3))
                plt.plot(tail.index, tail[col])
                plt.title(f"Feature: {col}")
                plt.xlabel("Time")
                plt.ylabel(col)
                plt.tight_layout()
                plt.savefig(out_dir / f"{col}.png")
                plt.close()
            except Exception as e:
                print(f"[Plot] Skipped {col}: {e}", flush=True)
                
    @staticmethod
    def _plot_feature_timeseries_plotly(df_feat: pd.DataFrame, out_dir: Path, max_points: int = 1000):
        out_dir.mkdir(parents=True, exist_ok=True)
        if df_feat.empty:
            return
        # Use tail to limit file size
        tail = df_feat.tail(max_points)
        # Ensure a Time CreateColumn for Plotly
        time_index = tail.index
        try:
            time_index = pd.to_datetime(time_index)
        except Exception:
            pass
        for col in tail.columns:
            try:
                df_one = pd.DataFrame({"Time": time_index, col: tail[col].values})
                if px is not None:
                    fig = px.line(df_one, x="Time", y=col, title=f"Feature: {col}")
                    fig.write_html(str(out_dir / f"{col}.html"), include_plotlyjs="cdn")
            except Exception as e:
                print(f"[Plot] Plotly feature '{col}' failed: {e}", flush=True)
                
    @staticmethod
    def train_and_forecast(frame: pd.DataFrame,
                           resample_rule: str = "5min",
                           lags=(1, 2, 3, 6, 12, 24),
                           holdout_ratio: float = 0.2,
                           horizon: int = 12):
        
        timestamp = MachineLearning._prepare_cdu_power(frame, resample_rule=resample_rule)
        if timestamp.empty or timestamp.size < max(lags) + 10:
            return {
                "ok": False,
                "reason": "Not enough CDU power data after filtering/resampling.",
            }

        X, y, idx = MachineLearning._make_supervised(timestamp, lags=lags)

        # Use Bangkok (+07:00) for plotting indices
        try:
            idx_local = idx.tz_convert('Asia/Bangkok')
        except Exception:
            # If index is naive, leave as-is for safety
            idx_local = idx

        # Build feature DataFrame for plotting (Bangkok time for plotting only)
        feat_cols = [f"lag_{L}" for L in lags] + ["hour", "dow"]
        df_feat = pd.DataFrame(X, index=idx_local, columns=feat_cols)
        # Plot features: prefer Plotly, fallback to Matplotlib
        if px is not None and go is not None:
            MachineLearning._plot_feature_timeseries_plotly(df_feat, PLOTS_DIR / "features_plotly")
        else:
            MachineLearning._plot_feature_timeseries(df_feat, PLOTS_DIR / "features")

        n = len(y)
        train_frac = 0.8 if holdout_ratio is None else (1 - holdout_ratio)
        split = max(int(n * train_frac), 1)
        print(f"[Split] Using {train_frac*100:.0f}% train / {100 - train_frac*100:.0f}% test -> train={split}, test={n - split}", flush=True)

        X_tr, y_tr = X[:split], y[:split]
        X_te, y_test = X[split:], y[split:]
        idx_tr, idx_te = idx[:split], idx[split:]

        # --- PyCaret pipeline (train on 80%, evaluate on 20%) ---
        # Build DataFrame for PyCaret
        df_full = pd.DataFrame(X, index=idx, columns=feat_cols)
        df_full['y'] = y
        df_tr = df_full.iloc[:split].copy()
        df_te = df_full.iloc[split:].copy()

        print("[PyCaret] setup(...) starting", flush=True)
        s = setup(
            data=df_tr,
            target='y',
            session_id=42,
            fold_strategy='timeseries',
            fold=5,
            data_split_shuffle=False,
            fold_shuffle=False,
            imputation_type='simple',
            numeric_imputation='median',
            normalize=True,
            transform_target=True,
            verbose=False
        )

        print("[PyCaret] compare_models(...)", flush=True)
        best = compare_models(include=None)  # let PyCaret pick
        print(f"[PyCaret] Best: {best}", flush=True)

        print("[PyCaret] tune_model(...)", flush=True)
        tuned = tune_model(best, choose_better=True)
        final = finalize_model(tuned)

        # Evaluate on held-out test set
        if len(df_te) > 0:
            preds_te = predict_model(final, data=df_te)
            # Handle PyCaret's prediction column name across versions
            pred_col = 'prediction_label' if 'prediction_label' in preds_te.columns else ('Label' if 'Label' in preds_te.columns else None)
            if pred_col is None:
                raise RuntimeError('PyCaret did not return a prediction column')
            y_pred = preds_te[pred_col].to_numpy()
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
        else:
            y_pred = np.array([])
            mae, r2 = None, None

        # --- Plotting (Pred vs Actual & Feature Correlations) [PLOTLY] ---
        try:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Only proceed if plotly is available
        if px is None or go is None:
            print("[Plot] Plotly not available; falling back to Matplotlib PNGs.", flush=True)
            # Scatter
            if len(y_pred) > 0:
                try:
                    plt.figure(figsize=(6, 6))
                    plt.scatter(y_test, y_pred, alpha=0.5)
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title(f"Predicted vs Actual (MAE={mae:.2f}, R2={r2:.2f})")
                    mn = min(float(np.min(y_test)), float(np.min(y_pred)))
                    mx = max(float(np.max(y_test)), float(np.max(y_pred)))
                    plt.plot([mn, mx], [mn, mx])
                    plt.tight_layout()
                    plt.savefig(PLOTS_DIR / "pred_vs_actual_scatter.png")
                    plt.close()
                except Exception as e:
                    print(f"[Plot] Matplotlib scatter failed: {e}", flush=True)
            # Time series
            if len(y_pred) > 0 and len(idx_te) == len(y_test):
                try:
                    plt.figure(figsize=(12, 4))
                    try:
                        idx_te_local = idx_te.tz_convert('Asia/Bangkok')
                    except Exception:
                        idx_te_local = idx_te
                    plt.plot(idx_te_local, y_test, label="Actual")
                    plt.plot(idx_te_local, y_pred, label="Predicted")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Power")
                    plt.title("Predicted vs Actual (Test Window)")
                    plt.tight_layout()
                    plt.savefig(PLOTS_DIR / "pred_vs_actual_timeseries.png")
                    plt.close()
                except Exception as e:
                    print(f"[Plot] Matplotlib timeseries failed: {e}", flush=True)
        else:
            # 1) Predicted vs Actual (interactive scatter + 1:1 line)
            if len(y_pred) > 0:
                try:
                    df_scatter = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                    fig_scatter = px.scatter(df_scatter, x="Actual", y="Predicted",
                                             title=f"Predicted vs Actual (MAE={mae:.2f}, R2={r2:.2f})",
                                             opacity=0.6)
                    mn = float(min(df_scatter["Actual"].min(), df_scatter["Predicted"].min()))
                    mx = float(max(df_scatter["Actual"].max(), df_scatter["Predicted"].max()))
                    fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="1:1"))
                    fig_scatter.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
                    fig_scatter.write_html(PLOTS_DIR / "pred_vs_actual_scatter.html", include_plotlyjs="cdn")
                except Exception as e:
                    print(f"[Plot] plotly scatter failed: {e}", flush=True)

            # 2) Predicted vs Actual over time (interactive lines)
            if len(y_pred) > 0 and len(idx_te) == len(y_test):
                try:
                    # Convert test index to Bangkok time for plotting
                    try:
                        idx_te_local = idx_te.tz_convert('Asia/Bangkok')
                    except Exception:
                        idx_te_local = idx_te
                    df_ts = pd.DataFrame({"Time": idx_te_local, "Actual": y_test, "Predicted": y_pred})
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=df_ts["Time"], y=df_ts["Actual"], name="Actual"))
                    fig_ts.add_trace(go.Scatter(x=df_ts["Time"], y=df_ts["Predicted"], name="Predicted"))
                    fig_ts.update_layout(title="Predicted vs Actual (Test Window)", xaxis_title="Time", yaxis_title="Power")
                    fig_ts.write_html(PLOTS_DIR / "pred_vs_actual_timeseries.html", include_plotlyjs="cdn")
                except Exception as e:
                    print(f"[Plot] plotly time-series failed: {e}", flush=True)

            # 3) Feature correlation heatmap (train split)
            try:
                df_corr = df_tr.corr(numeric_only=True)
                if df_corr.shape[0] > 0:
                    fig_corr = px.imshow(df_corr, color_continuous_midpoint=0, title="Feature Correlation Heatmap (Train)")
                    fig_corr.write_html(PLOTS_DIR / "feature_correlation_train.html", include_plotlyjs="cdn")
            except Exception as e:
                print(f"[Plot] plotly correlation failed: {e}", flush=True)

        # Recursive forecast using the last known window
        last_index = idx[-1]
        freq = pd.infer_freq(idx) or resample_rule  # fallback to provided rule
        # Build the latest feature row from the tail of timestamp
        tail = timestamp.copy()
        # Forecast horizon steps
        preds = []
        cur_ts = tail.copy()

        for step in range(horizon):
            feat_vals = []
            for L in lags:
                feat_vals.append(cur_ts.iloc[-L])
            next_time = last_index + pd.tseries.frequencies.to_offset(freq) * (step + 1)
            feat_vals.extend([next_time.hour, next_time.dayofweek])
            df_one = pd.DataFrame([feat_vals], columns=feat_cols)
            pred_one = predict_model(final, data=df_one)
            pred_col = 'prediction_label' if 'prediction_label' in pred_one.columns else ('Label' if 'Label' in pred_one.columns else None)
            if pred_col is None:
                raise RuntimeError('PyCaret did not return a prediction column for forecast step')
            y_next = float(pred_one.iloc[0][pred_col])
            preds.append((next_time, y_next))
            # append to series for subsequent lags
            cur_ts.loc[next_time] = y_next

        # Forecast index in Bangkok time (+07:00)
        try:
            forecast_index = pd.DatetimeIndex([t for t, _ in preds]).tz_convert('Asia/Bangkok')
        except Exception:
            # If timestamps are naive, localize as Bangkok
            forecast_index = pd.DatetimeIndex([t for t, _ in preds]).tz_localize('Asia/Bangkok')
        forecast_values = np.array([v for _, v in preds], dtype=float)
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="forecast")
        print(f"[Forecast] Index tz: {forecast_series.index.tz}", flush=True)

        return {
            "ok": True,
            "model": final,
            "mae": mae,
            "r2": r2,
            "y_test_index": idx_te,
            "y_test": y_test,
            "y_pred": y_pred,
            "forecast": forecast_series,
            "resample_rule": resample_rule,
            "lags": lags,
        }
        
class plotgraph:
    def plotgraph_timeseries():
        pass

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
full_df = None
existing_df = None

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
    # Try to use existing CSV for training if present
    try:
        if CSV_PATH.exists():
            if cols and len(cols) > 0:
                ts_col_name = cols[0]
                existing_df = pd.read_csv(CSV_PATH, parse_dates=[ts_col_name])
            else:
                existing_df = pd.read_csv(CSV_PATH, parse_dates=[0])
            if existing_df is not None and not existing_df.empty:
                full_df = existing_df
                print(f"Loaded existing CSV for training: {len(full_df)} rows from {CSV_PATH}", flush=True)
            else:
                print("Existing CSV is empty; cannot train.", flush=True)
        else:
            print("No existing CSV found; cannot train.", flush=True)
    except Exception as e:
        print(f"Warning: failed to read existing CSV for training ({e}).", flush=True)
else:
    # Build DataFrame with column names if available
    frame = pd.DataFrame(data, columns=cols if cols else None)

    # Ensure timestamp column is parsed as datetime
    if cols and len(cols) > 0:
        ts_col_name = cols[0]
        try:
            frame[ts_col_name] = pd.to_datetime(frame[ts_col_name], errors='coerce', utc=True)
        except Exception:
            pass
    else:
        # if no cols available, try first column
        try:
            frame[0] = pd.to_datetime(frame[0], errors='coerce', utc=True)
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

    frame.to_csv(CSV_PATH, mode="a" if CSV_PATH.exists() else "w", header=write_header, index=False)
    print(f"Appended {len(frame)} rows to {CSV_PATH}", flush=True)

    # Reload full exported CSV to train on entire history (append is only for ingestion)
    try:
        if cols and len(cols) > 0:
            ts_col_name = cols[0]
            full_df = pd.read_csv(CSV_PATH, parse_dates=[ts_col_name])
        else:
            full_df = pd.read_csv(CSV_PATH, parse_dates=[0])
        print(f"Loaded full dataset for training: {len(full_df)} rows from {CSV_PATH}", flush=True)
    except Exception as e:
        print(f"Warning: failed to read full CSV for training ({e}); falling back to in-memory frame.", flush=True)
        full_df = frame
    
    
# === Train & forecast CDU power ===
try:
    if full_df is None or len(full_df) == 0:
        print("[CDU ML] No data available for training (full_df is empty/undefined). Skipping.", flush=True)
    else:
        results = MachineLearning.train_and_forecast(full_df, resample_rule="5min", lags=(1,2,3,6,12,24), holdout_ratio=0.2, horizon=12)
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

#TRAIN THE ENTIRE DATASET TO FIND THE BEST 20 PREDICTION MODEL
#USING PYCARET

import pandas as pd
import numpy as np
from pycaret.regression import *
from sklearn.inspection import permutation_importance
from pathlib import Path
import matplotlib.pyplot as plt
# --- Figure export settings (exact 1920x1080 px) ---
FIG_DPI = 100
FIGSIZE = (1920 / FIG_DPI, 1080 / FIG_DPI)

file_path = "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/src/train/data/chilled_water_loop_1.csv"
tcol = "Plant - Cooling Load (Ton)"
test_path = "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/src/train/data/test_data.csv"
USE_OPTUNA = True

# --- Forecasting configuration ---
FORECAST_MINUTES = 20   # prediction horizon; change as needed
DROP_CURRENT_LOOP = True  # drop contemporaneous Loop features (use lags instead)
ADD_LOOP_LAGS = True      # create lagged/rolling Loop features from the past only
LAG_MINUTES = sorted({1, 5, 15, FORECAST_MINUTES, 30, 60})
ROLL_MINUTES = sorted({5, 15, FORECAST_MINUTES, 30, 60})
# Target-load lags/rolls (past-only), used to better capture peaks
TARGET_LAG_MINUTES = sorted({FORECAST_MINUTES, 30, 60})
TARGET_ROLL_MINUTES = [60]
ORIG_TCOL = tcol  

# --- Multi-round tuning config ---
MULTI_TUNE_ROUNDS = 1   # run a single tuning round
MULTI_TUNE_ITER = 1     # use a single Optuna trial in that round
OPTIMIZE_METRIC = "MAE" # objective for tuning/improvement printouts

def load_data(file_path):
    df = pd.read_csv(
        file_path,
        engine="python",
        quotechar='"',
        thousands=",",
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

data = load_data(file_path)
data[ORIG_TCOL] = pd.to_numeric(data[ORIG_TCOL], errors="coerce")
data = data.dropna(subset=[ORIG_TCOL, "Date"])
data["hour"] = data["Date"].dt.hour
data["dow"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month
data["is_weekend"] = data["dow"].isin([5, 6]).astype(int)

# cyclical encoding for hour of day
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

print(data.head())
print(len(data))

# --- Forecasting target alignment & lag features (TIME-BASED) ---
# Ensure rows are sorted by time
data = data.sort_values("Date").reset_index(drop=True)

# Create future target by aligning on time (robust to gaps/step changes)
tgt = data[["Date", ORIG_TCOL]].copy()
tgt["Date"] = tgt["Date"] - pd.Timedelta(minutes=FORECAST_MINUTES)
tgt = tgt.rename(columns={ORIG_TCOL: "target_t_plus"})
data = data.merge(tgt, on="Date", how="left")
tcol = "target_t_plus"
# Drop rows where future target is not available
data = data.dropna(subset=[tcol])

# Build time-based lag/rolling features using the timestamp index
loop_cols = [c for c in data.columns if "Loop" in c]
if ADD_LOOP_LAGS:
    _dt_index = data.set_index("Date")
    for c in loop_cols:
        s = _dt_index[c]
        # time-based lags: shift by minutes, then reindex back to the original timestamps
        for m in LAG_MINUTES:
            lag = s.shift(freq=f"{m}T").reindex(s.index)
            data[f"{c}_lag{m}m"] = lag.values
        # time-based rolling means over minutes, using lagged series to avoid leakage
        for m in ROLL_MINUTES:
            roll = s.shift(freq="1T").rolling(f"{m}T", min_periods=2).mean().reindex(s.index)
            data[f"{c}_roll{m}m"] = roll.values

    # ---- Add target-load lags/rolls (use past only; no leakage) ----
    try:
        s_load = data.set_index("Date")[ORIG_TCOL]
        # time-based lags of the original load
        for m in TARGET_LAG_MINUTES:
            lag = s_load.shift(freq=f"{m}T").reindex(s_load.index)
            data[f"load_lag_{m}m"] = lag.values
        # rolling means over minutes, computed on a 1-min-lagged series to avoid leakage
        for m in TARGET_ROLL_MINUTES:
            roll = s_load.shift(freq="1T").rolling(f"{m}T", min_periods=2).mean().reindex(s_load.index)
            data[f"load_roll_{m}m"] = roll.values
    except Exception as _e:
        print("[Warn] Skipped target-load lags due to:", _e)

if DROP_CURRENT_LOOP:
    print("Dropping contemporaneous loop features:", loop_cols)
    data = data.drop(columns=loop_cols, errors="ignore")

# --- Pre-setup leakage & correlation screening ---
# Compute correlations vs target using all current candidate features (except Date)
feature_candidates = [c for c in data.columns if c not in [tcol, "Date"]]
corrs_pre = data[feature_candidates + [tcol]].corr()[tcol].drop(labels=[tcol])
corrs_abs_pre = corrs_pre.abs().sort_values(ascending=False)

# Drop features that are obviously leaky / too correlated; always drop plant KPIs and original target
to_drop_pre = [col for col, corr in corrs_abs_pre.items() if corr > 0.995]
always_drop = ["Plant - Power (kW)", "Plant - Efficiency (kW/Ton)", ORIG_TCOL]
to_drop_pre = sorted(set(to_drop_pre + [c for c in always_drop if c in data.columns]))
print("Dropping before setup (leakage/high-corr):", to_drop_pre)
if to_drop_pre:
    data = data.drop(columns=to_drop_pre)

# Build ignore_features dynamically for safety (in case the column still exists)
ignore_feats = ["Date"]
for c in always_drop:
    if c in data.columns:
        ignore_feats.append(c)
print("Ignoring features (passed to setup):", ignore_feats)

def run_baseline(sort_metric: str = "MAE"):
    """Run a quick baseline to pick the best model by the chosen metric."""
    # Prefer tree boosters for short horizons (better peak capture)
    if FORECAST_MINUTES <= 60:
        best = compare_models(include=['gbr', 'lightgbm', 'catboost'], sort=sort_metric)
    else:
        best = compare_models(sort=sort_metric)
    print("[Baseline] Best model by", sort_metric, ":", best)
    return best

def tune_with_optuna(base_model, optimize: str = "MAE", n_iter: int = 50, fold: int = 5):
    """Tune a base model using Optuna via PyCaret; keep separate from the main flow."""
    tuned = tune_model(
        base_model,
        optimize=optimize,
        n_iter=n_iter,
        fold=fold,
        search_library="optuna",
        search_algorithm="tpe",
        choose_better=True,
        early_stopping=True,
    )
    print(f"[Optuna] Tuned model on {optimize} with {n_iter} iterations.")
    return tuned

def evaluate_on_holdout(model):
    """Compute quick holdout metrics using the current PyCaret context."""
    preds = predict_model(model)
    y_true = get_config('y_test').reset_index(drop=True)
    y_pred = preds['prediction_label'].reset_index(drop=True)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = float(1 - np.var(y_true - y_pred) / var) if var > 0 else float('nan')
    return mae, rmse, r2

def multi_round_tuning(base_model, rounds=MULTI_TUNE_ROUNDS, n_iter=MULTI_TUNE_ITER, optimize=OPTIMIZE_METRIC):
    """
    Run `tune_model` multiple times, feeding each tuned model into the next round.
    After each round, print the holdout metrics and the improvement vs the previous round.
    Returns (best_model, history_list).
    """
    current = base_model
    history = []
    models = []
    prev_mae = None
    for r in range(1, rounds + 1):
        print(f"[Tune] Round {r}/{rounds} — starting from: {current}")
        tuned = tune_model(
            current,
            optimize=optimize,
            n_iter=n_iter,
            fold=5,
            search_library="optuna",
            search_algorithm="tpe",
            choose_better=True,
            early_stopping=True,
        )
        mae, rmse, r2 = evaluate_on_holdout(tuned)
        delta = None if prev_mae is None else (prev_mae - mae)
        if delta is None:
            print(f"[Tune] Round {r} holdout -> MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
        else:
            print(f"[Tune] Round {r} holdout -> MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  ΔMAE={delta:+.4f}")
        history.append({"round": r, "mae": mae, "rmse": rmse, "r2": r2, "model": str(tuned)})
        models.append(tuned)
        prev_mae = mae
        current = tuned

    # pick best by MAE
    best_idx = int(np.argmin([h["mae"] for h in history]))
    best_model = models[best_idx]

    # save history to CSV for later inspection
    out_hist = Path(file_path).parent / "tuning_history.csv"
    pd.DataFrame(history).to_csv(out_hist, index=False)
    print(f"[Tune] Saved tuning history to {out_hist}")
    return best_model, history

s = setup(
    data=data,
    target=tcol,
    session_id=123,
    train_size=0.8,  # keep an internal holdout
    data_split_shuffle=False,
    fold=5,
    fold_strategy="timeseries",
    ignore_features=ignore_feats,
    remove_multicollinearity=False,
    multicollinearity_threshold=0.95,
    normalize=True,
    transform_target=True,
    transform_target_method='yeo-johnson',
    use_gpu=False,
    n_jobs=-1
)

# Ensure train/holdout have identical model schema
_train_cols = get_config('X').columns
_test_cols = get_config('X_test').columns
print("[Schema] Train cols:", len(_train_cols), "; Holdout cols:", len(_test_cols))
print("[Schema] Extra in holdout (should be 0):", sorted(list(set(_test_cols) - set(_train_cols))))
print("[Schema] Missing in holdout (should be 0):", sorted(list(set(_train_cols) - set(_test_cols))))

# --- Training flow ---
baseline_model = run_baseline(sort_metric="MAE")

if USE_OPTUNA:
    final_candidate, tuning_history = multi_round_tuning(
        baseline_model,
        rounds=MULTI_TUNE_ROUNDS,
        n_iter=MULTI_TUNE_ITER,
        optimize=OPTIMIZE_METRIC,
    )
else:
    final_candidate = baseline_model

final_model = finalize_model(final_candidate)
print("[Finalized]", final_model)

X_cols = get_config('X').columns.tolist()
print("Features used:\n", X_cols)
assert "Plant - Power (kW)" not in X_cols, "Plant - Power (kW) should have been excluded"
if "Plant - Efficiency (kW/Ton)" in data.columns:
    assert "Plant - Efficiency (kW/Ton)" not in X_cols, "Plant - Efficiency (kW/Ton) should have been excluded"
if ORIG_TCOL in data.columns:
    assert ORIG_TCOL not in X_cols, f"{ORIG_TCOL} should have been excluded"
assert tcol not in X_cols, "Target leaked into features!"
corrs = data[X_cols + [tcol]].corr()[tcol].drop(labels=[tcol])
print("Top absolute correlations with target:\n", corrs.abs().sort_values(ascending=False).head(20))
print()

# Evaluate on the holdout split and show a quick summary
holdout_results = predict_model(final_model)
print("Holdout predictions shape:", holdout_results.shape)
print(holdout_results.head())

# --- Predicted vs Actual (Holdout) ---
# Align y_true (holdout target) with predictions
y_holdout = get_config('y_test').reset_index(drop=True)
y_pred_holdout = holdout_results['prediction_label'].reset_index(drop=True)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
ax.scatter(y_holdout, y_pred_holdout, alpha=0.5)
# 45-degree reference line
min_val = float(np.nanmin([y_holdout.min(), y_pred_holdout.min()]))
max_val = float(np.nanmax([y_holdout.max(), y_pred_holdout.max()]))
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Predicted vs Actual (Holdout)')

plots_dir = Path(file_path).parent / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
plot_path_holdout = plots_dir / 'pred_vs_actual_holdout.png'
plt.savefig(plot_path_holdout, dpi=FIG_DPI)
plt.close(fig)
print(f"[Plot] Saved holdout Predicted vs Actual to {plot_path_holdout}")

# --- Load graph (time series) on Holdout ---
# Use X_test index to align back to the original 'data' rows to retrieve timestamps
test_index = get_config('X_test').index
dates_holdout = data.loc[test_index, "Date"].reset_index(drop=True)

fig_ts, ax_ts = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
ax_ts.plot(dates_holdout, y_holdout, label="Actual (t+H)")
ax_ts.plot(dates_holdout, y_pred_holdout, label="Predicted")
ax_ts.set_xlabel("Date/Time")
ax_ts.set_ylabel("Plant - Cooling Load (Ton)")
ax_ts.set_title(f"Load over time (Holdout) — H={FORECAST_MINUTES} min")
ax_ts.legend()

plot_path_ts = plots_dir / 'load_timeseries_holdout.png'
plt.savefig(plot_path_ts, dpi=FIG_DPI)
plt.close(fig_ts)
print(f"[Plot] Saved holdout Load over time to {plot_path_ts}")

# --- External test file (optional): same feature engineering, aligned schema ---
try:
    if test_path and isinstance(test_path, str):
        print("[External Test] Loading:", test_path)
        test_df = load_data(test_path)
        if "Date" not in test_df.columns:
            raise KeyError("External test CSV does not contain 'Date' column.")

        # Add time features
        test_df["hour"] = test_df["Date"].dt.hour
        test_df["dow"] = test_df["Date"].dt.dayofweek
        test_df["month"] = test_df["Date"].dt.month
        test_df["is_weekend"] = test_df["dow"].isin([5, 6]).astype(int)
        test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
        test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)

        # Ensure external data sorted by time
        test_df = test_df.sort_values("Date").reset_index(drop=True)

        # Time-based lag/rolling features on external set
        loop_cols_test = [c for c in test_df.columns if "Loop" in c]
        if ADD_LOOP_LAGS:
            _dt_index_t = test_df.set_index("Date")
            for c in loop_cols_test:
                s = _dt_index_t[c]
                for m in LAG_MINUTES:
                    lag = s.shift(freq=f"{m}T").reindex(s.index)
                    test_df[f"{c}_lag{m}m"] = lag.values
                for m in ROLL_MINUTES:
                    roll = s.shift(freq="1T").rolling(f"{m}T", min_periods=2).mean().reindex(s.index)
                    test_df[f"{c}_roll{m}m"] = roll.values
        if DROP_CURRENT_LOOP:
            test_df = test_df.drop(columns=loop_cols_test, errors="ignore")

        # Build external target by time alignment if the CSV has the original target
        if ORIG_TCOL in test_df.columns:
            tmp = test_df[["Date", ORIG_TCOL]].copy()
            tmp["Date"] = tmp["Date"] - pd.Timedelta(minutes=FORECAST_MINUTES)
            tmp = tmp.rename(columns={ORIG_TCOL: "target_t_plus"})
            test_df = test_df.merge(tmp, on="Date", how="left")

        # Drop always-excluded KPI columns after creating target
        for c in ["Plant - Power (kW)", "Plant - Efficiency (kW/Ton)", ORIG_TCOL]:
            if c in test_df.columns:
                test_df = test_df.drop(columns=[c])

        # Align to training feature schema
        X_cols = get_config('X').columns.tolist()
        X_test_aligned = test_df.reindex(columns=X_cols).copy()

        # Diagnostics: NA rates in aligned external features
        out_dir = Path(test_path).parent / "external_test_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        nan_rate = X_test_aligned.isna().mean().sort_values(ascending=False)
        (out_dir / "external_nan_rate.csv").write_text(nan_rate.to_csv(header=["nan_rate"]))
        print("[External Test] Top NA feature rates:\n", nan_rate.head(10))

        preds_ext = predict_model(final_model, data=X_test_aligned)

        # Compute simple metrics if ground truth is available
        if "target_t_plus" in test_df.columns and test_df["target_t_plus"].notna().any():
            mask = test_df["target_t_plus"].notna()
            y_true = test_df.loc[mask, "target_t_plus"].values
            y_pred = preds_ext.loc[mask, "prediction_label"].values
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            var = float(np.var(y_true))
            r2 = float(1 - np.var(y_true - y_pred) / var) if var > 0 else float('nan')
            print(f"[External Test] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
        else:
            print("[External Test] Ground truth not available in test file; wrote predictions only.")

        # Save predictions next to test file
        out_dir = Path(test_path).parent / "external_test_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "predictions_test.csv"
        out_df = test_df[["Date"]].copy()
        out_df["prediction_label"] = preds_ext["prediction_label"].values
        if "target_t_plus" in test_df.columns:
            out_df["target_t_plus"] = test_df["target_t_plus"].values
        out_df.to_csv(out_csv, index=False)
        print(f"[External Test] Saved predictions to {out_csv}")

        # --- Predicted vs Actual (External Test) ---
        if "target_t_plus" in test_df.columns and test_df["target_t_plus"].notna().any():
            mask_ext = test_df["target_t_plus"].notna()
            y_true_ext = test_df.loc[mask_ext, "target_t_plus"].values
            y_pred_ext = preds_ext.loc[mask_ext, "prediction_label"].values

            fig_ext, ax_ext = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
            ax_ext.scatter(y_true_ext, y_pred_ext, alpha=0.5)
            # 45-degree reference line
            min_val_ext = float(np.nanmin([y_true_ext.min(), y_pred_ext.min()]))
            max_val_ext = float(np.nanmax([y_true_ext.max(), y_pred_ext.max()]))
            ax_ext.plot([min_val_ext, max_val_ext], [min_val_ext, max_val_ext], linestyle='--')
            ax_ext.set_xlabel('Actual')
            ax_ext.set_ylabel('Predicted')
            ax_ext.set_title('Predicted vs Actual (External Test)')

            plot_path_ext = out_dir / 'pred_vs_actual_external.png'
            plt.savefig(plot_path_ext, dpi=FIG_DPI)
            plt.close(fig_ext)
            print(f"[Plot] Saved external Predicted vs Actual to {plot_path_ext}")

            # --- Load graph (time series) on External Test ---
            # If ground truth exists, align dates to non-NA rows; otherwise plot predictions only over time.
            try:
                if "target_t_plus" in test_df.columns and test_df["target_t_plus"].notna().any():
                    # Align to rows with ground truth for fair visual comparison
                    dates_ext = test_df.loc[mask_ext, "Date"].values
                    fig_ext_ts, ax_ext_ts = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
                    ax_ext_ts.plot(dates_ext, y_true_ext, label="Actual (t+H)")
                    ax_ext_ts.plot(dates_ext, y_pred_ext, label="Predicted")
                else:
                    # No ground truth: plot predictions over the entire external time range
                    dates_ext = test_df["Date"].values
                    fig_ext_ts, ax_ext_ts = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
                    ax_ext_ts.plot(dates_ext, preds_ext["prediction_label"].values, label="Predicted")
                ax_ext_ts.set_xlabel("Date/Time")
                ax_ext_ts.set_ylabel("Plant - Cooling Load (Ton)")
                ax_ext_ts.set_title(f"Load over time (External) — H={FORECAST_MINUTES} min")
                ax_ext_ts.legend()
                plot_path_ext_ts = out_dir / 'load_timeseries_external.png'
                plt.savefig(plot_path_ext_ts, dpi=FIG_DPI)
                plt.close(fig_ext_ts)
                print(f"[Plot] Saved external Load over time to {plot_path_ext_ts}")
            except Exception as _e_ts:
                print("[Plot] Skipped external load time series due to:", _e_ts)
except Exception as e:
    print("[External Test] Skipped due to:", e)

# Residuals and error plots (work for any regressor)
plot_model(final_model, plot="residuals")
plot_model(final_model, plot="error")

# Feature importance: fallback to permutation importance if estimator doesn't expose importances/coef
try:
    plot_model(final_model, plot="feature")
except TypeError:
    print("[Info] Native feature importance not available for this estimator. Using permutation importance on holdout.")
    X_holdout = get_config('X_test')
    y_holdout = get_config('y_test')
    perm = permutation_importance(
        final_model,
        X_holdout,
        y_holdout,
        n_repeats=5,
        random_state=123,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
    )
    imp = pd.Series(perm.importances_mean, index=X_holdout.columns).sort_values(ascending=False)
    print("Permutation importance (higher = more important):")
    print(imp.head(20))

# Optional: save the model artifact for reuse
save_model(final_model, "Plant_Cooling_Load_Model")

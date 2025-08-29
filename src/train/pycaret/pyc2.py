import pandas as pd
import numpy as np
from pycaret.regression import *
from sklearn.inspection import permutation_importance

file_path = "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/src/train/data/chilled_water_loop_1.csv"
tcol = "Plant - Cooling Load (Ton)"
USE_OPTUNA = False  # set True to enable Optuna tuning


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
data[tcol] = pd.to_numeric(data[tcol], errors="coerce")
data = data.dropna(subset=[tcol, "Date"])
data["hour"] = data["Date"].dt.hour
data["dow"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month
data["is_weekend"] = data["dow"].isin([5, 6]).astype(int)

# cyclical encoding for hour of day
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

print(data.head())
print(len(data))

def run_baseline(sort_metric: str = "MAE"):
    """Run a quick baseline to pick the best model by the chosen metric."""
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


s = setup(
    data=data,
    target=tcol,
    session_id=123,
    train_size=0.8,
    data_split_shuffle=False,
    fold=5,
    fold_strategy="timeseries",
    ignore_features=["Date"],
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    normalize=True,
    transform_target=True,
    transform_target_method='yeo-johnson',
    use_gpu=False,
    n_jobs=-1
)

# --- Training flow ---
baseline_model = run_baseline(sort_metric="MAE")

if USE_OPTUNA:
    final_candidate = tune_with_optuna(baseline_model, optimize="MAE", n_iter=50, fold=5)
else:
    final_candidate = baseline_model

final_model = finalize_model(final_candidate)
print("[Finalized]", final_model)

best = compare_models(include=[final_model], sort='RMSE')
print("[Re-compare Finalized] Best model by RMSE:", best)
print(best)

X_cols = get_config('X').columns.tolist()
print("Features used:\n", X_cols)
assert tcol not in X_cols, "Target leaked into features!" # Sanity Check
corrs = data[X_cols + [tcol]].corr()[tcol].drop(labels=[tcol])
corrs_abs = corrs.abs().sort_values(ascending=False)
print("Top absolute correlations with target:\n", corrs_abs.head(20))  # any feature with |corr| > 0.995 is suspicious
print()

# Drop columns with suspiciously high correlation to target
to_drop = [col for col, corr in corrs_abs.items() if corr > 0.995] + ["Plant - Power (kW)"]
print("Dropping highly correlated features:", to_drop)
data = data.drop(columns=to_drop)

# Evaluate on the holdout split and show a quick summary
holdout_results = predict_model(final_model)
print("Holdout predictions shape:", holdout_results.shape)
print(holdout_results.head())

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


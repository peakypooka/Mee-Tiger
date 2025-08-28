import pandas as pd
from pycaret.regression import *

file_path = "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/src/train/data/chilled_water_loop_1.csv"
tcol = "Plant - Cooling Load (Ton)"

def load_data(file_path):
    df = pd.read_csv(file_path)
    # parse datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # adjust column name if needed
    return df


data = load_data(file_path)
data[tcol] = pd.to_numeric(data[tcol], errors="coerce")
data = data.dropna(subset=[tcol, "Date"])

# make a daily group label
data["DayGroup"] = data["Date"].dt.date

# --- time-derived features (avoid raw Timestamp in model) ---
import numpy as np
data["hour"] = data["Date"].dt.hour
data["dow"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month
data["is_weekend"] = data["dow"].isin([5, 6]).astype(int)

# cyclical encoding for hour of day
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data = data[(data["hour"] >= 8) & (data["hour"] <= 18)]
data = data[(data["dow"] < 5)]  # keep only weekdays

s = setup(
    data=data,
    target=tcol,
    session_id=123,
    train_size=0.8,
    data_split_shuffle=False,
    fold=5,
    fold_strategy="timeseries",
    ignore_features=["Date","DayGroup","Plant - Power (kW)","Plant - Efficiency (kW/Ton)"],
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    normalize=True,
    transform_target=True,
    transform_target_method='yeo-johnson',
    use_gpu=False,
    n_jobs=-1
)

# get 3 best models
top3 = compare_models(
    sort="RMSE",
    turbo=False,
    n_select=3
)
#Tune top 3 via Optuna
_tuned_models = []
for _m in top3:
    try:
        _tm = tune_model(
            _m,
            optimize='RMSE',
            search_library='optuna',
            n_iter=300,          
            choose_better=True,
            tuner_verbose=True
        )
        _tuned_models.append(_tm)
    except Exception as e:
        print(f"Tuning failed for {_m}: {e}")

#best
best = compare_models(include=_tuned_models, sort='RMSE')
print(best)

X_cols = get_config('X').columns.tolist()
print("Features used:\n", X_cols)
assert tcol not in X_cols, "Target leaked into features!" # Sanity Check
corrs = data[X_cols + [tcol]].corr()[tcol].drop(labels=[tcol])
corrs_abs = corrs.abs().sort_values(ascending=False)
print("Top absolute correlations with target:\n", corrs_abs.head(20))  # any feature with |corr| > 0.995 is suspicious
print()

# Evaluate on holdout explicitly (forward-in-time)
X_test = get_config('X_test')
y_test = get_config('y_test')

df_test = X_test.copy()
df_test[tcol] = y_test.values
preds = predict_model(best, data=df_test)
print(preds[[tcol, 'prediction_label']].head())
print()

# Compute real RMSE on the holdout
rmse = np.sqrt(((preds[tcol] - preds['prediction_label'])**2).mean())
print("Holdout RMSE:", rmse)

plot_model(best, plot='residuals')
plot_model(best, plot='error')
plot_model(best, plot='feature')
#exp.plot_model(best, plot='logistic_regression')  #
#exp.plot_model(best, plot='forecast')  #
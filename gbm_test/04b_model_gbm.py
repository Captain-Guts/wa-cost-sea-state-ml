import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# --- Config ---
Data_Dir = "data/merged"
Out_Dir  = "gbm_test/models_gbm"

Buoy_Stations  = ["46041", "46087", "46211"]
Land_Stations  = ["DESW1", "LAPW1", "WPTW1"]

Buoy_Targets   = ["WVHT", "DPD", "MWD"]
Land_Targets   = ["WSPD", "WDIR", "PRES"]

Buoy_ERA5_Features = ['swh', 'mwp', 'mwd', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_ERA5_Features = ['u10', 'v10', 'sp', 't2m', 'tp']

def train_model(df, target, station, features):
    if target == "MWD" or target == "WDIR":
        df = df.copy()
        df["sin_target"] = np.sin(np.deg2rad(df[target]))
        df["cos_target"] = np.cos(np.deg2rad(df[target]))

        combined = df[features + ["sin_target", "cos_target"]].dropna()
        if len(combined) < 100:
            print(f"  {target}: Skipping — not enough data ({len(combined)} rows)")
            return

        X     = combined[features]
        y_sin = combined["sin_target"]
        y_cos = combined["cos_target"]

        X_train, X_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
            X, y_sin, y_cos, test_size=0.2, random_state=42)

        model_sin = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model_cos = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model_sin.fit(X_train, ys_train)
        model_cos.fit(X_train, yc_train)

        pred_angle = np.rad2deg(np.arctan2(
            model_sin.predict(X_test),
            model_cos.predict(X_test))) % 360
        true_angle = np.rad2deg(np.arctan2(ys_test, yc_test)) % 360

        mae  = mean_absolute_error(true_angle, pred_angle)
        rmse = np.sqrt(mean_squared_error(true_angle, pred_angle))

        joblib.dump((model_sin, model_cos),
                    os.path.join(Out_Dir, f"{station}_{target}_sin_cos.pkl"))
        print(f"  {target}: MAE={mae:.3f}  RMSE={rmse:.3f}")
        print(f"  Saved: {Out_Dir}/{station}_{target}_sin_cos.pkl")

    else:
        combined = df[features + [target]].dropna()
        if len(combined) < 100:
            print(f"  {target}: Skipping — not enough data ({len(combined)} rows)")
            return

        X = combined[features]
        y = combined[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        joblib.dump(model, os.path.join(Out_Dir, f"{station}_{target}.pkl"))
        print(f"  {target}: MAE={mae:.3f}  RMSE={rmse:.3f}")
        print(f"  Saved: {Out_Dir}/{station}_{target}.pkl")

# --- Main ---
print("Starting GBM model training...")
print("-" * 40)

print("\n BUOY STATIONS")
for station in Buoy_Stations:
    print(f"Training: {station}")
    df = pd.read_csv(os.path.join(Data_Dir, f"{station}_merged.csv"), index_col=0, parse_dates=True)
    for target in Buoy_Targets:
        if target in df.columns:
            train_model(df, target, station, Buoy_ERA5_Features)

print("\n LAND STATIONS")
for station in Land_Stations:
    print(f"Training: {station}")
    df = pd.read_csv(os.path.join(Data_Dir, f"{station}_merged.csv"), index_col=0, parse_dates=True)
    for target in Land_Targets:
        if target in df.columns:
            train_model(df, target, station, Land_ERA5_Features)

print("\nDone! All GBM models saved.")
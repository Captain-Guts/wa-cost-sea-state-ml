import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

Merged_DIR = 'data/merged'
Model_DIR = 'models'
os.makedirs(Model_DIR, exist_ok=True)

Buoy_stations = ['46041', '46087', '46211']
Land_stations = ['DESW1', 'LAPW1', 'WPTW1']

Buoy_features = ['swh', 'mwp', 'mwd_sin', 'mwd_cos', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_features = ['u10', 'v10', 'sp', 't2m', 'tp']

Buoy_targets = ['WVHT', 'DPD', 'MWD']
Land_targets = ['WSPD', 'WDIR', 'PRES']

def load_station(station, targets, features):
    path = os.path.join(Merged_DIR, f"{station}_merged.csv")
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    raw_features = [f for f in features if f not in ['mwd_sin', 'mwd_cos']]
    df = df.dropna(subset=raw_features)
    df['mwd_sin'] = np.sin(np.radians(df['mwd']))
    df['mwd_cos'] = np.cos(np.radians(df['mwd']))
    X = df[features]
    y = df[targets]
    print(f"  {station}: {len(df)} rows after dropping NaN")
    return X, y

def train_station(station, targets, features):
    print(f"\nTraining: {station}")
    X, y = load_station(station, targets, features)
    for target in targets:
        if target in ['MWD', 'WDIR']:
            mask = y[target].notna()
            X_t = X[mask]
            y_sin = np.sin(np.radians(y[target][mask]))
            y_cos = np.cos(np.radians(y[target][mask]))

            if len(X_t) < 100:
                print(f"  Skipping {target} — not enough data ({len(X_t)} rows)")
                continue

            X_train, X_test, y_sin_train, y_sin_test = train_test_split(X_t, y_sin, test_size=0.2, shuffle=False)
            _, _, y_cos_train, y_cos_test = train_test_split(X_t, y_cos, test_size=0.2, shuffle=False)

            model_sin = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_cos = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_sin.fit(X_train, y_sin_train)
            model_cos.fit(X_train, y_cos_train)

            pred_sin = model_sin.predict(X_test)
            pred_cos = model_cos.predict(X_test)
            y_pred = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
            y_actual = np.degrees(np.arctan2(y_sin_test, y_cos_test)) % 360

            diff = np.abs(y_pred - y_actual)
            diff = np.minimum(diff, 360 - diff)
            mae = diff.mean()
            rmse = np.sqrt((diff**2).mean())
            print(f"  {target}: MAE={mae:.3f}  RMSE={rmse:.3f}")

            joblib.dump(model_sin, os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
            joblib.dump(model_cos, os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))
            print(f"  Saved: models/{station}_{target}_sin/cos.pkl")

        else:
            mask = y[target].notna()
            X_t = X[mask]
            y_t = y[target][mask]

            if len(X_t) < 100:
                print(f"  Skipping {target} — not enough data ({len(X_t)} rows)")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, shuffle=False)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"  {target}: MAE={mae:.3f}  RMSE={rmse:.3f}")
            model_path = os.path.join(Model_DIR, f"{station}_{target}.pkl")
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path}")

if __name__ == '__main__':
    print("Starting model training...")
    print("-" * 40)
    print("\n BUOY STATION")
    for station in Buoy_stations:
        train_station(station, Buoy_targets, Buoy_features)
    print("\n LAND STATION")
    for station in Land_stations:
        train_station(station, Land_targets, Land_features)
    print("\nDone! All models saved to:", Model_DIR)
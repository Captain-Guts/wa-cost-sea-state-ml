import os
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import joblib

# --- Directories ---
Base_DIR = os.path.dirname(os.path.abspath(__file__))
Model_DIR = os.path.join(Base_DIR, 'models')
Output_DIR = os.path.join(Base_DIR, 'predict_output')
os.makedirs(Output_DIR, exist_ok=True)

# --- Stations ---
Buoy_stations = ['46041', '46087', '46211']
Land_stations  = ['DESW1', 'LAPW1', 'WPTW1']

Station_coords = {
    '46041': (47.353, -124.731),
    '46087': (48.494, -124.728),
    '46211': (47.116, -124.184),
    'DESW1': (47.677, -124.499),
    'LAPW1': (47.913, -124.637),
    'WPTW1': (46.904, -124.105),
}

Buoy_features = ['swh', 'mwp', 'mwd_sin', 'mwd_cos', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_features  = ['u10', 'v10', 'sp', 't2m', 'tp']

Buoy_targets = ['WVHT', 'DPD', 'MWD']
Land_targets  = ['WSPD', 'WDIR', 'PRES']
def fetch_forecast():
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    date_str = tomorrow.strftime('%Y-%m-%d')
    print(f"\n=== Washington Coast Sea State Forecast ===")
    print(f"Generating forecast for: {date_str}")
    print("-" * 40)

    station_data = {}

    for station, (lat, lon) in Station_coords.items():
        # Weather variables (u10, v10, sp, t2m, tp)
        weather = om.weather_api("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m", "wind_direction_10m",
                       "surface_pressure", "temperature_2m", "precipitation"],
            "wind_speed_unit": "ms",
            "start_date": date_str, "end_date": date_str,
            "timezone": "UTC"
        })[0].Hourly()

        times = pd.date_range(date_str, periods=24, freq='h', tz='UTC').tz_localize(None)
        df = pd.DataFrame(index=times)

        # Convert wind speed + direction → u10, v10 components
        wspd = weather.Variables(0).ValuesAsNumpy()
        wdir = weather.Variables(1).ValuesAsNumpy()
        df['u10'] = -wspd * np.sin(np.radians(wdir))
        df['v10'] = -wspd * np.cos(np.radians(wdir))
        df['sp']  = weather.Variables(2).ValuesAsNumpy() * 100  # hPa → Pa
        df['t2m'] = weather.Variables(3).ValuesAsNumpy() + 273.15  # °C → K
        df['tp']  = weather.Variables(4).ValuesAsNumpy() / 1000    # mm → m

        # Wave variables (swh, mwp, mwd) — buoy stations only
        if station in Buoy_stations:
            marine = om.weather_api("https://marine-api.open-meteo.com/v1/marine", params={
                "latitude": lat, "longitude": lon,
                "hourly": ["wave_height", "wave_period", "wave_direction"],
                "start_date": date_str, "end_date": date_str,
                "timezone": "UTC"
            })[0].Hourly()

            df['swh'] = marine.Variables(0).ValuesAsNumpy()
            df['mwp'] = marine.Variables(1).ValuesAsNumpy()
            mwd       = marine.Variables(2).ValuesAsNumpy()
            df['mwd_sin'] = np.sin(np.radians(mwd))
            df['mwd_cos'] = np.cos(np.radians(mwd))

        df = df.ffill().bfill()
        station_data[station] = df
        print(f"  {station}: fetched {len(df)} hours")

    return station_data, tomorrow
def run_predictions(station_data):
    print("\nRunning model predictions...")
    all_predictions = {}

    for station in Buoy_stations:
        df = station_data[station]
        X = df[Buoy_features]
        preds = {}

        for target in Buoy_targets:
            if target == 'MWD':
                model_sin = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
                model_cos = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))
                preds[target] = np.degrees(np.arctan2(
                    model_sin.predict(X), model_cos.predict(X))) % 360
            else:
                model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
                preds[target] = model.predict(X)

        all_predictions[station] = pd.DataFrame(preds, index=df.index)
        print(f"  {station}: done")

    for station in Land_stations:
        df = station_data[station]
        X = df[Land_features]
        preds = {}

        for target in Land_targets:
            if target == 'WDIR':
                model_sin = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
                model_cos = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))
                preds[target] = np.degrees(np.arctan2(
                    model_sin.predict(X), model_cos.predict(X))) % 360
            else:
                model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
                preds[target] = model.predict(X)

        all_predictions[station] = pd.DataFrame(preds, index=df.index)
        print(f"  {station}: done")

    return all_predictions


def print_forecast(all_predictions, tomorrow):
    date_str = tomorrow.strftime('%Y-%m-%d')

    print(f"\n{'='*55}")
    print(f"  BUOY STATION FORECAST — {date_str} (UTC)")
    print(f"{'='*55}")

    for station in Buoy_stations:
        pred = all_predictions[station]
        print(f"\n  Station {station}")
        print(f"  {'Hour':<6} {'WVHT(m)':<10} {'DPD(s)':<10} {'MWD(°)':<10} {'Warning'}")
        print(f"  {'-'*50}")
        for hour, row in pred.iterrows():
            h = hour.strftime('%H:00')
            warn = ''
            if row['WVHT'] >= 4.0:
                warn = '⚠ DANGEROUS'
            elif row['WVHT'] >= 2.5:
                warn = '! Rough'
            print(f"  {h:<6} {row['WVHT']:<10.2f} {row['DPD']:<10.1f} {row['MWD']:<10.0f} {warn}")

    print(f"\n{'='*55}")
    print(f"  LAND STATION FORECAST — {date_str} (UTC)")
    print(f"{'='*55}")

    for station in Land_stations:
        pred = all_predictions[station]
        print(f"\n  Station {station}")
        print(f"  {'Hour':<6} {'WSPD(m/s)':<12} {'WDIR(°)':<10} {'PRES(hPa)'}")
        print(f"  {'-'*45}")
        for hour, row in pred.iterrows():
            h = hour.strftime('%H:00')
            print(f"  {h:<6} {row['WSPD']:<12.1f} {row['WDIR']:<10.0f} {row['PRES']:.1f}")


def save_plots(all_predictions, tomorrow):
    date_str = tomorrow.strftime('%Y-%m-%d')
    print(f"\nSaving forecast plots to {Output_DIR}/")

    for station in Buoy_stations:
        pred = all_predictions[station]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{station} — Forecast for {date_str}', fontsize=13, fontweight='bold')

        for ax, target in zip(axes, Buoy_targets):
            ax.plot(pred.index, pred[target], color='tomato', linewidth=2)
            if target == 'WVHT':
                ax.axhline(2.5, color='orange', linestyle='--', alpha=0.7, label='Rough (2.5m)')
                ax.axhline(4.0, color='red',    linestyle='--', alpha=0.7, label='Dangerous (4.0m)')
                ax.legend(fontsize=8)
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(Output_DIR, f"{station}_forecast_{date_str}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")
if __name__ == '__main__':
    station_data, tomorrow = fetch_forecast()
    all_predictions = run_predictions(station_data)
    print_forecast(all_predictions, tomorrow)
    save_plots(all_predictions, tomorrow)
    print("\nDone! Check predict_output/ for plots.")
    
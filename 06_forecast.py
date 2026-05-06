import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

Base_DIR   = os.path.dirname(os.path.abspath(__file__))
Model_DIR  = os.path.join(Base_DIR, 'models_gbm')
Output_DIR = os.path.join(Base_DIR, 'forecast_output')
os.makedirs(Output_DIR, exist_ok=True)

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

Buoy_features = ['swh', 'mwp', 'mwd', 'u10', 'v10', 'sp', 't2m', 'tp']
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

    today    = pd.Timestamp.now().normalize()
    day1_str = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    day2_str = (today + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

    print(f"\n=== Washington Coast Sea State Forecast (GBM) ===")
    print(f"Generating 48-hour forecast: {day1_str} — {day2_str}")
    print("-" * 50)

    station_data = {}

    for station, (lat, lon) in Station_coords.items():
        weather = om.weather_api("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m", "wind_direction_10m",
                       "surface_pressure", "temperature_2m", "precipitation"],
            "wind_speed_unit": "ms",
            "start_date": day1_str, "end_date": day2_str,
            "timezone": "UTC"
        })[0].Hourly()

        times = pd.date_range(day1_str, periods=48, freq='h', tz='UTC').tz_localize(None)
        df = pd.DataFrame(index=times)

        wspd = weather.Variables(0).ValuesAsNumpy()
        wdir = weather.Variables(1).ValuesAsNumpy()
        df['u10'] = -wspd * np.sin(np.radians(wdir))
        df['v10'] = -wspd * np.cos(np.radians(wdir))
        df['sp']  = weather.Variables(2).ValuesAsNumpy() * 100    # hPa → Pa
        df['t2m'] = weather.Variables(3).ValuesAsNumpy() + 273.15  # °C → K
        df['tp']  = weather.Variables(4).ValuesAsNumpy() / 1000    # mm → m

        if station in Buoy_stations:
            marine = om.weather_api("https://marine-api.open-meteo.com/v1/marine", params={
                "latitude": lat, "longitude": lon,
                "hourly": ["wave_height", "wave_period", "wave_direction"],
                "start_date": day1_str, "end_date": day2_str,
                "timezone": "UTC"
            })[0].Hourly()

            df['swh'] = marine.Variables(0).ValuesAsNumpy()
            df['mwp'] = marine.Variables(1).ValuesAsNumpy()
            df['mwd'] = marine.Variables(2).ValuesAsNumpy()

        df = df.ffill().bfill()
        station_data[station] = df
        print(f"  {station}: fetched {len(df)} hours")

    return station_data, day1_str, day2_str


def run_predictions(station_data):
    print("\nRunning GBM predictions...")
    all_predictions = {}

    for station in Buoy_stations:
        df = station_data[station]
        X  = df[Buoy_features]
        preds = {}

        for target in Buoy_targets:
            if target == 'MWD':
                model_sin, model_cos = joblib.load(
                    os.path.join(Model_DIR, f"{station}_{target}_sin_cos.pkl"))
                preds[target] = np.degrees(np.arctan2(
                    model_sin.predict(X), model_cos.predict(X))) % 360
            else:
                model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
                preds[target] = model.predict(X)

        all_predictions[station] = pd.DataFrame(preds, index=df.index)
        print(f"  {station}: done")

    for station in Land_stations:
        df = station_data[station]
        X  = df[Land_features]
        preds = {}

        for target in Land_targets:
            if target == 'WDIR':
                model_path = os.path.join(Model_DIR, f"{station}_{target}_sin_cos.pkl")
                if not os.path.exists(model_path):
                    print(f"    {station} {target}: no model found, skipping")
                    continue
                model_sin, model_cos = joblib.load(model_path)
                preds[target] = np.degrees(np.arctan2(
                    model_sin.predict(X), model_cos.predict(X))) % 360
            else:
                model_path = os.path.join(Model_DIR, f"{station}_{target}.pkl")
                if not os.path.exists(model_path):
                    print(f"    {station} {target}: no model found, skipping")
                    continue
                model = joblib.load(model_path)
                preds[target] = model.predict(X)

        all_predictions[station] = pd.DataFrame(preds, index=df.index)
        print(f"  {station}: done")

    return all_predictions


def print_forecast(all_predictions, day1_str, day2_str):
    for label, date_str in [("Day 1", day1_str), ("Day 2", day2_str)]:
        date = pd.Timestamp(date_str).date()

        print(f"\n{'='*55}")
        print(f"  BUOY FORECAST — {label} ({date_str} UTC)")
        print(f"{'='*55}")

        for station in Buoy_stations:
            pred     = all_predictions[station]
            day_pred = pred[pred.index.date == date]
            print(f"\n  Station {station}")
            print(f"  {'Hour':<6} {'WVHT(m)':<10} {'DPD(s)':<10} {'MWD(°)':<10} {'Warning'}")
            print(f"  {'-'*50}")
            for hour, row in day_pred.iterrows():
                h    = hour.strftime('%H:00')
                warn = ''
                if row['WVHT'] >= 4.0:
                    warn = '⚠ DANGEROUS'
                elif row['WVHT'] >= 2.5:
                    warn = '! Rough'
                print(f"  {h:<6} {row['WVHT']:<10.2f} {row['DPD']:<10.1f} {row['MWD']:<10.0f} {warn}")

        print(f"\n{'='*55}")
        print(f"  LAND FORECAST — {label} ({date_str} UTC)")
        print(f"{'='*55}")

        for station in Land_stations:
            pred     = all_predictions[station]
            day_pred = pred[pred.index.date == date]
            print(f"\n  Station {station}")
            print(f"  {'Hour':<6} {'WSPD(m/s)':<12} {'WDIR(°)':<10} {'PRES(hPa)'}")
            print(f"  {'-'*45}")
            for hour, row in day_pred.iterrows():
                h    = hour.strftime('%H:00')
                wspd = row.get('WSPD', float('nan'))
                wdir = row.get('WDIR', float('nan'))
                pres = row.get('PRES', float('nan'))
                print(f"  {h:<6} {wspd:<12.1f} {wdir:<10.0f} {pres:.1f}")


def save_plots(all_predictions, day1_str, day2_str):
    print(f"\nSaving forecast plots to {Output_DIR}/")
    boundary = pd.Timestamp(day2_str)

    for station in Buoy_stations:
        pred = all_predictions[station]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'{station} — 48h GBM Forecast ({day1_str} — {day2_str})',
                     fontsize=13, fontweight='bold')

        for ax, target in zip(axes, Buoy_targets):
            ax.plot(pred.index, pred[target], color='tomato', linewidth=2)
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.6, label='Day boundary')
            if target == 'WVHT':
                ax.axhline(2.5, color='orange', linestyle='--', alpha=0.7, label='Rough (2.5m)')
                ax.axhline(4.0, color='red',    linestyle='--', alpha=0.7, label='Dangerous (4.0m)')
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(Output_DIR, f"{station}_forecast_{day1_str}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")


if __name__ == '__main__':
    station_data, day1_str, day2_str = fetch_forecast()
    all_predictions = run_predictions(station_data)
    print_forecast(all_predictions, day1_str, day2_str)
    save_plots(all_predictions, day1_str, day2_str)
    print("\nDone! Check forecast_output/ for plots.")

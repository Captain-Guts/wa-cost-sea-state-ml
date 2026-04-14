import os
import requests
import gzip
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import cdsapi

# Directories
Model_DIR = '../models'
Output_DIR = 'outputs'
Data_DIR = 'data'
os.makedirs(Output_DIR, exist_ok=True)
os.makedirs(Data_DIR, exist_ok=True)

# Stations
Buoy_stations = ['46041', '46087', '46211']
Land_stations = ['DESW1', 'LAPW1', 'WPTW1']

Buoy_features = ['swh', 'mwp', 'mwd_sin', 'mwd_cos', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_features = ['u10', 'v10', 'sp', 't2m', 'tp']

Buoy_targets = ['WVHT', 'DPD', 'MWD']
Land_targets = ['WSPD', 'WDIR', 'PRES']

# Forecast window — November 19-21 2024 storm
Forecast_year = '2024'
Forecast_months = ['11']
Forecast_days = [f'{d:02d}' for d in range(19, 22)]
Forecast_hours = [f'{h:02d}:00' for h in range(24)]

# Station coordinates
Station_coords = {
    '46041': (47.353, -124.731),
    '46087': (48.494, -124.728),
    '46211': (47.116, -124.184),
    'DESW1': (47.677, -124.499),
    'LAPW1': (47.913, -124.637),
    'WPTW1': (46.904, -124.105)
}

def extract_era5():
    import xarray as xr
    print("Extracting ERA5 at station locations...")

    ds_oper = xr.open_dataset('data/era5_unzipped/data_stream-oper_stepType-instant.nc', engine='netcdf4')
    ds_accum = xr.open_dataset('data/era5_unzipped/data_stream-oper_stepType-accum.nc', engine='netcdf4')
    ds_wave = xr.open_dataset('data/era5_unzipped/data_stream-wave_stepType-instant.nc', engine='netcdf4')

    station_era5 = {}
    for station, (lat, lon) in Station_coords.items():
        df = pd.DataFrame(index=pd.to_datetime(ds_oper.valid_time.values))

        pt = ds_oper.sel(latitude=lat, longitude=lon, method='nearest')
        for var in ['u10', 'v10', 'sp', 't2m']:
            df[var] = pt[var].values

        pt_acc = ds_accum.sel(latitude=lat, longitude=lon, method='nearest')
        df['tp'] = pt_acc['tp'].values

        # Wave variables — find nearest valid ocean point
        for var in ['swh', 'mwp', 'mwd']:
            point_data = ds_wave[var].sel(latitude=lat, longitude=lon, method='nearest').values
            if np.all(np.isnan(point_data)):
                for dlat, dlon in [(0, -0.5), (0.5, -0.5), (0, -1.0), (0.5, 0)]:
                    alt = ds_wave[var].sel(latitude=lat+dlat, longitude=lon+dlon, method='nearest').values
                    if not np.all(np.isnan(alt)):
                        point_data = alt
                        break
            df[var] = point_data

        df['mwd_sin'] = np.sin(np.radians(df['mwd']))
        df['mwd_cos'] = np.cos(np.radians(df['mwd']))
        df.index.name = 'time'
        station_era5[station] = df
        print(f"  {station}: {len(df)} timesteps extracted")

    return station_era5

def run_predictions(station_era5):
    print("\nRunning predictions...")
    all_predictions = {}

    for station in Buoy_stations:
        print(f"\n  {station}")
        df = station_era5[station]
        X = df[Buoy_features].ffill().bfill()
        preds = {}

        for target in Buoy_targets:
            if target == 'MWD':
                model_sin = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
                model_cos = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))
                pred_sin = model_sin.predict(X)
                pred_cos = model_cos.predict(X)
                preds[target] = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
            else:
                model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
                preds[target] = model.predict(X)
            print(f"    {target}: predicted {len(preds[target])} timesteps")

        all_predictions[station] = pd.DataFrame(preds, index=df.index)

    for station in Land_stations:
        print(f"\n  {station}")
        df = station_era5[station]
        X = df[Land_features].ffill().bfill()
        preds = {}

        for target in Land_targets:
            if target == 'WDIR':
                model_sin = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
                model_cos = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))
                pred_sin = model_sin.predict(X)
                pred_cos = model_cos.predict(X)
                preds[target] = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
            else:
                model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
                preds[target] = model.predict(X)
            print(f"    {target}: predicted {len(preds[target])} timesteps")

        all_predictions[station] = pd.DataFrame(preds, index=df.index)

    return all_predictions

def download_ndbc(station):
    url = f'https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h2024.txt.gz'
    r = requests.get(url)
    if r.status_code != 200:
        print(f"  {station}: failed to download (status {r.status_code})")
        return None

    with gzip.open(io.BytesIO(r.content)) as f:
        df = pd.read_csv(f, sep='\s+', skiprows=[1], na_values=[99, 999, 9999, 99.0, 999.0])

    df.columns = [c.strip('#') for c in df.columns]
    df['time'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
        columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
    df = df.set_index('time')
    df = df[(df.index >= '2024-11-19') & (df.index <= '2024-11-21 23:59')]
    print(f"  {station}: {len(df)} observations in storm window")
    return df

def download_all_ndbc():
    print("\nDownloading NDBC observations for Nov 19-21 2024...")
    observations = {}
    for station in Buoy_stations + Land_stations:
        obs = download_ndbc(station)
        if obs is not None:
            observations[station] = obs
    return observations

def plot_hindcast(all_predictions, observations):
    print("\nPlotting hindcast comparison...")

    for station in Buoy_stations:
        if station not in observations:
            print(f"  {station}: no observations available, skipping")
            continue

        pred = all_predictions[station]
        obs = observations[station]
        n = len(Buoy_targets)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        fig.suptitle(f'{station} — Hindcast vs Observed (Nov 19-21 2024 Storm)', fontsize=13, fontweight='bold')

        for ax, target in zip(axes, Buoy_targets):
            if target not in obs.columns:
                continue
            obs_series = obs[target].dropna()
            pred_series = pred[target]
            ax.plot(obs_series.index, obs_series.values, label='Observed', color='steelblue', linewidth=1.5)
            ax.plot(pred_series.index, pred_series.values, label='Predicted', color='tomato', linewidth=1.5, alpha=0.85)
            ax.axvline(pd.Timestamp('2024-11-20 17:40'), color='gray', linestyle='--', alpha=0.5, label='Storm peak')
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(Output_DIR, f"{station}_hindcast.png"), dpi=150)
        plt.show()
        print(f"  Saved: {Output_DIR}/{station}_hindcast.png")

    for station in Land_stations:
        if station not in observations:
            print(f"  {station}: no observations available, skipping")
            continue

        pred = all_predictions[station]
        obs = observations[station]
        n = len(Land_targets)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        fig.suptitle(f'{station} — Hindcast vs Observed (Nov 19-21 2024 Storm)', fontsize=13, fontweight='bold')

        for ax, target in zip(axes, Land_targets):
            if target not in obs.columns:
                continue
            obs_series = obs[target].dropna()
            pred_series = pred[target]
            ax.plot(obs_series.index, obs_series.values, label='Observed', color='steelblue', linewidth=1.5)
            ax.plot(pred_series.index, pred_series.values, label='Predicted', color='tomato', linewidth=1.5, alpha=0.85)
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(Output_DIR, f"{station}_hindcast.png"), dpi=150)
        plt.show()
        print(f"  Saved: {Output_DIR}/{station}_hindcast.png")

if __name__ == '__main__':
    print("Starting hindcast for Nov 19-21 2024 storm...")
    print("-" * 40)

    station_era5 = extract_era5()
    all_predictions = run_predictions(station_era5)
    observations = download_all_ndbc()
    plot_hindcast(all_predictions, observations)

    print("\nDone! All hindcast plots saved to:", Output_DIR)
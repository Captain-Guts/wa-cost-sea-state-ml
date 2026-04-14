import os
import requests
import gzip
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Directories
Model_DIR = 'gbm_test/models_gbm'
Output_DIR = 'gbm_test/outputs'
Data_DIR = 'forecast_test/data'
os.makedirs(Output_DIR, exist_ok=True)

# Stations
Buoy_stations = ['46041', '46087', '46211']
Land_stations  = ['DESW1', 'LAPW1', 'WPTW1']

Buoy_features = ['swh', 'mwp', 'mwd', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_features = ['u10', 'v10', 'sp', 't2m', 'tp']

Buoy_targets = ['WVHT', 'DPD', 'MWD']
Land_targets  = ['WSPD', 'WDIR', 'PRES']

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

    ds_oper  = xr.open_dataset(os.path.join(Data_DIR, 'era5_unzipped/data_stream-oper_stepType-instant.nc'), engine='netcdf4')
    ds_accum = xr.open_dataset(os.path.join(Data_DIR, 'era5_unzipped/data_stream-oper_stepType-accum.nc'),   engine='netcdf4')
    ds_wave  = xr.open_dataset(os.path.join(Data_DIR, 'era5_unzipped/data_stream-wave_stepType-instant.nc'), engine='netcdf4')

    station_era5 = {}
    for station, (lat, lon) in Station_coords.items():
        df = pd.DataFrame(index=pd.to_datetime(ds_oper.valid_time.values))

        pt = ds_oper.sel(latitude=lat, longitude=lon, method='nearest')
        for var in ['u10', 'v10', 'sp', 't2m']:
            df[var] = pt[var].values

        pt_acc = ds_accum.sel(latitude=lat, longitude=lon, method='nearest')
        df['tp'] = pt_acc['tp'].values

        for var in ['swh', 'mwp', 'mwd']:
            point_data = ds_wave[var].sel(latitude=lat, longitude=lon, method='nearest').values
            if np.all(np.isnan(point_data)):
                for dlat, dlon in [(0, -0.5), (0.5, -0.5), (0, -1.0), (0.5, 0)]:
                    alt = ds_wave[var].sel(latitude=lat+dlat, longitude=lon+dlon, method='nearest').values
                    if not np.all(np.isnan(alt)):
                        point_data = alt
                        break
            df[var] = point_data

        df.index.name = 'time'
        station_era5[station] = df
        print(f"  {station}: {len(df)} timesteps extracted")

    return station_era5

def run_predictions(station_era5):
    print("\nRunning GBM predictions...")
    all_predictions = {}

    for station in Buoy_stations:
        print(f"\n  {station}")
        df = station_era5[station]
        X = df[Buoy_features].ffill().bfill()
        preds = {}

        for target in Buoy_targets:
            model_path = os.path.join(Model_DIR, f"{station}_{target}_sin_cos.pkl")
            if target == 'MWD':
                model_sin, model_cos = joblib.load(model_path)
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
                model_path = os.path.join(Model_DIR, f"{station}_{target}_sin_cos.pkl")
                if not os.path.exists(model_path):
                    print(f"    {target}: no model found, skipping")
                    continue
                model_sin, model_cos = joblib.load(model_path)
                pred_sin = model_sin.predict(X)
                pred_cos = model_cos.predict(X)
                preds[target] = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
            else:
                model_path = os.path.join(Model_DIR, f"{station}_{target}.pkl")
                if not os.path.exists(model_path):
                    print(f"    {target}: no model found, skipping")
                    continue
                model = joblib.load(model_path)
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
            print(f"  {station}: no observations, skipping")
            continue

        pred = all_predictions[station]
        obs  = observations[station]
        fig, axes = plt.subplots(len(Buoy_targets), 1, figsize=(14, 4 * len(Buoy_targets)))
        fig.suptitle(f'{station} — GBM Hindcast vs Observed (Nov 19-21 2024)', fontsize=13, fontweight='bold')

        for ax, target in zip(axes, Buoy_targets):
            if target not in obs.columns:
                continue
            ax.plot(obs[target].dropna().index,  obs[target].dropna().values,  label='Observed',  color='steelblue', linewidth=1.5)
            ax.plot(pred[target].index, pred[target].values, label='GBM Predicted', color='tomato',    linewidth=1.5, alpha=0.85)
            ax.axvline(pd.Timestamp('2024-11-20 17:40'), color='gray', linestyle='--', alpha=0.5, label='Storm peak')
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(Output_DIR, f"{station}_gbm_hindcast.png"), dpi=150)
        plt.close()
        print(f"  Saved: {Output_DIR}/{station}_gbm_hindcast.png")

    for station in Land_stations:
        if station not in observations:
            print(f"  {station}: no observations, skipping")
            continue

        pred = all_predictions[station]
        obs  = observations[station]
        targets = [t for t in Land_targets if t in pred.columns]
        fig, axes = plt.subplots(len(targets), 1, figsize=(14, 4 * len(targets)))
        if len(targets) == 1:
            axes = [axes]
        fig.suptitle(f'{station} — GBM Hindcast vs Observed (Nov 19-21 2024)', fontsize=13, fontweight='bold')

        for ax, target in zip(axes, targets):
            if target not in obs.columns:
                continue
            ax.plot(obs[target].dropna().index,  obs[target].dropna().values,  label='Observed',  color='steelblue', linewidth=1.5)
            ax.plot(pred[target].index, pred[target].values, label='GBM Predicted', color='tomato',    linewidth=1.5, alpha=0.85)
            ax.set_ylabel(target)
            ax.set_title(target)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(Output_DIR, f"{station}_gbm_hindcast.png"), dpi=150)
        plt.close()
        print(f"  Saved: {Output_DIR}/{station}_gbm_hindcast.png")

if __name__ == '__main__':
    print("Starting GBM hindcast for Nov 19-21 2024 storm...")
    print("-" * 40)

    station_era5    = extract_era5()
    all_predictions = run_predictions(station_era5)
    observations    = download_all_ndbc()
    plot_hindcast(all_predictions, observations)

    print("\nDone! All GBM hindcast plots saved to:", Output_DIR)
import os
import zipfile
import xarray as xr
import pandas as pd
import numpy as np

ERA5_DIR = "data/wrf"
EXTRACT_DIR = "data/era5_extracted"
os.makedirs(EXTRACT_DIR, exist_ok=True)

zip_files = sorted([f for f in os.listdir(ERA5_DIR) if f.endswith(".nc")])
print(f"Found {len(zip_files)} zip files to extract")

for fname in zip_files:
    zip_path = os.path.join(ERA5_DIR, fname)
    folder_name = fname.replace(".nc", "")
    out_dir = os.path.join(EXTRACT_DIR, folder_name)
    
    if os.path.exists(out_dir):
        print(f"  Already extracted: {folder_name}, skipping.")
        continue
    
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    print(f"  Extracted: {folder_name}")

print("\nDone! All files extracted to:", EXTRACT_DIR)

STATIONS = {
    "46041": (47.352, -124.739),
    "46211": (46.857, -124.244),
    "46087": (48.493, -124.727),
    "LAPW1": (47.913, -124.637),
    "WPTW1": (46.904, -124.105),
    "DESW1": (47.675, -124.485),
}

WAVE_FILE = "data_stream-wave_stepType-instant.nc"
OPER_FILE = "data_stream-oper_stepType-instant.nc"
ACCUM_FILE = "data_stream-oper_stepType-accum.nc"

wave_list, oper_list, accum_list = [], [], []

for folder in sorted(os.listdir(EXTRACT_DIR)):
    folder_path = os.path.join(EXTRACT_DIR, folder)
    wave_list.append(xr.open_dataset(os.path.join(folder_path, WAVE_FILE)))
    oper_list.append(xr.open_dataset(os.path.join(folder_path, OPER_FILE)))
    accum_list.append(xr.open_dataset(os.path.join(folder_path, ACCUM_FILE)))

print("Concatenating all months...")
ds_wave = xr.concat(wave_list, dim="valid_time")
ds_oper = xr.concat(oper_list, dim="valid_time")
ds_accum = xr.concat(accum_list, dim="valid_time")
print(f"  Wave:  {ds_wave.dims}")
print(f"  Oper:  {ds_oper.dims}")
print(f"  Accum: {ds_accum.dims}")

os.makedirs("data/era5_stations", exist_ok=True)

WAVE_COORD_OVERRIDE = {
    "46211": (47.0, -124.5),  # nearest valid offshore wave grid point
}

for station, (lat, lon) in STATIONS.items():
    print(f"\nProcessing {station} ({lat}, {lon})")
    
    wave_lat, wave_lon = WAVE_COORD_OVERRIDE.get(station, (lat, lon))
    wave_point = ds_wave.sel(latitude=wave_lat, longitude=wave_lon, method="nearest")
    oper_point = ds_oper.sel(latitude=lat, longitude=lon, method="nearest")
    accum_point = ds_accum.sel(latitude=lat, longitude=lon, method="nearest")

    df_wave = wave_point[["swh", "mwp", "mwd"]].to_dataframe().drop(columns=["latitude", "longitude", "number", "expver"], errors="ignore")
    df_oper = oper_point[["u10", "v10", "sp", "t2m"]].to_dataframe().drop(columns=["latitude", "longitude", "number", "expver"], errors="ignore")
    df_accum = accum_point[["tp"]].to_dataframe().drop(columns=["latitude", "longitude", "number", "expver"], errors="ignore")
    
    # Merge all into one DataFrame
    df = df_wave.join(df_oper).join(df_accum)
    df.index.name = "time"
    
    out_path = f"data/era5_stations/{station}_era5.csv"
    df.to_csv(out_path)
    print(f"  Saved: {out_path} — shape: {df.shape}")

print("\nAll stations done!")
NDBC_DIR = "data/ndbc"
MISSING = [99.0, 999.0, 9999.0, 99.00, 999.0]  # sentinel values to replace with NaN

os.makedirs("data/ndbc_stations", exist_ok=True)

for station in STATIONS.keys():
    station_dir = os.path.join(NDBC_DIR, station)
    yearly_dfs = []
    
    for year in ["2021", "2022", "2023"]:
        fpath = os.path.join(station_dir, f"{station}_{year}.txt")
        print(f"  Looking for: {fpath} — exists: {os.path.exists(fpath)}")
        if not os.path.exists(fpath):
            print(f"  Missing file: {fpath}, skipping.")
            continue
        
        NDBC_COLS = ["YY", "MM", "DD", "hh", "mm", "WDIR", "WSPD", "GST",
                     "WVHT", "DPD", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE"]
        df_year = pd.read_csv(fpath, sep=r'\s+', skiprows=2, header=None,
                              names=NDBC_COLS, na_values=[99.0, 999.0, 9999.0])
        yearly_dfs.append(df_year)
    
    df = pd.concat(yearly_dfs, ignore_index=True)

    df["time"] = pd.to_datetime(df[["YY", "MM", "DD", "hh", "mm"]].rename(
        columns={"YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}))
    df = df.set_index("time").drop(columns=["YY", "MM", "DD", "hh", "mm"])
    
    out_path = f"data/ndbc_stations/{station}_ndbc.csv"
    df.to_csv(out_path)
    print(f"  {station}: saved {out_path} — shape: {df.shape}")

print("\nNDBC cleaning done!")
os.makedirs("data/merged", exist_ok=True)

for station in STATIONS.keys():
    ndbc_path = f"data/ndbc_stations/{station}_ndbc.csv"
    era5_path = f"data/era5_stations/{station}_era5.csv"

    df_ndbc = pd.read_csv(ndbc_path, index_col="time", parse_dates=True)
    df_era5 = pd.read_csv(era5_path, index_col="time", parse_dates=True)

    df_ndbc = df_ndbc.resample("1h").mean()
    df_merged = df_era5.join(df_ndbc, how="inner")

    out_path = f"data/merged/{station}_merged.csv"
    df_merged.to_csv(out_path)
    print(f"  {station}: merged shape {df_merged.shape}")

print("\nMerge done!")
print("\n--- Missing data summary (% NaN per station for key variables) ---")
TARGET_COLS = ["WVHT", "DPD", "MWD", "swh", "mwp", "mwd"]

for station in STATIONS.keys():
    df = pd.read_csv(f"data/merged/{station}_merged.csv", index_col="time", parse_dates=True)
    print(f"\n{station} ({len(df)} rows):")
    for col in TARGET_COLS:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            print(f"  {col}: {pct:.1f}% missing")
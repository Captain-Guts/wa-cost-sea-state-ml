#second step
import cdsapi
import os

Out_Dir = 'data/wrf'
os.makedirs(Out_Dir, exist_ok = True)

c = cdsapi.Client()

Area = [49,-126,46,-123]
years = ['2021','2022','2023']

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
    "total_precipitation",
    "surface_pressure",
    "2m_temperature",
]
def download_era5_year(year):
    for month in range(1,13):
        month_str = f'{month:02d}'
        save_path = os.path.join(Out_Dir, f'era5_{year}_{month_str}.nc')
        if os.path.exists(save_path):
            print(f' this path: {save_path} already exist, skip')
            return 

        print(f' Downloading ERA5 - {year}...')

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24)],
                "area": Area,
                "format": "netcdf",
            },
            save_path,
        )

        print(f"  Saved: {save_path}")

print("Starting ERA5 download...")
print(f"Years: {years}")
print(f"Area: {Area}")
print(f"Variables: {len(VARIABLES)} variables")
print("-" * 40)

for year in years:
    print(f"\nYear: {year}")
    download_era5_year(year)

print("\nDone! All ERA5 files saved to:", Out_Dir)
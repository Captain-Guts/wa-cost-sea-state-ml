import xarray as xr
import numpy as np

ds = xr.open_dataset("data/era5_extracted/era5_2021_01/data_stream-wave_stepType-instant.nc")

# Check the nearest point to 46211
point = ds.sel(latitude=46.857, longitude=-124.244, method="nearest")
print("Snapped to lat:", float(point.latitude), "lon:", float(point.longitude))
print("swh sample values:", point.swh.values[:10])
print("Any non-NaN?", np.any(~np.isnan(point.swh.values)))

# Check all grid points for NaN coverage
print("\nNaN count per grid point (swh, first timestep):")
print(np.isnan(ds.swh.values[0]))
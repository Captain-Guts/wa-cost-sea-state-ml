# Washington Coast Sea State ML

Machine learning pipeline for predicting sea state and coastal weather conditions along the Washington State coast, adapted from the [Sea-ML](https://github.com/granantuin/sea-ml) methodology.

## Overview

Station-specific models (Random Forest and XGBoost/GBM) are trained on ERA5 reanalysis data to forecast key oceanographic and meteorological variables at NDBC buoy and coastal weather stations along the Washington coast — one of the most active commercial fishing regions in North America.

**Forecast horizons:** 0–24 h and 24–48 h

## Stations

| ID | Type | Location |
|----|------|----------|
| 46041 | Buoy | Cape Elizabeth (47.35°N, 124.73°W) |
| 46087 | Buoy | Neah Bay (48.49°N, 124.73°W) |
| 46211 | Buoy | Grays Harbor (47.12°N, 124.18°W) |
| DESW1 | Land | Destruction Island (47.68°N, 124.50°W) |
| LAPW1 | Land | La Push (47.91°N, 124.64°W) |
| WPTW1 | Land | Westport (46.90°N, 124.11°W) |

## Predicted Variables

**Buoy stations:** Significant wave height (WVHT), dominant wave period (DPD), mean wave direction (MWD)

**Land stations:** Wind speed (WSPD), wind direction (WDIR), atmospheric pressure (PRES)

## Data Sources

- **[NDBC](https://www.ndbc.noaa.gov/)** — historical buoy and coastal station observations (training labels)
- **[ERA5](https://cds.climate.copernicus.eu/)** — ECMWF reanalysis (training features: `swh`, `mwp`, `mwd`, `u10`, `v10`, `sp`, `t2m`, `tp`)
- **[Open-Meteo](https://open-meteo.com/)** — real-time NWP forecast data (operational inference)

## Pipeline

| Script | Description |
|--------|-------------|
| `01_data_aquire.py` | Download NDBC station observations |
| `02_era5.py` | Download ERA5 reanalysis via CDS API |
| `03_preprocess.py` | Merge and align NDBC + ERA5 data |
| `04_work_model.py` | Train Random Forest models → `models/` |
| `04b_model_gbm.py` | Train XGBoost/GBM models → `models/` |
| `05_evaluation.py` | Evaluate models; generate metrics and feature importance plots |
| `06_forecast.py` | 48-hour GBM forecast (days 1–2) using Open-Meteo |
| `07_predict.py` | 24-hour RF forecast (day 1) using Open-Meteo |

## Models

**Random Forest** (`models/`): baseline models using scikit-learn `RandomForestRegressor`. Directional variables (MWD, WDIR) are decomposed into sin/cos components, each fitted separately, then reconstructed via `arctan2`.

**XGBoost/GBM** (`models/`): gradient boosted models using `XGBRegressor` (200 estimators, lr=0.05, max_depth=6). Same sin/cos decomposition for directional targets. Generally achieves lower MAE than RF.

## ERA5 Features

| Feature | Description | Stations |
|---------|-------------|----------|
| `swh` | Significant wave height | Buoy only |
| `mwp` | Mean wave period | Buoy only |
| `mwd` | Mean wave direction | Buoy only |
| `u10` | 10m zonal wind | All |
| `v10` | 10m meridional wind | All |
| `sp` | Surface pressure | All |
| `t2m` | 2m air temperature | All |
| `tp` | Total precipitation (input only — not predicted) | All |

## Usage

```bash
# Train GBM models (requires preprocessed data in data/merged/)
python 04b_model_gbm.py

# Run 48-hour GBM forecast (requires models_gbm/)
python 06_forecast.py

# Run 24-hour RF forecast (requires models/)
python 07_predict.py
```

Forecast plots are saved to `forecast_output/` and `predict_output/` respectively.

## Results

Both models were evaluated against NDBC observations across calm and storm conditions:

**Calm conditions (April 21 2026, station 46041):** The RF model predicted WVHT, DPD, and MWD with high accuracy — wave height held near 1.6–1.8m all day, consistent with observed conditions.

**Storm event (November 19–21 2024):** A hindcast was run against a significant storm along the Washington coast. GBM outperformed RF across all buoy stations, better capturing the wave height peak and directional shifts during the storm. This is the primary motivation for using GBM as the operational forecaster in `06_forecast.py`.

**Key finding:** RF is a strong baseline under calm, well-sampled conditions. GBM generalizes better to high-energy storm events — the more safety-critical scenario for commercial fishing operations.

## Use Case

Operational sea state forecasting to support maritime safety and route planning for commercial fishing vessels along the Washington coast.

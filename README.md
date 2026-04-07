# wa-cost-sea-state-ml
This project consist in creating a ML model to predict sea state and precipitation along the Washigton coast, this project has been adapted from [Sea Ml](https://github.com/granantuin/sea-ml) methodology.

## Overview
This project inteads in applying ML models to forescast key oceonography and meteorology variables at costall NDBC costal bouy stations along the Washigton State coast one of the most comercial fishing region in North American 

Predicted Variables:
- Significant wave heigth
- Swell period
- Wave prediction
- Precipitation Acumulation

Forecasting Horizons: **0-24h** and **24-48h**

Data sources:
[NDBC] ((https://www.ndbc.noaa.gov/) [Costal buoy observation records]
[NOAA WRF] (https://www.noaa.gov/) [Numerical weather prediction output]

## Methods 
- WRF grind point output used as predectife feature
- Station specific model trained per variables using scikit-learn
- Pipeline: preprocessig - feature extracting - training - cross validation

## Use Case

Operational sea state forecasting to support maritime fishing vessels safety and route along the Washigton Coast

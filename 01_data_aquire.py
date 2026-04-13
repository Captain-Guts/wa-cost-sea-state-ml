import os 
import io
import requests as rqs
import gzip 
import shutil 

Station = [
    '46041', #Cape Elizabeth 
    '46211', #Grays Harbor
    '46087', #Cape FLattery
    'LAPW1', #La push
    'WPTW1', #Wesport
    'DESW1', #Destruction Island
]

Years = range(2021,2024)
Output_DIR = 'data/ndbc'
os.makedirs(Output_DIR, exist_ok=True)

Bases_URL = 'https://www.ndbc.noaa.gov/data/historical/stdmet'

def download_station_year(station, year, Output_DIR):
    filename = f'{station.lower()}h{year}.txt.gz'
    url = f'{Bases_URL}/{filename}'
    
    station_DIR = os.path.join(Output_DIR, station)
    os.makedirs(station_DIR, exist_ok=True)
    save_path = os.path.join(station_DIR, f'{station}_{year}.txt')

    #skip if already downloaded                          
    if os.path.exists(save_path):
        print(f' Already Exist: {save_path}, skipping.')
        return

    print(f' Downloading {station} - {year}...')
    response = rqs.get(url, timeout=30)

    if response.status_code == 200:
        compressed = io.BytesIO(response.content)
        with gzip.open(compressed) as f_in:
            with open(save_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f' Saved"{save_path}')
    else:
        print(f' Not found: {url} (status {response.status_code})')

print(f' Starting Download of NDBC data ...')
print(f' Station: {Station}')
print(f' Years: {Years}' )
print(f'-' * 40)

for station in Station:
    print(f'\nStation {station}')
    for years in Years:
        download_station_year(station, years, Output_DIR)
print(f'Work has been completed')
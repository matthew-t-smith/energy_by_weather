import pandas as pd
import datetime as dt

ENERGY = 'energy_data_raw.csv'
WEATHER = 'wellington_weather_raw.csv'
RAIN = 'wellington_rain_raw.csv'

# Read half-hourly energy data and merge usages to hourly increments
raw_data = pd.read_csv(ENERGY, skiprows=(lambda x: x == 49*(int(x/49)) + 2))
reads = []
for i in range(0, raw_data.shape[0] - 2, 2):
    new_row = [pd.to_datetime(raw_data.at[i, 'reading_start'], dayfirst=True).round(
        'H'), (raw_data.at[i, 'usage'] + raw_data.at[(i+1), 'usage'])]
    reads.append(new_row)

energy = pd.DataFrame(reads, columns=['Date(UTC)', 'usage'])


# Timeslice rain, weather and energy data
def time_slice(dataset):
    dataset['Date(UTC)'] = pd.to_datetime(dataset['Date(UTC)'], dayfirst=True)
    dataset.set_index('Date(UTC)', inplace=True)
    dataset = dataset['2018-03-29':'2019-01-29']
    dataset.reset_index(inplace=True)
    return dataset


rain = pd.read_csv(RAIN, usecols=['Date(UTC)', 'Amount(mm)'], dayfirst=True)
rain = time_slice(rain)
weather = pd.read_csv(
    WEATHER, usecols=['Date(UTC)', 'Tmax(C)', 'Tmin(C)', 'RHmean(%)'], dayfirst=True)
weather = time_slice(weather)
energy = time_slice(energy)

# Create a single DataFrame to compile feed-in data
final = []
for index, row in weather.iterrows():
    energy_filter = energy['Date(UTC)'] == row['Date(UTC)']
    rain_filter = rain['Date(UTC)'] == row['Date(UTC)']

    rainfall = rain[rain_filter]['Amount(mm)'].values
    try:
        rainfall = float(rainfall)
    except (TypeError, ValueError):
        rainfall = -1

    usage = energy[energy_filter]['usage'].values
    try:
        usage = float(usage)
    except (TypeError, ValueError):
        usage = -1

    try:
        row['RHmean(%)'] = float(row['RHmean(%)']) / 100
    except (TypeError, ValueError):
        row['RHmean(%)'] = -1

    entry = [row['Date(UTC)'], row['Tmax(C)'], row['Tmin(C)'],
             row['RHmean(%)'], rainfall, usage]
    final.append(entry)

data = pd.DataFrame(final, columns=[
    'DateTime', 'Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
data.fillna(-1, inplace=True)

print(data.shape)
drops = []
for index, row in data.iterrows():
    if (row.isin(['-', -1.0]).any() or (row['Usage(kWh)'] >= 10)):
        drops.append(index)
data.drop(drops, axis='index', inplace=True)
print(data.shape)
data.to_csv('./data.csv')

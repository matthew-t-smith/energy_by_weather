
# A Case Study in Energy Efficiency
## Matthew T. Smith

### Problem Statement
A household’s normal energy usage most largely fluctuates due to the heating and cooling of the space based on weather. This particular analysis looks to model and predict a single household’s usage as it relates to that historical weather data. This compact model can be passed between an energy supplier or retailer and the consumer without having to hand off a user’s actual usage trends. Businesses can rely on normal meter reads and predictions from such a model to price their utility, and the consumers can better see how they use their energy and budget or plan accordingly. This small example of “federated learning” keeps privacy at the forefront of the model’s architecture. Through the use of the Keras Deep Learning library in Python on top of Google’s TensorFlow’s backend, a linear regression model is trained on a 10-month electricity usage report for an apartment in Wellington, New Zealand paired with the weather (high/low temperature, relative humidity, and rainfall) from that same time period. Predictions are then made on sample weather data to demonstrate probable energy consumption.

### Technology

#### Software
- **Keras Framework** [https://github.com/keras-team/keras]
- **TensorFlow Framework** [https://www.tensorflow.org/install/]
- **Python 3.7** [https://www.python.org/downloads/]
- **h5py** [http://docs.h5py.org/en/latest/build.html]
- **numpy** [https://scipy.org/install.html]
- **pandas** [https://pandas.pydata.org/pandas-docs/stable/install.html]
- **matplotlib** [https://scipy.org/install.html]

#### Hardware
- **Operating System:** macOS Mojave 10.14.4
- **Processor:** 2.3 GHz Intel Core i7
- **Memory:** 16 GB 1600 MHz DDR3
- **GPU**: N/A

### Data

#### Weather
- CliFlo – the web application providing access to New Zealand’s National Climate Database [https://cliflo.niwa.co.nz/]
- Collected from Wellington, NZ’s Greta Point Weather Station
    - Closest station to the home with complete data
- 10-month dataset of high and low temperatures (Celsius), relative humidity (percent), and rainfall (millimeters)

#### Energy Consumption
- Powershop NZ – Energy retailer for the home [https://www.powershop.co.nz/]
- Half-hourly consumption (kWh) fed from smart meter at the property
- The data downloaded is from my own home in NZ; consumption data is not freely available via the retailer


```python
import pandas as pd

ENERGY = './data/energy_data_raw.csv'
WEATHER = './data/wellington_weather_raw.csv'
RAIN = './data/wellington_rain_raw.csv'

E = pd.read_csv(ENERGY)
W = pd.read_csv(WEATHER)
R = pd.read_csv(RAIN)

print("\nEnergy: \n")
print(E.head(3))
print("\nWeather: \n")
print(W.head(3))
print("\nRainfall: \n")
print(R.head(3))
```

    
    Energy: 
    
             reading_start          reading_end  usage
    0  29/03/2018 00:00:01  29/03/2018 00:30:00   0.05
    1  29/03/2018 00:00:01  30/03/2018 00:00:00  11.51
    2  29/03/2018 00:30:01  29/03/2018 01:00:00   0.04
    
    Weather: 
    
                         Station         Date(UTC) Tmax(C) Period(Hrs)  Tmin(C)  \
    Wellington   Greta Point Cws  29/03/2018 00:00    20.5           1     19.5   
    Wellington   Greta Point Cws  29/03/2018 01:00    22.4           1     19.9   
    Wellington   Greta Point Cws  29/03/2018 02:00    23.2           1     22.4   
    
                Period(Hrs).1 Tgmin(C) Period(Hrs).2 Tmean(C) RHmean(%)  \
    Wellington              1        -             -     19.8        66   
    Wellington              1        -             -     20.8        61   
    Wellington              1        -             -     22.8        51   
    
               Period(Hrs).3 Freq  
    Wellington             1    H  
    Wellington             1    H  
    Wellington             1    H  
    
    Rainfall: 
    
                           Station         Date(UTC)  Amount(mm)  Period(Hrs) Freq
    0  Wellington, Greta Point Cws  29/03/2018 00:00         0.0            1    H
    1  Wellington, Greta Point Cws  29/03/2018 01:00         0.0            1    H
    2  Wellington, Greta Point Cws  29/03/2018 02:00         0.0            1    H


### Setup

#### Installation
The project and data can be downloaded from here: [https://github.com/matthew-t-smith/energy_by_weather]

From the home directory, run:

`~/energy_by_weather $ pip3 install -r requirements.txt`

#### Data Cleaning
The raw data already exists in the `/data/` directory, as does the cleaning script and output data. To re-clean the data, from the data directory, run:

`~/energy_by_weather/data $ python3 data_prep.py`

#### Model Creation and Training
The model will be created new, train, and save as an `.h5` file, as already exists in the directory currently. Plots will also be re-saved over the existing files when running. To re-create and re-train the model, from the home directory, run:

`~/energy_by_weather $ python3 model.py`


```python

```

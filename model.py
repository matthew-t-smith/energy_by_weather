import pandas as pd
import math
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# Normalizing function for dataset


def norm(d):
    return (d - d.min()) / (d.max() - d.min())


def denorm(y, d):
    return (y * (d.max() - d.min()) + d.min())


# Load the data
data = pd.read_csv('./data/data.csv',
                   usecols=['Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
data.fillna(0, inplace=True)
data['RHmean(%)'] = data['RHmean(%)'] / 100
dataset = data.values.astype(float)
X = dataset[:, 0:4].round(2)
Y = dataset[:, -1].round(2)

# Normalize the dataset
normed_X = np.apply_along_axis(norm, 0, X)
normed_Y = norm(Y)
print(normed_X[0:10, :])
print(normed_Y[0:10])

# Split datasets into training and test
split = 0.8
val = math.floor(split*len(normed_X))
train_X = normed_X[:val]
train_Y = normed_Y[:val]
test_X = normed_X[val:]
test_Y = normed_Y[val:]

# Build the model with layers
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
adam = Adam(lr=1e-5)
model.compile(loss='mean_squared_error',
              optimizer=adam, metrics=['mae', 'mse'])


# Fit the model
history = model.fit(train_X,
                    train_Y,
                    epochs=100,
                    validation_data=(test_X, test_Y))
model.save('energy.h5')

print(denorm(model.predict(test_X[-10:, :]), Y))

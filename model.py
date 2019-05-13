import pandas as pd
import math
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# Load the data
data = pd.read_csv('./data/data.csv',
                   usecols=['Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
dataset = data.values.astype(float)
X = dataset[:, 0:4] / 100
Y = dataset[:, 4]

# Split datasets into training and test
split = 0.8
val = math.floor(split*len(X))
train_X = X[:val]
train_Y = Y[:val]
test_X = X[val:]
test_Y = Y[val:]

# Scale datasets
max_energy = train_Y.max()
train_Y = train_Y / max_energy
test_Y = test_Y / max_energy
(trainX, trainY) = (train_X, train_Y)
(testX, testY) = (test_X, test_Y)

# Build the model with layers
model = Sequential()
model.add(Dense(4, input_dim=4,
                kernel_initializer='normal', activation='relu'))
model.add(Dense(4, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
adam = Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[
              'mean_absolute_error', 'mean_squared_error'])


# Fit the model
history = model.fit(trainX,
                    trainY,
                    epochs=200,
                    validation_split=0.8)
model.save('energy.h5')
results = model.evaluate(testX, testY)
print(results)

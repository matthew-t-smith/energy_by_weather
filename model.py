import pandas as pd
import math
from keras.layers import Dense
from keras.models import Sequential

# Load the data
data = pd.read_csv('./data/data.csv',
                   usecols=['Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
dataset = data.values.astype(float)
X = dataset[:, 0:4]
Y = dataset[:, 4]

# Split datasets into training and test
split = 0.8
val = math.floor(split*len(X))
train_X = X[:val]
train_Y = Y[:val]
test_X = X[val:]
test_Y = Y[val:]

print(dataset[0])

# Build the model with layers
model = Sequential()
model.add(Dense(4, input_dim=4,
                kernel_initializer='normal', activation='relu'))
model.add(Dense(4, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
history = model.fit(train_X,
                    train_Y,
                    epochs=200)
model.save('energy.h5')
results = model.evaluate(test_X, test_Y)
print(results)

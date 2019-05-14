import pandas as pd
import math
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Normalizing function for dataset


def norm(d):
    return (d - d.min()) / (d.max() - d.min())


def denorm(y, d):
    return (y * (d.max() - d.min()) + d.min())


# Load the data
data = pd.read_csv('./data/data.csv',
                   usecols=['Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
dataset = data.values.astype(float)
X = dataset[:, 0:4]
Y = dataset[:, 4]

# Normalize the dataset
normed_X = np.apply_along_axis(norm, axis=0, arr=X)
normed_Y = norm(Y)

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
model.add(Dropout(0.25))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
adam = Adam(lr=1e-5)
model.compile(loss='mean_squared_error',
              optimizer=adam, metrics=['mean_absolute_error', 'mean_squared_error'])

# Fit the model
history = model.fit(train_X,
                    train_Y,
                    epochs=300,
                    validation_data=(test_X, test_Y),
                    verbose=0)
model.save('energy.h5')

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
         label='Val Error')
plt.legend()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.plot(hist['epoch'], hist['mean_squared_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_squared_error'],
         label='Val Error')
plt.legend()
plt.show()

print(denorm(model.predict(test_X[-10:, :]), Y))

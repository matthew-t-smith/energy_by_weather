import pandas as pd
import math
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Normalizing function for dataset


def norm(d):
    return (d - d.min()) / (d.max() - d.min())


# Load the data
data = pd.read_csv('./data/data.csv',
                   usecols=['Tmax(C)', 'Tmin(C)', 'RHmean(%)', 'Rain(mm)', 'Usage(kWh)'])
data.fillna(0, inplace=True)
data['RHmean(%)'] = data['RHmean(%)'] / 100
dataset = data.values.astype(float)
X = dataset[:, 0:4]
Y = dataset[:, -1]

# Normalize the dataset
normed_X = np.apply_along_axis(norm, 0, X)
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
model.add(Dense(4, input_dim=4, kernel_regularizer=l2(l=1e-3), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, kernel_regularizer=l2(l=1e-3), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='linear'))
adam = Adam(lr=1e-5)
model.compile(loss='mean_squared_error',
              optimizer=adam, metrics=['accuracy'])


# Fit the model
history = model.fit(train_X,
                    train_Y,
                    epochs=100,
                    validation_data=(test_X, test_Y))
model.save('energy.h5')

# Plot the results of the training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

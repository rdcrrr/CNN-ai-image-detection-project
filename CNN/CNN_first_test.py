import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input
import numpy as np

# Load CIFAR-10 (already built into Keras)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Combine train and test for a similar feel to your original code
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# Convert to binary: classes 0-4 → 0, classes 5-9 → 1
# (simulates your original binary classification task)
y = (y >= 5).astype(int).flatten()

# Normalize
X = X / 255.0

# Build the model (same architecture as yours)
model = Sequential()

model.add(Input(shape=X.shape[1:]))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)


model.save('first_temp_module.keras')
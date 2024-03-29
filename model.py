import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers
import pickle

random.seed(0)

npz = np.load('train.npz')
train_x = npz['inputs']
train_y = npz['target']

npz = np.load('valid.npz')
valid_x = npz['inputs']
valid_y = npz['target']

npz = np.load('test.npz')
test_x = npz['inputs']
test_y = npz['target']

model = models.Sequential()
model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(32, (4,4), activation='relu', input_shape=(60, 64, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(16, (4,4), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(20))

lr = 0.01

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

epochs = 25

model.fit(
    train_x, train_y,
    epochs = epochs,
    validation_data = (valid_x, valid_y),
    verbose = 2
)

pickle.dump(model, open('model.pkl', 'wb'))
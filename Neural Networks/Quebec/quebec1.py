### Assignment Quebec 
### Part 1: 
### Train a regular Deep Neural Network to identify the 10 classes of objects in the CIFAR-10 dataset. 
## 	How good of an accuracy can you get on the test set?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


# CIFAR10
# 50000/10000 32x32 color images of various real-world items (cars, ships, etc...)
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preparing the data:
print("Scaling input data...")
max_val = np.max(x_train).astype(np.float32)
print("Max value: " +  str(max_val))
x_train = x_train.astype(np.float32) / max_val
x_test = x_test.astype(np.float32) / max_val
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# converting class vectors to binary class matrices:
num_classes = len(np.unique(y_train))
print("Number of classes in this dataset: " + str(num_classes))
if num_classes > 2:
	print("One hot encoding targets...")
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

print("Original input shape: " + str(x_train.shape[1:]))

from keras.models import  #
from keras.layers import Dropout, Dense, Flatten, BatchNormalization, Activation
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam

def leakyReLU(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

hidden1 = 4800
hidden2= 1200
hidden4 = 300
hidden5 = 300
hidden6 = 300
epochs = 20
act = 'relu'
init = 'he_uniform'
mloss = 'categorical_crossentropy'
opt = 'Adam'

#using batch normalization and dropout
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(hidden1, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.5)) #0.3 = percentage
model.add(Dense(hidden2, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(hidden4, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(hidden5, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(hidden6, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=mloss,
              optimizer=opt,
              metrics=['accuracy'])



print("\nTraining with " +  act + ", " + init + ", BatchNormalization, and Dropout:")
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test),
              		shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])

"""
epoch: 10 
test accuracy: 0.5085
training with elu, he_normal, BatchNormalization, and Dropout (0.3)
optimizer: Adam

epoch:  30
test accuracy: 0.513
training with elu, he_normal, BatchNormalization, and Dropout
optimizer: nesterov momentum 

epoch: 30 
test accuracy: 0.543
2 hidden layers: 300, 100 
training with relu, he_uniform, BatchNormalization, and Dropout
optimizer: Adam

epoch: 20 
test accuracy: 0.529
hidden1 = 4800
hidden2= 1200
hidden4 = 300
hidden5 = 300
hidden6 = 300
training with relu, he_uniform, BatchNormalization, and Dropout
optimizer: adam

epoch: 30 
test accuracy: 0.52680
training with relu, he_uniform, BatchNormalization, and Dropout
optimizer: AdaGrad

epoch: 30
test accuracy: 0.5264
training with relu, he_uniform, BatchNormalization, and Dropout
optimizer: momentum

epoch: 40
test accuracy: 0.5311
training with relu, he_uniform, BatchNormalization, and Dropout 
optimizer: momentum 

epoch: 20
test accuracy: 0.4913
training with relu, he_
uniform, BatchNormalization, and Dropout (0.5)
optimizer: momentum

"""









### ASSIGNMENT QUEBEC
# Mitsuka Kiyohara
# Block D | AT CS

### Part 1: 
### Train a regular Deep Neural Network to identify the 10 classes of objects in the CIFAR-10 dataset. 
## 	How good of an accuracy can you get on the test set?

"""
Best accuracy: 0.543
epoch: 30 
2 hidden layers: 300, 100 
training with relu, he_uniform, BatchNormalization, and Dropout
optimizer: Adam
running 6 hidden layers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


# load CIFAR10 dataset
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

from keras.models import  Sequential
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
Test Runs: 

epoch: 10 
test accuracy: 0.5085
training with elu, he_normal, BatchNormalization, and Dropout (0.3)
optimizer: Adam

epoch:  30
test accuracy: 0.513
training with elu, he_normal, BatchNormalization, and Dropout
optimizer: nesterov momentum 

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








### Part 2: 
### How good of an accuracy can you get on the test set? How does that compare to Part One's accuracy? 

"""
Best accuracy: 0.8539

Running CNNs produces a significantly higher accuracy results than regular deep neural nets, especially when we
compare the accuracy of the first epoch. 
"""

#libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#tensorflow and keras 
import tensorflow as tf
from tensorflow import keras

# CIFAR10
# 50000/10000 32x32 color images of various real-world items (cars, ships, etc...)
# https://www.cs.toronto.edu/~kriz/cifar.html
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preparing the data:
print("Scaling input data...")
max_val = np.max(x_train).astype(np.float32)
print("Max value: " +  str(max_val))
x_train = x_train.astype(np.float32) / max_val
x_test = x_test.astype(np.float32) / max_val
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Convert class vectors to binary class matrices.
num_classes = len(np.unique(y_train))
print("Number of classes in this dataset: " + str(num_classes))
if num_classes > 2:
  print("One hot encoding targets...")
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

print("Original input shape: " + str(x_train.shape[1:]))


from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization #importing stuff to do for layers

model = Sequential()

# switched of with relu/elu and 'valid'/'same' padding 
# also increased the number of feature maps by a multiplier of 2 
# added dropout of 0.4, batch normalization, and max pooling

model.add(Conv2D(32, (3,3), padding='valid', input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Conv2D(64, (3,3), padding='valid'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 

model.add(Flatten())
model.add(Dense(2048)) 
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
 

from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam

mloss = 'categorical_crossentropy'
opt = 'Adam' #found that the optimizer with the highest accuracy was Adam 

model.compile(loss=mloss, optimizer=opt, metrics=['accuracy'])

epochs = 73

history = model.fit(x_train, y_train, epochs=epochs,verbose=2, validation_data=(x_test, y_test), shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])

"""
Test Runs: 

test accuracy: 0.7810
epochs: 20 
opt: Adam
total params: 1,715,242

test accuracy: 0.8104
epochs: 20
opt: Adagrad
total params: 300,..

test accuracy: 0.8157 
epochs: 43
opt: Adagrad 
total params: 1,359,914

test accuracy: 0.8245
epochs: 59 
opt: Adagrad 
total params: 1,359,914


"""











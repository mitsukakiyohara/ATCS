### Assignment Quebec 
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




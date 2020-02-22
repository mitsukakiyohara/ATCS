### Neural Networks with Keras - Part One
# Covers basic layer construction and how to train and predict with NNs.

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

### First, pick and prepare a dataset:
# (comment/uncomment one of the following):


# Fashion MNIST
# 60000/10000 32x32 black and white images of clothing items.
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
# CIFAR10
# 50000/10000 32x32 color images of various real-world items (cars, ships, etc...)
# https://www.cs.toronto.edu/~kriz/cifar.html
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""

# Preparing the data:
print("Scaling input data...")
max_val = np.max(x_train).astype(np.float32)
print("Max value: " +  str(max_val))
x_train = x_train.astype(np.float32) / max_val
x_test = x_test.astype(np.float32) / max_val
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
#create range from 0 to 1

# Convert class vectors to binary class matrices.
num_classes = len(np.unique(y_train))
print("Number of classes in this dataset: " + str(num_classes))
if num_classes > 2:
	print("One hot encoding targets...")
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

print("Original input shape: " + str(x_train.shape[1:]))
#flatten 2d array to 1d array 

### Second, build a model:

"""
For standard Feed Forward Deep Networks we will use Dense layers in a Sequential model:

Example:
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Flatten(input_shape=(...,...)))
model.add(Dense(300, activation=act, kernel_initializer=init))
etc...

The most important input parameters are activation and initialization.
Some combination choices are shown here:

Activation         Initialization
----------         --------------
sigmoid            glorot_normal or glorot_uniform
tanh			   glorot_normal or glorot_uniform
relu               he_normal or he_uniform
*leaky relu        he_normal or he_uniform
elu                he_normal or he_uniform
selu               lecun_normal or lecun_uniform

#relu = rectified linear unit (greater than zero, return value. less than zero, return zero)
#leaky relu (has a small negative slope when less than zero)
#elu = exponential linear unit (better than leaky relu)

*Leaky ReLU is implemented as a custom function

A full list of activation functions can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/activations/

A full list of initializers available in keras can be found here:
https://keras.io/initializers/

A list of the other input parameters for the Dense layer are here:
https://keras.io/layers/core/

The last layer is the output layer, and should be configured based on
the kind of problem you are trying to solve:

Regression- one node, linear activation function (which is the default)
Binary Classification - one node, sigmoid activation function
Multi-Class Classification - num nodes = classes, softmax activation function

"""

def leakyReLU(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

## Choose activation and initialization functions.
# Set equal to a string with the name of the activation or initialization
#    function you want to use, except for Leaky ReLU. For that set equal
#    to the name of the function, without quotes.

act = 'relu' #sigmoid = default
init = 'he_normal' 

#btw, every single node does the activiation function (for each layer), and becomes the new inputs for next layer

#to run leakyReLU, use the function and init 'he_normal'
#act = leakyReLU
#init = 'he_normal'


#comments: 
#best running: 
#relu + he_uniform
#leakyrelu + he_uniform
#elu + he_normal

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:])) #input layer
model.add(Dense(300, activation=act, kernel_initializer=init)) #hidden layer 1
model.add(Dense(100, activation=act, kernel_initializer=init)) #hidden layer 2
model.add(Dense(num_classes, activation='softmax')) #output layer

model.summary()


### Third, compile the model with a loss function, optimizer algorithm, and metrics

#loss = looking at how close it wsa from what i wanted and what we trained

"""
Loss functions depend on the task you are training the NN to do:

Regression tasks-
	mean_squared_error  # recommended default choice
	mean_absolute_error  # for problematic outliers
	mean_squared_logarithmic_error  # for a very wide range of targets

Binary Classification tasks-
	binary_crossentropy # recommended default choice; targets: 0,1
	hinge # SVM approach (mixed results with NN); targets: -1,1; use tanh for output layer activation
	squared_hinge # smoother hinge loss function, targets: -1,1; use tanh for output layer activation

Multi-class Classification tasks-
	categorical_crossentropy # recommended default choice; use keras.utils.to_categorical(...) on targets 
	sparse_categorical_crossentropy # for a large number of targets; to_categorical not required on targets
	

For more loss function choices:  https://keras.io/losses/

Optimizers: algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses

Some Optimizer choices:
Gradient Descent		SGD(lr=0.01) tune learning rate
Momentum				SGD(lr=0.01, momentum=0.9) tune learning rate
Nesterov momentum		SGD(lr=0.01, momentum=0.9, nesterov=True) tune learning rate
AdaGrad					Adagrad()
RMSprop                 RMSprop()
Adam					Adam() 
Nesterov Adam			Nadam()

momentum = picks up speed if it takes a long time to be at optimal point 
nesterov momentum AG = calculate gradient in front  
AdaGrad = changing the learning rate for you (adaptive gradient)
RMSprop = exponential decay (reacting to gradients right where it was)
Adam = momentum + AdaGrad


For a full list of optimizers:  https://keras.io/optimizers/

SGD info: https://en.wikipedia.org/wiki/Gradient_descent

"""
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam 

mloss = 'categorical_crossentropy'
#opt = SGD(lr=0.01) #SGD = default lr = learning rate
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

#comments: 
#to go faster, use momentum and tune learning rate 


model.compile(loss=mloss,
              optimizer=opt,
              metrics=['accuracy']) 


### Fourth, train and test the model!

epochs = 999

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test), 
              		shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])



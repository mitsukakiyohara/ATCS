
__author__  = "Mitsuka Kiyohara"
__version__ = "3.7.4"

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

#Import Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

"""
Credits: 
@online{clanuwat2018deep,
  author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
  title        = {Deep Learning for Classical Japanese Literature},
  date         = {2018-12-03},
  year         = {2018},
  eprintclass  = {cs.CV},
  eprinttype   = {arXiv},
  eprint       = {cs.CV/1812.01718},
}
"""
# Input image dimensions
img_rows, img_cols = 28, 28

def load(f):
    return np.load(f)['arr_0']

# Load the data
x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test = load('kmnist-test-labels.npz')


# Preparing the data:
print("Scaling input data...")
max_val = np.max(x_train).astype(np.float32)
print("Max value: " +  str(max_val))
x_train = x_train.astype(np.float32) / max_val
x_test = x_test.astype(np.float32) / max_val
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

"""
Image Processing:

[info]
	Channels Last: Image data is represented in a three-dimensional array where the last channel represents the color channels, e.g. [rows][cols][channels].
	Channels First: Image data is represented in a three-dimensional array where the first channel represents the color channels, e.g. [channels][rows][cols].
"""
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# Convert class vectors to binary class matrices (One hot encoding)
num_classes = len(np.unique(y_train))
print("Number of classes in this dataset: " + str(num_classes))
if num_classes > 2:
	print("One hot encoding targets...")
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

print("Original input shape: " + str(x_train.shape[1:]))



from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization

model = Sequential()

"""
1st Case: Running a CNN [Using a similar code to Assignment Quebec]

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
opt = 'Adam' 

model.compile(loss=mloss, optimizer=opt, metrics=['accuracy'])

epochs = 10

history = model.fit(x_train, y_train, epochs=epochs,verbose=2, validation_data=(x_test, y_test), shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])
"""


#2nd case: running a normal CNN using only 10 layers 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(num_classes, activation='softmax'))

#Compiling [Optimizer: Adam, Loss: Categorical Crossentropy]
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#Fittting [Epochs: 100, Batch Class: 128]
hist = model.fit(x_train, y_train,
                 batch_size=128,
                 epochs=100,
                 verbose=2,
                 validation_data=(x_test, y_test),
                )

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])

"""
Plotting Method

Credits to: https://github.com/netsatsawat
The author used plotly as a way to plot down the points of each epoch run 

"""
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

def reshape_3d_to_2d(data_array: np.ndarray) -> np.ndarray:
    """
    Function to convert 3d numpy array to 2d numpy array
    @Arg:
      data_array: the numpy array, we want to convert
      
    Return:
      The converted numpy array
    """
    out_array = data_array.reshape(data_array.shape[0], 
                                   data_array.shape[1] * data_array.shape[2])
    return out_array


def compute_variance(kmnist_img: np.ndarray) -> (list, np.ndarray):
    """
    Function to compute the individual and cumulative variance explained 
    @Args:
      kmnist_img: the numpy array contains the MNIST data
      
    Return:
      Individual and cumulative explained variances
    """
    _img = kmnist_img.astype(np.float32)
    if len(_img.shape) == 3:
        _img = reshape_3d_to_2d(_img)
    
    _img_scale = StandardScaler().fit_transform(_img)
    # compute mean and conv matrix for eigenvalue, eigenvector
    mean_vec = np.mean(_img_scale, axis=0)
    conv_mat = np.cov(_img_scale.T)
    eig_vals, eig_vecs = np.linalg.eig(conv_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)  # sort the eigenvalue, eigenvector pairs from high to low
    total_ = sum(eig_vals)
    ind_var_exp_ = [(i/total_)*100 for i in sorted(eig_vals, reverse=True)]  
    cum_var_exp_ = np.cumsum(ind_var_exp_)
    return ind_var_exp_, cum_var_exp_


def plot_variance_explain(ind_var_exp: list, cum_var_exp: list,
                          n_dimension: int=786, variance_limit: int=50
                         ):
    """
    Function to plot the individual and cumulative explained variances 
      with zoom into specific limit numbers
    @Args:
      ind_var_exp:
      cum_var_exp:
      n_dimension:
      variance_limit:
      
    Return:
      No object returned, only visualization on console / notebook
    """
    # Trace 1 and 2 will act as partial plot, whereas 3 and 4 will be full plot
    trace1 = go.Scatter(x=list(range(n_dimension)),
                        y=cum_var_exp,
                        mode='lines+markers',
                        name='Cumulative Explained Variance (Partial)',
                        hoverinfo='x+y',
                        line=dict(shape='linear',
                                  color='red'
                                 )
                       )
    
    trace2 = go.Scatter(x=list(range(n_dimension)),
                        y=ind_var_exp,
                        mode='lines+markers',
                        name='Individual Explained Variance (Partial)',
                        hoverinfo='x+y',
                        line=dict(shape='linear',
                                  color='white'
                                 )
                       )

    trace3 = go.Scatter(x=list(range(n_dimension)),
                        y=cum_var_exp,
                        mode='lines+markers',
                        name='Cumulative Explained Variance (Full)',
                        hoverinfo='x+y',
                        line=dict(shape='linear',
                                  color='blue'
                                 )
                       )
    
    trace4 = go.Scatter(x=list(range(n_dimension)),
                        y=ind_var_exp,
                        mode='lines+markers',
                        name='Individual Explained Variance (Full)',
                        hoverinfo='x+y',
                        line=dict(shape='linear',
                                  color='gray'
                                 )
                       )

    fig = tls.make_subplots(rows=2, cols=1, print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig['layout']['xaxis1'].update(range=[0, variance_limit], color='gray', showgrid=False)
    fig['layout']['yaxis1'].update(range=[0, 90], color='gray', showgrid=False)
    fig['layout']['xaxis2'].update(color='gray', showgrid=False)
    fig['layout']['yaxis2'].update(color='gray', showgrid=False)
    fig['layout'].update(title=dict(text='<b>Variances Explained</b> (zoom to %s variances)' % 
                                    variance_limit,
                                    font=dict(size=20, color='white')
                                   ), 
                         height=600, width=850, 
                         showlegend=True, paper_bgcolor='#000', plot_bgcolor='#000',
                         autosize=False,
                         legend=dict(traceorder='normal',
                                     font=dict(size=12,
                                               color='white'
                                              )
                                    )
                         )
    py.iplot(fig)
    

ind_var_exp, cum_var_exp = compute_variance(kmnist_train_image)






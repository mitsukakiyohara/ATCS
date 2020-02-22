import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Linear Regression with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_reg = LinearRegression()

__author__ = "Mitsuka Kiyohara"

# Explore the Boston Housing data set
boston = pd.read_csv('boston_housing.csv')
print(boston.info())
"""

print(boston.head())

# Description of Boston Housing data set:
# CRIME RATE =  per capita crime rate by town
# LARGE LOT = proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUSTRY = proportion of non-retail business acres per town
# RIVER = Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX = nitric oxides concentration (parts per 10 million)
# ROOMS = average number of rooms per dwelling
# PRIOR 1940 = proportion of owner-occupied units built prior to 1940
# EMP DISTANCE = weighted distances to five Boston employment centres
# HWY ACCESS = index of accessibility to radial highways
# PROP TAX RATE = full-value property-tax rate per $10,000
# STU TEACH RATIO = pupil-teacher ratio by town
# AFR AMER = 1000(AFA - 0.63)^2 where AFA is the proportion of African Americans by town
# LOW STATUS = % lower status of the population
# MEDIAN VALUE = Median value of owner-occupied homes in $1000’s

# Creator: Harrison, D. and Rubinfeld, D.L.
# This is a copy of UCI ML housing dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


#MSE = mean squared error
#R^2 = percentage 

boston.hist(figsize=(14,7))
plt.title("Histogram")
plt.show()
boston.boxplot(figsize=(14,7))
plt.title("Box plot")
plt.show()

#compares all of the columns to the median value --- can figure out which column most impacts the median value
corr_matrix=boston.corr()
print(corr_matrix["MEDIAN VALUE"].sort_values(ascending=False))

pd.plotting.scatter_matrix(boston[ ['MEDIAN VALUE','LOW STATUS','ROOMS','INDUSTRY','NOX','PROP TAX RATE','STU TEACH RATIO'] ], figsize=(14,7))
plt.title("Scatter Matrix")
plt.show()

plt.scatter(boston['LOW STATUS'], boston['MEDIAN VALUE'])
plt.title("Scatter of Median Value(y) vs Low Status(x)")
plt.show()
"""

# Setup a sample regression, using scikit
boston_inputs = boston[ ['LOW STATUS'] ] # You can add more columns to this list...
boston_targets = boston['MEDIAN VALUE']

# Train the weights
lin_reg.fit(boston_inputs,boston_targets)

# Generate outputs / Make Predictions
boston_outputs = lin_reg.predict(boston_inputs)

# What's our error?
boston_mse = mean_squared_error(boston_targets, boston_outputs)
# What's our R^2? (amount of output variance explained by these inputs)
boston_r2 = r2_score(boston_targets, boston_outputs)
"""
print("MSE using LOW STATUS (scikit way): " + str(boston_mse*len(boston)))
print("R^2 using LOW STATUS (scikit way): " + str(boston_r2))
print("Weights/Coefficients of Regression: " + str(lin_reg.coef_))

plt.scatter(boston_inputs, boston_targets)
plt.plot(boston_inputs, boston_outputs, c="orange")
plt.xlabel("% lower status of the population")
plt.ylabel("Median value of homes in $1000's")
plt.title('Regression of % lower status vs median home value')
plt.show()

residuals = boston_targets-boston_outputs
plt.scatter(boston_inputs, residuals, c="red")
plt.title('Residuals from regression of % lower status vs median home value')
plt.show()
"""

"""
# Linear Regression the numpy way, for comparison:

inputs = boston.as_matrix(columns=['LOW STATUS'])
inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
targets = boston.as_matrix(columns=['MEDIAN VALUE'])

weights = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)
outputs = np.dot(inputs,weights)
error = np.sum((targets-outputs)**2)

print("MSE using LOW STATUS (numpy way): " +str(error))
"""

def findWeights(inputs, target):
    boston_inputs = boston[inputs]
    boston_targets = boston[target]
    
    lin_reg.fit(boston_inputs, boston_targets)
    
    print("Weights/Coefficients of Regression: " + str(lin_reg.coef_))
    
def findMAPE(target, outputs): 
    boston_targets = boston[target]
    
    boston_targets, outputs = np.array(boston_targets), np.array(outputs)
    mape = np.mean(np.abs((boston_targets - outputs) / boston_targets)) * 100
    
    return mape


#1: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS as the input
print("Linear Regression:")
findWeights(['LOW STATUS'], ['MEDIAN VALUE'])

boston_inputs = boston[ ['LOW STATUS']]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))

#2: Predict MEDIAN VALUE in the Boston Housing dataset using ROOMS as the input
print("Linear Regression:")
findWeights(['ROOMS'], ['MEDIAN VALUE'])

boston_inputs = boston[ ['ROOMS'] ]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))

#3: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS AND Rooms as inputs.
print("Linear Regression:")
findWeights(['LOW STATUS', 'ROOMS'], ['MEDIAN VALUE'])

boston_inputs = boston[ ['LOW STATUS','ROOMS'] ]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))


#4: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS and LOW STATUS^2 as inputs
print("Linear Regression:")
boston['LOW STATUS SQUARED'] = boston['LOW STATUS']**2
findWeights(['LOW STATUS', 'LOW STATUS SQUARED'], ['MEDIAN VALUE'])

boston_inputs = boston[['LOW STATUS', 'LOW STATUS SQUARED']]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))

#5: Predict MEDIAN VALUE in the Boston Housing dataset using ROOMS and ROOMS^2 as inputs.
print("Linear Regression:")
boston['ROOMS SQUARED'] = boston['ROOMS']**2
findWeights(['ROOMS', 'ROOMS SQUARED'], ['MEDIAN VALUE'])

boston_inputs = boston[['ROOMS', 'ROOMS SQUARED']]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))

#6: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS, LOW STATUS^2, ROOMS, and ROOMS^2 as inputs
print("Linear Regression:")
findWeights(['ROOMS', 'ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED'], ['MEDIAN VALUE'])

boston_inputs = boston[['ROOMS', 'ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED']]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))


#7: Predict MEDIAN VALUE in the Boston Housing dataset using LOW STATUS, LOW STATUS^2, ROOMS, ROOMS^2, 
#   AND 'LOWROOMS' as inputs. LOWROOMS is an interaction term: LOW STATUS * ROOMS.
print("Linear Regression:")
boston['LOWROOMS'] = boston['LOW STATUS'] * boston['ROOMS']
findWeights(['ROOMS','ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED','LOWROOMS'], ['MEDIAN VALUE'])

boston_inputs = boston[['ROOMS', 'ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED','LOWROOMS']]
boston_outputs = lin_reg.predict(boston_inputs)
mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
print("MAPE (Mean Absolute Percentage Error): " + str(mape))
boston_mse = mean_squared_error(boston_targets, boston_outputs)
print("MSE (scikit way): " + str(boston_mse*len(boston)))

#8: Starting with the inputs listed in #6, add each of the other inputs (only one at a time) 
#and see if any of them have a greater than 3% improvement on the overall error. 
#For any that do, report the input name and the new error.

def findNewError(inputs, oldmape):
    boston_inputs = boston[['ROOMS', 'ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED']+inputs]
    
    boston_targets = boston['MEDIAN VALUE']
    lin_reg.fit(boston_inputs, boston_targets)
    
    boston_outputs = lin_reg.predict(boston_inputs)
    mape = findMAPE(['MEDIAN VALUE'], boston_outputs)
    percent_difference = (mape - oldmape)/oldmape * 100
    
    return percent_difference


boston_inputs = boston[['ROOMS', 'ROOMS SQUARED','LOW STATUS','LOW STATUS SQUARED']]
boston_targets = boston['MEDIAN VALUE']

lin_reg.fit(boston_inputs, boston_targets)
boston_outputs = lin_reg.predict(boston_inputs)

old_mape = findMAPE(['MEDIAN VALUE'], boston_outputs)

print("crime rate error %: ", findNewError(['CRIME RATE'], old_mape))
print("industry error %: ", findNewError(['INDUSTRY'], old_mape))
print("student teacher ratio error %: ", findNewError(['STU TEACH RATIO'], old_mape))
print("african american error %: ", findNewError(['AFR AMER'], old_mape))
print("river error %: ", findNewError(['RIVER'], old_mape))
print("nox error %: ", findNewError(['NOX'], old_mape))
print("large lot error %: ", findNewError(['LARGE LOT'], old_mape))
print("prior 1940 error %: ", findNewError(['PRIOR 1940'], old_mape))
print("emp distance error %: ", findNewError(['EMP DISTANCE'], old_mape))
print("highway access error %: ", findNewError(['HWY ACCESS'], old_mape))
print("property tax rate error %: ", findNewError(['PROP TAX RATE'], old_mape))


#9: What's the best error rate you can get with any set of columns? Which columns did you use?
#the best error rate we could get is #7 (16.1%) 
#columns used: OW STATUS, LOW STATUS^2, ROOMS, ROOMS^2, AND 'LOWROOMS'


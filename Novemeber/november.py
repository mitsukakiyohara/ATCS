import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

#author: mitsukakiyohara
#block: D

mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

test_targets = mnist_test[0]
test_inputs = mnist_test[mnist_test.columns[1:]]

#shuffle training set 
from sklearn.utils import shuffle
train_targets, train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

"""
Find the values for max_depth and min_samples_leaf  that produces the best Mean accuracy rating for a Decision Tree 
on the MNIST test data set. Print the values, and accuracy and confusion matrix resulting from those values. 
"""

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 25, min_samples_leaf = 50)

for i in range(1, 6): 
    
    tree.fit(train_inputs, train_targets)
    print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())
    print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
    train_outputs = tree.predict(train_inputs)
    print("Confusion Matrix (train):")
    print(confusion_matrix(train_targets, train_outputs))
    print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
    test_outputs = tree.predict(test_inputs)
    print("Confusion Matrix (test):")
    print(confusion_matrix(test_targets, test_outputs))

    tree = DecisionTreeClassifier(max_depth = 25 + i, min_samples_leaf = 50 + 5*i)

    
"""
Test cases I tried: (every run, I increased depth by 1, increased leaf by 5)

max depth = 10, min_samples_leaf = 40 
1st run: train = 0.856, test = 0.846
2nd run: train = 0.86, test = 0.8473 **
3rd run: train = 0.858, test = 0.8463
4th run: train = 0.858, test = 0.8463
5th run: train = 0.854, test = 0.8421

max depth = 20, min_samples_leaf = 40
1st run: train = 0.86708, test = 0.852 *** best accuracy so far
2nd run: train = 0.863, test = 0.8492 **
3rd run: train = 0.8587, test = 0.8467
4th run: train = 0.85475, test = 0.8421
5th run: train = 0.851, test = 0.8391

max_depth = 20, min_samples_leaf = 80
1st run: train = 0.8368, test = 0.8284
2nd run: train = 0.8629, test = 0.8492 **
3rd run: train = 0.8587, test = 0.8467
4th run: train = 0.85475, test = 0.8421
5th run: train = 0.851, test = 0.8391

max_depth = 30, min_samples_leaf = 40
1st run: train = 0.86708, test = 0.8518 **
2nd run: train = 0.8629, test = 0.8492 **
3rd run: train = 0.8587, test = 0.8467
4th run: train = 0.85475, test = 0.8421
5th run: train = 0.851, test = 0.8391

max_depth = 25, min_samples_leaf = 50
1st run: train = 0.8587, test = 0.8467 
2nd run: train = 0.85475, test = 0.8421 
3rd run: train = 0.8511, test = 0.8391
4th run: train = 0.8427, test = 0.8346
5th run: train = 0.8429, test = 0.8339

"""

"""
Using the same inputs you used with the Perceptron, does a Decision Tree do better or worse at predicting who survived
the Titanic? Print out  what values for max_depth and min_samples_leaf that you used and why those values, 
as well as your accuracy and conclusion.
"""

titanic_train = pd.read_csv("titanic_train2.csv")
titanic_test = pd.read_csv("titanic_test2.csv")

titanic_train_inputs = titanic_train.drop(titanic_train['Name', 'PassengerId', 'Cabin', 'Ticket'], axis=1)
titanic_train_targets = titanic_test[titanic_test.columns[0]]

#delete not useful columns
del titanic_test['Name']
del titanic_test['PassengerId']
del titanic_test['Cabin']
del titanic_test['Ticket']


titanic_test_targets = titanic_test[titanic_test.columns[0]]
titanic_test_inputs = titanic_test[titanic_test.columns[2:]]


cols = titanic_train_inputs.loc[:, titanic_train_inputs.dtypes == object]
for col in cols:
    titanic_train_inputs = pd.concat([titanic_train_inputs, pd.get_dummies(titanic_train_inputs[col])], axis=1)
    del titanic_train_inputs[col]

#shuffle training set
from sklearn.utils import shuffle
train_targets, train_inputs = shuffle(titanic_train_targets, titanic_train_inputs, random_state=42)
    
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 40)

for i in range(1, 6): 
    
    tree.fit(train_inputs, train_targets)
    print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())
    print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
    titanic_train_outputs = tree.predict(train_inputs)
    print("Confusion Matrix (train):")
    print(confusion_matrix(train_targets, train_outputs))
    print("Mean test accuracy: ", tree.score(titanic_test_inputs, titanic_test_targets))
    titanic_test_outputs = tree.predict(titanic_test_inputs)
    print("Confusion Matrix (test):")
    print(confusion_matrix(titanic_test_targets, titanic_test_outputs))

    tree = DecisionTreeClassifier(max_depth = 20 + i, min_samples_leaf = 40 + 5*i)


"""
Use Logistic Regression, Decision Trees, and K Nearest Neighbors to detect Pulsars from background noise 
in radio signal data. Which model performs best? worst? Why might that be? Don't forget that Decision Trees and 
K Nearest Neighbors need some tuning. Print your final mean accuracy and confusion matrix for each model, as 
well as any model hyper-parameters you used.
"""

pulsar_train = pd.read_csv("pulsar_train.csv", header=None)
pulsar_test = pd.read_csv("pulsar_test.csv", header=None)

pulsar_train_targets = pulsar_train[pulsar_train.columns[8]]
pulsar_train_inputs = pulsar_train[pulsar_train.columns[1:8]]

pulsar_test_targets = pulsar_test[pulsar_test.columns[8]]
pulsar_test_inputs = pulsar[pulsar_train.column[1:8]]

#Using Multinomial Logisitc Regresssion
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

print("Training a Multinomial Logistic Regression classifier for all inputs")
softmax_reg.fit(pulsar_train_inputs, pulsar_train_targets)
softmax_outputs = softmax_reg.predict(pulsar_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(pulsar_train_inputs, pulsar_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(pulsar_train_targets, softmax_outputs))



#Using Decision Trees
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 25, min_samples_leaf = 50)

from sklearn.utils import shuffle
train_targets, train_inputs = shuffle(pulsar_train_targets, pulsar_train_inputs, random_state=42)

for i in range(1, 6): 
    
    tree.fit(train_inputs, train_targets)
    print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())
    print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
    train_outputs = tree.predict(train_inputs)
    print("Confusion Matrix (train):")
    print(confusion_matrix(train_targets, train_outputs))
    print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
    test_outputs = tree.predict(test_inputs)
    print("Confusion Matrix (test):")
    print(confusion_matrix(test_targets, test_outputs))

    tree = DecisionTreeClassifier(max_depth = 25 + i, min_samples_leaf = 50 + 5*i)


#Using K Nearest Neighbors 
from sklearn.neighbors import KNeighborsClassifier

print("K Nearest Neighbors classifier")
knn = KNeighborsClassifier() 

knn.fit(train_inputs, train_targets)

outputs = knn.predict(pulsar_test_inputs)
print("Mean test accuracy:", knn.score(pulsar_test_inputs, pulsar_test_targets))
print_conf_matrix(pulsar_test_targets, outputs)
print("Precision = TP / (TP + FP) = ", precision_score(pulsar_test_targets, outputs))
print("Recall = TP / (TP + FN) = ", recall_score(pulsar_test_targets, outputs))



"""
Use Logistic Regression, Decision Trees, and K Nearest Neighbors to detect malignant cancer cells in breast tissue.  
NOTE: a '4' in the Class column represents the presence of malignant cancer cells, and there may be a couple of rows 
with missing data.
Which model performs best? worst? Why might that be? Don't forget that Decision Trees and K Nearest Neighbors need 
some tuning. Print your final mean accuracy and confusion matrix for each model, as well as any model hyper-parameters you used.
"""

cancer_train = pd.read_csv("cancer_train.csv", header=None)
cancer_test = pd.read_csv("cancer_test.csv", header=None)

cancer_train_targets = cancer_train[10]
cancer_train_inputs = cancer_train[cancer_train.columns[1:10]]

#Using Multinomial Logisitc Regresssion
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

print("Training a Multinomial Logistic Regression classifier for all inputs")
softmax_reg.fit(cancer_train_inputs, cancer_train_targets)
softmax_outputs = softmax_reg.predict(cancer_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(cancer_train_inputs, cancer_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(cancer_train_targets, softmax_outputs))


#Using Decision Trees
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 25, min_samples_leaf = 50)

from sklearn.utils import shuffle
train_targets, train_inputs = shuffle(cancer_train_targets, cancer_train_inputs, random_state=42)

for i in range(1, 6): 
    
    tree.fit(train_inputs, train_targets)
    print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())
    print("Mean train accuracy: ", tree.score(train_inputs, train_targets))
    train_outputs = tree.predict(train_inputs)
    print("Confusion Matrix (train):")
    print(confusion_matrix(train_targets, train_outputs))
    print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
    test_outputs = tree.predict(test_inputs)
    print("Confusion Matrix (test):")
    print(confusion_matrix(test_targets, test_outputs))

    tree = DecisionTreeClassifier(max_depth = 25 + i, min_samples_leaf = 50 + 5*i)


#Using K Nearest Neighbors 
from sklearn.neighbors import KNeighborsClassifier

print("K Nearest Neighbors classifier")
knn = KNeighborsClassifier() 

knn.fit(train_inputs, train_targets)

outputs = knn.predict(cancer_test_inputs)
print("Mean test accuracy:", knn.score(cancer_test_inputs, cancer_test_targets))
print_conf_matrix(cancer_test_targets, outputs)
print("Precision = TP / (TP + FP) = ", precision_score(cancer_test_targets, outputs))
print("Recall = TP / (TP + FN) = ", recall_score(cancer_test_targets, outputs))



    
    




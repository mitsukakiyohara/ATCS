import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Helper functions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

__author__ = "Mitsuka Kiyohara"
#Block D | AT Comp Sci

def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def print_class_results(targets, outputs):
	print_conf_matrix(targets, outputs)

	# Precision - How accurate are the positive predictions?
	print("Precision (TP / (TP + FP)):", precision_score(targets, outputs))

	# Recall - How correctly are positives predicted?
	print("Recall (TP / (TP + FN)):", recall_score(targets, outputs))


"""
# Logistic Regression (even though it is a classifier)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]

# Now let's shuffle the training set to reduce bias opportunities
from sklearn.utils import shuffle
smn_train_targets, smn_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

# Let's try our Logistic Classifier on the MNIST data, predicting digit 5
from sklearn.linear_model import LogisticRegression

smn_train_target5 = (smn_train_targets==5)
smn_test_target5 = (mnist_test_targets==5)

print("Training...")
log_reg.fit(smn_train_inputs, smn_train_target5)
smn_train_outputs5 = log_reg.predict(smn_train_inputs)
# Classification error metrics:
print("Mean accuracy:", log_reg.score(smn_train_inputs, smn_train_target5))
print_class_results(smn_train_target5, smn_train_outputs5)

# But how does it perform on the test set?
print()
print("And on the test set...")
smn_test_outputs5 = log_reg.predict(mnist_test_inputs)
print("Mean accuracy:", log_reg.score(mnist_test_inputs, smn_test_target5))
print_class_results(smn_test_target5, smn_test_outputs5)
"""

# L1 Multinomial Logistic Regression 
print("Multinomial Logistic Regression (L1)")
softmax_reg = LogisticRegression(penalty="l1",multi_class="multinomial", solver="saga")
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))

# L2 Multinomial Logistic Regression 
print("Multinomial Logistic Regression (L2)")
softmax_reg = LogisticRegression(penalty="l2",multi_class="multinomial", solver="saga")
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))

#input 1:  total amt of ink 
#decreased its accuracy 
mnist_train['Total'] = mnist_train.apply(sum, axis = 1)

mnist_train_inputs = mnist_train[mnist_train.columns[2:]]
softmax_outputs = softmax_reg.predict(mnist_train_inputs)

print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))

#input 2: symmetry 
#reshape, compare each column, find the difference between the values of 2 pixels 
#-- not fully done -- 
print(raw_digits.head())
def transpose(mat, tr, N): 
    for i in range(N): 
        for j in range(N): 
            tr[i][j] = mat[j][i] 

            
#mat: image = row.to_numpy().reshape(28, 28)
def isSymmetric(mat, N):
    tr = [ [0 for j in range(len(mat[0])) ] for i in range(len(mat)) ] 
    transpose(image, tr, N) 
    diff = mat - tr 
    return(sum(diff*diff))

image = raw_digits.iloc[1].to_numpy().reshape(28, 28)
isSymmetric(image, 1)


    

#input 3: looking only at top 1/3 ink
mnist_train['Total'] = raw_digits.apply(sum, axis = 1)
mnist_train_inputs = mnist_train[mnist_train.columns[1:261]]

softmax_outputs = softmax_reg.predict(mnist_train_inputs)

print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))   

#input 4: got loops? 
#basic structure -- need to write floodfill 
def gotLoops(row): 
    grid = row.to_numpy().reshape(28,28)
    floodfill(grid)
    
    if 0 in grid: 
        return 1
    else:
        return 0 
gotLoops(mnist_train_inputs)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__   = "Mitsuka Kiyohara"
#Block D, AT Computer Science

#Load in data set and name columns
mushroom_data = pd.read_csv("mushrooms.csv", header=None, names=['E/P','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']) 

mushroom_targets = mushroom_data[['E/P']]


#Complete a Stratified Shuffle Split to dataset in 80/20 ratio
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)
sss.get_n_splits(mushroom_data)

for train_index, test_index in sss.split(mushroom_data, mushroom_targets):
    mushroom_train_set = mushroom_data.loc[train_index]
    mushroom_test_set = mushroom_data.loc[test_index]


#Split the training and test datasets into inputs and targets dataframes
mushroom_train_inputs = mushroom_train_set.drop(['E/P'], axis=1)
mushroom_train_targets = mushroom_train_set[['E/P']]

mushroom_test_inputs = mushroom_test_set.drop(['E/P'], axis=1)
mushroom_test_targets = mushroom_test_set[['E/P']]

#Encode the categorical input columns and print the total number of columns from each encoding
train_inputs = mushroom_train_set.drop('E/P', axis=1)

#One-Hot Encoding (scikit func)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
print("Number of columns from One Hot Encoding: 117")
print(pd.DataFrame(onehotencoder.fit_transform(train_inputs).toarray()))

#Ordinal Encoding (scikit func)
test_inputs = mushroom_test_set.drop('E/P', axis=1)
train_inputs = mushroom_train_set.drop('E/P', axis=1)

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
print("Number of columns from Ordinal Encoding: 22")
print(pd.DataFrame(enc.fit_transform(train_inputs, test_inputs)))

from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

""" 
What the Binary Encoder Class Looks Like: 

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.base_n_encoder = ce.BaseNEncoder(base=2, verbose=self.verbose, cols=self.cols, mapping=self.mapping,
                                              drop_invariant=self.drop_invariant, return_df=self.return_df,
                                              handle_unknown=self.handle_unknown, handle_missing=self.handle_missing)
    
    def fit(self, X, y=None, **kwargs):
        self.base_n_encoder.fit(X,y,**kwargs)
        return self
    
    def transform(self, X, override_return_df=False):
        return self.base_n_encoder.transform(X)
"""

#Binary Encoding 
test_inputs = mushroom_test_set.drop('E/P', axis=1)
train_inputs = mushroom_train_set.drop('E/P', axis=1)

binary = BinaryEncoder()
binary.fit(train_inputs, test_inputs)
print("Number of columns from Binary Encoding: 76")
print(pd.DataFrame(binary.transform(train_inputs, test_inputs)))


#For each encoding, a timed fit is done for each of three models: Logistic Regression, Decision Tree, K Nearest Neighbors. 
#For each model, it will predict and print the time elapsed, mean accuracy, confusion matrix, and precision/recall for each of the 9 combinations.
log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

#Confusion Metrix (function)
def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

#Class Results: Confusion Metrix, Precision and Recall (function)
def print_class_results(targets, outputs):
    print_conf_matrix(targets, outputs)

    # Precision - How accurate are the positive predictions?
    print("Precision (TP / (TP + FP)):", precision_score(targets, outputs))

    # Recall - How correctly are positives predicted?
    print("Recall (TP / (TP + FN)):", recall_score(targets, outputs))

#Timed fit (function)
import time
def print_timed_fit(model, train_inputs, train_targets):
    start = time.perf_counter()
    model.fit(train_inputs, train_targets)
    outputs = model.predict(train_inputs)
    stop = time.perf_counter()
    elapsed = stop - start
    
    print("Timed elapsed: ", elapsed)

models = [log_reg, tree, knn] #three models we're using: Logistic Regression, Decision Tree, and K-Nearest Neighbors
encoders = [onehotencoder, enc, binary] #three encoders we're using: One Hot Encoder, Ordinal Encoder, Binary Encoder

#nested for loop (testing each model with each encoder)
for model in models:
    print("Testing with: ", model)
    #call for train_targets and new train_inputs
    train_targets = mushroom_train_set['E/P'].apply(lambda x : 0 if x == 'e' else 1)
    train_inputs = mushroom_train_set.drop('E/P', axis=1)
    
    for encode in encoders:
        print("Testing with: ", encode)
        #encode data  
        new_mushroom_data = pd.DataFrame(encode.fit_transform(train_inputs).toarray())
        train_inputs = new_mushroom_data
 
    model.fit(train_inputs, train_targets)
    test_outputs = model.predict(train_inputs)

    #timed fit
    print_timed_fit(model, train_inputs, train_targets)
    #confusion matrix + precision/recall
    print_class_results(train_targets, test_outputs)
    #mean accuracy
    print("Mean Accuracy:", model.score(train_inputs, train_targets))

    
    
    
    
    





# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 15:54:40 2017

@author: alexj
"""
#------------------------------------------------------------------------------
# ReadMe
# Pipeline to predict T2D associative loci
# Data matrix (loci x in regulatory features)
#------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from IPython.display import Image  

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

start = time.time()

#------------------------------------------------------------------------------
# Select method
#------------------------------------------------------------------------------
estimators = ['logistic']
model = 'logistic'
num_folds = 3

print("Estimator: %s" % model)

#------------------------------------------------------------------------------
# import data matrix
#------------------------------------------------------------------------------

# Training (development) set
#file_train = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/10.31.16_T2D_noHLA_all_bed_table_grouped_train.txt"
file_train = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_sample/sample_train.txt"
data_train = pd.read_table(file_train,sep='\t',header=(0))
data_train = pd.DataFrame(data_train)
print("Training set: %s" % file_train)


# Test (hold out) set
#file_test = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/holdout_basemodel_bed_table_grouped_reformat.txt"
file_test = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_sample/sample_test_match.txt"
data_test = pd.read_table(file_test,sep='\t',header=(0))
data_test = pd.DataFrame(data_test)
print("Test set: %s" % file_test)
print('\n')

assert(len(data_train.columns) == len(data_test.columns))

nCols = len(data_train.columns)
features = list(data_train.columns)[2:nCols]
label = data_train.columns[1]

# Transform string labels into float (0.0, 1.0)
train_labels = data_train.loc[:,label]
test_labels_actual = data_test.loc[:,label]
#train_labels = map(lambda x: 1.0 if x == 'index' else 0.0, train_labels)
#test_labels_actual = map(lambda x: 1.0 if x == 'index' else 0.0, test_labels_actual)

# Transform boolean into float (0.0, 1.0)
train_markers = data_train.loc[:,features]
train_markers = train_markers.astype(np.float)
test_markers = data_test.loc[:,features]
test_markers = test_markers.astype(np.float)

#------------------------------------------------------------------------------
# K-fold cross validation and hyper parameter tuning suing GridSearchCV:
#
# 1.  Pick a set of hyper-parameters from the grid (representing the possible 
#     hyper-parameters you wish to consider)
# 2.  Train model to fit parameters using k-fold CV.  Average the performance 
#     on the validated set across the k trials
# 3.  Repeat 1 and 2 across all grid points.  Each grid point will have a 
#     corresponding average performance score
# 4.  Select best average score
# 5.  Using the corresponding set of hyper-parameters associated with the best 
#     average score, refit the model to retrieve the estimated parameters
#
#------------------------------------------------------------------------------
# Set the parameters by cross-validation
if model == 'logistic':
    
    # C is the inverse regularization parameter (1/lambda).  Lower C = high regularization.
    # Lambda (regularization) is the penalty against the complexity of the model 
    # as we want to avoid overfitting.
    # This lambda term is added to the cost function in order to penalize
    # higher weighted coefficients.
    lambda_ = np.linspace(0.0015, 0.003, 10)
    tuned_parameters = {'penalty':['l1'], 'C':1/lambda_, 'fit_intercept':[True, False],
                        'random_state':[100]}

#Specify scoring metric to use for truning hyper-parameters: 
# accuracy rate of correctly labeled = (TP+TN)/(Total)
# precision is the rate of TP = TP/(TP+FP)
# recall is the sensitivity score = TP/(TP+FN) 
scores = ['accuracy'] 


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('\n')

    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    if model == 'logistic':
        estimator = LogisticRegression()        
    
    clf = GridSearchCV(estimator, tuned_parameters, cv=num_folds, scoring=score)    
    clf.fit(train_markers, train_labels)

    # Best hyper-parameter set selected from Grid Search
    print("Best hyper-parameter set found on training set:")
    print('\n')
    print(clf.best_params_)
    print('\n')
    
    #Best model parameters
    if model == 'logistic':
        print("Best parameters (coefficients) found on training set:")
        print(clf.best_estimator_.coef_)
        
    print('\n')
    print("Grid scores on training set:")
    print('\n')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('\n')

    print("Detailed classification report:")
    print('\n')
    print("Performance of the best model on the test set")
    print('\n')

    
    y_test = np.array(test_labels_actual)
    y_true, y_pred = y_test, clf.predict(test_markers)
    print(classification_report(y_true, y_pred))
    print('\n')
    print(clf.best_estimator_.coef_)
    
##------------------------------------------------------------------------------
## Performance statistics
## Think about visualizations********************
##------------------------------------------------------------------------------    


   
end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('-------------%d:%d:%d-----------------' %(hours,minutes,seconds)) 
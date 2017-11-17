# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 15:54:40 2017

@author: alexj
"""
#------------------------------------------------------------------------------
# ReadMe
# Pipeline to predict T2D associative loci
# Data matrix (loci x regulatory features)
# 
# Implementation for a single random seed
#------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score
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
num_folds = 10

print("Estimator: %s \n" % model)

#------------------------------------------------------------------------------
# import data matrix
#------------------------------------------------------------------------------

# Training (development) set
file_train = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/10.31.16_T2D_noHLA_all_bed_table_grouped.txt"
data_train = pd.read_table(file_train,sep='\t',header=(0))
data_train = pd.DataFrame(data_train)
print("Training set: %s" % file_train)


# Test (hold out) set
file_test = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/holdout_basemodel_bed_table_grouped_rm_tr_ctrls_reformat.txt"
data_test = pd.read_table(file_test,sep='\t',header=(0))
data_test = pd.DataFrame(data_test)
print("Test set: %s \n" % file_test)

assert(len(data_train.columns) == len(data_test.columns))

nCols = len(data_train.columns)
features = list(data_train.columns)[2:nCols]
label = data_train.columns[1]

# Transform string labels ('index', 'control') into float (0.0, 1.0)
train_labels = data_train.loc[:,label]
test_labels_actual = data_test.loc[:,label]
train_labels = map(lambda x: 1.0 if x == 'index' else 0.0, train_labels)
test_labels_actual = map(lambda x: 1.0 if x == 'index' else 0.0, test_labels_actual)

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

    tuned_parameters = {'C': np.linspace(1e-10,1, 20), 'penalty':['l1'], 
                        'random_state':[100]}

#------------------------------------------------------------------------------
# Specify scoring metric to use for truning hyper-parameters: 
# f1:  average of the predicted positive rate (recall) and TP rate (precision) 
# rates so it is mostly concerned with the positive cases (as opposed to the controls)
# roc_auc: area under the ROC curve, which is the tradeoff between the ratio of 
# the TP rate  and FP rate
#------------------------------------------------------------------------------ 

scores = ['f1','roc_auc'] 

fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()
average_precision = dict()

for i in range(len(scores)):
    print("# Tuning hyper-parameters for %s" % scores[i])
    print('\n')

    #--------------------------------------------------------------------------
    # Fit model
    #--------------------------------------------------------------------------
    if model == 'logistic':
        estimator = LogisticRegression()        
    
    clf = GridSearchCV(estimator, tuned_parameters, cv=num_folds, scoring=scores[i])    
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
   

    #--------------------------------------------------------------------------
    # Performance of model
    #--------------------------------------------------------------------------       
    print("Performance of the best model on the test set")
    print('\n')
    y_test = np.array(test_labels_actual)
    y_true, y_pred = y_test, clf.predict_proba(test_markers)

    #--------------------------------------------------------------------------
    # Compute ROC curve and AUC
    # ROC curve indicates how the cost:benefit ratio (measured by the radeoff between 
    # TP (probability of detection - sensitivity)and FP (probability of false alarm) 
    # changes as the decision boundary for a given classifier varies
    # In order to generate this ROC curve, first convert your classifier output labels 
    # to predicted probability scores (sklearn has function 'predict_proba')
    # Then, set a constant threshold to determine the predicted cutoff of positive and 
    # negative and compare these predictions to the true labels to get values for the confusion matrix.
    # Vary threshold and repeat to get points on the curve
    #
    #
    # Compute precision-recall curve
    # A high area under the curve represents both high recall and high precision
    # Precision (TP/TP+FP):  Percentage of positive results that were identified correctly 
    # Recall (TP/TP+FN):  Percentage of positive results that are identified by our method
    # Note:
    # High Precision, low recall: Very few positive results, but most of its predicted positive labels are correct
    # Low precision, high recall: Many positive results, but most of its predicted labels are incorrect                                         
    #------------------------------------------------------------------------- 
    
    fpr[i], tpr[i], thresholds_roc = roc_curve(y_true, y_pred[:,1])
    precision[i], recall[i], thresholds_pr = precision_recall_curve(y_true, y_pred[:,1])
    average_precision[i] = average_precision_score(y_test, y_pred[:,1])
    roc_auc[i] = auc(fpr[i], tpr[i])
 
    #-------------------------------------------------------------------------   
    # Output important features
    #-------------------------------------------------------------------------
    if model == 'logistic':
        trained_coeff = pd.DataFrame(zip(clf.best_estimator_.coef_[0], features), columns=["Weight", "Feature"])
        trained_coeff.to_csv("C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/coeff.csv", sep=',')

#------------------------------------------------------------------------------   
# Plotting
#------------------------------------------------------------------------------        
colors = ["darkorange", "magenta"]
lw = 2

plt.figure()
for i in range(len(scores)):
    plt.plot(fpr[i], tpr[i], color=colors[i],lw=lw, label='(area = %0.2f) using %s for tuning' % (roc_auc[i], scores[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC using %s model' % model)
    plt.legend(loc="lower right")
plt.savefig("C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/ROC_combined.jpg")
plt.show()

plt.figure(2)
for i in range(len(scores)):
    plt.plot(recall[i], precision[i], lw=lw, color=colors[i], label='(average precision = %0.2f) using %s for tuning' % (average_precision[i], scores[i]))      
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class precision-recall curve using %s model' % model)
    plt.legend(loc="upper right")
plt.savefig("C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/Precision_recall_combined.jpg")
plt.show()
    
 
end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('-------------%d:%d:%d-----------------' %(hours,minutes,seconds)) 

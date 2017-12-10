# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:23:40 2017

@author: alexj
"""

#------------------------------------------------------------------------------
# ReadMe
# Read in .csv files of trained weights for all features from multiple estimators
# Plot histogram of weights (logistic regression) and importance score (random forest)
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.plotly as py

#------------------------------------------------------------------------------
# import data matrix
#------------------------------------------------------------------------------

# Training (development) set
file_LR = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/coeff_multEstimators/coeff_LR_roc_auc.csv"
data_LR = pd.read_table(file_LR,sep=',',header=(0))
data_LR = pd.DataFrame(data_LR)
print("Logistic regression result: %s" % file_LR)
LR_weights = data_LR.loc[:,'Weight']

plt.hist(LR_weights, bins=100)
plt.ylim((0,50))
plt.title("Logistic Regression Weights Histogram")
plt.xlabel("Weights")
plt.ylabel("Frequency")


#plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

# Test (hold out) set
file_RF = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/coeff_multEstimators/coeff_RF_roc_auc.csv"
data_RF = pd.read_table(file_RF,sep=',',header=(0))
data_RF = pd.DataFrame(data_RF)
print("Random Forest result: %s \n" % file_RF)
RF_score = data_RF.loc[:,'Score']

plt.hist(RF_score, bins=100)
#plt.ylim((0,10))
plt.title("Random Forest Scores Histogram")
plt.xlabel("Importance score")
plt.ylabel("Frequency")

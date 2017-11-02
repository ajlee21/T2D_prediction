# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:43:17 2017

@author: alexj
"""
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------
# import my data matrix
#------------------------------------------------------------------------------

# Training (development) set
file_train = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/10.31.16_T2D_noHLA_all_bed_table_grouped.txt"
#file_train = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_sample/sample_train.txt"
data_train = pd.read_table(file_train,sep='\t',header=(0))
data_train = pd.DataFrame(data_train)
print("Training set: %s" % file_train)


# Test (hold out) set
file_test = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/holdout_basemodel_bed_table_grouped_rm_tr_ctrls_reformat.txt"
#file_test = "C:/Users/alexj/Documents/UPenn/BVoight/dataset_sample/sample_test_match.txt"
data_test = pd.read_table(file_test,sep='\t',header=(0))
data_test = pd.DataFrame(data_test)
print("Test set: %s" % file_test)
print('\n')

#read in kim data and overlay


train_means = data_train.groupby('snp_group').agg(np.mean)
test_means = data_test.groupby('snp_group').agg(np.mean)


r_control, p_control = scipy.stats.pearsonr(train_means.loc['control',:], test_means.loc['control',:])
r_index, p_index = scipy.stats.pearsonr(train_means.loc['index',:], test_means.loc['index',:])
print('Pearson correlation score for control group between training and test sets is %f' % r_control)
print('Pearson correlation score for index group between training and test sets is %f'% r_index)

plt.figure(1)
plt.scatter(train_means.loc['control',:].values, test_means.loc['control',:].values)
plt.xlabel('training control mean')
plt.ylabel('test control mean')
plt.savefig("C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/control_mean.jpg")

plt.figure(2)
plt.scatter(train_means.loc['index',:].values, test_means.loc['index',:].values)
plt.xlabel('training index mean')
plt.ylabel('test index mean')
plt.savefig("C:/Users/alexj/Documents/UPenn/BVoight/dataset_separate/index_mean.jpg")
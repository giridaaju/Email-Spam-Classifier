## Tweak baseline classifier to see how it can be improved
# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

'''
def aucCV(features, labels):
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('imputer', SimpleImputer(missing_values=-1, strategy='median')), 
                           ('svm', SVC(kernel='rbf', probability=True))])
    param_grid = {
        "svm__C": [0.25, 0.5, 1, 2.5, 5, 10],
        "svm__degree": [2, 3, 5, 7],
        "svm__gamma": ['scale', 'auto', 1, 3, 5]
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(features, labels)
'''

def aucCV(features,labels):
    # model = GaussianNB()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='median'), StandardScaler(),
                          SVC(kernel='rbf', C=0.25, degree=2, probability=True, gamma='scale'))
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
    
    return scores


def predictTest(trainFeatures,trainLabels,testFeatures):
    # model = GaussianNB()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='median'), StandardScaler(), ## Changed imputation strategy
                          SVC(probability=True, kernel='rbf', C=0.25, degree=2, gamma='scale'))
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('C:/Users/anebe/VSCode Projects/CSDS_340/Case Study 1/data/spamTrain1.csv',delimiter=',') ## Chnage path 
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()
    

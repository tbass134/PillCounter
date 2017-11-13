
# coding: utf-8

# In[1]:


from __future__ import print_function
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT
from sklearn import svm
import sklearn.svm
import cPickle
import numpy as np


# In[2]:


def train(files):
    #extract feature
    features, classes, filenames = aF.dirsWavFeatureExtraction(files, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep)
    #normalize
    [featuresNorm, MEAN, STD] = aT.normalizeFeatures(features)
    [X, Y] = aT.listOfFeatures2Matrix(featuresNorm)
    #train using SVM
    clf = sklearn.svm.SVC(kernel = 'linear',  probability = True)        
    clf.fit(X, Y)
    return clf, MEAN, STD


# In[3]:


def saveTraining(Classifier, MEAN, STD, modelName):
    with open(modelName, 'wb') as fid:                                            # save to file
        cPickle.dump(Classifier, fid)            
    fo = open(modelName + "MEANS", "wb")
    cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()        


# In[4]:


files = ["audio/pills_1/","audio/pills_10/", "audio/pills_25/", "audio/pills_50/"]
clf, MEAN, STD = train(files)


# In[5]:


saveTraining(clf, MEAN, STD, "myMod")


# In[ ]:





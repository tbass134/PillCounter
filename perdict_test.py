
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
import re


# In[2]:


def loadSVModel(SVMmodelName):
    '''
    This function loads an SVM model either for classification or training.
    ARGMUMENTS:
        - SVMmodelName:     the path of the model to be loaded
        - isRegression:        a flag indigating whereas this model is regression or not
    '''
    try:
        fo = open(SVMmodelName+"MEANS", "rb")
    except IOError:
            print ("Load SVM Model: Didn't find file")
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = np.array(MEAN)
    STD = np.array(STD)

    COEFF = []
    with open(SVMmodelName, 'rb') as fid:
        SVM = cPickle.load(fid)    

    return(SVM, MEAN, STD)


# In[3]:


def perdict(files, file, modelName):
    #read audio file, convert to mono (if needed)
    [Fs, x] = audioBasicIO.readAudioFile(file)
    x = audioBasicIO.stereo2mono(x)
    
    if modelName:
        mtWin, mtStep, stWin, stStep = 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep
        Classifier, MEAN, STD = loadSVModel(modelName)
    else:
        with open(SVMmodelName, 'rb') as fid:
            Classifier = cPickle.load(fid)
            MEAN = cPickle.load(fo)
            STD = cPickle.load(fo)
            mtWin, mtStep, stWin, stStep = 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep


    #extract features from sample
    [MidTermFeatures, s] = aF.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * stWin), round(Fs * stStep))
    MidTermFeatures = MidTermFeatures.mean(axis=1)        # long term averaging of mid-term statistics
    curFV = (MidTermFeatures - MEAN) / STD                # normalization

    #predict
    result = Classifier.predict(curFV.reshape(1,-1))[0]
    prob = Classifier.predict_proba(curFV.reshape(1,-1))[0]
    s = files[int(result)]
    
    return re.findall(r'\d+',s)[0] + " pills", prob


# In[4]:


files = ["audio/pills_1/","audio/pills_10/", "audio/pills_25/", "audio/pills_50/"]
prediction = perdict(files, "audio/test/pills_50.wav", "myMod")
print("prediction {}".format(prediction))

# In[ ]:





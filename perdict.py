from __future__ import print_function

import pyaudio
import wave
 

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT
from sklearn import svm
import sklearn.svm
import cPickle
import numpy as np
import re

def record():
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	CHUNK = 1024
	RECORD_SECONDS = 10
	WAVE_OUTPUT_FILENAME = "file.wav"
	 
	audio = pyaudio.PyAudio()
	 
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
	                rate=RATE, input=True,
	                frames_per_buffer=CHUNK)
	print ("recording...")
	frames = []
	 
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)
	print ("finished recording")
	 
	 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	 
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()
	return WAVE_OUTPUT_FILENAME

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

def perdict(files, file, modelName):
    #read audio file, convert to mono (if needed)
    [Fs, x] = audioBasicIO.readAudioFile(file)
    x = audioBasicIO.stereo2mono(x)
    print("perfict")
    
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


fileName = record()
files = ["audio/pills_1/","audio/pills_10/", "audio/pills_25/", "audio/pills_50/"]
result = perdict(files, fileName, "myMod")
print(result)
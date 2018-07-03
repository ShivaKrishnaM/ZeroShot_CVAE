#=====================================================================# 
# Author : Shiva Krishna Reddy M , IIT Madras                         #
#                                                                     #
# You can use this code in any way you want.                          #
#                                                                     #
#=====================================================================#

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
# import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
#import cv2
import glob, os
from sklearn.preprocessing import normalize
from keras.preprocessing import image
from random import shuffle
import keras.backend.tensorflow_backend as KTF

# ================== LAB RESOURCES ARE LIMITED.. PLEASE HELP !! =================== #
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

#===================================================================#

path = '../../Datasets/CUB/'

# Loads features
X = (np.load(open(path + 'dataFeatures' , 'r')) )
Y_temp = np.load(open(path +'dataLabels' , 'r'))
Y = np.array([x-1 for x in Y_temp])
ATTR = np.load(open(path +'dataAttributes' , 'r'))

# ====================================================================#
# Change as per dataset...

noClasses = 200
trClasses = 150
teClasses = 50
noExs = X.shape[0]

#=====================================================================#
 # TEST TRAIN SPLIT
fp = open(path + 'allclasses.txt' ,'r')
allclasses = [x.split()[0] for x in fp.readlines()]
fp.close()

allLabels = {}
count = 0 
for cl in allclasses:
	allLabels[cl] = count
	count = count + 1

# Changed Here.. -1
testClasses = [allLabels[x.split()[0]] for x in open(path + 'testclasses_ps.txt').readlines()]
trainClasses = [allLabels[x.split()[0]] for x in open(path + 'trainclasses_ps.txt').readlines()]


# ============================================================================== #

trainDataX = []
trainDataLabels = [] 
trainDataAttrs = [] 

testDataX = []
testDataLabels = [] 
testDataAttrs = []


for ii in range(0,noExs):
	if(Y[ii] in trainClasses):
		trainDataX = trainDataX + [X[ii]]
		trainDataLabels = trainDataLabels + [Y[ii]]
		trainDataAttrs = trainDataAttrs + [ATTR[Y[ii]]]
	elif(Y[ii] in testClasses):
		testDataX = testDataX + [X[ii]]
		testDataLabels = testDataLabels + [Y[ii]]
		testDataAttrs = testDataAttrs + [ATTR[Y[ii]]]
	else:
		print 'Fatal Error... Please check code/data'
trainDataX = np.array(trainDataX)
trainDataLabels = np.array(trainDataLabels)
trainDataAttrs = np.array(trainDataAttrs)

testDataX = np.array(testDataX)
testDataLabels = np.array(testDataLabels)
testDataAttrs = np.array(testDataAttrs)

np.save(open('trainData' , 'w') , trainDataX)
np.save(open('trainLabels' , 'w') , trainDataLabels)
np.save(open('trainAttributes' , 'w') , trainDataAttrs)


np.save(open('testData' , 'w') , testDataX)
np.save(open('testLabels' , 'w') , testDataLabels)
np.save(open('testAttributes' , 'w') , testDataAttrs)



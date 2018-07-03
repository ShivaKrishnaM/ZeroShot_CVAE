from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.layers import Input, Dense, Lambda, merge, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import numpy as np
# import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
#import cv2
import glob, os
from sklearn.preprocessing import normalize
import random
import keras.backend.tensorflow_backend as KTF
from sklearn.metrics import accuracy_score
from sklearn import svm



# ================== LAB RESOURCES ARE LIMITED=================== #

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
# Some Constants

m = 50
n_x = 2048
n_y = 312
n_z = 50
interNo = n_x/4
n_epoch = 25
path = '../../Datasets/CUB/'
nSamples = 200
nTrain = 150
nTest = 50
# ============================================================ #

input_ic = Input(shape=[n_x+n_y], name = 'img_class' )
cond  = Input(shape=[n_y] , name='class')
temp_h_q = Dense(interNo, activation='relu')(input_ic)
h_q_zd = Dropout(rate=0.7)(temp_h_q)
h_q = Dense(interNo, activation='relu')(h_q_zd)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=[n_z], mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps

z = Lambda(sample_z)([mu, log_sigma])

# Depending on the keras version...
# z_cond = merge([z, cond] , mode='concat', concat_axis=1)
z_cond = concatenate([z, cond])

decoder_hidden = Dense(1024, activation='relu')
decoder_out = Dense(n_x, activation='linear')
h_p = decoder_hidden(z_cond)
reconstr = decoder_out(h_p)
vae = Model(inputs=[input_ic , cond], outputs=[reconstr])



encoder = Model(inputs=[input_ic , cond], outputs=[mu])


d_in = Input(shape=[n_z+n_y])
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.mean(K.square(y_pred - y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    #print 'kl : ' + str(kl)
    return recon + kl

encoder.summary()
decoder.summary()
vae.compile(optimizer='adam', loss=vae_loss)

# ======================================================================= #


trainData = np.load(open('trainData' , 'r'))
trainLabels = np.load(open('trainLabels' , 'r'))
trainLabelVectors = np.load(open('trainAttributes' , 'r'))

X_train = np.concatenate([trainData , trainLabelVectors], axis=1)

print 'Fitting VAE Model...'
vae.fit({'img_class' : X_train , 'class' : trainLabelVectors}, trainData, batch_size=m, nb_epoch=n_epoch)

# =========================== UNSEEN CLASSES ======================================#

 # TEST TRAIN SPLIT
fp = open(path + 'allclasses.txt' ,'r')
allclasses = [x.split()[0] for x in fp.readlines()]
fp.close()

allLabels = {}
count = 0 
for cl in allclasses:
	allLabels[cl] = count
	count = count + 1

testClasses = [allLabels[x.split()[0]] for x in open(path + 'testclasses_ps.txt').readlines()]
trainClasses = [allLabels[x.split()[0]] for x in open(path + 'trainclasses_ps.txt').readlines()]

ATTR = np.load(open(path+'dataAttributes'))
ATTR.shape

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# ==================================================




pseudoTrainData = []
pseudoTrainLabels =[]
pseudoTrainAttr = []
totalExs = len(testClasses)*nSamples

with sess.as_default():
	noise_gen = K.random_normal(shape=(totalExs, n_z), mean=0., stddev=1.).eval()

for tc in testClasses:
	for ii in range(0,nSamples):
		pseudoTrainAttr.append(ATTR[tc])
		pseudoTrainLabels.append(tc)
# ===================================================


pseudoTrainAttr = np.array(pseudoTrainAttr)
pseudoTrainLabels = np.array(pseudoTrainLabels)

dec_ip = np.concatenate((noise_gen, pseudoTrainAttr) , axis=1)
pseudoTrainData = decoder.predict(dec_ip)

testData = np.load(open('testData'))
testLabels = np.load(open('testLabels'))

pseudoTrainData = normalize(pseudoTrainData , axis =1)
testData = normalize(testData , axis=1)

print 'Training SVM-100'
clf5 = svm.SVC(C=100)
clf5.fit(pseudoTrainData, pseudoTrainLabels)
print 'Predicting...'
pred = clf5.predict(testData)

print accuracy_score(testLabels , pred)

# dict_correct = {}
# allTestLabels = 
allTestClasses = sorted(list(set(testLabels.tolist())))
dict_correct = {}
dict_total = {}

for ii in allTestClasses:
	dict_total[ii] = 0 
	dict_correct[ii] = 0

for ii in range(0,testLabels.shape[0]):
	if(testLabels[ii] == pred[ii]):
	    dict_correct[testLabels[ii]] = dict_correct[testLabels[ii]] + 1
	dict_total[testLabels[ii]] = dict_total[testLabels[ii]] + 1 

avgAcc = 0.0
for ii in allTestClasses:
	avgAcc = avgAcc + (dict_correct[ii]*1.0)/(dict_total[ii])

avgAcc = avgAcc/len(allTestClasses) 
print 'Average Class Accuracy = ' + str(avgAcc)

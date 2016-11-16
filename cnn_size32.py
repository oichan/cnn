# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import numpy as np
# Data loading and preprocessing
#train ../fig/train/train.txt
#test ../fig/test/test.txt
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)
print ("X:")
print (type(X))
print (X)
print ("Y:")
print (type(Y))
print (Y)
print ("X_test:")
print (X_test)
print ("Y_test:")
print (Y_test)
print ("original")
print (type(X_test[0][0][0][0]))
print (type(X_test))
print (type(Y_test[0][0]))
print (type(Y_test))
#自作データ読込み
dic={"B":"0","C":"1","D":"2","E":"3"}
X=[]
Y=[]
X_test=[]
Y_test=[]
#train data import
f = open("../fig/train/train.txt", 'r')
for line in f:
	#改行除いて、スペース区切り
	line = line.rstrip()
	l = line.split()
	#print (l[0])
        img = cv2.imread(l[0])
        img = cv2.resize(img, (32, 32))
	X.append(img)
	tmp = np.zeros(4)
	label_number = dic[l[1]]
	tmp[int(label_number)] = 1
	Y.append(tmp)
f.close()
X = np.asarray(X)
X=X.astype(float)/255.
Y = np.asarray(Y).astype(float)
print ("Oi mathod")
print (type(X.astype(float)[0][0][0][0]))
print (type(X))
print (X/255.)
print (type(Y.astype(float)[0][0]))
print (type(Y))
print (Y)
print ("-------------------------------")
#print (type(np.asarray(X)))
X,Y = shuffle(X,Y)
print (Y)
print ("-------------------------------")
#Y = to_categorical(Y,4)

#test data import
f = open("../fig/test/test.txt", 'r')
for line in f:
	#改行除いて、スペース区切り
	line = line.rstrip()
	l = line.split()
	#print (l[0])
        img = cv2.imread(l[0])
	#img = list(img)
        img = cv2.resize(img, (32, 32))
	X_test.append(img)
	tmp = np.zeros(4)
	label_number = dic[l[1]]
	tmp[int(label_number)] = 1.
	Y_test.append(tmp)
f.close()
X_test = np.asarray(X_test)
X_test = X_test.astype(float)
X_test = X_test/255.
#Y_test = to_categorical(Y_test, 4)
Y_test = np.array(Y_test)
Y_test = Y_test.astype(float)
print ("Oi mathod")
print (type(X_test[0][0][0][0]))
print (type(X_test))
print (X_test/255.)
print (type(Y_test.astype(float)[0][0]))
print (type(Y_test))
print (Y_test)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10000, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=500, run_id='av_cnn')

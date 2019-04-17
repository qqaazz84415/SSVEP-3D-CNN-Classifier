"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import scipy.io


########## CCA FFT MSC data  #####################

ccaFeature_train = scipy.io.loadmat(r"I:\3dcnn_example/ccaFeature_train.mat")
ccaLabel_train = scipy.io.loadmat(r"I:\3dcnn_example/ccaLabel_train.mat")
ccaFeature_test = scipy.io.loadmat(r"I:\3dcnn_example/ccaFeature_test.mat")
ccaLabel_test = scipy.io.loadmat(r"I:\3dcnn_example/ccaLabel_test.mat")

fftFeature_train = scipy.io.loadmat(r"I:\3dcnn_example/fftFeature_train.mat")
fftLabel_train = scipy.io.loadmat(r"I:\3dcnn_example/fftLabel_train.mat")
fftFeature_test = scipy.io.loadmat(r"I:\3dcnn_example/fftFeature_test.mat")
fftLabel_test = scipy.io.loadmat(r"I:\3dcnn_example/fftLabel_test.mat")

mscFeature_train = scipy.io.loadmat(r"I:\3dcnn_example/mscFeature_train.mat")
mscLabel_train = scipy.io.loadmat(r"I:\3dcnn_example/mscLabel_train.mat")

mscFeature_test = scipy.io.loadmat(r"I:\3dcnn_example/mscFeature_test.mat")
mscLabel_test = scipy.io.loadmat(r"I:\3dcnn_example/mscLabel_test.mat")

############ CCA FFT MSC label ##################

ccatrain_data = ccaFeature_train['ccaFeature']
ccatrain_label = ccaLabel_train['ccaLabel']
ccatest_data = ccaFeature_test['ccaFeature']
ccatest_label = ccaLabel_test['ccaLabel']

ffttrain_data = fftFeature_train['fftFeature']
ffttrain_label = fftLabel_train['fftLabel']
ffttest_data = fftFeature_test['fftFeature']
ffttest_label = fftLabel_test['fftLabel']

msctrain_data = mscFeature_train['mscFeature']
msctrain_label = mscLabel_train['mscLabel']
msctest_data = mscFeature_test['mscFeature']
msctest_label = mscLabel_test['mscLabel']

cca_data = np.hstack((ccatrain_data,ccatest_data))
fft_data = np.hstack((ffttrain_data,ffttest_data))
msc_data = np.hstack((msctrain_data,msctest_data))

    ########## CCA FFT MSC train data separate  #####################

for i in range(5):
    for j in range(15):
        if i ==0 and j == 0 :
            cca_datas = cca_data[0][0]
            #ccatrainset_label = ccatrain_label[0][0]
            fft_datas = fft_data[0][0]
            #ffttrainset_label = ffttrain_label[0][0]
            msc_datas = msc_data[0][0]
            #msctrainset_label = msctrain_label[0][0]
        else:
            cca_datas = np.vstack((cca_datas,cca_data[i][j]))
            #ccatrainset_label = np.vstack((ccatrainset_label,ccatrain_label[i][j]))
            fft_datas = np.vstack((fft_datas,fft_data[i][j]))
            #ffttrainset_label = np.vstack((ffttrainset_label,ffttrain_label[i][j]))
            msc_datas = np.vstack((msc_datas,msc_data[i][j]))
            #msctrainset_label = np.vstack((msctrainset_label,msctrain_label[i][j]))


    ########## CCA FFT MSC train data reshape  #####################

CCA_datas = np.array(cca_datas,dtype = np.float32)
#ccatrainset_label_reshape = ccatrainset_label.reshape(-1)
#ccatrainset_labels = np.array(ccatrainset_label_reshape, dtype = np.int)

FFT_datas = np.array(fft_datas,dtype = np.float32)
#ffttrainset_label_reshape = ffttrainset_label.reshape(-1)
#ffttrainset_labels = np.array(ffttrainset_label_reshape, dtype = np.int)

MSC_datas = np.array(msc_datas,dtype = np.float32)
#msctrainset_label_reshape = msctrainset_label.reshape(-1)
#msctrainset_labels = np.array(msctrainset_label_reshape, dtype = np.int)

data=np.dstack((CCA_datas,FFT_datas,MSC_datas))









train_label0 = np.array([1., 0., 0., 0., 0.])
train_label1 = np.array([0., 1., 0., 0., 0.])
train_label2 = np.array([0., 0., 1., 0., 0.])
train_label3 = np.array([0., 0., 0., 1., 0.])
train_label4 = np.array([0., 0., 0., 0., 1.])

train_label = []

for l0 in range(450):
    if l0 == 0 :
        train_label = train_label0
    else :
        train_label = np.vstack((train_label,train_label0))

for l1 in range(450):
    train_label = np.vstack((train_label,train_label1))

for l2 in range(450):
    train_label = np.vstack((train_label,train_label2))

for l3 in range(450):
    train_label = np.vstack((train_label,train_label3))

for l4 in range(450):
    train_label = np.vstack((train_label,train_label4))
















tf.set_random_seed(1)
np.random.seed(1)

permutation = np.random.permutation(train_label.shape[0])
shuffled_dataset = data[permutation, :]
shuffled_labels = train_label[permutation]



BATCH_SIZE = 50
LR = 0.001              # learning rate


tf_x = tf.placeholder(tf.float32, [None, 5, 3])

image = tf.reshape(tf_x, [-1, 1, 5, 3, 1])  
tf_y = tf.placeholder(tf.int32, [None, 5])            # input y

# CNN

conv1 = tf.layers.conv3d(   
    inputs=image,
    filters=16,
    kernel_size=1,
    strides=1,
    padding='same',
    activation=tf.nn.relu)

conv2 = tf.layers.conv3d(conv1, 32, 1, 1, 'same', activation=tf.nn.relu)
flat = tf.reshape(conv2, [-1, 15*32])          # -> (7*7*32, )

output = tf.layers.dense(flat, 5)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

turns = 0
for step in range(2250*500):
    
    if step % 2249 == 0: 
        turns = 0
        

        test_x = shuffled_dataset[:200]
        test_y = shuffled_labels[:200]
        
        b_x = shuffled_dataset[turns:(turns+1)*1]
        b_y = shuffled_labels[turns:(turns+1)*1]
    else :
        turns = turns + 1
        b_x = shuffled_dataset[turns:(turns+1)*1]
        b_y = shuffled_labels[turns:(turns+1)*1]
    
        
    
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.8f' % loss_, '| test accuracy: %.4f' % accuracy_)
    if (step+1) % (2250*5) == 0:
        with open('tensorflow_cnnclassifier.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step+1, accuracy_])

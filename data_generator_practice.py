import os
import glob
import cv2 as cv
import numpy as np
from make_training_data_v2 import get_data
import tensorflow as tf
import matplotlib.pyplot as plt

############### Parameter ################
global total_size ## it initialize later
batch_size = 128
learning_rate = 0.01
epoch_size = 25

def image_gen(train_img, train_label):
    for train_img_one, train_label_one in zip(train_img, train_label) :
        yield(train_img_one, train_label_one)

def Training_filter(train_img, train_label, test_img, test_label):
    global total_size
    total_size = len(train_img)

    train_img = np.reshape(train_img, (-1, 768))
    train_img = train_img / 256
    test_img = np.reshape(test_img, (-1, 768))
    test_img = test_img / 256
    print(test_img.shape)

    x = tf.placeholder(tf.float32, shape=[None,768])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal(shape=[768,1], stddev=5e-2))
    b = tf.Variable(tf.constant(0.1, shape=[1]))
    feature = tf.matmul(x, W) + b
    y_pred = tf.sigmoid(feature)

    loss = tf.reduce_mean(tf.square(y_pred-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    
    predicted = tf.cast(y_pred > 0.5, dtype=tf.float32)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_size):
            average_loss = 0
            total_batch = int(total_size / batch_size)
            for i in range(total_batch):
                _, current_loss = sess.run([train_step, loss], feed_dict={x:train_img, y:train_label})
                average_loss += current_loss/total_batch
                if i % 500 == 0:
                    print("      ----- batch_num : %d, current_loss : %f, average_loss : %f"%(i+1, current_loss, average_loss))
            if epoch % 1 == 0:
                print("Epoch: %d, 손실 함수(Loss): %f" %((epoch+1), average_loss))
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))
        y_pred, y, f, a = sess.run([y_pred, y, feature, accuracy], feed_dict={x:test_img, y:test_label})
        print("예측: %d , true값: %d , 정확도(Accuracy): %f" %(np.count_nonzero(y_pred), np.count_nonzero(y), a))
        print(test_img.shape)
        print(feature.shape)
        plot_show(test_img, test_label, f)
        
        
def plot_show(test_img, test_label, feature):

    plt.subplot(2,1,1)
    test_img_MinMax = test_img.mean(axis=1)
    print(test_img_MinMax.shape)
    test_img_MinMax = (test_img_MinMax - test_img_MinMax.min()) / (test_img_MinMax.max() - test_img_MinMax.min())
    plt.plot(test_img_MinMax, test_label, 'bo')
    plt.xlabel('images mean value(normalized 0 ~ 1)')
    plt.ylabel('1 is building, 0 is ground')

    plt.subplot(2,1,2)
    feature_MinMax = (feature - feature.min()) / (feature.max() - feature.min())
    plt.plot(feature_MinMax, test_label, 'go')
    plt.xlabel('images mean value(normalized 0 ~ 1)')
    plt.ylabel('1 is building, 0 is ground')
    plt.show()

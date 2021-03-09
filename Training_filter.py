import os
import glob
import cv2 as cv
import numpy as np
from make_training_data_v2 import get_data
import tensorflow as tf
import matplotlib.pyplot as plt

############### Parameter ################
global total_size ## it initialize later
batch_size = 256
learning_rate = 0.001
epoch_size = 5

def Training_filter(train_input, train_output, test_input, test_output):
    global total_size
    total_size = len(train_input)
    print(train_input.shape, total_size)

    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
    dataset = dataset.shuffle(total_size).repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    train_input_stacked, train_output_stacked = iterator.get_next()

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_normal(shape=[3,3,3,1], stddev=5e-2))
    b = tf.Variable(tf.constant(0.1, shape=[1]))
    conv1 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1])+ b
    conv1 = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1 = tf.reshape(conv1, [-1,49])

    W_output = tf.Variable(tf.random_normal(shape=[49,1], stddev=5e-2))
    b_output = tf.Variable(tf.constant(0.1, shape=[1]))

    logits = tf.sigmoid(tf.matmul(conv1, W_output) + b_output)

    loss = tf.reduce_mean(tf.square(logits-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    
    predicted = tf.cast(logits > 0.5, dtype=tf.float32)
import os
import glob
import cv2 as cv
import numpy as np
from make_training_data_v2 import get_data
import tensorflow as tf


############### Parameter ################
global total_size ## it initialize later
batch_size = 256
learning_rate = 0.001
epoch_size = 5

def Make_dataset_batch(train_input, train_output):
    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
    dataset = dataset.shuffle(total_size).repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    train_input_stacked, train_output_stacked = iterator.get_next()
    return iterator, train_input_stacked, train_output_stacked


def Logistic_Regression_train(train_input, train_output, test_input, test_output):
    train_input = train_input.mean(axis=(1,2,3)).reshape(-1, 1)
    train_input = train_input / 255
    global total_size 
    total_size = len(train_input)

    iterator, train_input_stacked, train_output_stacked = Make_dataset_batch(train_input, train_output) #create batch dataset

    test_input = test_input.mean(axis=(1,2,3)).reshape(-1, 1)
    test_input = test_input / 255

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_normal(shape=[1]))
    b = tf.Variable(tf.random_normal(shape=[1]))
    
    linear_model = W * x + b
    loss = tf.reduce_mean(tf.square(linear_model-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_size):
            sess.run(iterator.initializer)
            average_loss = 0
            total_batch = int(total_size / batch_size)
            for i in range(total_batch):
                _, curr_W, curr_b, current_loss = sess.run([train_step, W, b, loss], feed_dict={x:train_input, y:train_output})
                average_loss += current_loss/total_batch
                if i % 500 == 0:
                    print("      ----- batch_num : %d, current_loss : %f, average_loss : %f"%(i+1, current_loss, average_loss))
            if epoch % 1 == 0:
                print("Epoch: %d, W: %f, b: %f, 손실 함수(Loss): %f" %((epoch+1), curr_W, curr_b, average_loss))
            correct_prediction = tf.equal(tf.argmax(linear_model,1), tf.argmax(y,1))

        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("정확도(Accuracy): %f" %(accuracy.eval(feed_dict={x:test_input, y:test_output})))
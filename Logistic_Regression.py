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

def Logistic_Regression_train(train_input, train_output, test_input, test_output):
    global total_size
    total_size = len(train_input)
    print(train_input.shape, total_size)

    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
    dataset = dataset.shuffle(total_size).repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    train_input_stacked, train_output_stacked = iterator.get_next()

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_normal(shape=[16,16,3,1], stddev=5e-2))
    b = tf.Variable(tf.constant(0.1, shape=[1]))
    logits = tf.sigmoid(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.square(logits-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    
    predicted = tf.cast(logits > 0.5, dtype=tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_size):
            sess.run(iterator.initializer)
            average_loss = 0
            total_batch = int(total_size / batch_size)
            for i in range(total_batch):
                train_input_batch, train_output_batch = sess.run([train_input_stacked, train_output_stacked])
                _, curr_W, curr_b, current_loss = sess.run([train_step, W, b, loss], feed_dict={x:train_input_batch, y:train_output_batch})
                average_loss += current_loss/total_batch
                if i % 500 == 0:
                    print("      ----- batch_num : %d, current_loss : %f, average_loss : %f"%(i+1, current_loss, average_loss))
            if epoch % 1 == 0:
                print("Epoch: %d, W: %f, b: %f, 손실 함수(Loss): %f" %((epoch+1), curr_W, curr_b, average_loss))
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))
        W_output, b_output, P, Y, a = sess.run([W, b, predicted, y, accuracy], feed_dict={x:test_input, y:test_output})
        print("예상 건물의 수 %d, 실제 건물의 수 %d, 정확도(Accuracy): %f" %(np.count_nonzero(P), np.count_nonzero(Y), a))
        plot_show(train_input, train_output, test_input, test_output, W_output, b_output)
        
        
def plot_show(train_input, train_output, test_input, test_output, W_output, b_output):
    plt.subplot(2,1,1)
    plt.plot(train_input, train_output, 'bo')
    plt.plot(train_input, W_output * train_input + b_output, 'r-')
    plt.xlabel('images mean value(normalized 0 ~ 1)')
    plt.ylabel('1 is building, 0 is ground')

    plt.subplot(2,1,2)
    plt.plot(test_input, test_output, 'go')
    plt.plot(train_input, W_output * train_input + b_output, 'r-')
    plt.xlabel('images mean value(normalized 0 ~ 1)')
    plt.ylabel('1 is building, 0 is ground')
    plt.show()
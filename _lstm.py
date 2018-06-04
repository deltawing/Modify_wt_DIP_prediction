# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:35:40 2017

@author: 罗骏
"""
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn

learning_rate = 0.001
#raining_steps = 10000
repeat_times = 20
batch_size = 128
display_step = 20
num_input = 14
timesteps = 40
num_hidden = 256
num_classes = 2 

def define_timesteps_numinput(mod, max_level):
    global num_input, timesteps
    num_input = 14*mod
    timesteps = (max_level+1)*8

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {'out': tf.Variable(tf.random_normal([num_hidden*2, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def RNN_cell(num_hidden):
    lstm_cell = rnn.GRUCell(num_hidden, kernel_initializer=tf.orthogonal_initializer())
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.7)
    return lstm_cell
def RNN(x, weights, biases):
    with tf.variable_scope('rnn_layer',initializer=tf.orthogonal_initializer()):
        x = tf.unstack(x, axis = 1)
        cells_fw = [RNN_cell(num_hidden) for _ in range(2)]
        cells_bw = [RNN_cell(num_hidden) for _ in range(2)]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, 
                                                                       inputs=X, dtype=tf.float32)
        return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#####################################
train_set = tf.data.TFRecordDataset(["train.tfrecord"])
test_set = tf.data.TFRecordDataset(["test.tfrecord"])
def parse_function(example_proto):
    global num_input, timesteps
    print('in parse_function ', num_input*timesteps)
    dics = {'X_batch': tf.FixedLenFeature(shape=(num_input*timesteps,), dtype=tf.float32),      #
            'Y_batch': tf.FixedLenFeature(shape=(1,), dtype=tf.float32)}
    parsed_example = tf.parse_single_example(example_proto, dics)  
    return parsed_example
new_train_set = train_set.map(parse_function)
train_iterator = new_train_set.shuffle(buffer_size=60000).repeat(repeat_times).batch(batch_size).make_one_shot_iterator()
train_next_element = train_iterator.get_next()

new_test_set = test_set.map(parse_function)
test_iterator = new_test_set.shuffle(buffer_size=60000).batch(batch_size).make_one_shot_iterator()
test_next_element = test_iterator.get_next()
#####################################

def modify_y(batch_y):
    modify = []
    for i in batch_y:
        if i[0] == 0: temp = [1.0, 0.0]
        elif i[0] == 1: temp = [0.0, 1.0]
        modify += [temp]
    modify = np.array(modify)
    return modify

def apply_lstm(train_data_row = 1000, test_data_row = 1000, num_input=1, timesteps=5):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("LSTM optimizing ...")
        for step in range(1, train_data_row*repeat_times//batch_size+1):
            batch_x, batch_y = sess.run([train_next_element['X_batch'], train_next_element['Y_batch']])
            batch_x = batch_x.reshape((-1, 280))
            batch_x = batch_x.reshape((batch_size, num_input*14, (timesteps+1)*8)).transpose((0,2,1))
            batch_y = modify_y(batch_y)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if (step+1)%display_step == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        print("Optimization Finished!")
        test_cost, test_acc = 0, 0
        for j in range(test_data_row//batch_size):
            X_batch_1, Y_batch_1 = sess.run([test_next_element['X_batch'], test_next_element['Y_batch']])
            X_batch_1 = X_batch_1.reshape((-1, 280))
            X_batch_1 = X_batch_1.reshape((batch_size, num_input*14, (timesteps+1)*8)).transpose((0,2,1))
            Y_batch_1 = modify_y(Y_batch_1)
            _cost, _acc = sess.run([loss_op, accuracy], feed_dict={X:X_batch_1, Y:Y_batch_1})
            test_cost += _cost
            test_acc += _acc
        print(test_cost/(test_data_row//batch_size), test_acc/(test_data_row//batch_size))

if __name__ == '__main__':
    apply_lstm(630, 158)
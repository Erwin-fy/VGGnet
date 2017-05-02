#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf

batch_size = 32
max_step = 10000

img_width = 224
img_height = 224
img_channel = 3

image_holder = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_channel])
label_holder = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

start_learning_rate = 1e-2
decay_rate = 0.95
decay_steps = 1000

steps = '10000'
param_dir = '../params/'
save_filename = 'model'
load_filename = 'model-' + steps
checkpointer_iter = 2000

log_dir = '../log/'
summary_iter = 20000


#Todo
'''
images_train, labels_train
images_test, labels_test
'''

def print_tensor(tensor):
    print tensor.op.name, ' ', tensor.get_shape().as_list()

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def _activation_summary(tensor):
    name = tensor.op.name
    tf.summary.histogram(name + '/activatins', tensor)
    tf.summary.scalar(name + '/sparsity', tf.nn.zeros_fraction(tensor))

def conv_layer(fm, channels):
    '''
    Arg fm: feather maps
    '''
    shape = fm.get_shape()
    kernel = variable_with_weight_loss(shape=[3, 3, shape[-1].value, channels], stddev=1e-2, wl=0.0)
    conv = tf.nn.conv2d(fm, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[channels]))
    pre_activation = tf.nn.bias_add(conv, biases)

    activation = tf.nn.relu(pre_activation)

    print_tensor(activation)
    _activation_summary(activation)

    return activation

def fc_layer(input_op, fan_out):
    '''
    input_op: 输入tensor
    fan_in: 输入节点数
    fan_out： 输出节点数
    '''
    fan_in = input_op.get_shape()[1].value

    weights = variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=0.04)
    biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fan_out]))
    pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)

    activation = tf.nn.relu(pre_activation)

    print_tensor(activation)
    _activation_summary(tensor)

    return activation


def inference(images, keep_prob):
    with tf.name_scope('conv1') as scope:
        conv1_1 = conv_layer(images, 64)
        conv1_2 = conv_layer(conv1_1, 64)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2') as scope:
        conv2_1 = conv_layer(pool1, 128)
        conv2_2 = conv_layer(conv2_1, 128)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3') as scope:
        conv3_1 = conv_layer(pool2, 256)
        conv3_2 = conv_layer(conv3_1, 256)
        conv3_3 = conv_layer(conv3_2, 256)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    with tf.name_scope('conv4') as scope:
        conv4_1 = conv_layer(pool3, 512)
        conv4_2 = conv_layer(conv4_1, 512)
        conv4_3 = conv_layer(conv4_2, 512)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv5') as scope:
        conv5_1 = conv_layer(pool4, 512)
        conv5_2 = conv_layer(conv5_1, 512)
        conv5_3 = conv_layer(conv5_2, 512)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print_tensor(pool5)


    with tf.name_scope('fc1') as scope:
        reshape = tf.reshape(pool5, [batch_size, -1])
        print_tensor(reshape)
        fc1 = fc_layer(reshape, 4096)
        drop1 = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        fc2 = fc_layer(drop1, 4096)
        drop2 = tf.nn.dropout(fc2, keep_prob)

    with tf.name_scope('final_fc') as scope:
        logits = fc_layer(drop2, 21)

    print_tensor(logits)

    return logits

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)

    cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_sum')
    cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    tf.summary.scalar(total_loss.op.name + ' (raw)', loss)

    return total_loss


def train_op(total_loss, global_step):
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)

def main():
    global_step = tf.get_variable('global_step', initializer=0, 
                        dtype=tf.int32, trainable=False)

    logits = inference(image_holder, keep_prob)
    loss = loss(logits, label_holder)
    train_op = train_op(loss, global_step)

    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        logdir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

        #start training
        print 'start training'
        for step in range(max_step):
            #start_time = time.time()

            with tf.device('/cpu:0'):
                image_batch, label_batch = sess.run([images_train, labels_train])

            feed_dict={
                image_holder:image_batch,
                label_holder:label_batch,
                keep_prob:0.5
            }

            with tf.device('/gpu:0'):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            if step%20 == 0:
                print 'step %d, loss = ' % step, loss_value

            with tf.device('/cpu:0'):
                if (step+1)%checkpointer_iter == 0:
                    saver.save(sess, param_dir+save_filename, global_step.eval())
                
                if (step+1)%summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, global_step.eval())

            
            #test
            num_examples = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
            num_iter = int(math.ceil(num_examples/batch_size))
            true_count = 0
            total_sample_count = num_iter*batch_size
            step = 0
            while step < num_iter:
                step += 1

                with tf.device("/cpu:0"):
                    image_batch, label_batch = sess.run([images_test, labels_test])

                with tf.device("/gpu:0"):
                    accuracy = sess.run([top_k_op], feed_dict=feed_dict)

                true_count += np.sum(accuracy)

                precision = 1.0*true_count / total_sample_count

                print 'precision @ 1 = %.3f' % precision



if __name__ == '__main__':
    main()



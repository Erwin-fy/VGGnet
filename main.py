#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

batch_size = 32

img_width = 224
img_height = 224
img_channel = 3

image_holder = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_channel])
label_holder = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

start_learning_rate = 1e-2


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
        predictions = fc_layer(drop2, 21)

    print_tensor(predictions)

    return predictions

def main():
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        predictions = inference(image_holder, keep_prob)


if __name__ == '__main__':
    main()



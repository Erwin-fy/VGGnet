#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import tensorflow as tf

class VGG():
    def __init__(self, config):
        self.global_step = tf.get_variable('global_step', initializer=0, 
                        dtype=tf.int32, trainable=False)

        self.batch_size = config.batch_size
    
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.img_channel = config.img_channel

        self.start_learning_rate = config.start_learning_rate
        self.decay_rate = config.decay_rate
        self.decay_steps = config.decay_steps

        self.image_holder = tf.placeholder(tf.float32,
                                [self.batch_size, self.img_width, self.img_height, self.img_channel])
        self.label_holder = tf.placeholder(tf.int32, [self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)


    def print_tensor(self, tensor):
        print tensor.op.name, ' ', tensor.get_shape().as_list()

    def variable_with_weight_loss(self, shape, stddev, wl):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    def _activation_summary(self, tensor):
        name = tensor.op.name
        tf.summary.histogram(name + '/activatins', tensor)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))

    def conv_layer(self, fm, channels, scope):
        '''
        Arg fm: feather maps
        '''
        shape = fm.get_shape()
        kernel = self.variable_with_weight_loss(shape=[3, 3, shape[-1].value, channels], stddev=1e-2, wl=0.0)
        #kernel = tf.get_variable(scope + 'kernel', shape=[3, 3, shape[-1].value, channels], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(fm, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[channels]))
        pre_activation = tf.nn.bias_add(conv, biases)

        activation = tf.nn.relu(pre_activation)

        self.print_tensor(activation)
        self._activation_summary(activation)


        self.print_tensor(kernel)

        return activation

    def fc_layer(self, input_op, fan_out, is_train, scope):
        '''
        input_op: 输入tensor
        fan_in: 输入节点数
        fan_out： 输出节点数
        is_train: True --- fc   Flase --- conv
        '''
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fan_out]))

        if is_train:
            reshape = tf.reshape(input_op, [self.batch_size, -1])
            fan_in = reshape.get_shape()[1].value
            #weights = self.variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=0.004)
            weights = tf.get_variable(scope + 'weights', shape=[fan_in, fan_out], 
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            
            pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)
        else:
            shape = input_op.get_shape()
            weights = tf.get_variable(scope + 'weights', 
                    shape=[shape[1].value, shape[2].value, shape[3].value, fan_out], 
                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input_op, weights, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(conv, biases)

        activation = tf.nn.relu(pre_activation)
        self.print_tensor(activation)
        self._activation_summary(activation)
        return activation

    def final_fc_layer(self, input_op, fan_out, is_train, scope):
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fan_out]))

        if is_train:
            reshape = tf.reshape(input_op, [self.batch_size, -1])
            fan_in = reshape.get_shape()[1].value
            #weights = self.variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=0.0)
            weights = tf.get_variable(scope + 'weights', shape=[fan_in, fan_out], 
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
            pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)
        else:
            shape = input_op.get_shape()
            weights = tf.get_variable(scope + 'weights', 
                    shape=[shape[1].value, shape[2].value, shape[3].value, fan_out], 
                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input_op, weights, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(conv, biases)

        self.print_tensor(pre_activation)
        self._activation_summary(pre_activation)

        return pre_activation
  

    def inference(self, is_train):
        with tf.name_scope('conv1') as scope:
            conv1_1 = self.conv_layer(self.image_holder, 64, 'conv1_1')
            conv1_2 = self.conv_layer(conv1_1, 64, 'conv1_2')
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('conv2') as scope:
            conv2_1 = self.conv_layer(pool1, 128, 'conv2_1')
            conv2_2 = self.conv_layer(conv2_1, 128, 'conv2_2')
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('conv3') as scope:
            conv3_1 = self.conv_layer(pool2, 256, 'conv3_1')
            conv3_2 = self.conv_layer(conv3_1, 256, 'conv3_2')
            conv3_3 = self.conv_layer(conv3_2, 256, 'conv3_3')
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        with tf.name_scope('conv4') as scope:
            conv4_1 = self.conv_layer(pool3, 512, 'conv4_1')
            conv4_2 = self.conv_layer(conv4_1, 512, 'conv4_2')
            conv4_3 = self.conv_layer(conv4_2, 512, 'conv4_3')
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('conv5') as scope:
            conv5_1 = self.conv_layer(pool4, 512, 'conv5_1')
            conv5_2 = self.conv_layer(conv5_1, 512, 'conv5_2')
            conv5_3 = self.conv_layer(conv5_2, 512, 'conv5_3')
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.print_tensor(pool5)


        with tf.name_scope('fc1') as scope:
            fc1 = self.fc_layer(pool5, 4096, is_train, 'fc1')
            drop1 = tf.nn.dropout(fc1, self.keep_prob)

        with tf.name_scope('fc2') as scope:
            fc2 = self.fc_layer(drop1, 4096, is_train, 'fc2')
            drop2 = tf.nn.dropout(fc2, self.keep_prob)

        with tf.name_scope('final_fc') as scope:
            logits = self.final_fc_layer(drop2, 20, is_train, 'final_fc')

        return tf.reduce_mean(logits, axis=[1,2])

    def loss(self, logits):
        labels = tf.cast(self.label_holder, tf.int64)

        cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_sum')
        cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)

        return total_loss


    def train_op(self, total_loss):
        learning_rate = tf.train.exponential_decay(self.start_learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=self.global_step)
        return train_op

    def top_k_op(self, logits):
        return tf.nn.in_top_k(logits, self.label_holder, 1)

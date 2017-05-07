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

        self.start_learning_rate = 1e-3
        self.decay_rate = 0.95
        self.decay_steps = 200

        self.vgg_npy_path = './vgg16.npy'
        self.data_dict = np.load(self.vgg_npy_path, encoding='latin1').item()

        self.wl = 5e-3

        self.image_holder = tf.placeholder(tf.float32,
                                [self.batch_size, self.img_width, self.img_height, self.img_channel])
        self.label_holder = tf.placeholder(tf.int32, [self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)


    def print_tensor(self, tensor):
        print tensor.op.name, ' ', tensor.get_shape().as_list()

    def variable_with_weight_loss(self, shape, stddev, wl):
        var = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev))
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    def _activation_summary(self, tensor):
        name = tensor.op.name
        tf.summary.histogram(name + '/activatins', tensor)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))

    def conv_layer(self, fm, channels, name):
        '''
        Arg fm: feather maps
        '''
        with tf.variable_scope(name) as scope:
            #shape = fm.get_shape()
            #kernel = self.variable_with_weight_loss(shape=[3, 3, shape[-1].value, channels], stddev=1e-2, wl=0.0)
            #kernel = tf.get_variable(scope + 'kernel', shape=[3, 3, shape[-1].value, channels], 
            #    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
            kernel = self.get_conv_kernel(name)
            biases = self.get_bias(name)
            conv = tf.nn.conv2d(fm, kernel, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)

            activation = tf.nn.relu(pre_activation)

            self.print_tensor(activation)
            self._activation_summary(activation)

            return activation

    def fc_layer(self, input_op, fan_out, name):
        '''
        input_op: 输入tensor
        fan_in: 输入节点数
        fan_out： 输出节点数
        is_train: True --- fc   Flase --- conv
        '''
        with tf.variable_scope(name) as scope:
            reshape = tf.reshape(input_op, [self.batch_size, -1])
            #fan_in = reshape.get_shape()[1].value
            #weights = self.variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=0.004)
            #weights = tf.get_variable(scope + 'weights', shape=[fan_in, fan_out], 
            #    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            #biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fan_out]))
            
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            pre_activation = tf.nn.bias_add(tf.matmul(reshape, weights), biases)
            activation = tf.nn.relu(pre_activation)

            self.print_tensor(activation)
            self._activation_summary(activation)
            return activation

    def final_fc_layer(self, input_op, fan_out, name):
        with tf.variable_scope(name) as scope:
            fan_in = input_op.get_shape()[1].value
            #weights = tf.get_variable(name + 'weights', shape=[fan_in, fan_out], 
            #    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            
            #weights = self.get_fc_weight(name)
            #biases = self.get_bias(name)

            weights = self.variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=self.wl)
            biases = tf.Variable(tf.constant(0.1, shape=[fan_out], dtype=tf.float32), name='biases')
            
            pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)

            self.print_tensor(pre_activation)
            self._activation_summary(pre_activation)

            return pre_activation
  

    def inference(self):
        conv1_1 = self.conv_layer(self.image_holder, 64, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 64, 'conv1_2')
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print pool1

        conv2_1 = self.conv_layer(pool1, 128, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 128, 'conv2_2')
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3_1 = self.conv_layer(pool2, 256, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 256, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 256, 'conv3_3')
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv4_1 = self.conv_layer(pool3, 512, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 512, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 512, 'conv4_3')
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv5_1 = self.conv_layer(pool4, 512, 'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 512, 'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 512, 'conv5_3')
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.print_tensor(pool5)


        fc6 = self.fc_layer(pool5, 4096, 'fc6')
        drop1 = tf.nn.dropout(fc6, self.keep_prob)

        fc7 = self.fc_layer(drop1, 4096, 'fc7')
        drop2 = tf.nn.dropout(fc7, self.keep_prob)

        #fc8 = self.fc_layer(drop2, 1000, 'fc8')
        #drop3 = tf.nn.dropout(fc8, self.keep_prob)

        logits = self.final_fc_layer(drop2, 20, 'final_fc_layer')


        return logits

    def loss(self, logits):
        labels = tf.cast(self.label_holder, tf.int64)
        cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_sum')
        cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')
       
        '''
        #softmax loss
        logits = tf.nn.softmax(logits) + 1e-4
        labels = tf.one_hot(self.label_holder, 20)
        labels = tf.cast(labels, tf.float32)
        
        #self.print_tensor(tf.multiply(logits, labels))

        softmax_loss_sum = -tf.log(tf.reduce_sum(tf.multiply(logits, labels), axis=1))
        softmax_loss = tf.reduce_mean(softmax_loss_sum, name='softmax_loss')
        '''


        tf.add_to_collection('losses', cross_entropy)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')


        tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)

        return total_loss


    def train_op(self, total_loss):
        learning_rate = tf.train.exponential_decay(self.start_learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=self.global_step)
        
        return train_op

    def predictions(self, logits): 
        return tf.nn.softmax(logits)


    def top_k_op(self, logits):
        return tf.nn.in_top_k(logits, self.label_holder, 1)


    def get_conv_kernel(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name='kernel', shape=shape, initializer=init)
    
        #if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wl, name='weight_loss')
        tf.add_to_collection("losses", weight_decay)
        print name+'/kernel', shape

        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print self.data_dict[name][0]
        var = tf.get_variable(name='weights', shape=shape, initializer=init)
        
        #if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wl, name='weight_loss')
        tf.add_to_collection("losses", weight_decay)
        #print 'l2loss'
        print name+'/weights', shape
        return var


    def get_bias(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][1], dtype=tf.float32)
        shape = self.data_dict[name][1].shape
        biases = tf.get_variable(name='biases', shape=shape, initializer=init)
        print name+'/biases', shape
        return biases


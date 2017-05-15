#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG():
    def __init__(self, config):
        self.global_step = tf.get_variable('global_step', initializer=0, 
                        dtype=tf.int32, trainable=False)

        self.batch_size = config.batch_size

        self.img_width = config.img_width
        self.img_height = config.img_height
        self.img_channel = config.img_channel

        self.start_learning_rate = 1e-4
        self.decay_rate = 0.99
        self.decay_steps = 100

        self.vgg_path = config.vgg_path
        self.data_dict = np.load(self.vgg_path, encoding='latin1').item()
        self.var_dict = {}

        self.wl = 5e-4

        self.image_holder = tf.placeholder(tf.float32,
                                [self.batch_size, self.img_height, self.img_width, self.img_channel])
        self.label_holder = tf.placeholder(tf.float32, [self.batch_size, 20])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)


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

    def inference(self):
        images = self.image_holder * 255.0
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[red - VGG_MEAN[0], 
                                       green - VGG_MEAN[1],
                                       blue - VGG_MEAN[2]])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 64, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 'conv1_2')
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 128, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 'conv2_2')
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 256, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 'conv3_3')
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        
        self.conv4_1 = self.conv_layer(self.pool3, 512, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 'conv4_3')
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 'conv5_3')
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
        self.print_tensor(self.pool5)

        self.fc6 = self.fc_layer(self.pool5, 4096, 'fc6')
        if self.is_train is not None:
            self.fc6 = tf.cond(self.is_train, lambda: tf.nn.dropout(self.fc6, 0.5), lambda: self.fc6)

        self.fc7 = self.fc_layer(self.fc6, 4096, 'fc7')
        if self.is_train is not None:
            self.fc7 = tf.cond(self.is_train, lambda: tf.nn.dropout(self.fc7, 0.5), lambda: self.fc7)

        self.fc8 = self.fc_layer(self.fc7, 1000, 'fc8')

        self.fc9 = self.final_fc_layer(self.fc8, 20, 'fc9')
        
        '''
        if not is_train:
            self.logits =tf.reduce_mean(self.logits, axis=[1, 2])
        '''

        self.prob = tf.nn.softmax(self.fc9, name='prob')

        self.loss = self.loss()
        self.train_op = self.train_op()


    def conv_layer(self, fm, channels, name):
        '''
        Arg fm: feather maps
        '''
        with tf.variable_scope(name):
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
        with tf.variable_scope(name):
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            
            reshape = tf.reshape(input_op, [self.batch_size, -1])
            pre_activation = tf.nn.bias_add(tf.matmul(reshape, weights), biases)
            ''''
            if is_train:
                reshape = tf.reshape(input_op, [self.batch_size, -1])
                pre_activation = tf.nn.bias_add(tf.matmul(reshape, weights), biases)
            else:
                if name == 'fc6':
                    kernels_reshape = tf.reshape(weights, [7, 7, 512, 4096])
                else:
                    kernels_reshape = tf.reshape(weights, [1, 1, 4096, 4096])

                conv = tf.nn.conv2d(input_op, kernels_reshape, [1, 1, 1, 1], padding='VALID')
                pre_activation = tf.nn.bias_add(conv, biases)
            '''
            if name == 'fc8':
                activation = pre_activation
            else:
                activation = tf.nn.relu(pre_activation)
            
            self.print_tensor(activation)
            self._activation_summary(activation)
            return activation

    def final_fc_layer(self, input_op, fan_out, name):
        with tf.variable_scope(name):
            #weights = self.get_fc_weight_reshape(name, [4096, 1000], num_classes=20)
            #biases = self.get_bias_reshape(name, num_new=20)

            if name in self.data_dict:
                weights = self.data_dict[name][0]
                biases = self.data_dict[name][1]
            else:
                weights = self.variable_with_weight_loss([1000, 20], 0.01, self.wl)
                biases = tf.Variable(tf.constant(0.1, shape=[20], dtype=tf.float32))

            self.var_dict[(name, 0)] = weights
            self.var_dict[(name, 1)] = biases

            pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)
            '''
            if is_train:
                pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)
            else:
                kernels_reshape = tf.reshape(weights, [1, 1, 4096, 20])
                conv = tf.nn.conv2d(input_op, kernels_reshape, [1, 1, 1, 1], padding='VALID')
                pre_activation = tf.nn.bias_add(conv, biases)
            '''

            self.print_tensor(pre_activation)
            self._activation_summary(pre_activation)


            return pre_activation


    def loss(self):
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label_holder * tf.log(self.prob), reduction_indices=[1]))
        cross_entropy = tf.reduce_sum((self.prob - self.label_holder) ** 2)
        self.print_tensor(self.prob)

        '''
        tf.add_to_collection('losses', cross_entropy)
        
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)
        '''

        tf.summary.scalar(cross_entropy.op.name, cross_entropy)
        return cross_entropy

    def accuracy(self):
        correct_predition = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.label_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        return accuracy

    def train_op(self, total_loss):
        learning_rate = tf.train.exponential_decay(self.start_learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=self.global_step)
        
        return train_op

    def top_k_op(self, logits, index):
        return tf.nn.in_top_k(logits, self.label_holder, index)

    def save_npy(self, sess, npy_path):
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)


    def get_conv_kernel(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name='kernel', shape=shape, initializer=init)
    
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wl, name='weight_loss')
            tf.add_to_collection("losses", weight_decay)

        self.var_dict[(name, 0)] = var

        return var


    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name='weights', shape=shape, initializer=init)
        
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wl, name='weight_loss')
            tf.add_to_collection("losses", weight_decay)

        self.var_dict[(name, 0)] = var

        return var

    def get_bias(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][1], dtype=tf.float32)
        shape = self.data_dict[name][1].shape
        biases = tf.get_variable(name='biases', shape=shape, initializer=init)

        self.var_dict[(name, 1)] = biases

        return biases
    
    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        #return weights[:, 0:num_classes]

        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
            print weights.shape
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var

    def get_bias_reshape(self, name, num_new):
        biases = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        return biases[0: num_new]
        

        num_orig = shape[0]
        n_averaged_elements = num_orig//num_new
        avg_biases = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_biases[avg_idx] = np.mean(biases[start_idx:end_idx])
        return avg_biases

    def _summary_reshape(self, fweight, shape, num_new):
        num_orig = shape[1]
        shape[1] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, avg_idx] = np.mean(
                fweight[:, start_idx:end_idx], axis=1)
        return avg_fweight

    '''
    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection('losses',
                                 weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`
        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _add_wd_and_summary(self, var, wd):
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var
    '''

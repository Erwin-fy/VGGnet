#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf

'''
a = np.array([[1,3], [4,5,], [7,8]])
b = np.array([1,2])
c = np.subtract(a,b)
print c
'''
'''
a = tf.one_hot([11,8,9,9], 12)

sess = tf.Session()

print sess.run(a)
'''

data_dict = np.load('./vgg16.npy', encoding='latin1').item()
'''
kernel = tf.Variable(tf.constant_initializer(data_dict['conv4_3'][0], dtype=tf.float32))
biases = tf.Variable(tf.constant_initializer(data_dict['conv4_3'][1], dtype=tf.float32))
print kernel.get_shape().as_list()
print biases.get_shape().as_list()

weights6 = tf.Variable(tf.constant_initializer(data_dict['fc6'][0],dtype=tf.float32))
biases6 = tf.Variable(tf.constant_initializer(data_dict['fc6'][1], dtype=tf.float32))
print weights6.get_shape().as_list()
print biases6.get_shape().as_list()

weights7 = tf.Variable(tf.constant_initializer(data_dict['fc7'][0], dtype=tf.float32))
biases7 = tf.Variable(tf.constant_initializer(data_dict['fc7'][1]))
print weights7.get_shape().as_list()
print biases7.get_shape().as_list()

weights8 = tf.Variable(tf.constant_initializer(data_dict['fc8'][0]))
biases8 = tf.Variable(tf.constant_initializer(data_dict['fc8'][1]))
print weights8.get_shape().as_list()
print biases8.get_shape().as_list()
'''

print data_dict['conv4_3'][0].shape
print data_dict['conv4_3'][1].shape

print data_dict['fc6'][0].shape
print data_dict['fc6'][1].shape


print data_dict['fc7'][0].shape
print data_dict['fc7'][1].shape
print data_dict['fc8'][0].shape
print data_dict['fc8'][1].shape


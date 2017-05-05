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

a = tf.one_hot([11,8,9,9], 12)

sess = tf.Session()

print sess.run(a)
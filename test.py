#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import model
import read_data_new
import math


class Config():
    batch_size = 16
    max_step = 6000

    img_width = 224
    img_height = 224
    img_channel = 3

    steps = '2600'
    param_dir = './params/'
    load_filename = 'vgg16-' + steps
    vgg_path = param_dir + load_filename +  '.npy'
    checkpointer_iter = 2000

    

    log_dir = './log/'
    summary_iter = 200

    degree = 10
    test_size = 5823 

    trainset_file = './labels/val_image.txt'
    imgs_path = './data/images/'
    labels_file = './labels/val_labels.txt'


def main():
    config = Config()

    modeler = model.VGG(config)

    #read data to val("data/val")
    val_reader = read_data_new.Reader(config)

    modeler.inference()
    accuracy = modeler.accuracy()

    init = tf.global_variables_initializer()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        #saver.restore(sess, config.param_dir+config.load_filename)
        print 'restore params' + config.steps

        #testing


        num_iter = int(math.ceil(config.test_size / config.batch_size))
        batch_accuracy = []
        for i in range(num_iter):  
            with tf.device('/cpu:0'):
                images_val, labels_val = val_reader.batch()

            with tf.device("/gpu:0"):
                acc = accuracy.eval({modeler.image_holder:images_val, modeler.label_holder: labels_val, modeler.is_train:False})
                batch_accuracy.append(acc)
                #print acc
                '''
                accuracy1, accuracy5 = sess.run([top_k_1, top_k_5], feed_dict={
                modeler.image_holder:images_val,
                modeler.label_holder:labels_val
                })
                '''



            #true_count1 += np.sum(accuracy1)
            #true_count5 += np.sum(accuracy5)
            #precision1 = 1.0*true_count1 / config.test_size
            #precision5 = 1.0*true_count5 / config.test_size
            #print 'precision of testing @ 1 = %.3f' % precision1
            #print 'precision of testing @ 5 = %.3f' % precision5

        mean_accuracy = np.mean(batch_accuracy)
        print 'precision of testing is: %.3f' % mean_accuracy


if __name__ == '__main__':
    main()



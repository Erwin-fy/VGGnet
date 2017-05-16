#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import model
import read_data
import math


class Config():
    batch_size = 1
    max_step = 6000

    img_width = 224
    img_height = 224
    img_channel = 3

    steps = '1000'
    param_dir = './params/'
    load_filename = 'vgg16-' + steps
    vgg_path = param_dir + load_filename +  '.npy'
    checkpointer_iter = 2000

    label_path='./labels/val_labels_new.txt'
    data_path = './data/images/'

    log_dir = './log/'
    summary_iter = 200

    degree = 10
    test_size = 5823 


def main():
    config = Config()

    modeler = model.VGG(config)

    #read data to val("data/val")
    val_reader = read_data.VGGReader(config)

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
        confmat = np.zeros((20, 20))
        count = 0
        num_iter = config.test_size // config.batch_size
        for i in range(num_iter):  
            with tf.device('/cpu:0'):
                images_val, labels_val, filenames_val = val_reader.batch()

            with tf.device("/gpu:0"):
                
                predict = sess.run(modeler.pred, feed_dict={modeler.image_holder:images_val, modeler.is_train:False})
                #print predict, labels_val
                if labels_val[0][predict[0]] > 0:
                    count += 1
                    confmat[predict[0]][predict[0]] += 1
                else:
                    for i in range(20):
                        if labels_val[0][i] > 0:
                            confmat[i][predict[0]] += 1

        confmat = confmat * 20 / 5823


        print 'AP: ',  count * 1.0 / config.test_size
        print 'confusion matrix: '
        print confmat


if __name__ == '__main__':
    main()



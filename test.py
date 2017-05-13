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
    batch_size = 16
    max_step = 6000

    img_width = 384
    img_height = 384
    img_channel = 3

    steps = '4000'
    param_dir = './params/'
    save_filename = 'modeler'
    load_filename = 'modeler-' + steps
    checkpointer_iter = 2000

    log_dir = './log/'
    summary_iter = 200

    degree = 10
    test_size = 3837

def main():
    config = Config()

    modeler = model.VGG(config)

    #read data to val("data/val")
    val_reader = read_data.VGGReader("./labels/val_labels.txt", "./data/images", config)

    logits = modeler.inference(False)

    top_k_1 = modeler.top_k_op(logits, 1)
    top_k_5 = modeler.top_k_op(logits, 5)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        saver.restore(sess, config.param_dir+config.load_filename)
        print 'restore params' + config.steps

        #testing
        true_count1 = 0
        true_count5 = 0
        num_iter = int(math.ceil(config.test_size / config.batch_size))
        for i in range(num_iter):  
            with tf.device('/cpu:0'):
                images_val, labels_val, val_filenames = val_reader.get_batch(False)

            with tf.device("/gpu:0"):
                accuracy1, accuracy5 = sess.run([top_k_1, top_k_5], feed_dict={
                    modeler.image_holder:images_val,
                    modeler.label_holder:labels_val
                })

            true_count1 += np.sum(accuracy1)
            true_count5 += np.sum(accuracy5)

        precision1 = 1.0*true_count1 / config.test_size
        precision5 = 1.0*true_count5 / config.test_size
        print 'precision of testing @ 1 = %.3f' % precision1
        print 'precision of testing @ 5 = %.3f' % precision5

if __name__ == '__main__':
    main()



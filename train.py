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
    batch_size = 32
    max_step = 100000

    img_width = 224
    img_height = 224
    img_channel = 3

    steps = '5800'
    param_dir = './params/'
    save_filename = 'vgg16-'
    checkpointer_iter = 200

    #vgg_path = param_dir + save_filename + steps + '.npy'
    vgg_path = './vgg16.npy'

    label_path = './labels/train_labels_new.txt'
    data_path = './data/images/'


    log_dir = './log/'
    summary_iter = 200

    degree = 10
    val_size = 32


def main():
    config = Config()

    modeler = model.VGG(config)

    # read data to train("data/train")
    train_reader = read_data.VGGReader(config)

    modeler.inference()
    loss = modeler.loss
    train_op = modeler.train_op(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        #saver.restore(sess, config.param_dir + config.load_filename)
        #print "restore params" + config.steps

        merged = tf.summary.merge_all()
        logdir = os.path.join(config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

        #start training
        print 'start training'
        for step in range(config.max_step):
            #start_time = time.time()

            with tf.device('/cpu:0'):
                images_train, labels_train, filesname_train = train_reader.get_random_batch(True)

            feed_dict = {
                modeler.image_holder:images_train,
                modeler.label_holder:labels_train,
                modeler.is_train:True
            }

            with tf.device('/gpu:0'):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            with tf.device('/cpu:0'):
                if (step+1)%config.checkpointer_iter == 0:
                    modeler.save_npy(sess, config.param_dir + config.save_filename + str(modeler.global_step.eval()) + '.npy')
                    
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, modeler.global_step.eval())
            
            if step%10 == 0:
                print 'step %d, loss = %.3f' % (step, loss_value)
                #print prediction
                #print labels_train

if __name__ == '__main__':
    main()



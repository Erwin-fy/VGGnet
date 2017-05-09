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
    max_step = 10000

    img_width = 224
    img_height = 224
    img_channel = 3

    steps = '-1'
    param_dir = './params/'
    save_filename = 'modeler'
    load_filename = 'modeler-' + steps
    checkpointer_iter = 2000

    log_dir = './log/'
    summary_iter = 200

    degree = 10
    val_size = 32
    test_size = 8400

def main():
    config = Config()

    modeler = model.VGG(config)

    # read data to train("data/train")
    train_reader = read_data.VGGReader("./labels/clear_train_labels.txt", "./data/images", config)

    logits = modeler.inference(True)
    loss = modeler.loss(logits)
    train_op = modeler.train_op(loss)

    predictions = tf.argmax(tf.nn.softmax(logits), 1)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        
        merged = tf.summary.merge_all()
        logdir = os.path.join(config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

        #start training
        print 'start training'
        for step in range(config.max_step):
            #start_time = time.time()

            with tf.device('/cpu:0'):
                images_train, labels_train, train_filenames = train_reader.get_random_batch(False)

            feed_dict = {
                modeler.image_holder:images_train,
                modeler.label_holder:labels_train
            }

            with tf.device('/gpu:0'):
                _, loss_value, prediction = sess.run([train_op, loss, predictions], feed_dict=feed_dict)

            with tf.device('/cpu:0'):
                if (step+1)%config.checkpointer_iter == 0:
                    saver.save(sess, config.param_dir+config.save_filename, modeler.global_step.eval())
                    
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, modeler.global_step.eval())
            
            if step%10 == 0:
                print 'step %d, loss = %.3f' % (step, loss_value)
                print prediction
                print labels_train

if __name__ == '__main__':
    main()


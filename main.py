#!/usr/bin/env python
# encoding: utf-8

import os
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import model


class Config():
    batch_size = 32
    max_step = 10000

    img_width = 224
    img_height = 224
    img_channel = 3

    start_learning_rate = 1e-2
    decay_rate = 0.95
    decay_steps = 1000

    steps = '10000'
    param_dir = '../params/'
    save_filename = 'model'
    load_filename = 'model-' + steps
    checkpointer_iter = 2000

    log_dir = '../log/'
    summary_iter = 20000


#Todo
'''
images_train, labels_train
images_test, labels_test
'''



def main():
    config = Config()

    Model = model.VGG(config)


    logits = Model.inference()
    loss = Model.loss(logits)
    train_op = Model.train_op(loss)

    top_k = Model.accuracy(logits)

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
                image_batch, label_batch = sess.run([images_train, labels_train])

            feed_dict = {
                Model.image_holder:image_batch,
                Model.label_holder:label_batch,
                Model.keep_prob:0.5
            }

            with tf.device('/gpu:0'):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            if step%20 == 0:
                print 'step %d, loss = ' % step, loss_value

            with tf.device('/cpu:0'):
                if (step+1)%config.checkpointer_iter == 0:
                    saver.save(sess, config.param_dir+config.save_filename, Model.global_step.eval())
                
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, Model.global_step.eval())

            
            #test
            num_examples = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
            num_iter = int(math.ceil(num_examples/config.batch_size))
            true_count = 0
            total_sample_count = num_iter*config.batch_size
            step = 0
            while step < num_iter:
                step += 1

                with tf.device("/cpu:0"):
                    image_batch, label_batch = sess.run([images_test, labels_test])

                feed_dict = {
                    Model.image_holder:image_batch,
                    Model.label_holder:label_batch,
                    Model.keep_prob:1.0
                }

                with tf.device("/gpu:0"):
                    accuracy = sess.run([top_k], feed_dict=feed_dict)

                true_count += np.sum(accuracy)

                precision = 1.0*true_count / total_sample_count

                print 'precision @ 1 = %.3f' % precision
    


if __name__ == '__main__':
    main()



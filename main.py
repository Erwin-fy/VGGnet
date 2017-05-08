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
    def __init__(self):
        self.batch_size = 32
        self.max_step = 10000

        self.img_width = 224
        self.img_height = 224
        self.img_channel = 3

        self.steps = '-1'
        self.param_dir = './params/'
        self.save_filename = 'modeler'
        self.load_filename = 'modeler-' + self.steps
        self.checkpointer_iter = 2000

        self.log_dir = './log/'
        self.summary_iter = 200

        self.degree = 10
        self.val_size = 1000
        self.test_size = 8400

def main():
    config = Config()

    modeler = model.VGG(config)
    
    # read data to train("data/train")
    train_reader = read_data.VGGReader("./labels/train_labels.txt", "./data/images", config)
    
    #read data to val("data/val")
    val_reader = read_data.VGGReader("./labels/train_labels.txt", "./data/images", config)

    logits = modeler.inference()
    loss = modeler.loss(logits)
    train_op = modeler.train_op(loss)

    #predictions = tf.nn.softmax(logits)
    top_k_op = modeler.top_k_op(logits)

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
	
            	#print train_filenames, labels_train
            

	    feed_dict = {
                modeler.image_holder:images_train,
                modeler.label_holder:labels_train,
                modeler.keep_prob:0.5
            }

            with tf.device('/gpu:0'):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)


            with tf.device('/cpu:0'):
                if (step+1)%config.checkpointer_iter == 0:
                    saver.save(sess, config.param_dir+config.save_filename, modeler.global_step.eval())
                
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, modeler.global_step.eval())
            
            #val
            if (step % 100 == 0) and step:
	        true_count = 0
	        num_iter = int(math.ceil(config.val_size / config.batch_size))
                
	        for i in range(num_iter):  
                    with tf.device('/cpu:0'):
                        images_val, labels_val, val_filenames = val_reader.get_random_batch(False)              

           	    feed_dict = {
                        modeler.image_holder:images_val,
                        modeler.label_holder:labels_val,
                        modeler.keep_prob:1.0
           	    }

           	    with tf.device("/gpu:0"):
                        accuracy = sess.run([top_k_op], feed_dict=feed_dict)
                        #prediction, accuracy = sess.run((predictions,top_k_op), feed_dict=feed_dict)
		
		    #print labels_val, accuracy, '\n'

                    true_count += np.sum(accuracy)

                precision = 1.0*true_count / config.val_size
                print 'precision @ 1 = %.3f' % precision
            
	    if step%10 == 0:
		#print modeler.global_step.eval()
                print 'step %d, loss = %.3f' % (step, loss_value)


        #testing
	true_count = 0
	num_iter = int(math.ceil(config.test_size / config.batch_size))
	for i in range(num_iter):  
            with tf.device('/cpu:0'):
                images_val, labels_val, val_filenames = val_reader.get_random_batch(False)              
            feed_dict = {
                modeler.image_holder:images_val,
                modeler.label_holder:labels_val,
                modeler.keep_prob:1.0
            }

            with tf.device("/gpu:0"):
                accuracy = sess.run([top_k_op], feed_dict=feed_dict)
		

            true_count += np.sum(accuracy)

        precision = 1.0*true_count / config.test_size
        print 'precision @ 1 = %.3f' % precision


if __name__ == '__main__':
    main()



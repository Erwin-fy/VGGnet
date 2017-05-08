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
    val_size = 1000
    test_size = 8400

batch_size = 32
vgg_npy_path = './vgg16.npy'
global_step = tf.get_variable('global_step', initializer=0, 
                        dtype=tf.int32, trainable=False)
data_dict = np.load(vgg_npy_path, encoding='latin1').item()
wl = 5e-4
start_learning_rate = 1e-5
decay_rate = 0.99
decay_steps = 200

image_holder = tf.placeholder(tf.float32, [32, 224, 224, 3])
label_holder = tf.placeholder(tf.int32, [32])
keep_prob = tf.placeholder(tf.float32)

def print_tensor(tensor):
        print tensor.op.name, ' ', tensor.get_shape().as_list()

def _activation_summary(tensor):
        name = tensor.op.name
        tf.summary.histogram(name + '/activatins', tensor)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))

def inference(images, keep_prob):
    conv1_1 = conv_layer(images, 64, 'conv1_1')
    conv1_2 = conv_layer(conv1_1, 64, 'conv1_2')
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print pool1

    conv2_1 = conv_layer(pool1, 128, 'conv2_1')
    conv2_2 = conv_layer(conv2_1, 128, 'conv2_2')
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3_1 = conv_layer(pool2, 256, 'conv3_1')
    conv3_2 = conv_layer(conv3_1, 256, 'conv3_2')
    conv3_3 = conv_layer(conv3_2, 256, 'conv3_3')
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv4_1 = conv_layer(pool3, 512, 'conv4_1')
    conv4_2 = conv_layer(conv4_1, 512, 'conv4_2')
    conv4_3 = conv_layer(conv4_2, 512, 'conv4_3')
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv5_1 = conv_layer(pool4, 512, 'conv5_1')
    conv5_2 = conv_layer(conv5_1, 512, 'conv5_2')
    conv5_3 = conv_layer(conv5_2, 512, 'conv5_3')
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print_tensor(pool5)

    fc6 = fc_layer(pool5, 4096, 'fc6')
    drop1 = tf.nn.dropout(fc6, keep_prob)
    fc7 = fc_layer(drop1, 4096, 'fc7')
    drop2 = tf.nn.dropout(fc7, keep_prob)
    logits = final_fc_layer(drop2, 20, 'fc8')
    return logits

def conv_layer(fm, channels, name):
    '''
    Arg fm: feather maps
    '''
    with tf.variable_scope(name) as scope:
        #shape = fm.get_shape()
        #kernel = variable_with_weight_loss(shape=[3, 3, shape[-1].value, channels], stddev=1e-2, wl=0.0)
        #kernel = tf.get_variable(scope + 'kernel', shape=[3, 3, shape[-1].value, channels], 
        #    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        kernel = get_conv_kernel(name)
        biases = get_bias(name)
        conv = tf.nn.conv2d(fm, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)

        activation = tf.nn.relu(pre_activation)

        print_tensor(activation)
        _activation_summary(activation)

        return activation

def fc_layer(input_op, fan_out, name):
    '''
    input_op: 输入tensor
    fan_in: 输入节点数
    fan_out： 输出节点数
    is_train: True --- fc   Flase --- conv
    '''
    with tf.variable_scope(name) as scope:
        reshape = tf.reshape(input_op, [batch_size, -1])
        #fan_in = reshape.get_shape()[1].value
        #weights = variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=0.004)
        #weights = tf.get_variable(scope + 'weights', shape=[fan_in, fan_out], 
        #    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fan_out]))
        
        weights = get_fc_weight(name)
        biases = get_bias(name)
        pre_activation = tf.nn.bias_add(tf.matmul(reshape, weights), biases)
        activation = tf.nn.relu(pre_activation)

        print_tensor(activation)
        _activation_summary(activation)
        return activation

def final_fc_layer(input_op, fan_out, name):
    with tf.variable_scope(name) as scope:
        fan_in = input_op.get_shape()[1].value            
        #weights = get_fc_weight(name)
        #weights = variable_with_weight_loss(shape=[fan_in, fan_out], stddev=1e-2, wl=wl)
        
        weights = get_fc_weight_reshape(name, [fan_in, 1000], num_classes=20)
        biases = get_bias_reshape(name, num_new=20)

        pre_activation = tf.nn.bias_add(tf.matmul(input_op, weights), biases)

        print_tensor(pre_activation)
        _activation_summary(pre_activation)

        return pre_activation

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_sum')
    cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')
    
    tf.add_to_collection('losses', cross_entropy)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)

    return total_loss


def train_op(total_loss):
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    
    return train_op

def get_conv_kernel(name):
    init = tf.constant_initializer(value=data_dict[name][0],dtype=tf.float32)
    shape = data_dict[name][0].shape
    var = tf.get_variable(name='kernel', shape=shape, initializer=init)

    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection("losses", weight_decay)

    return var

def get_fc_weight(name):
    init = tf.constant_initializer(value=data_dict[name][0], dtype=tf.float32)
    shape = data_dict[name][0].shape
    var = tf.get_variable(name='weights', shape=shape, initializer=init)
    
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection("losses", weight_decay)
    return var

def get_bias(name):
    init = tf.constant_initializer(value=data_dict[name][1], dtype=tf.float32)
    shape = data_dict[name][1].shape
    biases = tf.get_variable(name='biases', shape=shape, initializer=init)
    return biases

def get_fc_weight_reshape(name, shape, num_classes=None):
    print('Layer name: %s' % name)
    print('Layer shape: %s' % shape)
    weights = data_dict[name][0]
    weights = weights.reshape(shape)
    if num_classes is not None:
        weights = _summary_reshape(weights, shape,
                                        num_new=num_classes)
        print weights.shape
    init = tf.constant_initializer(value=weights,
                                    dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    return var

def get_bias_reshape(name, num_new):
    biases = data_dict[name][1]
    shape = data_dict[name][1].shape

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

def _summary_reshape(fweight, shape, num_new):
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

def main():
    config = Config()
    
    # read data to train("data/train")
    train_reader = read_data.VGGReader("./labels/train_labels.txt", "./data/images", config)
    
    #read data to val("data/val")
    val_reader = read_data.VGGReader("./labels/train_labels.txt", "./data/images", config)

    logits = inference(image_holder, keep_prob)
    loss = loss(logits, label_holder)
    train_op = train_op(loss)

    #predictions = tf.nn.softmax(logits)
    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

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
                image_holder:images_train,
                label_holder:labels_train,
                keep_prob:0.5
            }

            with tf.device('/gpu:0'):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)


            with tf.device('/cpu:0'):
                if (step+1)%config.checkpointer_iter == 0:
                    saver.save(sess, config.param_dir+config.save_filename, global_step.eval())
                
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, global_step.eval())
            
            #val
            if (step % 100 == 0) and step:
	        true_count = 0
	        num_iter = int(math.ceil(config.val_size / config.batch_size))
                
	        for i in range(num_iter):
                    with tf.device('/cpu:0'):
                        images_val, labels_val, val_filenames = val_reader.get_random_batch(False)              


           	    with tf.device("/gpu:0"):
                        accuracy = sess.run([top_k_op], feed_dict={
                            image_holder:images_val,
                            label_holder:labels_val,
                            keep_prob:1.0
                        })
		
                    true_count += np.sum(accuracy)

                precision = 1.0*true_count / config.val_size
                print 'precision @ 1 = %.3f' % precision
            
	    if step%10 == 0:
                print 'step %d, loss = %.3f' % (step, loss_value)


        #testing
	true_count = 0
	num_iter = int(math.ceil(config.test_size / config.batch_size))
	for i in range(num_iter):  
            with tf.device('/cpu:0'):
                images_val, labels_val, val_filenames = val_reader.get_random_batch(False)

            with tf.device("/gpu:0"):
                accuracy = sess.run([top_k_op], feed_dict={
                    image_holder:images_val,
                    label_holder:labels_val,
                    keep_prob:1.0
                })

            true_count += np.sum(accuracy)

        precision = 1.0*true_count / config.test_size
        print 'precision @ 1 = %.3f' % precision


if __name__ == '__main__':
    main()



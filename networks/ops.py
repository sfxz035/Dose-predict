# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def weight_variable(shape,name=None,trainable=True, decay_mult = 0.0):
    weights = tf.get_variable(
        name, shape, tf.float32, trainable=trainable,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
        # regularizer=tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return weights

def bias_variable(shape,name=None, bias_start = 0.0, trainable = True, decay_mult = 0.0):
    bais = tf.get_variable(
        name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32)
        # regularizer = tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return bais

def conv_bn(inpt ,output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1], is_train = True, name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h,k_w,inpt.get_shape()[-1],output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        batch_norm = tf.layers.batch_normalization(conv, training=is_train) ###由contrib换成layers
    return batch_norm

def BatchNorm(
        value, is_train = True, name = 'BatchNorm',
        epsilon = 1e-5, momentum = 0.9
    ):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            # updates_collections = tf.GraphKeys.UPDATE_OPS,
            # updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name
        )

def conv_relu(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        pre_relu = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(pre_relu)
        return out


def conv(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        out = tf.nn.bias_add(conv, biases)
    return out


def ReLU(value, name = 'ReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

def Deconv2d(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        biases = bias_variable(name='biases', shape=[output_shape[-1]])
        deconv = tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv
def Deconv2d_bn(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        is_train=True, name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        batch_norm = tf.layers.batch_normalization(deconv, training=is_train) ###由contrib换成layers
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return batch_norm, weights
        else:
            return batch_norm


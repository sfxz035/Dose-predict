from networks.ops import *
import tensorflow as tf
import numpy as np
#paramaters
FILTER_DIM = 24
OUTPUT_C = 1
#deep 5
def inference(images,is_training,reuse = False,name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_1 = ReLU(conv_bn(images, FILTER_DIM, k_h=3,is_train=is_training, name='Conv2d_1_1'),name='ReLU_1_1')
        L1_2 = ReLU(conv_bn(L1_1, FILTER_DIM, k_h=3,is_train=is_training, name='Conv2d_1_2'),name='ReLU_1_2')
        L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')  ##

        L2_2 = ReLU(conv_bn(L2_1, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_1'),name='ReLU_2_1')
        L2_3 = ReLU(conv_bn(L2_2, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_2'),name='ReLU_2_2')
        L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling2')    ##

        L3_2 = ReLU(conv_bn(L3_1, FILTER_DIM*4, k_h=3, is_train=is_training,name='Conv2d_3_1'),name='ReLU_3_1')
        L3_3 = ReLU(conv_bn(L3_2, FILTER_DIM*4, k_h=3, is_train=is_training,name='Conv2d_3_2'),name='ReLU_3_2')
        L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling3')    ##

        L4_2 = ReLU(conv_bn(L4_1, FILTER_DIM*8, k_h=3, is_train=is_training,name='Conv2d_4_1'),name='ReLU_4_1')
        L4_3 = ReLU(conv_bn(L4_2, FILTER_DIM*8, k_h=3, is_train=is_training,name='Conv2d_4_2'),name='ReLU_4_2')
        L5_1 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L5_2 = ReLU(conv_bn(L5_1, FILTER_DIM*16, k_h=3, is_train=is_training,name='Conv2d_5_1'),name='ReLU_5_1')
        L5_3 = ReLU(conv_bn(L5_2, FILTER_DIM*16, k_h=3, is_train=is_training,name='Conv2d_5_2'),name='ReLU_5_2')

        L4_U1 = ReLU(Deconv2d_bn(L5_3, L4_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d4'),name='DeReLU4')
        L4_U1 = tf.concat((L4_3, L4_U1), -1)
        L4_U2 = ReLU(conv_bn(L4_U1, FILTER_DIM * 8, k_h=3, is_train=is_training,name='Conv2d_4_u1'),name='ReLU_4_u1')
        L4_U3 = ReLU(conv_bn(L4_U2, FILTER_DIM * 8, k_h=3, is_train=is_training,name='Conv2d_4_u2'),name='ReLU_4_u2')

        L3_U1 = ReLU(Deconv2d_bn(L4_U3,L3_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d3'),name = 'DeReLU3')
        L3_U1 = tf.concat((L3_3, L3_U1), -1)
        L3_U2 = ReLU(conv_bn(L3_U1, FILTER_DIM*4, k_h=3,is_train=is_training, name='Conv2d_3_u1'), name='ReLU_3_u1')
        L3_U3 = ReLU(conv_bn(L3_U2, FILTER_DIM*4, k_h=3,is_train=is_training, name='Conv2d_3_u2'), name='ReLU_3_u2')

        L2_U1 = ReLU(Deconv2d_bn(L3_U3,L2_3.get_shape(), k_h = 3,is_train=is_training,name = 'Deconv2d2'),name='DeReLU2')
        L2_U1 = tf.concat((L2_3, L2_U1), -1)
        L2_U2 = ReLU(conv_bn(L2_U1, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_u1'),name='ReLU_2_u1')
        L2_U3 = ReLU(conv_bn(L2_U2, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_u2'),name='ReLU_2_u2')

        L1_U1 = ReLU(Deconv2d_bn(L2_U3, L1_2.get_shape(),k_h=3,is_train=is_training,name='Deconv2d1'),name='DeReLU1')
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = ReLU(conv_bn(L1_U1, FILTER_DIM, k_h=3, is_train=is_training,name='Conv1d_1_u1'),name='ReLU_1_u1')
        L1_U3 = ReLU(conv_bn(L1_U2, FILTER_DIM, k_h=3, is_train=is_training,name='Conv1d_1_u2'),name='ReLU_1_u2')

        out = conv(L1_U3, OUTPUT_C,name='Conv1d_out')

    # variables = tf.contrib.framework.get_variables(name)

    return out

def losses(output, labels, name = 'losses'):
    with tf.name_scope(name):
        loss = tf.reduce_mean(tf.square(output - labels))
        return loss








